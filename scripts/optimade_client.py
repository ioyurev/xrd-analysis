"""OPTIMADE v1 REST API client and crystal structure integrator for xrd-analysis.

Provides zero-dependency, key-free access to federated OPTIMADE v1.1 providers:
1. Deterministic translation of ElementFilter into standard OPTIMADE v1.1 filter grammar.
2. Federated multi-provider searching across Materials Project, Alexandria (PBE/PBEsol),
   COD, and JARVIS-DFT.
3. Fetching and standardizing Cartesian crystal structures into conventional CIFs
   compatible with GSAS-II, including disordered solid solutions with partial occupancies.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from constants import VERSION
from pymatgen.core import Lattice, Structure  # type: ignore[attr-defined]
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if TYPE_CHECKING:
    from cif_retriever import ElementFilter, StructureSearchRecord

__all__ = [
    "KNOWN_PROVIDERS",
    "CellMetrics",
    "OptimadeHttpResponse",
    "OptimadeProvider",
    "ParsedStructureId",
    "StabilityMetrics",
    "download_optimade_cif",
    "element_filter_to_optimade",
    "optimade_resource_to_structure",
    "parse_composite_id",
    "parse_optimade_record",
    "search_optimade",
    "search_optimade_provider",
    "structure_to_cif_text",
]


@dataclass(frozen=True)
class OptimadeProvider:
    """Configuration for an OPTIMADE v1 provider endpoint."""

    id: str
    name: str
    base_url: str
    supports_has_any: bool = True


KNOWN_PROVIDERS: dict[str, OptimadeProvider] = {
    "mp": OptimadeProvider(
        id="mp",
        name="Materials Project (OPTIMADE)",
        base_url="https://optimade.materialsproject.org/v1",
        supports_has_any=True,
    ),
    "alexandria_pbe": OptimadeProvider(
        id="alexandria_pbe",
        name="Alexandria PBE",
        base_url="https://alexandria.icams.rub.de/pbe/v1",
        supports_has_any=True,
    ),
    "alexandria_pbesol": OptimadeProvider(
        id="alexandria_pbesol",
        name="Alexandria PBEsol",
        base_url="https://alexandria.icams.rub.de/pbesol/v1",
        supports_has_any=True,
    ),
    "cod": OptimadeProvider(
        id="cod",
        name="COD (OPTIMADE)",
        base_url="https://www.crystallography.net/cod/optimade/v1",
        supports_has_any=True,
    ),
    "jarvis": OptimadeProvider(
        id="jarvis",
        name="JARVIS-DFT",
        base_url="https://jarvis.nist.gov/optimade/jarvisdft/v1",
        supports_has_any=False,
    ),
}


@dataclass(frozen=True)
class CellMetrics:
    """Calculated unit cell parameters from 3x3 Cartesian lattice vectors."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float


@dataclass(frozen=True)
class StabilityMetrics:
    """Extracted thermodynamic stability metrics (energy above convex hull in eV/atom)."""

    energy_above_hull: float | None = None
    is_stable: bool | None = None


@dataclass(frozen=True)
class ParsedStructureId:
    """Parsed components of an OPTIMADE composite structure identifier."""

    provider_id: str
    native_id: str


@dataclass(frozen=True)
class OptimadeHttpResponse:
    """Typed HTTP response payload from an OPTIMADE server."""

    status: int
    data: dict[str, Any] | None = None
    error: str | None = None


def _build_http_opener() -> urllib.request.OpenerDirector:
    """Build a urllib opener respecting standard HTTPS_PROXY / HTTP_PROXY environment variables."""
    proxy_url = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if proxy_url:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
    return urllib.request.build_opener()


def _http_get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> OptimadeHttpResponse:
    """Execute GET request against an OPTIMADE endpoint and return parsed JSON."""
    req_headers = {
        "User-Agent": f"xrd-analysis/{VERSION} (OPTIMADE-Client)",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    opener = _build_http_opener()
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read()
            data = json.loads(raw.decode("utf-8", errors="replace"))
            return OptimadeHttpResponse(
                status=int(resp.status),
                data=data if isinstance(data, dict) else None,
            )
    except urllib.error.HTTPError as he:
        body = ""
        try:
            body = he.read()[:200].decode("utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            pass
        return OptimadeHttpResponse(
            status=int(he.code),
            error=f"HTTP {he.code}: {he.reason} ({body})",
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return OptimadeHttpResponse(status=0, error=str(exc))


def element_filter_to_optimade(
    elem_filter: ElementFilter,
    supports_has_any: bool = True,
) -> str:
    """Translate an ElementFilter into a deterministic OPTIMADE v1.1 filter string."""
    parts: list[str] = []

    if elem_filter.all_of:
        sorted_all = sorted(elem_filter.all_of)
        formatted_all = ", ".join(f'"{el}"' for el in sorted_all)
        parts.append(f"elements HAS ALL {formatted_all}")

    if elem_filter.any_of:
        sorted_any = sorted(elem_filter.any_of)
        if supports_has_any:
            formatted_any = ", ".join(f'"{el}"' for el in sorted_any)
            parts.append(f"elements HAS ANY {formatted_any}")
        else:
            disjunction = " OR ".join(f'elements HAS "{el}"' for el in sorted_any)
            parts.append(f"({disjunction})")

    if elem_filter.none_of:
        sorted_none = sorted(elem_filter.none_of)
        none_disjunction = " OR ".join(f'elements HAS "{el}"' for el in sorted_none)
        parts.append(f"NOT ({none_disjunction})")

    if elem_filter.strict:
        if not elem_filter.any_of and elem_filter.all_of:
            parts.append(f"nelements = {len(elem_filter.all_of)}")
        elif elem_filter.all_of or elem_filter.any_of:
            max_elems = len(elem_filter.all_of | elem_filter.any_of)
            parts.append(f"nelements <= {max_elems}")

    return " AND ".join(parts)


def _compute_cell_parameters(lattice_mat: list[list[float]]) -> CellMetrics | None:
    """Calculate CellMetrics (a, b, c, alpha, beta, gamma, volume) from a 3x3 Cartesian lattice matrix."""
    if len(lattice_mat) != 3 or any(len(row) != 3 for row in lattice_mat):
        return None

    a_vec = [float(x) for x in lattice_mat[0]]
    b_vec = [float(x) for x in lattice_mat[1]]
    c_vec = [float(x) for x in lattice_mat[2]]

    a = math.sqrt(sum(x * x for x in a_vec))
    b = math.sqrt(sum(x * x for x in b_vec))
    c = math.sqrt(sum(x * x for x in c_vec))
    if a == 0.0 or b == 0.0 or c == 0.0:
        return None

    def _dot(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=False))

    def _angle(u: list[float], v: list[float], len_u: float, len_v: float) -> float:
        val = max(-1.0, min(1.0, _dot(u, v) / (len_u * len_v)))
        return math.degrees(math.acos(val))

    alpha = _angle(b_vec, c_vec, b, c)
    beta = _angle(a_vec, c_vec, a, c)
    gamma = _angle(a_vec, b_vec, a, b)

    cross_bc = [
        b_vec[1] * c_vec[2] - b_vec[2] * c_vec[1],
        b_vec[2] * c_vec[0] - b_vec[0] * c_vec[2],
        b_vec[0] * c_vec[1] - b_vec[1] * c_vec[0],
    ]
    vol = abs(_dot(a_vec, cross_bc))

    return CellMetrics(
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        volume=vol,
    )


def _extract_energy_above_hull(attrs: dict[str, Any]) -> StabilityMetrics:
    """Extract StabilityMetrics across known provider attribute conventions."""
    if "energy_above_hull" in attrs and attrs["energy_above_hull"] is not None:
        try:
            val = float(attrs["energy_above_hull"])
            return StabilityMetrics(energy_above_hull=val, is_stable=val <= 1e-4)
        except (ValueError, TypeError):
            pass

    mp_stab = attrs.get("_mp_stability")
    if isinstance(mp_stab, dict):
        candidates: list[float] = []
        for calc in mp_stab.values():
            if (
                isinstance(calc, dict)
                and "energy_above_hull" in calc
                and calc["energy_above_hull"] is not None
            ):
                try:
                    candidates.append(float(calc["energy_above_hull"]))
                except (ValueError, TypeError):
                    pass
        if candidates:
            best_ehull = min(candidates)
            return StabilityMetrics(
                energy_above_hull=best_ehull, is_stable=best_ehull <= 1e-4
            )

    alex_hull = attrs.get("_alexandria_hull_distance")
    if alex_hull is None:
        alex_hull = attrs.get("_alexandria_scan_hull_distance")
    if alex_hull is not None:
        try:
            val = float(alex_hull)
            return StabilityMetrics(energy_above_hull=val, is_stable=val <= 1e-4)
        except (ValueError, TypeError):
            pass

    jarvis_hull = attrs.get("_jarvis_ehull")
    if jarvis_hull is not None:
        try:
            val = float(jarvis_hull)
            return StabilityMetrics(energy_above_hull=val, is_stable=val <= 1e-4)
        except (ValueError, TypeError):
            pass

    return StabilityMetrics(energy_above_hull=None, is_stable=None)


def parse_optimade_record(
    provider_id: str,
    item: dict[str, Any],
    *,
    preserve_mp_id: bool = False,
) -> StructureSearchRecord | None:
    """Parse one OPTIMADE data resource item into a StructureSearchRecord."""
    from cif_retriever import DatabaseSource, StructureSearchRecord

    struct_id = str(item.get("id", "")).strip()
    if not struct_id:
        return None

    if preserve_mp_id and provider_id == "mp":
        rec_id = struct_id if struct_id.startswith("mp-") else f"mp-{struct_id}"
    else:
        rec_id = f"optimade-{provider_id}-{struct_id}"

    attrs = item.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    formula = (
        attrs.get("chemical_formula_reduced")
        or attrs.get("chemical_formula_descriptive")
        or attrs.get("chemical_formula_anonymous")
        or struct_id
    )

    sg_sym = str(
        attrs.get("space_group_symbol")
        or attrs.get("_space_group_symbol")
        or attrs.get("_mp_spacegroup_symbol")
        or ""
    ).strip()

    lattice_mat = attrs.get("lattice_vectors")
    cell_metrics = (
        _compute_cell_parameters(lattice_mat) if isinstance(lattice_mat, list) else None
    )
    stab = _extract_energy_above_hull(attrs)

    if provider_id == "mp":
        source_val = DatabaseSource.MP
    elif provider_id == "cod":
        source_val = DatabaseSource.COD
    else:
        source_val = DatabaseSource.OPTIMADE

    return StructureSearchRecord(
        id=rec_id,
        source=source_val,
        formula=str(formula),
        spacegroup=sg_sym,
        a=cell_metrics.a if cell_metrics else None,
        b=cell_metrics.b if cell_metrics else None,
        c=cell_metrics.c if cell_metrics else None,
        alpha=cell_metrics.alpha if cell_metrics else None,
        beta=cell_metrics.beta if cell_metrics else None,
        gamma=cell_metrics.gamma if cell_metrics else None,
        volume=cell_metrics.volume if cell_metrics else None,
        energy_above_hull=stab.energy_above_hull,
        is_stable=stab.is_stable,
    )


def search_optimade_provider(
    provider: OptimadeProvider,
    elem_filter: ElementFilter,
    limit: int = 50,
    timeout: float = 15.0,
    *,
    preserve_mp_id: bool = False,
) -> list[StructureSearchRecord]:
    """Execute search against a single OPTIMADE provider."""
    from cif_retriever import extract_elements

    filter_expr = element_filter_to_optimade(
        elem_filter, supports_has_any=provider.supports_has_any
    )
    encoded_filter = urllib.parse.quote(filter_expr)
    page_limit = max(1, min(int(limit) if limit > 0 else 50, 100))
    query_url = f"{provider.base_url}/structures?filter={encoded_filter}&page_limit={page_limit}"

    resp = _http_get_json(query_url, timeout=timeout)
    if resp.status != 200 or not resp.data:
        return []

    data_entries = resp.data.get("data", [])
    if not isinstance(data_entries, list):
        return []

    records: list[StructureSearchRecord] = []
    for item in data_entries:
        if not isinstance(item, dict):
            continue
        rec = parse_optimade_record(
            provider.id, item, preserve_mp_id=preserve_mp_id
        )
        if rec is None:
            continue
        extracted = extract_elements(rec.formula)
        if elem_filter.matches(extracted):
            records.append(rec)

    return records


def search_optimade(
    elem_filter: ElementFilter,
    providers: list[str] | None = None,
    limit: int = 25,
    timeout: float = 15.0,
) -> list[StructureSearchRecord]:
    """Perform federated multi-provider search across configured OPTIMADE endpoints."""
    target_provider_ids = providers or [
        "mp",
        "alexandria_pbe",
        "alexandria_pbesol",
        "jarvis",
    ]
    all_results: list[StructureSearchRecord] = []

    for pid in target_provider_ids:
        prov = KNOWN_PROVIDERS.get(pid)
        if not prov:
            continue
        try:
            records = search_optimade_provider(
                prov, elem_filter, limit=limit, timeout=timeout
            )
            all_results.extend(records)
        except (urllib.error.URLError, TimeoutError, OSError):
            continue

    seen_keys: set[str] = set()
    deduped: list[StructureSearchRecord] = []
    for r in all_results:
        key = f"{r.id}_{r.formula}_{round(r.volume or 0.0, 1)}"
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)

    deduped.sort(
        key=lambda x: (
            not bool(x.is_stable),
            float(x.energy_above_hull if x.energy_above_hull is not None else 999.0),
            x.formula,
        )
    )
    return deduped


def optimade_resource_to_structure(attributes: dict[str, Any]) -> Structure:
    """Convert raw OPTIMADE structure attributes to pymatgen Structure with conventional cell standardization."""
    lattice_mat = attributes.get("lattice_vectors")
    cart_coords = attributes.get("cartesian_site_positions")
    species_at_sites = attributes.get("species_at_sites", [])
    raw_species_list = attributes.get("species") or []
    species_defs = {
        str(s["name"]): s
        for s in raw_species_list
        if isinstance(s, dict) and "name" in s
    }

    if not lattice_mat or not cart_coords or not species_at_sites:
        raise ValueError(
            "Invalid OPTIMADE structure: missing lattice_vectors, cartesian_site_positions, or species_at_sites"
        )

    lattice = Lattice(lattice_mat)

    site_species: list[Any] = []
    for sp_name in species_at_sites:
        sp_def = species_defs.get(str(sp_name))
        if sp_def:
            symbols = sp_def.get("chemical_symbols") or [str(sp_name)]
            concs = sp_def.get("concentration") or [1.0]
            if len(symbols) == 1 and abs(float(concs[0]) - 1.0) < 1e-5:
                site_species.append(str(symbols[0]))
            else:
                comp_dict = {
                    str(sym): float(conc)
                    for sym, conc in zip(symbols, concs, strict=False)
                    if str(sym).lower() != "vacancy"
                }
                site_species.append(comp_dict)
        else:
            site_species.append(str(sp_name))

    raw_struct = Structure(
        lattice=lattice,
        species=site_species,
        coords=cart_coords,
        coords_are_cartesian=True,
    )

    try:
        if raw_struct.is_ordered:
            sga = SpacegroupAnalyzer(raw_struct, symprec=0.01)
            conv_struct: Structure = sga.get_conventional_standard_structure()
            return conv_struct
    except (ValueError, RuntimeError, TypeError):
        pass

    return raw_struct


def structure_to_cif_text(struct: Structure, symprec: float = 0.01) -> str:
    """Serialize pymatgen Structure to clean CIF text with GSAS-II compatible space group symbol."""
    effective_symprec = symprec if struct.is_ordered else None
    raw_cif = str(CifWriter(struct, symprec=effective_symprec))
    return re.sub(
        r"(_symmetry_space_group_name_H-M\s+['\"]?)([^\s'\"]+)(['\"]?)",
        lambda m: f"{m.group(1)}{m.group(2).replace('_', '')}{m.group(3)}",
        raw_cif,
    )


def parse_composite_id(composite_id: str) -> ParsedStructureId:
    """Parse composite ID into ParsedStructureId(provider_id, native_id).

    Examples:
    - 'optimade-mp-mp-149' -> ParsedStructureId('mp', 'mp-149')
    - 'optimade-alexandria_pbe-agm001003285' -> ParsedStructureId('alexandria_pbe', 'agm001003285')
    - 'optimade-cod-1008317' -> ParsedStructureId('cod', '1008317')
    - 'mp-149' -> ParsedStructureId('mp', 'mp-149')
    - 'agm001003285' -> ParsedStructureId('alexandria_pbe', 'agm001003285')
    - 'JVASP-1002' -> ParsedStructureId('jarvis', 'JVASP-1002')
    """
    clean = composite_id.strip()
    if clean.startswith("optimade-"):
        parts = clean.split("-", 2)
        if len(parts) == 3:
            return ParsedStructureId(provider_id=parts[1], native_id=parts[2])
    if clean.startswith("mp-"):
        return ParsedStructureId(provider_id="mp", native_id=clean)
    if clean.startswith("agm"):
        return ParsedStructureId(provider_id="alexandria_pbe", native_id=clean)
    if clean.upper().startswith("JVASP-"):
        return ParsedStructureId(provider_id="jarvis", native_id=clean)
    if clean.isdigit():
        return ParsedStructureId(provider_id="cod", native_id=clean)
    return ParsedStructureId(provider_id="mp", native_id=clean)


def download_optimade_cif(
    composite_id: str,
    out_path: Path | None = None,
    symprec: float = 0.01,
    timeout: float = 15.0,
    formula_hint: str | None = None,
) -> Path | None:
    """Download crystal structure from an OPTIMADE provider and export as standardized CIF."""
    parsed = parse_composite_id(composite_id)
    provider = KNOWN_PROVIDERS.get(parsed.provider_id)
    if not provider:
        return None

    direct_url = f"{provider.base_url}/structures/{urllib.parse.quote(parsed.native_id)}"
    resp = _http_get_json(direct_url, timeout=timeout)

    entry: dict[str, Any] | None = None
    if resp.status == 200 and resp.data:
        data_field = resp.data.get("data")
        if isinstance(data_field, dict):
            entry = data_field
        elif isinstance(data_field, list) and data_field and isinstance(data_field[0], dict):
            entry = data_field[0]

    if entry is None:
        filter_expr = urllib.parse.quote(f'id="{parsed.native_id}"')
        filter_url = f"{provider.base_url}/structures?filter={filter_expr}"
        resp2 = _http_get_json(filter_url, timeout=timeout)
        if resp2.status == 200 and resp2.data:
            data_list = resp2.data.get("data")
            if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
                entry = data_list[0]

    if entry is None:
        return None

    attrs = entry.get("attributes") or {}
    if not isinstance(attrs, dict):
        return None

    try:
        struct = optimade_resource_to_structure(attrs)
    except (ValueError, TypeError, KeyError):
        return None

    cif_content = structure_to_cif_text(struct, symprec=symprec)

    formula = (
        formula_hint
        or attrs.get("chemical_formula_reduced")
        or attrs.get("chemical_formula_descriptive")
        or struct.composition.reduced_formula
        or parsed.native_id
    )
    safe_formula = "".join(
        c for c in str(formula) if c.isalnum() or c in ("-", "_", "(", ")")
    )
    if parsed.provider_id == "mp":
        cif_filename = f"{safe_formula}_{parsed.native_id}.cif"
    else:
        cif_filename = f"{safe_formula}_{parsed.provider_id}_{parsed.native_id}.cif"

    if out_path:
        dest_file = out_path / cif_filename if out_path.is_dir() else out_path
    else:
        cache_dir = Path.home() / ".cache" / "optimade_cifs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest_file = cache_dir / cif_filename

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_text(cif_content, encoding="utf-8")
    return dest_file
