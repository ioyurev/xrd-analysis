"""COD & Materials Project REST API client and crystal-chemical analog generator.

Supports:
1. Deterministic parameterized querying of Crystallography Open Database (COD) REST API.
2. Querying Materials Project (MP) REST API using official mp-api client.
3. Local caching of CIFs in ~/.cache/cod_cifs and ~/.cache/mp_cifs.
4. Generating isostructural analogs with Shannon radii scaling and pymatgen.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class DatabaseSource(StrEnum):
    """Supported crystal structure database sources."""

    COD = "cod"
    MP = "mp"
    ALL = "all"


@dataclass(frozen=True)
class CacheStatus:
    """Cache hit status and human-readable age for a database query."""

    cached: bool = False
    age: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return {"cached": self.cached, "age": self.age}


@dataclass(frozen=True)
class StructureSearchRecord:
    """Typed metadata record of a structure discovered in COD or Materials Project."""

    id: str
    source: DatabaseSource
    formula: str
    spacegroup: str
    a: float | None = None
    b: float | None = None
    c: float | None = None
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    volume: float | None = None
    energy_above_hull: float | None = None
    is_stable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": str(self.source),
            "formula": self.formula,
            "spacegroup": self.spacegroup,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "volume": self.volume,
            "energy_above_hull": self.energy_above_hull,
            "is_stable": self.is_stable,
        }


@dataclass(frozen=True)
class SearchCifsResponse:
    """Typed response of a multi-database structure search."""

    total_count: int
    displayed_count: int
    limit: int | None
    filter: dict[str, Any]
    filter_summary: str
    source: DatabaseSource | str
    cache_info: dict[str, CacheStatus]
    results: list[StructureSearchRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "displayed_count": self.displayed_count,
            "limit": self.limit,
            "filter": dict(self.filter),
            "filter_summary": self.filter_summary,
            "source": str(self.source),
            "cache_info": {k: v.to_dict() for k, v in self.cache_info.items()},
            "results": [r.to_dict() for r in self.results],
        }


if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from env_file import load_env_file
from loguru import logger
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Element, Species, Structure
from pymatgen.io.cif import CifWriter

ROMAN_TO_INT: dict[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "IVPY": 4,
    "IVSQ": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}
INT_TO_ROMAN: dict[int, str] = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
}


def normalize_cn(cn: str | int | None) -> str:
    """Normalize coordination number representation to Roman numerals."""
    if cn is None:
        return "VI"
    if isinstance(cn, int):
        return INT_TO_ROMAN.get(cn, "VI")
    cn_str = str(cn).strip().upper()
    if cn_str.isdigit():
        return INT_TO_ROMAN.get(int(cn_str), "VI")
    return cn_str


def extract_elements(formula_str: str) -> set[str]:
    """Extract chemical element symbols from formula string with fallback parsing."""
    clean = formula_str.replace(" ", "")
    clean_std = clean.replace("[", "(").replace("]", ")")
    if "·" in clean_std:
        clean_std = clean_std.split("·")[0]
    if "*" in clean_std:
        clean_std = clean_std.split("*")[0]
    try:
        comp = Composition(clean_std)
        elems: set[str] = set()
        for el in comp.elements:
            sym = el.symbol
            if sym in ("D", "T"):
                sym = "H"
            elems.add(sym)
        return elems
    except Exception:  # noqa: BLE001
        raw_syms = set(re.findall(r"[A-Z][a-z]?", formula_str))
        valid: set[str] = set()
        for s in raw_syms:
            if s in ("D", "T"):
                valid.add("H")
                continue
            try:
                valid.add(Element(s).symbol)
            except ValueError:
                pass
        return valid


def _float_or_none(val: Any) -> float | None:
    """Safely convert value to float or return None."""
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


DEFAULT_SEARCH_CACHE_TTL = 7 * 86400  # 7 days in seconds


def _format_cache_age(mtime: float) -> str:
    """Format cache age as human-readable string (e.g. 'just now', '45m ago', '2d ago')."""
    age_sec = max(0.0, time.time() - mtime)
    if age_sec < 60:
        return "just now"
    if age_sec < 3600:
        mins = int(age_sec // 60)
        return f"{mins}m ago"
    if age_sec < 86400:
        hours = int(age_sec // 3600)
        return f"{hours}h ago"
    days = int(age_sec // 86400)
    return f"{days}d ago"


@dataclass(frozen=True)
class ElementFilter:
    """Three-component conjunctive/disjunctive/exclusion chemical element filter.

    - all_of: AND (conjunction) - structure must contain all these elements.
    - any_of: OR (disjunction) - structure must contain at least one of these elements.
    - none_of: NOT / NOR (exclusion) - structure must not contain any of these elements.
    - strict: Closed chemical system - structure cannot contain any elements outside all_of | any_of.
    """

    all_of: frozenset[str] = field(default_factory=frozenset)
    any_of: frozenset[str] = field(default_factory=frozenset)
    none_of: frozenset[str] = field(default_factory=frozenset)
    strict: bool = False

    @classmethod
    def from_raw(
        cls,
        all_elements: str | list[str] | set[str] | frozenset[str] | None = None,
        any_elements: str | list[str] | set[str] | frozenset[str] | None = None,
        none_elements: str | list[str] | set[str] | frozenset[str] | None = None,
        strict: bool = False,
    ) -> ElementFilter:
        """Parse raw user inputs (comma-separated strings or collections) into validated ElementFilter."""

        def _parse(val: str | list[str] | set[str] | frozenset[str] | None) -> set[str]:
            if not val:
                return set()
            raw_list = val if not isinstance(val, str) else val.split(",")
            elems: set[str] = set()
            for r in raw_list:
                s = str(r).strip()
                if not s:
                    continue
                if s.upper() in ("D", "T"):
                    norm = "H"
                else:
                    norm = s.capitalize()
                try:
                    el = Element(norm)
                    elems.add(el.symbol)
                except ValueError:
                    raise ValueError(
                        f"Unknown chemical element '{s}'. Please use valid IUPAC element symbols."
                    ) from None
            return elems

        all_set = _parse(all_elements)
        any_set = _parse(any_elements)
        none_set = _parse(none_elements)

        if not all_set and not any_set:
            raise ValueError(
                "At least one positive inclusion criterion (--all or --any) must be provided."
            )

        contradiction = (all_set | any_set) & none_set
        if contradiction:
            raise ValueError(
                f"Contradictory filter: elements {sorted(contradiction)} cannot be both included and excluded."
            )

        # Prune elements from any_set that are already guaranteed by all_set
        redundant = any_set & all_set
        if redundant:
            any_set = any_set - redundant

        return cls(
            all_of=frozenset(all_set),
            any_of=frozenset(any_set),
            none_of=frozenset(none_set),
            strict=strict,
        )

    def matches(self, elements: set[str]) -> bool:
        """Check if candidate elemental composition matches filter specification."""
        if self.all_of and not self.all_of.issubset(elements):
            return False
        if self.any_of and not (self.any_of & elements):
            return False
        if self.none_of and bool(self.none_of & elements):
            return False
        return not (self.strict and not elements.issubset(self.all_of | self.any_of))

    def summary_str(self) -> str:
        """Human-readable description of active filter criteria."""
        parts: list[str] = []
        if self.all_of:
            parts.append(f"All: [{', '.join(sorted(self.all_of))}]")
        if self.any_of:
            parts.append(f"Any: [{', '.join(sorted(self.any_of))}]")
        if self.none_of:
            parts.append(f"None: [{', '.join(sorted(self.none_of))}]")
        if self.strict:
            parts.append("Strict: Yes")
        return ", ".join(parts) if parts else "No filters"


class CodClient:
    """Client for Crystallography Open Database (COD) REST API."""

    SEARCH_URL = "https://www.crystallography.net/cod/result"
    CIF_URL_TEMPLATE = "https://www.crystallography.net/cod/{cod_id}.cif"

    def __init__(
        self,
        cache_dir: Path | None = None,
        timeout: int = 15,
        cache_ttl: float = DEFAULT_SEARCH_CACHE_TTL,
    ) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "cod_cifs"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.search_cache_dir = self.cache_dir / "searches"
        self.search_cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.last_cache_info: CacheStatus = CacheStatus()

    def search(
        self,
        filter: ElementFilter | None = None,
        elements: list[str] | None = None,
        refresh: bool = False,
        subfiles: frozenset[str] | None = None,
    ) -> list[StructureSearchRecord]:
        """Search COD for structures matching element criteria (local HSRDB first, REST API fallback)."""
        if filter is None:
            if not elements:
                self.last_cache_info = CacheStatus()
                return []
            filter = ElementFilter.from_raw(all_elements=elements)

        try:
            from hs_rdb import HsRdbDatabase, HsRdbNotFoundError
            from pdf2_reader import CrystallographicFilter

            c_filter = CrystallographicFilter(subfiles=subfiles) if subfiles else None
            with HsRdbDatabase() as db:
                hs_records = db.search(element_filter=filter, crystal_filter=c_filter)
            self.last_cache_info = CacheStatus(cached=True, age="local-hsrdb")
            return [
                StructureSearchRecord(
                    id=f"cod-{r.cod_id}",
                    source=DatabaseSource.COD,
                    formula=r.formula,
                    spacegroup=r.spacegroup,
                    a=r.a,
                    b=r.b,
                    c=r.c,
                    alpha=r.alpha,
                    beta=r.beta,
                    gamma=r.gamma,
                    volume=r.volume,
                    energy_above_hull=None,
                    is_stable=None,
                )
                for r in hs_records
            ]
        except (ImportError, HsRdbNotFoundError, OSError, sqlite3.Error) as exc:
            logger.debug("Local COD HSRDB not used ({}); falling back to COD REST API", exc)

        queries_elements: list[list[str]] = []
        if filter.all_of:
            queries_elements.append(sorted(filter.all_of)[:8])
        elif filter.any_of:
            for el in sorted(filter.any_of):
                queries_elements.append([el])
        else:
            self.last_cache_info = CacheStatus()
            return []

        raw_results: list[dict[str, Any]] = []
        seen_files: set[str] = set()
        all_from_cache = True
        cache_mtimes: list[float] = []

        for q_elems in queries_elements:
            safe_q = "_".join(sorted(q_elems))
            cache_file = self.search_cache_dir / f"cod_query_{safe_q}.json"
            loaded = None
            is_from_cache = False

            if not refresh and cache_file.exists():
                try:
                    mtime = cache_file.stat().st_mtime
                    if (time.time() - mtime) < self.cache_ttl:
                        payload = json.loads(cache_file.read_text(encoding="utf-8"))
                        if isinstance(payload, dict) and "records" in payload:
                            loaded = payload["records"]
                            is_from_cache = True
                            cache_mtimes.append(float(payload.get("timestamp", mtime)))
                        elif isinstance(payload, list):
                            loaded = payload
                            is_from_cache = True
                            cache_mtimes.append(mtime)
                except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                    logger.warning(f"Failed to read COD search cache {cache_file}: {exc}")
                    loaded = None

            if not is_from_cache:
                all_from_cache = False
                params: dict[str, str] = {"format": "json"}
                for i, el in enumerate(q_elems, 1):
                    if i <= 8:
                        params[f"el{i}"] = el.strip()

                query_str = urllib.parse.urlencode(params)
                full_url = f"{self.SEARCH_URL}?{query_str}"
                logger.info(f"Querying COD: {full_url}")

                req = urllib.request.Request(
                    full_url,
                    headers={"User-Agent": "Antigravity-XRD-Agent/1.0"},
                )
                for attempt in range(1, 4):
                    try:
                        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                            data = resp.read()
                            loaded = json.loads(data.decode("utf-8"))
                            break
                    except (
                        urllib.error.URLError,
                        TimeoutError,
                        json.JSONDecodeError,
                        OSError,
                    ) as e:
                        logger.warning(f"COD search query attempt {attempt}/3 failed: {e}")
                        if attempt < 3:
                            time.sleep(1.0)

                if loaded is not None:
                    try:
                        cache_payload = {
                            "timestamp": time.time(),
                            "query": q_elems,
                            "records": loaded if isinstance(loaded, list) else [],
                        }
                        cache_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
                    except OSError as err:
                        logger.warning(f"Failed to write COD search cache {cache_file}: {err}")

            if isinstance(loaded, list):
                for item in loaded:
                    fid = str(item.get("file", "")).strip()
                    if fid and fid not in seen_files:
                        seen_files.add(fid)
                        raw_results.append(item)

        if all_from_cache and cache_mtimes:
            self.last_cache_info = CacheStatus(
                cached=True,
                age=_format_cache_age(min(cache_mtimes)),
            )
        else:
            self.last_cache_info = CacheStatus()

        cleaned_records: list[StructureSearchRecord] = []
        for r in raw_results:
            file_id = str(r.get("file", "")).strip()
            if not file_id:
                continue
            raw_formula = (
                str(r.get("formula", "") or r.get("chemname", "")).replace("-", "").strip()
            )
            extracted = extract_elements(raw_formula)
            if not filter.matches(extracted):
                continue
            sg_raw = r.get("sg")
            sg = str(sg_raw).strip() if sg_raw and str(sg_raw).strip().lower() != "none" else ""
            vol = _float_or_none(r.get("vol"))
            cleaned_records.append(
                StructureSearchRecord(
                    id=f"cod-{file_id}",
                    source=DatabaseSource.COD,
                    formula=raw_formula,
                    spacegroup=sg,
                    a=_float_or_none(r.get("a")),
                    b=_float_or_none(r.get("b")),
                    c=_float_or_none(r.get("c")),
                    alpha=_float_or_none(r.get("alpha")),
                    beta=_float_or_none(r.get("beta")),
                    gamma=_float_or_none(r.get("gamma")),
                    volume=vol,
                    energy_above_hull=None,
                    is_stable=None,
                )
            )
        return cleaned_records

    def download_cif(self, cod_id: str, out_path: Path | None = None) -> Path | None:
        """Download CIF by COD ID from local HSRDB, local cache, or COD REST API."""
        cached_file = self.cache_dir / f"{cod_id}.cif"
        if cached_file.exists() and cached_file.stat().st_size > 0:
            logger.debug(f"Using cached COD CIF: {cached_file}")
            if out_path:
                out_path.write_bytes(cached_file.read_bytes())
                return out_path
            return cached_file

        try:
            from hs_rdb import HsRdbDatabase, HsRdbNotFoundError

            with HsRdbDatabase() as db:
                db.export_cif(cod_id, out_path=cached_file, by_cod_id=True)
            if cached_file.exists() and cached_file.stat().st_size > 0:
                logger.info(f"Extracted COD entry {cod_id} from local HSRDB -> {cached_file}")
                if out_path:
                    out_path.write_bytes(cached_file.read_bytes())
                    return out_path
                return cached_file
        except (ImportError, HsRdbNotFoundError, KeyError, ValueError, OSError) as exc:
            logger.debug("Local HSRDB export notice for {}: {}", cod_id, exc)

        url = self.CIF_URL_TEMPLATE.format(cod_id=cod_id)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Antigravity-XRD-Agent/1.0"},
        )
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    content = resp.read()
                    cached_file.write_bytes(content)
                    logger.info(f"Downloaded COD entry {cod_id} ({len(content)} bytes)")
                    if out_path:
                        out_path.write_bytes(content)
                        return out_path
                    return cached_file
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                logger.warning(
                    f"Failed to download CIF for COD ID {cod_id} (attempt {attempt}/3): {e}"
                )
                if attempt < 3:
                    time.sleep(1.0)
                else:
                    return None
        return None


#: Environment variable naming the interpreter of the retrieval environment. The
#: Materials Project client cannot share an interpreter with the engine (see
#: ``mp_query.py``), so the retrieval route lives in a second environment and this
#: variable is how it is addressed. Like ``PDF2_DATA_DIR`` it is configuration, read
#: from the environment and, when not exported, from the ``.env`` file.
MP_PYTHON_ENV = "MP_PYTHON"

#: The bridge that runs under that interpreter; the only file importing mp_api.client.
MP_BRIDGE_PATH = Path(__file__).resolve().parent / "mp_query.py"


class MaterialsProjectUnavailable(RuntimeError):
    """The retrieval route could not be used, and the reason is known.

    Raised instead of a bare import failure so the caller reports a named obstacle --
    a missing interpreter, a missing key, a refused connection -- rather than an
    exception raised inside a third-party package.
    """


def resolve_mp_interpreter() -> str | None:
    """Return the interpreter of the retrieval environment, or None when unaddressed.

    A path that does not exist is reported and treated as unaddressed: silently
    handing a wrong path to ``subprocess`` would produce a ``FileNotFoundError`` that
    says nothing about the configuration being the problem.
    """
    load_env_file()
    candidate = os.environ.get(MP_PYTHON_ENV)
    if not candidate or not candidate.strip():
        return None
    path = Path(candidate.strip())
    if not path.is_file():
        logger.warning("{} names {}, which is not an existing file", MP_PYTHON_ENV, candidate)
        return None
    return str(path)


def run_mp_query(request: dict[str, Any]) -> dict[str, Any]:
    """Send one request to the retrieval environment and return its answer.

    The request travels as JSON on stdin, which is also how the API key travels: a
    command line is readable by every process on the machine, so the key does not go
    into argv.

    :raises MaterialsProjectUnavailable: when no interpreter is addressed, the bridge is
        missing, the second environment refuses the request, or it answers with something
        that is not JSON. Each case names its own obstacle.
    """
    interpreter = resolve_mp_interpreter()
    if interpreter is None:
        raise MaterialsProjectUnavailable(
            f"the Materials Project route is not addressed in this environment: set "
            f"{MP_PYTHON_ENV} to the python executable of an environment with mp-api "
            f"installed (the engine's own interpreter cannot import it, because "
            f"mp_api.client needs NumPy >= 2 while the GSAS-II binaries need NumPy < 2)"
        )
    if not MP_BRIDGE_PATH.is_file():
        raise MaterialsProjectUnavailable(f"bridge module is missing: {MP_BRIDGE_PATH}")

    try:
        completed = subprocess.run(
            [interpreter, str(MP_BRIDGE_PATH)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
            # The return code is inspected below: a nonzero exit is turned into a named
            # obstacle with the bridge's own message, so raising here would hide it.
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MaterialsProjectUnavailable(
            f"could not run the retrieval interpreter {interpreter}: {exc}"
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit code {completed.returncode}"
        raise MaterialsProjectUnavailable(reason)

    try:
        answer = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MaterialsProjectUnavailable(
            f"the retrieval environment answered with something that is not JSON: {exc}"
        ) from exc
    if not isinstance(answer, dict):
        raise MaterialsProjectUnavailable("the retrieval environment answered with a non-object")
    return answer


class MaterialsProjectClient:
    """Client for Materials Project, addressed through the retrieval environment.

    The client keeps the cache, the freshness policy and the record reading; the network
    calls belong to the second interpreter, reached through :func:`run_mp_query`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        timeout: int = 30,
        cache_ttl: float = DEFAULT_SEARCH_CACHE_TTL,
    ) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "mp_cifs"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.search_cache_dir = self.cache_dir / "searches"
        self.search_cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.last_cache_info: CacheStatus = CacheStatus()

        load_env_file()
        self.api_key = api_key or os.environ.get("MP_API_KEY")

    def is_configured(self) -> bool:
        """Check if an API key is provided."""
        return bool(self.api_key and self.api_key.strip())

    def search(
        self,
        filter: ElementFilter | None = None,
        elements: list[str] | None = None,
        refresh: bool = False,
    ) -> list[StructureSearchRecord]:
        """Search Materials Project for all structures matching chemical element criteria."""
        if filter is None:
            if not elements:
                self.last_cache_info = CacheStatus()
                return []
            filter = ElementFilter.from_raw(all_elements=elements)

        if filter.strict:
            chemsys = "-".join(sorted(filter.all_of | filter.any_of))
            cache_file = self.search_cache_dir / f"mp_strict_{chemsys}.json"
        elif filter.all_of:
            chemsys = "-".join(sorted(filter.all_of))
            cache_file = self.search_cache_dir / f"mp_all_{chemsys}.json"
        elif filter.any_of:
            chemsys = "-".join(sorted(filter.any_of))
            cache_file = self.search_cache_dir / f"mp_any_{chemsys}.json"
        else:
            self.last_cache_info = CacheStatus()
            return []

        raw_docs: list[Any] = []
        is_from_cache = False
        cache_mtime = 0.0

        if not refresh and cache_file.exists():
            try:
                mtime = cache_file.stat().st_mtime
                if (time.time() - mtime) < self.cache_ttl:
                    payload = json.loads(cache_file.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and "records" in payload:
                        raw_docs = payload["records"]
                        is_from_cache = True
                        cache_mtime = float(payload.get("timestamp", mtime))
                    elif isinstance(payload, list):
                        raw_docs = payload
                        is_from_cache = True
                        cache_mtime = mtime
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning(f"Failed to read MP search cache {cache_file}: {exc}")
                raw_docs = []

        if not is_from_cache:
            if not self.is_configured():
                logger.warning(
                    "Materials Project API key is missing. "
                    "Please add MP_API_KEY to your .env file in the project root. "
                    "Get your key at: https://materialsproject.org/dashboard"
                )
                self.last_cache_info = CacheStatus()
                return []

            try:
                # The element filter stays here, where its single definition lives
                # (ElementFilter); the bridge receives the resolved query and carries no
                # copy of the chemistry.
                if filter.strict:
                    request: dict[str, Any] = {
                        "action": "search",
                        "api_key": self.api_key.strip(),
                        "mode": "strict",
                        "chemsys": chemsys,
                    }
                elif filter.all_of:
                    request = {
                        "action": "search",
                        "api_key": self.api_key.strip(),
                        "mode": "all",
                        "elements": sorted(filter.all_of),
                        "exclude_elements": sorted(filter.none_of),
                    }
                else:
                    request = {
                        "action": "search",
                        "api_key": self.api_key.strip(),
                        "mode": "any",
                        "any_elements": sorted(filter.any_of),
                        "exclude_elements": sorted(filter.none_of),
                    }

                serialized_records = list(run_mp_query(request)["records"])

                raw_docs = serialized_records
                try:
                    cache_payload = {
                        "timestamp": time.time(),
                        "query_key": chemsys,
                        "records": serialized_records,
                    }
                    cache_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
                except OSError as exc:
                    logger.warning(f"Failed to write MP search cache {cache_file}: {exc}")
            except (MaterialsProjectUnavailable, KeyError, TypeError) as exc:
                logger.warning(f"Materials Project query failed: {exc}")
                self.last_cache_info = CacheStatus()
                return []

        if is_from_cache:
            self.last_cache_info = CacheStatus(cached=True, age=_format_cache_age(cache_mtime))
        else:
            self.last_cache_info = CacheStatus()

        def _extract_field(doc_obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(doc_obj, dict):
                return doc_obj.get(key, default)
            return getattr(doc_obj, key, default)

        results: list[StructureSearchRecord] = []
        for doc in raw_docs:
            mat_id = str(_extract_field(doc, "material_id", ""))
            formula_str = str(_extract_field(doc, "formula_pretty", ""))
            extracted = extract_elements(formula_str)
            if not filter.matches(extracted):
                continue
            e_hull = float(_extract_field(doc, "energy_above_hull", 0.0) or 0.0)
            is_st = bool(_extract_field(doc, "is_stable", False))
            sym_obj = _extract_field(doc, "symmetry", None)
            sg_sym = ""
            if sym_obj:
                sg_sym = str(
                    getattr(sym_obj, "symbol", "")
                    or (sym_obj.get("symbol") if isinstance(sym_obj, dict) else "")
                    or sym_obj
                )
            vol_val = _extract_field(doc, "volume", None)
            vol = float(vol_val) if vol_val is not None else None

            st = _extract_field(doc, "structure", None)
            lat = getattr(st, "lattice", None) if st is not None else None
            if lat is None and isinstance(st, dict):
                lat = st.get("lattice")

            if lat is not None:
                a = _float_or_none(
                    getattr(lat, "a", None) if not isinstance(lat, dict) else lat.get("a")
                )
                b = _float_or_none(
                    getattr(lat, "b", None) if not isinstance(lat, dict) else lat.get("b")
                )
                c = _float_or_none(
                    getattr(lat, "c", None) if not isinstance(lat, dict) else lat.get("c")
                )
                alpha = _float_or_none(
                    getattr(lat, "alpha", None) if not isinstance(lat, dict) else lat.get("alpha")
                )
                beta = _float_or_none(
                    getattr(lat, "beta", None) if not isinstance(lat, dict) else lat.get("beta")
                )
                gamma = _float_or_none(
                    getattr(lat, "gamma", None) if not isinstance(lat, dict) else lat.get("gamma")
                )
            else:
                a = b = c = alpha = beta = gamma = None

            clean_mp_id = mat_id if mat_id.startswith("mp-") else f"mp-{mat_id}"
            results.append(
                StructureSearchRecord(
                    id=clean_mp_id,
                    source=DatabaseSource.MP,
                    formula=formula_str,
                    spacegroup=sg_sym,
                    a=a,
                    b=b,
                    c=c,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    volume=vol,
                    energy_above_hull=e_hull,
                    is_stable=is_st,
                )
            )

        results.sort(key=lambda r: (not bool(r.is_stable), float(r.energy_above_hull or 0.0)))
        return results

    def download_cif(
        self,
        material_id: str,
        out_path: Path | None = None,
        formula_hint: str | None = None,
        symprec: float = 0.01,
    ) -> Path | None:
        """Download structure, standardize to conventional cell, remove oxidation charges, and export clean CIF."""
        safe_id = material_id.replace("/", "_")
        clean_id = safe_id if safe_id.startswith("mp-") else f"mp-{safe_id}"

        # If cache exists for this ID with or without formula prefix
        cached_matches = list(self.cache_dir.glob(f"*{clean_id}.cif"))
        if cached_matches and cached_matches[0].stat().st_size > 0:
            cached_file = cached_matches[0]
            logger.debug(f"Using cached Materials Project CIF: {cached_file}")
            if out_path:
                dest = out_path / cached_file.name if out_path.is_dir() else out_path
                dest.write_bytes(cached_file.read_bytes())
                return dest
            return cached_file

        if not self.is_configured():
            logger.warning(
                f"Cannot download {material_id}: Materials Project API key is not configured. "
                "Please add MP_API_KEY to your .env file in the project root."
            )
            return None

        try:
            answer = run_mp_query(
                {
                    "action": "download",
                    "api_key": self.api_key.strip(),
                    "material_id": clean_id,
                    "symprec": symprec,
                }
            )

            # The structure is fetched and standardized in the other environment and
            # arrives as CIF text; the file is written here, in the directory this client
            # owns as its cache, rather than wherever the other interpreter ran.
            formula = formula_hint or str(answer.get("reduced_formula") or clean_id)
            cif_name = f"{formula}_{clean_id}.cif"
            cached_file = self.cache_dir / cif_name
            cached_file.write_text(str(answer["cif_text"]), encoding="utf-8")
            logger.info(f"Retrieved and sanitized MP entry {clean_id} -> {cached_file.name}")

            if out_path:
                dest = out_path / cif_name if out_path.is_dir() else out_path
                dest.write_bytes(cached_file.read_bytes())
                return dest
            return cached_file
        except (MaterialsProjectUnavailable, KeyError, OSError) as exc:
            logger.warning(f"Failed to retrieve MP structure {material_id}: {exc}")
            return None


def get_effective_radius(element_symbol: str, cn: str | int | None = None) -> float:
    """Return effective crystal/ionic radius from pymatgen Shannon database for given CN."""
    target_cn = normalize_cn(cn)
    target_int = ROMAN_TO_INT.get(target_cn, 6)

    # 1. Попытка разобрать как Species (например Sm3+, Fe2+, O2-)
    try:
        sp = Species.from_str(element_symbol)
        shannon_data = sp._el.data.get("Shannon radii", {})
        oxi_key = str(int(sp.oxi_state))
        if oxi_key in shannon_data:
            cns = shannon_data[oxi_key]
            best_cn = min(cns.keys(), key=lambda c: abs(ROMAN_TO_INT.get(c, 6) - target_int))
            spin_dict = cns[best_cn]
            for pref in ["", "High Spin", "Low Spin"]:
                if pref in spin_dict:
                    return float(spin_dict[pref]["ionic_radius"])
            return float(next(iter(spin_dict.values()))["ionic_radius"])
    except (ValueError, KeyError, AttributeError):
        pass  # Fall through to neutral Element lookup

    # 2. Попытка для нейтрального символа элемента
    try:
        el = Element(element_symbol)
        shannon_data = el.data.get("Shannon radii", {})
        if shannon_data:
            neg_oxs = [ox for ox in shannon_data if ox.startswith("-")]
            common_oxs = [
                str(int(ox))
                for ox in (el.common_oxidation_states or el.icsd_oxidation_states or ())
            ]
            candidate_oxs = neg_oxs + [ox for ox in common_oxs if ox in shannon_data]
            if not candidate_oxs:
                candidate_oxs = list(shannon_data.keys())

            best_ox = candidate_oxs[0]
            cns = shannon_data[best_ox]
            best_cn = min(cns.keys(), key=lambda c: abs(ROMAN_TO_INT.get(c, 6) - target_int))
            spin_dict = cns[best_cn]
            for pref in ["", "High Spin", "Low Spin"]:
                if pref in spin_dict:
                    return float(spin_dict[pref]["ionic_radius"])
            return float(next(iter(spin_dict.values()))["ionic_radius"])

        if el.average_ionic_radius:
            return float(el.average_ionic_radius)
        if el.atomic_radius:
            return float(el.atomic_radius)
        if el.atomic_radius_calculated:
            return float(el.atomic_radius_calculated)
    except (ValueError, KeyError, AttributeError) as exc:
        logger.debug(f"Could not determine radius for {element_symbol}: {exc}")

    return 1.0


def generate_isostructural_analog(
    prototype_cif: Path,
    species_mapping: dict[str, str] | None = None,
    out_cif: Path = Path("./analog.cif"),
    anion_symbol: str = "O",
    cation_cn: str | int | None = None,
    anion_cn: str | int | None = None,
    scale_to_d: float | None = None,
    hkl: tuple[int, int, int] | list[int] | None = None,
    symprec: float = 0.01,
) -> Path:
    """Generate an isostructural analog by substituting species and/or scaling lattice parameters.

    Scaling modes:
    1. Direct experimental d-spacing:
       k = d_exp / d_calc(hkl)
       V_new = V_old * (k ** 3)
       Universal for all 7 crystal systems via metric tensor dilatational invariance.
    2. Ionic radius Shannon scaling:
       r_ratio = (r_new_cat + r_anion) / (r_old_cat + r_anion)
       V_new = V_old * (r_ratio ** 3)

    The CIF is written with symmetry re-detection (``symprec``), so the analog keeps the
    prototype's space group instead of being emitted as a P1 cell listing every site.
    That matters when the analog is used as a Rietveld phase: a P1 description of a
    cubic structure carries no space-group constraints and is not isostructural in the
    crystallographic sense.
    """
    struct = Structure.from_file(str(prototype_cif))
    old_vol = struct.volume

    if species_mapping:
        struct.replace_species(species_mapping)

    if scale_to_d is not None and hkl is not None:
        hkl_tuple = tuple(int(x) for x in hkl)
        d_calc = float(struct.lattice.d_hkl(hkl_tuple))
        if d_calc <= 1e-6:
            raise ValueError(f"Calculated d_spacing for hkl {hkl_tuple} is non-physical: {d_calc}")
        k = float(scale_to_d) / d_calc
        new_vol = old_vol * (k**3)
        logger.info(
            f"Analog lattice scaling via d-spacing: hkl={hkl_tuple}, d_calc={d_calc:.4f} Å, "
            f"d_exp={scale_to_d:.4f} Å -> linear factor k={k:.4f}, volume {old_vol:.2f} -> {new_vol:.2f} Å³"
        )
        struct.scale_lattice(new_vol)
    elif species_mapping:
        auto_cat_cn = cation_cn
        auto_ani_cn = anion_cn

        if auto_cat_cn is None or auto_ani_cn is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nn = CrystalNN()
                    for idx, site in enumerate(struct):
                        sym = site.specie.symbol
                        if auto_cat_cn is None and sym in species_mapping:
                            auto_cat_cn = nn.get_cn(struct, idx)
                        elif auto_ani_cn is None and sym == anion_symbol:
                            auto_ani_cn = nn.get_cn(struct, idx)
                        if auto_cat_cn is not None and auto_ani_cn is not None:
                            break
            except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as exc:
                logger.debug(f"CrystalNN CN auto-detection notice: {exc}")

        r_anion = get_effective_radius(anion_symbol, cn=auto_ani_cn or "VI")

        scale_factors: list[float] = []
        for old_sp, new_sp in species_mapping.items():
            r_old = get_effective_radius(old_sp, cn=auto_cat_cn or "VI")
            r_new = get_effective_radius(new_sp, cn=auto_cat_cn or "VI")
            factor = (r_new + r_anion) / (r_old + r_anion)
            scale_factors.append(factor)

        avg_linear_scale = sum(scale_factors) / len(scale_factors) if scale_factors else 1.0
        vol_scale = avg_linear_scale**3
        new_vol = old_vol * vol_scale

        logger.info(
            f"Analog generation: replacing {species_mapping} (CN_cat={auto_cat_cn or 'VI'}, "
            f"CN_ani={auto_ani_cn or 'VI'}), linear scale {avg_linear_scale:.4f}, "
            f"volume {old_vol:.2f} -> {new_vol:.2f} A^3"
        )
        struct.scale_lattice(new_vol)

    import gemmi

    out_cif.parent.mkdir(parents=True, exist_ok=True)
    doc = gemmi.cif.read_file(str(prototype_cif))
    block = doc.sole_block()
    sg_hm_raw = (
        block.find_value("_symmetry_space_group_name_H-M")
        or block.find_value("_space_group_name_H-M_alt")
        or ""
    ).strip("'\" ")
    if sg_hm_raw and sg_hm_raw.replace(" ", "") != "P1":
        lat = struct.lattice
        block.set_pair("_cell_length_a", f"{float(lat.a):.6f}")
        block.set_pair("_cell_length_b", f"{float(lat.b):.6f}")
        block.set_pair("_cell_length_c", f"{float(lat.c):.6f}")
        block.set_pair("_cell_angle_alpha", f"{float(lat.alpha):.4f}")
        block.set_pair("_cell_angle_beta", f"{float(lat.beta):.4f}")
        block.set_pair("_cell_angle_gamma", f"{float(lat.gamma):.4f}")
        block.set_pair("_cell_volume", f"{float(lat.volume):.4f}")
        if species_mapping:
            type_col = block.find_values("_atom_site_type_symbol")
            label_col = block.find_values("_atom_site_label")
            if type_col:
                for i in range(len(type_col)):
                    raw_sym = str(type_col[i]).strip("'\" ")
                    base_m = re.match(r"([A-Z][a-z]?)(.*)", raw_sym)
                    if base_m and base_m.group(1) in species_mapping:
                        old_el = base_m.group(1)
                        new_el = species_mapping[old_el]
                        type_col[i] = f"{new_el}{base_m.group(2)}"
                        if label_col and i < len(label_col):
                            lbl = str(label_col[i]).strip("'\" ")
                            if lbl.startswith(old_el):
                                label_col[i] = new_el + lbl[len(old_el) :]
            for f_tag in ("_chemical_formula_sum", "_chemical_formula_structural"):
                if block.find_value(f_tag):
                    block.set_pair(f_tag, f"'{struct.composition.reduced_formula}'")
        doc.write_file(str(out_cif))
    else:
        CifWriter(struct, symprec=symprec, refine_struct=False).write_file(str(out_cif))
    logger.info(f"Saved generated analog CIF: {out_cif}")
    return out_cif


def search_cifs(
    filter: ElementFilter | None = None,
    elements: list[str] | None = None,
    source: DatabaseSource | str = DatabaseSource.ALL,
    limit: int | None = 50,
    cod_client: CodClient | None = None,
    mp_client: MaterialsProjectClient | None = None,
    refresh: bool = False,
    subfiles: frozenset[str] | None = None,
) -> SearchCifsResponse:
    """Search COD and/or Materials Project for matching structures by element filter."""
    if filter is None:
        if not elements:
            return SearchCifsResponse(
                total_count=0,
                displayed_count=0,
                limit=limit,
                filter={},
                filter_summary="None",
                source=source,
                cache_info={},
                results=[],
            )
        filter = ElementFilter.from_raw(all_elements=elements)

    all_results: list[StructureSearchRecord] = []
    cache_info: dict[str, CacheStatus] = {}

    if source in (DatabaseSource.MP, DatabaseSource.ALL):
        if mp_client is None:
            mp_client = MaterialsProjectClient()
        mp_records = mp_client.search(filter=filter, refresh=refresh)
        all_results.extend(mp_records)
        cache_info["mp"] = getattr(mp_client, "last_cache_info", CacheStatus())

    if source in (DatabaseSource.COD, DatabaseSource.ALL):
        if cod_client is None:
            cod_client = CodClient()
        cod_records = cod_client.search(filter=filter, refresh=refresh, subfiles=subfiles)
        all_results.extend(cod_records)
        cache_info["cod"] = getattr(cod_client, "last_cache_info", CacheStatus())

    total_count = len(all_results)
    if limit is not None and limit > 0:
        displayed_results = all_results[:limit]
    else:
        displayed_results = all_results

    return SearchCifsResponse(
        total_count=total_count,
        displayed_count=len(displayed_results),
        limit=limit,
        filter={
            "all_of": sorted(filter.all_of),
            "any_of": sorted(filter.any_of),
            "none_of": sorted(filter.none_of),
            "strict": filter.strict,
        },
        filter_summary=filter.summary_str(),
        source=source,
        cache_info=cache_info,
        results=displayed_results,
    )


def download_cifs_by_ids(
    ids: list[str],
    out_dir: Path,
    cod_client: CodClient | None = None,
    mp_client: MaterialsProjectClient | None = None,
) -> list[Path]:
    """Download CIF files for explicitly specified structure IDs.

    Polymorphic ID routing:
    - IDs starting with 'mp-' / 'mp_' -> downloaded via MaterialsProjectClient
    - IDs starting with 'cod-' / 'cod_' or numeric -> downloaded via CodClient
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if cod_client is None:
        cod_client = CodClient()
    if mp_client is None:
        mp_client = MaterialsProjectClient()

    downloaded_paths: list[Path] = []
    for raw_id in ids:
        cid = raw_id.strip()
        if not cid:
            continue

        cid_lower = cid.lower()
        if cid_lower.startswith(("mp-", "mp_")):
            clean_id = cid[3:] if cid_lower.startswith("mp_") else cid
            path = mp_client.download_cif(clean_id, out_path=out_dir)
            if path and path.exists():
                downloaded_paths.append(path)
        else:
            if cid_lower.startswith(("cod-", "cod_")):
                cod_num = cid[4:]
            else:
                cod_num = cid
            dest = out_dir / f"cod_{cod_num}.cif"
            path = cod_client.download_cif(cod_num, out_path=dest)
            if path and path.exists():
                downloaded_paths.append(path)

    logger.info(f"Downloaded {len(downloaded_paths)} requested CIFs into {out_dir}")
    return downloaded_paths


def format_search_table(search_data: SearchCifsResponse) -> str:
    """Format search_cifs metadata into clean human-readable console table."""
    total = search_data.total_count
    displayed = search_data.displayed_count
    limit = search_data.limit
    crit_str = search_data.filter_summary or "None"
    source_str = str(search_data.source)
    cache_info = search_data.cache_info

    if source_str == "mp" and "mp" in cache_info:
        c = cache_info["mp"]
        source_display = f"mp [cached {c.age}]" if c.cached else "mp [live]"
    elif source_str == "cod" and "cod" in cache_info:
        c = cache_info["cod"]
        source_display = f"cod [cached {c.age}]" if c.cached else "cod [live]"
    elif source_str == "all" and cache_info:
        mp_c = cache_info.get("mp", CacheStatus())
        cod_c = cache_info.get("cod", CacheStatus())
        mp_tag = f"cached {mp_c.age}" if mp_c.cached else "live"
        cod_tag = f"cached {cod_c.age}" if cod_c.cached else "live"
        source_display = f"all (mp: {mp_tag}, cod: {cod_tag})"
    else:
        source_display = source_str

    results = search_data.results

    if total == 0:
        return f"No structures found for criteria ({crit_str}, source: {source_display})."

    lines: list[str] = []
    if limit is not None and limit > 0 and total > displayed:
        lines.append(
            f"Found {total} total structures ({crit_str}, source: {source_display}, "
            f"showing first {displayed}, limit: {limit}):"
        )
    else:
        lines.append(f"Found {total} total structures ({crit_str}, source: {source_display}):")

    def _fmt(val: Any, prec: int) -> str:
        if val is None:
            return "-"
        try:
            return f"{float(val):.{prec}f}"
        except (ValueError, TypeError):
            return str(val)

    header = (
        f"{'ID':<13} {'Source':<6} {'Formula':<12} {'Space Group':<12} "
        f"{'a':<7} {'b':<7} {'c':<7} {'alpha':<6} {'beta':<6} {'gamma':<6} "
        f"{'Vol [A^3]':<9} {'E_hull':<8} {'Stable':<6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        rid = str(r.id)
        src = str(r.source)
        formula = str(r.formula)
        sg_val = str(r.spacegroup or "").strip()
        sg = sg_val if sg_val and sg_val.lower() != "none" else "-"
        a_str = _fmt(r.a, 3)
        b_str = _fmt(r.b, 3)
        c_str = _fmt(r.c, 3)
        al_str = _fmt(r.alpha, 1)
        be_str = _fmt(r.beta, 1)
        ga_str = _fmt(r.gamma, 1)
        vol = _fmt(r.volume, 1)
        ehull = _fmt(r.energy_above_hull, 3)
        stable = str(r.is_stable) if r.is_stable is not None else "-"
        lines.append(
            f"{rid:<13} {src:<6} {formula:<12} {sg:<12} "
            f"{a_str:<7} {b_str:<7} {c_str:<7} {al_str:<6} {be_str:<6} {ga_str:<6} "
            f"{vol:<9} {ehull:<8} {stable:<6}"
        )

    return "\n".join(lines)


def validate_cif_identity(
    cif_path: Path,
    expected_elements: list[str] | None = None,
    expected_formula: str | None = None,
) -> bool:
    """Проверка химического состава структуры против ожидаемых элементов и формулы.

    Блокирует подмену фаз (например, Sm2O3 под видом Sm_metal.cif или Si под видом SiO2_quartz.cif).
    """
    struct = Structure.from_file(str(cif_path))
    actual_elements = {el.symbol for el in struct.composition.elements}

    if expected_elements:
        exp_set = set(expected_elements)
        # Если ожидался чистый моно-элемент (металл), а в составе есть кислород O или другие элементы — браковать!
        if len(expected_elements) == 1 and actual_elements != exp_set:
            raise ValueError(
                f"CIF {cif_path.name} содержит элементы {actual_elements}, ожидался чистый {expected_elements[0]}"
            )
        if not actual_elements.issubset(exp_set):
            raise ValueError(
                f"CIF {cif_path.name} содержит посторонние элементы {actual_elements - exp_set}, ожидались подмножества {exp_set}"
            )

    if expected_formula:
        red_formula = struct.composition.reduced_formula.replace(" ", "")
        req_formula = expected_formula.replace(" ", "")
        if red_formula.lower() != req_formula.lower():
            raise ValueError(
                f"CIF {cif_path.name} имеет формулу {red_formula}, ожидалась {req_formula}"
            )

    # Доменно-нейтральная проверка чистоты металла: если заявлен металл, а присутствуют неметаллы
    stem_lower = cif_path.stem.lower()
    if "_metal" in stem_lower and any(
        not getattr(el, "is_metal", False) for el in struct.composition.elements
    ):
        raise ValueError(
            f"CIF {cif_path.name} заявлен как чистый металл, но содержит неметаллические элементы: {actual_elements}"
        )

    return True


def harvest_local_cifs(
    local_dir: Path,
    out_dir: Path | None = None,
    filter: ElementFilter | None = None,
    elements: list[str] | None = None,
) -> list[Path]:
    """Scan local directory for CIF files, filter by element criteria with pymatgen,
    and optionally copy matching structures into out_dir."""
    if not local_dir.exists():
        logger.warning(f"Local CIF directory not found: {local_dir}")
        return []

    if filter is None and elements is not None:
        filter = ElementFilter.from_raw(all_elements=elements)

    matched_cifs: list[Path] = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for cif_path in sorted(local_dir.glob("**/*.cif")):
        try:
            struct = Structure.from_file(str(cif_path))
            struct_elems = {el.symbol for el in struct.composition.elements}

            if filter and not filter.matches(struct_elems):
                continue

            if out_dir is not None:
                dest_path = out_dir / cif_path.name
                if dest_path.resolve() != cif_path.resolve():
                    dest_path.write_bytes(cif_path.read_bytes())
                matched_cifs.append(dest_path)
            else:
                matched_cifs.append(cif_path)
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            logger.debug(f"Skipping unparseable or rejected local CIF {cif_path.name}: {e}")
            continue

    logger.info(f"Harvested {len(matched_cifs)} valid CIFs from local repository {local_dir}")
    return matched_cifs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="COD & Materials Project CIF Explorer and Downloader (Strict Two-Stage)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # 1. search
    p_search = subparsers.add_parser(
        "search", help="Search for CIF structures by chemical elements"
    )
    p_search.add_argument(
        "--all",
        type=str,
        help="All of (AND): comma-separated elements that must be present (e.g. 'Pb,S')",
    )
    p_search.add_argument(
        "--elements",
        type=str,
        help="Alias for --all (e.g. 'Sm,S' or 'Ba,Ti,O')",
    )
    p_search.add_argument(
        "--any",
        type=str,
        help="At least one of (OR): comma-separated alternative elements (e.g. 'Fe,Co,Ni')",
    )
    p_search.add_argument(
        "--none",
        type=str,
        help="None of (NOT / NOR): comma-separated forbidden elements (e.g. 'C,H,Cl')",
    )
    p_search.add_argument(
        "--strict",
        action="store_true",
        help="Closed chemical system: forbids any elements outside --all and --any",
    )
    p_search.add_argument(
        "--source",
        choices=["cod", "mp", "all"],
        default="all",
        help="Database source: cod, mp, or all (default: all)",
    )
    p_search.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max results to display in table/JSON (0 = no limit, reports total_count)",
    )
    p_search.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted table",
    )
    p_search.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass search cache and query database directly",
    )
    p_search.add_argument(
        "--subfiles",
        type=str,
        default=None,
        help="Comma-separated subfile filter for local COD HSRDB (inorganic, organic, mineral, metallic)",
    )

    # 2. download
    p_download = subparsers.add_parser(
        "download", help="Download CIF files by explicit structure IDs"
    )
    p_download.add_argument(
        "--ids",
        required=True,
        type=str,
        help="Comma-separated structure IDs (e.g. 'mp-1269,cod-1538008,1538008')",
    )
    p_download.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./cifs"),
        help="Directory to save CIF files (default: ./cifs)",
    )

    # 3. analog
    p_analog = subparsers.add_parser(
        "analog", help="Generate isostructural analog with ionic radius or d-spacing scaling"
    )
    p_analog.add_argument("--prototype", required=True, type=Path, help="Prototype CIF file")
    p_analog.add_argument(
        "--substitute",
        type=str,
        default=None,
        help="Species substitution in format Old:New (e.g. Nd:Sm). Optional if --scale-to-d is provided.",
    )
    p_analog.add_argument(
        "--anion", type=str, default="O", help="Anion symbol for bond distance scaling (default: O)"
    )
    p_analog.add_argument(
        "--cation-cn", type=str, help="Cation coordination number (e.g. 8 or VIII)"
    )
    p_analog.add_argument("--anion-cn", type=str, help="Anion coordination number (e.g. 4 or IV)")
    p_analog.add_argument(
        "--scale-to-d",
        type=float,
        default=None,
        help="Direct experimental d-spacing [A] to scale unit cell",
    )
    p_analog.add_argument(
        "--hkl",
        type=str,
        default=None,
        help="Miller indices (h,k,l) for --scale-to-d (e.g. '2,0,0')",
    )
    p_analog.add_argument(
        "--out-cif", required=True, type=Path, help="Output path for generated analog CIF"
    )

    # 4. harvest
    p_harvest = subparsers.add_parser(
        "harvest", help="Scan local directory for CIF files filtered by elements"
    )
    p_harvest.add_argument(
        "--local-dir", required=True, type=Path, help="Local directory containing CIF files"
    )
    p_harvest.add_argument(
        "--all",
        type=str,
        help="All of (AND): comma-separated elements that must be present",
    )
    p_harvest.add_argument("--elements", type=str, help="Alias for --all")
    p_harvest.add_argument(
        "--any",
        type=str,
        help="At least one of (OR): comma-separated alternative elements",
    )
    p_harvest.add_argument(
        "--none",
        type=str,
        help="None of (NOT / NOR): comma-separated forbidden elements",
    )
    p_harvest.add_argument(
        "--strict",
        action="store_true",
        help="Closed chemical system: forbids any elements outside --all and --any",
    )
    p_harvest.add_argument("--out-dir", type=Path, help="Directory to copy matched CIF files")

    args = parser.parse_args()

    if args.command == "search":
        all_val = args.all
        if args.elements:
            if all_val and all_val != args.elements:
                parser.error("Cannot specify both --all and --elements with conflicting values.")
            all_val = args.elements

        try:
            elem_filter = ElementFilter.from_raw(
                all_elements=all_val,
                any_elements=args.any,
                none_elements=args.none,
                strict=args.strict,
            )
        except ValueError as err:
            logger.error(f"Invalid element filter: {err}")
            return 1

        subfiles_set = (
            frozenset(s.strip() for s in args.subfiles.split(",") if s.strip())
            if args.subfiles
            else None
        )
        data = search_cifs(
            filter=elem_filter,
            source=args.source,
            limit=args.limit,
            refresh=args.refresh,
            subfiles=subfiles_set,
        )
        if args.json:
            print(json.dumps(data.to_dict(), indent=2))
        else:
            print(format_search_table(data))
        return 0

    if args.command == "download":
        id_list = [i.strip() for i in args.ids.split(",") if i.strip()]
        saved = download_cifs_by_ids(ids=id_list, out_dir=args.out_dir)
        print(f"Downloaded {len(saved)} CIFs into {args.out_dir}:")
        for p in saved:
            print(f"  - {p}")
        return 0

    if args.command == "analog":
        if not args.substitute and not args.scale_to_d:
            parser.error("Must provide either --substitute or --scale-to-d (with --hkl).")
        if args.scale_to_d is not None and not args.hkl:
            parser.error("--scale-to-d requires --hkl (e.g. --hkl '2,0,0').")

        mapping = {}
        if args.substitute:
            for pair in args.substitute.split(","):
                old_s, new_s = pair.split(":")  # noqa: PU001
                mapping[old_s.strip()] = new_s.strip()

        hkl_tuple = None
        if args.hkl:
            hkl_tuple = tuple(int(x.strip()) for x in args.hkl.split(","))

        args.out_cif.parent.mkdir(parents=True, exist_ok=True)
        generate_isostructural_analog(
            prototype_cif=args.prototype,
            species_mapping=mapping if mapping else None,
            out_cif=args.out_cif,
            anion_symbol=args.anion,
            cation_cn=args.cation_cn,
            anion_cn=args.anion_cn,
            scale_to_d=args.scale_to_d,
            hkl=hkl_tuple,
        )
        print(f"Generated analog CIF: {args.out_cif}")
        return 0

    if args.command == "harvest":
        all_val = args.all or args.elements
        elem_filter = None
        if all_val or args.any or args.none or args.strict:
            try:
                elem_filter = ElementFilter.from_raw(
                    all_elements=all_val,
                    any_elements=args.any,
                    none_elements=args.none,
                    strict=args.strict,
                )
            except ValueError as err:
                logger.error(f"Invalid element filter: {err}")
                return 1

        harvested = harvest_local_cifs(
            local_dir=args.local_dir,
            out_dir=args.out_dir,
            filter=elem_filter,
        )
        print(f"Harvested {len(harvested)} CIFs from {args.local_dir}")
        for p in harvested:
            print(f"  - {p}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
