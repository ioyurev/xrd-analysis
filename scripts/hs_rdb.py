"""Read-only interoperability adapter for local COD SQLite archives (.hsrdb).

Provides heuristic-free access to user-supplied SQLite archives of the public-domain
Crystallography Open Database (COD, CC0 1.0) addressed via COD_HSRDB_PATH:
- 128-bit element bitmask filtering (all_of, any_of, none_of, strict) in SQLite signed int64;
- Indexed SQL filtering by unit cell metrics, crystal system, space group, and subfiles;
- Fast first-pass peak matching against 60-byte strongest-lines blobs (pattern_strongestlines);
- Deterministic decoding of compressed Phases.Phase blobs (CRC32 + raw Deflate + UTF-16LE records)
  into atomic coordinates, Wyckoff positions, occupancies, and B_iso values;
- Standards-compliant CIF generation via gemmi.cif.Document.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import struct
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gemmi
import numpy as np
from env_file import load_env_file
from loguru import logger
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cif_retriever import ElementFilter
    from pdf2_reader import CrystallographicFilter

__all__ = [
    "COD_HSRDB_ENV",
    "AtomSite",
    "CodStructureRecord",
    "ElementMasks",
    "HsRdbDatabase",
    "HsRdbNotFoundError",
    "PeakFilter",
    "PhaseData",
    "SqlWhereClause",
    "build_crystal_sql_clauses",
    "build_element_sql_clauses",
    "elements_to_masks",
    "extract_cod_id",
    "masks_to_elements",
    "resolve_hsrdb_path",
    "to_signed_int64",
    "to_unsigned_int64",
]

COD_HSRDB_ENV = "COD_HSRDB_PATH"

Z_TO_ELEMENT: dict[int, str] = {z: gemmi.Element(z).name for z in range(1, 119)}
ELEMENT_TO_Z: dict[str, int] = {
    **{sym: z for z, sym in Z_TO_ELEMENT.items()},
    "D": 1,
    "T": 1,
}

CRYSTAL_SYSTEM_TO_HSRDB: dict[str, tuple[str, ...]] = {
    "cubic": ("C",),
    "c": ("C",),
    "tetragonal": ("T",),
    "t": ("T",),
    "orthorhombic": ("O",),
    "o": ("O",),
    "hexagonal": ("H",),
    "h": ("H",),
    "trigonal": ("R",),
    "rhombohedral": ("R",),
    "r": ("R",),
    "trigonal-hexagonal": ("H", "R"),
    "monoclinic": ("M",),
    "m": ("M",),
    "triclinic": ("A",),
    "anorthic": ("A",),
    "a": ("A",),
}

SUBFILE_MAP: dict[str, int] = {
    "inorganic": 0,
    "i": 0,
    "user inorganic": 0,
    "organic": 1,
    "o": 1,
    "user organic": 1,
    "mineral": 2,
    "m": 2,
    "user mineral": 2,
    "metallic": 3,
    "metal": 3,
    "user metallic": 3,
    "amorphous": 4,
    "user amorphous": 4,
    "ceramic": 5,
    "user ceramic": 5,
}

_COD_COMMENT_RE = re.compile(r"COD database code:\s*(\d+)")
_TEMP_COMMENT_RE = re.compile(r"Temperature:\s*([0-9.]+)\s*K", re.IGNORECASE)
_PRES_COMMENT_RE = re.compile(r"Pressure:\s*([0-9.]+)\s*(kPa|MPa|GPa)", re.IGNORECASE)
_WYCKOFF_RE = re.compile(r"^([0-9]+)[a-zA-Z]$")


class HsRdbNotFoundError(FileNotFoundError):
    """The COD HSRDB SQLite database was not located in configuration."""


@dataclass(frozen=True)
class ElementMasks:
    """Signed 64-bit integer bitmasks (e1, e2) for SQLite pattern_elements."""

    e1: int
    e2: int


@dataclass(frozen=True)
class SqlWhereClause:
    """SQL WHERE predicates and bound parameters."""

    clauses: list[str]
    params: list[Any]


def to_signed_int64(u: int) -> int:
    """Convert unsigned 64-bit integer to two's-complement signed int64 for SQLite."""
    u &= 0xFFFFFFFFFFFFFFFF
    return u if u < (1 << 63) else u - (1 << 64)


def to_unsigned_int64(s: int) -> int:
    """Convert SQLite signed int64 to unsigned 64-bit integer."""
    return s & 0xFFFFFFFFFFFFFFFF


def elements_to_masks(elements: Iterable[str]) -> ElementMasks:
    """Convert chemical element symbols to signed SQLite 64-bit masks (e1, e2).

    - e1: bits 1..63 correspond to atomic numbers Z = 1..63.
    - e2: bits 0..54 correspond to atomic numbers Z = 64..118 (bit = Z - 64).
    """
    e1_u = 0
    e2_u = 0
    for raw in elements:
        symbol = str(raw).strip().capitalize()
        z = ELEMENT_TO_Z.get(symbol)
        if z is None:
            continue
        if 1 <= z <= 63:
            e1_u |= 1 << z
        elif z >= 64:
            e2_u |= 1 << (z - 64)
    return ElementMasks(e1=to_signed_int64(e1_u), e2=to_signed_int64(e2_u))


def masks_to_elements(e1: int, e2: int) -> set[str]:
    """Convert SQLite signed int64 masks (e1, e2) back to IUPAC element symbols."""
    elems: set[str] = set()
    e1_u = to_unsigned_int64(e1)
    e2_u = to_unsigned_int64(e2)
    for z in range(1, 64):
        if (e1_u >> z) & 1:
            sym = Z_TO_ELEMENT.get(z)
            if sym:
                elems.add(sym)
    for b in range(64):
        if (e2_u >> b) & 1:
            sym = Z_TO_ELEMENT.get(64 + b)
            if sym:
                elems.add(sym)
    return elems


def resolve_hsrdb_path() -> Path:
    """Return the COD HSRDB file path declared in COD_HSRDB_PATH / .env.

    :raises HsRdbNotFoundError: when COD_HSRDB_PATH is unset or does not name a file.
    """
    load_env_file()
    env_val = os.environ.get(COD_HSRDB_ENV)
    if not env_val or not env_val.strip():
        raise HsRdbNotFoundError(
            f"COD HSRDB database not found: {COD_HSRDB_ENV} is unset. Point it at "
            "COD24_HS4x.hsrdb by exporting the variable or declaring it in .env."
        )
    resolved = Path(env_val.strip())
    if not resolved.is_file():
        raise HsRdbNotFoundError(
            f"{COD_HSRDB_ENV} names {resolved}, which is not an existing file"
        )
    return resolved


def extract_cod_id(product_id: str, comment: str | None = None) -> str:
    """Extract canonical 7-digit COD database code from Comment or ProductID.

    In .hsrdb COD archives, ProductID is '96' + (cod_id + 1), while Comment
    records 'COD database code: <cod_id>'.
    """
    if comment:
        m = _COD_COMMENT_RE.search(comment)
        if m:
            return m.group(1)
    clean = str(product_id).replace("-", "").strip()
    if clean.startswith("96") and len(clean) > 2 and clean[2:].isdigit():
        return str(max(1, int(clean[2:]) - 1))
    return clean


def build_element_sql_clauses(
    elem_filter: ElementFilter,
    table_alias: str = "pe",
) -> SqlWhereClause:
    """Translate ElementFilter into SQLite bitwise WHERE clauses on (e1, e2)."""
    pfx = f"{table_alias}." if table_alias else ""
    clauses: list[str] = []
    params: list[Any] = []

    if elem_filter.all_of:
        req = elements_to_masks(elem_filter.all_of)
        if req.e1 != 0:
            clauses.append(f"({pfx}e1 & ?) = ?")
            params.extend([req.e1, req.e1])
        if req.e2 != 0:
            clauses.append(f"({pfx}e2 & ?) = ?")
            params.extend([req.e2, req.e2])

    if elem_filter.any_of:
        any_m = elements_to_masks(elem_filter.any_of)
        if any_m.e1 != 0 and any_m.e2 != 0:
            clauses.append(f"(({pfx}e1 & ?) != 0 OR ({pfx}e2 & ?) != 0)")
            params.extend([any_m.e1, any_m.e2])
        elif any_m.e1 != 0:
            clauses.append(f"({pfx}e1 & ?) != 0")
            params.append(any_m.e1)
        elif any_m.e2 != 0:
            clauses.append(f"({pfx}e2 & ?) != 0")
            params.append(any_m.e2)

    if elem_filter.none_of:
        forb = elements_to_masks(elem_filter.none_of)
        if forb.e1 != 0:
            clauses.append(f"({pfx}e1 & ?) = 0")
            params.append(forb.e1)
        if forb.e2 != 0:
            clauses.append(f"({pfx}e2 & ?) = 0")
            params.append(forb.e2)

    if elem_filter.strict:
        allowed_set = getattr(elem_filter, "allowed", None) or (
            elem_filter.all_of | elem_filter.any_of
        )
        if allowed_set:
            allow = elements_to_masks(allowed_set)
            clauses.append(f"(({pfx}e1 | ?) = ? AND ({pfx}e2 | ?) = ?)")
            params.extend([allow.e1, allow.e1, allow.e2, allow.e2])
        else:
            clauses.append(f"({pfx}e1 = 0 AND {pfx}e2 = 0)")

    return SqlWhereClause(clauses=clauses, params=params)


def build_crystal_sql_clauses(
    crystal_filter: CrystallographicFilter,
    gi_alias: str = "gi",
    gs_alias: str = "gs",
) -> SqlWhereClause:
    """Translate CrystallographicFilter into SQLite WHERE clauses on general_indexed/stored."""
    gi_pfx = f"{gi_alias}." if gi_alias else ""
    gs_pfx = f"{gs_alias}." if gs_alias else ""
    clauses: list[str] = []
    params: list[Any] = []

    field_col_map = {
        "a": f"{gi_pfx}A",
        "b": f"{gi_pfx}B",
        "c": f"{gi_pfx}C",
        "alpha": f"{gi_pfx}ALPHA",
        "beta": f"{gi_pfx}BETA",
        "gamma": f"{gi_pfx}GAMMA",
        "volume": f"{gi_pfx}XTLVOL",
        "z": f"{gi_pfx}Z",
        "dx": f"{gi_pfx}DX",
        "dm": f"{gi_pfx}DM",
        "iic": f"{gi_pfx}IIC",
    }

    for axis, bounds in crystal_filter.cell_ranges.items():
        col = field_col_map.get(axis.lower())
        if not col:
            continue
        lo = bounds[0]
        hi = bounds[1]
        if lo is not None:
            clauses.append(f"{col} >= ?")
            params.append(float(lo))
        if hi is not None:
            clauses.append(f"{col} <= ?")
            params.append(float(hi))

    if crystal_filter.space_groups:
        sgs = list(crystal_filter.space_groups)
        placeholders = ", ".join(["?"] * len(sgs))
        clauses.append(f"({gs_pfx}SPGR IN ({placeholders}) OR {gs_pfx}XTLSG IN ({placeholders}))")
        params.extend(sgs)
        params.extend(sgs)

    if crystal_filter.crystal_systems:
        sys_codes: list[str] = []
        for cs in crystal_filter.crystal_systems:
            mapped = CRYSTAL_SYSTEM_TO_HSRDB.get(cs.strip().lower(), (cs.strip().upper(),))
            for code in mapped:
                if code not in sys_codes:
                    sys_codes.append(code)
        if sys_codes:
            placeholders = ", ".join(["?"] * len(sys_codes))
            clauses.append(f"{gi_pfx}XTSLSYS IN ({placeholders})")
            params.extend(sys_codes)

    if crystal_filter.subfiles:
        sf_ids: list[int] = []
        for sf in crystal_filter.subfiles:
            sf_id = SUBFILE_MAP.get(sf.strip().lower())
            if sf_id is not None and sf_id not in sf_ids:
                sf_ids.append(sf_id)
        if sf_ids:
            placeholders = ", ".join(["?"] * len(sf_ids))
            clauses.append(
                f"{gi_pfx}id IN (SELECT pid FROM SubFileRef WHERE SubFileID IN ({placeholders}))"
            )
            params.extend(sf_ids)

    return SqlWhereClause(clauses=clauses, params=params)


@dataclass(frozen=True)
class PeakFilter:
    """Filter candidate structures based on observed XRD diffraction peak windows."""

    peaks: list[float]
    tolerance: float = 0.20
    is_two_theta: bool = True
    wavelength: float = 1.540596
    min_intensity: float = 100.0
    min_hits: int = 1

    def get_d_windows(self) -> list[tuple[float, float]]:
        """Calculate (d_min, d_max) windows in Angstroms for each target peak."""
        windows: list[tuple[float, float]] = []
        for p in self.peaks:
            if self.is_two_theta:
                th_min = math.radians(max(0.05, (p - self.tolerance) / 2.0))
                th_max = math.radians(min(89.9, (p + self.tolerance) / 2.0))
                d_high = self.wavelength / (2.0 * math.sin(th_min))
                d_low = self.wavelength / (2.0 * math.sin(th_max))
                windows.append((min(d_low, d_high), max(d_low, d_high)))
            else:
                windows.append((p - self.tolerance, p + self.tolerance))
        return windows

    def matches_blob(self, lines_blob: bytes) -> bool:
        """Check if a Lines blob (<fH records) has at least min_hits matching windows."""
        windows = self.get_d_windows()
        if not windows:
            return True
        if not lines_blob or len(lines_blob) < 6:
            return False

        required = max(1, min(self.min_hits, len(windows)))
        hits = 0
        for d_min, d_max in windows:
            matched_win = False
            for pair in struct.iter_unpack("<fH", lines_blob):
                d_val = float(pair[0])
                i_val = float(pair[1])
                if i_val >= self.min_intensity and d_min <= d_val <= d_max:
                    matched_win = True
                    break
            if matched_win:
                hits += 1
                if hits >= required:
                    return True
        return False


@dataclass(frozen=True)
class AtomSite:
    """Decoded atomic position, occupancy, and isotropic displacement factor."""

    label: str
    element: str
    wyckoff: str = ""
    multiplicity: int = 1
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    sof: float = 1.0
    b_iso: float = 1.0


@dataclass(frozen=True)
class PhaseData:
    """Decoded Phases.Phase binary container."""

    hdr_ver: int
    atoms: dict[str, AtomSite]
    records: dict[str, float]


@dataclass(frozen=True)
class CodStructureRecord:
    """Typed metadata record of a structure in COD24_HS4x.hsrdb."""

    pid: int
    cod_id: str
    product_id: str
    formula: str
    name: str
    mineral_name: str
    spacegroup: str
    ita_number: int | None
    crystal_system: str
    a: float | None
    b: float | None
    c: float | None
    alpha: float | None
    beta: float | None
    gamma: float | None
    volume: float | None
    z: int | None
    density_calc: float | None
    density_meas: float | None
    rir: float | None
    temperature_k: float | None = None
    pressure_kpa: float | None = None
    has_phase: bool = True
    num_reflections: int = 0


class HsRdbDatabase:
    """Read-only interface to the local COD24_HS4x.hsrdb SQLite database."""

    def __init__(self) -> None:
        self.path = resolve_hsrdb_path()
        uri = f"file:{self.path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def __enter__(self) -> HsRdbDatabase:  # noqa: PYI034
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def count_total(self) -> int:
        """Return total number of patterns in general_indexed."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM general_indexed")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def pid_for_cod_id(self, cod_id_or_ref: str | int) -> int:
        """Resolve a COD ID ('1537844', 'cod-1537844') or ProductID ('961537845') to internal pid."""
        raw = str(cod_id_or_ref).strip().lower()
        for pfx in ("cod-", "cod_", "pdf:"):
            raw = raw.removeprefix(pfx)
        clean = raw.replace("-", "").strip()
        if not clean.isdigit():
            raise KeyError(f"Invalid COD identifier: {cod_id_or_ref!r}")

        cur = self._conn.cursor()
        candidates: list[str] = []
        if clean.startswith("96") and len(clean) == 9:
            candidates.append(clean)
        else:
            num = int(clean)
            candidates.append(f"96{num + 1:07d}")
            candidates.append(f"96{num:07d}")

        for cand_prod in candidates:
            cur.execute(
                """SELECT gi.id, gi.ProductID, gs.Comment
                   FROM general_indexed gi
                   JOIN general_stored gs ON gi.id = gs.pid
                   WHERE gi.ProductID = ?""",
                (cand_prod,),
            )
            row = cur.fetchone()
            if row is not None:
                actual_cod = extract_cod_id(str(row["ProductID"]), str(row["Comment"] or ""))
                if clean.startswith("96") or actual_cod == str(int(clean)):
                    return int(row["id"])

        # Fallback exact search in Comment
        cur.execute(
            "SELECT pid FROM general_stored WHERE Comment LIKE ? LIMIT 1",
            (f"%COD database code: {int(clean)}%",),
        )
        row_fb = cur.fetchone()
        if row_fb is not None:
            return int(row_fb["pid"])
        raise KeyError(f"COD entry {cod_id_or_ref!r} not found in {self.path.name}")

    def search(
        self,
        element_filter: ElementFilter | None = None,
        crystal_filter: CrystallographicFilter | None = None,
        peak_filter: PeakFilter | None = None,
        limit: int | None = None,
    ) -> list[CodStructureRecord]:
        """Execute fast indexed SQL search across 511,583 COD structures."""
        clauses: list[str] = []
        params: list[Any] = []

        if element_filter is not None:
            e_sql = build_element_sql_clauses(element_filter, table_alias="pe")
            clauses.extend(e_sql.clauses)
            params.extend(e_sql.params)

        if crystal_filter is not None:
            c_sql = build_crystal_sql_clauses(crystal_filter, gi_alias="gi", gs_alias="gs")
            clauses.extend(c_sql.clauses)
            params.extend(c_sql.params)

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        lines_col = ", gs.Lines AS full_lines" if peak_filter is not None else ""
        query = f"""
            SELECT
                gi.id, gi.ProductID, gi.XTSLSYS, gi.A, gi.B, gi.C,
                gi.ALPHA, gi.BETA, gi.GAMMA, gi.XTLVOL, gi.Z, gi.DX, gi.DM, gi.IIC,
                gs.SPGR, gs.Comment, LENGTH(gs.Lines) AS lines_len,
                gt.chemicalformula, gt.compoundname, gt.mineralname
                {lines_col}
            FROM general_indexed gi
            JOIN pattern_elements pe ON gi.id = pe.pid
            JOIN general_stored gs ON gi.id = gs.pid
            JOIN general_text gt ON gi.id = gt.docid
            {where_sql}
        """

        cur = self._conn.cursor()
        cur.execute(query, params)

        records: list[CodStructureRecord] = []
        for row in cur:
            if peak_filter is not None:
                blob = row["full_lines"] or b""
                if not peak_filter.matches_blob(blob):
                    continue

            pid_val = int(row["id"])
            prod_id = str(row["ProductID"] or "")
            comment_str = str(row["Comment"] or "")
            cod_id = extract_cod_id(prod_id, comment_str)
            spgr_str = str(row["SPGR"] or "").strip()
            sg_obj = gemmi.find_spacegroup_by_name(spgr_str) if spgr_str else None
            ita_num = int(sg_obj.number) if sg_obj is not None else None

            temp_k: float | None = None
            m_t = _TEMP_COMMENT_RE.search(comment_str)
            if m_t:
                temp_k = float(m_t.group(1))

            pres_kpa: float | None = None
            m_p = _PRES_COMMENT_RE.search(comment_str)
            if m_p:
                p_val = float(m_p.group(1))
                unit = m_p.group(2).lower()
                if unit == "mpa":
                    p_val *= 1000.0
                elif unit == "gpa":
                    p_val *= 1_000_000.0
                pres_kpa = p_val

            lines_len = int(row["lines_len"] or 0)
            records.append(
                CodStructureRecord(
                    pid=pid_val,
                    cod_id=cod_id,
                    product_id=prod_id,
                    formula=str(row["chemicalformula"] or "").strip(),
                    name=str(row["compoundname"] or "").strip(),
                    mineral_name=str(row["mineralname"] or "").strip(),
                    spacegroup=spgr_str,
                    ita_number=ita_num,
                    crystal_system=str(row["XTSLSYS"] or ""),
                    a=float(row["A"]) if row["A"] is not None else None,
                    b=float(row["B"]) if row["B"] is not None else None,
                    c=float(row["C"]) if row["C"] is not None else None,
                    alpha=float(row["ALPHA"]) if row["ALPHA"] is not None else None,
                    beta=float(row["BETA"]) if row["BETA"] is not None else None,
                    gamma=float(row["GAMMA"]) if row["GAMMA"] is not None else None,
                    volume=float(row["XTLVOL"]) if row["XTLVOL"] is not None else None,
                    z=int(row["Z"]) if row["Z"] is not None else None,
                    density_calc=float(row["DX"]) if row["DX"] is not None else None,
                    density_meas=float(row["DM"]) if row["DM"] is not None else None,
                    rir=float(row["IIC"]) if row["IIC"] is not None else None,
                    temperature_k=temp_k,
                    pressure_kpa=pres_kpa,
                    has_phase=True,
                    num_reflections=lines_len // 6,
                )
            )
            if limit is not None and limit > 0 and len(records) >= limit:
                break

        return records

    def survivors_by_strongest_lines(
        self,
        element_filter: ElementFilter,
        query_two_theta: NDArray[Any],
        to_two_theta: Callable[[NDArray[Any]], NDArray[Any]],
        tolerance_deg: float = 0.20,
        min_hits: int = 2,
        max_candidates: int | None = 500,
        crystal_filter: CrystallographicFilter | None = None,
    ) -> list[int]:
        """Fast first-pass filter using 60-byte pattern_strongestlines (top 3 Hanawalt lines)."""
        query = np.sort(np.asarray(query_two_theta, dtype=float))
        e_sql = build_element_sql_clauses(element_filter, table_alias="pe")
        clauses = list(e_sql.clauses)
        params = list(e_sql.params)

        join_gi = ""
        if crystal_filter is not None:
            c_sql = build_crystal_sql_clauses(crystal_filter, gi_alias="gi", gs_alias="gs")
            if c_sql.clauses:
                join_gi = (
                    "JOIN general_indexed gi ON pe.pid = gi.id "
                    "JOIN general_stored gs ON pe.pid = gs.pid"
                )
                clauses.extend(c_sql.clauses)
                params.extend(c_sql.params)

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT pe.pid, ps.Lines
            FROM pattern_elements pe
            JOIN pattern_strongestlines ps ON pe.pid = ps.pid
            {join_gi}
            {where_sql}
        """
        cur = self._conn.cursor()
        cur.execute(sql, params)

        if query.size == 0:
            pids = [int(r["pid"]) for r in cur]
            return pids[:max_candidates] if max_candidates else pids

        scored_hits: list[tuple[int, int]] = []
        for row in cur:
            pid_val = int(row["pid"])
            blob = row["Lines"] or b""
            n_top = min(3, len(blob) // 6)
            if n_top == 0:
                continue
            d_top = np.empty(n_top, dtype=float)
            for idx in range(n_top):
                pair = struct.unpack("<fH", blob[idx * 6 : (idx + 1) * 6])
                d_top[idx] = float(pair[0])
            angles = np.asarray(to_two_theta(d_top), dtype=float)
            hits = 0
            for ang in angles:
                if not np.isfinite(ang):
                    continue
                if float(np.min(np.abs(query - ang))) <= tolerance_deg:
                    hits += 1
            if hits >= min_hits:
                scored_hits.append((-hits, pid_val))

        scored_hits.sort()
        if max_candidates is not None and len(scored_hits) > max_candidates:
            logger.info(
                "HSRDB first pass left {} structures; keeping top {}",
                len(scored_hits),
                max_candidates,
            )
            scored_hits = scored_hits[:max_candidates]
        return [pid_val for _neg_h, pid_val in scored_hits]

    def get_literature(self, pid: int) -> dict[str, Any]:
        """Fetch bibliographic metadata for a pattern pid."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT lt.year, lt.authors, lt.journal, lt.volume, lt.pages
               FROM literature_map lm
               JOIN literature_text lt ON lm.doc = lt.docid
               WHERE lm.pid = ? LIMIT 1""",
            (pid,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        yr = str(row["year"] or "")
        return {
            "year": int(yr) if yr.isdigit() else None,
            "authors": str(row["authors"] or ""),
            "journal": str(row["journal"] or ""),
            "volume": str(row["volume"] or ""),
            "pages": str(row["pages"] or ""),
        }

    def get_candidate_lines(self, pid: int) -> Any:
        """Unpack Lines, LinesI, and HKL blobs into a search_match.CandidateLines object."""
        from search_match import CandidateLines

        cur = self._conn.cursor()
        cur.execute(
            """SELECT gi.ProductID, gi.XTLVOL, gi.A, gi.B, gi.C, gi.ALPHA, gi.BETA, gi.GAMMA,
                      gs.Comment, gs.Lines, gs.LinesI, gs.HKL, gt.chemicalformula
               FROM general_indexed gi
               JOIN general_stored gs ON gi.id = gs.pid
               JOIN general_text gt ON gi.id = gt.docid
               WHERE gi.id = ?""",
            (pid,),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError(f"Pattern pid {pid} not found")

        cod_id = extract_cod_id(str(row["ProductID"] or ""), str(row["Comment"] or ""))
        label = f"cod-{cod_id}"
        formula = str(row["chemicalformula"] or "")
        vol = float(row["XTLVOL"] or 0.0)
        cell = (
            float(row["A"]) if row["A"] is not None else None,
            float(row["B"]) if row["B"] is not None else None,
            float(row["C"]) if row["C"] is not None else None,
            float(row["ALPHA"]) if row["ALPHA"] is not None else None,
            float(row["BETA"]) if row["BETA"] is not None else None,
            float(row["GAMMA"]) if row["GAMMA"] is not None else None,
        )

        lines_blob = row["Lines"] or b""
        linesi_blob = row["LinesI"] or b""
        hkl_blob = row["HKL"] or b""
        n_peaks = len(lines_blob) // 6
        if n_peaks == 0:
            return CandidateLines(
                label=label,
                formula=formula,
                volume=vol,
                d_spacing=np.array([], dtype=float),
                intensity=np.array([], dtype=float),
                card_cell=cell,
                hkl=(),
            )

        d_arr = np.empty(n_peaks, dtype=float)
        int_arr = np.empty(n_peaks, dtype=float)
        if len(linesi_blob) == n_peaks * 4:
            int_arr[:] = struct.unpack(f"<{n_peaks}f", linesi_blob)
            for idx, pair in enumerate(struct.iter_unpack("<fH", lines_blob)):
                d_arr[idx] = float(pair[0])
        else:
            for idx, pair in enumerate(struct.iter_unpack("<fH", lines_blob)):
                d_arr[idx] = float(pair[0])
                int_arr[idx] = float(pair[1])

        hkl_list: list[tuple[int, ...]] = []
        if len(hkl_blob) == n_peaks * 6:
            for triple in struct.iter_unpack("<3h", hkl_blob):
                hkl_list.append((int(triple[0]), int(triple[1]), int(triple[2])))

        return CandidateLines(
            label=label,
            formula=formula,
            volume=vol,
            d_spacing=d_arr,
            intensity=int_arr,
            card_cell=cell,
            hkl=tuple(hkl_list),
        )

    def parse_phase(self, pid: int) -> PhaseData | None:
        """Deterministically decode Phases.Phase BLOB into atomic sites and parameters."""
        cur = self._conn.cursor()
        cur.execute("SELECT Phase FROM Phases WHERE ID = ?", (pid,))
        row = cur.fetchone()
        if not row or not row["Phase"]:
            return None

        phase_blob = bytes(row["Phase"])
        if len(phase_blob) < 6:
            return None

        hdr_ver = int(struct.unpack("<H", phase_blob[:2])[0])
        expected_crc = int(struct.unpack("<I", phase_blob[2:6])[0])
        try:
            raw = zlib.decompress(phase_blob[6:], -15)
        except zlib.error:
            return None

        if (zlib.crc32(raw) & 0xFFFFFFFF) != expected_crc:
            raise ValueError(f"Phase BLOB CRC32 mismatch for pid {pid}")

        pos = 0
        records: dict[str, float] = {}
        wyckoff_map: dict[str, str] = {}
        last_atom: str | None = None
        atom_params = {"X", "Y", "Z", "sof", "Biso", "Uiso", "B11", "B22", "B33"}

        while pos < len(raw) - 13:
            str_len = int(struct.unpack("<I", raw[pos : pos + 4])[0])
            if 1 <= str_len <= 150 and pos + 4 + str_len * 2 + 4 <= len(raw):
                str_bytes = raw[pos + 4 : pos + 4 + str_len * 2]
                try:
                    s = str_bytes.decode("utf-16le")
                    cur_idx = pos + 4 + str_len * 2
                    hdr = raw[cur_idx : cur_idx + 5]
                    if hdr == b"\x02\x00\x00\x00\x01" and cur_idx + 13 <= len(raw):
                        val = float(struct.unpack("<d", raw[cur_idx + 5 : cur_idx + 13])[0])
                        records[s] = val
                        parts = s.split()
                        if len(parts) >= 3 and parts[-1] in atom_params:
                            last_atom = parts[-2]
                        elif _WYCKOFF_RE.match(s) and last_atom is not None:
                            wyckoff_map[last_atom] = s
                        pos = cur_idx + 13 + 40
                        continue
                    if (
                        hdr[:4] == b"\x03\x00\x00\x00"
                        and cur_idx + 17 <= len(raw)
                        and raw[cur_idx + 4 : cur_idx + 9] == b"\x02\x00\x00\x00\x01"
                    ):
                        val = float(struct.unpack("<d", raw[cur_idx + 9 : cur_idx + 17])[0])
                        records[s] = val
                        parts = s.split()
                        if len(parts) >= 3 and parts[-1] in atom_params:
                            last_atom = parts[-2]
                        elif _WYCKOFF_RE.match(s) and last_atom is not None:
                            wyckoff_map[last_atom] = s
                        pos = cur_idx + 17 + 40
                        continue
                    if _WYCKOFF_RE.match(s) and last_atom is not None:
                        wyckoff_map[last_atom] = s
                except UnicodeDecodeError:
                    pass
            pos += 1

        atoms_dict: dict[str, dict[str, float]] = {}
        for name, val in records.items():
            parts = name.split()
            if len(parts) >= 3 and parts[-1] in atom_params:
                at_label = parts[-2]
                atoms_dict.setdefault(at_label, {})[parts[-1]] = val

        sites: dict[str, AtomSite] = {}
        for at_label, p in atoms_dict.items():
            elem_match = re.match(r"[A-Z][a-z]?", at_label)
            elem = elem_match.group(0) if elem_match else "X"
            wyck = wyckoff_map.get(at_label, "")
            m_mult = _WYCKOFF_RE.match(wyck)
            mult = int(m_mult.group(1)) if m_mult else 1
            sites[at_label] = AtomSite(
                label=at_label,
                element=elem,
                wyckoff=wyck,
                multiplicity=mult,
                x=float(p.get("X", 0.0)),
                y=float(p.get("Y", 0.0)),
                z=float(p.get("Z", 0.0)),
                sof=float(p.get("sof", 1.0)),
                b_iso=float(p.get("Biso", 1.0)),
            )

        return PhaseData(hdr_ver=hdr_ver, atoms=sites, records=records)

    def export_cif(
        self,
        pid_or_cod_id: int | str,
        out_path: Path | None = None,
        *,
        by_cod_id: bool = False,
    ) -> Path:
        """Build and write a standards-compliant CIF file from COD24_HS4x.hsrdb via gemmi."""
        pid = (
            self.pid_for_cod_id(pid_or_cod_id)
            if (by_cod_id or isinstance(pid_or_cod_id, str))
            else int(pid_or_cod_id)
        )
        cur = self._conn.cursor()
        cur.execute(
            """SELECT gi.id, gi.ProductID, gi.A, gi.B, gi.C, gi.ALPHA, gi.BETA, gi.GAMMA,
                      gi.XTLVOL, gs.SPGR, gs.Comment, gt.chemicalformula, gt.compoundname
               FROM general_indexed gi
               JOIN general_stored gs ON gi.id = gs.pid
               JOIN general_text gt ON gi.id = gt.docid
               WHERE gi.id = ?""",
            (pid,),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError(f"Pattern pid {pid} not found")

        cod_id = extract_cod_id(str(row["ProductID"] or ""), str(row["Comment"] or ""))
        spgr = str(row["SPGR"] or "P 1").strip()
        a = float(row["A"] or 5.0)
        b = float(row["B"] or 5.0)
        c = float(row["C"] or 5.0)
        alpha = float(row["ALPHA"] or 90.0)
        beta = float(row["BETA"] or 90.0)
        gamma = float(row["GAMMA"] or 90.0)
        vol = float(row["XTLVOL"] or gemmi.UnitCell(a, b, c, alpha, beta, gamma).volume)

        doc = gemmi.cif.Document()
        block = doc.add_new_block(f"cod_{cod_id}")
        formula_str = str(row["chemicalformula"] or "").strip()
        if formula_str:
            block.set_pair("_chemical_formula_sum", f"'{formula_str}'")
        block.set_pair("_cell_length_a", f"{a:.6f}")
        block.set_pair("_cell_length_b", f"{b:.6f}")
        block.set_pair("_cell_length_c", f"{c:.6f}")
        block.set_pair("_cell_angle_alpha", f"{alpha:.4f}")
        block.set_pair("_cell_angle_beta", f"{beta:.4f}")
        block.set_pair("_cell_angle_gamma", f"{gamma:.4f}")
        block.set_pair("_cell_volume", f"{vol:.4f}")
        block.set_pair("_symmetry_space_group_name_H-M", f"'{spgr}'")
        block.set_pair("_space_group_name_H-M_alt", f"'{spgr}'")

        sg_gemmi = gemmi.find_spacegroup_by_name(spgr)
        if sg_gemmi is not None:
            block.set_pair("_symmetry_Int_Tables_number", str(int(sg_gemmi.number)))
            block.set_pair("_space_group_IT_number", str(int(sg_gemmi.number)))

        phase_data = self.parse_phase(pid)
        if phase_data and phase_data.atoms:
            loop = block.init_loop(
                "_atom_site_",
                [
                    "label",
                    "type_symbol",
                    "fract_x",
                    "fract_y",
                    "fract_z",
                    "occupancy",
                    "B_iso_or_equiv",
                ],
            )
            for site in phase_data.atoms.values():
                loop.add_row(
                    [
                        site.label,
                        site.element,
                        f"{site.x:.6f}",
                        f"{site.y:.6f}",
                        f"{site.z:.6f}",
                        f"{site.sof:.4f}",
                        f"{site.b_iso:.4f}",
                    ]
                )

        target = out_path or (Path.home() / ".cache" / "cod_cifs" / f"{cod_id}.cif")
        target.parent.mkdir(parents=True, exist_ok=True)
        doc.write_file(str(target))
        return target
