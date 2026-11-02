"""Declarative Refinement-as-Code parameter engine (rietveld_params.py).

Provides bi-directional translation between GSAS-II projects (.gpx) and
human/agent-inspectable JSON parameter maps (params.json) for structured
crystallographic refinement control.

Enables precise, granular control over individual refinement flags (Cell, Scale,
Size, Mustrain, Zero, Background, Atoms), strictly preventing metric tensor
divergence on trace microphases while eliminating shell quoting issues.
"""

from __future__ import annotations

import shutil
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
from loguru import logger

# Sibling imports from skill scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from constants import (
    BOUND_DISPLACEX_MAX,
    BOUND_DISPLACEX_MIN,
    BOUND_MD_MAX,
    BOUND_MD_MIN,
    BOUND_MUSTRAIN_MAX,
    BOUND_MUSTRAIN_MIN,
    BOUND_SHL_MAX,
    BOUND_SHL_MIN,
    BOUND_SIZE_MAX,
    BOUND_SIZE_MIN,
    BOUND_ZERO_MAX,
    BOUND_ZERO_MIN,
    PHASE_MICRO_CELL_LOCK_PERCENT,
)
from gsas2_adapter import (
    calculate_profile_metrics,
    extract_cell_parameters,
    extract_phase_scales,
    extract_profile_curves,
    g2_safe_errors,
    get_project_qpa,
    sync_pawley_reflections_to_hist,
)
from stage_gate_rietveld import (
    G2AtomRecord,
    StageId,
    apply_box_constraints,
    instrumental_lower_bound_report,
    normalize_stage_name,
    set_param_limit,
)

# -----------------------------------------------------------------------------
# GSAS-II instrument parameter record layout
# -----------------------------------------------------------------------------
# GSAS-II stores entries of hist.data["Instrument Parameters"][0] as 3-field
# records [init, value, refine]. The effective value used by the profile
# calculation (and printed in the .lst) is field 1; the refinement flag is
# field 2. Field 0 is the stale initial value and must be neither read nor
# written -- doing so silently returns the init value and overwrites the
# effective value with a bool.

INST_PARAM_FIELDS = 3


#: Caglioti coefficients of the instrumental profile, in the order GSAS-II stores them.
#: The first three are the Gaussian terms, the last three the Lorentzian ones.
CAGLIOTI_COEFFICIENTS = ("U", "V", "W", "X", "Y", "Z")


def read_inst_param(inst_raw: dict[str, Any], name: str, default: float) -> tuple[float, bool]:
    """Return (value, refine) of an instrument parameter, validating the record layout."""
    rec = inst_raw.get(name)
    if rec is None:
        return float(default), False
    if len(rec) < INST_PARAM_FIELDS:
        raise ValueError(
            f"Instrument parameter {name!r} has {len(rec)} fields; GSAS-II expects "
            f"{INST_PARAM_FIELDS} as [init, value, refine]"
        )
    return float(rec[1]), bool(rec[2])


def write_inst_param(
    inst_raw: dict[str, Any],
    name: str,
    value: float | None = None,
    refine: bool | None = None,
) -> None:
    """Set value (field 1) and/or refinement flag (field 2) of an instrument parameter."""
    rec = inst_raw.get(name)
    if rec is None:
        raise KeyError(f"Instrument parameter {name!r} is absent from the histogram")
    if len(rec) < INST_PARAM_FIELDS:
        raise ValueError(
            f"Instrument parameter {name!r} has {len(rec)} fields; GSAS-II expects "
            f"{INST_PARAM_FIELDS} as [init, value, refine]"
        )
    if value is not None:
        rec[1] = float(value)
    if refine is not None:
        rec[2] = bool(refine)


def apply_declared_limits(gpx: Any, entry: dict[str, Any], mask: str) -> None:
    """Apply ``min``/``max`` from a params.json entry as GSAS-II box constraints.

    The declared bounds were previously carried through the document and never
    applied, so a refinement could run far outside the schema (size below its
    declared minimum, microstrain negative or unphysically large). ``mask`` must
    be the exact GSAS-II parameter name for the entry being bounded.
    """
    if "min" not in entry and "max" not in entry:
        return
    if "min" in entry:
        set_param_limit(gpx, "parmMin", mask, float(entry["min"]))
    if "max" in entry:
        set_param_limit(gpx, "parmMax", mask, float(entry["max"]))


def registered_limit(gpx: Any, control: str, mask: str, fallback: float) -> float:
    """Return a box limit registered in the project, or the fallback.

    ``params refine`` rewrites the document from the project, and a fresh dump used
    to restate the schema defaults for ``min``/``max`` whatever the caller had
    declared -- so a declared interval silently disappeared on the next round trip,
    and a violation check against the re-dumped document saw the schema instead of
    the declaration. Reading the registered value back keeps the document an
    honest mirror of the project.
    """
    if not hasattr(gpx, "get_Controls"):
        return float(fallback)
    with suppress(*g2_safe_errors()):
        value = gpx.get_Controls(control, variable=mask)
        if value is not None:
            return float(value)
    return float(fallback)


def _bounded_nodes(node: Any, path: tuple[str, ...] = ()):
    """Yield (path, entry) for every node carrying both a value and a bound.

    The bounds that matter are the ones the caller declared in the input
    document, not the schema defaults a fresh dump carries: a round trip through
    ``params refine`` rewrites the file from the project and would otherwise
    replace the caller's interval with the schema's.
    """
    if isinstance(node, dict):
        if "value" in node and ("min" in node or "max" in node):
            yield path, node
            return
        for key, child in node.items():
            yield from _bounded_nodes(child, path + (str(key),))


def _node_at(document: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the node at ``path`` inside ``document``, or None when absent."""
    node: Any = document
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, dict) else None


def enforce_declared_limits(updated: dict[str, Any], declared: dict[str, Any]) -> list[str]:
    """Report the declared limits that the refined model violates.

    Registering a box constraint and the optimizer respecting it are different
    things: GSAS-II bounds a parameter through Levenberg damping, and a step can
    still land outside the declared interval (a specimen displacement declared as
    +-500 um refined to -676 um). The comparison is made between the document
    dumped after the refinement and the document the caller declared, because a
    fresh dump restates the schema defaults rather than the declared limits.
    """
    violations: list[str] = []
    for path, declared_entry in _bounded_nodes(declared):
        target = _node_at(updated, path)
        if target is None or target.get("value") is None:
            continue
        label = ".".join(path)
        value = float(target["value"])
        lo = declared_entry.get("min")
        hi = declared_entry.get("max")
        if lo is not None and value < float(lo) - 1e-12:
            violations.append(f"{label} = {value} is below the declared min {lo}")
        elif hi is not None and value > float(hi) + 1e-12:
            violations.append(f"{label} = {value} is above the declared max {hi}")
    return violations


def clamp_to_declared_limits(updated: dict[str, Any], declared: dict[str, Any]) -> list[str]:
    """Clamp every value in ``updated`` to the interval declared in ``declared``."""
    clamped: list[str] = []
    for path, declared_entry in _bounded_nodes(declared):
        target = _node_at(updated, path)
        if target is None or target.get("value") is None:
            continue
        label = ".".join(path)
        value = float(target["value"])
        lo = declared_entry.get("min")
        hi = declared_entry.get("max")
        if lo is not None and value < float(lo):
            target["value"] = float(lo)
            clamped.append(label)
        elif hi is not None and value > float(hi):
            target["value"] = float(hi)
            clamped.append(label)
    return clamped


# -----------------------------------------------------------------------------
# Parameter Extraction (Dump)
# -----------------------------------------------------------------------------
def dump_project_params(
    gpx: Any,
    hist_idx: int = 0,
    preset: int | str | None = None,
) -> dict[str, Any]:
    """Serialize the complete refinement state of a GSAS-II project into a declarative dict."""
    histograms = gpx.histograms()
    phases = gpx.phases()

    if not histograms:
        raise ValueError("Project contains no powder histograms.")

    hist = histograms[hist_idx]
    hist_name: str = hist.name

    # 1. QPA mass fractions
    mass_fracs_dict: dict[str, tuple[float, float]] = get_project_qpa(gpx, hist)
    cells_dict = extract_cell_parameters(gpx)
    scales_dict = extract_phase_scales(gpx)

    # 2. Last fit e.s.d. values (shifts are not carried in the declarative document)
    last_esds: dict[str, float] = {}
    with suppress(*g2_safe_errors()):
        res = gpx.get_LastFitResults()
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            last_esds = res[1] if isinstance(res[1], dict) else {}

    # 3. Histogram parameters (Background, Instrument, Sample)
    # Background
    bkg_data = hist.data.get("Background", [[], {}])
    bkg_raw = bkg_data[0] if isinstance(bkg_data, list) and len(bkg_data) > 0 else []
    bkg_type = bkg_raw[0] if len(bkg_raw) > 0 else "chebyschev-1"
    bkg_refine = bool(bkg_raw[1]) if len(bkg_raw) > 1 else False
    bkg_n_terms = int(bkg_raw[2]) if len(bkg_raw) > 2 else 6
    bkg_coeffs = [float(x) for x in bkg_raw[3 : 3 + bkg_n_terms]] if len(bkg_raw) > 3 else []
    bkg_esds = [float(last_esds.get(f":{hist_idx}:Back;{i}", 0.0)) for i in range(len(bkg_coeffs))]

    # Instrument parameters (3-field records: value at [1], flag at [2])
    inst_raw = hist.data.get("Instrument Parameters", [{}, {}])[0]
    zero_val, zero_refine = read_inst_param(inst_raw, "Zero", 0.0)  # noqa: PU001
    zero_esd = float(last_esds.get(f":{hist_idx}:Zero", 0.0))

    shl_val, shl_refine = read_inst_param(inst_raw, "SH/L", 0.002)  # noqa: PU001
    # The Caglioti coefficients: U, V, W carry the Gaussian part of the instrumental
    # profile and X, Y, Z the Lorentzian one. GSAS-II's PXC instrument defines all six, but
    # only U/V/W were reachable through this document, so the Lorentzian half of the shape
    # could not be refined at all -- while a residual with W < 1 and serial correlation is
    # precisely a complaint about the tails. All six are carried now.
    caglioti_val: dict[str, float] = {}
    caglioti_refine: dict[str, bool] = {}
    for coefficient in CAGLIOTI_COEFFICIENTS:
        coefficient_value, coefficient_refine = read_inst_param(  # noqa: PU001
            inst_raw, coefficient, 0.0
        )
        caglioti_val[coefficient] = coefficient_value
        caglioti_refine[coefficient] = coefficient_refine
    # K-alpha2/K-alpha1 intensity ratio. A data set may carry the companion line
    # already removed; the ratio is then 0.0 and the modelled profile must not
    # carry a doublet.
    ka2_val, ka2_refine = read_inst_param(  # noqa: PU001
        inst_raw, "I(L2)/I(L1)", 0.5
    )

    # Sample parameters
    samp_raw = hist.data.get("Sample Parameters", {})
    dispx_val = float(samp_raw.get("DisplaceX", [0.0, False])[0])
    dispx_refine = bool(samp_raw.get("DisplaceX", [0.0, False])[1])
    dispx_esd = float(last_esds.get(f":{hist_idx}:DisplaceX", 0.0))

    # 4. Phase-level parameters
    phases_map: dict[str, Any] = {}
    for p_idx, p in enumerate(phases):
        p_name = p.name
        gen = p.data.get("General", {})
        sg = gen.get("SGData", {}).get("SpGrp", "Unknown")
        qpa_tuple = mass_fracs_dict.get(p_name, (0.0, None))
        wt_pct = float(qpa_tuple[0])
        # esd может отсутствовать: ноль означал бы «измерено точно», поэтому в
        # декларативном документе это null, а не 0.0
        wt_esd = float(qpa_tuple[1]) if qpa_tuple[1] is not None else None

        # Cell
        c_info = gen.get("Cell", [False, 0, 0, 0, 0, 0, 0, 0])
        cell_refine = bool(c_info[0]) if len(c_info) > 0 else False
        c_dict = cells_dict.get(p_name, {})
        a_val = float(c_dict.get("a", c_info[1] if len(c_info) > 1 else 0.0))
        b_val = float(c_dict.get("b", c_info[2] if len(c_info) > 2 else 0.0))
        c_val = float(c_dict.get("c", c_info[3] if len(c_info) > 3 else 0.0))
        alpha_val = float(c_dict.get("alpha", c_info[4] if len(c_info) > 4 else 90.0))
        beta_val = float(c_dict.get("beta", c_info[5] if len(c_info) > 5 else 90.0))
        gamma_val = float(c_dict.get("gamma", c_info[6] if len(c_info) > 6 else 90.0))
        vol_val = float(c_dict.get("volume", c_info[7] if len(c_info) > 7 else 0.0))

        cell_esds: dict[str, float] = {}
        for ax in ["a", "b", "c", "alpha", "beta", "gamma", "volume"]:
            if f"{ax}_esd" in c_dict and c_dict[f"{ax}_esd"] is not None:
                cell_esds[ax] = float(c_dict[f"{ax}_esd"])

        # HAP data
        hap = p.data.get("Histograms", {}).get(hist.name, {})

        # Scale
        sc_val = float(scales_dict.get(p_name, hap.get("Scale", [1.0, True])[0]))
        sc_refine = bool(hap.get("Scale", [1.0, True])[1])
        sc_esd = float(last_esds.get(f"{p_idx}:{hist_idx}:Scale", 0.0))

        # Size
        size_raw = hap.get("Size", ["isotropic", [0.1, 1.0, 1.0], [False, False, False]])
        size_type = str(size_raw[0])
        size_val = float(size_raw[1][0]) if len(size_raw) > 1 and len(size_raw[1]) > 0 else 0.1
        size_refine = bool(size_raw[2][0]) if len(size_raw) > 2 and len(size_raw[2]) > 0 else False
        size_esd = float(last_esds.get(f"{p_idx}:{hist_idx}:Size;i", 0.0))

        # Mustrain
        mustrain_raw = hap.get(
            "Mustrain", ["isotropic", [500.0, 1000.0, 1.0], [False, False, False]]
        )
        ms_type = str(mustrain_raw[0])
        ms_val = (
            float(mustrain_raw[1][0])
            if len(mustrain_raw) > 1 and len(mustrain_raw[1]) > 0
            else 500.0
        )
        ms_refine = (
            bool(mustrain_raw[2][0])
            if len(mustrain_raw) > 2 and len(mustrain_raw[2]) > 0
            else False
        )
        ms_esd = float(last_esds.get(f"{p_idx}:{hist_idx}:Mustrain;i", 0.0))

        # Preferred Orientation
        po_raw = hap.get("Pref.Ori.", ["MD", 1.0, False, [0, 0, 1]])
        po_type = str(po_raw[0])
        po_ratio = float(po_raw[1])
        po_refine = bool(po_raw[2])
        po_axis = (
            [int(x) for x in po_raw[3]]
            if len(po_raw) > 3 and isinstance(po_raw[3], (list, tuple))
            else [0, 0, 1]
        )
        po_esd = float(last_esds.get(f"{p_idx}:{hist_idx}:MD", 0.0))

        # Atoms summary
        atoms_list = []
        any_refine_xyz = False
        any_refine_uiso = False
        with suppress(*g2_safe_errors()):
            for at in p.atoms():
                rec = G2AtomRecord(at)
                r_xyz = bool(rec.refinement_flag in ("X", "XU", "U")) and "X" in rec.refinement_flag
                r_u = bool(rec.refinement_flag in ("U", "XU")) or (len(at) > 11 and at[11] != 0.0)
                if r_xyz:
                    any_refine_xyz = True
                if r_u:
                    any_refine_uiso = True
                atoms_list.append(
                    {
                        "label": rec.label,
                        "element": rec.element,
                        "xyz": [round(float(x), 5) for x in rec.xyz],
                        "uiso": round(float(rec.uiso), 4),
                        "mult": int(rec.mult),
                        "refine_xyz": r_xyz,
                        "refine_uiso": r_u,
                    }
                )

        pawley_flag = bool(gen.get("doPawley", False))
        phases_map[p_name] = {
            "space_group": sg,
            "pawley": pawley_flag,
            "weight_pct": round(wt_pct, 2),
            "weight_pct_esd": round(wt_esd, 2) if wt_esd is not None else None,
            "scale": {
                "value": sc_val,
                "refine": sc_refine,
                "esd": sc_esd,
                "min": 0.0,
            },
            "cell": {
                "refine": cell_refine,
                "a": round(a_val, 6),
                "b": round(b_val, 6),
                "c": round(c_val, 6),
                "alpha": round(alpha_val, 4),
                "beta": round(beta_val, 4),
                "gamma": round(gamma_val, 4),
                "volume": round(vol_val, 3),
                "esds": cell_esds,
            },
            "profile": {
                "size": {
                    "type": size_type,
                    "value": round(size_val, 4),
                    "refine": size_refine,
                    "esd": size_esd,
                    "min": BOUND_SIZE_MIN,
                    "max": BOUND_SIZE_MAX,
                },
                "mustrain": {
                    "type": ms_type,
                    "value": round(ms_val, 2),
                    "refine": ms_refine,
                    "esd": ms_esd,
                    "min": BOUND_MUSTRAIN_MIN,
                    "max": BOUND_MUSTRAIN_MAX,
                },
            },
            "preferred_orientation": {
                "type": po_type,
                "ratio": round(po_ratio, 4),
                "refine": po_refine,
                "axis": po_axis,
                "esd": po_esd,
                "min": BOUND_MD_MIN,
                "max": BOUND_MD_MAX,
            },
            "atoms": {
                "refine_xyz": any_refine_xyz,
                "refine_uiso": any_refine_uiso,
                "count": len(atoms_list),
            },
        }

    doc: dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "histogram": {
            "name": hist_name,
            "instrument": {
                "Zero": {
                    "value": round(zero_val, 6),
                    "refine": zero_refine,
                    "esd": zero_esd,
                    "min": registered_limit(gpx, "parmMin", f":{hist_idx}:Zero", BOUND_ZERO_MIN),
                    "max": registered_limit(gpx, "parmMax", f":{hist_idx}:Zero", BOUND_ZERO_MAX),
                },
                "SH/L": {
                    "value": round(shl_val, 6),
                    "refine": shl_refine,
                    "min": registered_limit(gpx, "parmMin", f":{hist_idx}:SH/L", BOUND_SHL_MIN),
                    "max": registered_limit(gpx, "parmMax", f":{hist_idx}:SH/L", BOUND_SHL_MAX),
                },
                **{
                    coefficient: {
                        "value": round(caglioti_val[coefficient], 6),
                        "refine": caglioti_refine[coefficient],
                        "min": registered_limit(gpx, "parmMin", f":{hist_idx}:{coefficient}", -1.0),
                        "max": registered_limit(gpx, "parmMax", f":{hist_idx}:{coefficient}", 1.0),
                    }
                    for coefficient in CAGLIOTI_COEFFICIENTS
                },
                "ka2_ratio": {
                    "value": round(ka2_val, 6),
                    "refine": ka2_refine,
                    "min": registered_limit(gpx, "parmMin", f":{hist_idx}:I(L2)/I(L1)", 0.0),
                    "max": registered_limit(gpx, "parmMax", f":{hist_idx}:I(L2)/I(L1)", 1.0),
                },
            },
            "sample": {
                "DisplaceX": {
                    "value": round(dispx_val, 6),
                    "refine": dispx_refine,
                    "esd": dispx_esd,
                    "min": registered_limit(
                        gpx, "parmMin", f":{hist_idx}:DisplaceX", BOUND_DISPLACEX_MIN
                    ),
                    "max": registered_limit(
                        gpx, "parmMax", f":{hist_idx}:DisplaceX", BOUND_DISPLACEX_MAX
                    ),
                },
            },
            "background": {
                "type": bkg_type,
                "refine": bkg_refine,
                "n_terms": bkg_n_terms,
                "coeffs": [round(c, 4) for c in bkg_coeffs],
                "esds": [round(e, 4) for e in bkg_esds],
            },
        },
        "phases": phases_map,
    }

    # 5. Apply stage preset if specified
    if preset is not None:
        apply_stage_preset_to_dict(doc, preset)

    return doc


def apply_stage_preset_to_dict(doc: dict[str, Any], preset: int | str) -> None:
    """Populate default Stage-Gate refinement booleans on the parameter dict.

    Presets:
      1: Scale + Background -> True; all others False.
      2: Zero Shift -> True.
      3: Unit Cell -> True for phases with W_p >= 2.0%, False for microphases (<2.0%).
         The lock on a microphase is reported in the log and carries a comment in the
         document, and it is only a default: a caller who declares `cell.refine = true`
         after the preset is applied gets the cell refined, because the refinement reads
         the document last (see the guardrail "Declared Document Wins").
      4: Atomic coordinates -> True for dominant phase (W_p >= 10.0%).
      5: Atomic thermal Uiso -> True for main phases (W_p >= 5.0%).
      7: Profile Size + Mustrain -> True for main phases (W_p >= 2.0%), False for microphases.
      8: Preferred orientation -> True for dominant phase.
    """
    canon = normalize_stage_name(str(preset))

    if canon == StageId.SCALE_AND_BACKGROUND:
        doc["histogram"]["background"]["refine"] = True
        doc["histogram"]["instrument"]["Zero"]["refine"] = False
        doc["histogram"]["sample"]["DisplaceX"]["refine"] = False
        for p_cfg in doc.get("phases", {}).values():
            p_cfg["scale"]["refine"] = not bool(p_cfg.get("pawley", False))
            p_cfg["cell"]["refine"] = False
            p_cfg["profile"]["size"]["refine"] = False
            p_cfg["profile"]["mustrain"]["refine"] = False
            p_cfg["preferred_orientation"]["refine"] = False
            p_cfg["atoms"]["refine_xyz"] = False
            p_cfg["atoms"]["refine_uiso"] = False

    elif canon == StageId.FINAL_SCALE:
        doc["histogram"]["background"]["refine"] = False
        for p_cfg in doc.get("phases", {}).values():
            p_cfg["scale"]["refine"] = not bool(p_cfg.get("pawley", False))

    elif canon == StageId.ZERO_SHIFT:
        doc["histogram"]["instrument"]["Zero"]["refine"] = True
        doc["histogram"]["sample"]["DisplaceX"]["refine"] = False

    elif canon == StageId.UNIT_CELL:
        locked: list[str] = []
        for name, p_cfg in doc.get("phases", {}).items():
            wt = float(p_cfg.get("weight_pct", 100.0))
            if bool(p_cfg.get("pawley", False)) or wt >= PHASE_MICRO_CELL_LOCK_PERCENT:
                p_cfg["cell"]["refine"] = True
            else:
                p_cfg["cell"]["refine"] = False
                p_cfg["cell"]["comment"] = (
                    "MICROPHASE: cell locked to the CIF reference to avoid a singular "
                    "metric; an explicit cell.refine = true in this document overrides "
                    "the lock"
                )
                locked.append(f"{name} ({wt:.2f}%)")
        if locked:
            logger.info(
                "Stage preset 3 locked the unit cell of {} phase(s) below {} wt%: {}",
                len(locked),
                PHASE_MICRO_CELL_LOCK_PERCENT,
                ", ".join(locked),
            )

    elif canon == StageId.ATOMIC_COORDINATES:
        for p_cfg in doc.get("phases", {}).values():
            wt = float(p_cfg.get("weight_pct", 100.0))
            p_cfg["atoms"]["refine_xyz"] = bool(wt >= 10.0)

    elif canon in (StageId.OVERALL_DISPLACEMENT, StageId.GROUP_B_ISO):
        for p_cfg in doc.get("phases", {}).values():
            wt = float(p_cfg.get("weight_pct", 100.0))
            p_cfg["atoms"]["refine_uiso"] = bool(wt >= 5.0)

    elif canon == StageId.SAMPLE_PROFILE_XY:
        for p_cfg in doc.get("phases", {}).values():
            wt = float(p_cfg.get("weight_pct", 100.0))
            is_main = wt >= PHASE_MICRO_CELL_LOCK_PERCENT
            p_cfg["profile"]["size"]["refine"] = is_main
            p_cfg["profile"]["mustrain"]["refine"] = is_main
            p_cfg["atoms"]["refine_xyz"] = False
            p_cfg["atoms"]["refine_uiso"] = False

    elif canon == StageId.PREFERRED_ORIENTATION:
        for p_cfg in doc.get("phases", {}).values():
            wt = float(p_cfg.get("weight_pct", 100.0))
            p_cfg["preferred_orientation"]["refine"] = bool(wt >= 20.0)

    elif canon == StageId.INSTRUMENT_PROFILE:
        for coeff in ("U", "V", "W"):
            if coeff in doc["histogram"]["instrument"]:
                doc["histogram"]["instrument"][coeff]["refine"] = True


# -----------------------------------------------------------------------------
# Parameter Application & Refinement
# -----------------------------------------------------------------------------
def apply_params_to_project(
    gpx: Any,
    params_dict: dict[str, Any],
    hist_idx: int = 0,
) -> None:
    """Inject parameter values, bounds, and refinement flags into GSAS-II data tree."""
    histograms = gpx.histograms()
    if not histograms or hist_idx >= len(histograms):
        raise ValueError(f"Histogram index {hist_idx} out of range.")
    hist = histograms[hist_idx]

    # index_ids() resolves phase and histogram identifiers and raises a bare
    # Exception("No Phases in GPX file") on a project without phases. The
    # precondition is checked here so the message names the action instead.
    if hasattr(gpx, "phases") and not gpx.phases():
        raise ValueError(
            "Project contains no phases: add a phase before applying a parameter document."
        )

    # Fill the rId lookup tables before touching any limit (cf. apply_box_constraints).
    if hasattr(gpx, "index_ids"):
        gpx.index_ids()

    h_cfg = params_dict.get("histogram", {})

    # 1. Background
    bkg_cfg = h_cfg.get("background", {})
    bkg_raw = hist.data["Background"][0]
    if "refine" in bkg_cfg:
        bkg_raw[1] = bool(bkg_cfg["refine"])
    if "coeffs" in bkg_cfg and isinstance(bkg_cfg["coeffs"], list):
        n_coeffs = len(bkg_cfg["coeffs"])
        bkg_raw[2] = n_coeffs
        bkg_raw[3 : 3 + n_coeffs] = [float(x) for x in bkg_cfg["coeffs"]]

    # 2. Instrument parameters (3-field records [init, value, refine])
    inst_cfg = h_cfg.get("instrument", {})
    inst_raw = hist.data["Instrument Parameters"][0]
    instrument_params = INSTRUMENT_PARAMETER_NAMES
    for cfg_name, gsas_name in instrument_params:
        entry = inst_cfg.get(cfg_name)
        if not entry or gsas_name not in inst_raw:
            continue
        write_inst_param(
            inst_raw,
            gsas_name,
            value=float(entry["value"]) if "value" in entry else None,
            refine=bool(entry["refine"]) if "refine" in entry else None,
        )
        apply_declared_limits(gpx, entry, f":{hist_idx}:{gsas_name}")

    # 3. Sample parameters
    samp_cfg = h_cfg.get("sample", {})
    samp_raw = hist.data["Sample Parameters"]
    if "DisplaceX" in samp_cfg and "DisplaceX" in samp_raw:
        dx_entry = samp_cfg["DisplaceX"]
        if "value" in dx_entry:
            samp_raw["DisplaceX"][0] = float(dx_entry["value"])
        if "refine" in dx_entry:
            samp_raw["DisplaceX"][1] = bool(dx_entry["refine"])
        apply_declared_limits(gpx, dx_entry, f":{hist_idx}:DisplaceX")

    # 4. Phase-level parameters
    phases_cfg = params_dict.get("phases", {})
    for p_idx, p in enumerate(gpx.phases()):
        p_cfg = phases_cfg.get(p.name)
        if not p_cfg:
            continue

        gen = p.data.get("General", {})
        hap = p.data.get("Histograms", {}).get(hist.name, {})

        # Scale
        sc_entry = p_cfg.get("scale", {})
        if bool(gen.get("doPawley", False)):
            gen["Pawley ref"] = True
            if "Scale" in hap:
                hap["Scale"] = [1.0, False]
        elif "Scale" in hap:
            if "value" in sc_entry:
                hap["Scale"][0] = float(sc_entry["value"])
            if "refine" in sc_entry:
                hap["Scale"][1] = bool(sc_entry["refine"])
            apply_declared_limits(gpx, sc_entry, f"{p_idx}:{hist_idx}:Scale")

        # Cell & Volume Synchronization via gemmi (Micro-accent #1)
        c_entry = p_cfg.get("cell", {})
        if "Cell" in gen and len(gen["Cell"]) >= 8:
            cell_raw = gen["Cell"]
            if "refine" in c_entry:
                cell_raw[0] = bool(c_entry["refine"])
            # Update lattice lengths / angles if specified
            a_val = float(c_entry.get("a", cell_raw[1]))
            b_val = float(c_entry.get("b", cell_raw[2]))
            c_val = float(c_entry.get("c", cell_raw[3]))
            alpha_val = float(c_entry.get("alpha", cell_raw[4]))
            beta_val = float(c_entry.get("beta", cell_raw[5]))
            gamma_val = float(c_entry.get("gamma", cell_raw[6]))
            cell_raw[1:7] = [a_val, b_val, c_val, alpha_val, beta_val, gamma_val]

            # Recompute exact metric tensor and volume through gemmi to prevent GSAS-II metric mismatch
            calc_vol = float(
                gemmi.UnitCell(a_val, b_val, c_val, alpha_val, beta_val, gamma_val).volume
            )
            cell_raw[7] = calc_vol

        # Profile (Size & Mustrain)
        prof_entry = p_cfg.get("profile", {})
        if "size" in prof_entry and "Size" in hap:
            sz_cfg = prof_entry["size"]
            if "value" in sz_cfg:
                hap["Size"][1][0] = float(sz_cfg["value"])
            if "refine" in sz_cfg:
                hap["Size"][2][0] = bool(sz_cfg["refine"])
            apply_declared_limits(gpx, sz_cfg, f"{p_idx}:{hist_idx}:Size;i")

        if "mustrain" in prof_entry and "Mustrain" in hap:
            ms_cfg = prof_entry["mustrain"]
            if "value" in ms_cfg:
                hap["Mustrain"][1][0] = float(ms_cfg["value"])
            if "refine" in ms_cfg:
                hap["Mustrain"][2][0] = bool(ms_cfg["refine"])
            apply_declared_limits(gpx, ms_cfg, f"{p_idx}:{hist_idx}:Mustrain;i")

        # Preferred Orientation
        po_entry = p_cfg.get("preferred_orientation", {})
        if "Pref.Ori." in hap:
            if "ratio" in po_entry:
                hap["Pref.Ori."][1] = float(po_entry["ratio"])
            if "refine" in po_entry:
                hap["Pref.Ori."][2] = bool(po_entry["refine"])
            if "axis" in po_entry and isinstance(po_entry["axis"], (list, tuple)):
                hap["Pref.Ori."][3] = [int(x) for x in po_entry["axis"]]
            apply_declared_limits(gpx, po_entry, f"{p_idx}:{hist_idx}:MD")

        # Atoms refinement flags
        at_cfg = p_cfg.get("atoms", {})
        ref_xyz = bool(at_cfg.get("refine_xyz", False))
        ref_u = bool(at_cfg.get("refine_uiso", False))
        if ref_xyz and ref_u:
            p.set_refinements({"Atoms": {"all": "XU"}})
        elif ref_xyz:
            p.set_refinements({"Atoms": {"all": "X"}})
        elif ref_u:
            p.set_refinements({"Atoms": {"all": "U"}})
        else:
            p.set_refinements({"Atoms": {"all": ""}})


def solve_sequential_scales(
    gpx: Any,
    hist_idx: int = 0,
    cycles_per_step: int = 2,
    matrix_idx: int = 0,
    matrix_scale_start: float = 0.15,
    minor_scale_start: float = 0.005,
    release_matrix_cell: bool = False,
) -> dict[str, float]:
    """Damped stepwise scale factorization for multiphase Rietveld fits.

    Refining the histogram scale, every phase scale and the Chebyshev background
    in one shot makes the design matrix collinear (rho ~ 1) and the
    Levenberg-Marquardt solver diverges on the very first cycle. This routine
    removes the collinearity by factorization:

    * **A** - refine the matrix phase scale alone, background frozen;
    * **B** - refine every secondary phase scale alone, one phase at a time;
    * **C** - refine all phase scales jointly - the background stays frozen;
    * **D** - unfreeze the background (and, only on request, the matrix unit cell).

    Secondary-phase cells are locked from step A to step C, so the metric tensor
    cannot diverge on underdetermined (trace) phases while the scales are being
    factorized. The lock is a protection, not an override: before step D every phase
    cell flag is restored to the value the caller's parameter document declared,
    because that document is the single source of truth for the model -- the same
    rule ``refine_project_with_params`` states for box constraints. A phase whose
    cell the document leaves locked stays locked, which is the case this protection
    was written for. Every phase scale is pinned to ``>= 0`` through native GSAS-II
    ``parmMin`` box constraints, which prevents the "negative scale collapse" that
    destabilises the profile.

    ``release_matrix_cell`` is off by default: releasing the matrix cell while the
    scale vector is still being factorized was measured to drive the metric tensor
    to absurd values (GSAS-II aborts with "Invalid cell metric tensor for phase").
    Refine the unit cell afterwards, as a separate controlled step, once the scale
    vector has settled. It may add a matrix-cell refinement the document left unset,
    and it may never remove one the document declared.

    Returns:
        Mapping ``phase_name -> refined scale`` for every phase in the project.
    """
    histograms = gpx.histograms()
    if not histograms:
        raise ValueError("Project contains no histograms; nothing to factorize.")
    if hist_idx >= len(histograms):
        raise ValueError(
            f"Histogram index {hist_idx} out of range ({len(histograms)} histogram(s) in project)."
        )
    hist = histograms[hist_idx]
    hist_name: str = hist.name

    phases = gpx.phases()
    if not phases:
        raise ValueError("Project contains no phases; nothing to factorize.")
    if not 0 <= matrix_idx < len(phases):
        raise ValueError(f"matrix_idx {matrix_idx} out of range ({len(phases)} phase(s)).")

    # 1. Remove the histogram-scale / phase-scale collinearity (trap 1).
    hist.SampleParameters["Scale"] = [1.0, False]

    # 2. Freeze the background for steps A-C: with a fixed background its
    #    covariance with the phase scales is exactly zero.
    background = hist.data["Background"][0]
    background_refine = bool(background[1])
    background[1] = False

    # 3. Sanitize the starting scales and pin them to non-negative values (traps 2, 3).
    # The declared model has already been applied, so the cell flags read here are the
    # caller's, not the schema's. They are captured because steps A-C must freeze the
    # cells to remove the scale/cell collinearity, and a freeze that is never lifted is
    # not a freeze but a silent override of the declared document: a caller who declared
    # `cell.refine = true` for a minor phase got it back as false, with no message.
    declared_cell_refine = {p.name: bool(p.data["General"]["Cell"][0]) for p in phases}
    for p_idx, p in enumerate(phases):
        p.data["General"]["Cell"][0] = False
        hap = p.data["Histograms"][hist_name]
        start = matrix_scale_start if p_idx == matrix_idx else minor_scale_start
        hap["Scale"] = [max(float(start), 0.0), False]
        set_param_limit(gpx, "parmMin", f"{p_idx}:{hist_idx}:Scale", 0.0)
    logger.info(
        "Sequential scales: {} phase(s), matrix='{}', background frozen (was refine={})",
        len(phases),
        phases[matrix_idx].name,
        background_refine,
    )

    def refine_once() -> None:
        gpx.set_Controls("cycles", int(cycles_per_step))
        gpx.refine(makeBack=False)

    # Step A: matrix scale alone.
    logger.info("Sequential scales step A: matrix phase '{}' only", phases[matrix_idx].name)
    phases[matrix_idx].data["Histograms"][hist_name]["Scale"][1] = True
    refine_once()
    phases[matrix_idx].data["Histograms"][hist_name]["Scale"][1] = False

    # Step B: each secondary phase scale alone, one at a time.
    for p_idx, p in enumerate(phases):
        if p_idx == matrix_idx:
            continue
        logger.info("Sequential scales step B: phase '{}' only", p.name)
        p.data["Histograms"][hist_name]["Scale"][1] = True
        refine_once()
        p.data["Histograms"][hist_name]["Scale"][1] = False

    # Step C: all scales jointly, background still frozen.
    logger.info("Sequential scales step C: joint scales, background still frozen")
    for p in phases:
        p.data["Histograms"][hist_name]["Scale"][1] = True
    refine_once()

    # Step D: unfreeze the background; the unit cell only on explicit request.
    logger.info(
        "Sequential scales step D: unfreezing background; restoring declared cell flags "
        "{} (explicit matrix cell release: {})",
        declared_cell_refine,
        release_matrix_cell,
    )
    background[1] = True
    # Restore every declared cell flag, not just the matrix phase's. The document stays the
    # single source of truth for the model; the factorization only borrows the flags while
    # it converges the scales, so what it borrowed is what it returns.
    for p in phases:
        p.data["General"]["Cell"][0] = declared_cell_refine.get(p.name, False)
    if release_matrix_cell:
        # An explicit request may add a cell refinement the document left unset, but it must
        # not be able to take one away: enabling is a permission, disabling is an override of
        # the caller's declaration.
        phases[matrix_idx].data["General"]["Cell"][0] = True
    refine_once()

    refined = {p.name: float(p.data["Histograms"][hist_name]["Scale"][0]) for p in phases}
    logger.info(
        "Sequential scales done: {}",
        {name: round(value, 6) for name, value in refined.items()},
    )
    return refined


#: A refined cell length may not leave this band relative to the value the caller declared.
#: The band is wide on purpose: a real solid solution or thermal expansion moves a cell by
#: a fraction of a percent, while the degeneracy that motivated the check moved it by two
#: orders of magnitude.
CELL_TOLERANCE_FRACTION = 0.25

#: No physically meaningful cell length in powder work is below this, in angstroms.
MIN_PHYSICAL_CELL_LENGTH = 1.0


#: (document key, GSAS-II record name) of every instrument parameter this document
#: carries. The document uses a readable key for the K-alpha2 ratio while the data tree
#: keeps the GSAS-II spelling.
INSTRUMENT_PARAMETER_NAMES = (
    ("Zero", "Zero"),
    ("SH/L", "SH/L"),
    ("ka2_ratio", "I(L2)/I(L1)"),
    ("U", "U"),
    ("V", "V"),
    ("W", "W"),
    ("X", "X"),
    ("Y", "Y"),
    ("Z", "Z"),
)

#: A parameter left pinned on a bound counts as a degeneracy once it has travelled this
#: fraction of its declared value away from it.
PINNED_TRAVEL_FRACTION = 0.5

#: A refinement that improves R_wp by less than this is treated as having gained nothing.
NO_GAIN_PERCENT = 0.01


def _pinned_declarations(declared: dict[str, Any], gpx: Any) -> list[str]:
    """Return declared-free instrument parameters the fit left pinned on a box bound.

    A parameter with no interior optimum runs to whatever bound it is given: measured, the
    Lorentzian terms ended at X = 1.00000, Y = -1.00000, Z = 1.00000 while R_wp moved from
    9.19 to 9.22 %, and the fit was reported as successful. A value sitting on its bound is
    not a fitted quantity, and reading it as one repeats the mistake of reading a collapsed
    cell as a lattice.
    """
    pinned: list[str] = []
    try:
        inst_raw = gpx.histograms()[0].data.get("Instrument Parameters", [{}])[0]
    except (IndexError, AttributeError, TypeError):
        return pinned

    instrument = (declared.get("histogram") or {}).get("instrument") or {}
    for key, gsas_name in INSTRUMENT_PARAMETER_NAMES:
        entry = instrument.get(key)
        if not isinstance(entry, dict) or not entry.get("refine"):
            continue
        start = entry.get("value")
        if start is None:
            continue
        value, _ = read_inst_param(inst_raw, gsas_name, 0.0)  # noqa: PU001
        for bound_label in ("min", "max"):
            bound = entry.get(bound_label)
            if bound is None:
                continue
            with suppress(TypeError, ValueError):
                bound_value = float(bound)
                start_value = float(start)
                if abs(value - bound_value) > 1e-9:
                    continue
                if abs(value - start_value) <= PINNED_TRAVEL_FRACTION * max(abs(start_value), 1e-9):
                    continue
                pinned.append(
                    f"{key} pinned at its {bound_label} bound "
                    f"({value:.5f}, declared {start_value:.5f})"
                )
    return pinned


def _cell_degeneracy(declared: dict[str, Any], gpx: Any) -> str:
    """Return a reason when a refined cell left the physically possible range, else "".

    A phase whose unit cell is free and whose peaks are too weak to constrain it lets the
    solver collapse the cell towards zero; its reflections then leave the data range and
    the scale explodes to compensate, which can *improve* R_wp. The fit is then reported as
    successful while the model is meaningless -- measured: a = 0.008 A with a 100 % phase
    fraction at R_wp 8.99 %. Refusing such a model is the only way to stop a good metric
    from hiding a broken result.
    """
    for name, config in (declared.get("phases") or {}).items():
        entry = (config or {}).get("cell") or {}
        fallback = entry.get("a")
        if not fallback:
            continue
        for phase in gpx.phases():
            if phase.name != name:
                continue
            refined = phase.data.get("General", {}).get("Cell", [])
            if len(refined) < 4:
                continue
            for label, value in zip(("a", "b", "c"), refined[1:4]):
                expected = entry.get(label) or fallback
                with suppress(TypeError, ValueError):
                    length = float(value)
                    reference = float(expected)
                    if length < MIN_PHYSICAL_CELL_LENGTH:
                        return (
                            f"Degenerate cell for phase '{name}': {label} refined to "
                            f"{length:.4f} A, below the physical floor of "
                            f"{MIN_PHYSICAL_CELL_LENGTH} A"
                        )
                    if abs(length - reference) > CELL_TOLERANCE_FRACTION * reference:
                        return (
                            f"Degenerate cell for phase '{name}': {label} moved from the "
                            f"declared {reference:.4f} A to {length:.4f} A, beyond "
                            f"{CELL_TOLERANCE_FRACTION:.0%}"
                        )
    return ""


def _unapplied_declarations(declared: dict[str, Any], gpx: Any) -> list[str]:
    """Return the declared refinement flags that the project does not hold afterwards.

    The document is the specification of the model, so a flag that silently fails to reach
    the project turns a declared refinement into a locked parameter. The only trace is then
    a zero esd read back much later -- that is exactly how the lost cell flag was found --
    so the discrepancy is named here, while the run is still being interpreted.
    """
    problems: list[str] = []
    try:
        hist_name = gpx.histograms()[0].name
    except (IndexError, AttributeError, TypeError):
        return problems

    for name, config in (declared.get("phases") or {}).items():
        for phase in gpx.phases():
            if phase.name != name:
                continue
            hap = phase.data.get("Histograms", {}).get(hist_name, {})
            if not isinstance(hap, dict):
                continue
            profile = (config or {}).get("profile") or {}
            for key, wanted, index in (
                ("Scale", (config.get("scale") or {}).get("refine"), 1),
                ("Size", (profile.get("size") or {}).get("refine"), None),
                ("Mustrain", (profile.get("mustrain") or {}).get("refine"), None),
            ):
                if wanted is None:
                    continue
                record = hap.get(key)
                with suppress(TypeError, IndexError, ValueError):
                    held = record[index] if index is not None else record[2][0]
                    if bool(held) != bool(wanted):
                        problems.append(f"{name}.{key} holds {bool(held)}, declared {bool(wanted)}")
    return problems


def refine_project_with_params(
    gpx: Any,
    params_dict: dict[str, Any],
    cycles: int = 5,
    project_path: Path | None = None,
    hist_idx: int = 0,
    sequential_scales: bool = False,
) -> tuple[dict[str, Any], bool, str]:
    """Execute Rietveld refinement step governed strictly by params_dict declarations.

    When ``sequential_scales`` is set, the declared model is applied first and a
    damped scale factorization (:func:`solve_sequential_scales`) then converges the
    scale vector on top of it, before the final coupled refinement. The JSON
    document stays the single source of truth for the model.

    Returns:
        (updated_params_dict, diverged_bool, fail_reason_str)
    """
    histograms = gpx.histograms()
    if not histograms:
        return params_dict, True, "No histograms in project."
    hist = histograms[hist_idx]

    # Measure baseline R_wp
    prev_r_wp = 99.0
    with suppress(*g2_safe_errors()):
        prev_curves = extract_profile_curves(hist)
        prev_m = calculate_profile_metrics(prev_curves)
        prev_r_wp = prev_m.r_wp_percent

    # 1. Create safety backup checkpoint if path provided
    if project_path:
        ckpt_path = project_path.parent / f"{project_path.stem}_checkpoint.gpx"
        with suppress(*g2_safe_errors()):
            shutil.copy2(project_path, ckpt_path)

    # 2. Default box constraints first, so that the declared model injected below
    #    can override them. Under the previous order apply_box_constraints ran
    #    after the document was applied and clobbered a caller's declared interval
    #    with the schema default on every refinement: a declared +-1 um
    #    specimen-displacement bound came back as +-1000 um.
    for p_idx in range(len(gpx.phases())):
        apply_box_constraints(gpx, phase_id=p_idx, hist_id=hist_idx)

    # 3. Inject the declared model (cells, refinement flags, declared limits) so
    #    that the factorization below runs on top of the declared lattice.
    apply_params_to_project(gpx, params_dict, hist_idx=hist_idx)

    # What the injection actually achieved is reported rather than assumed: a declared flag
    # that did not land would silently lock a parameter the caller asked to refine.
    unapplied = _unapplied_declarations(params_dict, gpx)
    if unapplied:
        logger.warning(
            "Declared refinement flags not present in the project after injection: {}",
            "; ".join(unapplied),
        )

    # --- Optional damped scale factorization (multiphase cold start) -------------
    # Runs AFTER the declared model is applied: the factorizer converges the scale
    # vector, and nothing may overwrite it afterwards. Placing it before
    # apply_params_to_project would reset the converged scales to the declared
    # starting values.
    if sequential_scales:
        phase_count = len(gpx.phases())
        if phase_count < 2:
            logger.warning(
                "sequential_scales requested with {} phase(s); factorization is a no-op.",
                phase_count,
            )
        try:
            solve_sequential_scales(gpx, hist_idx=hist_idx)
        except Exception as e_seq:  # noqa: BLE001
            fail_reason = f"Sequential scale factorization failed: {e_seq}"
            logger.error(fail_reason)
            if project_path:
                ckpt_path = project_path.parent / f"{project_path.stem}_checkpoint.gpx"
                if ckpt_path.exists():
                    with suppress(*g2_safe_errors()):
                        shutil.copy2(ckpt_path, project_path)
                        gpx.reload()
            return params_dict, True, fail_reason

    # 4. Set cycles in Controls and Controls.data (Micro-accent #2)
    target_cycles = max(1, int(cycles))
    gpx.set_Controls("cycles", target_cycles)
    gpx.data.setdefault("Controls", {}).setdefault("data", {})["max cyc"] = target_cycles

    # 5. Execute native refinement with current project settings
    from xrd_project import clamp_pawley_reflections, is_pawley_phase

    has_pawley = any(is_pawley_phase(p) for p in gpx.phases())
    diverged = False
    fail_reason = ""
    try:
        # Direct native refinement using exact flags configured in project
        gpx.refine(makeBack=False)
        if has_pawley:
            clamp_pawley_reflections(gpx, lock_nonpositive=False)
            sync_pawley_reflections_to_hist(gpx, hist)
    except Exception as e_refine:  # noqa: BLE001
        diverged = True
        fail_reason = f"Refinement exception: {e_refine}"

    # 6. Evaluate convergence & metrics
    new_r_wp = 99.0
    new_gof = 1.0
    new_dw = 2.0
    if not diverged:
        try:
            new_curves = extract_profile_curves(hist)
            vary_cnt = len(gpx.data.get("Controls", {}).get("VaryList", []))
            new_m = calculate_profile_metrics(new_curves, n_free_parameters=max(1, vary_cnt))
            new_r_wp = new_m.r_wp_percent
            new_gof = new_m.gof
            new_dw = new_m.durbin_watson

            if np.isnan(new_r_wp) or (prev_r_wp < 90.0 and new_r_wp > prev_r_wp * 1.35):
                diverged = True
                fail_reason = f"R_wp exploded from {prev_r_wp:.2f}% to {new_r_wp:.2f}%"

            # A refinement can also report a *better* metric while producing a physically
            # impossible model: an unbounded unit cell collapses towards zero, the phase's
            # reflections leave the data range and its scale explodes to compensate. Such a
            # fit is a degeneracy dressed as a success, so it is refused on the same footing
            # as an exploded R_wp -- and rolled back by the same path below.
            if not diverged:
                degeneracy = _cell_degeneracy(params_dict, gpx)
                if degeneracy:
                    diverged = True
                    fail_reason = degeneracy

            # A parameter pinned on its bound is a degeneracy only when the refinement also
            # gained nothing: with a real improvement the boundary may simply be the best
            # available, while without one the fit has produced no result *and* an undefined
            # quantity. Both halves are required, so a legitimate boundary fit is kept.
            if not diverged:
                pinned = _pinned_declarations(params_dict, gpx)
                gained = prev_r_wp >= 90.0 or new_r_wp <= prev_r_wp - NO_GAIN_PERCENT
                if pinned and not gained:
                    diverged = True
                    fail_reason = (
                        "Refinement gained nothing and left parameter(s) pinned: "
                        + "; ".join(pinned)
                    )
                elif pinned:
                    logger.warning("Parameter(s) pinned at a bound: {}", "; ".join(pinned))
        except Exception as e_met:  # noqa: BLE001
            diverged = True
            fail_reason = f"Metric calculation failed: {e_met}"

    # 7. Rollback if diverged and checkpoint exists
    if diverged and project_path:
        ckpt_path = project_path.parent / f"{project_path.stem}_checkpoint.gpx"
        if ckpt_path.exists():
            with suppress(*g2_safe_errors()):
                shutil.copy2(ckpt_path, project_path)
                gpx.reload()
            logger.warning(f"Refinement diverged ({fail_reason}). Rolled back to {ckpt_path.name}")
        return params_dict, True, fail_reason

    # Save cleanly
    gpx.save()

    # 8. Re-dump fresh parameters with updated values, shifts, and ESDs
    updated_dict = dump_project_params(gpx, hist_idx=hist_idx)

    # 8b. A registered bound is not a respected bound. Whatever the solver
    #     stepped over is clamped back to its declared interval, and the affected
    #     step is repeated once; the report names what moved.
    limits_clamped = clamp_to_declared_limits(updated_dict, params_dict)
    remaining_violations = enforce_declared_limits(updated_dict, params_dict)
    if limits_clamped:
        logger.warning(
            "Parameters refined outside their declared interval and were clamped: {}",
            ", ".join(limits_clamped),
        )
        apply_params_to_project(gpx, updated_dict, hist_idx=hist_idx)
        gpx.set_Controls("cycles", 1)
        with suppress(*g2_safe_errors()):
            gpx.refine(makeBack=False)
        updated_dict = dump_project_params(gpx, hist_idx=hist_idx)
        remaining_violations = enforce_declared_limits(updated_dict, params_dict)
        if remaining_violations:
            logger.warning(
                "Declared limits still exceeded after clamping: {}",
                "; ".join(remaining_violations),
            )

    # 8c. The stage path reports reflections sharper than the hardware limit; the
    #     declarative path has to do the same, otherwise a profile can be driven
    #     below the instrumental floor through this route alone.
    for message in instrumental_lower_bound_report(hist):
        logger.warning(message)

    # 9. Synchronize history sidecar (Micro-accent #3)
    if project_path:
        from xrd_project import load_sidecar_history, save_sidecar_history

        history = load_sidecar_history(project_path)
        history["iterations"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "params_refine",
                "status": "SUCCESS",
                "cycles": target_cycles,
                "r_wp_percent": round(new_r_wp, 2),
                "delta_r_wp": round(prev_r_wp - new_r_wp, 3),
                "gof": round(new_gof, 2),
                "durbin_watson": round(new_dw, 2),
            }
        )
        save_sidecar_history(project_path, history)

    return updated_dict, False, ""


# -----------------------------------------------------------------------------
# Parameter Map ASCII Table Formatter
# -----------------------------------------------------------------------------
def format_parameter_table(
    params_dict: dict[str, Any], last_fit_results: tuple | None = None
) -> str:
    """Format an informative ASCII parameter status table."""
    lines = [
        "=" * 92,
        "REFINEMENT PARAMETER MAP",
        "=" * 92,
        f"{'Category':<14} | {'Parameter':<16} | {'Value':<13} | {'Refine?':<7} | {'Shift (Δ)':<11} | {'ESD (σ)':<11} | {'Status':<10}",
        "-" * 92,
    ]

    h_cfg = params_dict.get("histogram", {})
    inst_cfg = h_cfg.get("instrument", {})
    bkg_cfg = h_cfg.get("background", {})

    # Zero
    z_ent = inst_cfg.get("Zero", {})
    z_ref = "ON" if z_ent.get("refine") else "OFF"
    z_val = f"{z_ent.get('value', 0.0):.6f}"
    z_esd = f"{z_ent.get('esd', 0.0):.6f}" if z_ent.get("esd") else "-"
    lines.append(
        f"{'Histogram':<14} | {'Zero':<16} | {z_val:<13} | {z_ref:<7} | {'-':<11} | {z_esd:<11} | {'LOCKED' if z_ref == 'OFF' else 'ACTIVE':<10}"
    )

    # Background
    b_ref = "ON" if bkg_cfg.get("refine") else "OFF"
    b_terms = bkg_cfg.get("n_terms", 6)
    lines.append(
        f"{'Histogram':<14} | {f'Back ({b_terms} terms)':<16} | {'Chebyshev':<13} | {b_ref:<7} | {'-':<11} | {'-':<11} | {'LOCKED' if b_ref == 'OFF' else 'ACTIVE':<10}"
    )

    # Phases
    for p_name, p_cfg in params_dict.get("phases", {}).items():
        wt = p_cfg.get("weight_pct", 0.0)
        p_hdr = f"{p_name[:12]} ({wt}%)"

        # Scale
        sc_ent = p_cfg.get("scale", {})
        sc_ref = "ON" if sc_ent.get("refine") else "OFF"
        sc_val = f"{sc_ent.get('value', 0.0):.5e}"
        sc_esd = f"{sc_ent.get('esd', 0.0):.5e}" if sc_ent.get("esd") else "-"
        lines.append(
            f"{p_hdr:<14} | {'Scale':<16} | {sc_val:<13} | {sc_ref:<7} | {'-':<11} | {sc_esd:<11} | {'ACTIVE' if sc_ref == 'ON' else 'LOCKED':<10}"
        )

        # Cell
        c_ent = p_cfg.get("cell", {})
        c_ref = "ON" if c_ent.get("refine") else "OFF"
        a_val = f"a={c_ent.get('a', 0.0):.4f}"
        esd_a = c_ent.get("esds", {}).get("a")
        esd_str = f"±{esd_a:.4f}" if esd_a else "-"
        status_lbl = "ACTIVE" if c_ref == "ON" else ("LOCKED (MICRO)" if wt < 2.0 else "LOCKED")
        lines.append(
            f"{p_hdr:<14} | {'Cell Lattice':<16} | {a_val:<13} | {c_ref:<7} | {'-':<11} | {esd_str:<11} | {status_lbl:<10}"
        )

        # Profile Size
        sz_ent = p_cfg.get("profile", {}).get("size", {})
        sz_ref = "ON" if sz_ent.get("refine") else "OFF"
        sz_val = f"{sz_ent.get('value', 0.0):.4f} um"
        lines.append(
            f"{p_hdr:<14} | {'Size Broadening':<16} | {sz_val:<13} | {sz_ref:<7} | {'-':<11} | {'-':<11} | {'ACTIVE' if sz_ref == 'ON' else 'LOCKED':<10}"
        )

        # Profile Mustrain
        ms_ent = p_cfg.get("profile", {}).get("mustrain", {})
        ms_ref = "ON" if ms_ent.get("refine") else "OFF"
        ms_val = f"{ms_ent.get('value', 0.0):.1f}"
        lines.append(
            f"{p_hdr:<14} | {'Microstrain':<16} | {ms_val:<13} | {ms_ref:<7} | {'-':<11} | {'-':<11} | {'ACTIVE' if ms_ref == 'ON' else 'LOCKED':<10}"
        )

    lines.append("=" * 92)
    return "\n".join(lines)
