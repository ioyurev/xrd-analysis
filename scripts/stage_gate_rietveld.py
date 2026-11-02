from __future__ import annotations

import math
import os
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class AtomCol(IntEnum):
    """Column indices in a raw GSAS-II phase atom record list."""

    LABEL = 0
    ELEMENT = 1
    FLAG = 2
    X = 3
    Y = 4
    Z = 5
    MULT = 6
    OCCUPANCY = 7
    UISO = 10


from pathlib import Path
from typing import Any, ClassVar

import gemmi
import numpy as np
from constants import (
    BOUND_CAGLIOTI_UW_MAX,
    BOUND_CAGLIOTI_UW_MIN,
    BOUND_CAGLIOTI_V_MAX,
    BOUND_CAGLIOTI_V_MIN,
    BOUND_DISPLACEX_MAX,
    BOUND_DISPLACEX_MIN,
    BOUND_EXTINCTION_MAX,
    BOUND_EXTINCTION_MIN,
    BOUND_MD_MAX,
    BOUND_MD_MIN,
    BOUND_MUSTRAIN_MAX,
    BOUND_MUSTRAIN_MIN,
    BOUND_SHL_MAX,
    BOUND_SHL_MIN,
    BOUND_SIZE_MAX,
    BOUND_SIZE_MIN,
    BOUND_UISO_MAX,
    BOUND_UISO_MIN,
    BOUND_ZERO_MAX,
    BOUND_ZERO_MIN,
    PHASE_MINOR_WEIGHT_PERCENT,
    PHASE_RB_DOMINANT_PERCENT,
    PHASE_RB_SUBDOMINANT_PERCENT,
    PHASE_RF_DOMINANT_PERCENT,
    PHASE_RF_SUBDOMINANT_PERCENT,
)
from loguru import logger
from pymatgen.core import Structure

__all__ = [
    "ADVANCED_STAGES",
    "PASS_CONTINUE",
    "PASS_EXHAUSTED",
    "PASS_PLATEAU",
    "PASS_REGRESSED",
    "G2AtomRecord",
    "PassDecision",
    "RefinementConfig",
    "RefinementStatus",
    "StageGateRietveld",
    "StageId",
    "apply_box_constraints",
    "build_stage_recipe",
    "clamp_stored_parameters",
    "decide_next_pass",
    "ensure_gsas2_environment",
    "reset_stage_flags",
    "set_param_limit",
    "setup_phase_dist_restraints",
]


class G2AtomRecord:
    """Strongly-typed wrapper around raw GSAS-II atom list entry or G2Atom object."""

    def __init__(self, raw_atom: Any):
        if hasattr(raw_atom, "data"):
            self._raw = raw_atom.data
            self._obj = raw_atom
        elif isinstance(raw_atom, list):
            self._raw = raw_atom
            self._obj = None
        else:
            self._raw = getattr(raw_atom, "_data", [])
            self._obj = raw_atom

    @property
    def label(self) -> str:
        if self._obj and hasattr(self._obj, "label"):
            return str(self._obj.label)
        return str(self._raw[AtomCol.LABEL]) if len(self._raw) > AtomCol.LABEL else ""

    @label.setter
    def label(self, val: str) -> None:
        if len(self._raw) > AtomCol.LABEL:
            self._raw[AtomCol.LABEL] = str(val)

    @property
    def element(self) -> str:
        if self._obj:
            if hasattr(self._obj, "element"):
                return str(self._obj.element)
            if hasattr(self._obj, "elem"):
                return str(self._obj.elem)
        return str(self._raw[AtomCol.ELEMENT]) if len(self._raw) > AtomCol.ELEMENT else ""

    @element.setter
    def element(self, val: str) -> None:
        if len(self._raw) > AtomCol.ELEMENT:
            self._raw[AtomCol.ELEMENT] = str(val)

    @property
    def refinement_flag(self) -> str:
        if self._obj and hasattr(self._obj, "refinement_flags"):
            return str(self._obj.refinement_flags)
        return str(self._raw[AtomCol.FLAG]) if len(self._raw) > AtomCol.FLAG else ""

    @refinement_flag.setter
    def refinement_flag(self, val: str) -> None:
        if len(self._raw) > AtomCol.FLAG:
            self._raw[AtomCol.FLAG] = str(val)

    @property
    def coordinates(self) -> tuple[float, float, float]:
        if self._obj:
            if hasattr(self._obj, "coordinates"):
                c = self._obj.coordinates
                return (float(c[0]), float(c[1]), float(c[2]))
            if hasattr(self._obj, "xyz"):
                c = self._obj.xyz
                return (float(c[0]), float(c[1]), float(c[2]))
        if len(self._raw) > AtomCol.Z:
            return (
                float(self._raw[AtomCol.X]),
                float(self._raw[AtomCol.Y]),
                float(self._raw[AtomCol.Z]),
            )
        return (0.0, 0.0, 0.0)

    @coordinates.setter
    def coordinates(self, coords: tuple[float, float, float]) -> None:
        if len(self._raw) > AtomCol.Z:
            self._raw[AtomCol.X] = float(coords[0])
            self._raw[AtomCol.Y] = float(coords[1])
            self._raw[AtomCol.Z] = float(coords[2])

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.coordinates

    @property
    def has_coordinates(self) -> bool:
        if self._obj:
            return hasattr(self._obj, "coordinates") or hasattr(self._obj, "xyz")
        return len(self._raw) > AtomCol.Z

    @property
    def mult(self) -> float:
        if self._obj and hasattr(self._obj, "mult"):
            return float(self._obj.mult)
        return float(self._raw[AtomCol.MULT]) if len(self._raw) > AtomCol.MULT else 1.0

    @property
    def occupancy(self) -> float:
        if self._obj and hasattr(self._obj, "occupancy"):
            return float(self._obj.occupancy)
        return float(self._raw[AtomCol.OCCUPANCY]) if len(self._raw) > AtomCol.OCCUPANCY else 1.0

    @occupancy.setter
    def occupancy(self, val: float) -> None:
        if len(self._raw) > AtomCol.OCCUPANCY:
            self._raw[AtomCol.OCCUPANCY] = float(val)

    @property
    def uiso(self) -> float:
        if self._obj and hasattr(self._obj, "uiso"):
            return float(self._obj.uiso)
        if len(self._raw) > AtomCol.UISO and self._raw[AtomCol.UISO] is not None:
            try:
                return float(self._raw[AtomCol.UISO])
            except (ValueError, TypeError):
                pass
        return 0.01

    @uiso.setter
    def uiso(self, val: float) -> None:
        if len(self._raw) > AtomCol.UISO:
            self._raw[AtomCol.UISO] = float(val)


def set_param_limit(
    gpx: Any,
    control: str,
    variable: str,
    value: float,
    *,
    required: bool = False,
    _index_already_synced: bool = False,
) -> bool:
    """Set one GSAS-II ``parmMin``/``parmMax`` limit and verify it was registered."""
    applied = False
    readback: float | None = None
    failure: str | None = None

    if not _index_already_synced:
        try:
            if hasattr(gpx, "index_ids"):
                gpx.index_ids()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "set_param_limit: index_ids() failed ({}); limit names may stay unresolved", exc
            )

    verify = hasattr(gpx, "get_Controls")

    def _read() -> tuple[float | None, str | None]:
        if not verify:
            return None, "get_Controls unavailable"
        try:
            return gpx.get_Controls(control, variable=str(variable)), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    for attempt in (1, 2):
        _drop_limits(gpx, control, variable)
        try:
            gpx.set_Controls(control, value=float(value), variable=str(variable))
            readback, failure = _read()  # noqa: PU001
        except Exception as exc:  # noqa: BLE001
            readback, failure = None, f"{type(exc).__name__}: {exc}"
        applied = not verify or (
            readback is not None and abs(float(readback) - float(value)) < 1e-9
        )
        if applied or attempt == 2:
            break
        _rebucket_limits(_limits_dict(gpx, control))

    if not applied:
        detail = failure if failure else f"read back {readback!r}"
        message = f"Box constraint not applied: {control}({variable!r}) = {value!r} ({detail})"
        if required:
            raise RuntimeError(message)
        logger.warning(message)
    return applied


def _limits_dict(gpx: Any, control: str) -> dict[Any, Any]:
    """Return the live GSAS-II limits dict for ``parmMin``/``parmMax``.

    Returns an empty dict when the project exposes no ``Controls`` block (mocked
    projects, or a project in which no limit was ever set).
    """
    data = getattr(gpx, "data", None)
    if not isinstance(data, dict):
        return {}
    return data.get("Controls", {}).get("data", {}).get(control + "Dict", {}) or {}


def _limit_name_parts(variable: str) -> tuple[str, str | None]:
    """Split a GSAS-II variable mask into (parameter name, atom spec)."""
    parts = str(variable).split(":")
    if len(parts) >= 4:
        return parts[2], parts[3]
    if len(parts) == 3:
        return parts[2], None
    return str(variable), None


def _rebucket_limits(limits: dict[Any, Any]) -> int:
    """Re-insert every key of a GSAS-II limits dict so hash and bucket agree.

    ``G2VarObj.__hash__`` hashes ``varname()``, which resolves rIds through
    ``PhaseRanIdLookup`` / ``HistRanIdLookup``. A project loaded from disk builds
    its limits dict while those tables are still empty, so each key lands in the
    bucket of ``hash('?:?:MD')``. As soon as ``index_ids()`` fills the tables the
    very same object hashes as ``'0:0:MD'``: the dict becomes inconsistent —
    iteration still yields its entries, but ``pop``/``in``/``=`` raise ``KeyError``
    or silently insert duplicates, and ``prmLookup`` (which iterates and compares
    ``str(key)``) keeps returning the stale entry. Observed consequence: a declared
    ``MD = 0.7`` read back as ``0.4``, and ``parmMinDict.pop(key)`` raising
    ``KeyError`` for a key taken from ``parmMinDict.keys()``.

    Re-inserting the identical key objects rebuilds the buckets for their current
    hashes without changing key identity.

    :returns: number of entries carried over.
    """
    if not limits:
        return 0
    entries = list(limits.items())
    limits.clear()
    for key, val in entries:
        limits[key] = val
    return len(entries)


def _drop_limits(gpx: Any, control: str, variable: str) -> int:
    """Remove stored limits that would shadow a freshly written limit.

    Must be called after ``gpx.index_ids()`` so that the hashes of the stored keys
    have reached their final values and the buckets can be repaired.

    :returns: number of entries removed.
    """
    target_name, target_atom = _limit_name_parts(variable)  # noqa: PU001
    limits = _limits_dict(gpx, control)
    _rebucket_limits(limits)
    removed = 0
    for key in list(limits.keys()):
        if str(getattr(key, "name", "")) != target_name:
            continue
        atom = getattr(key, "atom", None)
        if target_atom is None or atom is None or str(atom) == str(target_atom):
            del limits[key]
            removed += 1
    return removed


def apply_box_constraints(gpx: Any, phase_id: int = 0, hist_id: int = 0) -> None:
    """Apply native GSAS-II box constraints (parmMin/parmMax) via ``set_Controls``.

    Prevents unphysical divergence during Levenberg-Marquardt refinement:
    - Atom thermal displacement: Uiso in [0.001, 0.15] A^2
    - March-Dollase preferred orientation: MD in [0.4, 2.5]
    - Sample displacement: DisplaceX in [-1000.0, 1000.0] um
    - Lattice metric-tensor terms A0..A2 > 0
    - Zero shift: Zero in [-0.5, 0.5] deg
    - Axial divergence: SH/L in [0.0, 0.05]
    - Crystallite size: Size in [0.001, 10.0] um
    - Microstrain: Mustrain in [0.0, 50000.0]

    Limits are registered under ``Controls['data']['parmMinDict' | 'parmMaxDict']``
    and are read back through :func:`set_param_limit`; a limit that does not
    register raises (``required=True`` for the mandatory ones).
    """
    if not hasattr(gpx, "set_Controls"):
        return

    # index_ids() must run BEFORE any limit is written: it fills the rId lookup
    # tables that G2VarObj.varname() (and therefore __hash__) depends on. Calling
    # it after some limits are already stored changes their hash and produces
    # shadowing duplicates in parmMinDict/parmMaxDict.
    if hasattr(gpx, "index_ids"):
        gpx.index_ids()

    phases = gpx.phases() if hasattr(gpx, "phases") else []
    if phase_id < len(phases):
        target_p = phases[phase_id]
        atom_count = (
            len(list(target_p.atoms()))
            if hasattr(target_p, "atoms")
            else len(target_p.data.get("Atoms", []))
        )
        for at_idx in range(atom_count):
            set_param_limit(
                gpx, "parmMin", f"{phase_id}::AUiso:{at_idx}", BOUND_UISO_MIN, _index_already_synced=True
            )
            set_param_limit(
                gpx, "parmMax", f"{phase_id}::AUiso:{at_idx}", BOUND_UISO_MAX, _index_already_synced=True
            )
            set_param_limit(
                gpx, "parmMin", f"{phase_id}::Afrac:{at_idx}", 0.0, _index_already_synced=True
            )
            set_param_limit(
                gpx, "parmMax", f"{phase_id}::Afrac:{at_idx}", 1.0, _index_already_synced=True
            )

    set_param_limit(
        gpx, "parmMin", f"{phase_id}:{hist_id}:MD", BOUND_MD_MIN, _index_already_synced=True
    )
    set_param_limit(
        gpx, "parmMax", f"{phase_id}:{hist_id}:MD", BOUND_MD_MAX, _index_already_synced=True
    )

    for ax_name in ["A0", "A1", "A2"]:
        set_param_limit(gpx, "parmMin", f"{phase_id}::{ax_name}", 0.0001, _index_already_synced=True)
        set_param_limit(gpx, "parmMax", f"{phase_id}::{ax_name}", 1.0, _index_already_synced=True)

    set_param_limit(
        gpx, "parmMin", f"{phase_id}:{hist_id}:Size;i", BOUND_SIZE_MIN, required=True, _index_already_synced=True
    )
    set_param_limit(
        gpx, "parmMax", f"{phase_id}:{hist_id}:Size;i", BOUND_SIZE_MAX, required=True, _index_already_synced=True
    )
    set_param_limit(
        gpx, "parmMin", f"{phase_id}:{hist_id}:Mustrain;i", BOUND_MUSTRAIN_MIN, required=True, _index_already_synced=True
    )
    set_param_limit(
        gpx, "parmMax", f"{phase_id}:{hist_id}:Mustrain;i", BOUND_MUSTRAIN_MAX, required=True, _index_already_synced=True
    )

    if phase_id == 0:
        set_param_limit(
            gpx, "parmMin", f":{hist_id}:DisplaceX", BOUND_DISPLACEX_MIN, _index_already_synced=True
        )
        set_param_limit(
            gpx, "parmMax", f":{hist_id}:DisplaceX", BOUND_DISPLACEX_MAX, _index_already_synced=True
        )
        set_param_limit(
            gpx, "parmMin", f":{hist_id}:Zero", BOUND_ZERO_MIN, _index_already_synced=True
        )
        set_param_limit(
            gpx, "parmMax", f":{hist_id}:Zero", BOUND_ZERO_MAX, _index_already_synced=True
        )
        set_param_limit(
            gpx, "parmMin", f":{hist_id}:SH/L", BOUND_SHL_MIN, required=True, _index_already_synced=True
        )
        set_param_limit(
            gpx, "parmMax", f":{hist_id}:SH/L", BOUND_SHL_MAX, required=True, _index_already_synced=True
        )
        for name, low, high in (
            ("U", BOUND_CAGLIOTI_UW_MIN, BOUND_CAGLIOTI_UW_MAX),
            ("V", BOUND_CAGLIOTI_V_MIN, BOUND_CAGLIOTI_V_MAX),
            ("W", BOUND_CAGLIOTI_UW_MIN, BOUND_CAGLIOTI_UW_MAX),
        ):
            set_param_limit(gpx, "parmMin", f":{hist_id}:{name}", low, _index_already_synced=True)
            set_param_limit(gpx, "parmMax", f":{hist_id}:{name}", high, _index_already_synced=True)

    # Значение, уже лежащее за границей, граница не исправляет: она ограничивает
    # следующий шаг, а не возвращает назад. Хранимое приводится здесь.
    histograms = gpx.histograms() if hasattr(gpx, "histograms") else []
    if phase_id < len(phases) and hist_id < len(histograms):
        clamp_stored_parameters(gpx, phases[phase_id], histograms[hist_id])

    # Fail fast: on a real project the limits must have landed in the GSAS-II
    # Controls data tree. An object that does not expose that tree (a duck-typed
    # or mocked project) can only be trusted as-is.
    data = getattr(gpx, "data", None)
    exposes_controls = isinstance(data, dict) and "Controls" in data
    if exposes_controls and not (_limits_dict(gpx, "parmMin") and _limits_dict(gpx, "parmMax")):
        raise RuntimeError(
            "apply_box_constraints registered no limits: GSAS-II Controls['data'] "
            "parmMinDict/parmMaxDict are empty"
        )


def _specimen_displacement_notes(inst: Any, data: Any) -> list[str]:
    """State a refined specimen displacement in the angle it actually shifts.

    ``DisplaceX`` is a length in micrometres and its effect is an angular shift, so the
    two are comparable only through the goniometer radius. A displacement refinement and
    a Zero refinement both move every reflection in the same direction, and telling them
    apart means comparing one shift in degrees with the other.

    A project that refines no displacement keeps the report empty, which is what the
    caller of :func:`instrumental_lower_bound_report` expects of a canonical project.
    """
    # The sample block is one mapping, not a list of them: this mirrors the accessor of
    # the parameter reader, which reads hist.data["Sample Parameters"]["DisplaceX"].
    sample_raw = data.get("Sample Parameters", {}) if isinstance(data, dict) else {}
    if not isinstance(sample_raw, dict):
        return []
    record = sample_raw.get("DisplaceX")
    if not isinstance(record, (list, tuple)) or not record:
        return []
    try:
        displacement_um = float(record[0])
        refine_flag = bool(record[1]) if len(record) > 1 else False
    except (TypeError, ValueError):
        return []
    if displacement_um == 0.0 and not refine_flag:
        return []

    reference_angle = 20.0
    shift_deg = inst.displacement_angular_shift_deg(displacement_um / 1000.0, reference_angle)
    return [
        (
            f"Specimen displacement is {displacement_um:.1f} um, which shifts a reflection "
            f"at {reference_angle:.1f} deg by {shift_deg:.4f} deg on a "
            f"{inst.goniometer_radius_mm:.1f} mm goniometer. A displacement refinement can "
            "only be compared with a Zero refinement through this shift."
        )
    ]


def instrumental_lower_bound_report(hist: Any) -> list[str]:
    """Report an instrument profile sharpened below the hardware floor.

    The FWHM column of a GSAS-II reflection list carries the *instrumental
    Gaussian variance* -- ``U tan^2(theta) + V tan(theta) + W`` in deg^2 -- not
    the reflection's width. For the canonical instrument the column holds
    0.005733 deg^2 at the strongest reflection, whose calculated profile is
    0.18 deg wide, so comparing that column against a canonical FWHM is a
    dimensional error that fires on every project.

    What the check is for: an instrument injected through ``--override-json`` or
    an ``.instprm`` file can be sharper than the canonical optics, and the fit
    then sharpens reflections with the instrument instead of with the sample. A
    sample can only add broadening, so the Gaussian part of the refined
    instrument must not fall below the canonical one. With the stock canonical
    coefficients the report is empty by construction.
    """
    # Imported here because instrument is a sibling module and this function is
    # called from two different entry points.
    from instrument import CanonicalInstrument

    inst = CanonicalInstrument()
    data = getattr(hist, "data", None)
    inst_raw = data.get("Instrument Parameters", [{}])[0] if isinstance(data, dict) else {}
    if not isinstance(inst_raw, dict):
        return []

    def _coefficient(name: str, default: float) -> float:
        rec = inst_raw.get(name)
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            try:
                return float(rec[1])
            except (TypeError, ValueError):
                return float(default)
        return float(default)

    u = _coefficient("U", inst.u_inst)
    v = _coefficient("V", inst.v_inst)
    w = _coefficient("W", inst.w_inst)

    two_theta = np.linspace(20.0, 60.0, 81)
    canonical = inst.fwhm_inst_deg(two_theta)
    theta = np.radians(two_theta / 2.0)
    refined = np.sqrt(np.maximum(u * np.tan(theta) ** 2 + v * np.tan(theta) + w, 0.0))

    sharper = refined < canonical - 1e-3
    if not bool(np.any(sharper)):
        return _specimen_displacement_notes(inst, data)

    return [
        (
            "Instrument profile is narrower than the canonical Bragg-Brentano floor over "
            f"{int(np.sum(sharper))} of {two_theta.size} sampled angles (U={u:.5f}, "
            f"V={v:.5f}, W={w:.5f}; worst shortfall "
            f"{float(np.max(canonical - refined)):.4f} deg). The fit is sharpening "
            "reflections with the instrument instead of with the sample."
        )
    ]


def setup_phase_dist_restraints(
    gpx: Any,
    phase: Any,
    structure: Structure | None = None,
    factor: float = 1.15,
    esd: float = 0.02,
    weight: float = 100.0,
) -> int:
    """Universal Dynamic Distance Restraints Builder without chemical hardcoding.

    Extracts all 1st coordination sphere interatomic pairs from the structure / CIF
    (d_0 < 1.25 * (r_i + r_j)), initializes the Restraints data block, clears
    existing restraints on the phase, adds native restraints via phase.addDistRestraint,
    and sets restraint weight.
    Returns the count of added restraints.
    """
    p_name = getattr(phase, "name", "Phase")
    # 1. Ensure the Restraints data block exists in the shape GSAS-II expects.
    # GSASIIstrIO.SummRestraints / SetPhaseData read every present restraint section
    # as itemRest['Use'], itemRest['wtFactor'] and itemRest[rest]; a section written
    # without 'Use' makes the next gpx.refine() fail with KeyError: 'Use'. The key set
    # mirrors the GSAS-II constructors (GSASIIphsGUI: {'wtFactor':1.0,'Bonds':[],
    # 'Use':True}). An existing section is preserved rather than overwritten.
    if hasattr(gpx, "data") and isinstance(gpx.data, dict):
        rest_data = gpx.data.setdefault("Restraints", {}).setdefault("data", {})
        bond_block = rest_data.setdefault(p_name, {}).setdefault(
            "Bond", {"wtFactor": weight, "Bonds": [], "Use": True}
        )
        # Repair blocks written by earlier versions of this helper.
        bond_block.setdefault("Bonds", [])
        bond_block["Use"] = True
        bond_block["wtFactor"] = weight

    # 2. Clear existing distance restraints
    if hasattr(phase, "clearDistRestraint"):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            phase.clearDistRestraint()

    count = 0
    # 3. Get phase atoms (unique sites in asymmetric unit)
    phase_atoms = []
    if hasattr(phase, "atoms") and callable(phase.atoms):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            phase_atoms = list(phase.atoms())
    elif hasattr(phase, "data") and isinstance(phase.data, dict):
        raw_atoms = phase.data.get("Atoms", [])
        phase_atoms = [G2AtomRecord(at) for at in raw_atoms]

    if not phase_atoms or not hasattr(phase, "addDistRestraint"):
        return 0

    seen_pairs: set[tuple[str, str]] = set()
    n_atoms = len(phase_atoms)

    for i in range(n_atoms):
        at_i = phase_atoms[i]
        rec_i = G2AtomRecord(at_i)
        elem_i = rec_i.element
        lbl_i = rec_i.label or f"Site_{i}"
        if not elem_i:
            continue
        r_i = gemmi.Element(elem_i).covalent_r or 1.0

        for j in range(i + 1, n_atoms):
            at_j = phase_atoms[j]
            rec_j = G2AtomRecord(at_j)
            elem_j = rec_j.element
            lbl_j = rec_j.label or f"Site_{j}"
            if not elem_j or elem_i == elem_j:
                continue

            pair_key = tuple(sorted([lbl_i, lbl_j]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            r_j = gemmi.Element(elem_j).covalent_r or 1.0
            limit = 1.25 * (r_i + r_j)

            # Find minimum distance between this specific pair of sites in reference structure
            min_d = None
            if structure is not None and len(structure) > 0:
                sites_i = [idx for idx, s in enumerate(structure) if s.specie.symbol == elem_i]
                sites_j = [idx for idx, s in enumerate(structure) if s.specie.symbol == elem_j]

                # Match by fractional coordinates if available on both atom records
                if rec_i.has_coordinates and rec_j.has_coordinates and sites_i and sites_j:
                    xyz_i = rec_i.coordinates
                    xyz_j = rec_j.coordinates
                    matched_i = None
                    matched_j = None
                    best_diff_i = 1e9
                    for idx in sites_i:
                        diff = float(
                            np.linalg.norm(
                                np.mod(structure[idx].frac_coords - xyz_i + 0.5, 1.0) - 0.5
                            )
                        )
                        if diff < best_diff_i:
                            best_diff_i = diff
                            matched_i = idx
                    best_diff_j = 1e9
                    for idx in sites_j:
                        diff = float(
                            np.linalg.norm(
                                np.mod(structure[idx].frac_coords - xyz_j + 0.5, 1.0) - 0.5
                            )
                        )
                        if diff < best_diff_j:
                            best_diff_j = diff
                            matched_j = idx

                    if matched_i is not None and matched_j is not None:
                        d = float(structure.get_distance(matched_i, matched_j))
                        if 0.5 < d <= limit:
                            min_d = d

                # Fallback to minimum distance among sites of these elements
                if min_d is None:
                    for idx1 in sites_i:
                        for idx2 in sites_j:
                            d = float(structure.get_distance(idx1, idx2))
                            if 0.5 < d and (min_d is None or d < min_d):
                                min_d = d
            else:
                min_d = r_i + r_j

            if min_d is not None and 0.8 <= min_d <= limit:
                with suppress(
                    RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError
                ):
                    res = phase.addDistRestraint(
                        [at_i],
                        [at_j],
                        bond=round(min_d, 4),
                        factor=factor,
                        ESD=esd,
                    )
                    count += res if isinstance(res, int) else 1

    if hasattr(phase, "setDistRestraintWeight"):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            phase.setDistRestraintWeight(weight)

    return count


#: Флаги, которые стадия снимает у всех фаз и гистограммы перед тем, как освободить
#: свои. ``{'set': ...}`` в GSAS-II кумулятивен: поднятый флаг остаётся поднятым во
#: всех следующих стадиях, и протокол «сначала масштабы, потом ячейка, потом профиль»
#: вырождается в одновременное снятие всех параметров — ровно то, что он должен
#: предотвращать. Измерено на реальном образце: после серии 1,3,5,7 одновременно были
#: свободны ячейка, размер, микронапряжение и масштабы у всех трёх фаз, движок увёл
#: A0 в отрицательные значения и отказал с ``Invalid cell metric tensor``.
_STAGE_CLEAR_PHASE_KEYS: dict[str, Any] = {"Cell": False, "Atoms": ""}
_STAGE_CLEAR_HAP_KEYS: dict[str, Any] = {
    "Scale": False,
    "Size": False,
    "Mustrain": False,
    "Pref.Ori.": False,
    "Extinction": False,
}
_STAGE_CLEAR_HIST_KEYS: dict[str, Any] = {
    "Background": False,
    "Instrument Parameters": ["Zero", "SH/L", "U", "V", "W"],
    "Sample Parameters": ["DisplaceX", "Absorption"],
}


def reset_stage_flags(gpx: Any, hist: Any) -> list[str]:
    """Снять флаги уточнения всех фаз и гистограммы перед новой стадией.

    Стадия обязана освобождать только свои параметры. Без этого сброса каждая
    следующая стадия работает на модели, где уже свободно всё предыдущее, и
    Stage-Gate превращается в одновременное снятие всех параметров.

    :returns: имена фаз, у которых флаги сняты — чтобы вызывающий мог это записать.
    """
    cleared: list[str] = []
    for phase in gpx.phases():
        phase.clear_refinements(dict(_STAGE_CLEAR_PHASE_KEYS))
        phase.clear_HAP_refinements(dict(_STAGE_CLEAR_HAP_KEYS), [hist])
        cleared.append(str(phase.name))
    hist.clear_refinements(dict(_STAGE_CLEAR_HIST_KEYS))
    return cleared


#: Физические границы величин, которые GSAS-II хранит в HAP. Значение за границей —
#: не «подобранная величина», а наследие прежнего прогона, и оно не должно переезжать
#: в следующую стадию.
_STORED_VALUE_BOUNDS: dict[str, tuple[float, float]] = {
    "Size": (BOUND_SIZE_MIN, BOUND_SIZE_MAX),
    "Mustrain": (BOUND_MUSTRAIN_MIN, BOUND_MUSTRAIN_MAX),
    "Extinction": (BOUND_EXTINCTION_MIN, BOUND_EXTINCTION_MAX),
}


def clamp_stored_parameters(gpx: Any, phase: Any, hist: Any) -> list[str]:
    """Привести хранимые в HAP значения к объявленным границам и назвать исправленное.

    Граница ``parmMin/parmMax`` ограничивает следующий шаг оптимизатора, но не
    исправляет значение, которое уже лежит в проекте: измерено — у фазы хранилось
    ``Mustrain = -1124.9`` при ``refine = True``, и это значение переезжало из стадии
    в стадию, потому что граница «не даёт уйти дальше», а не «возвращает назад».
    Отрицательное микронапряжение физически невозможно, и приводить его как
    подобранное нельзя.

    :returns: список исправлений вида ``"Mustrain: -1124.9 -> 0.0"``.
    """
    changed: list[str] = []
    histograms = getattr(phase, "data", {}).get("Histograms", {})
    hist_name = getattr(hist, "name", "")
    hap = histograms.get(hist_name) if isinstance(histograms, dict) else None
    if not isinstance(hap, dict):
        return changed

    for key, (low, high) in _STORED_VALUE_BOUNDS.items():
        record = hap.get(key)
        if not isinstance(record, (list, tuple)) or len(record) < 2:
            continue
        # Две формы записи в HAP: [type, [values], [refine]] (Size, Mustrain) и
        # [value, refine] (Extinction). Их нельзя читать одним способом: у второй
        # record[1] — булев флаг, а не список значений.
        values = record[1] if isinstance(record[1], (list, tuple)) else record
        if not isinstance(values, (list, tuple)) or not values:
            continue
        try:
            current = float(values[0])
        except (TypeError, ValueError):
            continue
        clamped = min(max(current, low), high)
        if abs(clamped - current) <= 1e-12:
            continue
        values[0] = clamped
        changed.append(f"{key}: {current:g} -> {clamped:g}")
        logger.warning(
            "Phase {}: stored {} = {:g} is outside [{:g}, {:g}]; clamped to {:g} before refinement",
            getattr(phase, "name", "?"),
            key,
            current,
            low,
            high,
            clamped,
        )
    return changed


@dataclass(frozen=True)
class RefinementConfig:
    """Universal Rietveld refinement configuration passed from external callers.

    Encapsulates experimental and optical adaptations without polluting
    the core crystallographic engine with chemical-specific assumptions.
    """

    bkg_chebyshev_terms: int = 6  # Conservative default for clean patterns
    refine_sample_displacement: bool = (
        False  # Refine DisplaceX (sample displacement) instead of Zero
    )
    refine_axial_divergence: bool = False  # Refine instrumental SH/L (axial divergence asymmetry)
    texture_axes: dict[str, list[int]] = field(default_factory=dict)
    # Mapping phase identifier/formula -> Miller indices [h, k, l] for March-Dollase
    march_dollase_bounds: tuple[float, float] = (0.7, 1.3)


class StageId(StrEnum):
    """Canonical 12-stage Rietveld refinement stages."""

    SCALE_AND_BACKGROUND = "1_Scale_and_Background"
    ZERO_SHIFT = "2_Zero_Shift"
    UNIT_CELL = "3_Unit_Cell"
    ATOMIC_COORDINATES = "4_Atomic_Coordinates"
    OVERALL_DISPLACEMENT = "5_Overall_Displacement"
    GROUP_B_ISO = "6_Group_B_iso"
    SAMPLE_PROFILE_XY = "7_Sample_Profile_XY"
    PREFERRED_ORIENTATION = "8_Preferred_Orientation"
    OCCUPANCY = "9_Occupancy"
    INSTRUMENT_PROFILE = "10_Instrument_Profile"
    ABSORPTION_EXTINCTION = "11_Absorption_Extinction"
    FINAL_SCALE = "12_Final_Scale"


ADVANCED_STAGES: frozenset[StageId] = frozenset(
    {
        StageId.ATOMIC_COORDINATES,
        StageId.OVERALL_DISPLACEMENT,
        StageId.GROUP_B_ISO,
        StageId.SAMPLE_PROFILE_XY,
        StageId.PREFERRED_ORIENTATION,
        StageId.OCCUPANCY,
        StageId.INSTRUMENT_PROFILE,
        StageId.ABSORPTION_EXTINCTION,
        StageId.FINAL_SCALE,
    }
)

CANONICAL_STAGES: list[str] = [stage.value for stage in StageId]

STAGE_ALIASES: dict[str, StageId] = {
    "1": StageId.SCALE_AND_BACKGROUND,
    "2": StageId.ZERO_SHIFT,
    "3": StageId.UNIT_CELL,
    "4": StageId.ATOMIC_COORDINATES,
    "5": StageId.OVERALL_DISPLACEMENT,
    "6": StageId.GROUP_B_ISO,
    "7": StageId.SAMPLE_PROFILE_XY,
    "8": StageId.PREFERRED_ORIENTATION,
    "9": StageId.OCCUPANCY,
    "10": StageId.INSTRUMENT_PROFILE,
    "11": StageId.ABSORPTION_EXTINCTION,
    "12": StageId.FINAL_SCALE,
}


class PassDecision(StrEnum):
    """Stopping decision of a multi-pass stage sequence."""

    CONTINUE = "continue"
    PLATEAU = "plateau"
    REGRESSED = "regressed"
    EXHAUSTED = "exhausted"
    SINGLE_PASS = "single_pass"


class RefinementStatus(StrEnum):
    """Overall status of a refinement or export operation."""

    CONVERGED = "CONVERGED"
    NEEDS_REFINEMENT = "NEEDS_REFINEMENT"
    REJECTED_BY_GATEKEEPER = "REJECTED_BY_GATEKEEPER"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REVERTED = "REVERTED"
    SKIPPED = "SKIPPED"
    PASS = "PASS"
    FAIL = "FAIL"


PASS_CONTINUE = PassDecision.CONTINUE
PASS_PLATEAU = PassDecision.PLATEAU
PASS_REGRESSED = PassDecision.REGRESSED
PASS_EXHAUSTED = PassDecision.EXHAUSTED


def decide_next_pass(r_wp_history: list[float], plateau_delta: float) -> tuple[PassDecision, int]:
    """Решить, нужен ли ещё один проход, и какое состояние считать лучшим.

    Критерий — **улучшение**, а не модуль разности: проход, ухудшивший модель на
    0.05 %, не является плато, это шаг назад. Измерено трижды за одну сессию:
    финальная стадия масштабов давала 8.17 → 9.24 %, а «плато по модулю» приняло бы
    это за схождение.

    Возвращает пару (решение, индекс лучшего прохода):
    ``continue`` — улучшение не меньше порога; ``plateau`` — улучшение есть, но
    меньше порога; ``regressed`` — последний проход хуже предыдущего.
    ``exhausted`` возвращает вызывающий: предел проходов — это не сходимость,
    и формулировать его должно то место, которое знает про лимит.
    """
    if len(r_wp_history) < 2:
        return PASS_CONTINUE, len(r_wp_history) - 1
    best_index = min(range(len(r_wp_history)), key=lambda i: r_wp_history[i])
    improvement = r_wp_history[-2] - r_wp_history[-1]
    if improvement < 0.0:
        return PASS_REGRESSED, best_index
    if improvement < plateau_delta:
        return PASS_PLATEAU, best_index
    return PASS_CONTINUE, best_index


#: Декларативная таблица сопоставления ключевых подстрок с каноническими стадиями.
#: Порядок записей значим: более специфичные ключи («final», «instrument») идут раньше общих («scale», «profile»).
_STAGE_KEYWORD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("final", "scale_only"), "12_Final_Scale"),
    (("scale", "background"), "1_Scale_and_Background"),
    (("zero", "sample displacement"), "2_Zero_Shift"),
    (("cell", "unit"), "3_Unit_Cell"),
    (("coord", "atomic"), "4_Atomic_Coordinates"),
    (("group", "b_iso"), "6_Group_B_iso"),
    (("overall", "thermal"), "5_Overall_Displacement"),
    (("absor", "extinct"), "11_Absorption_Extinction"),
    (("instrument", "caglioti"), "10_Instrument_Profile"),
    (("profile", "mustrain", "size"), "7_Sample_Profile_XY"),
    (("texture", "preferred"), "8_Preferred_Orientation"),
    (("occup",), "9_Occupancy"),
)


def normalize_stage_name(stage: str) -> str:
    """Normalize numeric or alias stage name to canonical stage string."""
    s = stage.strip()
    if s in STAGE_ALIASES:
        return STAGE_ALIASES[s]
    s_low = s.lower()
    for canon in CANONICAL_STAGES:
        if s_low == canon.lower():
            return canon
    for keywords, target_stage in _STAGE_KEYWORD_RULES:
        if any(kw in s_low for kw in keywords):
            return target_stage
    return s


def build_stage_recipe(
    stage: str,
    bkg_terms: int = 6,
    refine_displacement: bool = True,
    refine_axial: bool = False,
    march_dollase_direction: list[int] | None = None,
) -> dict[str, Any]:
    """Build canonical GSAS-II declarative refinement recipe for a given stage.

    Supported stages:
        1_Scale_and_Background: Chebyshev polynomial background and phase scale factors.
        2_Zero_Shift: Sample displacement (DisplaceX) or instrument zero-shift (Zero).
        3_Unit_Cell: Unit cell lattice parameters (Cell).
        4_Atomic_Coordinates: Fractional atomic coordinates (X, Y, Z).
        5_Overall_Displacement: Overall isotropic thermal displacement (Uiso).
        6_Group_B_iso: Site/group thermal displacement factors (Uiso).
        7_Sample_Profile_XY: Microstrain (Mustrain) and crystallite size broadening (Size), optional SH/L.
        8_Preferred_Orientation: March-Dollase preferred orientation (Pref.Ori.).
        9_Occupancy: site occupancies (F); use when the amplitude pattern, not the
            overall scale, is wrong.
        10_Instrument_Profile: instrumental Caglioti terms (U, V, W); they describe
            the optics, so their lower bound is the canonical laboratory profile.
        11_Absorption_Extinction: sample absorption and phase extinction; both bias
            integrated intensities systematically, which is what R_B reads.
        12_Final_Scale: scale factors only, background locked. Ends a converged route
            so that the covariance of the last refinement contains the scales, which is
            where phase-fraction uncertainties come from.
    """
    canon = normalize_stage_name(stage)
    if canon == "1_Scale_and_Background":
        return {
            "Background": {
                "type": "chebyschev-1",
                "refine": True,
                "no. coeffs": int(bkg_terms),
            },
            "Scale": True,
        }
    if canon == "2_Zero_Shift":
        if refine_displacement:
            return {"Sample Parameters": ["DisplaceX"]}
        return {"Instrument Parameters": ["Zero"]}
    if canon == "3_Unit_Cell":
        return {"Cell": True}
    if canon == "4_Atomic_Coordinates":
        return {"Atoms": {"all": "X"}}
    if canon in ("5_Overall_Displacement", "6_Group_B_iso"):
        return {"Atoms": {"all": "U"}}
    if canon == "7_Sample_Profile_XY":
        rec: dict[str, Any] = {
            "Size": {"type": "isotropic", "refine": True},
            "Mustrain": {"type": "isotropic", "refine": True},
        }
        if refine_axial:
            rec["Instrument Parameters"] = ["SH/L"]
        return rec
    if canon == "8_Preferred_Orientation":
        rec: dict[str, Any] = {"Pref.Ori.": True}
        if march_dollase_direction is not None:
            rec["direction"] = list(march_dollase_direction)
        return rec
    if canon == "9_Occupancy":
        return {"Atoms": {"all": "F"}}
    if canon == "12_Final_Scale":
        # Финальная стадия существует ради одного: чтобы масштабные факторы попали
        # в ковариацию последнего уточнения, иначе у массовых долей нет esd. Фон
        # здесь НЕ освобождается: его совместное уточнение с масштабами поднимало
        # R_wp с 8.14 до 9.21 % на сошедшейся модели, то есть ухудшало результат.
        return {"Scale": True}
    if canon == "11_Absorption_Extinction":
        # Поглощение относится к образцу, экстинкция — к фазе: это разные модели
        # одного наблюдения (систематическое занижение интенсивностей), поэтому
        # освобождаются вместе, но в разных блоках.
        return {"Sample Parameters": ["Absorption"], "Extinction": True}
    if canon == "10_Instrument_Profile":
        rec: dict[str, Any] = {"Instrument Parameters": ["U", "V", "W"]}
        if refine_axial:
            rec["Instrument Parameters"] = ["U", "V", "W", "SH/L"]
        return rec

    raise ValueError(
        f"Unknown Rietveld refinement stage: '{stage}'. Valid stages: {CANONICAL_STAGES}"
    )


@dataclass(frozen=True)
class SuccessGateInputs:
    """The quantities the Stage-Gate success criteria are evaluated on."""

    r_wp_percent: float
    durbin_watson: float
    max_microstrain_percent: float
    max_crystallite_size_nm: float
    unindexed_count: int
    r_p_percent: float | None = None
    rb_by_phase: dict[str, float] | None = None
    rf_by_phase: dict[str, float] | None = None
    esd_problems: list[str] | None = None
    phase_weights_percent: dict[str, float] | None = None
    rule9_problems: list[str] | None = None
    rule9_measured: bool | None = None
    #: Причина, по которой esd массовых долей не существует. Отсутствие измерения
    #: называется, а не выдаётся за «нулевую погрешность».
    qpa_esd_reason: str | None = None


def shift_significance_problems(
    shifts: dict[str, float], sigmas: dict[str, float], threshold: float
) -> list[str]:
    """Параметры, у которых сдвиг не подтверждён данными.

    Критерий Rule 9: ``|Δ|/σ < threshold``. Параметр с нулевой, отрицательной или
    нечисловой σ не проверен вовсе, поэтому он попадает в список с отдельной
    пометкой, а не считается сошедшимся.
    """
    problems: list[str] = []
    for name in sorted(shifts):
        delta = float(shifts[name])
        sigma = float(sigmas.get(name, 0.0))
        if not math.isfinite(sigma) or sigma <= 0.0:
            problems.append(f"{name}: |Δ|/σ не определено (σ={sigma})")
        elif abs(delta) / sigma >= threshold:
            problems.append(f"{name}: |Δ|/σ = {abs(delta) / sigma:.2f} ≥ {threshold}")
    return problems


def undefined_esd_problems(sigmas: dict[str, float]) -> list[str]:
    """Возвращает параметры, у которых esd не определяет значение.

    Ноль, nan и бесконечность означают не «идеальную точность», а отсутствие
    оценки: параметр не определён данными, и это должно попадать в вердикт.
    """
    problems: list[str] = []
    for name, sigma in sorted(sigmas.items()):
        if not math.isfinite(sigma) or sigma <= 0.0:
            problems.append(f"{name}: esd не определена (σ={sigma})")
    return problems


def evaluate_success_gates(inputs: SuccessGateInputs) -> list[str]:
    """Return the unmet Stage-Gate success criteria, each one named.

    A single ``R_wp`` threshold is not a convergence criterion: a model at
    ``R_wp = 12 %`` with 1 % microstrain and serial residuals is not converged,
    and certifying it as such hides the defect the gates exist to catch. Naming
    each unmet criterion makes the verdict state what is wrong rather than only
    that something is.
    """
    unmet: list[str] = []

    # Абсолютное качество подгонки: отношение R_wp/R_exp его не заменяет.
    # Прежняя дизъюнкция пропускала R_wp = 40 % при отношении 1.0 — невязка
    # огромна, но «согласована с весами», и вердикт был положительным.
    if inputs.r_wp_percent > 10.0:
        unmet.append(f"R_wp = {inputs.r_wp_percent}% > 10%")

    # χ² в критерии не входит: в этой серии разброс весов между проектами
    # (χ² от 0.2 до 37.7) делает его индикатором весов, а не качества модели.
    # Печатается в отчёте и сводке экспорта.
    if not (1.5 <= inputs.durbin_watson <= 2.5):
        unmet.append(f"Durbin-Watson = {inputs.durbin_watson} outside [1.5, 2.5]")
    if inputs.max_microstrain_percent >= 0.8:
        unmet.append(f"microstrain = {inputs.max_microstrain_percent}% >= 0.8%")
    if inputs.max_crystallite_size_nm >= 900.0:
        unmet.append(
            f"crystallite size = {inputs.max_crystallite_size_nm} nm >= 900 nm "
            "(profile at its ceiling)"
        )
    if inputs.unindexed_count:
        unmet.append(f"{inputs.unindexed_count} unindexed reflection(s) above the noise floor")

    # Профильный критерий: R_p реагирует на отдельные сильные точки иначе, чем R_wp.
    if inputs.r_p_percent is not None and inputs.r_p_percent > 10.0:
        unmet.append(f"R_p = {inputs.r_p_percent}% > 10%")

    # Структурный критерий по фазам (SKILL.md): доминирующая фаза R_B <= 10 %,
    # R_F <= 6 %; субдоминантные (> 5 масс. %) R_B <= 15 %, R_F <= 8 %; фазы с
    # долей <= 5 масс. % печатаются информационно и в критерий не входят.
    weights = inputs.phase_weights_percent or {}
    rb_map = inputs.rb_by_phase or {}
    rf_map = inputs.rf_by_phase or {}
    dominant = max(weights, key=lambda name: weights[name]) if weights else None
    major_rb: list[float] = []
    for phase in sorted(set(rb_map) | set(rf_map)):
        weight = weights.get(phase)
        if weight is not None and weight < PHASE_MINOR_WEIGHT_PERCENT:
            continue
        is_dominant = phase == dominant
        rb_limit = PHASE_RB_DOMINANT_PERCENT if is_dominant else PHASE_RB_SUBDOMINANT_PERCENT
        rf_limit = PHASE_RF_DOMINANT_PERCENT if is_dominant else PHASE_RF_SUBDOMINANT_PERCENT
        rb_value = rb_map.get(phase)
        if rb_value is not None:
            major_rb.append(rb_value)
            if rb_value > rb_limit:
                unmet.append(
                    f"R_B({phase}) = {rb_value}% > {rb_limit}% — структура фазы не подтверждена"
                )
        rf_value = rf_map.get(phase)
        if rf_value is not None and rf_value > rf_limit:
            unmet.append(f"R_F({phase}) = {rf_value}% > {rf_limit}%")

    # Определённость параметров: без конечной положительной esd параметр не
    # измерен, и никакие R-факторы этого не компенсируют.
    if inputs.esd_problems:
        unmet.extend(inputs.esd_problems)

    # Отсутствие данных Rule 9 — не «всё сошлось»: пока стадия не прогнана после
    # введения критерия, сдвиги параметров не проверены, и это надо назвать,
    # иначе вердикт опирается на отсутствие измерения.
    if inputs.qpa_esd_reason:
        unmet.append(inputs.qpa_esd_reason)

    if inputs.rule9_measured is False:
        unmet.append("Rule 9 не измерялась: в истории нет сдвигов и esd по параметрам")

    # Rule 9 как критерий, а не только отчёт: сдвиг на величину порядка esd
    # означает, что параметр ещё не сошёлся, и приёмка это учитывает.
    if inputs.rule9_problems:
        shown = inputs.rule9_problems[:5]
        unmet.extend(f"сдвиг не подтверждён: {item}" for item in shown)
        if len(inputs.rule9_problems) > len(shown):
            unmet.append(
                f"сдвиг не подтверждён ещё для {len(inputs.rule9_problems) - len(shown)} параметр(ов)"
            )

    # Различение причин: высокий R_wp при низком R_B — ограничение профиля.
    if inputs.r_wp_percent > 10.0 and major_rb and max(major_rb) <= 5.0:
        unmet.append(
            "R_wp высокий при R_B <= 5% по всем фазам: ограничение профиля "
            "(форма, текстура, фон), а не структуры"
        )

    return unmet


def ensure_gsas2_environment() -> bool:
    """Discover GSAS-II installation, configure DLL directories for Windows,
    and ensure native Fortran/C extensions (pyspg, pypowder) load cleanly."""
    candidates = [
        os.environ.get("GSASII_PATH"),
        Path.home() / ".GSASII" / "GSAS-II",
        Path.home() / "GSAS-II",
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            c_str = str(Path(cand).resolve())
            if c_str not in sys.path:
                sys.path.insert(0, c_str)

    # Windows DLL resolution for MinGW runtimes
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        git_bin = shutil.which("git")
        if git_bin:
            git_mingw = Path(git_bin).resolve().parent.parent / "mingw64" / "bin"
            if git_mingw.exists():
                try:
                    os.add_dll_directory(str(git_mingw))
                except OSError:
                    pass

        for cand in candidates:
            if cand and Path(cand).exists():
                bindist = Path(cand) / "GSASII" / "bindist"
                if bindist.exists():
                    try:
                        os.add_dll_directory(str(bindist.resolve()))
                    except OSError:
                        pass

    try:
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            from GSASII import GSASIIpath, GSASIIstrIO, GSASIIstrMath

            math_file = Path(GSASIIstrMath.__file__)
            with suppress(OSError):
                src_text = math_file.read_text(encoding="utf-8", errors="ignore")
                bad_block = (
                    "for phasej in Phases:\n"
                    "        if Phases[phasej]['General']['Type'] == 'magnetic': continue\n"
                    "        if Phases[phase]['General']['doPawley']: continue"
                )
                fixed_block = (
                    "for phasej in Phases:\n"
                    "        if Phases[phasej]['General']['Type'] == 'magnetic': continue\n"
                    "        if Phases[phasej]['General']['doPawley']: continue"
                )
                if bad_block in src_text:
                    math_file.write_text(
                        src_text.replace(bad_block, fixed_block), encoding="utf-8"
                    )

            strio_file = Path(GSASIIstrIO.__file__)
            with suppress(OSError):
                strio_text = strio_file.read_text(encoding="utf-8", errors="ignore")
                bad_lim = (
                    "if Phases[phase]['General']['doPawley']:\n"
                    "                hapDict[pfx+'LeBail'] = False\n"
                    "                Tmin = G2lat.Dsp2pos(inst,dmin)\n"
                    "                if 'T' in inst['Type'][1]:\n"
                    "                    limits[0] = max(limits[0],Tmin)\n"
                    "                else:\n"
                    "                    limits[1] = min(limits[1],Tmin)"
                )
                fixed_lim = (
                    "if Phases[phase]['General']['doPawley']:\n"
                    "                hapDict[pfx+'LeBail'] = False"
                )
                if bad_lim in strio_text:
                    strio_file.write_text(
                        strio_text.replace(bad_lim, fixed_lim), encoding="utf-8"
                    )

            if not getattr(GSASIIstrIO, "_limits_guard_installed", False):
                orig_get_hpd = GSASIIstrIO.GetHistogramPhaseData

                def _safe_get_hpd(phases_arg: Any, histograms_arg: Any, *a: Any, **kw: Any) -> Any:
                    true_maxima: dict[str, float] = {}
                    if isinstance(histograms_arg, dict):
                        for h_name, h_data in histograms_arg.items():
                            if isinstance(h_data, dict) and "Limits" in h_data:
                                lim = h_data["Limits"][1]
                                if isinstance(lim, (list, tuple)) and len(lim) >= 2:
                                    true_maxima[h_name] = float(lim[1])
                    res = orig_get_hpd(phases_arg, histograms_arg, *a, **kw)
                    if isinstance(histograms_arg, dict):
                        for h_name, orig_max in true_maxima.items():
                            if h_name in histograms_arg and "Limits" in histograms_arg[h_name]:
                                lim = histograms_arg[h_name]["Limits"][1]
                                if isinstance(lim, list) and len(lim) >= 2:
                                    lim[1] = orig_max
                    return res

                GSASIIstrIO.GetHistogramPhaseData = _safe_get_hpd
                GSASIIstrIO._limits_guard_installed = True

            GSASIIpath.SetBinaryPath()
            if not GSASIIpath.BinaryPathFailed:
                from GSASII import GSASIIscriptable as G2sc

                if hasattr(G2sc, "G2SetPrintLevel"):
                    G2sc.G2SetPrintLevel("warn")
                return True
    except (ImportError, OSError):
        pass

    logger.warning(
        "GSAS-II calculation engine is not found or binaries failed to load. "
        "Run `py .agents/skills/xrd-analysis/scripts/install_gsas2.py` to set up GSAS-II."
    )
    return False


class StageGateRietveld:
    """Controls step-by-step Rietveld refinement according to the verified 8-stage protocol.

    Acts as an intelligent supervisor orchestrating the native GSAS-II engine (GSASIIscriptable),
    enforcing physical box constraints, Hessian condition limits, and parameter correlation guards.
    """

    STAGES: ClassVar[list[str]] = [
        "1_Scale_and_Background",
        "2_Zero_Shift",
        "3_Unit_Cell",
        "4_Atomic_Coordinates",
        "5_Overall_Displacement",
        "6_Group_B_iso",
        "7_Sample_Profile_XY",
        "8_Preferred_Orientation",
        "9_Occupancy",
        "10_Instrument_Profile",
        "11_Absorption_Extinction",
        "12_Final_Scale",
    ]

    def __init__(
        self,
        max_correlation: float = 0.85,
        max_condition_number: float = 1e5,
        zero_shift_limit_deg: float = 0.5,
        zero_shift_warn_deg: float = 0.05,
        displace_x_warn_um: float = 200.0,
        b_iso_bounds: tuple[float, float] = (0.2, 4.0),
        march_dollase_bounds: tuple[float, float] = (0.7, 1.3),
        stagnation_cycles: int = 5,
        stagnation_threshold: float = 1e-4,
    ):
        self.max_correlation = max_correlation
        self.max_condition_number = max_condition_number
        self.zero_shift_limit_deg = zero_shift_limit_deg
        self.zero_shift_warn_deg = zero_shift_warn_deg
        self.displace_x_warn_um = displace_x_warn_um
        self.b_iso_bounds = b_iso_bounds
        self.march_dollase_bounds = march_dollase_bounds
        self.stagnation_cycles = stagnation_cycles
        self.stagnation_threshold = stagnation_threshold

    def validate_correlations(
        self, correlation_matrix: np.ndarray, parameter_names: list[str]
    ) -> list[str]:
        """Flag parameter pairs exceeding correlation threshold |rho| > 0.85."""
        flagged = []
        n = len(parameter_names)
        for i in range(n):
            for j in range(i + 1, n):
                rho = abs(correlation_matrix[i, j])
                if rho > self.max_correlation:
                    flagged.append(
                        f"{parameter_names[i]} <-> {parameter_names[j]} (|rho|={rho:.3f})"
                    )
        return flagged

    def check_condition_number(self, hessian_or_cov: np.ndarray) -> tuple[float, bool]:
        """Compute condition number kappa(H). Returns (kappa, is_singular)."""
        try:
            cond = float(np.linalg.cond(hessian_or_cov))
            is_singular = cond > self.max_condition_number or np.isnan(cond)
            return cond, is_singular
        except (ValueError, np.linalg.LinAlgError):
            return float("inf"), True

    def audit_project_covariance(self, gpx: Any) -> tuple[list[str], float, bool]:
        """Audit live GSAS-II Covariance block for |rho_ij| > 0.85 and kappa > 10^5."""
        cov_block = (getattr(gpx, "data", {}) or {}).get("Covariance") or {}
        cov_data = cov_block.get("data", {}) if isinstance(cov_block, dict) else {}
        raw_cov = cov_data.get("covMatrix") if isinstance(cov_data, dict) else None
        if raw_cov is None and isinstance(cov_block, dict):
            raw_cov = cov_block.get("covMatrix")
        vary_list = cov_data.get("varyList") or cov_block.get("varyList") or []
        if raw_cov is None:
            return [], 1.0, False
        cov_arr = np.asarray(raw_cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] == 0 or cov_arr.shape[0] != cov_arr.shape[1]:
            return [], 1.0, False

        cond, is_singular = self.check_condition_number(cov_arr)  # noqa: PU001
        diag = np.maximum(np.diag(cov_arr), 1e-30)
        denom = np.sqrt(np.outer(diag, diag))
        corr_matrix = np.where(denom > 0.0, cov_arr / denom, 0.0)
        names = (
            [str(v) for v in vary_list]
            if len(vary_list) == cov_arr.shape[0]
            else [f"param_{i}" for i in range(cov_arr.shape[0])]
        )
        flagged = self.validate_correlations(corr_matrix, names)
        return flagged, cond, is_singular

    def enforce_box_constraints(
        self,
        zero_shift: float | None = None,
        b_iso_values: dict[str, float] | None = None,
        march_dollase_r: float | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate physical parameters against the hard domain limits.

        A zero shift inside ``zero_shift_limit_deg`` — the same ±0.5° that
        ``apply_box_constraints`` places on the parameter — is a physical
        possibility, not a violation: a real goniometer zero error of a few
        tenths of a degree is ordinary. Offsets above ``zero_shift_warn_deg`` are
        reported by :meth:`angular_origin_notes` as advice, so that the physical
        role of the parameter is not conflated with a misalignment symptom.
        """
        violations = []

        if zero_shift is not None and abs(zero_shift) > self.zero_shift_limit_deg:
            violations.append(
                f"Zero-shift |{zero_shift:.4f}°| exceeds hard limit {self.zero_shift_limit_deg}°"
            )

        if b_iso_values:
            for elem, b in b_iso_values.items():
                if b < self.b_iso_bounds[0] or b > self.b_iso_bounds[1]:
                    violations.append(
                        f"B_iso({elem})={b:.3f} Å² outside physical range {self.b_iso_bounds}"
                    )

        if march_dollase_r is not None and (
            march_dollase_r < self.march_dollase_bounds[0]
            or march_dollase_r > self.march_dollase_bounds[1]
        ):
            violations.append(
                f"March-Dollase r={march_dollase_r:.3f} outside valid interval {self.march_dollase_bounds}"
            )

        return len(violations) == 0, violations

    def angular_origin_notes(
        self,
        zero_shift: float | None = None,
        displace_x_um: float | None = None,
    ) -> list[str]:
        """Advisory notes naming the parameter that carries the angular offset.

        The constant part of the angular error belongs to the zero-shift
        parameter. ``DisplaceX`` follows a ``cos(theta)`` law and is therefore a
        different physical model: letting it carry the constant offset replaces a
        zero error by a fictitious specimen displacement of several tenths of a
        millimetre. Both parameters are reported side by side so the substitution
        is visible instead of being inferred from a systematic residual.
        """
        notes: list[str] = []

        if zero_shift is not None and abs(zero_shift) > self.zero_shift_warn_deg:
            notes.append(
                f"Zero-shift |{zero_shift:.4f}°| is above the well-aligned-goniometer "
                f"reference of {self.zero_shift_warn_deg}°; the constant offset belongs "
                "to Zero, not to the specimen displacement"
            )

        if displace_x_um is not None and abs(displace_x_um) > self.displace_x_warn_um:
            notes.append(
                f"DisplaceX = {displace_x_um:.1f} um is a large specimen displacement; "
                "if a constant angular offset is present, check whether Zero should "
                "carry it instead"
            )

        return notes
