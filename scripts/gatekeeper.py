"""Crystallochemical gatekeeper for structural model validation."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import gemmi
import numpy as np
import spglib
from constants import DEFAULT_GII_REJECT_THRESHOLD
from loguru import logger
from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
from pymatgen.core import Structure  # type: ignore[attr-defined]

# spglib 2.7.0 keeps its legacy error handling on by default and warns on every
# API call; from 2.8.0 the default flips to raising SpglibError, and in 3.0 the
# legacy path is removed. Opting in now keeps the behaviour identical across those
# generations and drops the transitional DeprecationWarning. The switch itself is
# scheduled for removal, so the assignment is deliberately tolerant.
try:
    spglib.error.OLD_ERROR_HANDLING = False
except AttributeError:
    logger.debug("spglib: legacy error-handling switch is absent; modern API assumed")

DEFAULT_MIN_DISTANCE_RATIO: float = 0.72
BOND_COLLAPSE_RATIO: float = 0.65
DEFAULT_SYMMETRY_TOLERANCE: float = 0.20


class GiiStatus(StrEnum):
    """Crystallochemical stability classification based on Global Instability Index."""

    NOT_APPLICABLE = "N/A"
    STABLE = "STABLE"
    ACCEPTABLE = "ACCEPTABLE"
    COVALENT_RELAXED = "COVALENT_RELAXED"
    REJECT = "REJECT"
    FALLBACK_RIGID_SPHERES = "FALLBACK_RIGID_SPHERES"
    PASSED = "PASSED"
    FAIL = "FAIL"


__all__ = [
    "BOND_COLLAPSE_RATIO",
    "DEFAULT_GII_REJECT_THRESHOLD",
    "DEFAULT_MIN_DISTANCE_RATIO",
    "DEFAULT_SYMMETRY_TOLERANCE",
    "AuditReport",
    "CrystallochemicalGatekeeper",
    "GiiStatus",
]


@dataclass
class AuditReport:
    """Detailed audit report of a candidate or refined crystal structure."""

    is_valid: bool
    gii_value: float | None = None
    gii_status: GiiStatus = GiiStatus.NOT_APPLICABLE
    spacegroup_initial: str = ""
    spacegroup_standard: str = ""
    supergroup_detected: bool = False
    suggested_spacegroup: str | None = None
    min_interatomic_distance: float = 0.0
    rigid_sphere_violations: list[str] = field(default_factory=list)
    b_iso_violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "gii_value": self.gii_value,
            "gii_status": str(self.gii_status),
            "spacegroup_initial": self.spacegroup_initial,
            "spacegroup_standard": self.spacegroup_standard,
            "supergroup_detected": self.supergroup_detected,
            "suggested_spacegroup": self.suggested_spacegroup,
            "min_interatomic_distance": self.min_interatomic_distance,
            "rigid_sphere_violations": list(self.rigid_sphere_violations),
            "b_iso_violations": list(self.b_iso_violations),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


class CrystallochemicalGatekeeper:
    """Rigorous crystallochemical validator enforcing physical, stereochemical,

    and symmetry constraints before a CIF model is accepted.
    """

    def __init__(
        self,
        gii_reject_threshold: float = DEFAULT_GII_REJECT_THRESHOLD,
        min_distance_ratio: float = DEFAULT_MIN_DISTANCE_RATIO,
        bond_collapse_ratio: float = BOND_COLLAPSE_RATIO,
        symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    ) -> None:
        self.gii_reject_threshold = gii_reject_threshold
        self.min_distance_ratio = min_distance_ratio
        self.bond_collapse_ratio = bond_collapse_ratio
        self.symmetry_tolerance = symmetry_tolerance
        # pymatgen BVAnalyzer.__init__ ships without type annotations.
        self.bv_analyzer = BVAnalyzer()  # type: ignore[no-untyped-call]

    def audit_structure(
        self, structure: Structure, b_iso_dict: dict[str, float] | None = None
    ) -> AuditReport:
        """Perform full crystallochemical audit on a pymatgen Structure.

        Args:
            structure: Structure to validate.
            b_iso_dict: Optional mapping of site/element to isotropic thermal factor B_iso.
        """
        report = AuditReport(is_valid=True)
        report.spacegroup_initial = structure.get_space_group_info()[0]

        # 1. Symmetry & Hidden Supergroup Check via spglib and gemmi
        self._check_symmetry(structure, report)

        # 2. Minimum Interatomic Distances & Rigid Sphere Check (gemmi fallback/guard)
        self._check_distances(structure, report)

        # 3. Bond Valence Sum (BVS) & Global Instability Index (GII)
        self._check_bvs_gii(structure, report)

        # 4. Thermal parameter hierarchy check: B_heavy <= B_light
        if b_iso_dict:
            self._check_thermal_parameters(structure, b_iso_dict, report)

        # Final verdict
        if len(report.errors) > 0:
            report.is_valid = False

        return report

    def _check_symmetry(self, structure: Structure, report: AuditReport) -> None:
        """Standardize space group to International Tables Vol. A and check for supergroups."""
        try:
            # Normalize with gemmi
            sg_gemmi = gemmi.find_spacegroup_by_name(report.spacegroup_initial)
            if sg_gemmi:
                report.spacegroup_standard = sg_gemmi.short_name()
            else:
                report.spacegroup_standard = report.spacegroup_initial
        except (RuntimeError, ValueError, KeyError) as e:
            logger.debug("gemmi spacegroup lookup notice: {}", e)
            report.warnings.append(f"gemmi spacegroup lookup notice: {e}")
            report.spacegroup_standard = report.spacegroup_initial

        # Search for hidden higher symmetry via spglib
        try:
            cell = (
                structure.lattice.matrix,
                structure.frac_coords,
                [site.specie.number for site in structure],
            )
            # spglib type stubs declare Sequence[Sequence[float]] while accepting ndarray at runtime.
            dataset = spglib.get_symmetry_dataset(cell, symprec=self.symmetry_tolerance)  # type: ignore[arg-type]
            if dataset:
                detected_sg = dataset.international
                if detected_sg != report.spacegroup_initial and dataset.number > 1:
                    report.supergroup_detected = True
                    report.suggested_spacegroup = detected_sg
                    report.warnings.append(
                        f"Possible higher symmetry / supergroup detected: {detected_sg} "
                        f"(initial was {report.spacegroup_initial}, tolerance={self.symmetry_tolerance} Å)"
                    )
        except (RuntimeError, ValueError, KeyError, spglib.SpglibError) as e:
            logger.debug("spglib symmetry analysis notice: {}", e)
            report.warnings.append(f"spglib symmetry analysis notice: {e}")

    def _check_distances(self, structure: Structure, report: AuditReport) -> None:
        """Check all interatomic distances against hard-sphere covalent/ionic limits."""
        dm = structure.distance_matrix
        np.fill_diagonal(dm, np.inf)
        min_d = float(np.min(dm))
        report.min_interatomic_distance = min_d

        if min_d < 1.0:
            report.errors.append(
                f"Severe atomic overlap detected: minimum distance {min_d:.3f} Å < 1.0 Å"
            )

        # Element-pair radii check
        sites = list(structure)
        for i in range(len(sites)):
            r_i = gemmi.Element(sites[i].specie.symbol).covalent_r or 1.0
            for j in range(i + 1, len(sites)):
                r_j = gemmi.Element(sites[j].specie.symbol).covalent_r or 1.0
                limit = self.min_distance_ratio * (r_i + r_j)
                actual_d = dm[i, j]
                if actual_d < limit:
                    msg = (
                        f"{sites[i].specie.symbol}-{sites[j].specie.symbol} distance "
                        f"{actual_d:.3f} Å < threshold {limit:.3f} Å"
                    )
                    report.rigid_sphere_violations.append(msg)
                    if actual_d < self.bond_collapse_ratio * (r_i + r_j):
                        report.errors.append(f"Unphysical bond collapse: {msg}")

    def _check_bvs_gii(self, structure: Structure, report: AuditReport) -> None:
        """Calculate Global Instability Index (GII) with domain-neutral fallback."""
        try:
            s_oxi = self.bv_analyzer.get_oxi_state_decorated_structure(structure)
            diffs = []

            for site in s_oxi:
                formal_v = abs(float(site.specie.oxi_state or 0.0))
                neighbors = s_oxi.get_neighbors(site, r=3.5)
                # pymatgen calculate_bv_sum ships without type annotations.
                v_calc = abs(float(calculate_bv_sum(site, neighbors)))  # type: ignore[no-untyped-call]
                diffs.append((formal_v - v_calc) ** 2)

            gii = float(np.sqrt(np.mean(diffs)))
            report.gii_value = gii

            ionic_limit = min(DEFAULT_GII_REJECT_THRESHOLD, self.gii_reject_threshold)
            if gii <= 0.10:
                report.gii_status = GiiStatus.STABLE
            elif gii <= ionic_limit:
                report.gii_status = GiiStatus.ACCEPTABLE
            elif gii <= self.gii_reject_threshold:
                report.gii_status = GiiStatus.COVALENT_RELAXED
            else:
                report.gii_status = GiiStatus.REJECT
                report.errors.append(
                    f"Global Instability Index GII={gii:.3f} v.u. exceeds reject threshold {self.gii_reject_threshold} v.u."
                )
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            # Domain-neutral fallback: metallic, intermetallic, zero-valence, or mixed-valence systems
            formula = structure.composition.reduced_formula
            logger.debug("BVS/BVAnalyzer notice for {}: {}", formula, e)
            report.warnings.append(
                f"Formal integer valence states cannot be partitioned statically by BVAnalyzer ({e}). "
                "BVS evaluation skipped; physical plausibility assessed via rigid-sphere and spacegroup criteria."
            )
            report.gii_status = GiiStatus.FALLBACK_RIGID_SPHERES

    def _check_thermal_parameters(
        self, structure: Structure, b_iso_dict: dict[str, float], report: AuditReport
    ) -> None:
        """Verify that B_heavy <= B_light holds physically (heavy cations have lower B_iso than anions)."""
        elements = {site.specie.symbol: site.specie.atomic_mass for site in structure}
        sorted_elems = sorted(elements.keys(), key=lambda e: float(elements[e]))
        if len(sorted_elems) >= 2:
            lightest = sorted_elems[0]
            heaviest = sorted_elems[-1]
            b_light = b_iso_dict.get(lightest)
            b_heavy = b_iso_dict.get(heaviest)
            if b_light is not None and b_heavy is not None and b_heavy > 1.5 * b_light:
                err = (
                    f"Unphysical thermal factor inversion: B_iso({heaviest}, M={elements[heaviest]:.1f}) = {b_heavy:.2f} "
                    f"> B_iso({lightest}, M={elements[lightest]:.1f}) = {b_light:.2f}"
                )
                report.b_iso_violations.append(err)
                report.errors.append(err)
