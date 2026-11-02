"""Continuous analytical candidate phase scoring engine for powder XRD patterns.

Implements an advisory multi-component scoring model:
- Fast 1D Bragg lattice strain search with tan(theta) dispersion
- Profile probability field U_exp(2theta) derived from counting statistics
- Continuous profile cross-correlation with canonical instrumental Caglioti convolution
- Line completeness factor (ML / TL) bounded strictly to experimental range
- Texture-aware 3-peak major reflection validation (Veto vs TEXTURE_SUSPECT)
- Spearman rank intensity correlation robust to preferred orientation
- Robust median peak-ratio scale factor estimation
- Diagnostic crystallographic flags for autonomous agent decision-making
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Element semantics are owned by cif_retriever.ElementFilter (SSOT: --all,
# --any, --none, --strict) and its extract_elements. Reusing them keeps one
# definition of "which elements does this candidate contain" for CIF files and
# for PDF-2 cards alike; a second implementation would drift.
from cif_retriever import ElementFilter, extract_elements
from constants import CANONICAL_WAVELENGTH_KA1
from gsas2_adapter import read_difference_curve, read_project_difference_curve
from instrument import CanonicalInstrument, fit_physical_background
from loguru import logger
from numpy.typing import NDArray
from peaks import extract_peaks_robust
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# pymatgen re-exports Structure from pymatgen.core but its stub does not declare
# the export. Suppressed rule: attr-defined.
from pymatgen.core import Structure  # type: ignore[attr-defined]
from scipy.signal import find_peaks
from scipy.stats import spearmanr

# GSAS-II ships compiled binaries built against the NumPy 1.x ABI only (build-tools
# release v1.0.1 offers no numpy-2 build for win_64_p3.11), so this project is pinned to
# NumPy < 2. NumPy 2 renamed trapz -> trapezoid; resolve whichever spelling exists instead
# of hard-coding one generation.
try:
    # trapezoid is NumPy >= 2, trapz is NumPy < 2. The skill pins NumPy < 2 for the
    # GSAS-II binaries, so the fallback is the live path and the shim is deliberate.
    # Suppressed rule: attr-defined.
    from numpy import trapezoid  # type: ignore[attr-defined]
except ImportError:  # NumPy < 2
    from numpy import trapz as trapezoid


@dataclass(frozen=True)
class CandidateMatchDetail:
    """Record of matching an individual theoretical candidate reflection."""

    two_theta_calc: float
    two_theta_shifted: float
    intensity_calc_rel: float
    matched: bool
    matched_exp_two_theta: float | None = None
    matched_exp_intensity: float | None = None
    snr_local: float = 0.0


@dataclass(frozen=True)
class CandidateLines:
    """A candidate supplied as bare reflection lines rather than as a structure."""

    label: str
    formula: str
    volume: float
    d_spacing: NDArray[Any]
    intensity: NDArray[Any]
    card_cell: tuple[float | None, ...] = ()
    hkl: tuple[tuple[int, ...], ...] = ()


@dataclass
class CandidateScoringResult:
    """Comprehensive crystallographic diagnostic scoring descriptor for a candidate phase."""

    phase_name: str
    formula: str
    cif_file: str
    structure: Structure | None
    score: float  # Composite total score in [0.0, 100.0%]
    matched_lines: int  # Matched Lines (ML)
    total_lines: int  # Total Lines (TL) in experimental range
    completeness_ml_tl: float  # ML / TL ratio in [0.0, 1.0]
    intensity_match: float  # Spearman rank correlation score in [0.0, 100.0%]
    lattice_shift_da_a: float  # Optimal delta_a / a strain
    zero_shift: float  # Zero shift (deg 2theta)
    scale_factor: float  # Robust median peak scale factor
    pos_overlap: float  # Continuous profile cross-correlation S_pos in [0.0, 1.0]
    strain_penalty: float  # P_strain in [0.0, 1.0]
    missing_penalty: float  # P_missing in [0.0, 1.0]
    occam_penalty: float  # P_occam in [0.0, 1.0]
    strong_line_penalty: float  # P_strong in [0.0, 1.0]; graded DemoteUnmatchedStrong
    diagnostic_flags: list[str]  # Crystallographic diagnostic flags
    missing_strong_lines: list[dict[str, Any]] = field(default_factory=list)
    matched_lines_detail: list[CandidateMatchDetail] = field(default_factory=list)
    volume: float = 0.0
    cell: tuple[float | None, ...] = ()  # Card cell (a, b, c, alpha, beta, gamma)
    explained_peaks_count: int = 0
    explained_2thetas: list[float] = field(default_factory=list)
    # Coverage of the scored signal: which fraction of its significant peaks the
    # candidate accounts for, and the factor derived from it (1.0 when the profile
    # leaves the term out of the score).
    observed_peaks_total: int = 0
    observed_peaks_explained: int = 0
    observed_peak_coverage: float = 1.0
    coverage_penalty: float = 1.0
    space_group: str = ""  # Space group of the candidate structure; "" for line lists
    raw_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "phase_name": self.phase_name,
            "formula": self.formula,
            "cif_file": self.cif_file,
            "score": round(self.score, 2),
            "raw_score": round(self.raw_score if self.raw_score > 0.0 else self.score, 2),
            "matched_lines": self.matched_lines,
            "total_lines": self.total_lines,
            "completeness_ml_tl": round(self.completeness_ml_tl, 4),
            "intensity_match": round(self.intensity_match, 2),
            "lattice_shift_da_a": round(self.lattice_shift_da_a, 5),
            "zero_shift": round(self.zero_shift, 4),
            "scale_factor": round(self.scale_factor, 6),
            "pos_overlap": round(self.pos_overlap, 4),
            "strain_penalty": round(self.strain_penalty, 4),
            "missing_penalty": round(self.missing_penalty, 4),
            "occam_penalty": round(self.occam_penalty, 4),
            "strong_line_penalty": round(self.strong_line_penalty, 4),
            "diagnostic_flags": self.diagnostic_flags,
            "missing_strong_lines": self.missing_strong_lines,
            "matched_lines_detail": [
                {
                    "two_theta_calc": round(d.two_theta_calc, 4),
                    "two_theta_shifted": round(d.two_theta_shifted, 4),
                    "intensity_calc_rel": round(d.intensity_calc_rel, 2),
                    "matched": d.matched,
                    "matched_exp_two_theta": round(d.matched_exp_two_theta, 4)
                    if d.matched_exp_two_theta is not None
                    else None,
                    "matched_exp_intensity": round(d.matched_exp_intensity, 2)
                    if d.matched_exp_intensity is not None
                    else None,
                    "snr_local": round(d.snr_local, 2),
                }
                for d in self.matched_lines_detail
            ],
            "volume": round(self.volume, 2),
            "cell": [None if v is None else round(v, 5) for v in self.cell],
            "space_group": self.space_group,
            "explained_peaks_count": self.explained_peaks_count,
            "explained_2thetas": [round(x, 4) for x in self.explained_2thetas],
            "observed_peaks_total": self.observed_peaks_total,
            "observed_peaks_explained": self.observed_peaks_explained,
            "observed_peak_coverage": round(self.observed_peak_coverage, 4),
            "coverage_penalty": round(self.coverage_penalty, 4),
        }


def apply_bragg_cell_shift(
    two_theta: NDArray[Any],
    delta_a_over_a: float,
    zero_shift: float = 0.0,
    max_zero_shift: float = 0.04,
) -> NDArray[Any]:
    """Apply first-order analytical Bragg peak shift:
    d(2th) = -2*tan(theta) * (delta_a/a) + zero.
    Derived from differentiating Bragg's Law: lambda = 2*d*sin(theta).
    Zero-shift is clamped to [-max_zero_shift, max_zero_shift].
    """
    clamped_zero = float(np.clip(zero_shift, -abs(max_zero_shift), abs(max_zero_shift)))
    th_rad = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    shift_deg = np.degrees(2.0 * np.tan(th_rad) * delta_a_over_a)
    # numpy stubs type the arithmetic on ndarray as Any. Suppressed rule:
    # no-any-return.
    return two_theta - shift_deg + clamped_zero  # type: ignore[no-any-return]


def apply_anisotropic_bragg_shift(
    two_theta: NDArray[Any],
    delta_a_over_a: float,
    delta_c_over_c: float,
    c_weights: NDArray[Any],
    zero_shift: float = 0.0,
    max_zero_shift: float = 0.04,
) -> NDArray[Any]:
    """Apply axial anisotropic Bragg shift with directional weights w_c = (l * d / c)^2."""
    clamped_zero = float(np.clip(zero_shift, -abs(max_zero_shift), abs(max_zero_shift)))
    th_rad = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    w_c = np.clip(np.asarray(c_weights, dtype=float), 0.0, 1.0)
    eff_strain = (1.0 - w_c) * float(delta_a_over_a) + w_c * float(delta_c_over_c)
    shift_deg = np.degrees(2.0 * np.tan(th_rad) * eff_strain)
    return two_theta - shift_deg + clamped_zero  # type: ignore[no-any-return]


def bragg_two_theta(d_spacing: NDArray[Any], wavelength: float) -> NDArray[Any]:
    """Convert d-spacings to angles: 2theta = 2 * arcsin(lambda / (2 d)), in degrees.

    The relation is stated once, here, because three callers need it: the scoring engine,
    the first-pass line index and the tests that check a card against its own cell. A
    non-finite input stays non-finite, which the line index relies on: the columns it does
    not read are left as NaN and must not turn into angles.
    """
    d = np.asarray(d_spacing, dtype=float)
    arg = wavelength / (2.0 * np.maximum(d, 1e-6))
    # numpy stubs type the ufunc chain as Any. Suppressed rule: no-any-return.
    return np.degrees(2.0 * np.arcsin(np.clip(arg, -1.0, 1.0)))  # type: ignore[no-any-return]


def select_by_elements(
    candidate_elements: dict[str, set[str]],
    elem_filter: ElementFilter | None,
) -> list[str]:
    """Return the names of candidates satisfying the element filter.

    ``candidate_elements`` maps a candidate name to the set of elements it
    contains. The caller builds that set with ``extract_elements`` -- from a CIF
    structure for local candidates, from ``PDFCard.structural_formula`` for PDF-2
    cards -- so the extraction rules stay in one place.

    With ``elem_filter is None`` nothing is removed: the filter is an optional
    restriction, not a default narrowing. This matters because the candidate
    pool must cover every thermodynamically plausible product of the synthesis,
    container and atmosphere; silently dropping phases is prohibited by the
    skill's candidate-pool rule.
    """
    if elem_filter is None:
        return list(candidate_elements)
    return [name for name, elems in candidate_elements.items() if elem_filter.matches(elems)]


def calculate_complexity_penalty(line_count: float, n0: float = 20.0, alpha: float = 0.40) -> float:
    """Occam penalty for a candidate measured by its number of theoretical lines in the window.

    The complexity of a candidate governs how easily it matches the observed
    pattern *by chance*: the expected number of random coincidences is
    ``E[ML] = TL * P * 2 * delta / W`` (TL = theoretical lines, P = observed
    peaks, delta = angular tolerance, W = window width). Up to a constant factor
    that depends only on the experiment, the line count *is* the expected number
    of spurious matches, so it is the correct parsimony variable.

    Cell volume is not: a 34-line candidate and a 12-line candidate can have
    nearly equal volumes and therefore near-identical volume penalties, which is
    exactly how a rich-line candidate outranked the physically correct one.

    ``line_count <= n0`` is complexity-free; the penalty decays logarithmically
    beyond that, so a candidate is not punished for being merely well indexed.
    """
    if line_count <= n0:
        return 1.0
    excess_ratio = max(0.0, (line_count - n0) / n0)
    return float(1.0 / (1.0 + alpha * np.log(1.0 + excess_ratio)))


def count_unique_matched_peaks(
    matched_exp_indices: list[int],
    exp_two_theta: NDArray[Any],
    covering_2thetas: NDArray[Any],
    angular_tolerance: float = 0.20,
) -> int:
    """Count matched experimental peaks not already covered by accepted phases.

    ``covering_2thetas`` holds the positions explained by the phases accepted so
    far; a peak is credited only if none of them accounts for it within the
    angular tolerance. Under this sequential residual phase-exclusion criterion,
    pass 1 scores every candidate against the whole pattern, and subsequent passes
    credit only peaks not yet accounted for by higher-ranked accepted phases.

    Which lines belong to the covering set is the caller's decision -- it is the
    set of *accepted* phases, never the set of all candidates. Discounting
    against unaccepted candidates was measured to strip a 95 % dominant phase of
    a third of its credit.
    """
    covering = np.asarray(covering_2thetas, dtype=float)
    if covering.size == 0:
        return len(matched_exp_indices)

    unique_count = 0
    last = exp_two_theta.size - 1
    for exp_idx in matched_exp_indices:
        # ``np.searchsorted`` returns the array length for a position beyond the
        # last grid point, so an unclamped index raises IndexError here instead of
        # being counted as a miss.
        if not 0 <= exp_idx <= last:
            logger.warning("Matched peak index {} is outside the pattern grid", exp_idx)
            continue
        pos = exp_two_theta[exp_idx]
        if np.min(np.abs(covering - pos)) > angular_tolerance:
            unique_count += 1
    return unique_count


class ProfileProbabilityField:
    """Continuous statistical significance probability field U_exp(2theta)."""

    def __init__(
        self,
        two_theta: NDArray[Any],
        y_net: NDArray[Any],
        sigma: NDArray[Any],
        clamp_positive: bool = True,
    ):
        self.two_theta = np.asarray(two_theta, dtype=float)
        y_net_arr = np.asarray(y_net, dtype=float)
        # A raw pattern carries a positive net signal above the fitted baseline,
        # so clipping is harmless there. A Rietveld difference curve does not: its
        # information is the sign, and clipping it positive turns the field into a
        # broadly positive surface on which every candidate line "matches", so
        # completeness saturates at 1.0 for unrelated phases.
        self.y_net = np.maximum(0.0, y_net_arr) if clamp_positive else y_net_arr
        self.sigma = np.maximum(1e-3, np.asarray(sigma, dtype=float))
        self.u_field = self.y_net / self.sigma
        self.clamp_positive = clamp_positive

    @classmethod
    def from_raw_pattern(
        cls,
        two_theta: NDArray[Any],
        y_obs: NDArray[Any],
        y_bkg: NDArray[Any] | None = None,
    ) -> ProfileProbabilityField:
        """Construct probability field from raw experimental pattern.
        If background is not supplied, fits physical baseline using fit_physical_background.
        """
        two_theta = np.asarray(two_theta, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)
        if y_bkg is None:
            y_bkg = fit_physical_background(two_theta, y_obs, n_terms=6)
        else:
            y_bkg = np.asarray(y_bkg, dtype=float)

        y_net = np.maximum(0.0, y_obs - y_bkg)
        sigma = np.sqrt(np.maximum(1.0, y_obs))
        return cls(two_theta, y_net, sigma)

    @classmethod
    def from_difference_curve(
        cls,
        two_theta: NDArray[Any],
        y_obs: NDArray[Any],
        y_calc: NDArray[Any],
    ) -> ProfileProbabilityField:
        """Construct a signed probability field from the Rietveld difference curve.

        The field keeps the sign of ``Y_obs - Y_calc`` and is scaled by the scatter
        of the difference curve itself, estimated robustly from the high-angle
        region where reflections are sparse. Judging a candidate line against
        Poisson noise of the counts would credit a systematic residual with
        significance it does not have; judging the clipped positive part, as was
        done before, makes the significance test vacuous.
        """
        two_theta = np.asarray(two_theta, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)
        y_calc = np.asarray(y_calc, dtype=float)
        y_diff = y_obs - y_calc
        if y_diff.size:
            tail = y_diff[int(0.7 * y_diff.size) :]
            mad = float(np.median(np.abs(tail - np.median(tail)))) / 0.6745
        else:
            mad = 0.0
        sigma = np.full(y_diff.shape, max(mad, 1.0))
        return cls(two_theta, y_diff, sigma, clamp_positive=False)


def synthesize_candidate_profile(
    two_theta_grid: NDArray[Any],
    reflections_2th: NDArray[Any],
    reflections_i: NDArray[Any],
    instrument: CanonicalInstrument | None = None,
    eta: float = 0.5,
    window_multiplier: float = 4.0,
) -> NDArray[Any]:
    """Synthesize continuous pseudo-Voigt profile on the experimental angular grid."""
    if instrument is None:
        instrument = CanonicalInstrument()

    profile = np.zeros_like(two_theta_grid, dtype=float)
    if len(reflections_2th) == 0:
        return profile

    fwhms = instrument.fwhm_inst_deg(reflections_2th)
    c_g = 4.0 * np.log(2.0)

    for pos, height, fwhm in zip(reflections_2th, reflections_i, fwhms, strict=False):
        if height <= 0:
            continue
        half_w = max(window_multiplier * fwhm, 0.20)
        idx_start = int(np.searchsorted(two_theta_grid, pos - half_w))
        idx_end = int(np.searchsorted(two_theta_grid, pos + half_w))
        if idx_start >= idx_end:
            continue

        x_slice = two_theta_grid[idx_start:idx_end]
        delta2 = ((x_slice - pos) / max(fwhm, 1e-4)) ** 2
        lor = 1.0 / (1.0 + 4.0 * delta2)
        # exp(-large) underflows to zero by design in the far tails. The error
        # state is fixed locally so that a profile synthesis does not depend on a
        # caller's -- or a test session's -- numpy error policy.
        with np.errstate(under="ignore"):
            gauss = np.exp(-c_g * delta2)
        pv = height * (eta * lor + (1.0 - eta) * gauss)
        profile[idx_start:idx_end] += pv

    return profile


# Named scoring presets for major/general phase identification ("ide-all") and
# trace secondary phase detection ("minors"). A minor phase has few strong lines
# and is not expected to explain dominant matrix peaks, so unmatched strong lines
# are not demoted and weaker reference reflections remain active.
SCORING_PROFILES: dict[str, dict[str, float]] = {
    "ide-all": {
        "angular_tolerance": 0.20,
        "min_line_intensity": 1.0,
        "max_delta_a": 0.020,
        "max_zero_shift": 0.03,
        "strong_line_threshold": 0.50,
        "demote_unmatched_strong": 1.0,
        "coverage_weight": 0.0,
    },
    "minors": {
        "angular_tolerance": 0.20,
        "min_line_intensity": 0.30,
        "max_delta_a": 0.020,
        "max_zero_shift": 0.03,
        "strong_line_threshold": 0.75,
        "demote_unmatched_strong": 0.0,
        "coverage_weight": 0.0,
    },
}


def _split_criteria(raw: str | None) -> frozenset[str] | None:
    """Turn a comma-separated flag value into a criterion set, or None when not asked.

    ``None`` and an empty string both mean "do not narrow": an absent criterion is not a
    criterion that rejects everything.
    """
    if raw is None:
        return None
    parts = frozenset(part.strip() for part in raw.split(",") if part.strip())
    return parts or None


def resolve_angular_tolerance(profile: dict[str, float], explicit: float | None) -> float:
    """Return the matching window in degrees 2theta: an explicit flag, else the profile.

    Stated once because both the scoring engine and the first pass need it, and they must
    agree: a first pass narrower than the scoring that follows it would drop cards the
    scorer would have accepted.
    """
    return float(explicit) if explicit is not None else float(profile["angular_tolerance"])


class CandidateScoringEngine:
    """Analytical candidate phase scoring engine.

    Evaluates candidate crystal structures against experimental XRD patterns or
    residual difference curves without greedy subtractive loops.
    """

    def __init__(
        self,
        wavelength: float = CANONICAL_WAVELENGTH_KA1,
        angular_tolerance: float = 0.20,
        min_line_intensity: float = 1.0,
        max_delta_a: float = 0.020,
        max_zero_shift: float = 0.03,
        strong_line_threshold: float = 0.50,
        demote_unmatched_strong: float = 1.0,
        degenerate_overlap_max: float = 0.30,
        coverage_weight: float = 0.5,
    ):
        self.wavelength = wavelength
        self.angular_tolerance = angular_tolerance
        self.min_line_intensity = min_line_intensity
        self.max_delta_a = max_delta_a
        self.max_zero_shift = max_zero_shift
        # Exponent of the observed-pattern coverage term. 0.0 removes it from the
        # product without removing the measurement; the coverage is still reported.
        self.coverage_weight = coverage_weight
        # DemoteUnmatchedStrong (graded). "Strong" is relative to the candidate's
        # own strongest line, so the threshold is instrument-independent.
        self.strong_line_threshold = strong_line_threshold
        # 1.0 = full penalty, 0.0 = disabled.
        self.demote_unmatched_strong = demote_unmatched_strong
        # A candidate that "explains" every one of its own lines while the profile
        # cross-correlation with the field stays below this value is matching a
        # featureless surface rather than a pattern; the flag says so.
        self.degenerate_overlap_max = degenerate_overlap_max
        self.calculator = XRDCalculator(wavelength=self.wavelength)
        self.instrument = CanonicalInstrument(wavelength_ka1=self.wavelength)

    def calculate_cif_pattern(
        self,
        structure_or_path: Structure | str | Path,
        two_theta_range: tuple[float, float],
    ) -> tuple[Structure, NDArray[Any], NDArray[Any]]:
        """Compute Bragg reflections strictly within the experimental angular window."""
        if isinstance(structure_or_path, (str, Path)):
            structure = Structure.from_file(str(structure_or_path))
        else:
            structure = structure_or_path

        pattern = self.calculator.get_pattern(structure, two_theta_range=two_theta_range)
        peaks_2th = np.asarray(pattern.x, dtype=float)
        peaks_i = np.asarray(pattern.y, dtype=float)

        # Restrict strictly to experimental window
        in_range = (peaks_2th >= two_theta_range[0]) & (peaks_2th <= two_theta_range[1])
        peaks_2th = peaks_2th[in_range]
        peaks_i = peaks_i[in_range]

        if len(peaks_i) == 0:
            return structure, np.array([], dtype=float), np.array([], dtype=float)

        # Normalize relative intensities to max = 100.0 within range
        max_i = float(np.max(peaks_i))
        if max_i > 0:
            peaks_i = (peaks_i / max_i) * 100.0

        # Filter lines meeting the intensity cutoff
        cutoff_mask = peaks_i >= self.min_line_intensity
        return structure, peaks_2th[cutoff_mask], peaks_i[cutoff_mask]

    def calculate_lines_pattern(
        self,
        source: CandidateLines,
        two_theta_range: tuple[float, float],
    ) -> tuple[Structure | None, NDArray[Any], NDArray[Any]]:
        """Window and normalize a bare line list exactly like a calculated pattern.

        The d-spacings are converted to angles by Bragg's law at the engine
        wavelength, restricted to the experimental window and scaled to a maximum
        of 100.0, so a card candidate enters the scoring on the same footing as a
        CIF candidate.
        """
        d = np.asarray(source.d_spacing, dtype=float)
        intensities = np.asarray(source.intensity, dtype=float)
        # The field carries d-spacings by contract; no guessing between units.
        peaks_2th = self.d_to_two_theta(d)
        peaks_i = intensities

        in_range = (peaks_2th >= two_theta_range[0]) & (peaks_2th <= two_theta_range[1])
        peaks_2th = peaks_2th[in_range]
        peaks_i = peaks_i[in_range]
        if len(peaks_i) == 0:
            return None, np.array([], dtype=float), np.array([], dtype=float)

        max_i = float(np.max(peaks_i))
        if max_i > 0:
            peaks_i = (peaks_i / max_i) * 100.0
        cutoff_mask = peaks_i >= self.min_line_intensity
        return None, peaks_2th[cutoff_mask], peaks_i[cutoff_mask]

    def d_to_two_theta(self, d_spacing: NDArray[Any]) -> NDArray[Any]:
        """Bragg's law at this engine's wavelength; see :func:`bragg_two_theta`."""
        return bragg_two_theta(d_spacing, self.wavelength)

    def score_candidates(
        self,
        two_theta: NDArray[Any],
        y_net: NDArray[Any],
        sigma: NDArray[Any],
        candidate_structures: dict[str, Structure | str | Path | CandidateLines],
        refine_zero: bool = False,
        min_score: float = 0.0,
        observed_signal: NDArray[Any] | None = None,
    ) -> list[CandidateScoringResult]:
        """Evaluate and rank a pool of candidate CIFs against experimental data.

        ``observed_signal`` is the measured intensity the coverage denominator is drawn
        from. Without it the coverage term stays inert and says so: it is deliberately not
        measured on ``y_net``, which :meth:`ProfileProbabilityField.from_raw_pattern` clips
        at zero. Clipping removes the negative half of the noise, the MAD estimate falls
        with it (measured on the CeO2 fixture: 3.51 -> 0.68), and the threshold then admits a
        ridge of the detector's own smoothing as a peak -- a maximum of height 13.9 against
        a true noise threshold of 14.05. A denominator counting that ridge penalises every
        candidate, the correct one included, so a caller that has no measured signal gets no
        coverage term rather than a wrong one.
        """
        results: list[CandidateScoringResult] = []
        if len(two_theta) == 0 or len(candidate_structures) == 0:
            return results

        two_theta = np.asarray(two_theta, dtype=float)
        prob_field = ProfileProbabilityField(two_theta, y_net, sigma)
        t_min = float(np.min(two_theta))
        t_max = float(np.max(two_theta))
        t_range = (t_min, t_max)

        u_exp = prob_field.u_field
        norm_u = float(np.sqrt(trapezoid(u_exp**2, two_theta)))
        if norm_u <= 1e-6:
            return results

        # Denominator of the coverage term: the significant observed peaks of the pattern
        # being scored. The extraction carries a MAD noise threshold, and that is not a
        # detail: a plain maximum search without one returned 259 "peaks" on a pattern
        # whose nine peaks are visible to the eye, and a term built on that denominator
        # cut the true phase's score from 91.75 to 18.91 -- it penalised the correct answer.
        #
        # The source must be the uncut measured intensity, so a caller that has none gets no
        # term instead of a wrong one: the clipped net field removes the negative half of the
        # noise, and the threshold that follows from it counts ridges of the detector's own
        # smoothing as peaks (measured: the denominator grew from 9 to 10 on the CeO2 fixture
        # and the true phase's score fell twice as far as it should).
        significant_observed: NDArray[Any] | None = None
        if observed_signal is None:
            if self.coverage_weight > 0.0:
                logger.info(
                    "Coverage term inert: no measured signal was supplied, so the "
                    "denominator cannot be counted without counting clipped-noise ridges. "
                    "Pass observed_signal (the measured intensity) to use it."
                )
        else:
            coverage_table = extract_peaks_robust(two_theta, observed_signal)
            if len(coverage_table):
                significant_observed = coverage_table.two_theta
                logger.debug(
                    "Coverage denominator: {} significant observed peak(s)",
                    len(coverage_table),
                )

        # 1D Strain grid for fast peak lookup
        da_grid = np.linspace(-self.max_delta_a, self.max_delta_a, 41)
        z_grid = (
            np.linspace(-self.max_zero_shift, self.max_zero_shift, 7)
            if refine_zero
            else np.array([0.0])
        )

        for name, struct_src in candidate_structures.items():
            try:
                if isinstance(struct_src, CandidateLines):
                    struct, cand_2th, cand_i = self.calculate_lines_pattern(  # noqa: PU001
                        struct_src, two_theta_range=t_range
                    )
                    cif_path_str = name
                    formula_str = struct_src.formula
                    struct_vol = float(struct_src.volume)
                    candidate_cell = struct_src.card_cell
                    candidate_space_group = ""
                else:
                    struct, cand_2th, cand_i = self.calculate_cif_pattern(  # noqa: PU001
                        struct_src, two_theta_range=t_range
                    )
                    cif_path_str = str(struct_src) if isinstance(struct_src, (str, Path)) else name
                    formula_str = struct.composition.reduced_formula
                    struct_vol = float(getattr(struct, "volume", 100.0))
                    candidate_cell = ()
                    # The space-group token lets two candidates of one structure type
                    # be recognised as a solid-solution pair rather than as a strong
                    # and a weak, unrelated phase.
                    candidate_space_group = str(struct.get_space_group_info()[0])
            except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError) as err:
                logger.warning(f"Failed to calculate pattern for candidate '{name}': {err}")
                continue

            tl_count = len(cand_2th)
            if tl_count == 0:
                continue

            # tl_count is the parsimony variable: it scales the expected number of
            # random coincidences E[ML], whereas the cell volume does not.
            complexity_penalty = calculate_complexity_penalty(float(tl_count))

            # Stage 1: Fast vector lookup for optimal strain (da/a, zero)
            best_fast_score = -1e9
            best_da = 0.0
            best_z = 0.0

            for z_val in z_grid:
                for da_val in da_grid:
                    shifted_2th = apply_bragg_cell_shift(
                        cand_2th,
                        delta_a_over_a=float(da_val),
                        zero_shift=float(z_val),
                        max_zero_shift=self.max_zero_shift,
                    )
                    # Vectorized lookup in probability field
                    u_sampled = np.interp(shifted_2th, two_theta, u_exp, left=0.0, right=0.0)
                    fast_score = float(np.sum(cand_i * u_sampled))
                    if fast_score > best_fast_score:
                        best_fast_score = fast_score
                        best_da = float(da_val)
                        best_z = float(z_val)

            # Stage 2: Single pseudo-Voigt profile synthesis at optimal strain
            opt_shifted_2th = apply_bragg_cell_shift(
                cand_2th,
                delta_a_over_a=best_da,
                zero_shift=best_z,
                max_zero_shift=self.max_zero_shift,
            )
            p_ref = synthesize_candidate_profile(
                two_theta_grid=two_theta,
                reflections_2th=opt_shifted_2th,
                reflections_i=cand_i,
                instrument=self.instrument,
            )

            norm_pref = float(np.sqrt(trapezoid(p_ref**2, two_theta)))
            if norm_pref <= 1e-6:
                continue

            # Continuous profile cross-correlation S_pos (linear and sqrt-compressed matching U_exp)
            linear_overlap = float(trapezoid(u_exp * p_ref, two_theta) / (norm_u * norm_pref))
            p_ref_sqrt = np.sqrt(np.maximum(p_ref, 0.0))
            norm_pref_sqrt = float(np.sqrt(trapezoid(p_ref_sqrt**2, two_theta)))
            sqrt_overlap = (
                float(trapezoid(u_exp * p_ref_sqrt, two_theta) / (norm_u * norm_pref_sqrt))
                if norm_pref_sqrt > 1e-6
                else 0.0
            )
            pos_overlap = float(np.clip(max(linear_overlap, sqrt_overlap), 0.0, 1.0))

            # Coverage of the observed pattern: which fraction of its significant peaks
            # this candidate accounts for. This is the quantity that was missing, and its
            # absence is measured: every top-ranked card on the Sm-Gd-S series explained
            # only 2-6 of the 10-20 observed peaks while reaching perfect completeness of
            # its own line list. The window is the nominal angular tolerance and not the
            # per-line FWHM widening used for counting the candidate's own lines, so the
            # term cannot overstate what is explained.
            observed_total = 0
            observed_explained = 0
            if significant_observed is not None:
                observed_total = int(significant_observed.size)
                nearest = np.min(
                    np.abs(significant_observed[:, None] - opt_shifted_2th[None, :]), axis=1
                )
                observed_explained = int(np.sum(nearest <= self.angular_tolerance))
            observed_coverage = observed_explained / observed_total if observed_total > 0 else 1.0
            coverage_penalty = (
                float(observed_coverage**self.coverage_weight)
                if self.coverage_weight > 0.0
                else 1.0
            )

            # Reflection matching analysis (ML / TL & line details)
            matched_pairs: list[tuple[float, float, float]] = []  # (exp_int, ref_int, shifted_2th)
            explained_2thetas: list[float] = []
            matched_details: list[CandidateMatchDetail] = []
            fwhms = self.instrument.fwhm_inst_deg(opt_shifted_2th)

            for orig_pos, pos, ref_i, fwhm in zip(
                cand_2th, opt_shifted_2th, cand_i, fwhms, strict=False
            ):
                window = max(self.angular_tolerance, fwhm)
                idx_lo = int(np.searchsorted(two_theta, pos - window))
                idx_hi = int(np.searchsorted(two_theta, pos + window))
                if idx_lo >= idx_hi:
                    matched_details.append(
                        CandidateMatchDetail(
                            two_theta_calc=float(orig_pos),
                            two_theta_shifted=float(pos),
                            intensity_calc_rel=float(ref_i),
                            matched=False,
                        )
                    )
                    continue

                local_u = u_exp[idx_lo:idx_hi]
                local_y_net = prob_field.y_net[idx_lo:idx_hi]
                max_local_u = float(np.max(local_u))
                max_local_y = float(np.max(local_y_net))
                is_matched = max_local_u >= 2.5

                if is_matched:
                    matched_pairs.append((max_local_y, float(ref_i), float(pos)))
                    explained_2thetas.append(float(pos))

                matched_details.append(
                    CandidateMatchDetail(
                        two_theta_calc=float(orig_pos),
                        two_theta_shifted=float(pos),
                        intensity_calc_rel=float(ref_i),
                        matched=is_matched,
                        matched_exp_two_theta=float(pos) if is_matched else None,
                        matched_exp_intensity=max_local_y if is_matched else None,
                        snr_local=max_local_u,
                    )
                )

            ml_count = len(matched_pairs)
            ml_tl_ratio = ml_count / tl_count if tl_count > 0 else 0.0
            s_ml = float(ml_tl_ratio**0.75)

            # Three-tier texture-aware Veto logic
            # Inspect top 3 major lines
            sort_order = np.argsort(cand_i)[::-1]
            top_indices = sort_order[: min(3, len(sort_order))]
            major_snrs: list[float] = []

            for top_idx in top_indices:
                p_pos = opt_shifted_2th[top_idx]
                p_fwhm = fwhms[top_idx]
                w = max(self.angular_tolerance, p_fwhm)
                i_lo = int(np.searchsorted(two_theta, p_pos - w))
                i_hi = int(np.searchsorted(two_theta, p_pos + w))
                top_u = float(np.max(u_exp[i_lo:i_hi])) if i_lo < i_hi else 0.0
                major_snrs.append(top_u)

            p_missing = 1.0
            veto_flag: str | None = None
            texture_suspect: bool = False

            # Hard veto: all 3 top lines in noise floor (< 2.5 sigma)
            if all(snr < 2.5 for snr in major_snrs):
                p_missing = 0.05
                veto_flag = "VETO: NO_MAJOR_PEAKS"
            elif major_snrs and major_snrs[0] < 2.0:
                # 100% primary peak missing, but another major peak is detected
                p_missing = 0.50
                texture_suspect = True

            # Spearman rank intensity agreement (robust to texture)
            if len(matched_pairs) < 3:
                s_int = 0.50
            else:
                exp_vals = [p[0] for p in matched_pairs]
                ref_vals = [p[1] for p in matched_pairs]
                r_s, _ = spearmanr(exp_vals, ref_vals)  # noqa: PU001
                s_int = max(0.50, float(1.0 + r_s) / 2.0) if np.isfinite(r_s) else 0.50

            # Strain and zero penalties
            p_strain = float(np.exp(-50.0 * (best_da**2) - 20.0 * (best_z**2)))

            # Robust median peak scale factor
            if matched_pairs:
                scale_ratios = []
                for exp_y, _, pos_val in matched_pairs:
                    p_ref_val = float(np.interp(pos_val, two_theta, p_ref))
                    if p_ref_val > 1e-4:
                        scale_ratios.append(exp_y / p_ref_val)
                scale_c = float(np.median(scale_ratios)) if scale_ratios else 0.0
            else:
                scale_c = 0.0

            # DemoteUnmatchedStrong (graded): a candidate is down-weighted in
            # proportion to the reference intensity of its own STRONG lines that
            # find no observed counterpart. A rich-line decoy carries many strong
            # lines that are absent in the experiment and is demoted; a true minor
            # phase, whose strong lines are all observed, is not. The alternative
            # reading (strong observed peaks left unexplained by the candidate)
            # would demote every minor phase and would not touch the rich-line
            # decoy at all, so it is not the implemented semantics.
            #
            # No penalty when the missing primary line is already explained as
            # preferred orientation: P_missing and P_strong describe the same
            # observation, and compounding them double-counts it.
            strong_cut = self.strong_line_threshold * float(np.max(cand_i))
            sum_strong = 0.0
            sum_unmatched_strong = 0.0
            for detail in matched_details:
                if detail.intensity_calc_rel >= strong_cut:
                    sum_strong += detail.intensity_calc_rel
                    if not detail.matched:
                        sum_unmatched_strong += detail.intensity_calc_rel

            if texture_suspect:
                p_strong = 1.0
            elif sum_strong > 0.0 and self.demote_unmatched_strong > 0.0:
                raw_strong = 1.0 - (sum_unmatched_strong / sum_strong)
                p_strong = float(1.0 - self.demote_unmatched_strong * (1.0 - raw_strong))
                p_strong = float(np.clip(p_strong, 0.0, 1.0))
            else:
                p_strong = 1.0

            # Composite total score [0, 100]
            total_score = (
                100.0
                * pos_overlap
                * s_ml
                * s_int
                * p_missing
                * p_strain
                * complexity_penalty
                * p_strong
                * coverage_penalty
            )
            total_score = float(np.clip(total_score, 0.0, 100.0))

            # Crystallographic diagnostic flags
            flags: list[str] = []
            if veto_flag:
                flags.append(veto_flag)
            elif texture_suspect:
                flags.append("TEXTURE_SUSPECT (Missing primary line)")

            if observed_total >= 5 and observed_coverage < 0.5:
                flags.append(f"LOW_OBSERVED_COVERAGE ({observed_explained}/{observed_total})")

            if ml_tl_ratio < 0.20 and tl_count >= 10:
                flags.append(f"LOW_COMPLETENESS (ML/TL={ml_count}/{tl_count})")

            # Vacuous completeness: every line "matched" while the profile
            # cross-correlation stayed low. On a difference curve this is the
            # signature of matching residual structure, not of a phase.
            if ml_tl_ratio >= 0.999 and pos_overlap < self.degenerate_overlap_max:
                flags.append(
                    f"DEGENERATE_COMPLETENESS (ML/TL=1.00 at pos_overlap={pos_overlap:.2f})"
                )

            if abs(best_da) > 0.005:
                sign = "+" if best_da > 0 else ""
                flags.append(f"LATTICE_STRAINED (da/a={sign}{best_da * 100:.2f}%)")

            if tl_count <= 4 and ml_count >= 1 and total_score >= 40.0:
                flags.append(f"STRONG_ISOLATED ({ml_count}/{tl_count})")

            if total_score >= 85.0 and ml_tl_ratio >= 0.80 and not veto_flag:
                flags.append("PERFECT_MATCH")
            elif total_score >= 65.0 and ml_tl_ratio >= 0.70 and not veto_flag:
                flags.append("STRONG_MATCH")
            elif 35.0 <= total_score < 65.0:
                flags.append("MINOR_MATCH")

            if not flags:
                flags.append("EVALUATED")

            if total_score >= min_score:
                results.append(
                    CandidateScoringResult(
                        phase_name=name,
                        formula=formula_str,
                        cif_file=cif_path_str,
                        structure=struct,
                        score=total_score,
                        matched_lines=ml_count,
                        total_lines=tl_count,
                        completeness_ml_tl=ml_tl_ratio,
                        intensity_match=s_int * 100.0,
                        lattice_shift_da_a=best_da,
                        zero_shift=best_z,
                        scale_factor=scale_c,
                        pos_overlap=pos_overlap,
                        strain_penalty=p_strain,
                        missing_penalty=p_missing,
                        occam_penalty=complexity_penalty,
                        strong_line_penalty=p_strong,
                        diagnostic_flags=flags,
                        matched_lines_detail=matched_details,
                        volume=struct_vol,
                        cell=candidate_cell,
                        space_group=candidate_space_group,
                        explained_peaks_count=ml_count,
                        explained_2thetas=explained_2thetas,
                        observed_peaks_total=observed_total,
                        observed_peaks_explained=observed_explained,
                        observed_peak_coverage=observed_coverage,
                        coverage_penalty=coverage_penalty,
                        raw_score=total_score,
                    )
                )

        # ---- Phase 2: sequential residual pass --------------------------
        # Only genuinely accepted phases (non-vetoed, explaining >= 2 unique peaks with
        # post-residual score >= 15%) contribute their peaks to the covering set.
        results.sort(key=lambda x: x.score, reverse=True)
        if len(results) > 1:
            covering: list[float] = []
            for r in results:
                has_veto = any("VETO" in f for f in r.diagnostic_flags)
                if covering:
                    last_idx = len(two_theta) - 1
                    matched_idx = [
                        min(int(np.searchsorted(two_theta, p)), last_idx)
                        for p in r.explained_2thetas
                    ]
                    unique_count = count_unique_matched_peaks(
                        matched_exp_indices=matched_idx,
                        exp_two_theta=two_theta,
                        covering_2thetas=np.asarray(covering, dtype=float),
                        angular_tolerance=self.angular_tolerance,
                    )
                    residual_factor = unique_count / r.matched_lines if r.matched_lines > 0 else 1.0
                    r.score = float(np.clip(r.score * residual_factor, 0.0, 100.0))
                    r.diagnostic_flags.append(f"RESIDUAL_UNIQUE ({unique_count}/{r.matched_lines})")
                    if not has_veto and unique_count >= 2 and r.score >= 15.0:
                        covering.extend(r.explained_2thetas)
                elif not has_veto and r.score >= 15.0:
                    covering.extend(r.explained_2thetas)

        # ---- Phase 3: isostructural ambiguity note -----------------------
        # Two candidates of one structure type at different cell constants are a
        # solid-solution pair, not a strong and a weak phase. Ranking them apart
        # hides the fact that the observed cell lies between them, which is how a
        # trace member of such a series is dismissed and its reflections are then
        # hunted as an unknown impurity.
        by_space_group: dict[str, list[CandidateScoringResult]] = {}
        for scored in results:
            if scored.space_group:
                by_space_group.setdefault(scored.space_group, []).append(scored)
        for space_group, group in by_space_group.items():
            if len(group) < 2:
                continue
            for i, first in enumerate(group):
                for other in group[i + 1 :]:
                    # The gate is line completeness, not score. The residual pass
                    # strips the second member of the pair of every peak the first
                    # one already explains, so its score goes to zero by
                    # construction -- a score-based gate could never fire in the
                    # very case it was written for.
                    if min(first.completeness_ml_tl, other.completeness_ml_tl) < 0.5:
                        continue
                    if max(first.score, other.score) < 35.0:
                        continue
                    delta_da = abs(other.lattice_shift_da_a - first.lattice_shift_da_a) * 100.0
                    if delta_da < 0.05:
                        continue
                    note = (
                        f"ISOSTRUCTURAL_AMBIGUITY ({first.phase_name} vs "
                        f"{other.phase_name}, {space_group}, required da/a differ by "
                        f"{delta_da:.2f}%)"
                    )
                    first.diagnostic_flags.append(note)
                    other.diagnostic_flags.append(note)

        results.sort(key=lambda x: x.score, reverse=True)
        return results


def _short_formula(text: str, limit: int = 22) -> str:
    """Trim a formula for the text table; the JSON payload keeps it in full."""
    text = text or "-"
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Rules, header and data rows all share this width: nine column widths and
# nine " | " separators, plus the open-ended flags column.
_TABLE_FIXED_WIDTH = 4 + 24 + 22 + 6 + 9 + 8 + 12 + 8 + 6 + 3 * 9


def format_candidate_table(results: list[CandidateScoringResult]) -> str:
    """Format candidate evaluation results into a clean aligned ASCII report table."""
    lines: list[str] = []
    flag_cells = ["; ".join(r.diagnostic_flags) for r in results]
    # The flags column is the only open-ended one, so the width is taken from the
    # content. A hard-coded rule drifts as soon as a flag grows: the closing column
    # then floats and the table stops being a table.
    flags_width = max([len("Diagnostic Flags"), *(len(cell) for cell in flag_cells)])
    width = _TABLE_FIXED_WIDTH + flags_width
    header_sep = "=" * width
    col_sep = "-" * width

    lines.append(header_sep)
    lines.append(
        f"CANDIDATE RANKING LIST (Analytical Profile Scoring, {len(results)} candidate(s) evaluated):"
    )
    lines.append(header_sep)
    lines.append(
        f"{'Rank':<4} | {'Candidate / CIF':<24} | {'Formula':<22} | {'Score':<6} | "
        f"{'ML / TL':<9} | {'IntMatch':<8} | {'Shift (da/a)':<12} | {'Scale':<8} | "
        f"{'Cov':<6} | {'Diagnostic Flags':<{flags_width}}"
    )
    lines.append(col_sep)

    for rank, r in enumerate(results, start=1):
        da_pct = f"{r.lattice_shift_da_a * 100:+.2f}%"
        ml_tl_str = f"{r.matched_lines} / {r.total_lines}"
        cov_str = f"{r.observed_peak_coverage:.2f}"
        lines.append(
            f"{rank:<4} | {r.phase_name:<24} | {_short_formula(r.formula):<22} | "
            f"{r.score:>5.1f}% | {ml_tl_str:<9} | {r.intensity_match:>7.1f}% | "
            f"{da_pct:>12} | {r.scale_factor:>8.4f} | {cov_str:>6} | "
            f"{flag_cells[rank - 1]:<{flags_width}}"
        )

    lines.append(header_sep)
    return "\n".join(lines)


def build_element_filter(
    all_elements: str | None = None,
    any_elements: str | None = None,
    none_elements: str | None = None,
    strict: bool = False,
) -> ElementFilter | None:
    """Build the element filter from raw flag values, or ``None`` when nothing was asked.

    ``None`` means "do not narrow the pool". An absent filter is an optional
    restriction and never a default narrowing: the candidate pool has to cover every
    thermodynamically plausible product of the synthesis, container and atmosphere
    (SKILL.md, Principle 11), so the narrowing happens only when a caller asks for it.

    :raises ValueError: on a contradictory or empty criterion; the message comes from
        ``ElementFilter``, which owns these semantics.
    """
    parts = [part.strip() for part in (any_elements or "").split(",") if part.strip()]
    if not (all_elements or parts or none_elements or strict):
        return None
    return ElementFilter.from_raw(
        all_elements=all_elements,
        any_elements=",".join(parts) if parts else None,
        none_elements=none_elements,
        strict=strict,
    )


def select_candidates(
    candidates: dict[str, Structure | str | Path | CandidateLines],
    elem_filter: ElementFilter | None,
) -> dict[str, Structure | str | Path | CandidateLines]:
    """Apply the element filter, keeping candidates whose composition is unreadable.

    Extraction rules stay in ``extract_elements`` and filter semantics stay in
    ``ElementFilter``; this function only decides what to do when the criterion
    cannot be evaluated at all. Such a candidate is kept: dropping it would narrow
    the pool on a technicality rather than on chemistry.
    """
    if elem_filter is None:
        return dict(candidates)

    elems_map: dict[str, set[str]] = {}
    unverifiable: list[str] = []
    for name, src in candidates.items():
        try:
            if isinstance(src, (str, Path)):
                elems_map[name] = extract_elements(
                    Structure.from_file(str(src)).composition.formula
                )
            elif isinstance(src, Structure):
                elems_map[name] = extract_elements(src.composition.formula)
            else:
                elems_map[name] = extract_elements(getattr(src, "formula", ""))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Cannot read elements of {}; keeping it in the pool: {}", name, exc)
            unverifiable.append(name)

    kept = select_by_elements(elems_map, elem_filter) + unverifiable
    dropped = len(candidates) - len(kept)
    if dropped:
        logger.info("Element filter removed {} of {} candidates", dropped, len(candidates))
    return {name: candidates[name] for name in kept}


def query_lines_from_pattern(
    two_theta: NDArray[Any],
    intensity: NDArray[Any],
    count: int = 3,
    tolerance_deg: float = 0.20,
) -> NDArray[Any]:
    """Return the positions of the strongest observed peaks, for the first pass.

    Three lines and not five: on a card self-query, widening the query to the five
    strongest lines *increases* the number of survivors (1384 against 4276 at two hits),
    because a weaker line is easier to match by chance while the hit requirement stays the
    same.

    Peaks are taken above a fitted baseline, so the query describes the pattern rather than
    its background, and they must lie at least ``tolerance_deg`` apart so that one broad
    maximum does not enter the query three times.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if two_theta.size == 0:
        return np.array([], dtype=float)

    baseline = fit_physical_background(two_theta, intensity, n_terms=6)
    net = np.maximum(0.0, intensity - baseline)
    step = float(np.mean(np.diff(two_theta))) if two_theta.size > 1 else 0.02
    distance = max(1, round(tolerance_deg / max(step, 1e-9)))
    peaks, _ = find_peaks(net, distance=distance)  # noqa: PU001
    if peaks.size == 0:
        return np.array([], dtype=float)
    if peaks.size > count:
        strongest = np.argsort(net[peaks])[::-1][:count]
        peaks = peaks[strongest]
    return np.asarray(np.sort(two_theta[peaks]), dtype=float)


def load_pdf2_candidates(
    elem_filter: ElementFilter,
    query_two_theta: NDArray[Any] | None = None,
    tolerance_deg: float = 0.20,
    min_hits: int = 2,
    wavelength: float = CANONICAL_WAVELENGTH_KA1,
    allowed_classes: frozenset[str] | None = None,
    max_candidates: int | None = None,
    pearson_symbols: frozenset[str] | None = None,
    subfiles: frozenset[str] | None = None,
) -> dict[str, CandidateLines]:
    """Return the reference card pool as candidate line lists, narrowed by the filters.

    The installation location is owned by ``pdf2_reader`` (``PDF2_DATA_DIR``, or the
    ``.env`` of the working directory) and is never resolved here.

    The filter is required, not optional, and is applied while the records are still
    cards: materialising the entire reference database as candidate line lists before
    filtering would be prohibitively expensive, and early reduction is what keeps a
    large card pool tractable.

    Two narrowings run here, and their order is dictated by cost. The line test is answered
    from the shipped strongest-lines file without touching a card, so it runs first and
    only the cards it leaves are read -- otherwise the first pass would cost one seek per
    card of the database, which is exactly what makes a full pass unusable. The element
    test stays at card level, where the full formula is available: the formula in the
    strongest-lines file is truncated, so deciding chemistry there could drop a card whose
    truncated formula happens to miss a required element.

    Measured on the shipped database: the line test alone leaves a median of 343 cards and
    up to 7752, so it narrows the reading but does not shrink the pool to a working size on
    its own. That is why the element criterion is mandatory in the CLI rather than
    optional, and why ``max_candidates`` exists.

    With ``query_two_theta`` empty or omitted the prefilter is skipped and the whole
    element-filtered database is returned. That is kept for a caller that has no pattern to
    query with, and it is the slow path by design.

    ``pearson_symbols`` and ``subfiles`` are the remaining two card-level criteria of the
    reference database: the cell, the space group and the crystal system were already
    available, the Pearson symbol only through its crystal system, and the subfiles not at
    all. Both are compared where the card is complete, for the same reason as the element
    test.

    Three outcomes are distinguished: no installation gives an empty pool with an
    explanatory note, since an absent optional database is not a failure of the
    measurement; an installation that exists but cannot be read raises, because
    silence there would look like "no card matched" when nothing was ever read; a
    usable installation gives every non-deleted card that satisfies the filter and
    carries at least one reflection.
    """
    try:
        from pdf2_reader import PDF2Database, PDF2NotFoundError, StrongestLinesIndex
    except ImportError:
        return {}

    pool: dict[str, CandidateLines] = {}
    read = 0
    first_pass: int | None = None
    try:
        with PDF2Database() as db:
            positions: list[int] | None = None
            if query_two_theta is not None and len(query_two_theta) > 0:
                index = StrongestLinesIndex(db.db_dir)
                positions = index.survivors(
                    query_two_theta,
                    lambda d: bragg_two_theta(d, wavelength),
                    tolerance_deg=tolerance_deg,
                    min_hits=min_hits,
                    allowed_classes=allowed_classes,
                    max_candidates=max_candidates,
                )
                # The two files are shipped together and were measured to agree, but they
                # are separate files: a position past the end of the card stream is named
                # rather than left to raise from inside the reader.
                beyond = [position for position in positions if position >= db.card_count]
                if beyond:
                    logger.warning(
                        "Strongest-lines file names {} card(s) beyond the card stream "
                        "({} cards); they are dropped",
                        len(beyond),
                        db.card_count,
                    )
                    positions = [position for position in positions if position < db.card_count]
                first_pass = len(positions)
                logger.info(
                    "PDF-2 first pass: {} of {} cards keep >= {} of 3 strongest lines "
                    "within {:.2f} deg of the measured peaks",
                    first_pass,
                    db.card_count,
                    min_hits,
                    tolerance_deg,
                )

            card_filter = None
            if pearson_symbols is not None or subfiles is not None:
                from pdf2_reader import CrystallographicFilter

                card_filter = CrystallographicFilter(
                    pearson_symbols=pearson_symbols, subfiles=subfiles
                )

            cards = (
                (db.get_card(position) for position in positions)
                if positions is not None
                else db.iter_cards()
            )
            for card in cards:
                read += 1
                if card.is_deleted:
                    continue
                if allowed_classes is not None and card.quality_class not in allowed_classes:
                    continue
                if card_filter is not None and not card_filter.matches(card):
                    continue
                formula = card.structural_formula or card.empirical_formula
                if not elem_filter.matches(extract_elements(formula)):
                    continue
                d_vals = np.asarray([refl.d_spacing for refl in card.reflections], dtype=float)
                if d_vals.size == 0:
                    continue
                i_vals = np.asarray([refl.intensity for refl in card.reflections], dtype=float)
                pool[card.pdf_number] = CandidateLines(
                    label=card.pdf_number,
                    formula=formula,
                    volume=float(card.cell_volume or 0.0),
                    d_spacing=d_vals,
                    intensity=i_vals,
                    card_cell=(
                        card.cell_a,
                        card.cell_b,
                        card.cell_c,
                        card.cell_alpha,
                        card.cell_beta,
                        card.cell_gamma,
                    ),
                    hkl=tuple(refl.hkl for refl in card.reflections),
                )
    except PDF2NotFoundError as exc:
        logger.info("PDF-2 pool absent: {}", exc)
        return {}
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"cannot read the PDF-2 database: {exc}") from exc

    if first_pass is None:
        logger.info("PDF-2 pool: {} cards kept of {} read (no line first pass)", len(pool), read)
    else:
        logger.info(
            "PDF-2 pool: {} cards kept of {} read after the line first pass left {}",
            len(pool),
            read,
            first_pass,
        )
    return pool


def load_hsrdb_candidates(
    elem_filter: ElementFilter,
    query_two_theta: NDArray[Any] | None = None,
    tolerance_deg: float = 0.20,
    min_hits: int = 2,
    wavelength: float = CANONICAL_WAVELENGTH_KA1,
    max_candidates: int | None = 500,
    subfiles: frozenset[str] | None = None,
) -> dict[str, CandidateLines]:
    """Return candidate line lists from the local COD24_HS4x.hsrdb database (511,583 structures)."""
    try:
        from hs_rdb import HsRdbDatabase, HsRdbNotFoundError
        from pdf2_reader import CrystallographicFilter
    except ImportError:
        return {}

    c_filter = CrystallographicFilter(subfiles=subfiles) if subfiles else None
    pool: dict[str, CandidateLines] = {}
    try:
        with HsRdbDatabase() as db:
            query_arr = (
                np.asarray(query_two_theta, dtype=float)
                if query_two_theta is not None
                else np.array([], dtype=float)
            )
            pids = db.survivors_by_strongest_lines(
                element_filter=elem_filter,
                query_two_theta=query_arr,
                to_two_theta=lambda d: bragg_two_theta(d, wavelength),
                tolerance_deg=tolerance_deg,
                min_hits=min_hits,
                max_candidates=max_candidates,
                crystal_filter=c_filter,
            )
            for pid_val in pids:
                cand = db.get_candidate_lines(pid_val)
                if cand.d_spacing.size > 0:
                    pool[cand.label] = cand
    except HsRdbNotFoundError as exc:
        logger.info("COD HSRDB pool absent: {}", exc)
        return {}
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"cannot read COD HSRDB database: {exc}") from exc

    logger.info("COD HSRDB pool: {} candidate structures loaded", len(pool))
    return pool


def main() -> int:
    """CLI entrypoint for standalone candidate phase search-matching."""
    parser = argparse.ArgumentParser(
        description="Autonomous analytical candidate phase scoring against powder XRD data."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to experimental XY powder pattern file (mutually exclusive with --residual)",
    )
    parser.add_argument(
        "--residual",
        type=str,
        default=None,
        help="Path to a difference curve (2theta, Yobs, Ycalc): either plain columns or "
        "the skill's own profile_curves.csv written by 'xrd_project.py export'; "
        "candidates are then scored against the residual instead of the raw pattern",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Path to a GSAS-II .gpx project; candidates are scored against the signed "
        "difference curve of its refined model, read from the project itself",
    )
    parser.add_argument(
        "--candidate-cifs",
        default=None,
        help="Directory or path to candidate CIF files",
    )
    parser.add_argument(
        "--pdf2",
        action="store_true",
        help="Also score the reference card database (requires an element criterion and a "
        "PDF-2 installation declared through the environment)",
    )
    parser.add_argument(
        "--hsrdb",
        action="store_true",
        help="Also score the local COD HSRDB database (511,583 structures; requires an "
        "element criterion and COD_HSRDB_PATH in .env)",
    )
    parser.add_argument(
        "--pdf2-min-hits",
        type=int,
        default=2,
        help="First-pass requirement: how many of a card's three strongest lines must fall "
        "on a measured peak (default: 2). Requiring all three collapses to zero as soon "
        "as one line is missing.",
    )
    parser.add_argument(
        "--pdf2-class",
        type=str,
        default=None,
        help="Comma-separated card quality classes to admit, for example 'C'. Omitted "
        "admits every value, because the semantics of the class column are only partly "
        "established.",
    )
    parser.add_argument(
        "--pdf2-max-pool",
        type=int,
        default=None,
        help="Upper bound on the cards the first pass hands to scoring; the survivor count "
        "and the bound are both reported when it bites.",
    )
    parser.add_argument(
        "--no-pdf2-prefilter",
        action="store_true",
        help="Skip the strongest-lines first pass and read the whole element-filtered "
        "database. This is the slow path, kept for a caller that must not narrow by lines.",
    )
    parser.add_argument(
        "--pearson",
        type=str,
        default=None,
        help="Comma-separated Pearson symbols to admit, for example 'cF8'. The letters are "
        "compared exactly and an absent count means any, so 'cF' admits every face-centred "
        "cubic cell.",
    )
    parser.add_argument(
        "--subfiles",
        type=str,
        default=None,
        help="Comma-separated reference-database subfile codes to admit; a card is kept when "
        "it belongs to any of them.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=CANONICAL_WAVELENGTH_KA1,
        help="Diffractometer radiation wavelength in Angstroms",
    )
    parser.add_argument(
        "--elements",
        type=str,
        default=None,
        help="AND (all_of): comma-separated elements every candidate must contain",
    )
    parser.add_argument(
        "--any",
        type=str,
        default=None,
        help="OR (any_of): comma-separated elements of which a candidate must contain at least one",
    )
    parser.add_argument(
        "--none",
        type=str,
        default=None,
        help="NOT (none_of): comma-separated elements a candidate must not contain",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Closed chemical system: a candidate may contain no element outside --elements / --any",
    )
    parser.add_argument(
        "--pool-file",
        type=Path,
        default=None,
        help="JSON file mapping tier names to element lists. Required by --analogs / "
        "--contamination. The lists live outside the skill on purpose: the skill "
        "supplies the mechanism, the research script supplies the chemistry.",
    )
    parser.add_argument(
        "--analogs",
        action="store_true",
        help="Also accept candidates containing any element of the 'analogs' tier "
        "from --pool-file (isostructural substitutions).",
    )
    parser.add_argument(
        "--contamination",
        action="store_true",
        help="Also accept candidates containing any element of the 'contamination' "
        "tier from --pool-file (oxidation, sorbed gases, container products).",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(SCORING_PROFILES),
        default="ide-all",
        help="Named scoring profile; explicit flags below override its values (default: ide-all)",
    )
    parser.add_argument(
        "--angular-tolerance",
        type=float,
        default=None,
        help="Matching angular window tolerance in degrees 2theta (default: taken from --profile)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum total score threshold for output inclusion",
    )
    parser.add_argument(
        "--refine-zero",
        action="store_true",
        help="Enable refinement of detector zero-shift (|Z| <= 0.03 deg)",
    )
    parser.add_argument(
        "--strip-ka2",
        action="store_true",
        help="Strip the K-alpha2 companion line before scoring (Rachinger/Ladell recursion)",
    )
    parser.add_argument(
        "--strong-line-threshold",
        type=float,
        default=None,
        help="Relative reference intensity above which a candidate line counts as "
        "strong for DemoteUnmatchedStrong (default: taken from --profile)",
    )
    parser.add_argument(
        "--demote-unmatched-strong",
        type=float,
        default=None,
        help="Weight of the DemoteUnmatchedStrong penalty; 0.0 disables it "
        "(default: taken from --profile)",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=None,
        help="Exponent of the observed-pattern coverage term (fraction of significant "
        "observed peaks a candidate explains); 0.0 removes it from the score while the "
        "coverage is still reported (default: taken from --profile)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON output to stdout"
    )
    parser.add_argument("--out", type=str, default=None, help="Save output JSON/table to file")

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parser.parse_args()

    if args.strip_ka2 and (args.project or args.residual):
        logger.error(
            "--strip-ka2 removes the companion line of a measured doublet and applies "
            "to --data only; a modelled difference curve already carries the "
            "wavelength state of the refinement"
        )
        return 2

    declared = {
        "--data": args.data,
        "--project": args.project,
        "--residual": args.residual,
    }
    chosen = [flag for flag, value in declared.items() if value]
    if len(chosen) != 1:
        logger.error(
            "Pass exactly one of --data (raw pattern), --project (GSAS-II project) or "
            "--residual (difference curve); got {}",
            ", ".join(chosen) or "none",
        )
        return 2

    cif_files: list[Path] = []
    if args.candidate_cifs:
        cif_p = Path(args.candidate_cifs).resolve()
        if cif_p.is_dir():
            cif_files = sorted(cif_p.glob("*.cif"))
        elif cif_p.is_file():
            cif_files = [cif_p]
        else:
            logger.error(f"No candidate CIF files found at: {cif_p}")
            return 1

    # --- Pattern source -------------------------------------------------------
    # A raw pattern is scored against the net signal above its fitted baseline; a
    # refined model is scored against its signed difference curve, which is read
    # either from the GSAS-II project that owns it or from a difference-curve file.
    # The field is built here, in the branch that knows which source it came from,
    # so that no optional value has to be carried further down the function.
    two_theta: NDArray[Any]
    prob_field: ProfileProbabilityField

    if args.data:
        data_p = Path(args.data).resolve()
        if not data_p.exists():
            logger.error(f"Data file not found: {data_p}")
            return 1
        raw_data: NDArray[Any] = np.atleast_2d(np.loadtxt(str(data_p)))
        if raw_data.ndim != 2 or raw_data.shape[1] < 2:
            logger.error(f"{data_p.name} has shape {raw_data.shape}; expected >= 2 columns")
            return 1
        two_theta = raw_data[:, 0]
        intensity = raw_data[:, 1]
        # The observed profile is kept separately from the signal that scoring uses: the
        # first pass over the card database must query the measurement, not a copy of it
        # that a strip step may have rewritten.
        observed_intensity = np.asarray(raw_data[:, 1], dtype=float)
        if args.strip_ka2:
            # The Rachinger/Ladell recursion applies to the LINE profile only: the
            # background is not Ka1/Ka2 paired, so stripping the raw signal would
            # attenuate the baseline (s = y - r*s). Subtract, strip, restore.
            instrument = CanonicalInstrument(wavelength_ka1=args.wavelength)
            baseline = fit_physical_background(two_theta, intensity, n_terms=6)
            intensity = instrument.strip_ka2(two_theta, intensity - baseline) + baseline
        prob_field = ProfileProbabilityField.from_raw_pattern(two_theta, intensity)
    elif args.residual:
        residual_p = Path(args.residual).resolve()
        if not residual_p.exists():
            logger.error(f"Difference-curve file not found: {residual_p}")
            return 1
        try:
            difference = read_difference_curve(residual_p)
        except (OSError, ValueError) as exc:
            logger.error("Cannot read {}: {}", residual_p.name, exc)
            return 1
        two_theta = difference[:, 0]
        observed_intensity = np.asarray(difference[:, 1], dtype=float)
        prob_field = ProfileProbabilityField.from_difference_curve(
            two_theta, difference[:, 1], difference[:, 2]
        )
    else:
        project_p = Path(args.project).resolve()
        if not project_p.exists():
            logger.error(f"Project file not found: {project_p}")
            return 1
        try:
            difference = read_project_difference_curve(project_p)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Cannot read the difference curve from {}: {}", project_p.name, exc)
            return 2
        two_theta = difference[:, 0]
        observed_intensity = np.asarray(difference[:, 1], dtype=float)
        prob_field = ProfileProbabilityField.from_difference_curve(
            two_theta, difference[:, 1], difference[:, 2]
        )

    # Initialize candidate dictionary
    candidates: dict[str, Structure | str | Path | CandidateLines] = {
        cf.stem: cf for cf in cif_files
    }

    # --- Element filter: semantics owned by cif_retriever.ElementFilter -------
    # The tier lists are deliberately NOT hard-coded here (SKILL.md, Principle 8,
    # Inversion of Control): --pool-file supplies them from outside the skill. The
    # filter is built before the reference pool, because that pool is narrowed by it
    # while the records are still cards.
    any_parts = [p.strip() for p in (args.any or "").split(",") if p.strip()]
    if args.analogs or args.contamination:
        if args.pool_file is None:
            logger.error("--analogs / --contamination require --pool-file")
            return 2
        try:
            pools = json.loads(Path(args.pool_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Cannot read --pool-file {}: {}", args.pool_file, exc)
            return 2
        for tier, wanted in (("analogs", args.analogs), ("contamination", args.contamination)):
            if not wanted:
                continue
            tier_elements = pools.get(tier)
            if not tier_elements:
                logger.error("--pool-file has no non-empty tier '{}'", tier)
                return 2
            any_parts.extend(str(e).strip() for e in tier_elements if str(e).strip())

    try:
        elem_filter = build_element_filter(
            all_elements=args.elements,
            any_elements=",".join(any_parts) if any_parts else None,
            none_elements=args.none,
            strict=args.strict,
        )
    except ValueError as exc:
        logger.error("Invalid element filter: {}", exc)
        return 2

    # --- PDF-2 pool -----------------------------------------------------------
    # Requested explicitly so routine calls do not scan the entire reference card
    # database unless asked. The location still comes only from the environment.
    if args.pdf2:
        if elem_filter is None:
            logger.error(
                "--pdf2 scores the reference card database; give an "
                "element criterion (--elements / --any / --none) so the pool is narrowed "
                "while the records are still cards"
            )
            return 2

        allowed_classes = (
            frozenset(part.strip() for part in args.pdf2_class.split(",") if part.strip())
            if args.pdf2_class
            else None
        )
        tolerance = resolve_angular_tolerance(
            SCORING_PROFILES[args.profile], args.angular_tolerance
        )
        query: NDArray[Any] = (
            np.array([], dtype=float)
            if args.no_pdf2_prefilter
            else query_lines_from_pattern(
                two_theta, observed_intensity, count=3, tolerance_deg=tolerance
            )
        )
        if query.size:
            logger.info(
                "PDF-2 first pass will query the {} strongest observed peaks at {}",
                query.size,
                ", ".join(f"{value:.2f}" for value in query),
            )
        else:
            logger.info("PDF-2 first pass disabled; the whole filtered database is read")

        try:
            candidates.update(
                load_pdf2_candidates(
                    elem_filter,
                    query_two_theta=query,
                    tolerance_deg=tolerance,
                    min_hits=args.pdf2_min_hits,
                    wavelength=args.wavelength,
                    allowed_classes=allowed_classes,
                    max_candidates=args.pdf2_max_pool,
                    pearson_symbols=_split_criteria(args.pearson),
                    subfiles=_split_criteria(args.subfiles),
                )
            )
        except RuntimeError as exc:
            logger.error("{}", exc)
            return 2

    if args.hsrdb:
        if elem_filter is None:
            logger.error(
                "--hsrdb scores the COD HSRDB database (511,583 structures); give an "
                "element criterion (--elements / --any / --none)"
            )
            return 2
        tolerance = resolve_angular_tolerance(
            SCORING_PROFILES[args.profile], args.angular_tolerance
        )
        query_hs: NDArray[Any] = (
            np.array([], dtype=float)
            if args.no_pdf2_prefilter
            else query_lines_from_pattern(
                two_theta, observed_intensity, count=3, tolerance_deg=tolerance
            )
        )
        try:
            candidates.update(
                load_hsrdb_candidates(
                    elem_filter,
                    query_two_theta=query_hs,
                    tolerance_deg=tolerance,
                    min_hits=args.pdf2_min_hits,
                    wavelength=args.wavelength,
                    max_candidates=args.pdf2_max_pool or 500,
                    subfiles=_split_criteria(args.subfiles),
                )
            )
        except RuntimeError as exc:
            logger.error("{}", exc)
            return 2

    candidates = select_candidates(candidates, elem_filter)
    if not candidates:
        logger.error(
            "No candidates: pass --candidate-cifs, --pdf2, and/or --hsrdb with an element "
            "criterion"
        )
        return 2

    profile = dict(SCORING_PROFILES[args.profile])
    engine = CandidateScoringEngine(
        wavelength=args.wavelength,
        angular_tolerance=(
            args.angular_tolerance
            if args.angular_tolerance is not None
            else profile["angular_tolerance"]
        ),
        min_line_intensity=profile["min_line_intensity"],
        max_delta_a=profile["max_delta_a"],
        max_zero_shift=profile["max_zero_shift"],
        strong_line_threshold=(
            args.strong_line_threshold
            if args.strong_line_threshold is not None
            else profile["strong_line_threshold"]
        ),
        demote_unmatched_strong=(
            args.demote_unmatched_strong
            if args.demote_unmatched_strong is not None
            else profile["demote_unmatched_strong"]
        ),
        coverage_weight=(
            args.coverage_weight if args.coverage_weight is not None else profile["coverage_weight"]
        ),
    )

    results = engine.score_candidates(
        two_theta=two_theta,
        y_net=prob_field.y_net,
        sigma=prob_field.sigma,
        candidate_structures=candidates,
        refine_zero=args.refine_zero,
        min_score=args.min_score,
        observed_signal=observed_intensity,
    )

    table_output = format_candidate_table(results)
    json_output = [r.to_dict() for r in results]

    if args.json:
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
    else:
        print(table_output)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            if out_path.suffix.lower() == ".json":
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            else:
                f.write(table_output + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
