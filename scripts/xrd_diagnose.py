"""Automated crystallographic residual diagnostic engine (Yobs - Ycalc).

Detects:
1. Unindexed impurity peaks (Delta I > 4 * sigma_bg).
2. Derivative (+/-) residuals (unit cell parameter mismatch or zero-shift error).
3. W-shaped (+/- /+) profile shape errors (pseudo-Voigt U, V, W or eta mixing).
4. Systematically suppressed/enhanced families indicating preferred orientation.

Outputs machine-readable JSON and human/LLM-readable clinical diagnostic reports
based on references/rietveld-troubleshooting.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from constants import CANONICAL_WAVELENGTH_KBETA, DEFAULT_WAVELENGTH_KA1
from gsas2_adapter import read_difference_curve, read_project_difference_curve
from loguru import logger
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_widths
from scipy.stats import norm


class DiagnosticStatus(StrEnum):
    """Overall status of residual diagnostics."""

    PASS = "PASS"
    NEEDS_REFINEMENT = "NEEDS_REFINEMENT"
    FAIL = "FAIL"


class SymptomSeverity(StrEnum):
    """Clinical severity level of a residual symptom."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SymptomCode(StrEnum):
    """Canonical residual symptom identifiers."""

    UNINDEXED_PHASE = "SYM_UNINDEXED_PHASE"
    INTENSITY_MISMATCH = "SYM_INTENSITY_MISMATCH"
    DERIVATIVE_RESIDUAL = "SYM_DERIVATIVE_RESIDUAL"
    PROFILE_SHAPE_MISMATCH = "SYM_PROFILE_SHAPE_MISMATCH"
    DURBIN_WATSON_AUTOCORRELATION = "SYM_DURBIN_WATSON_AUTOCORRELATION"
    RWP_RP_DISCREPANCY = "SYM_RWP_RP_DISCREPANCY"
    BACKGROUND_WIGGLE = "SYM_BACKGROUND_WIGGLE"


class ResidualPeakType(StrEnum):
    """Classification of a peak anomaly on the difference curve."""

    TRUE_UNINDEXED_IMPURITY = "TRUE_UNINDEXED_IMPURITY"
    SHIFTED_KNOWN_LINE = "SHIFTED_KNOWN_LINE"
    INTENSITY_DEFECT = "INTENSITY_DEFECT (texture/occupancy/coordinates)"
    WEAK_KNOWN_LINE = "WEAK_KNOWN_LINE (reflection present, model height below noise floor)"
    OVER_CALCULATED_PHANTOM = (
        "OVER_CALCULATED_PHANTOM (Y_calc >> Y_obs, absent or suppressed reflection)"
    )
    KBETA_SATELLITE = (
        "KBETA_SATELLITE (residual Cu K-beta companion of a strong Bragg reflection)"
    )
    RESIDUAL_ARTIFACT = "RESIDUAL_ARTIFACT (no observed peak at this angle)"


@dataclass(frozen=True)
class ResidualPeak:
    """Typed descriptor of a classified residual peak on Yobs - Ycalc."""

    two_theta: float
    d_spacing_angstrom: float
    residual_height: float
    significance_sigma: float
    peak_type: ResidualPeakType
    matched_phase: str | None = None
    bragg_2theta: float | None = None
    delta_2theta: float | None = None

    @property
    def defect_type(self) -> ResidualPeakType:
        return self.peak_type

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "2theta": self.two_theta,
            "d_spacing_angstrom": self.d_spacing_angstrom,
            "residual_height": self.residual_height,
            "significance_sigma": self.significance_sigma,
        }
        if self.matched_phase is not None:
            payload["matched_phase"] = self.matched_phase
        if self.bragg_2theta is not None:
            payload["bragg_2theta"] = self.bragg_2theta
        if self.delta_2theta is not None:
            payload["delta_2theta"] = self.delta_2theta
        if self.peak_type in (
            ResidualPeakType.INTENSITY_DEFECT,
            ResidualPeakType.WEAK_KNOWN_LINE,
            ResidualPeakType.OVER_CALCULATED_PHANTOM,
        ):
            payload["defect_type"] = str(self.peak_type)
        else:
            payload["type"] = str(self.peak_type)
        return payload


@dataclass(frozen=True)
class PhaseBraggMetrics:
    """Phase-specific Bragg R_B and Structure Factor R_F metrics."""

    r_bragg_percent: float
    r_f_percent: float
    integrated_obs_intensity: float = 0.0
    reflections_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "r_bragg_percent": self.r_bragg_percent,
            "r_f_percent": self.r_f_percent,
            "integrated_obs_intensity": self.integrated_obs_intensity,
            "reflections_count": self.reflections_count,
        }


@dataclass(frozen=True)
class MicrostructureMetrics:
    """Physical crystallite size and microstrain parameters."""

    crystallite_size_nm: float | None = None
    microstrain_percent: float | None = None
    x_lorentzian_deg: float | None = None
    y_lorentzian_deg: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "crystallite_size_nm": self.crystallite_size_nm,
            "microstrain_percent": self.microstrain_percent,
            "x_lorentzian_deg": self.x_lorentzian_deg,
            "y_lorentzian_deg": self.y_lorentzian_deg,
        }


@dataclass(frozen=True)
class DiagnosticSymptom:
    """Clinical crystallographic symptom detected in the residual profile."""

    code: SymptomCode
    severity: SymptomSeverity
    name: str
    description: str
    details: list[ResidualPeak] = field(default_factory=list)
    inflections: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": str(self.code),
            "severity": str(self.severity),
            "name": self.name,
            "description": self.description,
        }
        if self.details:
            payload["details"] = [d.to_dict() for d in self.details]
        if self.inflections is not None:
            payload["inflections"] = self.inflections
        return payload


@dataclass(frozen=True)
class ResidualDiagnosisReport:
    """Full typed diagnostic report of a Rietveld difference curve."""

    overall_status: DiagnosticStatus
    r_p_percent: float
    r_wp_percent: float
    r_wp_over_r_p_ratio: float
    durbin_watson: float
    noise_sigma: float
    i_max_net: float
    lod_3sigma_percent: float
    true_unindexed_count: int
    residual_artifact_count: int
    residual_artifacts: list[ResidualPeak]
    intensity_defects_count: int
    true_unindexed_reflections: list[ResidualPeak]
    intensity_mismatch_reflections: list[ResidualPeak]
    phase_r_factors: dict[str, PhaseBraggMetrics]
    total_symptoms: int
    symptoms: list[DiagnosticSymptom]
    recommended_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": str(self.overall_status),
            "r_p_percent": self.r_p_percent,
            "r_wp_percent": self.r_wp_percent,
            "r_wp_over_r_p_ratio": self.r_wp_over_r_p_ratio,
            "durbin_watson": self.durbin_watson,
            "noise_sigma": self.noise_sigma,
            "i_max_net": self.i_max_net,
            "lod_3sigma_percent": self.lod_3sigma_percent,
            "true_unindexed_count": self.true_unindexed_count,
            "residual_artifact_count": self.residual_artifact_count,
            "residual_artifacts": [p.to_dict() for p in self.residual_artifacts],
            "intensity_defects_count": self.intensity_defects_count,
            "true_unindexed_reflections": [p.to_dict() for p in self.true_unindexed_reflections],
            "intensity_mismatch_reflections": [
                p.to_dict() for p in self.intensity_mismatch_reflections
            ],
            "phase_r_factors": {k: v.to_dict() for k, v in self.phase_r_factors.items()},
            "total_symptoms": self.total_symptoms,
            "symptoms": [s.to_dict() for s in self.symptoms],
            "recommended_actions": list(self.recommended_actions),
        }


def classify_residual_peaks(
    two_theta: NDArray[Any],
    y_obs: NDArray[Any],
    y_calc: NDArray[Any],
    phase_bragg_peaks: dict[str, list[float]]
    | dict[str, list[tuple[float, float]]]
    | list[float]
    | None = None,
    angular_tolerance: float = 0.20,
    snr_threshold: float = 3.0,
    wavelength: float = DEFAULT_WAVELENGTH_KA1,
    min_peak_snr: float = 3.0,
    observed_peaks_2theta: NDArray[Any] | None = None,
) -> tuple[list[ResidualPeak], list[ResidualPeak], list[ResidualPeak]]:
    """Разделяет пики разностной кривой Delta Y на две непересекающиеся категории:

    1. true_unindexed_peaks: в радиусе +-angular_tolerance deg нет ни одного брэгговского рефлекса
       ни одной из включенных фаз. Это реальные кандидаты в неизвестные примеси.
    2. intensity_profile_defects: пик совпадает с положением рефлекса известной фазы.
       Это НЕ посторонняя фаза, а невязка модели (координаты, B_iso, текстура, вакансии).
    """
    y_diff = y_obs - y_calc
    step_size = float(np.mean(np.diff(two_theta))) if len(two_theta) > 1 else 0.02

    # Шум оценивается по второй разности остатка: это поточечный разброс, который
    # не раздувается широкой невязкой модели и не принимает одиночный выброс за пик.
    second = np.diff(y_diff, n=2)
    sigma_bg = float(np.median(np.abs(second)) / 0.6745 / np.sqrt(6.0))
    sigma_bg = max(sigma_bg, 1e-6)

    # Порог с контролем ложных срабатываний: при N точках ожидаемое число
    # превышений 3σ равно ~2.8, поэтому «критический» пик появлялся почти на
    # каждом образце. z_crit — квантиль для уровня 1 % на весь профиль.
    n_points = max(len(y_diff), 2)
    z_crit = float(norm.isf(0.01 / n_points))
    z_crit = max(float(snr_threshold), z_crit)

    min_dist_pts = max(3, int(0.2 / max(step_size, 1e-4)))
    positive_peaks, _ = find_peaks(  # noqa: PU001
        y_diff,
        height=z_crit * sigma_bg,
        prominence=max(1e-6, (z_crit - 1.0) * sigma_bg),
        distance=min_dist_pts,
    )
    # Настоящее отражение имеет ширину порядка инструментальной линии; одиночный
    # канал выше порога — выброс, а не пик. Отсекаем всё уже двух точек.
    if len(positive_peaks) > 0:
        widths = peak_widths(y_diff, positive_peaks, rel_height=0.5)[0]
        positive_peaks = positive_peaks[widths >= 2.0]

    phases_dict: dict[str, list[Any]] = {}
    if phase_bragg_peaks is not None:
        if isinstance(phase_bragg_peaks, dict):
            phases_dict = phase_bragg_peaks
        elif isinstance(phase_bragg_peaks, (list, tuple, np.ndarray)):
            phases_dict = {"Identified_Phase": list(phase_bragg_peaks)}

    true_unindexed: list[ResidualPeak] = []
    intensity_defects: list[ResidualPeak] = []
    residual_artifacts: list[ResidualPeak] = []
    observed_grid = (
        np.sort(np.asarray(observed_peaks_2theta, dtype=float))
        if observed_peaks_2theta is not None and len(observed_peaks_2theta)
        else None
    )

    def _has_observed_peak(angle: float) -> bool:
        if observed_grid is None:
            return True
        nearest = float(observed_grid[np.argmin(np.abs(observed_grid - angle))])
        return abs(nearest - angle) <= angular_tolerance

    # Предварительный расчёт положений Cu K-beta спутников от сильных брэгговских линий
    # (Правило 13 SKILL.md, категория 2; rietveld-troubleshooting.md §9):
    # sin(theta_Kb) = (lambda_Kb / lambda_Ka1) * sin(theta_Ka1), интенсивность <= 3% от главного пика.
    kbeta_candidates: list[tuple[float, str, float, float]] = []
    b_height: float | None = None
    max_calc_span = max(float(np.max(y_calc) - np.min(y_calc)) if len(y_calc) > 0 else 1.0, 1.0)
    for ph_name, peaks in phases_dict.items():
        for peak_entry in peaks:
            b_peak = float(peak_entry[0] if isinstance(peak_entry, (list, tuple)) else peak_entry)
            if isinstance(peak_entry, (list, tuple)) and len(peak_entry) > 1:
                b_height = float(peak_entry[1])
            else:
                b_idx = int(np.clip(np.argmin(np.abs(two_theta - b_peak)), 0, len(two_theta) - 1))
                b_height = float(max(0.0, y_calc[b_idx] - float(np.min(y_calc))))
            if b_height >= 0.15 * max_calc_span:
                sin_kb = (CANONICAL_WAVELENGTH_KBETA / max(wavelength, 1e-6)) * float(
                    np.sin(np.radians(b_peak / 2.0))
                )
                if 0.0 < sin_kb < 1.0:
                    kb_2th = float(np.degrees(2.0 * np.arcsin(sin_kb)))
                    kbeta_candidates.append((kb_2th, ph_name, b_peak, b_height))

    for p_idx in positive_peaks:
        t_val = float(two_theta[p_idx])
        theta_rad = np.radians(t_val / 2.0)
        d_val = float(wavelength / (2.0 * np.sin(theta_rad))) if np.sin(theta_rad) > 1e-4 else 0.0
        peak_h = float(y_diff[p_idx])
        sig = float(peak_h / sigma_bg)

        nearest_any: tuple[float, str, float, float | None] | None = None
        for ph_name, peaks in phases_dict.items():
            for peak_entry in peaks:
                b_peak = float(
                    peak_entry[0] if isinstance(peak_entry, (list, tuple)) else peak_entry
                )
                b_height = (
                    float(peak_entry[1])
                    if isinstance(peak_entry, (list, tuple)) and len(peak_entry) > 1
                    else None
                )
                dist = abs(b_peak - t_val)
                if nearest_any is None or dist < nearest_any[0]:
                    nearest_any = (dist, ph_name, b_peak, b_height)

        if (
            nearest_any is not None
            and nearest_any[0] <= angular_tolerance
            and nearest_any[3] is not None
            and nearest_any[3] < min_peak_snr * sigma_bg
        ):
            intensity_defects.append(
                ResidualPeak(
                    two_theta=round(t_val, 3),
                    d_spacing_angstrom=round(d_val, 4),
                    residual_height=round(peak_h, 1),
                    significance_sigma=round(sig, 1),
                    peak_type=ResidualPeakType.WEAK_KNOWN_LINE,
                    matched_phase=nearest_any[1],
                    bragg_2theta=round(nearest_any[2], 3),
                    delta_2theta=round(t_val - nearest_any[2], 3),
                )
            )
            continue

        if nearest_any is not None and nearest_any[0] <= angular_tolerance:
            intensity_defects.append(
                ResidualPeak(
                    two_theta=round(t_val, 3),
                    d_spacing_angstrom=round(d_val, 4),
                    residual_height=round(peak_h, 1),
                    significance_sigma=round(sig, 1),
                    peak_type=ResidualPeakType.INTENSITY_DEFECT,
                    matched_phase=nearest_any[1],
                    bragg_2theta=round(nearest_any[2], 3),
                    delta_2theta=round(t_val - nearest_any[2], 3),
                )
            )
            continue

        kbeta_match: tuple[float, str, float] | None = None
        for kb_2th, ph_name, main_2th, main_h in kbeta_candidates:
            dist_kb = abs(t_val - kb_2th)
            if (
                dist_kb <= angular_tolerance
                and peak_h <= 0.03 * main_h
                and (kbeta_match is None or dist_kb < kbeta_match[0])
            ):
                kbeta_match = (dist_kb, ph_name, main_2th)

        if kbeta_match is not None:
            residual_artifacts.append(
                ResidualPeak(
                    two_theta=round(t_val, 3),
                    d_spacing_angstrom=round(d_val, 4),
                    residual_height=round(peak_h, 1),
                    significance_sigma=round(sig, 1),
                    peak_type=ResidualPeakType.KBETA_SATELLITE,
                    matched_phase=kbeta_match[1],
                    bragg_2theta=round(kbeta_match[2], 3),
                    delta_2theta=round(t_val - kbeta_match[2], 3),
                )
            )
        elif not _has_observed_peak(t_val):
            residual_artifacts.append(
                ResidualPeak(
                    two_theta=round(t_val, 3),
                    d_spacing_angstrom=round(d_val, 4),
                    residual_height=round(peak_h, 1),
                    significance_sigma=round(sig, 1),
                    peak_type=ResidualPeakType.RESIDUAL_ARTIFACT,
                )
            )
        elif nearest_any is not None and angular_tolerance < nearest_any[0] <= 0.5:
            true_unindexed.append(
                ResidualPeak(
                    two_theta=round(t_val, 3),
                    d_spacing_angstrom=round(d_val, 4),
                    residual_height=round(peak_h, 1),
                    significance_sigma=round(sig, 1),
                    peak_type=ResidualPeakType.SHIFTED_KNOWN_LINE,
                    matched_phase=nearest_any[1],
                    bragg_2theta=round(nearest_any[2], 3),
                    delta_2theta=round(t_val - nearest_any[2], 3),
                )
            )
        else:
            true_unindexed.append(
                ResidualPeak(
                    two_theta=round(t_val, 3),
                    d_spacing_angstrom=round(d_val, 4),
                    residual_height=round(peak_h, 1),
                    significance_sigma=round(sig, 1),
                    peak_type=ResidualPeakType.TRUE_UNINDEXED_IMPURITY,
                )
            )

    # Detect severe negative residual spikes (Y_calc >> Y_obs: phantom or over-calculated Bragg lines)
    negative_peaks, _ = find_peaks(  # noqa: PU001
        -y_diff,
        height=z_crit * sigma_bg,
        prominence=max(1e-6, (z_crit - 1.0) * sigma_bg),
        distance=min_dist_pts,
    )
    if len(negative_peaks) > 0:
        neg_widths = peak_widths(-y_diff, negative_peaks, rel_height=0.5)[0]
        negative_peaks = negative_peaks[neg_widths >= 2.0]

    for n_idx in negative_peaks:
        t_val = float(two_theta[n_idx])
        theta_rad = np.radians(t_val / 2.0)
        d_val = float(wavelength / (2.0 * np.sin(theta_rad))) if np.sin(theta_rad) > 1e-4 else 0.0
        peak_h = float(y_diff[n_idx])  # negative value (Y_obs - Y_calc < 0)
        sig = float(abs(peak_h) / sigma_bg)

        nearest_bragg: tuple[float, str, float] | None = None
        for ph_name, peaks in phases_dict.items():
            for peak_entry in peaks:
                b_peak = float(
                    peak_entry[0] if isinstance(peak_entry, (list, tuple)) else peak_entry
                )
                dist = abs(b_peak - t_val)
                if nearest_bragg is None or dist < nearest_bragg[0]:
                    nearest_bragg = (dist, ph_name, b_peak)

        if nearest_bragg is not None and nearest_bragg[0] <= angular_tolerance:
            intensity_defects.append(
                ResidualPeak(
                    two_theta=round(t_val, 3),
                    d_spacing_angstrom=round(d_val, 4),
                    residual_height=round(peak_h, 1),
                    significance_sigma=round(sig, 1),
                    peak_type=ResidualPeakType.OVER_CALCULATED_PHANTOM,
                    matched_phase=nearest_bragg[1],
                    bragg_2theta=round(nearest_bragg[2], 3),
                    delta_2theta=round(t_val - nearest_bragg[2], 3),
                )
            )

    return true_unindexed, intensity_defects, residual_artifacts


def calculate_durbin_watson(y_diff: NDArray[Any], weights: NDArray[Any] | None = None) -> float:
    """Calculate the Durbin-Watson serial correlation d-statistic of residuals:

    d = sum_{i=2}^N (Delta Y_i - Delta Y_{i-1})^2 / sum_{i=1}^N Delta Y_i^2,
    where Delta Y_i = sqrt(w_i) * (Y_obs,i - Y_calc,i) when weights are supplied.
    Ideal uncorrelated white noise: d ~ 2.0. Strong serial correlation: d < 1.0.
    """
    diff_arr = np.asarray(y_diff, dtype=float)
    if weights is not None:
        w_arr = np.maximum(np.asarray(weights, dtype=float), 0.0)
        delta = np.sqrt(w_arr) * diff_arr
    else:
        delta = diff_arr
    denom = float(np.sum(delta**2))
    if denom <= 1e-12 or delta.size <= 1:
        return 2.0
    num = float(np.sum(np.diff(delta) ** 2))
    return float(num / denom)


def calculate_bragg_r_factors(
    two_theta: NDArray[Any],
    y_obs: NDArray[Any],
    y_calc: NDArray[Any],
    phase_bragg_peaks: dict[str, list[Any]],
    y_bkg: NDArray[Any] | None = None,
    peak_window: float = 0.25,
) -> dict[str, PhaseBraggMetrics]:
    """Estimate phase-specific Bragg R-factor (R_B) and Structure Factor (R_F)
    from raw profile arrays using 1D Voronoi midpoint partitioning (for standalone
    curve files when a GSAS-II RefList table is unavailable; for live .gpx projects
    the authoritative SSOT is gsas2_adapter.phase_rb_percent / phase_rf_percent).
    """
    results: dict[str, PhaseBraggMetrics] = {}
    if not phase_bragg_peaks or len(two_theta) == 0:
        return results

    half_win = float(peak_window / 2.0)
    n_pts = len(two_theta)

    for ph_name, peaks in phase_bragg_peaks.items():
        if not peaks:
            continue

        # Edge case 1: Ensure peaks are strictly sorted and unique
        sorted_peaks = sorted({float(p[0] if isinstance(p, (list, tuple)) else p) for p in peaks})
        k_count = len(sorted_peaks)

        sum_abs_diff_i = 0.0
        sum_obs_i = 0.0
        sum_abs_diff_f = 0.0
        sum_obs_f = 0.0

        prev_end_i = 0
        for k, b_2th in enumerate(sorted_peaks):
            left_bound = (
                b_2th - half_win
                if k == 0
                else max(b_2th - half_win, (sorted_peaks[k - 1] + b_2th) / 2.0)
            )
            right_bound = (
                b_2th + half_win
                if k == k_count - 1
                else min(b_2th + half_win, (b_2th + sorted_peaks[k + 1]) / 2.0)
            )

            # Edge case 2: Avoid boundary point collisions and guarantee non-overlapping slices
            start_i = max(prev_end_i, int(np.searchsorted(two_theta, left_bound, side="left")))
            start_i = min(start_i, n_pts)
            end_i = max(start_i, int(np.searchsorted(two_theta, right_bound, side="left")))
            end_i = min(end_i, n_pts)
            prev_end_i = end_i

            if start_i >= end_i:
                idx = int(np.clip(np.argmin(np.abs(two_theta - b_2th)), 0, n_pts - 1))
                if idx >= prev_end_i:
                    start_i = idx
                    end_i = min(idx + 1, n_pts)
                    prev_end_i = end_i
                else:
                    continue

            if y_bkg is not None:
                slice_bkg = y_bkg[start_i:end_i]
                i_obs = max(0.0, float(np.sum(np.maximum(0.0, y_obs[start_i:end_i] - slice_bkg))))
                i_calc = max(0.0, float(np.sum(np.maximum(0.0, y_calc[start_i:end_i] - slice_bkg))))
            else:
                i_obs = max(0.0, float(np.sum(y_obs[start_i:end_i])))
                i_calc = max(0.0, float(np.sum(y_calc[start_i:end_i])))

            sum_abs_diff_i += abs(i_obs - i_calc)
            sum_obs_i += i_obs

            f_obs = np.sqrt(i_obs)
            f_calc = np.sqrt(i_calc)
            sum_abs_diff_f += abs(f_obs - f_calc)
            sum_obs_f += f_obs

        r_b = (sum_abs_diff_i / sum_obs_i * 100.0) if sum_obs_i > 0 else 0.0
        r_f = (sum_abs_diff_f / sum_obs_f * 100.0) if sum_obs_f > 0 else 0.0

        results[ph_name] = PhaseBraggMetrics(
            r_bragg_percent=round(r_b, 2),
            r_f_percent=round(r_f, 2),
            integrated_obs_intensity=round(sum_obs_i, 1),
            reflections_count=k_count,
        )

    return results


def calculate_microstructure_parameters(
    x_lorentzian: float,
    y_lorentzian: float,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_KA1,
    scherrer_k: float = 0.90,
) -> MicrostructureMetrics:
    """Converts Thompson-Cox-Hastings Lorentzian profile coefficients (X, Y in degrees)
    into volume-weighted crystallite size <L> (nm) and lattice microstrain epsilon (%).

    Scherrer: <L> = (180 * K * lambda) / (pi * X_sample)
    Stokes-Wilson (beta_2theta = 4 * epsilon * tan(theta)):
        epsilon = (pi * Y_sample) / 720 * 100%
    """
    x_val = max(float(x_lorentzian), 1e-6)
    y_val = max(float(y_lorentzian), 0.0)
    lambda_nm = float(wavelength_angstrom) / 10.0  # Convert Angstrom to nm

    crystallite_size_nm = (180.0 * scherrer_k * lambda_nm) / (np.pi * x_val)
    microstrain_pct = (np.pi * y_val / 720.0) * 100.0

    return MicrostructureMetrics(
        crystallite_size_nm=round(float(crystallite_size_nm), 2),
        microstrain_percent=round(float(microstrain_pct), 4),
        x_lorentzian_deg=round(float(x_val), 4),
        y_lorentzian_deg=round(float(y_val), 4),
    )


def diagnose_residuals(
    two_theta: NDArray[Any],
    y_obs: NDArray[Any],
    y_calc: NDArray[Any],
    wavelength: float = DEFAULT_WAVELENGTH_KA1,
    bragg_peaks_2theta: list[float] | None = None,
    phase_bragg_peaks: dict[str, list[Any]] | None = None,
    y_bkg: NDArray[Any] | None = None,
    angular_tolerance: float = 0.20,
    snr_threshold: float = 3.0,
    min_peak_snr: float = 3.0,
    observed_peaks_2theta: NDArray[Any] | None = None,
) -> ResidualDiagnosisReport:
    """Analyze Rietveld difference curve (Yobs - Ycalc) and classify clinical crystallographic symptoms."""
    y_diff = y_obs - y_calc
    n_points = len(two_theta)
    step_size = float(np.mean(np.diff(two_theta))) if len(two_theta) > 1 else 0.02

    # Estimate background noise standard deviation from tail or low-intensity region
    abs_diff = np.abs(y_diff)
    sigma_bg = float(np.median(abs_diff) / 0.6745)  # MAD estimator of noise
    sigma_bg = max(sigma_bg, 1e-6)

    # Estimate net peak intensity and IUPAC 3-sigma LOD
    y_net = y_obs - y_bkg if y_bkg is not None else y_obs
    i_max_net = float(np.max(y_net)) if len(y_net) > 0 else 1000.0
    lod_3sigma = (3.0 * sigma_bg / max(i_max_net, 1.0)) * 100.0

    # R-factors calculation with Poisson counting weights w_i = 1 / max(Y_obs,i, 1.0)
    weights = 1.0 / np.maximum(y_obs, 1.0)
    sum_obs = float(np.sum(np.abs(y_obs)))
    sum_w_diff_sq = float(np.sum(weights * (y_diff**2)))
    sum_w_obs_sq = float(np.sum(weights * (y_obs**2)))

    r_p = float(np.sum(abs_diff) / sum_obs * 100.0) if sum_obs > 0 else 0.0
    r_wp = float(np.sqrt(sum_w_diff_sq / sum_w_obs_sq) * 100.0) if sum_w_obs_sq > 0 else 0.0
    r_ratio = float(r_wp / r_p) if r_p > 0 else 1.0

    d_stat = calculate_durbin_watson(y_diff, weights=weights)

    symptoms: list[DiagnosticSymptom] = []
    actions: list[str] = []

    # 1. Dual residual classification (True unindexed impurity peaks vs Intensity defects)
    pb_input: dict[str, list[Any]] | None = phase_bragg_peaks
    if pb_input is None and bragg_peaks_2theta is not None:
        pb_input = {"Identified_Phase": list(bragg_peaks_2theta)}

    true_unindexed, intensity_defects, residual_artifacts = classify_residual_peaks(  # noqa: PU001
        two_theta=two_theta,
        y_obs=y_obs,
        y_calc=y_calc,
        phase_bragg_peaks=pb_input,
        angular_tolerance=angular_tolerance,
        snr_threshold=snr_threshold,
        wavelength=wavelength,
        min_peak_snr=min_peak_snr,
        observed_peaks_2theta=observed_peaks_2theta,
    )

    phase_r_factors: dict[str, PhaseBraggMetrics] = {}
    if pb_input:
        phase_r_factors = calculate_bragg_r_factors(
            two_theta=two_theta,
            y_obs=y_obs,
            y_calc=y_calc,
            phase_bragg_peaks=pb_input,
            y_bkg=y_bkg,
        )

    if true_unindexed:
        symptoms.append(
            DiagnosticSymptom(
                code=SymptomCode.UNINDEXED_PHASE,
                severity=SymptomSeverity.CRITICAL,
                name="Unindexed Impurity Reflections",
                description=f"Found {len(true_unindexed)} unexplained positive residual peaks exceeding {snr_threshold}*sigma.",
                details=true_unindexed,
            )
        )
        actions.append(
            "Execute search_match against secondary phases (precursor oxides, hydroxides, oxysulfides, or crucible contamination)."
        )

    if intensity_defects:
        symptoms.append(
            DiagnosticSymptom(
                code=SymptomCode.INTENSITY_MISMATCH,
                severity=SymptomSeverity.WARNING,
                name="Phase Intensity / Profile Defects",
                description=f"Found {len(intensity_defects)} residual peaks aligned with known Bragg reflections. Indicates preferred orientation, site occupancy deficiency, or unrefined atomic positions.",
                details=intensity_defects,
            )
        )
        actions.append(
            "Refine atomic displacement parameters (B_iso), preferred orientation (March-Dollase), or check atomic site occupancies."
        )

    # 2. Detect first-derivative shaped residuals (+/- pairs)
    diff_derivative = np.gradient(y_diff, two_theta)
    max_gradient = float(np.max(np.abs(diff_derivative)))
    mean_obs_intensity = float(np.mean(y_obs))

    if max_gradient > 10.0 * (mean_obs_intensity / (n_points * step_size)):
        symptoms.append(
            DiagnosticSymptom(
                code=SymptomCode.DERIVATIVE_RESIDUAL,
                severity=SymptomSeverity.WARNING,
                name="First-Derivative Residual (Peak Position Mismatch)",
                description="Calculated peak positions are systematically shifted relative to observed peak centers.",
            )
        )
        actions.append(
            "Refine Instrumental Zero-shift if low-angle peaks are shifted, then refine Unit Cell Parameters (a, b, c)."
        )

    # 3. Detect W-shaped or M-shaped profile errors (+/-/+ or -/+/-)
    kurtosis_diff = (
        float(np.mean((y_diff - np.mean(y_diff)) ** 4) / (np.var(y_diff) ** 2))
        if np.var(y_diff) > 0
        else 0.0
    )
    if kurtosis_diff > 8.0:
        symptoms.append(
            DiagnosticSymptom(
                code=SymptomCode.PROFILE_SHAPE_MISMATCH,
                severity=SymptomSeverity.WARNING,
                name="Peak Profile Mismatch (W-shaped ripple)",
                description="Peak tails or peak core poorly fit by current pseudo-Voigt profile function.",
            )
        )
        actions.append(
            "Refine pseudo-Voigt mixing parameter (Gaussian/Lorentzian ratio) and Caglioti parameters (U, V, W). If low-angle peaks are asymmetric, enable Finger-Cox-Jephcoat (FCJ) asymmetry."
        )

    # 4. Detect Durbin-Watson serial correlation anomaly
    if d_stat < 1.0:
        symptoms.append(
            DiagnosticSymptom(
                code=SymptomCode.DURBIN_WATSON_AUTOCORRELATION,
                severity=SymptomSeverity.WARNING,
                name="Durbin-Watson Serial Correlation Anomaly",
                description=f"Durbin-Watson statistic d={d_stat:.2f} < 1.0 indicates strong serial autocorrelation of residuals (oscillations from unmodeled asymmetry, incorrect peak tails, or missing weak reflections).",
            )
        )
        actions.append(
            "Check for unindexed trace impurity peaks, refine pseudo-Voigt profile asymmetry (FCJ), or increase Chebyshev background order."
        )

    if r_ratio > 1.45:
        symptoms.append(
            DiagnosticSymptom(
                code=SymptomCode.RWP_RP_DISCREPANCY,
                severity=SymptomSeverity.INFO,
                name="High R_wp/R_p Discrepancy",
                description=f"R_wp/R_p ratio is {r_ratio:.2f} > 1.40. Weighted residuals are dominated by low-intensity background regions or weak reflection tails.",
            )
        )
        actions.append(
            "Inspect background curve curvature or adjust pseudo-Voigt Lorentzian mixing parameter in tail regions."
        )

    # 5. Detect polynomial background overfitting / oscillations (Background Wiggle)
    if y_bkg is not None and len(y_bkg) > 4:
        bkg_d2 = np.diff(np.diff(y_bkg))
        bkg_d2[np.abs(bkg_d2) < 1e-8] = 0.0
        signs = np.sign(bkg_d2)
        non_zero = signs[signs != 0]
        inflections = int(np.sum(np.diff(non_zero) != 0)) if len(non_zero) > 1 else 0
        if inflections > 4:
            symptoms.append(
                DiagnosticSymptom(
                    code=SymptomCode.BACKGROUND_WIGGLE,
                    severity=SymptomSeverity.WARNING,
                    name="Background Overfitting (Polynomial Ripple)",
                    description=(
                        f"Базовая линия имеет {inflections} перегибов. Полином фона переобучен "
                        "и осциллирует, математически искажая реальные интенсивности и уширенные пики."
                    ),
                    inflections=inflections,
                )
            )
            actions.append(
                "Снизить степень полинома Чебышева (step --stage 1_Scale_and_Background --bkg-terms 3)."
            )

    # 6. Overall quality gate
    overall_status = DiagnosticStatus.PASS
    if any(s.severity == SymptomSeverity.CRITICAL for s in symptoms):
        overall_status = DiagnosticStatus.FAIL
    elif any(s.severity == SymptomSeverity.WARNING for s in symptoms) or r_wp > 10.0:
        overall_status = DiagnosticStatus.NEEDS_REFINEMENT

    return ResidualDiagnosisReport(
        overall_status=overall_status,
        r_p_percent=round(r_p, 2),
        r_wp_percent=round(r_wp, 2),
        r_wp_over_r_p_ratio=round(r_ratio, 2),
        durbin_watson=round(d_stat, 2),
        noise_sigma=round(sigma_bg, 2),
        i_max_net=round(i_max_net, 1),
        lod_3sigma_percent=round(lod_3sigma, 2),
        true_unindexed_count=len(true_unindexed),
        residual_artifact_count=len(residual_artifacts),
        residual_artifacts=residual_artifacts,
        intensity_defects_count=len(intensity_defects),
        true_unindexed_reflections=true_unindexed,
        intensity_mismatch_reflections=intensity_defects,
        phase_r_factors=phase_r_factors,
        total_symptoms=len(symptoms),
        symptoms=symptoms,
        recommended_actions=actions,
    )


def format_diagnostic_report(diagnosis: ResidualDiagnosisReport) -> str:
    """Format diagnosis into clean Markdown for display or LLM consumption."""
    lines = [
        "# Rietveld Residual Diagnostic Report",
        "",
        f"- **Status**: `{diagnosis.overall_status}`",
        f"- **R_wp**: {diagnosis.r_wp_percent}%",
        f"- **R_p**: {diagnosis.r_p_percent}%",
        f"- **R_wp / R_p Ratio**: {diagnosis.r_wp_over_r_p_ratio}",
        f"- **Durbin-Watson d-statistic**: {diagnosis.durbin_watson}",
        f"- **Estimated Noise Sigma**: {diagnosis.noise_sigma}",
        f"- **IUPAC 3-sigma LOD**: ~{diagnosis.lod_3sigma_percent}% (I_max_net={diagnosis.i_max_net})",
        f"- **True Unindexed Peaks**: {diagnosis.true_unindexed_count}",
        f"- **Intensity Defect Peaks**: {diagnosis.intensity_defects_count}",
        "",
        "## Detected Crystallographic Symptoms",
    ]

    if not diagnosis.symptoms:
        lines.append(
            "*No significant systematic residuals detected. Refinement is well-converged.*"
        )
    else:
        for sym in diagnosis.symptoms:
            lines.append(f"### [{sym.severity}] {sym.name} (`{sym.code}`)")
            lines.append(f"{sym.description}")
            if sym.details:
                lines.append("Peaks:")
                for d in sym.details:
                    matched = (
                        f" (matched phase: {d.matched_phase}, Delta 2theta = {d.delta_2theta or 0.0:+.3f} deg)"
                        if d.matched_phase
                        else ""
                    )
                    lines.append(
                        f"  - 2theta = {d.two_theta} deg, d = {d.d_spacing_angstrom} A "
                        f"(intensity: {d.residual_height}, {d.significance_sigma} sigma){matched}"
                    )

    lines.append("")
    lines.append("## Recommended Actions")
    if not diagnosis.recommended_actions:
        lines.append(
            "*Proceed to crystallochemical validation (BVS, GII) and structure publication export.*"
        )
    else:
        for idx, act in enumerate(diagnosis.recommended_actions, 1):
            lines.append(f"{idx}. {act}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("MANDATORY NEXT STEPS (READY TO EXECUTE):")
    lines.append("=" * 70)

    step_idx = 1
    if diagnosis.true_unindexed_count > 0:
        lines.append(
            f"{step_idx}. Query COD / Materials Project or local cifs for candidates explaining unindexed peaks:"
        )
        lines.append(
            '   uv run py .agents/skills/xrd-analysis/scripts/cif_retriever.py search --all "<elements>"'
        )
        step_idx += 1

    if diagnosis.r_wp_percent > 10.0:
        lines.append(
            f"{step_idx}. Refine in-memory sequence of physical parameters (Displacements, Peak broadness):"
        )
        lines.append(
            '   uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py --dir "<sample>" step --sequence "5,7"'
        )
        step_idx += 1

    if diagnosis.intensity_defects_count > 0:
        lines.append(
            f"{step_idx}. If platy/needle morphology is observed, refine March-Dollase texture (Stage 8):"
        )
        lines.append(
            '   uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py --dir "<sample>" step --stage 8_Preferred_Orientation --texture-axis "<Phase>:0,0,1"'
        )
        step_idx += 1

    lines.append(
        f"{step_idx}. Export final Rietveld artifacts and verify convergence (R_wp <= 10%):"
    )
    lines.append(
        '   uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py --dir "<sample>" export'
    )
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XRD Residual Diagnostic Engine; the residual is read either from a "
        "GSAS-II project or from a difference-curve file"
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        type=Path,
        help="Difference-curve file: plain 2theta/Yobs/Ycalc columns, or the profile CSV "
        "written by 'xrd_project.py export'",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="GSAS-II .gpx project; the residual is read from its refined model",
    )
    parser.add_argument(
        "--residual",
        type=Path,
        default=None,
        help="Difference-curve file, same layouts as the positional argument",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=DEFAULT_WAVELENGTH_KA1,
        help=f"X-ray wavelength in Angstroms (default: {DEFAULT_WAVELENGTH_KA1})",
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--out", type=Path, help="Write output report to file")
    args = parser.parse_args()

    sources = [p for p in (args.data_file, args.project, args.residual) if p is not None]
    if len(sources) != 1:
        logger.error(
            "Pass exactly one residual source: a difference-curve file, --residual or --project"
        )
        return 1

    try:
        if args.project is not None:
            table = read_project_difference_curve(args.project)
        else:
            table = read_difference_curve(sources[0])

        two_theta = table[:, 0]
        y_obs = table[:, 1]
        y_calc = table[:, 2]

        report = diagnose_residuals(two_theta, y_obs, y_calc, wavelength=args.wavelength)

        if args.out:
            if args.json:
                args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            else:
                args.out.write_text(format_diagnostic_report(report), encoding="utf-8")
            logger.info(f"Report written to {args.out}")

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_diagnostic_report(report))
        return 0
    except (ValueError, OSError, RuntimeError) as e:
        logger.error(f"Diagnosis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
