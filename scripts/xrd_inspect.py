"""Fast pre-inspection of powder XRD experimental data (.xy).

Evaluates:
1. Background curvature and maximum/minimum baseline levels.
2. Signal-to-noise ratio (SNR) and maximum diffraction intensity.
3. Instrumental/sample peak broadening (FWHM) at low and high 2-theta angles.
4. Detection of amorphous haloes or broad substrate humps (FWHM > 3.0 deg).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from instrument import fit_physical_background
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, peak_widths


class CrystallinityGrade(StrEnum):
    """Qualitative crystallinity grade from pre-inspection."""

    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low / Poorly Crystalline"


@dataclass(frozen=True)
class AmorphousFeature:
    """Detected broad amorphous hump or substrate feature."""

    two_theta: float
    fwhm_deg: float
    intensity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "2theta": self.two_theta,
            "fwhm_deg": self.fwhm_deg,
            "intensity": self.intensity,
        }


@dataclass(frozen=True)
class PatternInspectionReport:
    """Typed crystallographic pre-inspection report of a raw XRD scan."""

    range_2theta: tuple[float, float]
    step_size: float
    total_points: int
    max_intensity: float
    min_intensity: float
    estimated_noise_sigma: float
    signal_to_noise_ratio: float
    detected_peaks_count: int
    low_angle_fwhm_deg: float | None
    high_angle_fwhm_deg: float | None
    has_amorphous_halo: bool
    amorphous_features: list[AmorphousFeature]
    crystallinity_grade: CrystallinityGrade

    def to_dict(self) -> dict[str, Any]:
        return {
            "range_2theta": [self.range_2theta[0], self.range_2theta[1]],
            "step_size": self.step_size,
            "total_points": self.total_points,
            "max_intensity": self.max_intensity,
            "min_intensity": self.min_intensity,
            "estimated_noise_sigma": self.estimated_noise_sigma,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
            "detected_peaks_count": self.detected_peaks_count,
            "low_angle_fwhm_deg": self.low_angle_fwhm_deg,
            "high_angle_fwhm_deg": self.high_angle_fwhm_deg,
            "has_amorphous_halo": self.has_amorphous_halo,
            "amorphous_features": [f.to_dict() for f in self.amorphous_features],
            "crystallinity_grade": str(self.crystallinity_grade),
        }


def load_xy_data(path: Path) -> tuple[NDArray[Any], NDArray[Any]]:
    """Load two-column powder diffraction pattern (2theta, intensity) with container shape check."""
    raw = np.atleast_2d(np.loadtxt(str(path)))
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(
            f"{path.name} has shape {raw.shape}; expected at least 2 columns (2theta, intensity)"
        )
    two_theta = np.asarray(raw[:, 0], dtype=float)
    intensity = np.asarray(raw[:, 1], dtype=float)
    return two_theta, intensity


def estimate_background(
    intensity: NDArray[Any],
    two_theta: NDArray[Any] | None = None,
    window_pts: int = 50,
    n_terms: int = 6,
) -> NDArray[Any]:
    """Robust physical baseline calculation using Chebyshev peak-clipping or rolling minimum."""
    if two_theta is not None and len(intensity) >= n_terms:
        try:
            return fit_physical_background(two_theta, intensity, n_terms=n_terms)
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.debug(f"Chebyshev background fit fallback: {err}")

    if len(intensity) < window_pts:
        window_pts = max(3, len(intensity) // 4)
    # Apply rolling minimum followed by uniform smoothing
    eroded = minimum_filter1d(intensity, size=window_pts, mode="nearest")
    background = uniform_filter1d(eroded, size=window_pts, mode="nearest")
    return np.asarray(np.minimum(background, intensity), dtype=float)


def inspect_powder_pattern(
    two_theta: NDArray[Any],
    intensity: NDArray[Any],
) -> PatternInspectionReport:
    """Perform comprehensive crystallographic pre-inspection of XRD scan."""
    n_points = len(two_theta)
    if n_points < 10:
        raise ValueError("Diffraction pattern has fewer than 10 data points.")

    step_size = float(np.mean(np.diff(two_theta)))
    t_min = float(np.min(two_theta))
    t_max = float(np.max(two_theta))
    i_max = float(np.max(intensity))
    i_min = float(np.min(intensity))

    # Baseline and noise estimation using canonical Chebyshev fit
    bkg = estimate_background(intensity, two_theta=two_theta, n_terms=6)

    # Noise estimated in upper 15% 2theta region where Bragg reflections are typically weaker
    high_tail_idx = int(0.85 * n_points)
    noise_region = (intensity - bkg)[high_tail_idx:]
    noise_std = (
        float(np.std(noise_region)) if len(noise_region) > 5 else float(np.std(intensity - bkg))
    )
    noise_std = max(noise_std, 1e-6)

    snr = float((i_max - np.min(bkg)) / noise_std)

    # Peak detection
    min_prominence = max(float(3.0 * noise_std), float(0.01 * (i_max - np.min(bkg))))
    peaks, _properties = find_peaks(  # noqa: PU001
        intensity,
        prominence=min_prominence,
        distance=max(3, int(0.15 / max(step_size, 1e-4))),
    )

    low_angle_fwhm: list[float] = []
    high_angle_fwhm: list[float] = []
    amorphous_humps: list[AmorphousFeature] = []

    if len(peaks) > 0:
        widths_res = peak_widths(intensity, peaks, rel_height=0.5)
        widths_deg = widths_res[0] * step_size

        for p_idx, width in zip(peaks, widths_deg, strict=False):
            tt_val = float(two_theta[p_idx])
            p_height = float(intensity[p_idx])

            if width >= 3.0:
                amorphous_humps.append(
                    AmorphousFeature(
                        two_theta=round(tt_val, 2),
                        fwhm_deg=round(float(width), 2),
                        intensity=round(p_height, 1),
                    )
                )
            elif tt_val <= 35.0:
                low_angle_fwhm.append(float(width))
            elif tt_val >= 60.0:
                high_angle_fwhm.append(float(width))

    median_fwhm_low = float(np.median(low_angle_fwhm)) if low_angle_fwhm else None
    median_fwhm_high = float(np.median(high_angle_fwhm)) if high_angle_fwhm else None

    # Assessment
    has_amorphous_halo = len(amorphous_humps) > 0
    crystallinity_grade = (
        CrystallinityGrade.HIGH
        if (len(peaks) >= 8 and not has_amorphous_halo and snr > 20)
        else CrystallinityGrade.MODERATE
    )
    if snr < 5 or (len(peaks) < 3 and has_amorphous_halo):
        crystallinity_grade = CrystallinityGrade.LOW

    return PatternInspectionReport(
        range_2theta=(round(t_min, 3), round(t_max, 3)),
        step_size=round(step_size, 4),
        total_points=n_points,
        max_intensity=round(i_max, 2),
        min_intensity=round(i_min, 2),
        estimated_noise_sigma=round(noise_std, 2),
        signal_to_noise_ratio=round(snr, 1),
        detected_peaks_count=len(peaks),
        low_angle_fwhm_deg=round(median_fwhm_low, 3) if median_fwhm_low is not None else None,
        high_angle_fwhm_deg=round(median_fwhm_high, 3) if median_fwhm_high is not None else None,
        has_amorphous_halo=has_amorphous_halo,
        amorphous_features=amorphous_humps,
        crystallinity_grade=crystallinity_grade,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="XRD Experimental Pre-Inspector")
    parser.add_argument("xy_file", type=Path, help="Path to .xy diffraction file")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--out", type=Path, help="Write output JSON to specified file")
    args = parser.parse_args()

    if not args.xy_file.exists():
        logger.error(f"File not found: {args.xy_file}")
        return 1

    try:
        two_theta, intensity = load_xy_data(args.xy_file)  # noqa: PU001
        report = inspect_powder_pattern(two_theta, intensity)

        if args.out:
            args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            logger.info(f"Report saved to {args.out}")

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("==================================================")
            print("XRD PRE-INSPECTION SUMMARY")
            print("==================================================")
            print(
                f"2-Theta Range:  {report.range_2theta[0]} - {report.range_2theta[1]} deg (step: {report.step_size})"
            )
            print(f"Max Intensity:  {report.max_intensity} counts")
            print(
                f"Estimated SNR:  {report.signal_to_noise_ratio} (sigma={report.estimated_noise_sigma})"
            )
            print(f"Bragg Peaks:    {report.detected_peaks_count} detected")
            print(f"FWHM (<35 deg): {report.low_angle_fwhm_deg} deg")
            print(f"FWHM (>60 deg): {report.high_angle_fwhm_deg} deg")
            print(f"Amorphous Halo: {report.has_amorphous_halo}")
            print(f"Crystallinity:  {report.crystallinity_grade}")
            print("==================================================")
        return 0
    except (ValueError, OSError, RuntimeError) as e:
        logger.error(f"Inspection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
