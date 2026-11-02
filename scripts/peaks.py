"""Extraction of the observed reflection list from a measured powder pattern.

This module exists so that two layers can share one definition of "which peaks the
measurement shows": the pipeline, which reports them, and the scoring engine, which needs
them as the denominator of any statement about how much of a pattern a candidate explains.
It was moved out of the pipeline because the pipeline imports the scoring engine, so the
engine could not import it back.

The extraction is a judgement and not a reading, which is the whole reason it must not be
duplicated. Peaks are taken above a fitted physical baseline with a noise threshold from
the median absolute deviation, so counting them does not count noise: measured, a plain
``find_peaks`` without a significance threshold returned 259 "peaks" on a pattern whose
nine peaks are visible to the eye. Any consumer that needs the number of observed peaks
needs that threshold, not a raw maximum search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from constants import CANONICAL_WAVELENGTH_KA1
from instrument import fit_physical_background
from numpy.typing import NDArray
from scipy.signal import find_peaks, savgol_filter

__all__ = ["PeakTable", "extract_peaks_robust"]


@dataclass(frozen=True)
class PeakTable:
    """The observed reflection list extracted from a raw diffraction pattern.

    Every array is indexed by reflection, so the table carries the full list of
    experimental lines rather than bare positions: search-match needs the
    interplanar spacing, and the residual diagnostics need the net height on top
    of the fitted baseline plus the width, because a line blended with a close
    neighbour cannot be attributed to a single phase.
    """

    two_theta: NDArray[np.float64]
    d_spacing: NDArray[np.float64]
    height: NDArray[np.float64]
    fwhm: NDArray[np.float64]
    overlapped: NDArray[np.bool_]

    def __len__(self) -> int:
        return int(self.two_theta.size)


def _half_max_widths(
    two_theta: NDArray[np.float64],
    profile: NDArray[np.float64],
    indices: NDArray[np.intp],
    baseline: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Width at half of each peak's net height, in degrees 2theta.

    ``scipy.signal.peak_widths`` evaluates a single contour level across the whole
    pattern, so when the half-height levels of two lines interleave it attributes
    the neighbour's flank to one of them: a symmetric doublet measured 0.097 deg
    against 0.378 deg. Walking outward from each maximum until the first sample
    below half height yields the width of that line by definition and stops at the
    neighbour instead of absorbing it.
    """
    widths = np.empty(indices.size, dtype=float)
    last = int(profile.size) - 1
    for i, raw_index in enumerate(indices):
        peak = int(raw_index)
        half = float(baseline[peak]) + 0.5 * float(profile[peak] - baseline[peak])
        left = peak
        while left > 0 and profile[left - 1] > half:
            left -= 1
        right = peak
        while right < last and profile[right + 1] > half:
            right += 1
        widths[i] = float(two_theta[right] - two_theta[left])
    return widths


def extract_peaks_robust(
    two_theta: NDArray[np.float64],
    intensity: NDArray[np.float64],
    wavelength: float = CANONICAL_WAVELENGTH_KA1,
    overlap_factor: float = 1.5,
) -> PeakTable:
    """Extract the full observed reflection list from a raw diffraction pattern.

    Peaks are located with ``scipy.signal.find_peaks`` on a Savitzky-Golay
    smoothed profile using a MAD-based noise threshold. Each line is then
    characterised against the canonical physical baseline
    (:func:`instrument.fit_physical_background`): net height above that baseline,
    observed FWHM, interplanar spacing, and a flag marking lines blended with a
    neighbour closer than ``overlap_factor`` FWHMs. A blended line is not
    attributable to one phase, so downstream matching must see the flag instead of
    silently treating both lines as resolved.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n_points = int(intensity.size)

    def _empty() -> PeakTable:
        empty = np.array([], dtype=float)
        return PeakTable(empty, empty, empty, empty, np.array([], dtype=bool))

    if n_points < 10:
        return _empty()

    # Adaptive Savitzky-Golay filtering window
    win_len = min(15, n_points if n_points % 2 != 0 else n_points - 1)
    if win_len >= 5:
        smoothed = savgol_filter(intensity, window_length=win_len, polyorder=2)
    else:
        smoothed = intensity

    # Robust noise estimation using Median Absolute Deviation (MAD)
    diff = np.abs(intensity - smoothed)
    noise_mad = float(np.median(diff)) * 1.4826
    max_val = float(np.max(smoothed))
    bg_level = float(np.percentile(smoothed, 15))
    peak_dyn_range = max(1.0, max_val - bg_level)

    prominence_threshold = max(4.0 * noise_mad, 0.015 * peak_dyn_range)
    height_threshold = bg_level + 3.0 * noise_mad

    peak_indices, _ = find_peaks(  # noqa: PU001
        smoothed,
        prominence=prominence_threshold,
        height=height_threshold,
        distance=4,
    )

    if len(peak_indices) == 0:
        # Fallback to local maxima if SNR threshold was overly strict
        peak_indices, _ = find_peaks(  # noqa: PU001
            smoothed, height=bg_level + noise_mad
        )

    if len(peak_indices) == 0:
        return _empty()

    positions = two_theta[peak_indices]
    baseline = fit_physical_background(two_theta, intensity, n_terms=6)
    heights = np.maximum(0.0, smoothed[peak_indices] - baseline[peak_indices])

    # Distance to the closest neighbour: the observable width of a line cannot
    # exceed the gap to the next one, and an unresolved neighbour would otherwise
    # stretch the half-height walk across the whole pattern.
    step = float(np.mean(np.diff(two_theta))) if n_points > 1 else 0.0
    gaps = np.full(positions.size, np.inf)
    for idx in range(positions.size):
        separations = np.abs(positions - positions[idx])
        separations[idx] = np.inf
        gaps[idx] = float(np.min(separations))

    fwhm = np.minimum(
        _half_max_widths(two_theta, smoothed, peak_indices, baseline),
        np.maximum(gaps, max(step, 1e-6)),
    )

    theta_rad = np.radians(positions / 2.0)
    sine = np.sin(theta_rad)
    d_spacing = np.where(sine > 1e-6, wavelength / (2.0 * sine), np.nan)

    overlapped = gaps < overlap_factor * fwhm

    return PeakTable(
        two_theta=positions,
        d_spacing=d_spacing,
        height=heights,
        fwhm=fwhm,
        overlapped=overlapped,
    )
