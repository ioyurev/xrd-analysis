"""Canonical instrumental profile and hard lower-bound constraints for laboratory XRD."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from constants import CANONICAL_WAVELENGTH_KA1, CANONICAL_WAVELENGTH_KA2
from numpy.typing import NDArray


@dataclass(frozen=True)
class CanonicalInstrument:
    """Standard laboratory Bragg-Brentano diffractometer profile (Cu Ka radiation).

    Serves as an immutable physical lower bound (hard bound) to prevent
    the optimizer from absorbing sample/crystallite size effects into fictional
    instrumental peak sharpening.

    ``ka2_ratio`` is the K-alpha2/K-alpha1 intensity ratio of the modelled
    doublet. A data set may arrive with the companion line already removed: the
    state belongs to the data set and is recorded in the analysis project rather
    than assumed here. The ratio is then 0.0, and the modelled profile must be
    single-wavelength -- keeping the canonical 0.5 adds a shoulder to every
    reflection the measurement does not contain, which forces the fitted peaks
    wider than the data and leaves systematic residuals.
    """

    wavelength_ka1: float = CANONICAL_WAVELENGTH_KA1
    wavelength_ka2: float = CANONICAL_WAVELENGTH_KA2
    ka2_ratio: float = 0.5
    goniometer_radius_mm: float = 217.0

    # Caglioti instrumental coefficients (deg^2)
    u_inst: float = 0.008
    v_inst: float = -0.003
    w_inst: float = 0.006

    def fwhm_inst_deg(self, two_theta_deg: float | NDArray[Any]) -> NDArray[Any]:
        """Compute the instrumental full-width at half-maximum (FWHM) in degrees 2theta."""
        theta_rad = np.radians(np.asarray(two_theta_deg) / 2.0)
        tan_theta = np.tan(theta_rad)
        fwhm2 = self.u_inst * (tan_theta**2) + self.v_inst * tan_theta + self.w_inst
        # Guard against numerical artifacts near zero
        fwhm2 = np.maximum(fwhm2, 1e-6)
        # numpy stubs type sqrt() as Any for the array/scalar union; the declared
        # return type is the contract. Suppressed rule: no-any-return.
        return np.sqrt(fwhm2)  # type: ignore[no-any-return]

    def enforce_lower_bound(
        self, two_theta_deg: NDArray[Any], observed_fwhm_deg: NDArray[Any]
    ) -> NDArray[Any]:
        """Ensure observed FWHM never falls below the hardware instrumental limit.

        Rule: FWHM_calc = max(FWHM_observed, FWHM_inst).
        """
        fwhm_min = self.fwhm_inst_deg(two_theta_deg)
        # Same numpy typing gap as fwhm_inst_deg. Suppressed rule: no-any-return.
        return np.maximum(observed_fwhm_deg, fwhm_min)

    def extract_sample_broadening(
        self, two_theta_deg: NDArray[Any], total_fwhm_deg: NDArray[Any]
    ) -> NDArray[Any]:
        """Deconvolve instrumental broadening using Gaussian approximation:

        beta_sample^2 = beta_total^2 - beta_inst^2 (or 0 if within instrumental resolution).
        """
        inst_fwhm = self.fwhm_inst_deg(two_theta_deg)
        diff = np.square(total_fwhm_deg) - np.square(inst_fwhm)
        return np.sqrt(np.maximum(diff, 0.0))  # type: ignore[no-any-return]

    def displacement_angular_shift_deg(self, displacement_mm: float, two_theta_deg: float) -> float:
        """Return the angular shift a specimen displacement causes at one reflection.

        A flat specimen standing ``s`` millimetres off the goniometer axis shifts every
        reflection by ``d(2theta) = -2 * s * cos(theta) / R`` radians, where ``theta`` is
        half the given angle and ``R`` is the goniometer radius. This is the only relation
        that connects the ``DisplaceX`` parameter -- which GSAS-II states in micrometres --
        with a shift in degrees, and therefore the only way to compare a displacement
        refinement with the Zero correction it competes with.

        The sign is that of the shift in 2theta: a specimen displaced towards the source
        lowers the angles.
        """
        theta_rad = np.radians(two_theta_deg / 2.0)
        shifted = -2.0 * displacement_mm * np.cos(theta_rad) / self.goniometer_radius_mm
        # float() is not decoration: np.degrees() is typed as Any for a scalar argument,
        # and this states the declared return type without a suppression comment.
        return float(np.degrees(shifted))

    def strip_ka2(self, two_theta_deg: NDArray[Any], intensity: NDArray[Any]) -> NDArray[Any]:
        """Remove the K-alpha2 companion line by the RACHINGER / LADELL recursion.

        The measured profile superimposes the Ka1 and Ka2 lines of every
        reflection. Denoting by ``g`` the angular displacement that carries a Ka1
        line onto its own Ka2 partner -- ``sin(theta_Ka2) = (lam2/lam1) *
        sin(theta_Ka1)``, hence ``g(x) > x`` -- the observed profile is

            O(x) = A(x) + ka2_ratio * A(g^-1(x))

        where ``A`` is the Ka1-only profile. Inverting gives the stripping rule

            A(x) = O(x) - ka2_ratio * A(g^-1(x))

        with ``g^-1(x) < x``. The recursion therefore runs from low to high
        angle: every right-hand term it needs has already been stripped. Running
        it in the opposite direction (against ``g(x)``) leaves the Ka2 companion
        in place instead of removing it.

        Removing Ka2 is not cosmetic. The companion line is a peak of
        non-structural origin (SKILL.md, Rule 13, category 2); against a large
        reference card pool a Ka2 satellite inevitably finds a matching card,
        so identification degrades rather than improves.

        The output is not clipped at zero: the recursion legitimately yields small
        negative values just below intense lines, and clipping would bias the
        baseline that the background fit later relies on.
        """
        x = np.asarray(two_theta_deg, dtype=float)
        y = np.asarray(intensity, dtype=float)
        if x.size != y.size:
            raise ValueError("two_theta and intensity must have the same length")
        if x.size == 0:
            return y.copy()

        order = np.argsort(x, kind="stable")
        xs = x[order]
        ys = y[order]

        # Inverse mapping: the Ka1 position whose Ka2 lands exactly on xs[i].
        arg = (self.wavelength_ka1 / self.wavelength_ka2) * np.sin(np.radians(xs / 2.0))
        ka1_prev = np.degrees(2.0 * np.arcsin(np.clip(arg, -1.0, 1.0)))

        xmin = xs[0]
        stripped_sorted = np.zeros_like(ys)
        for i in range(xs.size):
            prev = float(ka1_prev[i])
            if prev < xmin or prev >= xs[i]:
                carry = 0.0
            else:
                carry = float(np.interp(prev, xs, stripped_sorted))
            stripped_sorted[i] = ys[i] - self.ka2_ratio * carry

        stripped = np.empty_like(stripped_sorted)
        stripped[order] = stripped_sorted
        return stripped

    def as_g2_dict(self) -> list[dict[str, Any]]:
        """Return the canonical instrument parameter dictionary for in-memory use in GSAS-II."""
        return [
            {
                "Type": ["PXC", "PXC", 0],
                "Lam": [self.wavelength_ka1, self.wavelength_ka1, 0],
                "Lam1": [self.wavelength_ka1, self.wavelength_ka1, 0],
                "Lam2": [self.wavelength_ka2, self.wavelength_ka2, 0],
                "I(L2)/I(L1)": [self.ka2_ratio, self.ka2_ratio, 0],
                "Zero": [0.0, 0.0, 0],
                "Polariz.": [0.99, 0.99, 0],
                "U": [self.u_inst, self.u_inst, 0],
                "V": [self.v_inst, self.v_inst, 0],
                "W": [self.w_inst, self.w_inst, 0],
                "X": [0.0, 0.0, 0],
                "Y": [0.0, 0.0, 0],
                "Z": [0.0, 0.0, 0],
                "SH/L": [0.002, 0.002, 0],
                "Azimuth": [0.0, 0.0, 0],
                "Bank": [1, 1, 0],
            },
            {},
        ]


def fit_physical_background(
    two_theta: NDArray[Any],
    intensity: NDArray[Any],
    n_terms: int = 6,
    max_iter: int = 40,
    tol: float = 1e-4,
    sigma_clip: float = 2.0,
) -> NDArray[Any]:
    """Канонический расчет физической базовой линии через итеративное отсечение пиков (Peak-Clipped Chebyshev).

    Гарантирует прохождение кривой ровно через центр шумового коридора в долинах между дифракционными пиками.

    :param two_theta: Массив углов дифракции 2theta (в градусах).
    :param intensity: Массив наблюдаемых интенсивностей.
    :param n_terms: Число коэффициентов полинома Чебышева (по умолчанию 6).
    :param max_iter: Максимальное число итераций отсечения пиков.
    :param tol: Критерий сходимости формы фона относительно динамического диапазона.
    :param sigma_clip: Множитель шумового коридора (точки выше bkg + sigma_clip*sigma отсекаются).
    :return: Вычисленная одномерная фоновая кривая той же длины.
    """
    two_theta_arr = np.asarray(two_theta, dtype=float)
    intensity_arr = np.asarray(intensity, dtype=float)
    n_pts = len(two_theta_arr)
    if n_pts < n_terms:
        raise ValueError(f"Число точек дифрактограммы ({n_pts}) должно быть >= n_terms ({n_terms})")

    deg = max(1, n_terms - 1)
    y_work = intensity_arr.copy()
    bkg = np.zeros_like(intensity_arr)

    for _ in range(max_iter):
        # scipy Chebyshev.fit ships without annotations; there is no typed
        # alternative in the public API. Suppressed rule: no-untyped-call.
        cheb = np.polynomial.chebyshev.Chebyshev.fit(  # type: ignore[no-untyped-call]
            two_theta_arr, y_work, deg=deg
        )
        bkg_new = cheb(two_theta_arr)

        diff = intensity_arr - bkg_new
        neg_diff = diff[diff <= 0]
        if len(neg_diff) > 5:
            # Оценка СКО шума в долинах по отрицательным флуктуациям ниже фона
            noise_sigma = float(np.sqrt(np.mean(neg_diff**2)))
        else:
            noise_sigma = float(np.std(diff))
        noise_sigma = max(noise_sigma, 1e-6)

        # Сохранение симметричного шумового коридора в долинах, отсечение дифракционных пиков
        y_next = np.where(diff > sigma_clip * noise_sigma, bkg_new, intensity_arr)

        norm_range = float(np.max(intensity_arr) - np.min(intensity_arr) + 1e-9)
        delta = float(np.max(np.abs(bkg_new - bkg)) / norm_range)
        bkg = bkg_new
        y_work = y_next

        if delta < tol:
            break

    return bkg
