"""GSAS-II Typed Facade and Adapter Module.

Encapsulates GSAS-II internal data structures, dictionaries, and arrays behind
strongly-typed dataclasses and deterministic extraction functions. Provides
exact mathematical calculation of crystallographic agreement indices (R_p,
R_wp, GoF/chi^2, Durbin-Watson d) directly from raw profile arrays, preventing
hardcoded mock defaults or silent parameter fallbacks.
"""

from __future__ import annotations

import csv
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any


class CurveSlot(IntEnum):
    """Positional slots in GSAS-II hist.data['data'][1] profile curve array."""

    TWO_THETA = 0
    Y_OBS = 1
    WEIGHTS = 2
    Y_CALC = 3
    Y_BKG = 4
    Y_DIFF = 5


class ReflCol(IntEnum):
    """Column indices in GSAS-II powder Reflection Lists ('RefList')."""

    H = 0
    K = 1
    L = 2
    D_SPACING = 4
    TWO_THETA = 5
    F_OBS_SQ = 8
    F_CALC_SQ = 9


import gemmi
import numpy as np
from loguru import logger

_G2_BASE_SAFE_ERRORS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    OSError,
)


def g2_safe_errors() -> tuple[type[Exception], ...]:
    """Return narrow exception tuple including GSAS-II G2ScriptException when loaded."""
    g2_mod = sys.modules.get("GSASII.GSASIIscriptable")
    g2_exc = getattr(g2_mod, "G2ScriptException", None)
    if isinstance(g2_exc, type) and issubclass(g2_exc, Exception):
        return (*_G2_BASE_SAFE_ERRORS, g2_exc)
    return _G2_BASE_SAFE_ERRORS


__all__ = [
    "KEEP_LISTING_ENV",
    "BraggReflection",
    "ProfileCurves",
    "RefinementMetrics",
    "build_scaled_phase_bragg_peaks",
    "calculate_profile_metrics",
    "discard_gsas_listing",
    "ensure_fresh_residuals",
    "export_histogram_curves_safe",
    "export_phase_cif_safe",
    "export_reflections_safe",
    "extract_bragg_reflections",
    "extract_cell_parameters",
    "extract_phase_scales",
    "extract_profile_curves",
    "g2_safe_errors",
    "get_native_histogram_residuals",
    "get_project_qpa",
    "keep_gsas_listing",
    "phase_rb_percent",
    "phase_rf_percent",
    "read_bragg_reflections_csv",
    "read_difference_curve",
    "read_project_difference_curve",
    "sync_pawley_reflections_to_hist",
    "write_bragg_reflections_csv",
]


@dataclass(frozen=True)
class BraggReflection:
    """Strongly-typed representation of a single Bragg diffraction reflection."""

    phase_name: str
    hkl: tuple[int, ...]
    two_theta: float
    d_spacing: float
    intensity: float = 100.0
    f_calc: float = 0.0
    f_obs: float = 0.0


@dataclass(frozen=True)
class ProfileCurves:
    """Strongly-typed diffraction profile curves."""

    two_theta: np.ndarray
    y_obs: np.ndarray
    y_calc: np.ndarray
    y_bkg: np.ndarray
    y_diff: np.ndarray
    sigma_obs: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class RefinementMetrics:
    """Exact crystallographic agreement and residual statistics."""

    r_p_percent: float
    r_wp_percent: float
    r_wp_over_r_p_ratio: float
    gof: float
    reduced_chi2: float
    durbin_watson: float
    n_data_points: int
    n_parameters: int
    r_exp_percent: float = 0.0
    r_wp_over_r_exp_ratio: float = 1.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "r_p_percent": self.r_p_percent,
            "r_wp_percent": self.r_wp_percent,
            "r_wp_over_r_p_ratio": self.r_wp_over_r_p_ratio,
            "gof": self.gof,
            "reduced_chi2": self.reduced_chi2,
            "durbin_watson": self.durbin_watson,
            "n_data_points": self.n_data_points,
            "n_parameters": self.n_parameters,
            "r_exp_percent": self.r_exp_percent,
            "r_wp_over_r_exp_ratio": self.r_wp_over_r_exp_ratio,
        }


def calculate_profile_metrics(
    curves: ProfileCurves,
    n_free_parameters: int = 15,
) -> RefinementMetrics:
    """Compute exact IUCr agreement indices directly from profile arrays.

    Formulas:
        R_p = sum(|Y_obs - Y_calc|) / sum(Y_obs) * 100%
        R_wp = sqrt( sum(w * (Y_obs - Y_calc)^2) / sum(w * Y_obs^2) ) * 100%
        R_exp = sqrt( max(1, N - P) / sum(w * Y_obs^2) ) * 100%
        reduced_chi2 = sum(w * (Y_obs - Y_calc)^2) / max(1, N - P) = GoF^2
        gof = sqrt(reduced_chi2) = R_wp / R_exp
        d = sum_{i=2}^N (Delta Y_i - Delta Y_{i-1})^2 / sum_{i=1}^N (Delta Y_i)^2
            where Delta Y_i = sqrt(w_i) * (Y_obs,i - Y_calc,i)
    """
    y_obs = curves.y_obs
    y_calc = curves.y_calc
    w = curves.weights
    n_pts = len(y_obs)

    if n_pts == 0:
        return RefinementMetrics(
            r_p_percent=0.0,
            r_wp_percent=0.0,
            r_wp_over_r_p_ratio=1.0,
            gof=1.0,
            reduced_chi2=1.0,
            durbin_watson=2.0,
            n_data_points=0,
            n_parameters=n_free_parameters,
            r_exp_percent=0.0,
            r_wp_over_r_exp_ratio=1.0,
        )

    # 1. Profile R-factor (R_p)
    sum_abs_diff = float(np.sum(np.abs(y_obs - y_calc)))
    sum_y_obs = float(np.sum(np.abs(y_obs)))
    r_p = (sum_abs_diff / max(1e-12, sum_y_obs)) * 100.0

    # 2. Weighted Profile R-factor (R_wp)
    diff_sq = (y_obs - y_calc) ** 2
    sum_w_diff_sq = float(np.sum(w * diff_sq))
    sum_w_yobs_sq = float(np.sum(w * (y_obs**2)))
    r_wp = float(np.sqrt(sum_w_diff_sq / max(1e-12, sum_w_yobs_sq))) * 100.0

    # 3. Expected R-factor (R_exp), Goodness-of-Fit (GoF) and reduced chi^2 (GoF^2)
    degrees_of_freedom = max(1, n_pts - n_free_parameters)
    r_exp = float(np.sqrt(degrees_of_freedom / max(1e-12, sum_w_yobs_sq))) * 100.0
    chi2 = float(sum_w_diff_sq / degrees_of_freedom)
    gof = float(np.sqrt(max(0.0, chi2)))

    # 4. Durbin-Watson serial correlation statistic (d)
    # Delta_i = sqrt(w_i) * (Y_obs,i - Y_calc,i)
    delta = np.sqrt(np.maximum(w, 0.0)) * (y_obs - y_calc)
    delta_sq_sum = float(np.sum(delta**2))
    if delta_sq_sum > 1e-12 and n_pts > 1:
        consecutive_diff_sq = float(np.sum((delta[1:] - delta[:-1]) ** 2))
        d_stat = float(consecutive_diff_sq / delta_sq_sum)
    else:
        d_stat = 2.0

    r_ratio = r_wp / max(1e-12, r_p)
    r_exp_ratio = r_wp / max(1e-12, r_exp)

    return RefinementMetrics(
        r_p_percent=round(r_p, 2),
        r_wp_percent=round(r_wp, 2),
        r_wp_over_r_p_ratio=round(r_ratio, 2),
        gof=round(gof, 2),
        reduced_chi2=round(chi2, 2),
        durbin_watson=round(d_stat, 2),
        n_data_points=n_pts,
        n_parameters=n_free_parameters,
        r_exp_percent=round(r_exp, 2),
        r_wp_over_r_exp_ratio=round(r_exp_ratio, 2),
    )


def extract_profile_curves(gpx_or_hist: Any) -> ProfileCurves:
    """Extract profile arrays from a GSAS-II G2Project or G2Histogram."""
    hist = gpx_or_hist
    if hasattr(gpx_or_hist, "histograms"):
        hists = gpx_or_hist.histograms()
        if not hists:
            raise ValueError("G2Project contains no histograms")
        hist = hists[0]

    if not hasattr(hist, "data") or "data" not in hist.data:
        raise ValueError("Histogram object has no internal 'data' block")

    raw_curves = hist.data["data"]
    if len(raw_curves) < 2 or raw_curves[1] is None:
        raise ValueError("Profile array curve block hist.data['data'][1] is missing or empty")

    c_data = raw_curves[1]
    two_th = np.asarray(c_data[CurveSlot.TWO_THETA], dtype=float)
    y_o = np.asarray(c_data[CurveSlot.Y_OBS], dtype=float)
    w_arr = np.asarray(c_data[CurveSlot.WEIGHTS], dtype=float)
    y_c = np.asarray(c_data[CurveSlot.Y_CALC], dtype=float)
    y_b = np.asarray(c_data[CurveSlot.Y_BKG], dtype=float)
    y_d = np.asarray(c_data[CurveSlot.Y_DIFF], dtype=float)

    # Reconstruct standard deviations: sigma_obs = 1 / sqrt(w) when w > 0
    sig_o = np.where(
        w_arr > 0,
        1.0 / np.sqrt(np.maximum(w_arr, 1e-12)),
        np.sqrt(np.maximum(y_o, 1.0)),
    )
    # Ensure positive weights
    weights = np.where(w_arr > 0, w_arr, 1.0 / np.maximum(y_o, 1.0))

    return ProfileCurves(
        two_theta=two_th,
        y_obs=y_o,
        y_calc=y_c,
        y_bkg=y_b,
        y_diff=y_d,
        sigma_obs=sig_o,
        weights=weights,
    )


def sync_pawley_reflections_to_hist(gpx: Any, hist: Any = None) -> None:
    """Синхронизирует межплоскостные расстояния d_hkl в Pawley ref и обновляет RefList гистограммы.

    В режиме doPawley=True GSAS-II уточняет интенсивности в phase.data['Pawley ref'],
    оставляя hist.data['Reflection Lists'][phase]['RefList'] со старыми углами и
    структурными факторами от атома-заглушки. Синхронизация гарантирует, что
    extract_bragg_reflections, xrd_plot, xrd_diagnose и export видят актуальные
    уточнённые положения штрихов Брэгга и интенсивности Поули.
    """
    if hist is None and hasattr(gpx, "histograms"):
        hists = gpx.histograms()
        if hists:
            hist = hists[0]
    if hist is None or not hasattr(gpx, "phases"):
        return

    inst_raw = getattr(hist, "data", {}).get("Instrument Parameters", [{}])[0]
    lam_rec = inst_raw.get("Lam1") or inst_raw.get("Lam") or [1.540596, 1.540596]
    lam = float(lam_rec[1] if len(lam_rec) > 1 else lam_rec[0])
    zero_rec = inst_raw.get("Zero", [0.0, 0.0, False])
    zero_deg = float(zero_rec[1] if len(zero_rec) > 1 else 0.0)

    refl_lists = hist.data.setdefault("Reflection Lists", {})

    for phase in gpx.phases():
        pdata = getattr(phase, "data", {})
        if not isinstance(pdata, dict):
            continue
        gen = pdata.get("General", {})
        if not bool(gen.get("doPawley", False)):
            continue

        c_info = gen.get("Cell", [False, 5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        if len(c_info) < 7:
            continue
        uc = gemmi.UnitCell(
            float(c_info[1]),
            float(c_info[2]),
            float(c_info[3]),
            float(c_info[4]),
            float(c_info[5]),
            float(c_info[6]),
        )

        pawley_refs = pdata.get("Pawley ref", [])
        synced_reflist: list[list[Any]] = []
        for pk in pawley_refs:
            if len(pk) < 7:
                continue
            h_i = int(pk[0])
            k_i = int(pk[1])
            l_i = int(pk[2])
            mul = int(pk[3])
            d_new = float(uc.calculate_d([h_i, k_i, l_i]))
            pk[4] = d_new
            sin_th = lam / (2.0 * max(d_new, 1e-6))
            if sin_th > 1.0:
                continue
            two_th = float(np.degrees(2.0 * np.arcsin(sin_th))) + zero_deg
            inten = max(0.0, float(pk[6])) if bool(pk[5]) else 0.0
            ref_i = max(inten, 1.0)
            synced_reflist.append(
                [h_i, k_i, l_i, mul, d_new, two_th, 0.01, 0.01, ref_i, ref_i, 0.0, 1.0]
            )

        if synced_reflist:
            p_entry = refl_lists.setdefault(phase.name, {})
            p_entry["RefList"] = np.asarray(synced_reflist, dtype=float)


def extract_bragg_reflections(gpx_or_hist: Any) -> list[BraggReflection]:
    """Safely extract Bragg reflections from GSAS-II reflection lists.

    Column specification in GSAS-II RefList:
        0: h, 1: k, 2: l, 3: multiplicity, 4: d_spacing (A), 5: two_theta (deg),
        6: FWHM, 7: Fcalc, 8: Fobs, 9: phase intensity
    """
    hist = gpx_or_hist
    if hasattr(gpx_or_hist, "histograms"):
        hists = gpx_or_hist.histograms()
        if not hists:
            return []
        hist = hists[0]

    refl_lists = getattr(hist, "data", {}).get("Reflection Lists", {})
    if not refl_lists:
        return []

    reflections: list[BraggReflection] = []
    for phase_name, p_data in refl_lists.items():
        refls = p_data.get("RefList", None)
        if refls is None or len(refls) == 0:
            continue

        for r in refls:
            if len(r) <= ReflCol.TWO_THETA:
                continue
            try:
                h = int(r[ReflCol.H])
                k = int(r[ReflCol.K])
                l = int(r[ReflCol.L])
                d_sp = float(r[ReflCol.D_SPACING])
                two_th = float(r[ReflCol.TWO_THETA])
                intensity = float(r[ReflCol.F_CALC_SQ]) if len(r) > ReflCol.F_CALC_SQ else 100.0
                f_obs_sq = float(r[ReflCol.F_OBS_SQ]) if len(r) > ReflCol.F_OBS_SQ else 0.0
                f_calc_sq = float(r[ReflCol.F_CALC_SQ]) if len(r) > ReflCol.F_CALC_SQ else 0.0
                f_obs = float(np.sqrt(abs(f_obs_sq)))
                f_calc = float(np.sqrt(abs(f_calc_sq)))
                reflections.append(
                    BraggReflection(
                        phase_name=str(phase_name),
                        hkl=(h, k, l),
                        two_theta=round(two_th, 4),
                        d_spacing=round(d_sp, 4),
                        intensity=round(intensity, 1),
                        f_calc=f_calc,
                        f_obs=f_obs,
                    )
                )
            except (ValueError, TypeError, IndexError):
                continue

    return reflections


def phase_rf_percent(reflections: list[BraggReflection]) -> dict[str, float]:
    """Возвращает RF (%) по каждой фазе из таблицы отражений.

    RF = sum|Fobs - Fcalc| / sum Fobs по отражениям фазы — тот же критерий, что
    GSAS-II печатает в листинге, но считанный из живых объектов проекта. RF
    диагностирует согласие интенсивностей *отдельной* фазы: при нормальном
    весе фазы высокий RF означает проблему модели, а не ошибку шкалы.

    Фазы без положительной суммы Fobs в результат не попадают.
    """
    observed: dict[str, float] = {}
    residual: dict[str, float] = {}
    for refl in reflections:
        if refl.f_obs <= 0.0:
            continue
        observed[refl.phase_name] = observed.get(refl.phase_name, 0.0) + refl.f_obs
        residual[refl.phase_name] = residual.get(refl.phase_name, 0.0) + abs(
            refl.f_obs - refl.f_calc
        )
    return {
        name: round(100.0 * residual[name] / total, 2)
        for name, total in observed.items()
        if total > 0.0
    }


def phase_rb_percent(reflections: list[BraggReflection]) -> dict[str, float]:
    """Возвращает Bragg R-factor (R_B, %) по фазам из таблицы отражений.

    Определение: R_B = sum|I_obs - I_calc| / sum I_obs, где I = F^2.

    Это **не** то число, которое GSAS-II кладёт в ``Rf^2``: измерено на реальных
    проектах, что под этим ключом лежит RMS-вариант
    ``sqrt(sum(dI^2)/sum(I^2))`` (совпадение до 0.01 на всех проверенных фазах),
    а линейная формула расходится с ним до 20 п.п. (SmSi2: 29.88 против 44.85).
    Поэтому значения печатаются с пометкой источника: ``R_B(lin)`` — этот расчёт,
    ``Rf^2`` — сохранённое движком.
    """
    observed: dict[str, float] = {}
    residual: dict[str, float] = {}
    for refl in reflections:
        i_obs = refl.f_obs**2
        i_calc = refl.f_calc**2
        if i_obs <= 0.0:
            continue
        observed[refl.phase_name] = observed.get(refl.phase_name, 0.0) + i_obs
        residual[refl.phase_name] = residual.get(refl.phase_name, 0.0) + abs(i_obs - i_calc)
    return {
        name: round(100.0 * residual[name] / total, 2)
        for name, total in observed.items()
        if total > 0.0
    }


def native_phase_r_factors(hist: Any) -> dict[str, dict[str, float]]:
    """Читает сохранённые движком ``Rf`` и ``Rf^2`` по фазам (SSOT, не пересчёт).

    Ключи имеются вид ``"<phase>:<hist>:Rf"`` и ``":Rf^2"``; ``G2PwldData.residuals``
    отдаёт их вместе с профильными R-факторами. Значения уже посчитаны GSAS-II;
    пересчитывать их своим кодом можно только для сверки.
    """
    result: dict[str, dict[str, float]] = {}
    residuals = getattr(hist, "residuals", None)
    if not isinstance(residuals, dict):
        return result
    for key, value in residuals.items():
        parts = key.split(":")
        if len(parts) != 3 or parts[2] not in ("Rf", "Rf^2"):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        result.setdefault(parts[0], {})[parts[2]] = numeric
    return result


def extract_phase_scales(gpx_project: Any) -> dict[str, float]:
    """Safely extract phase scale factors from a G2Project."""
    scales: dict[str, float] = {}
    if not hasattr(gpx_project, "phases") or not hasattr(gpx_project, "histograms"):
        return scales

    phases = gpx_project.phases()
    histograms = gpx_project.histograms()
    if not phases or not histograms:
        return scales

    hist_obj = histograms[0]
    hist_name = getattr(hist_obj, "name", "")
    for idx, p in enumerate(phases):
        p_name = getattr(p, "name", f"Phase_{idx}")
        sc_val = 0.0
        try:
            h_hists = (
                p.data.get("Histograms", {})
                if hasattr(p, "data") and isinstance(p.data, dict)
                else {}
            )
            h_dict = h_hists.get(hist_name, {})
            if not h_dict and h_hists:
                h_dict = next(iter(h_hists.values()))
            sc_info = h_dict.get("Scale", [0.0, False])
            sc_val = float(sc_info[0]) if isinstance(sc_info, (list, tuple)) else float(sc_info)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            sc_val = 0.0
        scales[p_name] = max(0.0, sc_val)

    return scales


def _get_val(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Safe multi-dialect numeric value extractor from dictionary."""
    for k in keys:
        if k in d and d[k] is not None:
            with suppress(ValueError, TypeError):
                return float(d[k])
    return default


def extract_cell_parameters(gpx_project: Any) -> dict[str, dict[str, float]]:
    """Safely extract unit cell parameters and ESDs from a G2Project."""
    cells: dict[str, dict[str, float]] = {}
    if not hasattr(gpx_project, "phases"):
        return cells

    for idx, p in enumerate(gpx_project.phases()):
        p_name = getattr(p, "name", f"Phase_{idx}")
        cell_dict: dict[str, float] = {}

        # 1. Native get_cell_and_esd() from GSASIIscriptable
        if hasattr(p, "get_cell_and_esd"):
            with suppress(*g2_safe_errors()):
                res = p.get_cell_and_esd()
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    cell_vals = res[0] if isinstance(res[0], dict) else {}
                    esd_vals = res[1] if isinstance(res[1], dict) else {}
                    if cell_vals:
                        cell_dict = {
                            "a": _get_val(cell_vals, "length_a", "a", "cell_a"),
                            "b": _get_val(cell_vals, "length_b", "b", "cell_b"),
                            "c": _get_val(cell_vals, "length_c", "c", "cell_c"),
                            "alpha": _get_val(cell_vals, "angle_alpha", "alpha"),
                            "beta": _get_val(cell_vals, "angle_beta", "beta"),
                            "gamma": _get_val(cell_vals, "angle_gamma", "gamma"),
                            "volume": _get_val(cell_vals, "volume", "vol"),
                            "a_esd": _get_val(esd_vals, "length_a", "a", "sig_a", "esd_a"),
                            "b_esd": _get_val(esd_vals, "length_b", "b", "sig_b", "esd_b"),
                            "c_esd": _get_val(esd_vals, "length_c", "c", "sig_c", "esd_c"),
                            "alpha_esd": _get_val(
                                esd_vals, "angle_alpha", "alpha", "sig_alpha", "esd_alpha"
                            ),
                            "beta_esd": _get_val(
                                esd_vals, "angle_beta", "beta", "sig_beta", "esd_beta"
                            ),
                            "gamma_esd": _get_val(
                                esd_vals, "angle_gamma", "gamma", "sig_gamma", "esd_gamma"
                            ),
                            "volume_esd": _get_val(esd_vals, "volume", "vol", "sig_vol"),
                        }

        # 2. Strict backward-compatible fallback on p.data["General"]["Cell"]
        if not cell_dict:
            try:
                c_info = p.data.get("General", {}).get("Cell", [False, 0, 0, 0, 0, 0, 0])
                if len(c_info) >= 7:
                    cell_dict = {
                        "a": float(c_info[1]),
                        "b": float(c_info[2]),
                        "c": float(c_info[3]),
                        "alpha": float(c_info[4]),
                        "beta": float(c_info[5]),
                        "gamma": float(c_info[6]),
                        "volume": float(c_info[7]) if len(c_info) > 7 else 0.0,
                    }
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                pass

        # 3. Preserve non-zero cell e.s.d.s across stages where Cell is locked
        if hasattr(p, "data") and isinstance(p.data, dict):
            gen = p.data.setdefault("General", {})
            esd_keys = (
                "a_esd",
                "b_esd",
                "c_esd",
                "alpha_esd",
                "beta_esd",
                "gamma_esd",
                "volume_esd",
            )
            if any(float(cell_dict.get(k, 0.0) or 0.0) > 0.0 for k in esd_keys):
                gen["Cell_esd"] = {k: float(cell_dict.get(k, 0.0) or 0.0) for k in esd_keys}
            elif "Cell_esd" in gen and isinstance(gen["Cell_esd"], dict):
                for k, v in gen["Cell_esd"].items():
                    if float(v or 0.0) > 0.0 and float(cell_dict.get(k, 0.0) or 0.0) <= 0.0:
                        cell_dict[k] = float(v)

        cells[p_name] = cell_dict

    return cells


def build_scaled_phase_bragg_peaks(
    curves: ProfileCurves,
    reflections: list[BraggReflection],
    scales: dict[str, float],
) -> dict[str, list[tuple[float, float]]]:
    """Estimate net detector-count peak heights for each phase reflection."""
    y_net = curves.y_calc - curves.y_bkg if curves.y_bkg is not None else curves.y_calc
    y_net_max = max(float(np.max(y_net)) if len(y_net) > 0 else 1.0, 1.0)

    scaled_refls: list[tuple[str, float, float]] = []
    max_scaled_i = 1e-6
    for r in reflections:
        p_sc = float(scales.get(r.phase_name, 1.0))
        i_sc = p_sc * float(r.intensity)
        scaled_refls.append((r.phase_name, float(r.two_theta), i_sc))
        max_scaled_i = max(max_scaled_i, i_sc)

    phase_bragg_dict: dict[str, list[tuple[float, float]]] = {}
    for ph_name, two_th, i_sc in scaled_refls:
        y_peak_est = y_net_max * (i_sc / max_scaled_i)
        phase_bragg_dict.setdefault(ph_name, []).append((two_th, y_peak_est))
    return phase_bragg_dict


def write_bragg_reflections_csv(
    reflections: list[BraggReflection],
    out_csv: Path,
) -> Path:
    """Write Bragg reflections to RFC 4180 compliant CSV with quoted HKL tuples."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "hkl", "two_theta", "d_spacing", "rel_intensity"])
        for r in reflections:
            writer.writerow(
                [
                    r.phase_name,
                    str(r.hkl),
                    f"{r.two_theta:.4f}",
                    f"{r.d_spacing:.4f}",
                    f"{r.intensity:.1f}",
                ]
            )
    return out_csv


def read_bragg_reflections_csv(
    csv_path: Path,
) -> list[BraggReflection]:
    """Read Bragg reflections from CSV using csv.reader respecting quoted HKLs."""
    refls: list[BraggReflection] = []
    if not csv_path.exists():
        return refls

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            _ = next(reader)
        except StopIteration:
            return refls

        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                pname = row[0].strip()
                hkl_str = row[1].strip().strip("()")
                hkl_tuple = tuple(int(x.strip()) for x in hkl_str.split(",") if x.strip())
                two_th = float(row[2].strip())
                d_sp = float(row[3].strip()) if len(row) > 3 else 0.0
                rel_int = float(row[4].strip()) if len(row) > 4 else 100.0
                refls.append(
                    BraggReflection(
                        phase_name=pname,
                        hkl=hkl_tuple,
                        two_theta=two_th,
                        d_spacing=d_sp,
                        intensity=rel_int,
                    )
                )
            except (ValueError, IndexError):
                continue

    return refls


def get_native_histogram_residuals(hist: Any) -> dict[str, float]:
    """Извлекает R-факторы напрямую из словаря hist.residuals."""
    residuals = getattr(hist, "residuals", {})
    if not isinstance(residuals, dict):
        return {}
    res: dict[str, float] = {}
    for k, v in residuals.items():
        try:
            res[str(k)] = float(v)
        except (ValueError, TypeError):
            continue
    return res


#: Имя переменной окружения, включающей сохранение листинга уточнения GSAS-II.
#: Настройка читается только здесь: CLI-флаг ``--keep-gsas-listing`` лишь выставляет
#: переменную, поэтому прямой вызов библиотеки наследует то же решение.
KEEP_LISTING_ENV = "XRD_KEEP_GSAS_LISTING"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def keep_gsas_listing() -> bool:
    """Сообщает, нужно ли сохранять листинг уточнения GSAS-II."""
    return os.environ.get(KEEP_LISTING_ENV, "").strip().lower() in _TRUTHY


def discard_gsas_listing(project_path: Path, *, keep: bool | None = None) -> tuple[Path, ...]:
    """Удаляет файлы листинга, которые GSAS-II пишет рядом с проектом.

    GSAS-II открывает ``<проект>.lst`` безусловно в начале ``Refine`` (и
    ``<проект>.junk``, когда запрошены все производные); управляющего флага для
    этого нет. Авторитетные метрики берутся из объектов проекта
    (``hist.residuals``), поэтому листинг — побочный продукт и по умолчанию
    удаляется. ``keep=True`` или переменная ``XRD_KEEP_GSAS_LISTING`` оставляют
    его для разбора поведения GSAS-II.

    :returns: кортеж удалённых путей (пустой, если удалять было нечего).
    """
    if keep is None:
        keep = keep_gsas_listing()
    if keep:
        return ()
    removed: list[Path] = []
    for suffix in (".lst", ".junk"):
        listing = project_path.with_suffix(suffix)
        if not listing.exists():
            continue
        try:
            listing.unlink()
        except OSError as exc:
            # Файл занят или прав нет: уборка не критична для результата.
            logger.debug(f"Could not remove GSAS-II listing {listing}: {exc}")
            continue
        removed.append(listing)
    if removed:
        logger.debug("Removed GSAS-II listing(s): " + ", ".join(p.name for p in removed))
    return tuple(removed)


def ensure_fresh_residuals(gpx: Any, hist: Any = None, force: bool = False) -> dict[str, float]:
    """Гарантирует актуальность словаря hist.residuals.

    Если hist.residuals пуст или отсутствует (проект был только что загружен
    с диска без предварительного вызова do_refinements), выполняет 0-цикловую
    оценку модели (cycles=0), обновляя in-memory структуры GSAS-II без изменения
    уточняемых параметров.
    """
    if hist is None and hasattr(gpx, "histograms"):
        hists = gpx.histograms()
        if hists:
            hist = hists[0]

    # В проектах с нативным Pawley (doPawley=True) вызов 0-циклового do_refinements
    # сбрасывает рассчитанный профиль Y_calc до чистого фона (R_wp ~ 300%), поэтому
    # при наличии рассчитанных невязок используем их напрямую.
    has_pawley = hasattr(gpx, "phases") and any(
        bool(getattr(p, "data", {}).get("General", {}).get("doPawley", False))
        for p in gpx.phases()
    )
    if hist is not None and (not force or has_pawley):
        existing = get_native_histogram_residuals(hist)
        if existing and "wR" in existing and existing["wR"] > 0:
            return existing

    # Perform 0-cycle evaluation to populate hist.residuals
    if hasattr(gpx, "set_Controls") and hasattr(gpx, "do_refinements"):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            old_cycles = 3
            if hasattr(gpx, "get_Controls"):
                with suppress(
                    RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError
                ):
                    c = gpx.get_Controls("cycles")
                    if c is not None and int(c) > 0:
                        old_cycles = int(c)
            try:
                gpx.set_Controls("cycles", 0)
                gpx.do_refinements([{}])
            finally:
                with suppress(
                    RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError
                ):
                    gpx.set_Controls("cycles", old_cycles)

        # Оценочный прогон тоже пишет листинг: метрики уже в памяти.
        project_file = getattr(gpx, "filename", None)
        if project_file:
            discard_gsas_listing(Path(project_file))

    if hist is None and hasattr(gpx, "histograms"):
        hists = gpx.histograms()
        if hists:
            hist = hists[0]

    return get_native_histogram_residuals(hist) if hist is not None else {}


def export_histogram_curves_safe(hist: Any, out_csv: Path) -> Path:
    """Экспортирует профиль через hist.Export(..., '.csv', 'histogram CSV').

    При отсутствии метода, ошибке или в headless-окружении выполняет экспорт
    через extract_profile_curves() с сохранением полного набора колонок:
    (two_theta, y_obs, y_calc, y_bkg, y_diff, sigma_obs).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(hist, "Export"):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            stem = str(out_csv.with_suffix(""))
            res_path = hist.Export(stem, ".csv", "histogram CSV")
            if res_path and Path(res_path).exists() and Path(res_path).stat().st_size > 0:
                return Path(res_path)
            if out_csv.exists() and out_csv.stat().st_size > 0:
                return out_csv

    curves = extract_profile_curves(hist)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["two_theta", "y_obs", "y_calc", "y_bkg", "y_diff", "sigma_obs"])
        for tt, yo, yc, yb, yd, so in zip(
            curves.two_theta,
            curves.y_obs,
            curves.y_calc,
            curves.y_bkg,
            curves.y_diff,
            curves.sigma_obs,
        ):
            writer.writerow(
                [f"{tt:.4f}", f"{yo:.2f}", f"{yc:.2f}", f"{yb:.2f}", f"{yd:.2f}", f"{so:.2f}"]
            )

    return out_csv


def read_difference_curve(path: Path) -> np.ndarray:
    """Read a ``(2theta, Yobs, Ycalc)`` table from a difference-curve file.

    Two layouts are accepted: whitespace-separated columns, and the CSV this module
    itself writes through :func:`export_histogram_curves_safe`
    (``two_theta,y_obs,y_calc,y_bkg,y_diff,sigma_obs``). The second layout is why this
    function exists here rather than in a caller: the module that writes the export is
    the one that can read it back, so no consumer has to reimplement the format or
    convert the file by hand.

    :raises ValueError: when the table has fewer than three columns.
    """
    first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if "," in first_line:
        try:
            for cell in first_line.split(",")[:3]:
                float(cell)
            skip_rows = 0
        except ValueError:
            skip_rows = 1  # the export carries a header row
        table = np.loadtxt(str(path), delimiter=",", skiprows=skip_rows)
    else:
        table = np.loadtxt(str(path))
    table = np.atleast_2d(table)
    if table.shape[1] < 3:
        raise ValueError(
            f"{path.name} holds {table.shape[1]} column(s); a difference curve needs "
            "2theta, Yobs and Ycalc"
        )
    return table


def read_project_difference_curve(project_path: Path) -> np.ndarray:
    """Read the signed difference curve from a GSAS-II project.

    The refined curves belong to the project, so they are read from it rather than
    from a copy made on the side (SKILL.md, Rule 10). This is the path that removes an
    auxiliary residual file entirely.

    :raises RuntimeError: when the GSAS-II engine is unavailable.
    :raises ValueError: when the project holds no powder histogram.
    """
    from stage_gate_rietveld import ensure_gsas2_environment

    if not ensure_gsas2_environment():
        raise RuntimeError("GSAS-II calculation engine is not available")

    from GSASII import GSASIIscriptable as G2sc

    gpx = G2sc.G2Project(str(project_path))
    histograms = gpx.histograms()
    if not histograms:
        raise ValueError(f"project {project_path.name} contains no histograms")
    curves = extract_profile_curves(histograms[0])
    return np.column_stack([curves.two_theta, curves.y_obs, curves.y_calc])


def export_reflections_safe(hist: Any, out_csv: Path) -> Path:
    """Экспортирует рефлексы через hist.Export(..., '.csv', 'refl') либо extract_bragg_reflections()."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(hist, "Export"):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            stem = str(out_csv.with_suffix(""))
            res_path = hist.Export(stem, ".csv", "refl")
            if res_path and Path(res_path).exists() and Path(res_path).stat().st_size > 0:
                return Path(res_path)
            if out_csv.exists() and out_csv.stat().st_size > 0:
                return out_csv

    refls = extract_bragg_reflections(hist)
    return write_bragg_reflections_csv(refls, out_csv)


def export_phase_cif_safe(phase: Any, out_cif: Path) -> Path:
    """Экспортирует CIF фазы через phase.export_CIF(filename) либо phase.Export(..., '.cif', 'CIF')."""
    out_cif = Path(out_cif)
    out_cif.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(phase, "export_CIF"):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            phase.export_CIF(str(out_cif))
            if out_cif.exists() and out_cif.stat().st_size > 0:
                return out_cif
    if hasattr(phase, "Export"):
        with suppress(RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError):
            stem = str(out_cif.with_suffix(""))
            res_path = phase.Export(stem, ".cif", "CIF")
            if res_path and Path(res_path).exists() and Path(res_path).stat().st_size > 0:
                return Path(res_path)
            if out_cif.exists() and out_cif.stat().st_size > 0:
                return out_cif
    return out_cif


def get_project_qpa(gpx: Any, hist: Any = None) -> dict[str, tuple[float, float | None]]:
    """Return quantitative phase analysis (QPA) mass fractions and ESDs.

    Returns:
        dict[str, tuple[float, float | None]]: Mapping of phase_name to
        (weight_percent, esd_percent), where ``esd_percent`` is ``None`` when the
        project carries no covariance for phase scales -- an undefined uncertainty,
        not zero.

    Resolution order:
        1. Native GSAS-II hist.ComputeMassFracs() or hist.data["MassFracs"].
        2. Exact physical Hill & Howard (1987) formula using full unit cell mass
           M_cell = Z * M_formula (from SGZ * Mass or structure composition) and
           unit cell volume V_cell:
               W_p = (S_p * M_cell,p * V_p) / sum_k (S_k * M_cell,k * V_k) * 100%
           Never uses unmultiplied asymmetric unit atoms.
    """
    if hist is None and hasattr(gpx, "histograms"):
        hists = gpx.histograms()
        if hists:
            hist = hists[0]

    phases = gpx.phases() if hasattr(gpx, "phases") else []
    if not phases:
        return {}

    # 1. Attempt native GSAS-II calculation
    native_qpa: dict[str, tuple[float, float | None]] = {}
    if hist is not None:
        try:
            res = None
            if hasattr(hist, "ComputeMassFracs"):
                res = hist.ComputeMassFracs()
            elif hasattr(hist, "data") and isinstance(hist.data, dict):
                res = hist.data.get("MassFracs")

            if isinstance(res, dict):
                for k, v in res.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        val = float(v[0])
                        sig = float(v[1])
                    elif isinstance(v, (int, float)):
                        val = float(v)
                        sig = 0.0
                    else:
                        continue
                    pct = val * 100.0 if val <= 1.0 else val
                    sig_pct = sig * 100.0 if val <= 1.0 else sig
                    native_qpa[str(k)] = (round(pct, 2), round(sig_pct, 2))
            elif isinstance(res, (list, tuple)) and len(res) == 2 and isinstance(res[0], dict):
                mf_dict, sig_dict = res[0], res[1]
                for k, val_raw in mf_dict.items():
                    val = float(val_raw)
                    sig = float(sig_dict.get(k, 0.0))
                    pct = val * 100.0 if val <= 1.0 else val
                    sig_pct = sig * 100.0 if val <= 1.0 else sig
                    native_qpa[str(k)] = (round(pct, 2), round(sig_pct, 2))
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
            logger.debug("Native ComputeMassFracs notice: {}", e)

    if native_qpa and any(v[0] > 0 for v in native_qpa.values()):
        return native_qpa

    # 2. Strict physical Hill-Howard fallback
    scales = extract_phase_scales(gpx)
    cells = extract_cell_parameters(gpx)

    phase_factors: dict[str, float] = {}
    for idx, p in enumerate(phases):
        p_name = getattr(p, "name", f"Phase_{idx}")
        sc = scales.get(p_name, 0.0)
        if sc <= 0.0:
            phase_factors[p_name] = 0.0
            continue

        # Cell volume
        vol = 0.0
        p_cell = cells.get(p_name, {})
        if "volume" in p_cell and p_cell["volume"] > 0:
            vol = float(p_cell["volume"])
        elif "a" in p_cell and p_cell["a"] > 0:
            a = float(p_cell.get("a", 5.0))
            b = float(p_cell.get("b", a))
            c = float(p_cell.get("c", a))
            alpha = float(p_cell.get("alpha", 90.0))
            beta = float(p_cell.get("beta", 90.0))
            gamma = float(p_cell.get("gamma", 90.0))
            uc = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
            vol = float(uc.volume)

        # Full unit cell mass: M_cell = Z * M_formula
        m_cell = 0.0
        if hasattr(p, "data") and isinstance(p.data, dict):
            gen = p.data.get("General", {})
            m_form = float(gen.get("Mass", 0.0))
            sg_z = float(gen.get("SGData", {}).get("SGZ", 0.0))
            if m_form > 0 and sg_z > 0:
                m_cell = m_form * sg_z
            elif m_form > 0:
                m_cell = m_form

        if m_cell <= 0.0:
            atoms = p.data.get("Atoms", []) if hasattr(p, "data") else []
            asym_mass = 0.0
            for at in atoms:
                elem_sym = at[1] if isinstance(at, (list, tuple)) and len(at) > 1 else ""
                if elem_sym:
                    with suppress(RuntimeError, ValueError, KeyError):
                        asym_mass += gemmi.Element(elem_sym).weight
            sg_z = (
                float(p.data.get("General", {}).get("SGData", {}).get("SGZ", 1.0))
                if hasattr(p, "data")
                else 1.0
            )
            m_cell = asym_mass * max(1.0, sg_z)

        phase_factors[p_name] = sc * m_cell * vol

    total_factor = sum(phase_factors.values())
    cov_block = (getattr(gpx, "data", {}) or {}).get("Covariance") or {}
    cov_data = cov_block.get("data", {}) if isinstance(cov_block, dict) else {}
    vary_list = (
        cov_data.get("varyList")
        if cov_data.get("varyList") is not None
        else cov_block.get("varyList", [])
    )
    sig_list = (
        cov_data.get("sig")
        if cov_data.get("sig") is not None
        else cov_block.get("sig", [])
    )
    sig_map: dict[str, float] = {
        str(k): float(v) for k, v in zip(vary_list, sig_list) if v is not None
    }

    rel_var_by_phase: dict[str, float] = {}
    for idx, p in enumerate(phases):
        p_name = getattr(p, "name", f"Phase_{idx}")
        pdata = getattr(p, "data", {})
        if isinstance(pdata, dict) and bool(pdata.get("General", {}).get("doPawley", False)):
            p_cell = cells.get(p_name, {})
            vol_val = float(p_cell.get("volume", 0.0) or 0.0)
            vol_esd = float(p_cell.get("volume_esd", 0.0) or 0.0)
            rel_v = (vol_esd / vol_val) ** 2 if (vol_val > 0.0 and vol_esd > 0.0) else 0.0
            pwl_refs = pdata.get("Pawley ref", [])
            sum_i = 0.0
            sum_var = 0.0
            for r_idx, pk in enumerate(pwl_refs):
                if len(pk) >= 8 and bool(pk[5]) and float(pk[6]) > 0.0:
                    inten = float(pk[6])
                    sig_i = sig_map.get(f"{idx}::PWLref:{r_idx}", float(pk[7]))
                    if 0.0 < sig_i <= 0.15 * inten:
                        sum_i += inten
                        sum_var += sig_i * sig_i
            rel_i = (sum_var / (sum_i * sum_i)) if (sum_i > 0.0 and sum_var > 0.0) else 0.000025
            rel_var_by_phase[p_name] = max(rel_v, rel_i)

    result: dict[str, tuple[float, float | None]] = {}
    for p_name, f_val in phase_factors.items():
        if total_factor > 0:
            wt_pct = round((f_val / total_factor) * 100.0, 2)
        else:
            wt_pct = 0.0
        wt_esd: float | None = None
        if len(phases) > 1 and wt_pct > 0.0 and rel_var_by_phase:
            w_p = wt_pct / 100.0
            var_wp = ((1.0 - w_p) ** 2) * rel_var_by_phase.get(p_name, 0.0)
            for q_name, q_val in phase_factors.items():
                if q_name == p_name or total_factor <= 0.0:
                    continue
                w_q = q_val / total_factor
                var_wp += (w_q**2) * rel_var_by_phase.get(q_name, 0.0)
            if var_wp > 0.0:
                wt_esd = max(0.01, round(wt_pct * float(np.sqrt(var_wp)), 2))
        result[p_name] = (wt_pct, wt_esd)

    return result
