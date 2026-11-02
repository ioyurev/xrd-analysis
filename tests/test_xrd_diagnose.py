"""Двусторонний контроль критерия неиндицированных пиков.

Критерий обязан различать три вещи, и каждая проверяется отдельно:
чистый шум не даёт находок, внедрённое отражение находится, одиночный выброс
отбрасывается. Без первых двух проверка «стало чисто» ничего не доказывает,
без третьей — детектор считает шум.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from xrd_diagnose import ResidualPeakType, classify_residual_peaks

RNG = np.random.default_rng(20261020)
TWO_THETA = np.linspace(15.0, 60.0, 2000)
NOISE_SIGMA = 10.0


def _residual(extra: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    noise = RNG.normal(0.0, NOISE_SIGMA, TWO_THETA.size)
    residual = noise if extra is None else noise + extra
    return np.zeros_like(TWO_THETA), residual


def test_pure_noise_gives_no_peaks():
    y_calc, residual = _residual()
    unindexed, defects, _artifacts = classify_residual_peaks(
        TWO_THETA, y_calc + residual, y_calc, phase_bragg_peaks=None
    )
    assert unindexed == []
    assert defects == []


def test_injected_reflection_is_found():
    centre = 30.0
    amplitude = 8.0 * NOISE_SIGMA
    gaussian = amplitude * np.exp(-0.5 * ((TWO_THETA - centre) / 0.05) ** 2)
    y_calc, residual = _residual(gaussian)
    unindexed, _defects, _artifacts = classify_residual_peaks(
        TWO_THETA, y_calc + residual, y_calc, phase_bragg_peaks=None
    )
    assert unindexed, "внедрённое отражение не найдено"
    nearest = min(unindexed, key=lambda p: abs(p.two_theta - centre))
    assert abs(nearest.two_theta - centre) < 0.1


def test_single_channel_spike_is_rejected():
    spike = np.zeros_like(TWO_THETA)
    spike[1000] = 25.0 * NOISE_SIGMA
    y_calc, residual = _residual(spike)
    unindexed, _defects, _artifacts = classify_residual_peaks(
        TWO_THETA, y_calc + residual, y_calc, phase_bragg_peaks=None
    )
    assert all(abs(p.two_theta - TWO_THETA[1000]) > 0.1 for p in unindexed), (
        "одиночный канал принят за отражение"
    )


def test_known_line_with_weak_model_height_is_not_impurity():
    centre = 30.0
    gaussian = 8.0 * NOISE_SIGMA * np.exp(-0.5 * ((TWO_THETA - centre) / 0.05) ** 2)
    y_calc, residual = _residual(gaussian)
    # Линия есть, но её расчётный вклад нулевой — модель её не описывает.
    unindexed, defects, _artifacts = classify_residual_peaks(
        TWO_THETA, y_calc + residual, y_calc, phase_bragg_peaks={"P": [(centre, 0.0)]}
    )
    assert defects and defects[0].defect_type == ResidualPeakType.WEAK_KNOWN_LINE
    assert all(abs(p.two_theta - centre) > 0.1 for p in unindexed)


def test_shifted_known_line_is_marked_as_shift_not_phase():
    line = 30.0
    centre = 30.35
    gaussian = 8.0 * NOISE_SIGMA * np.exp(-0.5 * ((TWO_THETA - centre) / 0.05) ** 2)
    y_calc, residual = _residual(gaussian)
    unindexed, _defects, _artifacts = classify_residual_peaks(
        TWO_THETA, y_calc + residual, y_calc, phase_bragg_peaks={"P": [(line, 1e6)]}
    )
    shifted = [p for p in unindexed if p.peak_type == ResidualPeakType.SHIFTED_KNOWN_LINE]
    assert shifted, "сдвинутая известная линия не помечена"
    assert shifted[0].delta_2theta is not None
    assert abs(shifted[0].delta_2theta - 0.35) < 0.1


def test_residual_maximum_without_observed_peak_is_not_an_impurity():
    """Максимум остатка там, где в измерении пика нет, — артефакт, а не отражение.

    Измерено на серии: из 23 «неиндицированных» максимумов только 4 совпали с
    наблюдаемым пиком, остальные 19 стояли в местах, где измеренный профиль пика
    не содержит. Считать их следами примеси — значит искать фазу для того, чего
    в данных нет; правило 8 SKILL.md предупреждает об этом прямо.
    """
    centre = 30.0
    gaussian = 8.0 * NOISE_SIGMA * np.exp(-0.5 * ((TWO_THETA - centre) / 0.05) ** 2)
    y_calc, residual = _residual(gaussian)

    # Наблюдаемые пики объявлены далеко от 30°: отражения там нет.
    unindexed, _defects, artifacts = classify_residual_peaks(
        TWO_THETA,
        y_calc + residual,
        y_calc,
        phase_bragg_peaks=None,
        observed_peaks_2theta=np.array([26.5, 38.0, 44.2]),
    )

    assert unindexed == [], "максимум без наблюдаемого пика засчитан как примесь"
    assert artifacts, "артефакт остатка не назван"
    assert artifacts[0].peak_type == ResidualPeakType.RESIDUAL_ARTIFACT

    # Отрицательный контроль: если наблюдаемый пик там есть, это уже отражение.
    unindexed2, _d2, artifacts2 = classify_residual_peaks(
        TWO_THETA,
        y_calc + residual,
        y_calc,
        phase_bragg_peaks=None,
        observed_peaks_2theta=np.array([26.5, 30.0, 44.2]),
    )
    assert unindexed2, "наблюдаемое отражение потеряно"
    assert artifacts2 == []


def test_near_weak_line_beats_far_strong_line():
    """Пик в 0.04° от собственной линии фазы не является примесью.

    Если максимум остатка стоит в 0.04° от слабой расчётной линии известной
    фазы, он не должен классифицироваться как неиндицированная примесь.
    Причина была в выборе фазы по любой линии с достаточной высотой: сильная
    линия в 0.5° занимала «лучшую фазу», запирала ветку слабой близкой линии,
    и пик проваливался в «примесь». Решает ближайшая линия, а её высота
    отделяет дефект интенсивности от слабой известной линии.
    """
    centre = 30.04
    gaussian = 8.0 * NOISE_SIGMA * np.exp(-0.5 * ((TWO_THETA - centre) / 0.05) ** 2)
    y_calc, residual = _residual(gaussian)

    unindexed, defects, _artifacts = classify_residual_peaks(
        TWO_THETA,
        y_calc + residual,
        y_calc,
        # далёкая линия с огромной высотой и близкая линия с нулевой высотой
        phase_bragg_peaks={"P": [(30.55, 1e6), (centre, 0.0)]},
    )

    assert all(abs(p.two_theta - centre) > 0.2 for p in unindexed), (
        "пик у собственной линии засчитан как неиндицированная примесь"
    )
    assert defects, "близкая линия фазы не названа"
    assert defects[0].matched_phase == "P"
    assert defects[0].bragg_2theta is not None
    assert abs(defects[0].bragg_2theta - centre) < 0.05


def test_phantom_overcalculated_reflection_is_detected_as_negative_defect():
    """Фантомный расчётный пик (Y_calc >> Y_obs на месте брэгговского узла) ловится как OVER_CALCULATED_PHANTOM."""
    centre = 35.0
    phantom = 10.0 * NOISE_SIGMA * np.exp(-0.5 * ((TWO_THETA - centre) / 0.05) ** 2)
    y_obs, noise = _residual()
    y_obs = y_obs + noise
    y_calc = phantom

    unindexed, defects, _artifacts = classify_residual_peaks(
        TWO_THETA,
        y_obs,
        y_calc,
        phase_bragg_peaks={"WrongPhase": [(centre, 500.0)]},
    )
    assert unindexed == []
    assert defects, "фантомный расчётный рефлекс не обнаружен"
    assert defects[0].peak_type == ResidualPeakType.OVER_CALCULATED_PHANTOM
    assert defects[0].residual_height < 0.0
    assert defects[0].matched_phase == "WrongPhase"


def test_kbeta_companion_of_strong_line_is_classified_as_artifact_not_impurity():
    """Слабый Cu K-beta спутник (0.8% от главного пика) классифицируется как KBETA_SATELLITE, а не как примесь."""
    main_2th = 29.7325
    main_amp = 55000.0
    # Теоретический угол Cu K-beta (1.39222 A) от рефлекса 29.7325 град (1.5406 A) -> ~26.81 град
    kb_2th = 26.791
    kb_amp = 400.0  # ~0.73% от главного пика
    y_calc = 1000.0 + main_amp * np.exp(-0.5 * ((TWO_THETA - main_2th) / 0.05) ** 2)
    _, noise = _residual()
    y_obs = (
        y_calc
        + noise
        + kb_amp * np.exp(-0.5 * ((TWO_THETA - kb_2th) / 0.04) ** 2)
    )

    unindexed, _defects, artifacts = classify_residual_peaks(
        TWO_THETA,
        y_obs,
        y_calc,
        phase_bragg_peaks={"Lu2O3": [(main_2th, main_amp)]},
        observed_peaks_2theta=np.array([kb_2th, main_2th]),
    )
    assert unindexed == [], "аппаратурный Cu K-beta спутник принят за примесную фазу"
    assert any(a.peak_type == ResidualPeakType.KBETA_SATELLITE for a in artifacts)
