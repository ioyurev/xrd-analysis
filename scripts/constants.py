"""Числовые константы скилла: одно значение — одно место.

Ранее эти величины были продублированы литералами (длина волны — 12 вхождений,
порог GII — объявление и литерал рядом, порог Rule 9 — расчёт и подпись таблицы),
поэтому смена калибровки или порога требовала помнить обо всех копиях. Здесь они
объявлены один раз и импортируются.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _resolve_version() -> str:
    """Read package version from pyproject.toml (SSOT)."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        proj = data.get("project")
        if isinstance(proj, dict) and "version" in proj:
            return str(proj["version"])
    return "0.0.0"


#: Версия пакета xrd-analysis, прочитанная из pyproject.toml.
VERSION: str = _resolve_version()

#: Длина волны Cu Kα1 по умолчанию для рецептов образца и диагностики, Å.
DEFAULT_WAVELENGTH_KA1: float = 1.5406

#: Табличные длины волн дублета Cu Kα1, Cu Kα2 и линии Cu Kβ для эталонного профиля прибора, Å.
CANONICAL_WAVELENGTH_KA1: float = 1.540596
CANONICAL_WAVELENGTH_KA2: float = 1.544426
CANONICAL_WAVELENGTH_KBETA: float = 1.39222

#: Отношение |сдвиг|/σ, ниже которого параметр считается сошедшимся (Rule 9).
RULE9_SHIFT_SIGMA_RATIO: float = 0.15

# Пороги брэгговских факторов по фазам (SKILL.md, критерий полного успеха).
# R_B и R_F нормированы на сумму наблюдаемых интенсивностей фазы, поэтому у
# примеси высокое значение говорит о её малой доле, а не о качестве модели:
# фазы с долей ниже порога печатаются информационно и в критерии не входят.
PHASE_MINOR_WEIGHT_PERCENT: float = 5.0
PHASE_RB_DOMINANT_PERCENT: float = 10.0
PHASE_RF_DOMINANT_PERCENT: float = 6.0
PHASE_RB_SUBDOMINANT_PERCENT: float = 15.0
PHASE_RF_SUBDOMINANT_PERCENT: float = 8.0

#: Порог массовой доли (масс. %), ниже которого ячейка микрофазы фиксируется.
PHASE_MICRO_CELL_LOCK_PERCENT: float = 2.0

#: Порог GII (v.u.), выше которого кристаллохимический шлюз отклоняет экспорт.
DEFAULT_GII_REJECT_THRESHOLD: float = 0.25

# Канонические аппаратные границы параметров МНК (Box Constraints, SSOT)
BOUND_UISO_MIN: float = 0.001
BOUND_UISO_MAX: float = 0.15
BOUND_MD_MIN: float = 0.40
BOUND_MD_MAX: float = 2.50
BOUND_DISPLACEX_MIN: float = -1000.0
BOUND_DISPLACEX_MAX: float = 1000.0
BOUND_ZERO_MIN: float = -0.5
BOUND_ZERO_MAX: float = 0.5
BOUND_SHL_MIN: float = 0.0
BOUND_SHL_MAX: float = 0.05
BOUND_SIZE_MIN: float = 0.001
BOUND_SIZE_MAX: float = 10.0
BOUND_MUSTRAIN_MIN: float = 0.0
BOUND_MUSTRAIN_MAX: float = 50000.0
BOUND_EXTINCTION_MIN: float = 0.0
BOUND_EXTINCTION_MAX: float = 1000.0
BOUND_CAGLIOTI_UW_MIN: float = 0.0
BOUND_CAGLIOTI_UW_MAX: float = 0.10
BOUND_CAGLIOTI_V_MIN: float = -0.05
BOUND_CAGLIOTI_V_MAX: float = 0.05

__all__ = [
    "BOUND_CAGLIOTI_UW_MAX",
    "BOUND_CAGLIOTI_UW_MIN",
    "BOUND_CAGLIOTI_V_MAX",
    "BOUND_CAGLIOTI_V_MIN",
    "BOUND_DISPLACEX_MAX",
    "BOUND_DISPLACEX_MIN",
    "BOUND_EXTINCTION_MAX",
    "BOUND_EXTINCTION_MIN",
    "BOUND_MD_MAX",
    "BOUND_MD_MIN",
    "BOUND_MUSTRAIN_MAX",
    "BOUND_MUSTRAIN_MIN",
    "BOUND_SHL_MAX",
    "BOUND_SHL_MIN",
    "BOUND_SIZE_MAX",
    "BOUND_SIZE_MIN",
    "BOUND_UISO_MAX",
    "BOUND_UISO_MIN",
    "BOUND_ZERO_MAX",
    "BOUND_ZERO_MIN",
    "CANONICAL_WAVELENGTH_KA1",
    "CANONICAL_WAVELENGTH_KA2",
    "CANONICAL_WAVELENGTH_KBETA",
    "DEFAULT_GII_REJECT_THRESHOLD",
    "DEFAULT_WAVELENGTH_KA1",
    "PHASE_MICRO_CELL_LOCK_PERCENT",
    "PHASE_MINOR_WEIGHT_PERCENT",
    "PHASE_RB_DOMINANT_PERCENT",
    "PHASE_RB_SUBDOMINANT_PERCENT",
    "PHASE_RF_DOMINANT_PERCENT",
    "PHASE_RF_SUBDOMINANT_PERCENT",
    "RULE9_SHIFT_SIGMA_RATIO",
    "VERSION",
]
