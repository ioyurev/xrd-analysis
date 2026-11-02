"""Критерий определённости параметров: esd должна существовать и быть положительной."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage_gate_rietveld import (
    SuccessGateInputs,
    evaluate_success_gates,
    undefined_esd_problems,
)


def test_undefined_esd_is_reported_by_name():
    problems = undefined_esd_problems({"a": 0.0, "b": float("nan"), "c": 0.01})
    assert len(problems) == 2
    assert any("a" in p for p in problems)
    assert any("b" in p for p in problems)
    assert all("c" not in p for p in problems)


def test_undefined_esd_enters_the_verdict():
    inputs = SuccessGateInputs(
        r_wp_percent=8.0,
        durbin_watson=1.9,
        max_microstrain_percent=0.2,
        max_crystallite_size_nm=120.0,
        unindexed_count=0,
        esd_problems=["1:0:Size;i: esd не определена (σ=0.0)"],
    )
    unmet = evaluate_success_gates(inputs)
    assert unmet and "esd не определена" in unmet[0]


def test_shift_significance_requires_a_defined_sigma():
    from stage_gate_rietveld import shift_significance_problems

    problems = shift_significance_problems(
        {"converged": 0.01, "drifting": 0.40, "undefined": 0.05},
        {"converged": 0.10, "drifting": 0.10, "undefined": 0.0},
        threshold=0.15,
    )
    assert len(problems) == 2
    assert any("drifting" in item and "≥" in item for item in problems)
    assert any("undefined" in item for item in problems)
    assert all("converged" not in item for item in problems)


def test_rule9_problems_enter_the_verdict():
    inputs = SuccessGateInputs(
        r_wp_percent=8.0,
        durbin_watson=1.9,
        max_microstrain_percent=0.2,
        max_crystallite_size_nm=120.0,
        unindexed_count=0,
        rule9_problems=["SmS:0:a: |Δ|/σ = 0.40 ≥ 0.15"],
    )
    unmet = evaluate_success_gates(inputs)
    assert unmet and "не подтверждён" in unmet[0]


def test_missing_rule9_data_is_not_a_pass():
    inputs = SuccessGateInputs(
        r_wp_percent=8.0,
        durbin_watson=1.9,
        max_microstrain_percent=0.2,
        max_crystallite_size_nm=120.0,
        unindexed_count=0,
        rule9_measured=False,
    )
    unmet = evaluate_success_gates(inputs)
    assert unmet and "не измерялась" in unmet[0]


def test_rb_criterion_ignores_minor_phases():
    inputs = SuccessGateInputs(
        r_wp_percent=8.0,
        durbin_watson=1.9,
        max_microstrain_percent=0.2,
        max_crystallite_size_nm=120.0,
        unindexed_count=0,
        rb_by_phase={"SmS": 6.0, "Sm2O2S": 40.0},
        phase_weights_percent={"SmS": 88.0, "Sm2O2S": 3.0},
    )
    assert evaluate_success_gates(inputs) == []

    inputs = SuccessGateInputs(
        r_wp_percent=8.0,
        durbin_watson=1.9,
        max_microstrain_percent=0.2,
        max_crystallite_size_nm=120.0,
        unindexed_count=0,
        rb_by_phase={"SmS": 12.0, "Sm2O2S": 40.0},
        phase_weights_percent={"SmS": 88.0, "Sm2O2S": 3.0},
    )
    unmet = evaluate_success_gates(inputs)
    assert len(unmet) == 1 and "R_B(SmS)" in unmet[0]


def test_two_tier_limits_for_dominant_and_subdominant_phases():
    base = {
        "r_wp_percent": 8.0,
        "durbin_watson": 1.9,
        "max_microstrain_percent": 0.2,
        "max_crystallite_size_nm": 120.0,
        "unindexed_count": 0,
        "phase_weights_percent": {"SmS": 80.0, "Sm3S4": 15.0},
    }
    ok = SuccessGateInputs(
        **base,
        rb_by_phase={"SmS": 10.0, "Sm3S4": 15.0},
        rf_by_phase={"SmS": 6.0, "Sm3S4": 8.0},
    )
    assert evaluate_success_gates(ok) == []

    bad = SuccessGateInputs(
        **base,
        rb_by_phase={"SmS": 11.0, "Sm3S4": 16.0},
        rf_by_phase={"SmS": 6.5, "Sm3S4": 8.5},
    )
    assert len(evaluate_success_gates(bad)) == 4


def test_instrument_profile_stage_exists_and_refines_caglioti_terms():
    """Стадия инструментального профиля заведена и уточняет U, V, W.

    До неё все три коэффициента стояли на значениях по умолчанию и не
    уточнялись ни разу: скилл не умел трогать ту часть профиля, которую задаёт
    прибор, а не образец.
    """
    from stage_gate_rietveld import (
        CANONICAL_STAGES,
        build_stage_recipe,
        normalize_stage_name,
    )

    assert "10_Instrument_Profile" in CANONICAL_STAGES
    recipe = build_stage_recipe("10_Instrument_Profile")
    assert recipe == {"Instrument Parameters": ["U", "V", "W"]}
    assert build_stage_recipe("10_Instrument_Profile", refine_axial=True) == {
        "Instrument Parameters": ["U", "V", "W", "SH/L"]
    }

    # Псевдонимы: «instrument» не должен уводить к стадии 7 по слову «profile».
    assert normalize_stage_name("10") == "10_Instrument_Profile"
    assert normalize_stage_name("instrument profile") == "10_Instrument_Profile"
    assert normalize_stage_name("Caglioti") == "10_Instrument_Profile"
    assert normalize_stage_name("7") == "7_Sample_Profile_XY"


def test_stored_microstrain_beyond_its_bound_is_clamped():
    """Значение за границей должно исправляться, а не наследоваться.

    Граница parmMin/parmMax ограничивает следующий шаг оптимизатора и не возвращает
    назад уже записанное. Измерено на реальном проекте: у фазы лежало
    Mustrain = -1124.9 при refine = True, и значение переезжало из стадии в стадию.
    Отрицательное микронапряжение физически невозможно.
    """
    from stage_gate_rietveld import apply_box_constraints

    class _Hist:
        name = "PWDR test"

    class _Phase:
        def __init__(self) -> None:
            self.name = "Sm3S4"
            self.data = {
                "Atoms": [],
                "Histograms": {
                    "PWDR test": {
                        "Mustrain": ["isotropic", [-1124.9, 1000.0, 1.0], [True, False, False]],
                        "Size": ["isotropic", [0.0002, 1.0, 1.0], [True, False, False]],
                    }
                },
            }

        def atoms(self):
            return []

    class _Project:
        def __init__(self, phase, hist) -> None:
            self._phase = phase
            self._hist = hist
            self.controls = {"parmMin": {}, "parmMax": {}}

        def phases(self):
            return [self._phase]

        def histograms(self):
            return [self._hist]

        def set_Controls(self, control, value, variable=None):
            self.controls[control][variable] = value

    phase, hist = _Phase(), _Hist()
    gpx = _Project(phase, hist)

    apply_box_constraints(gpx, phase_id=0, hist_id=0)

    hap = phase.data["Histograms"]["PWDR test"]
    assert hap["Mustrain"][1][0] == 0.0
    assert hap["Size"][1][0] == 0.001


def test_absorption_and_extinction_stage_releases_both_models():
    """Поглощение и экстинкция освобождаются вместе, но в разных блоках.

    Обе величины систематически занижают интегральные интенсивности — именно то,
    что читает R_B, — но принадлежат разным моделям: поглощение описывает образец,
    экстинкция — фазу. Стадия заведена, чтобы агент не подменял одну другой.
    """
    from stage_gate_rietveld import (
        CANONICAL_STAGES,
        build_stage_recipe,
        normalize_stage_name,
    )

    assert "11_Absorption_Extinction" in CANONICAL_STAGES
    assert build_stage_recipe("11_Absorption_Extinction") == {
        "Sample Parameters": ["Absorption"],
        "Extinction": True,
    }
    assert normalize_stage_name("11") == "11_Absorption_Extinction"
    assert normalize_stage_name("absorption") == "11_Absorption_Extinction"
    assert normalize_stage_name("Extinction") == "11_Absorption_Extinction"


def test_negative_extinction_is_clamped():
    """Отрицательная экстинкция невозможна: она означала бы усиление отражений.

    Измерено на сошедшейся модели: стадия 11 дала Extinction = -25.1 и -0.9 у двух
    фаз, а R_wp не улучшился (19.18 -> 19.25 %). Такое значение — наследие
    вырожденного уточнения, а не подобранная величина.
    """
    from stage_gate_rietveld import clamp_stored_parameters

    class _Hist:
        name = "PWDR test"

    class _Phase:
        name = "Sm3S4"

        def __init__(self) -> None:
            self.data = {
                "Histograms": {
                    "PWDR test": {"Extinction": [-25.14, True]},
                }
            }

    phase = _Phase()
    changed = clamp_stored_parameters(object(), phase, _Hist())

    assert phase.data["Histograms"]["PWDR test"]["Extinction"][0] == 0.0
    assert any("Extinction" in item for item in changed)


def test_final_scale_stage_refines_scales_only():
    """Финальная стадия освобождает только масштабы: фон сюда не входит.

    Она существует ради ковариации последнего уточнения — без масштабов в varyList
    у массовых долей нет esd. Совместное освобождение фона с масштабами измеренно
    ухудшало модель: R_wp 8.14 → 9.21 % на сошедшемся образце.
    """
    from stage_gate_rietveld import (
        CANONICAL_STAGES,
        build_stage_recipe,
        normalize_stage_name,
    )

    assert "12_Final_Scale" in CANONICAL_STAGES
    assert build_stage_recipe("12_Final_Scale") == {"Scale": True}
    assert "Background" not in build_stage_recipe("12_Final_Scale")
    assert normalize_stage_name("12") == "12_Final_Scale"
    assert normalize_stage_name("final_scale") == "12_Final_Scale"


def test_cell_drift_guard_is_declared_in_the_stage_path():
    """Защита от ухода ячейки объявлена в стадийном маршруте и называет порог.

    Измерено: у минорной фазы ячейка ушла с a=3.8811, c=6.6968 на a=3.7281,
    c=7.7183 (-4 %, +15 %), её доля выросла с 3.85 до 34 %, и R_wp застрял на
    7.38 % вместо 5.68 %. Данные не ограничивают ячейку примеси, и оптимизатор
    использует её как убежище; такой шаг обязан откатываться.
    """
    import inspect

    import xrd_project

    source = inspect.getsource(xrd_project._run_stage_in_project)
    assert "3_Unit_Cell" in source
    assert "different phase, not a better model" in source
    assert "CELL_DRIFT_LIMIT" in source


def test_decide_next_pass_uses_improvement_not_absolute_difference():
    """Правило остановки смотрит на улучшение, а не на |ΔR_wp|.

    Проход, ухудшивший модель, не является плато: измерено 8.17 → 9.24 % после
    переоткрытия масштабов, и критерий по модулю разности принял бы это за
    схождение (|Δ| = 1.07 % > 0.1 % — ещё нет, но при 0.05 % принял бы).
    """
    from stage_gate_rietveld import (
        PASS_CONTINUE,
        PASS_PLATEAU,
        PASS_REGRESSED,
        decide_next_pass,
    )

    assert decide_next_pass([20.0], 0.1)[0] == PASS_CONTINUE
    assert decide_next_pass([46.0, 20.0], 0.1)[0] == PASS_CONTINUE
    assert decide_next_pass([19.5, 19.45], 0.1)[0] == PASS_PLATEAU
    decision, best = decide_next_pass([8.18, 9.24], 0.1)
    assert decision == PASS_REGRESSED
    assert best == 0
    # Лучший проход — не обязательно последний.
    decision, best = decide_next_pass([20.0, 19.0, 19.05], 0.01)
    assert decision == PASS_REGRESSED and best == 1
