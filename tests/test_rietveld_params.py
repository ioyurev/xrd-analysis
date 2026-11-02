"""Unit tests for Declarative Refinement-as-Code engine (rietveld_params.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pymatgen.core import Structure
from rietveld_params import (
    apply_stage_preset_to_dict,
    format_parameter_table,
    refine_project_with_params,
    solve_sequential_scales,
)
from stage_gate_rietveld import ensure_gsas2_environment


@pytest.fixture(scope="module")
def gsas2_available():
    """Check GSAS-II availability."""
    return ensure_gsas2_environment()


def test_preset_rules():
    """Verify Stage-Gate preset rules on a synthetic parameter dictionary."""
    mock_dict = {
        "histogram": {
            "background": {"refine": False},
            "instrument": {"Zero": {"refine": False}},
            "sample": {"DisplaceX": {"refine": False}},
        },
        "phases": {
            "MainPhase": {
                "weight_pct": 96.5,
                "scale": {"refine": False},
                "cell": {"refine": False},
                "profile": {"size": {"refine": False}, "mustrain": {"refine": False}},
                "preferred_orientation": {"refine": False},
                "atoms": {"refine_xyz": False, "refine_uiso": False},
            },
            "MicroPhase": {
                "weight_pct": 3.5,
                "scale": {"refine": False},
                "cell": {"refine": False},
                "profile": {"size": {"refine": False}, "mustrain": {"refine": False}},
                "preferred_orientation": {"refine": False},
                "atoms": {"refine_xyz": False, "refine_uiso": False},
            },
            "TracePhase": {
                "weight_pct": 0.8,
                "scale": {"refine": False},
                "cell": {"refine": False},
                "profile": {"size": {"refine": False}, "mustrain": {"refine": False}},
                "preferred_orientation": {"refine": False},
                "atoms": {"refine_xyz": False, "refine_uiso": False},
            },
        },
    }

    # Preset 1: Scale & Background
    d1 = json.loads(json.dumps(mock_dict))
    apply_stage_preset_to_dict(d1, 1)
    assert d1["histogram"]["background"]["refine"] is True
    assert d1["phases"]["MainPhase"]["scale"]["refine"] is True
    assert d1["phases"]["MicroPhase"]["scale"]["refine"] is True
    assert d1["phases"]["MainPhase"]["cell"]["refine"] is False

    # Preset 3: Unit Cell (Main & Micro >= 2% get Cell=True; Trace < 2% gets Cell=False)
    d3 = json.loads(json.dumps(mock_dict))
    apply_stage_preset_to_dict(d3, 3)
    assert d3["phases"]["MainPhase"]["cell"]["refine"] is True
    assert d3["phases"]["MicroPhase"]["cell"]["refine"] is True
    assert d3["phases"]["TracePhase"]["cell"]["refine"] is False
    assert "MICROPHASE" in d3["phases"]["TracePhase"]["cell"].get("comment", "")

    # Preset 7: Profile (Main & Micro >= 2% get Size/Mustrain=True; Trace < 2% get False)
    d7 = json.loads(json.dumps(mock_dict))
    apply_stage_preset_to_dict(d7, 7)
    assert d7["phases"]["MainPhase"]["profile"]["size"]["refine"] is True
    assert d7["phases"]["MainPhase"]["profile"]["mustrain"]["refine"] is True
    assert d7["phases"]["TracePhase"]["profile"]["size"]["refine"] is False
    assert d7["phases"]["TracePhase"]["profile"]["mustrain"]["refine"] is False

    # Preset 10: Instrument Profile (must NOT trigger preset 1 due to substring '1')
    d10 = json.loads(json.dumps(mock_dict))
    d10["histogram"]["instrument"]["U"] = {"refine": False}
    d10["histogram"]["instrument"]["V"] = {"refine": False}
    d10["histogram"]["instrument"]["W"] = {"refine": False}
    apply_stage_preset_to_dict(d10, 10)
    assert d10["histogram"]["background"]["refine"] is False
    assert d10["histogram"]["instrument"]["U"]["refine"] is True
    assert d10["histogram"]["instrument"]["V"]["refine"] is True
    assert d10["histogram"]["instrument"]["W"]["refine"] is True

    # Preset 12: Final Scale (scales True, background locked False)
    d12 = json.loads(json.dumps(mock_dict))
    d12["histogram"]["background"]["refine"] = True
    apply_stage_preset_to_dict(d12, 12)
    assert d12["histogram"]["background"]["refine"] is False
    assert d12["phases"]["MainPhase"]["scale"]["refine"] is True


def test_parameter_table_formatting():
    """Verify ASCII parameter map table renders cleanly without exceptions."""
    sample_doc = {
        "histogram": {
            "instrument": {"Zero": {"value": 0.5176, "refine": True, "esd": 0.003}},
            "background": {"refine": True, "n_terms": 6},
        },
        "phases": {
            "Sm(GdS2)2": {
                "weight_pct": 96.0,
                "scale": {"value": 0.233, "refine": True, "esd": 0.002},
                "cell": {"refine": True, "a": 8.5158, "esds": {"a": 0.0016}},
                "profile": {
                    "size": {"value": 0.12, "refine": True},
                    "mustrain": {"value": 520.0, "refine": True},
                },
            },
            "B-Sm2O3": {
                "weight_pct": 4.0,
                "scale": {"value": 0.012, "refine": True, "esd": 0.001},
                "cell": {"refine": False, "a": 14.197, "esds": {}},
                "profile": {
                    "size": {"value": 0.50, "refine": False},
                    "mustrain": {"value": 500.0, "refine": False},
                },
            },
        },
    }
    table = format_parameter_table(sample_doc)
    assert "REFINEMENT PARAMETER MAP" in table
    assert "Sm(GdS2)2" in table
    assert "B-Sm2O3" in table
    assert "LOCKED" in table
    assert "ACTIVE" in table


def test_declarative_workbench_e2e(gsas2_available, tmp_path):
    """E2E test: dump parameters from real CeO2 project, modify JSON, refine with params."""
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    import argparse

    from xrd_project import cmd_add_phase, cmd_init, cmd_params

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    xy_file = fixtures_dir / "patterns" / "CeO2.xy"
    cif_file = fixtures_dir / "cifs" / "CeO2.cif"

    if not xy_file.exists() or not cif_file.exists():
        pytest.skip("Test fixtures CeO2_114 not available.")

    project_gpx = tmp_path / "test_decl.gpx"
    params_json = tmp_path / "test_decl_params.json"

    # 1. Initialize project
    args_init = argparse.Namespace(
        data=xy_file,
        project=project_gpx,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    assert cmd_init(args_init) == 0

    # 2. Add phase
    args_add = argparse.Namespace(
        project=project_gpx,
        cif=cif_file,
        name="CeO2",
        scale=0.05,
    )
    assert cmd_add_phase(args_add) == 0

    # 3. Dump params with preset 1
    args_dump = argparse.Namespace(
        action="dump",
        project=project_gpx,
        out=params_json,
        preset="1",
    )
    assert cmd_params(args_dump) == 0
    assert params_json.exists()

    with open(params_json, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["histogram"]["background"]["refine"] is True
    assert doc["phases"]["CeO2"]["scale"]["refine"] is True
    assert doc["phases"]["CeO2"]["cell"]["refine"] is False

    # 4. Refine Stage 1 via params refine
    args_refine = argparse.Namespace(
        action="refine",
        project=project_gpx,
        params=params_json,
        out=params_json,
        cycles=2,
        sequential_scales=False,
    )
    assert cmd_params(args_refine) == 0

    # 5. Enable unit cell refinement and test volume sync
    with open(params_json, "r", encoding="utf-8") as f:
        doc = json.load(f)
    doc["phases"]["CeO2"]["cell"]["refine"] = True
    # Test slightly adjusted starting parameter a
    doc["phases"]["CeO2"]["cell"]["a"] = 5.4120
    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    assert cmd_params(args_refine) == 0

    # Check updated JSON has refined value and cell volume synced
    with open(params_json, "r", encoding="utf-8") as f:
        updated = json.load(f)
    refined_a = updated["phases"]["CeO2"]["cell"]["a"]
    assert 5.40 < refined_a < 5.43
    assert updated["phases"]["CeO2"]["cell"]["volume"] > 150.0


# ---------------------------------------------------------------------------
# solve_sequential_scales: damped scale factorization
# ---------------------------------------------------------------------------
class _MockHistogram:
    """Minimal G2PwdrData stand-in exposing only what the factorizer touches."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.SampleParameters: dict[str, Any] = {"Scale": [1.0, True]}
        self.data: dict[str, Any] = {
            "Background": [["chebyschev-1", True, 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {}]
        }


class _MockPhase:
    """Minimal G2Phase stand-in holding a HAP 'Scale' entry per histogram."""

    def __init__(self, name: str, hist_name: str, scale: float) -> None:
        self.name = name
        self.data: dict[str, Any] = {
            "General": {"Cell": [False, 5.0, 5.0, 5.0, 90.0, 90.0, 90.0, 125.0]},
            "Histograms": {hist_name: {"Scale": [scale, False]}},
        }


class _MockProject:
    """Minimal G2Project stand-in recording refine ordering and box constraints."""

    def __init__(
        self,
        phases: list[_MockPhase],
        hist: _MockHistogram,
        histograms: list[_MockHistogram] | None = None,
    ) -> None:
        self._phases = phases
        self.hist = hist
        self._hists = [hist] if histograms is None else histograms
        self.refine_log: list[list[str]] = []
        self.background_log: list[bool] = []
        self.cycles_log: list[int] = []
        self.limits: dict[str, dict[str, float]] = {"parmMin": {}, "parmMax": {}}

    def histograms(self) -> list[_MockHistogram]:
        return self._hists

    def phases(self) -> list[_MockPhase]:
        return self._phases

    def set_Controls(self, control: str, value: Any, variable: str | None = None) -> None:
        if control == "cycles":
            self.cycles_log.append(int(value))
            return
        assert variable is not None, "GSAS-II requires variable= for parmMin/parmMax"
        self.limits[control][variable] = float(value)

    def refine(self, makeBack: bool = False) -> None:
        hist_name = self.hist.name
        active = [p.name for p in self._phases if p.data["Histograms"][hist_name]["Scale"][1]]
        self.refine_log.append(active)
        self.background_log.append(bool(self.hist.data["Background"][0][1]))
        for p in self._phases:
            if p.data["Histograms"][hist_name]["Scale"][1]:
                p.data["Histograms"][hist_name]["Scale"][0] += 0.01


def test_solve_sequential_scales_factorization_order():
    """Verify A -> B(xN) -> C -> D ordering, frozen background and non-negative scale pins."""
    hist = _MockHistogram("PWDR sample.csv")
    phases = [
        _MockPhase("Matrix", hist.name, 0.0),
        _MockPhase("Minor1", hist.name, 0.0),
        _MockPhase("Minor2", hist.name, 0.0),
    ]
    gpx = _MockProject(phases, hist)

    refined = solve_sequential_scales(gpx, hist_idx=0, cycles_per_step=3)

    # Histogram scale is pinned to break histogram/phase scale collinearity.
    assert hist.SampleParameters["Scale"] == [1.0, False]

    # A: matrix alone; B: each minor alone; C: joint; D: joint with unfrozen background.
    assert gpx.refine_log == [
        ["Matrix"],
        ["Minor1"],
        ["Minor2"],
        ["Matrix", "Minor1", "Minor2"],
        ["Matrix", "Minor1", "Minor2"],
    ]
    assert gpx.background_log == [False, False, False, False, True]
    assert gpx.cycles_log == [3, 3, 3, 3, 3]

    # Unit cells stay locked through the whole sequence: releasing the matrix cell
    # while the scales are still being factorized produced
    # "Invalid cell metric tensor for phase #0".
    assert all(p.data["General"]["Cell"][0] is False for p in phases)

    # Every phase scale carries a native parmMin >= 0 box constraint.
    assert gpx.limits["parmMin"] == {"0:0:Scale": 0.0, "1:0:Scale": 0.0, "2:0:Scale": 0.0}

    # Start scales: matrix 0.15, minors 0.005; +0.01 for each active refine pass.
    assert refined == pytest.approx({"Matrix": 0.18, "Minor1": 0.035, "Minor2": 0.035})
    assert all(value > 0.0 for value in refined.values())


def test_solve_sequential_scales_release_matrix_cell_is_opt_in():
    """Verify the matrix cell is released only when explicitly requested."""
    hist = _MockHistogram("PWDR sample.csv")
    phases = [_MockPhase("Matrix", hist.name, 0.0), _MockPhase("Minor", hist.name, 0.0)]
    gpx = _MockProject(phases, hist)

    solve_sequential_scales(gpx, release_matrix_cell=True)

    assert phases[0].data["General"]["Cell"][0] is True
    assert phases[1].data["General"]["Cell"][0] is False


def test_solve_sequential_scales_rejects_degenerate_projects():
    """Verify explicit failure for projects without histograms, phases or a valid index."""
    hist = _MockHistogram("PWDR sample.csv")

    with pytest.raises(ValueError, match="no histograms"):
        solve_sequential_scales(
            _MockProject([_MockPhase("Matrix", hist.name, 0.0)], hist, histograms=[])
        )

    with pytest.raises(ValueError, match="no phases"):
        solve_sequential_scales(_MockProject([], hist))

    with pytest.raises(ValueError, match="out of range"):
        solve_sequential_scales(
            _MockProject([_MockPhase("Matrix", hist.name, 0.0)], hist), hist_idx=5
        )


def test_refine_project_with_params_propagates_sequential_scales():
    """Verify the factorization flag exists, defaults to False and is documented."""
    import inspect

    parameters = inspect.signature(refine_project_with_params).parameters
    assert "sequential_scales" in parameters
    assert parameters["sequential_scales"].default is False
    assert "solve_sequential_scales" in inspect.getsource(refine_project_with_params)


# ---------------------------------------------------------------------------
# Regressions: instrument slots, declared limits, restraints and JSON output
# ---------------------------------------------------------------------------
def test_degenerate_cell_is_refused_and_a_physical_one_passes():
    """A collapsed cell must be reported; a physical cell and a real shift must not.

    A free cell on a phase the data cannot constrain collapses towards zero; the reflections
    then leave the data range and the scale explodes, which can *improve* R_wp while the
    model is meaningless (measured: a = 0.008 A with a 100 % fraction at R_wp 8.99 %).
    This is the check that stops such a fit from being reported as a success.

    Built on a stand-in project rather than on GSAS-II on purpose: what is tested is the
    refusal rule, not the solver that walks into the degeneracy.
    """
    from rietveld_params import _cell_degeneracy

    class _Phase:
        def __init__(self, name: str, cell: list[object]) -> None:
            self.name = name
            self.data = {"General": {"Cell": cell}}

    class _Project:
        def __init__(self, phases: list[_Phase]) -> None:
            self._phases = phases

        def phases(self) -> list[_Phase]:
            return self._phases

    declared = {"phases": {"SmS": {"cell": {"a": 5.95, "b": 5.95, "c": 5.95}}}}

    collapsed = _Project([_Phase("SmS", [True, 0.008, 0.008, 0.008, 90, 90, 90, 0.0])])
    assert "Degenerate cell" in _cell_degeneracy(declared, collapsed)

    healthy = _Project([_Phase("SmS", [True, 5.9531, 5.9531, 5.9531, 90, 90, 90, 210.9])])
    assert _cell_degeneracy(declared, healthy) == ""

    # A solid-solution or thermal shift of a few tenths of a percent is a result, not a
    # degeneracy, and must stay inside the band.
    shifted = _Project([_Phase("SmS", [True, 5.90, 5.90, 5.90, 90, 90, 90, 205.0])])
    assert _cell_degeneracy(declared, shifted) == ""


def test_unapplied_declarations_are_named_and_matching_ones_are_silent():
    """A declared flag the project does not hold must be named; a matching one must not.

    The document is the specification of the model. A flag that silently fails to reach the
    project locks a parameter the caller asked to refine, and the only trace is a zero esd
    discovered much later. Built on a stand-in project, because what is tested is the
    comparison, not the solver.
    """
    from rietveld_params import _unapplied_declarations

    class _Histogram:
        name = "PWDR test"

    class _Phase:
        def __init__(self, name: str, scale_refine: bool, size_refine: bool) -> None:
            self.name = name
            self.data = {
                "Histograms": {
                    "PWDR test": {
                        "Scale": [1.0, scale_refine],
                        "Size": ["isotropic", [0.04, 1.0, 1.0], [size_refine, False, False]],
                    }
                }
            }

    class _Project:
        def __init__(self, phases: list[_Phase]) -> None:
            self._phases = phases

        def phases(self) -> list[_Phase]:
            return self._phases

        def histograms(self) -> list[_Histogram]:
            return [_Histogram()]

    declared = {
        "phases": {
            "SmS": {
                "scale": {"refine": True},
                "profile": {"size": {"refine": False}},
            }
        }
    }

    matching = _Project([_Phase("SmS", scale_refine=True, size_refine=False)])
    assert _unapplied_declarations(declared, matching) == []

    # The project kept the cell-like size flag locked while the document asked to refine it.
    mismatched = _Project([_Phase("SmS", scale_refine=True, size_refine=False)])
    declared_asking_size = {
        "phases": {"SmS": {"scale": {"refine": True}, "profile": {"size": {"refine": True}}}}
    }
    problems = _unapplied_declarations(declared_asking_size, mismatched)
    assert len(problems) == 1
    assert "SmS.Size" in problems[0]


def test_pinned_parameters_are_reported_and_gentle_moves_are_not():
    """A free parameter left on its bound is reported; one barely moved there is not.

    Measured: the Lorentzian terms ended at X = 1.00000, Y = -1.00000, Z = 1.00000 while
    R_wp moved from 9.19 to 9.22 %, and the fit was reported as successful. A value on its
    bound is not a fitted quantity. The travel threshold keeps an honest boundary solution
    (the asymmetry term resting on its floor) out of the report.
    """
    from rietveld_params import _pinned_declarations

    class _Histogram:
        def __init__(self, record: dict[str, list[object]]) -> None:
            self.data = {"Instrument Parameters": [record]}

    class _Project:
        def __init__(self, record: dict[str, list[object]]) -> None:
            self._histogram = _Histogram(record)

        def histograms(self) -> list[_Histogram]:
            return [self._histogram]

    def document(value: float, refine: bool = True, low: float = -1.0, high: float = 1.0) -> dict:
        return {
            "histogram": {
                "instrument": {"X": {"value": value, "refine": refine, "min": low, "max": high}}
            }
        }

    # The measured case: a declared-free coefficient parked on its upper bound.
    pinned = _pinned_declarations(document(0.0), _Project({"X": [0.0, 1.0, True]}))
    assert len(pinned) == 1
    assert "X pinned at its max bound" in pinned[0]

    # An interior optimum is not a degeneracy.
    assert _pinned_declarations(document(0.0), _Project({"X": [0.0, 0.3, True]})) == []

    # A parameter the caller left locked is not judged at all.
    assert (
        _pinned_declarations(document(0.0, refine=False), _Project({"X": [0.0, 1.0, True]})) == []
    )

    # Resting on a bound after barely moving is an honest boundary solution.
    landed = document(0.0008, low=0.0005, high=0.05)
    assert _pinned_declarations(landed, _Project({"X": [0.0, 0.0005, True]})) == []


def _ceo2_project(tmp_path: Path) -> Path:
    """Build a CeO2 project from the skill's own fixtures and return its path."""
    import argparse

    from xrd_project import cmd_add_phase, cmd_init

    fixtures = Path(__file__).resolve().parent / "fixtures"
    project = tmp_path / "regression.gpx"
    assert (
        cmd_init(
            argparse.Namespace(
                data=fixtures / "patterns" / "CeO2.xy",
                project=project,
                instrument="canonical",
                wavelength=1.5406,
                bkg_terms=6,
            )
        )
        == 0
    )
    assert (
        cmd_add_phase(
            argparse.Namespace(
                project=project,
                cif=fixtures / "cifs" / "CeO2.cif",
                name="CeO2",
                scale=0.05,
            )
        )
        == 0
    )
    return project


def test_declared_cell_refinement_survives_scale_factorization(gsas2_available, tmp_path):
    """A declared cell refinement must outlive the scale factorization.

    Steps A-C freeze every phase cell, because a free cell and a free scale are
    collinear while the scales are being factored. That freeze used to outlive the step
    that needed it, so a phase whose cell the caller had declared refined came back
    locked, with an esd of zero and a lattice that never moved -- the declaration was
    accepted and then quietly not honoured.
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from rietveld_params import solve_sequential_scales

    gpx = G2sc.G2Project(str(_ceo2_project(tmp_path)))
    phase = gpx.phases()[0]

    # The project as built leaves the cell locked. Declaring it refined is what a caller
    # document does, and is the state the factorization is required to give back.
    phase.data["General"]["Cell"][0] = True
    solve_sequential_scales(gpx, hist_idx=0)

    assert phase.data["General"]["Cell"][0] is True, (
        "the factorization borrowed the cell flag and never returned it"
    )


def test_document_cell_flag_reaches_the_project_through_a_refinement(gsas2_available, tmp_path):
    """The declarative document must be the model that is actually refined.

    This is the end-to-end form of the same rule: dump a document, declare a cell
    refinement in it, refine through the sequential-scales route, and read the flag back.
    The route is the one the CLI uses, and the flag has to survive it.
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from rietveld_params import dump_project_params, refine_project_with_params

    project = _ceo2_project(tmp_path)
    gpx = G2sc.G2Project(str(project))
    document = dump_project_params(gpx, hist_idx=0)
    document["phases"]["CeO2"]["cell"]["refine"] = True

    updated, diverged, reason = refine_project_with_params(
        gpx,
        document,
        cycles=2,
        project_path=project,
        hist_idx=0,
        sequential_scales=True,
    )

    assert not diverged, reason
    assert updated["phases"]["CeO2"]["cell"]["refine"] is True, (
        "the document declared a cell refinement that the refinement run did not honour"
    )


def test_instrument_slot_round_trip(gsas2_available, tmp_path):
    """dump -> apply -> dump must preserve instrument values and refine flags.

    GSAS-II stores each instrument entry as a 3-field record [init, value, refine].
    Reading or writing field 0 silently returns the stale initial value and
    overwrites the effective one with a bool, which is how a refinement flag once
    became a peak-shape value.
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from rietveld_params import apply_params_to_project, dump_project_params, read_inst_param

    gpx = G2sc.G2Project(str(_ceo2_project(tmp_path)))
    doc = dump_project_params(gpx)
    doc["histogram"]["instrument"]["Zero"].update({"value": 0.123, "refine": True})
    doc["histogram"]["instrument"]["SH/L"].update({"value": 0.011, "refine": False})

    # The six Caglioti coefficients travel through the document as well. The Lorentzian
    # triple (X, Y, Z) had no route through it at all, so the tails of the instrumental
    # profile could not be refined while the residuals complained about exactly them.
    caglioti = [
        ("U", 0.0081, True),
        ("V", -0.0032, False),
        ("W", 0.0063, True),
        ("X", 0.0011, True),
        ("Y", 0.0022, False),
        ("Z", 0.0033, False),
    ]
    for coefficient, value, refine in caglioti:
        doc["histogram"]["instrument"][coefficient].update({"value": value, "refine": refine})

    apply_params_to_project(gpx, doc)
    again = dump_project_params(gpx)

    assert again["histogram"]["instrument"]["Zero"]["value"] == pytest.approx(0.123, abs=1e-9)
    assert again["histogram"]["instrument"]["Zero"]["refine"] is True
    assert again["histogram"]["instrument"]["SH/L"]["value"] == pytest.approx(0.011, abs=1e-9)
    assert again["histogram"]["instrument"]["SH/L"]["refine"] is False

    for coefficient, value, refine in caglioti:
        slot = again["histogram"]["instrument"][coefficient]
        assert slot["value"] == pytest.approx(value, abs=1e-9), coefficient
        assert slot["refine"] is refine, coefficient

    raw = gpx.histograms()[0].data["Instrument Parameters"][0]
    assert len(raw["Zero"]) == 3
    zero_value, zero_refine = read_inst_param(raw, "Zero", 0.0)
    assert zero_value == pytest.approx(0.123, abs=1e-9)
    assert zero_refine is True


def test_declared_limits_bind_the_parameter(gsas2_available, tmp_path):
    """Every min/max declared in params.json must read back through get_Controls.

    Guards the mutable-hash defect: a limits dict loaded from disk holds G2VarObj
    keys whose hash changes once index_ids() resolves the rId tables, so the stale
    entry shadows the freshly written one (a declared displacement of -5.0 read
    back as -1000.0).
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from rietveld_params import apply_params_to_project, dump_project_params

    gpx = G2sc.G2Project(str(_ceo2_project(tmp_path)))
    doc = dump_project_params(gpx)
    doc["histogram"]["instrument"]["Zero"].update({"min": -0.1, "max": 0.1})
    doc["histogram"]["instrument"]["SH/L"].update({"min": 0.0, "max": 0.05})
    doc["histogram"]["sample"]["DisplaceX"].update({"min": -5.0, "max": 5.0})
    phase = next(iter(doc["phases"].values()))
    phase["profile"]["size"].update({"min": 5.0, "max": 6.0})
    phase["profile"]["mustrain"].update({"min": 0.0, "max": 300.0})
    phase["preferred_orientation"].update({"min": 0.7, "max": 1.3})

    apply_params_to_project(gpx, doc)

    expected = {
        ("parmMin", ":0:Zero"): -0.1,
        ("parmMax", ":0:Zero"): 0.1,
        ("parmMin", ":0:SH/L"): 0.0,
        ("parmMax", ":0:SH/L"): 0.05,
        ("parmMin", ":0:DisplaceX"): -5.0,
        ("parmMax", ":0:DisplaceX"): 5.0,
        ("parmMin", "0:0:Size;i"): 5.0,
        ("parmMax", "0:0:Size;i"): 6.0,
        ("parmMin", "0:0:Mustrain;i"): 0.0,
        ("parmMax", "0:0:Mustrain;i"): 300.0,
        ("parmMin", "0:0:MD"): 0.7,
        ("parmMax", "0:0:MD"): 1.3,
    }
    for (control, variable), wanted in expected.items():
        got = gpx.get_Controls(control, variable=variable)
        assert got == pytest.approx(wanted, abs=1e-9), f"{control}({variable!r})"


def test_restraints_block_carries_use_flag(gsas2_available, tmp_path):
    """A phase restraint block must carry 'Use'; without it refine() dies.

    GSASIIstrIO.SummRestraints reads itemRest['Use'] for every restraint section,
    so a block built without that key makes the next refinement fail with
    KeyError: 'Use'.
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from stage_gate_rietveld import setup_phase_dist_restraints

    fixtures = Path(__file__).resolve().parent / "fixtures"
    project = _ceo2_project(tmp_path)
    gpx = G2sc.G2Project(str(project))
    phase = gpx.phases()[0]
    structure = Structure.from_file(str(fixtures / "cifs" / "CeO2.cif"))

    assert setup_phase_dist_restraints(gpx, phase, structure=structure, weight=100.0) > 0

    block = gpx.data["Restraints"]["data"][phase.name]["Bond"]
    assert {"Use", "wtFactor", "Bonds"} <= set(block)
    assert block["Use"] is True
    gpx.save()

    reloaded = G2sc.G2Project(str(project))
    reloaded.set_Controls("cycles", 0)
    reloaded.do_refinements([{}])  # previously raised KeyError: 'Use'


def test_json_payload_is_strict():
    """Non-finite floats must serialize as null so strict parsers accept the output.

    json.dumps emits bare NaN/Infinity tokens, which Python accepts but strict
    consumers (e.g. ConvertFrom-Json) reject.
    """
    from xrd_project import _json_safe

    payload = {
        "nan": float("nan"),
        "pos": float("inf"),
        "neg": float("-inf"),
        "ok": 1.5,
        "nested": [float("nan"), {"deep": float("inf")}],
    }
    safe = _json_safe(payload)
    assert safe == {
        "nan": None,
        "pos": None,
        "neg": None,
        "ok": 1.5,
        "nested": [None, {"deep": None}],
    }
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe
    with pytest.raises(ValueError):
        json.dumps(payload, allow_nan=False)


def test_ka2_ratio_round_trip(gsas2_available, tmp_path):
    """The K-alpha2/K-alpha1 ratio must survive dump -> apply -> dump.

    A data set can carry the companion line already removed, and the ratio is
    then 0.0; the document used to expose only Zero and SH/L from the instrument
    block, so that state could not be declared at all.
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from rietveld_params import apply_params_to_project, dump_project_params

    gpx = G2sc.G2Project(str(_ceo2_project(tmp_path)))
    doc = dump_project_params(gpx)
    assert doc["histogram"]["instrument"]["ka2_ratio"]["value"] == pytest.approx(0.5)

    doc["histogram"]["instrument"]["ka2_ratio"]["value"] = 0.0
    apply_params_to_project(gpx, doc)

    again = dump_project_params(gpx)
    assert again["histogram"]["instrument"]["ka2_ratio"]["value"] == pytest.approx(0.0)
    raw = gpx.histograms()[0].data["Instrument Parameters"][0]
    assert float(raw["I(L2)/I(L1)"][1]) == pytest.approx(0.0)


def test_declared_limits_are_compared_against_the_declaration():
    """The clamp helpers must read the caller's interval, not the schema's.

    The first implementation compared the refined value with the bounds of a
    freshly dumped document, which restates the schema defaults: a declared +-1 um
    displacement bound was therefore never enforced, and a value of -9.9 um was
    reported as within limits.
    """
    from rietveld_params import clamp_to_declared_limits, enforce_declared_limits

    declared = {"histogram": {"sample": {"DisplaceX": {"value": 0.0, "min": -1.0, "max": 1.0}}}}
    updated = {"histogram": {"sample": {"DisplaceX": {"value": -9.9, "min": -50.0, "max": 50.0}}}}

    assert enforce_declared_limits(updated, declared) == [
        "histogram.sample.DisplaceX = -9.9 is below the declared min -1.0"
    ]
    assert clamp_to_declared_limits(updated, declared) == ["histogram.sample.DisplaceX"]
    assert updated["histogram"]["sample"]["DisplaceX"]["value"] == pytest.approx(-1.0)


def test_declared_limits_survive_a_refinement(gsas2_available, tmp_path):
    """A declared interval must still be in force after a refinement.

    apply_box_constraints used to run after the document was applied and to
    overwrite the caller's interval with the schema default, so the declaration
    was lost on every refinement.
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from rietveld_params import dump_project_params, refine_project_with_params

    project = _ceo2_project(tmp_path)
    gpx = G2sc.G2Project(str(project))
    doc = dump_project_params(gpx)
    doc["histogram"]["sample"]["DisplaceX"].update({"refine": True, "min": -1.0, "max": 1.0})

    updated, diverged, reason = refine_project_with_params(
        gpx=gpx, params_dict=doc, cycles=3, project_path=project
    )

    assert diverged is False, reason
    displace = updated["histogram"]["sample"]["DisplaceX"]
    assert displace["max"] == pytest.approx(1.0)
    assert displace["min"] == pytest.approx(-1.0)
    assert abs(float(displace["value"])) <= 1.0 + 1e-9


def test_hardware_lower_bound_report_is_silent_on_canonical_optics(gsas2_available, tmp_path):
    """The instrument-sharpening report must not fire on the stock optics.

    The check compared the reflection list's FWHM column -- which holds the
    instrumental Gaussian variance ``U tan^2 + V tan + W`` in deg^2, 0.005733 for
    the canonical instrument -- against a canonical FWHM in degrees, so it fired
    on every project.
    """
    if not gsas2_available:
        pytest.skip("GSAS-II engine not available.")

    from GSASII import GSASIIscriptable as G2sc
    from stage_gate_rietveld import instrumental_lower_bound_report

    gpx = G2sc.G2Project(str(_ceo2_project(tmp_path)))
    hist = gpx.histograms()[0]
    assert instrumental_lower_bound_report(hist) == []

    # Optics sharpened below the canonical floor must be reported.
    hist.data["Instrument Parameters"][0]["W"][1] = 0.0005
    assert instrumental_lower_bound_report(hist) != []
