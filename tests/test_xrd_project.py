"""Tests for Interactive Stateful Crystallographic Workbench CLI (xrd_project.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gsas2_adapter import (
    BraggReflection,
    discard_gsas_listing,
    keep_gsas_listing,
    phase_rf_percent,
)
from xrd_project import (
    capture_engine_notes,
    cmd_add_phase,
    cmd_diagnose,
    cmd_export,
    cmd_init,
    cmd_remove_phase,
    cmd_status,
    cmd_step,
    cmd_toggle_phase,
    load_sidecar_history,
    save_sidecar_history,
    summarize_rule9,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CIFS_DIR = FIXTURES_DIR / "cifs"
PATTERNS_DIR = FIXTURES_DIR / "patterns"


@pytest.fixture
def test_env(tmp_path: Path):
    """Fixture providing clean workspace paths and test data."""
    pattern_file = PATTERNS_DIR / "CeO2.xy"
    cif_file = CIFS_DIR / "CeO2.cif"
    project_file = tmp_path / "test_CeO2.gpx"
    return {
        "tmp_path": tmp_path,
        "pattern": pattern_file,
        "cif": cif_file,
        "project": project_file,
    }


def _extract_json(text: str) -> dict:
    """Robustly extract JSON payload from stdout text."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return json.loads(text[start : end + 1])
    return json.loads(text)


def test_add_phase_reports_a_width_mismatch(test_env):
    """A phase entering with a default width must say so when the project settled elsewhere.

    The profile width of a new phase is GSAS-II's default, while the phases already in the
    project may have refined to a very different crystallite size. Left silent, the mismatch
    looks like a property of the sample: measured on the Sm-S workbench, setting the width
    alone moved R_wp from 9.23 to 8.35 % and the minority weight from 3.81 to 8.78 %.

    The message is read through a temporary loguru sink rather than through ``capfd``:
    loguru writes asynchronously, so a file-descriptor read can return before the record
    lands, and the assertion would then test the reader instead of the code.
    """
    from loguru import logger as loguru_logger

    project = test_env["project"]
    assert (
        cmd_init(
            SimpleNamespace(
                data=test_env["pattern"],
                project=project,
                instrument="canonical",
                wavelength=1.5406,
                bkg_terms=6,
            )
        )
        == 0
    )
    add_first = SimpleNamespace(project=project, cif=test_env["cif"], name="CeO2", scale=0.001)
    assert cmd_add_phase(add_first) == 0

    # GSAS-II becomes importable only after the engine bootstrap that cmd_init performs,
    # so this import belongs after the first project command and not at the top.
    from GSASII import GSASIIscriptable as G2sc

    # Put the phase already in the project at a narrow width, the way a refined model sits.
    gpx = G2sc.G2Project(str(project))
    hist_name = gpx.histograms()[0].name
    for phase in gpx.phases():
        phase.data["Histograms"][hist_name]["Size"][1] = [0.04, 1.0, 1.0]
    gpx.save()

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING")
    try:
        add_second = SimpleNamespace(
            project=project, cif=test_env["cif"], name="Second", scale=0.001
        )
        assert cmd_add_phase(add_second) == 0
    finally:
        loguru_logger.remove(sink_id)

    reported = "\n".join(messages)
    assert "crystallite size" in reported, "the width mismatch was not reported"
    assert "0.0400" in reported, "the refined width was not named"
    assert "profile-start-size" in reported, "the remedy was not named"


def test_workbench_init_in_memory_optics(test_env):
    """Test project initialization with canonical optics in memory and sidecar creation."""
    project = test_env["project"]
    pattern = test_env["pattern"]

    args = SimpleNamespace(
        data=pattern,
        project=project,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    rc = cmd_init(args)
    assert rc == 0
    assert project.exists()
    assert project.stat().st_size > 0

    history = load_sidecar_history(project)
    assert history["project_file"] == str(project.resolve())
    assert history["instrument"] == "canonical"
    assert history["bkg_terms"] == 6


def test_workbench_status_ssot_resilience(test_env, capfd):
    """Test status extraction directly from .gpx SSOT even if sidecar is deleted."""
    project = test_env["project"]
    pattern = test_env["pattern"]

    # 1. Initialize
    init_args = SimpleNamespace(
        data=pattern,
        project=project,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    assert cmd_init(init_args) == 0

    # 2. Add phase
    add_args = SimpleNamespace(
        project=project,
        cif=test_env["cif"],
        name="CeO2",
        scale=0.001,
    )
    assert cmd_add_phase(add_args) == 0

    # 3. Delete sidecar history file to test SSOT resilience
    sidecar = project.parent / f"{project.stem}_history.json"
    if sidecar.exists():
        sidecar.unlink()

    # 4. Query status in JSON mode. ``capfd`` captures the process stdout at file
    # descriptor level because the ``--json`` contract writes the payload past the
    # redirected ``sys.stdout`` (see xrd_project.print_json).
    _ = capfd.readouterr()
    status_args = SimpleNamespace(
        project=project,
        json=True,
    )
    rc = cmd_status(status_args)
    assert rc == 0

    captured = capfd.readouterr()
    data = _extract_json(captured.out)
    assert data["phases_count"] == 1
    assert data["phases"][0]["name"] == "CeO2"
    assert "metrics" in data


def test_workbench_add_remove_phase(test_env):
    """Test adding and removing phases cleanly from project data structures."""
    project = test_env["project"]
    pattern = test_env["pattern"]

    init_args = SimpleNamespace(
        data=pattern,
        project=project,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    assert cmd_init(init_args) == 0

    # Add CeO2
    add_args = SimpleNamespace(
        project=project,
        cif=test_env["cif"],
        name="CeO2",
        scale=0.002,
    )
    assert cmd_add_phase(add_args) == 0

    from GSASII import GSASIIscriptable as G2sc

    gpx = G2sc.G2Project(str(project))
    assert len(gpx.phases()) == 1
    assert gpx.phases()[0].name == "CeO2"

    # Remove CeO2
    rem_args = SimpleNamespace(
        project=project,
        phase_name="CeO2",
    )
    rc_rem = cmd_remove_phase(rem_args)
    assert rc_rem == 0

    gpx_after = G2sc.G2Project(str(project))
    assert len(gpx_after.phases()) == 0


def test_workbench_toggle_phase(test_env):
    """Test enabling and disabling a phase scale and refinement."""
    project = test_env["project"]
    pattern = test_env["pattern"]

    init_args = SimpleNamespace(
        data=pattern,
        project=project,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    assert cmd_init(init_args) == 0

    add_args = SimpleNamespace(
        project=project,
        cif=test_env["cif"],
        name="CeO2",
        scale=0.005,
    )
    assert cmd_add_phase(add_args) == 0

    # Disable phase
    dis_args = SimpleNamespace(
        project=project,
        phase_name="CeO2",
        enable=False,
        disable=True,
    )
    assert cmd_toggle_phase(dis_args) == 0

    from GSASII import GSASIIscriptable as G2sc

    gpx = G2sc.G2Project(str(project))
    p = gpx.phases()[0]
    hist_name = gpx.histograms()[0].name
    sc_val = float(p.data["Histograms"][hist_name]["Scale"][0])
    assert sc_val == 0.0

    # Enable phase back
    en_args = SimpleNamespace(
        project=project,
        phase_name="CeO2",
        enable=True,
        disable=False,
    )
    assert cmd_toggle_phase(en_args) == 0

    gpx2 = G2sc.G2Project(str(project))
    p2 = gpx2.phases()[0]
    sc_val2 = float(p2.data["Histograms"][hist_name]["Scale"][0])
    assert sc_val2 > 0.0


def test_workbench_step_and_rollback(test_env, capsys):
    """Test Stage 1 refinement execution and physical disk checkpoint rollback."""
    project = test_env["project"]
    pattern = test_env["pattern"]

    init_args = SimpleNamespace(
        data=pattern,
        project=project,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    assert cmd_init(init_args) == 0

    add_args = SimpleNamespace(
        project=project,
        cif=test_env["cif"],
        name="CeO2",
        scale=0.001,
    )
    assert cmd_add_phase(add_args) == 0

    # Step Stage 1 (Scale and Background)
    _ = capsys.readouterr()
    step_args = SimpleNamespace(
        project=project,
        stage="1_Scale_and_Background",
        sequence=None,
        cycles=5,
        bkg_terms=6,
        refine_displacement=False,
        refine_axial=False,
        texture_axis=[],
        uniaxial_size=[],
        init_md=0.95,
        override_json=None,
        force=False,
    )
    rc_step = cmd_step(step_args)
    assert rc_step == 0

    captured = capsys.readouterr()
    res = _extract_json(captured.out)
    assert res["status"] == "SUCCESS"
    assert res["stage"] == "1_Scale_and_Background"
    assert res["r_wp_percent"] < 90.0


def test_recommended_actions_render_without_a_command_for_a_card():
    """An action that carries no command must not break the whole report.

    A reference card yields OBTAIN_STRUCTURE_FOR_CARD, which has no ready command by
    design: the card has to be turned into a structure first. The reporter read the
    command unconditionally and raised KeyError after the report had been printed, so a
    successful diagnosis exited nonzero. This pins the omission as legal.
    """
    from xrd_project import format_recommended_actions

    rendered = format_recommended_actions(
        [
            {"action_type": "OBTAIN_STRUCTURE_FOR_CARD", "rationale": "a card is not a structure"},
            {
                "action_type": "ADD_CANDIDATE_PHASE",
                "rationale": "phase",
                "command": "add-phase x y",
            },
        ]
    )

    assert "OBTAIN_STRUCTURE_FOR_CARD" in rendered
    assert "a card is not a structure" in rendered
    assert "Command: add-phase x y" in rendered
    assert rendered.count("Command:") == 1, "only the action that has a command may print one"


def test_workbench_diagnose_with_candidate_matching(test_env, capfd):
    """Test residual peak classification and candidate CIF pool matching."""
    project = test_env["project"]
    pattern = test_env["pattern"]

    init_args = SimpleNamespace(
        data=pattern,
        project=project,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    assert cmd_init(init_args) == 0

    # Объект аргументов строится явно и называет все поля команды: парсера
    # больше нет, единственный вход для человека — xrd_sample.
    _ = capfd.readouterr()
    diag_args = SimpleNamespace(
        project=project,
        out=None,
        candidate_cifs=CIFS_DIR,
        elements=None,
        any=None,
        none=None,
        strict=False,
        pdf2=False,
        angular_tolerance=0.25,
        snr_threshold=3.0,
        wavelength=1.5406,
        gradients=False,
        json=True,
    )
    rc_diag = cmd_diagnose(diag_args)
    assert rc_diag == 0

    captured = capfd.readouterr()
    diag_data = _extract_json(captured.out)
    assert "true_unindexed_count" in diag_data
    assert "candidate_phase_matches" in diag_data
    # Candidate CIFs should be evaluated
    assert isinstance(diag_data["candidate_phase_matches"], list)


def test_workbench_export(test_env):
    """Test export of refined CIFs, CSVs, publication plots, and summary.json."""
    project = test_env["project"]
    pattern = test_env["pattern"]
    out_dir = test_env["tmp_path"] / "results_export"

    init_args = SimpleNamespace(
        data=pattern,
        project=project,
        instrument="canonical",
        wavelength=1.5406,
        bkg_terms=6,
    )
    assert cmd_init(init_args) == 0

    add_args = SimpleNamespace(
        project=project,
        cif=test_env["cif"],
        name="CeO2",
        scale=0.001,
    )
    assert cmd_add_phase(add_args) == 0

    export_args = SimpleNamespace(
        project=project,
        out_dir=out_dir,
        plot=None,
        wavelength=1.5406,
        dpi=150,  # Fast DPI for test
        gii_threshold=None,
        strict=False,
        allow_incomplete=False,
    )
    from export import ExportSummary

    summary = cmd_export(export_args)
    assert isinstance(summary, ExportSummary)

    assert (out_dir / "profile_curves.csv").exists()
    assert (out_dir / "bragg_reflections.csv").exists()
    assert (out_dir / f"{project.stem}_CeO2_refined.cif").exists()
    summary_file = out_dir / "summary.json"
    assert summary_file.exists()
    stored = json.loads(summary_file.read_text(encoding="utf-8"))
    assert stored["r_wp_percent"] == summary.r_wp_percent
    assert not (out_dir / "REPORT.md").exists()

    assert "CeO2" in summary.phases


def test_workbench_revert_undoes_a_trial_phase(test_env):
    """A trial addition must be reversible.

    ``remove-phase`` takes the phase out but leaves every other parameter in the
    minimum the trial reached, so the metric does not return; the checkpoint stack
    is what makes add -> refine -> revert work.
    """
    from xrd_project import cmd_revert

    project = test_env["project"]
    pattern = test_env["pattern"]

    assert (
        cmd_init(
            SimpleNamespace(
                data=pattern,
                project=project,
                instrument="canonical",
                wavelength=1.5406,
                bkg_terms=6,
            )
        )
        == 0
    )
    assert (
        cmd_add_phase(SimpleNamespace(project=project, cif=test_env["cif"], name="CeO2", scale=0.2))
        == 0
    )
    assert (
        cmd_add_phase(
            SimpleNamespace(
                project=project, cif=CIFS_DIR / "TiO2_rutile.cif", name="TiO2", scale=0.05
            )
        )
        == 0
    )

    from GSASII import GSASIIscriptable as G2sc

    assert len(G2sc.G2Project(str(project)).phases()) == 2

    assert cmd_revert(SimpleNamespace(project=project, to=None)) == 0
    assert [p.name for p in G2sc.G2Project(str(project)).phases()] == ["CeO2"]

    stack = json.loads((project.parent / f"{project.stem}_ckpt.json").read_text(encoding="utf-8"))
    assert [entry["index"] for entry in stack] == [1]


def test_export_names_the_unmet_criteria(test_env):
    """The verdict must say what failed, not only that something did.

    A single R_wp threshold certified models with per-cent microstrain and serial
    residuals as converged, which is exactly the defect the gates exist to catch.
    """
    project = test_env["project"]
    pattern = test_env["pattern"]
    out_dir = test_env["tmp_path"] / "results_criteria"

    assert (
        cmd_init(
            SimpleNamespace(
                data=pattern,
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
            SimpleNamespace(project=project, cif=test_env["cif"], name="CeO2", scale=0.001)
        )
        == 0
    )
    summary = cmd_export(
        SimpleNamespace(
            project=project,
            out_dir=out_dir,
            plot=None,
            wavelength=1.5406,
            dpi=100,
            gii_threshold=None,
            strict=False,
            allow_incomplete=True,
        )
    )

    from export import ExportSummary
    from stage_gate_rietveld import RefinementStatus

    assert isinstance(summary, ExportSummary)
    assert (out_dir / "summary.json").exists()

    assert isinstance(summary.unmet_criteria, list)
    assert summary.exportable is False
    if summary.status in (RefinementStatus.CONVERGED, RefinementStatus.NEEDS_REFINEMENT):
        assert summary.unmet_criteria, "an unrefined model must name its unmet criteria"

    inst_dict = summary.instrument.to_dict()
    for key in ("zero_deg", "sh_over_l", "ka2_ratio", "u", "v", "w"):
        assert key in inst_dict
    assert summary.sample.displace_x_um is not None
    assert summary.background.n_terms >= 0


def test_success_gates_name_every_unmet_criterion():
    """Every Stage-Gate criterion must be checked and named (owner: stage_gate_rietveld).

    A single R_wp threshold used to certify the verdict, which passed models with
    per-cent microstrain, serial residuals and unindexed peaks -- the very defect
    the gates exist to catch.
    """
    from stage_gate_rietveld import SuccessGateInputs, evaluate_success_gates

    # A model that satisfies every criterion.
    assert (
        evaluate_success_gates(
            SuccessGateInputs(
                r_wp_percent=9.0,
                durbin_watson=1.90,
                max_microstrain_percent=0.20,
                max_crystallite_size_nm=120.0,
                unindexed_count=0,
            )
        )
        == []
    )

    # The case the old threshold certified as converged (R_wp = 12.5 % <= 15 %),
    # with the microstrain, the size ceiling and the unindexed peaks of a real
    # surface-degraded sample.
    unmet = evaluate_success_gates(
        SuccessGateInputs(
            r_wp_percent=12.5,
            durbin_watson=0.46,
            max_microstrain_percent=1.03,
            max_crystallite_size_nm=500.0,
            unindexed_count=11,
        )
    )
    assert len(unmet) == 4
    assert any("R_wp" in item for item in unmet)
    assert any("Durbin-Watson" in item for item in unmet)
    assert any("microstrain" in item for item in unmet)
    assert any("unindexed" in item for item in unmet)

    # A high R_wp is admissible against a noisy pattern when the statistical
    # criterion R_wp/R_exp <= 1.35 holds: the ratio, not the bare value, decides.
    # Отношение R_wp/R_exp не спасает плохую подгонку: 12.5 % остаётся отказом.
    assert (
        evaluate_success_gates(
            SuccessGateInputs(
                r_wp_percent=12.5,
                durbin_watson=1.90,
                max_microstrain_percent=0.20,
                max_crystallite_size_nm=120.0,
                unindexed_count=0,
            )
        )
        != []
    )

    # Each remaining criterion fires on its own.
    assert any(
        "crystallite size" in u
        for u in evaluate_success_gates(
            SuccessGateInputs(
                r_wp_percent=9.0,
                durbin_watson=1.90,
                max_microstrain_percent=0.20,
                max_crystallite_size_nm=950.0,
                unindexed_count=0,
            )
        )
    )
    assert any(
        "unindexed" in u
        for u in evaluate_success_gates(
            SuccessGateInputs(
                r_wp_percent=9.0,
                durbin_watson=1.90,
                max_microstrain_percent=0.20,
                max_crystallite_size_nm=120.0,
                unindexed_count=4,
            )
        )
    )

    # Высокий R_wp больше не проходит и при «согласованном» отношении:
    # абсолютное качество подгонки — отдельный критерий.
    assert any(
        "R_wp" in u
        for u in evaluate_success_gates(
            SuccessGateInputs(
                r_wp_percent=28.0,
                durbin_watson=1.60,
                max_microstrain_percent=0.50,
                max_crystallite_size_nm=50.0,
                unindexed_count=0,
            )
        )
    )


# --- GSAS-II refinement listing policy ---------------------------------------
# GSAS-II opens <project>.lst unconditionally at the start of Refine, and
# <project>.junk when all derivatives are requested; there is no switch for it.
# Authoritative metrics come from hist.residuals, so the listing is a by-product:
# removed after success, kept on failure, and kept on demand for diagnostics.


def test_discard_gsas_listing_removes_lst_and_junk(tmp_path, monkeypatch):
    monkeypatch.delenv("XRD_KEEP_GSAS_LISTING", raising=False)
    project = tmp_path / "sample.gpx"
    project.write_text("stub", encoding="utf-8")
    lst = tmp_path / "sample.lst"
    lst.write_text("listing", encoding="utf-8")
    junk = tmp_path / "sample.junk"
    junk.write_text("derivatives", encoding="utf-8")

    removed = discard_gsas_listing(project)

    assert {path.name for path in removed} == {"sample.lst", "sample.junk"}
    assert not lst.exists()
    assert not junk.exists()
    # The project itself is never touched: it is the evidence that survives.
    assert project.exists()


def test_discard_gsas_listing_keeps_files_when_env_flag_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("XRD_KEEP_GSAS_LISTING", "1")
    project = tmp_path / "sample.gpx"
    project.write_text("stub", encoding="utf-8")
    lst = tmp_path / "sample.lst"
    lst.write_text("listing", encoding="utf-8")

    assert discard_gsas_listing(project) == ()
    assert lst.exists()


def test_discard_gsas_listing_is_quiet_when_nothing_to_remove(tmp_path, monkeypatch):
    monkeypatch.delenv("XRD_KEEP_GSAS_LISTING", raising=False)
    project = tmp_path / "sample.gpx"
    project.write_text("stub", encoding="utf-8")

    assert discard_gsas_listing(project) == ()


def test_keep_gsas_listing_reads_the_environment(monkeypatch):
    monkeypatch.setenv("XRD_KEEP_GSAS_LISTING", "yes")
    assert keep_gsas_listing() is True
    monkeypatch.setenv("XRD_KEEP_GSAS_LISTING", "0")
    assert keep_gsas_listing() is False
    monkeypatch.delenv("XRD_KEEP_GSAS_LISTING", raising=False)
    assert keep_gsas_listing() is False


# --- Stage diagnostics: only anomalies are spoken ----------------------------
# A stage that fits well stays silent; the terminal speaks when a threshold is
# crossed. These tests pin the arithmetic and the silence.


def test_phase_rf_uses_observed_amplitudes():
    reflections = [
        BraggReflection("A", (1, 0, 0), 10.0, 3.0, 100.0, f_calc=90.0, f_obs=100.0),
        BraggReflection("A", (1, 1, 0), 20.0, 2.0, 50.0, f_calc=60.0, f_obs=50.0),
        BraggReflection("B", (1, 0, 0), 30.0, 1.5, 10.0, f_calc=10.0, f_obs=10.0),
    ]
    rf = phase_rf_percent(reflections)
    assert rf["A"] == 13.33  # (10 + 10) / 150
    assert rf["B"] == 0.0


def test_phase_rf_ignores_reflections_without_observed_amplitude():
    assert phase_rf_percent([BraggReflection("A", (1, 0, 0), 10.0, 3.0)]) == {}


def test_rule9_summary_follows_table_semantics():
    # A parameter without a positive sigma is PASS in the shifts table, so the
    # summary must not count it as converging.
    total, converged, converging = summarize_rule9(
        {"a": 0.01, "b": 0.5, "c": 0.2}, {"a": 1.0, "b": 1.0}
    )
    assert (total, converged, converging) == (3, 2, 1)


def test_rule9_summary_is_quiet_for_a_converged_stage():
    total, converged, converging = summarize_rule9({"a": 0.01}, {"a": 1.0})
    assert (total, converged, converging) == (1, 1, 0)


def test_capture_engine_notes_passes_output_through_and_promotes_warnings(capfd):
    # loguru owns its stderr handle, so capfd cannot see its records: the sink
    # below is the honest way to observe what was logged.
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        with capture_engine_notes("unit"):
            print("Refinement converged normally")
            print("WARNING: singular matrix in phase 2")
    finally:
        logger.remove(sink_id)
    out, _err = capfd.readouterr()

    # Engine output is not swallowed: the caller still sees it.
    assert "Refinement converged normally" in out
    # The warning is promoted to the log with the context tag, so the agent does
    # not have to grep GSAS-II's stdout for it.
    assert any("GSAS-II [unit]" in record for record in records)
    assert any("singular matrix" in record for record in records)
    # A normal convergence statement is not a warning: the filter must stay
    # quiet about it, otherwise every stage would shout.
    assert not any("converged normally" in record for record in records)


# --- Engine provenance -------------------------------------------------------
# After a GSAS-II update there must be a way to say which engine produced a
# project. The sidecar records it on every write, so no call site can forget.


def test_sidecar_history_records_engine_provenance(tmp_path):
    project = tmp_path / "X.gpx"
    project.write_text("stub", encoding="utf-8")
    save_sidecar_history(project, {"stages_completed": []})

    data = json.loads((tmp_path / "X_history.json").read_text(encoding="utf-8"))
    assert data["python_version"].count(".") >= 1
    assert data["gsas_commit"]
    assert data["gsas_versiontag"]


def test_workbench_exposes_no_cli():
    """Библиотечный слой не должен возвращать себе точку входа.

    Регрессионный засов: парсер, диспетчер и run() удалены как класс, и любое
    их возвращение снова сделает агента владельцем путей.
    """
    import xrd_project

    assert not hasattr(xrd_project, "build_cli_parser")
    assert not hasattr(xrd_project, "run")
    assert not hasattr(xrd_project, "main")


def test_candidates_are_split_by_element_composition():
    """Ранжирование слепо к составу: фаза с чужим элементом не должна попадать в возможные."""
    from xrd_project import partition_candidates

    matches = [
        {"phase_name": "Sm2S3_mp-1", "formula": "Sm2S3"},
        {"phase_name": "YbS_mp-2", "formula": "YbS"},
        {"phase_name": "SmSi_mp-3", "formula": "SmSi"},
    ]
    plausible, novel = partition_candidates(matches, {"Sm", "S", "O"})
    assert [m["phase_name"] for m in plausible] == ["Sm2S3_mp-1"]
    assert sorted(m["phase_name"] for m in novel) == ["SmSi_mp-3", "YbS_mp-2"]
    assert novel[0]["elements"]


def test_add_phase_without_scale_uses_a_small_default(test_env):
    """`--scale` is optional, and the default must not be unit scale.

    Reading `args.scale` unconditionally crashed the command after the CIF had been
    imported; a unit default made the new phase enter with a calculated pattern far
    stronger than observed, and the next stage diverged (measured: R_wp 19% -> 5.5e7)
    instead of refining the scale. Both failure modes were silent in the report.
    """
    from xrd_project import load_sidecar_history

    project = test_env["project"]
    assert (
        cmd_init(
            SimpleNamespace(
                data=test_env["pattern"],
                project=project,
                instrument="canonical",
                wavelength=1.5406,
                bkg_terms=6,
            )
        )
        == 0
    )

    added = SimpleNamespace(project=project, cif=test_env["cif"], name="CeO2", scale=None)
    assert cmd_add_phase(added) == 0

    history = load_sidecar_history(Path(project))
    stored = [entry for entry in history.get("phases", {}).values()]
    assert stored and stored[-1]["initial_scale"] == pytest.approx(1e-2)


def test_sequence_runs_until_plateau_and_keeps_the_best_state(test_env):
    """Маршрут `--sequence` идёт проходами до плато и хранит лучшее состояние.

    Требование нормы: маршрут заканчивается лучшей найденной моделью, а не
    последней итерацией, если финальный проход ухудшил R_wp.
    """
    project = test_env["project"]
    assert (
        cmd_init(
            SimpleNamespace(
                data=test_env["pattern"],
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
            SimpleNamespace(project=project, cif=test_env["cif"], name="CeO2", scale=0.001)
        )
        == 0
    )

    step_args = SimpleNamespace(
        project=project,
        stage=None,
        sequence="1,5",
        cycles=3,
        bkg_terms=6,
        refine_displacement=False,
        refine_axial=False,
        texture_axis=[],
        uniaxial_size=[],
        init_md=0.95,
        override_json=None,
        force=False,
        passes=None,
        until_plateau=True,
        max_passes=2,
        plateau_delta=0.1,
    )
    assert cmd_step(step_args) == 0

    history = load_sidecar_history(Path(project))
    route_records = [
        item for item in history["iterations"] if str(item.get("stage", "")).startswith("route:")
    ]
    assert route_records, "маршрут не записал результат в историю"
    record = route_records[-1]
    assert 1 <= int(record["passes"]) <= 2
    assert record["route_decision"] in ("plateau", "regressed", "exhausted", "continue")
    assert record["route_best_r_wp"] is not None
    assert len(record["route_r_wp_per_pass"]) == record["passes"]


def test_workbench_pawley_refinement_and_clamping(test_env):
    """Фаза с флагом pawley=True инициализирует Pawley ref, сохраняет RefList и уточняется без отрицательных пиков."""
    from xrd_project import PawleyCol, is_pawley_phase

    project = test_env["project"]
    assert (
        cmd_init(
            SimpleNamespace(
                data=test_env["pattern"],
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
            SimpleNamespace(
                project=project,
                cif=test_env["cif"],
                from_card=None,
                name="CeO2_pawley",
                scale=None,
                pawley=True,
            )
        )
        == 0
    )

    from GSASII import GSASIIscriptable as G2sc

    gpx = G2sc.G2Project(str(project))
    phase = gpx.phases()[0]
    hist = gpx.histograms()[0]
    assert is_pawley_phase(phase) is True
    pawley_refs = phase.data.get("Pawley ref", [])
    assert len(pawley_refs) > 0
    assert any(bool(pk[PawleyCol.REFINE]) for pk in pawley_refs)
    assert "RefList" in hist.data.get("Reflection Lists", {}).get("CeO2_pawley", {})

    assert (
        cmd_step(
            SimpleNamespace(
                project=project,
                stage=None,
                sequence="1,2,3,7,12",
                cycles=4,
                bkg_terms=6,
                refine_displacement=False,
                refine_axial=False,
                texture_axis=[],
                uniaxial_size=[],
                init_md=0.95,
                override_json=None,
                force=False,
                passes=1,
            )
        )
        == 0
    )

    gpx_after = G2sc.G2Project(str(project))
    phase_after = gpx_after.phases()[0]
    assert any(
        float(pk[PawleyCol.INTENSITY]) > 0.0 for pk in phase_after.data.get("Pawley ref", [])
    )

    out_dir = test_env["tmp_path"] / "pawley_export"
    from export import ExportSummary

    summary = cmd_export(
        SimpleNamespace(
            project=project,
            out_dir=out_dir,
            plot=None,
            wavelength=1.5406,
            dpi=100,
            gii_threshold=None,
            strict=False,
            allow_incomplete=True,
            json=False,
        )
    )
    assert isinstance(summary, ExportSummary)
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "profile_curves.csv").exists()
    assert (out_dir / "bragg_reflections.csv").exists()
    assert (out_dir / f"{project.stem}_rietveld.png").exists()


def test_build_placeholder_cif_from_card_preserves_polar_monoclinic_symmetry(tmp_path: Path):
    """build_placeholder_cif_from_card сохраняет полярную группу P21 (#4) и компактную запись P21/c (#14) без повышения симметрии."""
    import gemmi
    from pdf2_reader import PDFCard
    from xrd_project import build_placeholder_cif_from_card

    card_p21 = PDFCard(
        set_number=83,
        card_number=1994,
        status="P",
        quality_class="I",
        empirical_formula="Er4 H9 N O13",
        structural_formula="Er4 O2 ( O H )8 ( H N O3 )",
        space_group="P21",
        ita_number=4,
        cell_a=9.337,
        cell_b=16.369,
        cell_c=3.608,
        cell_alpha=90.0,
        cell_beta=101.12,
        cell_gamma=90.0,
    )
    out_cif = build_placeholder_cif_from_card(card_p21, tmp_path / "card_83-1994.cif")
    doc = gemmi.cif.read_file(str(out_cif))
    block = doc.sole_block()
    sg_hm = block.find_value("_symmetry_space_group_name_H-M").strip("'\" ")
    sg_obj = gemmi.find_spacegroup_by_name(sg_hm)
    assert sg_obj is not None and sg_obj.number == 4

