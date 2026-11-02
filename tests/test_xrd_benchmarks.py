"""Comprehensive 4-tier benchmark suite for XRD crystallographic analysis."""

import json
import socket
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gemmi
import pytest
import spglib
from charge_flipping import MultiStartChargeFlipping, calc_pbc_distance
from gatekeeper import CrystallochemicalGatekeeper
from instrument import CanonicalInstrument
from peaks import extract_peaks_robust
from search_match import CandidateScoringEngine, ProfileProbabilityField
from stage_gate_rietveld import StageGateRietveld
from xrd_pipeline import PipelineReport, run_pipeline

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CIF_DIR = FIXTURES_DIR / "cifs"
PATTERNS_DIR = FIXTURES_DIR / "patterns"


@pytest.fixture(autouse=True)
def _ensure_numpy_error_state():
    """Ensure numpy float underflow does not raise errors across tests."""
    np.seterr(under="ignore")
    yield
    np.seterr(under="ignore")


# =========================================================================
# TIER 1: Base Calibration & Stage-Gate Protocol (1 phase, high symmetry)
# =========================================================================
def test_tier1_canonical_instrument_lower_bound():
    """Verify that canonical instrument profile acts as an immutable lower bound."""
    instrument = CanonicalInstrument()
    two_theta = np.array([20.0, 40.0, 60.0, 80.0])
    inst_fwhm = instrument.fwhm_inst_deg(two_theta)

    # Basic physical sanity check for Cu Ka Bragg-Brentano (0.035 - 0.12 deg)
    assert np.all(inst_fwhm > 0.03)
    assert np.all(inst_fwhm < 0.20)

    # Enforce lower bound on artificially sharp peaks
    sharp_fwhm = np.array([0.01, 0.02, 0.015, 0.02])
    bounded = instrument.enforce_lower_bound(two_theta, sharp_fwhm)
    assert np.all(bounded >= inst_fwhm)

    # Sample broadening deconvolution
    broadened_fwhm = inst_fwhm + 0.10
    sample_fwhm = instrument.extract_sample_broadening(two_theta, broadened_fwhm)
    assert np.all(sample_fwhm > 0.05)


def test_tier1_stage_gate_rietveld_protocol_and_guards():
    """Verify 8-stage sequence and automated guardrails."""
    controller = StageGateRietveld()

    # Protocol structure
    assert len(controller.STAGES) == 12
    assert controller.STAGES[0] == "1_Scale_and_Background"
    assert controller.STAGES[1] == "2_Zero_Shift"
    assert controller.STAGES[3] == "4_Atomic_Coordinates"
    assert controller.STAGES[6] == "7_Sample_Profile_XY"
    assert controller.STAGES[-3:] == [
        "10_Instrument_Profile",
        "11_Absorption_Extinction",
        "12_Final_Scale",
    ]

    # Correlation matrix safeguard (|rho| > 0.85)
    params = ["Scale", "B_iso", "Zero", "Cell_a"]
    corr_matrix = np.array(
        [
            [1.00, 0.92, 0.10, 0.05],  # Scale <-> B_iso is 0.92
            [0.92, 1.00, 0.15, 0.08],
            [0.10, 0.15, 1.00, 0.30],
            [0.05, 0.08, 0.30, 1.00],
        ]
    )
    flagged = controller.validate_correlations(corr_matrix, params)
    assert len(flagged) == 1
    assert "Scale <-> B_iso" in flagged[0]

    # Condition number check kappa(H) > 10^5
    singular_matrix = np.array([[1.0, 1.0], [1.0, 1.000001]])
    cond, is_sing = controller.check_condition_number(singular_matrix)
    assert cond > 1e5
    assert is_sing is True

    # Box constraints check. A zero shift of 0.08 deg is a real goniometer offset
    # inside the hard +-0.5 deg bound, so it is advice rather than a violation:
    # a reference export of a real sample carries Zero = 0.262 deg.
    valid, errs = controller.enforce_box_constraints(
        zero_shift=0.08,
        b_iso_values={"Ti": 0.10, "O": 0.80},  # Ti < 0.2 limit
        march_dollase_r=1.5,  # > 1.3 limit
    )
    assert valid is False
    assert len(errs) == 2

    # The real offset is reported as advice that names the owning parameter, so a
    # zero error is not silently absorbed by a fictitious specimen displacement.
    notes = controller.angular_origin_notes(zero_shift=0.262, displace_x_um=-676.0)
    assert any("belongs" in note and "Zero" in note for note in notes)
    assert any("DisplaceX" in note for note in notes)

    # Beyond the hard bound it is a violation again.
    valid_hard, errs_hard = controller.enforce_box_constraints(zero_shift=0.9)
    assert valid_hard is False
    assert len(errs_hard) == 1

    # A well-aligned goniometer with a small displacement produces no advice.
    assert controller.angular_origin_notes(zero_shift=0.01, displace_x_um=5.0) == []


def test_tier1_real_rietveld_profile_refinement_ceo2(tmp_path):
    """Verify Stage-Gate Rietveld profile refinement runs natively via xrd_project."""
    import argparse

    from stage_gate_rietveld import ensure_gsas2_environment
    from xrd_project import cmd_add_phase, cmd_export, cmd_init, cmd_step

    if not ensure_gsas2_environment():
        pytest.skip("GSAS-II engine not available")

    gpx_path = tmp_path / "CeO2.gpx"
    # 1. Initialize project
    rc = cmd_init(
        argparse.Namespace(
            project=str(gpx_path),
            data=str(PATTERNS_DIR / "CeO2.xy"),
            instrument="canonical",
            wavelength=1.5406,
            bkg_terms=6,
        )
    )
    assert rc == 0
    assert gpx_path.exists()

    # 2. Add CeO2 phase
    rc = cmd_add_phase(
        argparse.Namespace(
            project=str(gpx_path),
            cif=str(CIF_DIR / "CeO2.cif"),
            name="CeO2",
            scale=0.001,
        )
    )
    assert rc == 0

    # 3. Refine Stage 1: Scale and Background
    rc = cmd_step(
        argparse.Namespace(
            project=str(gpx_path),
            stage="1_Scale_and_Background",
            sequence=None,
            cycles=3,
            bkg_terms=6,
            refine_displacement=False,
            refine_axial=False,
            texture_axis=[],
            uniaxial_size=[],
            init_md=0.95,
            override_json=None,
            force=False,
        )
    )
    assert rc == 0

    # 4. Export artifacts
    out_dir = tmp_path / "export"
    summary = cmd_export(
        argparse.Namespace(
            project=str(gpx_path),
            out_dir=str(out_dir),
            plot=None,
            wavelength=1.5406,
            dpi=100,
            gii_threshold=None,
            strict=False,
            allow_incomplete=False,
        )
    )
    from export import ExportSummary

    assert isinstance(summary, ExportSummary)
    summary_file = out_dir / "summary.json"
    assert summary_file.exists()
    assert (
        json.loads(summary_file.read_text(encoding="utf-8"))["r_wp_percent"] == summary.r_wp_percent
    )
    assert not (out_dir / "REPORT.md").exists()
    assert summary.r_wp_percent < 80.0
    assert "CeO2" in summary.microstructure


def test_tier1_fail_fast_on_gsas2_exception(tmp_path):
    """Verify that any exception inside GSAS-II fails fast with error code 1."""
    import argparse

    from xrd_project import cmd_step

    # Pointing to non-existent project fails fast
    rc = cmd_step(
        argparse.Namespace(
            project=str(tmp_path / "nonexistent.gpx"),
            stage="1_Scale_and_Background",
            sequence=None,
            cycles=3,
            bkg_terms=6,
            refine_displacement=False,
            refine_axial=False,
            texture_axis=[],
            uniaxial_size=[],
            init_md=0.95,
            override_json=None,
            force=False,
        )
    )
    assert rc == 1


def test_tier1_gsas2_stage_delegation():
    """Verify that StageGateRietveld defines canonical recipes exclusively for GSAS-II."""
    from stage_gate_rietveld import StageGateRietveld, build_stage_recipe

    assert len(StageGateRietveld.STAGES) == 12
    for stage in StageGateRietveld.STAGES:
        recipe = build_stage_recipe(stage)
        assert isinstance(recipe, dict)
        assert len(recipe) > 0


# =========================================================================
# TIER 2: Candidate Scoring Engine (Multiphase analytical scoring)
# =========================================================================
def test_tier2_candidate_scoring_pure_phase():
    """Verify single-phase identification and candidate scoring on CeO2."""
    data = np.loadtxt(PATTERNS_DIR / "CeO2.xy")
    two_theta, intensity = data[:, 0], data[:, 1]
    prob_field = ProfileProbabilityField.from_raw_pattern(two_theta, intensity)

    engine = CandidateScoringEngine(angular_tolerance=0.20)
    cifs = {
        "CeO2": CIF_DIR / "CeO2.cif",
        "TiO2_rutile": CIF_DIR / "TiO2_rutile.cif",
    }
    results = engine.score_candidates(
        two_theta=two_theta,
        y_net=prob_field.y_net,
        sigma=prob_field.sigma,
        candidate_structures=cifs,
    )
    assert len(results) >= 1
    assert results[0].phase_name == "CeO2"
    assert results[0].score > 80.0
    assert results[0].completeness_ml_tl > 0.80
    assert "PERFECT_MATCH" in results[0].diagnostic_flags


def test_tier2_candidate_scoring_multiphase_mixture():
    """Verify identification of dominant Rutile (95%) and minor Anatase (5%) in two-phase mixture."""
    data = np.loadtxt(PATTERNS_DIR / "Mixture_Rutile95_Anatase5.xy")
    two_theta, intensity = data[:, 0], data[:, 1]
    prob_field = ProfileProbabilityField.from_raw_pattern(two_theta, intensity)

    engine = CandidateScoringEngine(angular_tolerance=0.20)
    cifs = {
        "TiO2_rutile": CIF_DIR / "TiO2_rutile.cif",
        "TiO2_anatase": CIF_DIR / "TiO2_anatase.cif",
        "CeO2": CIF_DIR / "CeO2.cif",
    }
    results = engine.score_candidates(
        two_theta=two_theta,
        y_net=prob_field.y_net,
        sigma=prob_field.sigma,
        candidate_structures=cifs,
    )

    assert len(results) >= 2
    # Dominant phase is Rutile with top score and high scale
    assert results[0].phase_name == "TiO2_rutile"
    assert results[0].score > 70.0

    # Anatase is detected with positive score (minor 5% phase in full pattern)
    anatase_res = next((r for r in results if r.phase_name == "TiO2_anatase"), None)
    assert anatase_res is not None
    assert anatase_res.score > 1.0
    # Scale of rutile must be significantly larger than anatase
    assert results[0].scale_factor > anatase_res.scale_factor

    # Anatase ranks above completely irrelevant candidate CeO2
    ceo2_res = next((r for r in results if r.phase_name == "CeO2"), None)
    ceo2_score = ceo2_res.score if ceo2_res else 0.0
    assert anatase_res.score > ceo2_score


# =========================================================================
# TIER 3: Fourier Truncation Ripple Pruning with PBC
# =========================================================================
def test_tier3_multistart_charge_flipping_supervision():
    """Verify Fourier series truncation ripple pruning under Periodic Boundary Conditions."""
    cf_supervisor = MultiStartChargeFlipping(
        ensemble_size=20, r_dense_threshold=0.30, min_distance_angstrom=1.20
    )
    cell_lengths = (4.5937, 4.5937, 2.9587)
    cell_angles = (90.0, 90.0, 90.0)

    clean_sites = [
        ("Ti", np.array([0.0, 0.0, 0.0])),
        ("O", np.array([0.305, 0.305, 0.0])),
    ]
    kept_clean, pruned_clean = cf_supervisor.prune_close_contacts(
        cell_lengths, cell_angles, clean_sites
    )
    assert pruned_clean == 0
    assert len(kept_clean) == 2

    sites_with_ripple = [
        clean_sites[0],
        ("Ti", np.array([0.01, 0.01, 0.01])),
        clean_sites[1],
    ]
    pruned_sites, pruned_count = cf_supervisor.prune_close_contacts(
        cell_lengths, cell_angles, sites_with_ripple
    )
    assert pruned_count == 1
    assert len(pruned_sites) == 2


# =========================================================================
# TIER 4: Stress-test Gatekeepers (Invalid & degenerate data)
# =========================================================================
def test_tier4_gatekeeper_valid_rutile():
    """Verify that physically correct rutile structure passes gatekeeper."""
    gatekeeper = CrystallochemicalGatekeeper(gii_reject_threshold=0.25)
    struct = Structure.from_file(str(CIF_DIR / "TiO2_rutile.cif"))
    b_iso = {"Ti": 0.5, "O": 0.8}

    report = gatekeeper.audit_structure(struct, b_iso_dict=b_iso)
    assert report.is_valid is True
    assert report.min_interatomic_distance > 1.80  # Ti-O is ~1.95 A
    assert len(report.errors) == 0


def test_tier4_gatekeeper_reject_inverted_chemistry():
    """Verify that swapped cation/anion (Fe on O position) is rejected by GII / BVS."""
    gatekeeper = CrystallochemicalGatekeeper(gii_reject_threshold=0.25)
    inv_struct = Structure.from_file(str(CIF_DIR / "Inverted_Rutile_Invalid.cif"))

    report = gatekeeper.audit_structure(inv_struct)
    assert report.is_valid is False
    assert report.gii_status == "REJECT" or len(report.errors) > 0


def test_tier4_gatekeeper_reject_thermal_factor_inversion():
    """Verify that heavy cation B_iso > 1.5 * anion B_iso triggers rejection."""
    gatekeeper = CrystallochemicalGatekeeper()
    struct = Structure.from_file(str(CIF_DIR / "TiO2_rutile.cif"))

    # Unphysical inversion: Ti (M=47.9) has B_iso=2.5 while O (M=16.0) has B_iso=0.4
    unphysical_b_iso = {"Ti": 2.5, "O": 0.4}
    report = gatekeeper.audit_structure(struct, b_iso_dict=unphysical_b_iso)
    assert report.is_valid is False
    assert len(report.b_iso_violations) > 0


def test_tier4_gatekeeper_survives_spglib_error(monkeypatch):
    """A symmetry-analysis failure is recorded as a warning, not raised.

    spglib 2.7 still runs in legacy mode and returns ``None`` on failure, but from
    2.8.0 the default flips to raising ``SpglibError``, which derives from
    ``Exception`` rather than from the ``ValueError``/``RuntimeError`` family. The
    audit has to behave identically in both generations.
    """
    struct = Structure.from_file(str(CIF_DIR / "TiO2_rutile.cif"))

    def _raise_symmetry_failure(*_args, **_kwargs):
        raise spglib.SpglibError("simulated symmetry analysis failure")

    monkeypatch.setattr(spglib, "get_symmetry_dataset", _raise_symmetry_failure)

    report = CrystallochemicalGatekeeper().audit_structure(struct)

    assert report.is_valid is True
    assert not report.errors
    assert any("spglib symmetry analysis notice" in w for w in report.warnings)


def test_tier3_calc_pbc_distance_boundary_wrap():
    """Verify Minimum Image Convention handles boundary wraparound correctly."""
    cell = gemmi.UnitCell(4.5937, 4.5937, 2.9587, 90.0, 90.0, 90.0)
    # Positions near 0.0 and 1.0 along x-axis
    f1 = np.array([0.01, 0.5, 0.5])
    f2 = np.array([0.99, 0.5, 0.5])

    dist_pbc = calc_pbc_distance(cell, f1, f2)
    # With PBC, delta_x is 0.02 * 4.5937 = ~0.0919 A
    assert np.isclose(dist_pbc, 0.02 * 4.5937, atol=1e-4)
    assert dist_pbc < 1.20


def test_tier3_prune_close_contacts_pbc():
    """Verify prune_close_contacts eliminates periodic boundary ghost ripples."""
    supervisor = MultiStartChargeFlipping(min_distance_angstrom=1.20)
    cell_lengths = (4.5937, 4.5937, 2.9587)
    cell_angles = (90.0, 90.0, 90.0)

    # Site 1 at (0.01, 0.5, 0.5) and Site 2 at (0.99, 0.5, 0.5)
    # Naive Euclidean distance without PBC is |0.01 - 0.99| * 4.5937 = 4.50 A (would not be pruned)
    # With PBC, distance is 0.092 A < 1.2 A (must be pruned!)
    sites = [
        ("Ti", np.array([0.01, 0.5, 0.5])),
        ("Ti", np.array([0.99, 0.5, 0.5])),
        ("O", np.array([0.50, 0.5, 0.5])),
    ]
    pruned, pruned_count = supervisor.prune_close_contacts(cell_lengths, cell_angles, sites)
    assert pruned_count == 1
    assert len(pruned) == 2


def test_extract_peaks_robust_noise_rejection():
    """Verify robust peak extraction filters out baseline noise."""
    rng = np.random.default_rng(123)
    two_theta = np.linspace(10.0, 90.0, 800)
    flat_noise = rng.normal(50.0, 2.0, size=len(two_theta))

    # Pattern with 2 sharp peaks above noise
    signal = flat_noise.copy()
    signal += 150.0 * np.exp(-0.5 * ((two_theta - 27.4) / 0.15) ** 2)
    signal += 80.0 * np.exp(-0.5 * ((two_theta - 36.1) / 0.15) ** 2)

    peaks = extract_peaks_robust(two_theta, signal)
    # Should identify the 2 genuine peaks and not noise fluctuations
    assert len(peaks) == 2
    assert np.any(np.isclose(peaks.two_theta, 27.4, atol=0.2))
    assert np.any(np.isclose(peaks.two_theta, 36.1, atol=0.2))

    # Every line carries its own characterisation, not just a position.
    assert peaks.d_spacing.shape == peaks.two_theta.shape
    assert peaks.height.shape == peaks.two_theta.shape
    assert peaks.fwhm.shape == peaks.two_theta.shape
    assert peaks.overlapped.dtype == bool
    assert np.all(peaks.height > 0.0)
    assert np.all(peaks.fwhm > 0.0)
    # Interplanar spacing must follow Bragg's law for the given wavelength.
    assert np.allclose(
        peaks.d_spacing,
        1.540596 / (2.0 * np.sin(np.radians(peaks.two_theta / 2.0))),
    )


def test_extract_peaks_marks_blended_doublet():
    """Lines closer than 1.5 FWHM are flagged blended; isolated lines are not.

    A blended line cannot be attributed to a single phase, so the flag has to be
    part of the extracted table rather than inferred later by the caller.
    """
    rng = np.random.default_rng(7)
    # A fine grid matters: the Savitzky-Golay detection window is a fixed number of
    # samples (15), so at a coarse step it is wider than the line itself and would
    # smear a one-FWHM doublet into a single maximum.
    two_theta = np.linspace(20.0, 60.0, 8000)  # step = 0.005 deg
    background = rng.normal(30.0, 1.0, size=two_theta.size)

    single = extract_peaks_robust(
        two_theta, background + 200.0 * np.exp(-0.5 * ((two_theta - 30.0) / 0.05) ** 2)
    )
    assert len(single) == 1
    assert bool(single.overlapped[0]) is False

    # Two lines 1.2 FWHM apart: separated by more than 2 sigma, so still resolved
    # as two maxima, but closer than 1.5 FWHM and therefore blended.
    separation = 1.2 * float(single.fwhm[0])
    doublet = extract_peaks_robust(
        two_theta,
        background
        + 200.0 * np.exp(-0.5 * ((two_theta - (30.0 - separation / 2.0)) / 0.05) ** 2)
        + 200.0 * np.exp(-0.5 * ((two_theta - (30.0 + separation / 2.0)) / 0.05) ** 2),
    )
    assert len(doublet) == 2, f"doublet not resolved: {doublet.two_theta}"
    assert bool(np.all(doublet.overlapped))


def test_pipeline_e2e_mixture_report(tmp_path):
    """Verify full end-to-end execution of thin-glue pipeline producing a strongly typed PipelineReport."""
    data_file = PATTERNS_DIR / "Mixture_Rutile95_Anatase5.xy"
    report = run_pipeline(
        data_path=data_file,
        cif_dir=CIF_DIR,
        elements=["Ti", "O"],
        output_dir=tmp_path,
    )
    assert isinstance(report, PipelineReport)
    assert report.status == "SUCCESS"
    assert report.data_points > 100
    assert len(report.phases_identified) >= 1
    assert report.qpa_weights_wt
    assert sum(report.qpa_weights_wt.values()) == pytest.approx(100.0, abs=0.5)
    assert all(p.weight_fraction_wt is not None for p in report.phases_identified)
    assert report.audit_passed is True
    assert len(report.stage_gate_protocol) == 12


def test_gsas2_installer_contract(tmp_path, monkeypatch):
    """Verify install_gsas2 workflow and target layout creation."""
    from install_gsas2 import install_gsas2

    def mock_run(cmd, *args, **_):
        class MockRes:
            stdout = "Cloned"
            stderr = ""

        (tmp_path / "GSASII").mkdir(parents=True, exist_ok=True)
        return MockRes()

    monkeypatch.setattr("subprocess.run", mock_run)

    install_gsas2(target_dir=tmp_path)
    assert (tmp_path / "GSASII").exists()


# =========================================================================
# TIER 5: Phase 8 Expert Crystallographer Tools Tests
# =========================================================================
def test_cif_retriever_parameterized_search_and_analog_generation(tmp_path, monkeypatch):
    """Verify parameterized COD search query execution and isostructural analog generation."""
    from cif_retriever import CodClient, generate_isostructural_analog

    # Test CodClient.search URL generation and parsing with mocked urlopen
    monkeypatch.delenv("COD_HSRDB_PATH", raising=False)
    monkeypatch.setattr("env_file.load_env_file", lambda: None)
    monkeypatch.setattr("hs_rdb.load_env_file", lambda: None)
    client = CodClient(cache_dir=tmp_path / "cache")

    def mock_urlopen(req, *args, **_):
        class MockResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                # Return dummy COD search result JSON
                return b'[{"file": "9008719", "chemname": "Sm S", "a": "5.97", "sg": "F m -3 m"}]'

        return MockResp()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    results = client.search(elements=["Sm", "S"])
    assert len(results) == 1
    assert results[0].id == "cod-9008719"
    assert results[0].formula == "Sm S"

    # Test analog generation: CeO2 (fluorite) -> ThO2
    proto_cif = CIF_DIR / "CeO2.cif"
    out_analog = tmp_path / "ThO2_analog.cif"
    generated = generate_isostructural_analog(
        prototype_cif=proto_cif,
        species_mapping={"Ce": "Th"},
        out_cif=out_analog,
        anion_symbol="O",
    )
    assert generated.exists()
    assert generated.stat().st_size > 0
    # Verify resulting structure
    th_struct = Structure.from_file(str(generated))
    species_symbols = [s.symbol for s in th_struct.composition.elements]
    assert "Th" in species_symbols
    assert "Ce" not in species_symbols

    # The written cell must carry the symmetry detected from the prototype structure.
    # Writing without symmetry re-detection emits a P1 cell listing every site, which
    # is not an isostructural analog for Rietveld use: no space-group constraints
    # apply and the phase refines as triclinic. The check reads the CIF text, because
    # parsing the P1 file back would re-detect the symmetry and hide the defect.
    def _declared_spacegroup(cif_path: Path) -> str:
        for line in cif_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("_symmetry_space_group_name_H-M"):
                return line.split()[-1].strip("'\"").replace(" ", "")
        return ""

    # Compared against the symmetry detected from the sites, not against the label the
    # prototype file happens to declare: the test fixtures themselves are written as P1.
    detected = Structure.from_file(str(proto_cif)).get_space_group_info()[0].replace(" ", "")
    assert detected == "Fm-3m"
    assert _declared_spacegroup(generated) == detected


def test_xrd_inspect_pre_inspection():
    """Verify experimental pattern pre-inspection for SNR, FWHM and crystallinity."""
    from xrd_inspect import CrystallinityGrade, inspect_powder_pattern

    data = np.loadtxt(PATTERNS_DIR / "Mixture_Rutile95_Anatase5.xy")
    two_theta, intensity = data[:, 0], data[:, 1]

    report = inspect_powder_pattern(two_theta, intensity)
    assert report.total_points > 100
    assert report.signal_to_noise_ratio > 10.0
    assert report.detected_peaks_count >= 5
    assert report.crystallinity_grade in (
        CrystallinityGrade.HIGH,
        CrystallinityGrade.MODERATE,
    )
    assert report.has_amorphous_halo is False


def test_xrd_diagnose_residual_classification():
    """Verify classification of unindexed impurity peaks and residual symptoms."""
    from xrd_diagnose import DiagnosticStatus, SymptomCode, diagnose_residuals

    two_theta = np.linspace(10.0, 70.0, 600)
    # Simulate observed pattern with a known peak at 27.5 and an impurity at 38.0
    y_obs = 50.0 + 200.0 * np.exp(-0.5 * ((two_theta - 27.5) / 0.15) ** 2)
    y_obs += 80.0 * np.exp(-0.5 * ((two_theta - 38.0) / 0.15) ** 2)

    # Calculated model only knows about the 27.5 peak
    y_calc = 50.0 + 200.0 * np.exp(-0.5 * ((two_theta - 27.5) / 0.15) ** 2)

    # Bragg peaks for model: 27.5
    diagnosis = diagnose_residuals(
        two_theta=two_theta,
        y_obs=y_obs,
        y_calc=y_calc,
        wavelength=1.5406,
        bragg_peaks_2theta=[27.5],
    )

    assert diagnosis.overall_status == DiagnosticStatus.FAIL
    assert diagnosis.total_symptoms >= 1
    sym_codes = [s.code for s in diagnosis.symptoms]
    assert SymptomCode.UNINDEXED_PHASE in sym_codes
    assert len(diagnosis.recommended_actions) >= 1


def test_xrd_plot_publication_export(tmp_path):
    """Verify publication-grade 600 DPI plot creation."""
    from gsas2_adapter import ProfileCurves
    from xrd_plot import plot_gsas2_histogram

    two_theta = np.linspace(10.0, 70.0, 300)
    y_obs = 100.0 + 500.0 * np.exp(-0.5 * ((two_theta - 30.0) / 0.2) ** 2)
    y_calc = 100.0 + 490.0 * np.exp(-0.5 * ((two_theta - 30.0) / 0.2) ** 2)
    y_bkg = np.full_like(two_theta, 100.0)
    y_diff = y_obs - y_calc
    sigma_obs = np.sqrt(np.maximum(y_obs, 1.0))
    weights = np.where(y_obs > 0, 1.0 / np.maximum(y_obs, 1.0), 1.0)

    plot_file = tmp_path / "test_refinement.png"
    curves = ProfileCurves(
        two_theta=two_theta,
        y_obs=y_obs,
        y_calc=y_calc,
        y_bkg=y_bkg,
        y_diff=y_diff,
        sigma_obs=sigma_obs,
        weights=weights,
    )
    out_path = plot_gsas2_histogram(
        hist=curves,
        out_path=plot_file,
        dpi=300,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_stage_gate_phase_data_cif_reconstruction(tmp_path):
    """Verify sanitize_cif_for_gsas2 strips non-standard oxidation states while preserving crystal metrics."""
    from xrd_project import sanitize_cif_for_gsas2

    raw_cif = tmp_path / "source_oxi.cif"
    raw_cif.write_text(
        (CIF_DIR / "CeO2.cif").read_text(encoding="utf-8").replace(" Ce ", " Ce4+ "),
        encoding="utf-8",
    )
    staged = sanitize_cif_for_gsas2(raw_cif, tmp_path / "staging")
    assert staged.exists()
    assert "Ce4+" not in staged.read_text(encoding="utf-8")
    loaded_struct = Structure.from_file(str(staged))
    assert loaded_struct.composition.reduced_formula == "CeO2"


# =========================================================================
# TIER 5: Advanced Algorithms and Diagnostics
# =========================================================================
def test_tier5_hill_howard_qpa_exact_weights():
    """Verify Hill-Howard (1987) QPA formula across multi-phase structures via get_project_qpa."""
    import types

    from gsas2_adapter import get_project_qpa

    ceo2_struct = Structure.from_file(str(CIF_DIR / "CeO2.cif"))
    tio2_struct = Structure.from_file(str(CIF_DIR / "TiO2_rutile.cif"))

    def _make_phase(name, struct, scale):
        gen = {
            "SGData": {"SGZ": 1.0},
            "Mass": float(struct.composition.weight),
            "Cell": [
                0,
                struct.lattice.a,
                struct.lattice.b,
                struct.lattice.c,
                struct.lattice.alpha,
                struct.lattice.beta,
                struct.lattice.gamma,
                float(struct.volume),
            ],
        }
        hists = {"PWDR.csv": {"Scale": [scale, True]}}
        return types.SimpleNamespace(
            name=name,
            data={"General": gen, "Histograms": hists},
        )

    p1 = _make_phase("CeO2", ceo2_struct, 1.0)
    p2 = _make_phase("TiO2", tio2_struct, 2.0)
    hist = types.SimpleNamespace(name="PWDR.csv", data={"Histograms": {}})
    gpx = types.SimpleNamespace(phases=lambda: [p1, p2], histograms=lambda: [hist])

    weights = get_project_qpa(gpx, hist)
    assert "CeO2" in weights
    assert "TiO2" in weights
    total_wt = sum(w[0] for w in weights.values())
    assert abs(total_wt - 100.0) < 0.5

    # Check that scale = 0 yields 0 wt.%
    p2_zero = _make_phase("TiO2", tio2_struct, 0.0)
    gpx_zero = types.SimpleNamespace(phases=lambda: [p1, p2_zero], histograms=lambda: [hist])
    zero_weights = get_project_qpa(gpx_zero, hist)
    assert zero_weights["CeO2"][0] == 100.0
    assert zero_weights["TiO2"][0] == 0.0


def test_tier5_analytical_bragg_shift():
    """Verify first-order analytical Bragg peak shift and tan(theta) dispersion."""
    from search_match import apply_bragg_cell_shift

    two_theta = np.array([20.0, 50.0, 80.0])

    # Lattice expansion (da > 0) -> peaks shift to smaller angles
    expanded = apply_bragg_cell_shift(two_theta, delta_a_over_a=0.01)
    assert np.all(expanded < two_theta)

    # Lattice contraction (da < 0) -> peaks shift to larger angles
    contracted = apply_bragg_cell_shift(two_theta, delta_a_over_a=-0.01)
    assert np.all(contracted > two_theta)

    # Check tan(theta) angular dispersion: shift at 80 deg must be strictly greater than shift at 20 deg
    shift_20 = abs(expanded[0] - two_theta[0])
    shift_80 = abs(expanded[2] - two_theta[2])
    assert shift_80 > shift_20


def test_tier5_local_cif_harvesting(tmp_path):
    """Verify local CIF harvesting with pymatgen elemental filtering."""
    import shutil

    from cif_retriever import harvest_local_cifs

    local_repo = tmp_path / "repo"
    local_repo.mkdir()
    shutil.copy(CIF_DIR / "CeO2.cif", local_repo / "CeO2.cif")
    shutil.copy(CIF_DIR / "TiO2_rutile.cif", local_repo / "TiO2_rutile.cif")

    out_cifs = tmp_path / "selected"

    # Filter strictly for Ce, O
    harvested = harvest_local_cifs(local_repo, out_cifs, elements=["Ce", "O"])
    assert len(harvested) == 1
    assert "CeO2.cif" in harvested[0].name

    # Filter for Ti, O
    harvested_ti = harvest_local_cifs(local_repo, out_cifs, elements=["Ti", "O"])
    assert len(harvested_ti) == 1
    assert "TiO2_rutile.cif" in harvested_ti[0].name


def test_tier5_publication_plot_with_3sigma_corridor(tmp_path):
    """Verify publication-grade plotting with multi-phase Bragg ticks and +-3sigma noise corridor."""
    from gsas2_adapter import ProfileCurves
    from xrd_plot import plot_gsas2_histogram

    rng = np.random.default_rng(42)
    two_theta = np.linspace(10.0, 80.0, 500)
    y_obs = 100.0 + 500.0 * np.exp(-0.5 * ((two_theta - 28.5) / 0.15) ** 2) + rng.normal(0, 5, 500)
    y_calc = 100.0 + 490.0 * np.exp(-0.5 * ((two_theta - 28.5) / 0.15) ** 2)
    y_bkg = np.full_like(two_theta, 100.0)
    y_diff = y_obs - y_calc
    weights = np.where(y_obs > 0, 1.0 / np.maximum(y_obs, 1.0), 1.0)
    sigma_obs = np.sqrt(np.maximum(y_obs, 1.0))

    plot_file = tmp_path / "test_pub_plot.png"

    curves = ProfileCurves(
        two_theta=two_theta,
        y_obs=y_obs,
        y_calc=y_calc,
        y_bkg=y_bkg,
        y_diff=y_diff,
        sigma_obs=sigma_obs,
        weights=weights,
    )

    out_img = plot_gsas2_histogram(curves, out_path=plot_file, dpi=300)
    assert out_img.exists()
    assert out_img.stat().st_size > 5000


# =========================================================================
# TIER 6: Truthful Reporting, Occam's Razor & Residual Classification
# =========================================================================
def test_tier6_residual_classification_bragg_vs_impurity():
    """Verify dual residual classification:
    - Peak at known Bragg angle is classified as INTENSITY_DEFECT.
    - Peak in background without Bragg lines is classified as TRUE_UNINDEXED_IMPURITY.
    """
    from xrd_diagnose import classify_residual_peaks, diagnose_residuals

    two_theta = np.linspace(10.0, 60.0, 1000)
    # Background
    y_calc = np.full_like(two_theta, 100.0)
    y_obs = np.full_like(two_theta, 100.0)

    # Add peak at 25.53 deg (Bragg position of Phase A)
    y_obs += 300.0 * np.exp(-0.5 * ((two_theta - 25.53) / 0.10) ** 2)
    # Add peak at 42.00 deg (no Bragg line nearby - true impurity)
    y_obs += 250.0 * np.exp(-0.5 * ((two_theta - 42.00) / 0.10) ** 2)

    phase_braggs = {
        "Phase_A": [25.50, 30.10, 35.20],
    }

    true_unindexed, intensity_defects, _artifacts = classify_residual_peaks(
        two_theta, y_obs, y_calc, phase_bragg_peaks=phase_braggs, angular_tolerance=0.20
    )

    from xrd_diagnose import ResidualPeakType

    assert len(intensity_defects) == 1
    assert intensity_defects[0].matched_phase == "Phase_A"
    assert abs(intensity_defects[0].two_theta - 25.53) < 0.05
    assert intensity_defects[0].defect_type == ResidualPeakType.INTENSITY_DEFECT

    assert len(true_unindexed) == 1
    assert abs(true_unindexed[0].two_theta - 42.00) < 0.05
    assert true_unindexed[0].peak_type == ResidualPeakType.TRUE_UNINDEXED_IMPURITY

    # Test full diagnose_residuals output
    diag = diagnose_residuals(two_theta, y_obs, y_calc, phase_bragg_peaks=phase_braggs)
    assert diag.true_unindexed_count == 1
    assert diag.intensity_defects_count == 1


def test_tier6_complexity_penalty_many_lines():
    """Verify the parsimony penalty scales with the number of theoretical lines.

    The complexity variable is the line count in the window (TL), not the cell
    volume: E[ML] = TL * P * 2*delta/W, so TL governs how many matches a
    candidate obtains by chance. Two candidates of equal volume but different TL
    must not receive equal penalties -- that is how a rich-line candidate
    outranked the physically correct low-line phase on the real series.
    """
    from search_match import calculate_complexity_penalty, count_unique_matched_peaks

    # Penalty tests: TL <= n0 (20 lines) is complexity-free, many lines are not
    pen_few = calculate_complexity_penalty(20.0)
    assert pen_few == 1.0  # <= n0 has no penalty

    pen_many = calculate_complexity_penalty(200.0)
    assert pen_many < 0.70  # 200 lines in the window are penalised significantly
    assert calculate_complexity_penalty(60.0) < calculate_complexity_penalty(30.0)

    # Residual gate: peaks already explained by the ACCEPTED phase keep no credit
    covering = np.array([20.0, 30.0, 40.0])
    exp_2th = np.array([20.01, 30.02, 39.99])  # only peaks that overlap with simple phase

    unique_count = count_unique_matched_peaks(
        matched_exp_indices=[0, 1, 2],
        exp_two_theta=exp_2th,
        covering_2thetas=covering,
        angular_tolerance=0.20,
    )
    assert unique_count == 0  # everything is already explained by the accepted phase

    # A peak the accepted phase does NOT explain must keep its credit.
    exp_mixed = np.array([20.01, 30.02, 55.0])
    assert (
        count_unique_matched_peaks(
            matched_exp_indices=[0, 1, 2],
            exp_two_theta=exp_mixed,
            covering_2thetas=covering,
            angular_tolerance=0.20,
        )
        == 1
    )

    # Empty covering set credits every matched peak.
    assert (
        count_unique_matched_peaks(
            matched_exp_indices=[0, 1, 2],
            exp_two_theta=exp_2th,
            covering_2thetas=np.array([]),
        )
        == 3
    )


def test_tier6_deterministic_report_no_hallucinations(tmp_path):
    """Verify report generation strictly uses calculation metrics without optimistic rhetoric."""
    from export import (
        BackgroundSummary,
        ExportSummary,
        InstrumentSummary,
        PhaseExportSummary,
        SampleSummary,
    )
    from gatekeeper import GiiStatus
    from stage_gate_rietveld import RefinementStatus
    from xrd_diagnose import (
        DiagnosticStatus,
        MicrostructureMetrics,
        ResidualDiagnosisReport,
        ResidualPeak,
        ResidualPeakType,
    )
    from xrd_report import generate_honest_markdown_report

    defect_peak = ResidualPeak(
        two_theta=25.53,
        d_spacing_angstrom=3.486,
        residual_height=140.0,
        significance_sigma=5.2,
        peak_type=ResidualPeakType.INTENSITY_DEFECT,
        matched_phase="Sm3S4",
        bragg_2theta=25.50,
        delta_2theta=0.03,
    )
    diag_data = ResidualDiagnosisReport(
        overall_status=DiagnosticStatus.NEEDS_REFINEMENT,
        r_p_percent=11.20,
        r_wp_percent=15.50,
        r_wp_over_r_p_ratio=1.38,
        durbin_watson=1.80,
        noise_sigma=5.0,
        i_max_net=1000.0,
        lod_3sigma_percent=1.50,
        true_unindexed_count=0,
        residual_artifact_count=0,
        residual_artifacts=[],
        intensity_defects_count=1,
        true_unindexed_reflections=[],
        intensity_mismatch_reflections=[defect_peak],
        phase_r_factors={},
        total_symptoms=0,
        symptoms=[],
        recommended_actions=[],
    )
    summary_data = ExportSummary(
        project="test.gpx",
        status=RefinementStatus.SUCCESS,
        r_wp_percent=15.50,
        r_p_percent=11.20,
        chi_square=3.70,
        phase_rb_percent={"Sm3S4": 6.2, "Sm2O2S": 9.1},
        r_exp_percent=8.06,
        r_wp_over_r_exp=1.92,
        wr_bkg_percent=14.10,
        gof=1.92,
        durbin_watson=1.80,
        gii_value=0.12,
        gii_status=GiiStatus.ACCEPTABLE,
        free_parameters=12,
        true_unindexed_peaks=[],
        intensity_mismatch_peaks=[defect_peak],
        symptoms=[],
        phases={
            "Sm3S4": PhaseExportSummary(
                weight_fraction_percent=88.5,
                weight_fraction_esd=0.4,
                scale=1.2e-3,
                cell_parameters={"a": 8.54, "b": 8.54, "c": 8.54, "volume": 622.8},
                microstructure=MicrostructureMetrics(
                    crystallite_size_nm=85.0, microstrain_percent=0.12
                ),
                r_bragg_percent=6.2,
                r_f_percent=4.1,
            ),
            "Sm2O2S": PhaseExportSummary(
                weight_fraction_percent=11.5,
                weight_fraction_esd=0.4,
                scale=1.5e-4,
                cell_parameters={"a": 3.89, "b": 3.89, "c": 6.70, "volume": 87.8},
                microstructure=MicrostructureMetrics(
                    crystallite_size_nm=60.0, microstrain_percent=0.15
                ),
                r_bragg_percent=9.1,
                r_f_percent=5.8,
            ),
        },
        microstructure={},
        instrument=InstrumentSummary(),
        sample=SampleSummary(),
        background=BackgroundSummary(),
        unmet_criteria=[],
        exportable=False,
        history=[],
    )

    out_file = tmp_path / "HONEST_REPORT.md"
    content = generate_honest_markdown_report(summary_data, diag_data, out_path=out_file)

    assert out_file.exists()
    assert "15.50%" in content
    assert "3.70" in content
    assert "Истинные неиндексированные пики примесей**: **0**" in content
    assert "Дефекты интенсивности/текстуры известных фаз**: **1**" in content
    assert "Sm3S4" in content
    assert "R_wp < 10%" not in content
    assert "R_{wp} < 10%" not in content


# =========================================================================
# TIER 7: IUCr Publication Standards, e.s.d. & Curve Export
# =========================================================================
def test_tier7_durbin_watson_and_rp_diagnostics():
    """Verify Durbin-Watson d-statistic and R_wp/R_p discrepancy diagnostics."""
    from xrd_diagnose import calculate_durbin_watson, diagnose_residuals

    rng = np.random.default_rng(42)

    # 1. Pure white noise residual -> d approx 2.0
    white_noise = rng.normal(0.0, 5.0, size=2000)
    dw_white = calculate_durbin_watson(white_noise)
    assert 1.8 <= dw_white <= 2.2

    # 2. Strongly correlated low-frequency oscillation -> d < 0.5
    t = np.linspace(0.0, 4.0 * np.pi, 2000)
    correlated_wave = 10.0 * np.sin(t)
    dw_corr = calculate_durbin_watson(correlated_wave)
    assert dw_corr < 0.20

    # 3. Verify symptom emission in diagnose_residuals
    two_theta = np.linspace(10.0, 70.0, 1000)
    y_calc = np.full_like(two_theta, 100.0)
    y_obs = y_calc + 20.0 * np.sin(np.linspace(0, 2 * np.pi, 1000))

    from xrd_diagnose import SymptomCode

    diag = diagnose_residuals(two_theta, y_obs, y_calc)
    assert diag.durbin_watson < 1.0
    symptom_codes = [s.code for s in diag.symptoms]
    assert SymptomCode.DURBIN_WATSON_AUTOCORRELATION in symptom_codes


def test_tier7_bragg_r_factors_rb_rf():
    """Verify calculation of phase-specific Bragg R-factor (R_B) and Structure Factor (R_F)."""
    from xrd_diagnose import calculate_bragg_r_factors

    two_theta = np.linspace(10.0, 50.0, 800)
    y_obs = np.full_like(two_theta, 50.0)
    y_calc = np.full_like(two_theta, 50.0)

    # Identical reflections for Phase_1
    phase_braggs = {"Phase_1": [20.0, 30.0, 40.0]}
    for bp in [20.0, 30.0, 40.0]:
        peak = 200.0 * np.exp(-0.5 * ((two_theta - bp) / 0.15) ** 2)
        y_obs += peak
        y_calc += peak

    r_factors_ideal = calculate_bragg_r_factors(two_theta, y_obs, y_calc, phase_braggs)
    assert r_factors_ideal["Phase_1"].r_bragg_percent == 0.0
    assert r_factors_ideal["Phase_1"].r_f_percent == 0.0

    # Perturbed peak intensity (mismatch in structure factor)
    y_obs_perturbed = y_obs.copy()
    y_obs_perturbed += 50.0 * np.exp(-0.5 * ((two_theta - 20.0) / 0.15) ** 2)

    r_factors_pert = calculate_bragg_r_factors(two_theta, y_obs_perturbed, y_calc, phase_braggs)
    assert r_factors_pert["Phase_1"].r_bragg_percent > 0.0
    assert r_factors_pert["Phase_1"].r_f_percent > 0.0


def test_tier7_microstructure_scherrer_stokes_wilson():
    """Verify physical microstructure deconvolution (Crystallite size <L> and Microstrain epsilon)."""
    from xrd_diagnose import calculate_microstructure_parameters

    # Case 1: X = 0.10 deg, Y = 0.04 deg on Cu Ka (1.5406 A)
    # Scherrer: <L> = 180 * 0.90 * 0.15406 / (pi * 0.10) = 79.44 nm
    # Stokes-Wilson (beta = 4 * eps * tan(theta)): eps = pi * 0.04 / 720 * 100% = 0.01745%
    res = calculate_microstructure_parameters(
        x_lorentzian=0.10, y_lorentzian=0.04, wavelength_angstrom=1.5406
    )
    assert res.crystallite_size_nm is not None
    assert res.microstrain_percent is not None
    assert abs(res.crystallite_size_nm - 79.44) < 0.2
    assert abs(res.microstrain_percent - 0.0175) < 0.001

    res_nano = calculate_microstructure_parameters(
        x_lorentzian=0.50, y_lorentzian=0.0, wavelength_angstrom=1.5406
    )
    assert res_nano.crystallite_size_nm is not None
    assert abs(res_nano.crystallite_size_nm - 15.89) < 0.1
    assert res_nano.microstrain_percent == 0.0


def test_tier7_profile_curves_csv_export(tmp_path):
    """Verify export of profile_curves.csv and bragg_reflections.csv for Origin / Excel."""
    data_file = PATTERNS_DIR / "Mixture_Rutile95_Anatase5.xy"
    report = run_pipeline(
        data_path=data_file,
        cif_dir=CIF_DIR,
        elements=["Ti", "O"],
        output_dir=tmp_path,
    )

    prof_csv = tmp_path / "profile_curves.csv"
    assert prof_csv.exists()
    assert prof_csv.stat().st_size > 1000

    # Verify CSV format and columns
    lines = prof_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "two_theta,y_obs,y_calc,y_bkg,y_diff,sigma_obs"
    # Verify numerical parse of data rows
    first_row = [float(x) for x in lines[1].split(",")]
    assert len(first_row) == 6

    bragg_csv = tmp_path / "bragg_reflections.csv"
    assert bragg_csv.exists()
    b_lines = bragg_csv.read_text(encoding="utf-8").strip().splitlines()
    assert "phase" in b_lines[0] and "two_theta" in b_lines[0] and "d_spacing" in b_lines[0]

    # Verify report fields
    assert report.profile_curves_path == str(prof_csv)
    assert report.bragg_reflections_path == str(bragg_csv)
    assert report.durbin_watson > 0.0


def test_tier7_bragg_r_factors_voronoi_partitioning():
    """Verify 1D Voronoi midpoint partitioning prevents double-counting in close doublets."""
    from xrd_diagnose import calculate_bragg_r_factors

    two_theta = np.linspace(28.0, 32.0, 401)  # step = 0.01 deg
    y_bkg = np.full_like(two_theta, 20.0)
    y_obs = y_bkg.copy()
    y_calc = y_bkg.copy()

    # Close doublet separated by 0.10 deg (peak_window = 0.25 deg)
    p1, p2 = 30.00, 30.10
    peak1 = 100.0 * np.exp(-0.5 * ((two_theta - p1) / 0.04) ** 2)
    peak2 = 150.0 * np.exp(-0.5 * ((two_theta - p2) / 0.04) ** 2)
    y_obs += peak1 + peak2
    y_calc += peak1 + peak2

    # Test with unsorted peaks to test sorting guarantee
    phase_braggs = {"Doublet_Phase": [30.10, 30.00]}

    res = calculate_bragg_r_factors(
        two_theta=two_theta,
        y_obs=y_obs,
        y_calc=y_calc,
        phase_bragg_peaks=phase_braggs,
        y_bkg=y_bkg,
        peak_window=0.25,
    )

    # Identical observed and calculated -> R_B = 0%
    assert res["Doublet_Phase"].r_bragg_percent == 0.0
    assert res["Doublet_Phase"].r_f_percent == 0.0
    assert res["Doublet_Phase"].reflections_count == 2

    # Net integrated intensity under peaks above background
    total_net_integral = float(np.sum(np.maximum(0.0, y_obs - y_bkg)))
    integrated_obs = res["Doublet_Phase"].integrated_obs_intensity

    # Thanks to Voronoi partitioning, integrated_obs cannot exceed total_net_integral (no double counting!)
    assert integrated_obs <= total_net_integral * 1.001


def test_tier7_native_gsas2_profile_curves_structure():
    """Verify RefinementMetrics.to_dict() and AuditReport.to_dict() preserve contract fields."""
    from gsas2_adapter import ProfileCurves, calculate_profile_metrics

    two_theta = np.array([10.0, 10.02, 10.04])
    y_obs = np.array([100.0, 105.0, 98.0])
    y_calc = np.array([99.0, 104.0, 97.0])
    curves = ProfileCurves(
        two_theta=two_theta,
        y_obs=y_obs,
        y_calc=y_calc,
        y_bkg=np.full(3, 10.0),
        y_diff=y_obs - y_calc,
        sigma_obs=np.sqrt(y_obs),
        weights=1.0 / y_obs,
    )
    metrics_dict = calculate_profile_metrics(curves, n_free_parameters=1).to_dict()
    assert metrics_dict["n_data_points"] == 3
    assert metrics_dict["n_parameters"] == 1
    assert "gof" in metrics_dict and "reduced_chi2" in metrics_dict


def test_fit_physical_background_valleys_noise():
    """Verify Peak-Clipped Chebyshev baseline runs symmetrically through valley noise."""
    from instrument import fit_physical_background

    data = np.loadtxt(PATTERNS_DIR / "CeO2.xy")
    two_theta, intensity = data[:, 0], data[:, 1]

    bkg = fit_physical_background(two_theta, intensity, n_terms=6)
    assert len(bkg) == len(intensity)

    # In peak-free flat valley region (e.g. 2theta between 35 and 45 deg)
    valley_mask = (two_theta >= 35.0) & (two_theta <= 45.0)
    residuals_valley = intensity[valley_mask] - bkg[valley_mask]

    # Mean residual in pure valley should be close to 0 (center of noise corridor)
    mean_valley_res = float(np.mean(residuals_valley))
    std_valley_res = float(np.std(residuals_valley))
    # Relative center offset within small fraction of standard deviation
    assert abs(mean_valley_res) < 0.8 * std_valley_res


def test_validate_cif_identity():
    """Verify CIF identity validation blocks counterfeit or contaminated phases."""
    import pytest
    from cif_retriever import validate_cif_identity

    ceo2_path = CIF_DIR / "CeO2.cif"

    # Valid check with both elements
    assert validate_cif_identity(ceo2_path, expected_elements=["Ce", "O"]) is True

    # Rejection: expected single metal Ce, but CIF contains O
    with pytest.raises(ValueError, match="содержит элементы"):
        validate_cif_identity(ceo2_path, expected_elements=["Ce"])

    # Rejection: foreign elements
    with pytest.raises(ValueError, match="посторонние элементы"):
        validate_cif_identity(ceo2_path, expected_elements=["Sm", "S"])


def test_tier1_refinement_config_domain_neutrality():
    """Verify RefinementConfig defaults, custom parameters, and strict domain neutrality."""
    from stage_gate_rietveld import RefinementConfig

    # 1. Defaults verification
    default_cfg = RefinementConfig()
    assert default_cfg.bkg_chebyshev_terms == 6
    assert default_cfg.refine_sample_displacement is False
    assert default_cfg.refine_axial_divergence is False
    assert default_cfg.texture_axes == {}
    assert default_cfg.march_dollase_bounds == (0.7, 1.3)

    # 2. Custom configuration instantiation (IoC)
    custom_cfg = RefinementConfig(
        bkg_chebyshev_terms=10,
        refine_sample_displacement=True,
        refine_axial_divergence=True,
        texture_axes={"PhaseA": [1, 0, 0]},
        march_dollase_bounds=(0.65, 1.35),
    )
    assert custom_cfg.bkg_chebyshev_terms == 10
    assert custom_cfg.refine_sample_displacement is True
    assert custom_cfg.refine_axial_divergence is True
    assert custom_cfg.texture_axes["PhaseA"] == [1, 0, 0]

    # 3. Source code audit: Strict Domain Neutrality in stage_gate_rietveld.py
    source = (SCRIPTS_DIR / "stage_gate_rietveld.py").read_text(encoding="utf-8")
    assert "SmS" not in source, (
        "Violation of Principle 8: Hardcoded 'SmS' found in stage_gate_rietveld.py"
    )
    assert "Quartz" not in source, (
        "Violation of Principle 8: Hardcoded 'Quartz' found in stage_gate_rietveld.py"
    )


def test_pipeline_refinement_config_propagation():
    """Verify run_pipeline signature accepts RefinementConfig and propagates it."""
    import inspect

    from xrd_pipeline import run_pipeline

    sig = inspect.signature(run_pipeline)
    assert "config" in sig.parameters
    param = sig.parameters["config"]
    assert param.default is None

    pipeline_source = (SCRIPTS_DIR / "xrd_pipeline.py").read_text(encoding="utf-8")
    assert "config" in pipeline_source
    assert "config.bkg_chebyshev_terms" in pipeline_source


def test_sym_background_wiggle_detector():
    """Verify SYM_BACKGROUND_WIGGLE detection, noise-gate deadband, and recommendations."""
    from xrd_diagnose import diagnose_residuals

    two_theta = np.linspace(10.0, 70.0, 500)
    y_obs = np.full_like(two_theta, 100.0)
    y_calc = np.full_like(two_theta, 100.0)

    # 1. Smooth linear background + micro floating point noise (noise-gate test)
    from xrd_diagnose import SymptomCode, SymptomSeverity

    rng = np.random.default_rng(42)
    y_bkg_smooth = 100.0 + 0.5 * two_theta + rng.normal(0, 1e-12, size=len(two_theta))
    diag_smooth = diagnose_residuals(two_theta, y_obs, y_calc, y_bkg=y_bkg_smooth)
    symptom_codes_smooth = [s.code for s in diag_smooth.symptoms]
    assert SymptomCode.BACKGROUND_WIGGLE not in symptom_codes_smooth

    # 2. Severely oscillating polynomial background (6+ inflections)
    y_bkg_wiggle = 100.0 + 20.0 * np.sin(0.5 * two_theta)
    diag_wiggle = diagnose_residuals(two_theta, y_obs, y_calc, y_bkg=y_bkg_wiggle)
    symptoms_wiggle = [s for s in diag_wiggle.symptoms if s.code == SymptomCode.BACKGROUND_WIGGLE]
    assert len(symptoms_wiggle) == 1
    assert symptoms_wiggle[0].severity == SymptomSeverity.WARNING
    assert symptoms_wiggle[0].inflections is not None
    assert symptoms_wiggle[0].inflections > 4

    assert any("bkg-terms 3" in a for a in diag_wiggle.recommended_actions)


def test_native_project_qpa_ssot():
    """Verify get_project_qpa Hill-Howard fallback returns correct types and sums ~100%.

    Uses SimpleNamespace mock gpx with 2 phases. No GSAS-II installation required.
    Covers SSOT canonical QPA path added in Plan v2.0 (Step 9).
    """
    import types

    from gsas2_adapter import get_project_qpa

    # Build minimal mock phases matching GSAS-II internal dictionary layout
    def _make_phase(name, sgz, mass, volume, scale):
        gen = {
            "SGData": {"SGZ": sgz},
            "Mass": mass,
            "Cell": [0, 5.0, 5.0, 5.0, 90.0, 90.0, 90.0, volume],  # [unused, a,b,c,α,β,γ, V]
        }
        hists = {"PWDR test.csv": {"Scale": [scale, True]}}
        phase = types.SimpleNamespace(
            name=name,
            data={"General": gen, "Histograms": hists},
        )
        return phase

    p1 = _make_phase("PhaseA", sgz=4, mass=160.0, volume=125.0, scale=3.0)
    p2 = _make_phase("PhaseB", sgz=2, mass=80.0, volume=64.0, scale=5.0)

    hist = types.SimpleNamespace(
        name="PWDR test.csv",
        data={"Histograms": {}},
    )

    gpx = types.SimpleNamespace(
        phases=lambda: [p1, p2],
        histograms=lambda: [hist],
    )

    result = get_project_qpa(gpx, hist=None)

    assert isinstance(result, dict), "get_project_qpa must return a dict"
    for phase_name, (wt, esd) in result.items():
        assert isinstance(phase_name, str)
        assert isinstance(wt, float)
        # esd отсутствует, если в проекте нет ковариации масштабов: это None,
        # а не 0.0, иначе неопределённость печаталась бы как точное число
        assert esd is None or isinstance(esd, float)
        assert esd is None or esd >= 0.0
        assert 0.0 <= wt <= 100.0, f"Weight fraction out of range: {wt}"
    total = sum(v[0] for v in result.values())
    assert abs(total - 100.0) < 0.5, f"QPA fractions must sum to ~100%, got {total:.2f}"


def test_build_stage_recipe_ssot():
    """Verify build_stage_recipe returns correct keys for all 8 canonical stage names.

    Pure unit test — no GSAS-II required. Covers SSOT canonical recipe generator
    added in Plan v2.0 (Step 10).
    """
    from stage_gate_rietveld import build_stage_recipe

    stage_checks = [
        ("1_Scale_and_Background", {"Background", "Scale"}),
        ("2_Zero_Shift", {"Sample Parameters"}),
        ("3_Unit_Cell", {"Cell"}),
        ("4_Atomic_Coordinates", {"Atoms"}),
        ("5_Overall_Displacement", {"Atoms"}),
        ("6_Group_B_iso", {"Atoms"}),
        ("7_Sample_Profile_XY", {"Size"}),
        ("8_Preferred_Orientation", {"Pref.Ori."}),
    ]

    for stage_name, expected_keys in stage_checks:
        recipe = build_stage_recipe(stage_name)
        assert isinstance(recipe, dict), f"Stage '{stage_name}': recipe must be dict"
        for key in expected_keys:
            assert key in recipe, (
                f"Stage '{stage_name}': expected key '{key}' missing. Got: {set(recipe)}"
            )

    # Stage 7 with refine_axial=True must include SH/L
    recipe7_axial = build_stage_recipe("7_Sample_Profile_XY", refine_axial=True)
    assert "Instrument Parameters" in recipe7_axial
    assert "SH/L" in recipe7_axial["Instrument Parameters"]

    # Stage 8 with march_dollase_direction must include direction key
    recipe8_md = build_stage_recipe("8_Preferred_Orientation", march_dollase_direction=[0, 0, 1])
    assert "direction" in recipe8_md
    assert recipe8_md["direction"] == [0, 0, 1]

    # Unknown stage must raise ValueError
    with pytest.raises(ValueError, match="Unknown Rietveld refinement stage"):
        build_stage_recipe("99_Nonexistent_Stage")


def test_crystallochemical_gatekeeper_rejection_audit():
    """Verify that structure with severe strain (GII > 0.25 v.u.) triggers REJECTED_BY_GATEKEEPER."""
    from gatekeeper import CrystallochemicalGatekeeper
    from pymatgen.core import Lattice, Structure

    # Moderately compress lattice parameters to induce severe bond-valence strain
    # Nominal Rutile TiO2 has standard a=4.59, c=2.96. Compression to a=4.10, c=2.70 produces severe overbonding GII > 0.25 v.u.
    strained_rutile = Structure(
        Lattice.from_parameters(4.10, 4.10, 2.70, 90.0, 90.0, 90.0),
        ["Ti", "Ti", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.305, 0.305, 0.0],
            [0.695, 0.695, 0.0],
            [0.805, 0.195, 0.5],
            [0.195, 0.805, 0.5],
        ],
    )
    gk = CrystallochemicalGatekeeper()
    audit = gk.audit_structure(strained_rutile)
    assert audit.gii_value is not None
    assert audit.gii_value > 0.25
    assert audit.gii_status == "REJECT"
    assert audit.is_valid is False


def test_strict_domain_neutrality_in_codebase():
    """Verify Rule 8: zero chemical element branching or hardcoded system tuples in skill scripts."""
    import re

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    py_files = list(scripts_dir.glob("*.py"))
    assert len(py_files) > 0, "No script files found to audit."

    forbidden_patterns = [
        re.compile(r'in\s*\(\s*["\'](?:Sm|Eu|Yb|Ce)["\']', re.IGNORECASE),
        re.compile(r'in\s*\(\s*["\'](?:S|Se|Te)["\']', re.IGNORECASE),
        re.compile(r"\bis_mixed_valence\b"),
    ]

    violations = []
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pat in forbidden_patterns:
                if pat.search(line):
                    violations.append(f"{py_file.name}:{lineno}: {line.strip()}")

    assert not violations, "Domain neutrality violations detected (Rule 8):\n" + "\n".join(
        violations
    )


def test_expected_r_factor_calculation():
    """Verify mathematical calculation of R_exp and R_wp / R_exp ratio."""
    from gsas2_adapter import ProfileCurves, calculate_profile_metrics

    two_theta = np.linspace(20.0, 60.0, 100)
    y_obs = np.full(100, 100.0)
    y_calc = np.full(100, 95.0)
    sigma_obs = np.sqrt(y_obs)

    curves = ProfileCurves(
        two_theta=two_theta,
        y_obs=y_obs,
        y_calc=y_calc,
        y_bkg=np.zeros(100),
        y_diff=y_obs - y_calc,
        sigma_obs=sigma_obs,
        weights=1.0 / (sigma_obs**2),
    )

    metrics = calculate_profile_metrics(curves, n_free_parameters=10)
    assert metrics.r_exp_percent > 0.0
    assert np.isclose(metrics.r_exp_percent, np.sqrt(90.0 / 10000.0) * 100.0, atol=1e-2)
    assert metrics.r_wp_over_r_exp_ratio > 0.0
    assert np.isclose(
        metrics.r_wp_over_r_exp_ratio, metrics.r_wp_percent / metrics.r_exp_percent, atol=1e-2
    )
    assert np.isclose(metrics.gof, metrics.r_wp_percent / metrics.r_exp_percent, atol=1e-2)
    assert np.isclose(
        metrics.reduced_chi2, (metrics.r_wp_percent / metrics.r_exp_percent) ** 2, atol=1e-2
    )


# =========================================================================
# K-alpha2 stripping (Rachinger / Ladell recursion)
# =========================================================================
def test_tier7_ka2_stripping_removes_companion_keeps_main_line():
    """The Ka2 companion must vanish while the Ka1 line survives.

    The recursion is A(x) = O(x) - r * A(g^-1(x)) with g^-1(x) < x, so it runs
    from low to high angle. Reversing the direction leaves the companion in
    place: the stripped pattern would still contain a Ka2 satellite, which against
    a large reference card pool always finds a spurious match.
    Ka1 and Ka2 are only ~0.075 deg apart at 2theta = 28 deg and therefore merge
    into one maximum at ordinary lab widths; they are genuinely resolvable only at
    high angle (~0.27 deg at 80 deg). The test therefore uses a high-angle line
    with narrow peaks, where the companion is a separate maximum.
    """
    inst = CanonicalInstrument()
    two_theta = np.linspace(70.0, 90.0, 4001)  # step 0.005 deg

    tt1 = 80.0
    arg = np.sin(np.radians(tt1 / 2.0)) * inst.wavelength_ka2 / inst.wavelength_ka1
    tt2 = float(np.degrees(2.0 * np.arcsin(arg)))
    assert tt2 - tt1 > 0.20  # the Ka2 partner is resolved at this angle

    y = 1000.0 * np.exp(-0.5 * ((two_theta - tt1) / 0.03) ** 2)
    y += 1000.0 * inst.ka2_ratio * np.exp(-0.5 * ((two_theta - tt2) / 0.03) ** 2)

    stripped = inst.strip_ka2(two_theta, y)

    ka1_peak = float(np.max(stripped[(two_theta > tt1 - 0.08) & (two_theta < tt1 + 0.08)]))
    ka2_peak = float(np.max(stripped[(two_theta > tt2 - 0.08) & (two_theta < tt2 + 0.08)]))

    assert ka1_peak > 800.0, f"Ka1 line lost: {ka1_peak}"
    assert ka2_peak < 0.25 * ka1_peak, f"Ka2 companion survived: {ka2_peak}"


def test_tier7_ka2_stripping_preserves_the_ka1_line_position():
    """Stripping must not shift the Ka1 line, only remove its companion."""
    inst = CanonicalInstrument()
    two_theta = np.linspace(70.0, 90.0, 4001)
    tt1 = 80.0
    arg = np.sin(np.radians(tt1 / 2.0)) * inst.wavelength_ka2 / inst.wavelength_ka1
    tt2 = float(np.degrees(2.0 * np.arcsin(arg)))

    y = 1000.0 * np.exp(-0.5 * ((two_theta - tt1) / 0.03) ** 2)
    y += 1000.0 * inst.ka2_ratio * np.exp(-0.5 * ((two_theta - tt2) / 0.03) ** 2)

    stripped = inst.strip_ka2(two_theta, y)
    pos_before = float(two_theta[int(np.argmax(y))])
    pos_after = float(two_theta[int(np.argmax(stripped))])

    assert abs(pos_after - pos_before) < 0.02, f"Ka1 line shifted: {pos_before} -> {pos_after}"


def test_tier7_ka2_stripping_preserves_input_array():
    """Unsorted input must be handled; the caller's arrays must not be mutated."""
    inst = CanonicalInstrument()
    rng = np.random.default_rng(5)
    two_theta = np.linspace(20.0, 60.0, 501)
    intensity = 100.0 + rng.normal(0.0, 1.0, size=two_theta.size)
    x_before, y_before = two_theta.copy(), intensity.copy()

    out = inst.strip_ka2(two_theta, intensity)

    assert np.array_equal(two_theta, x_before)
    assert np.array_equal(intensity, y_before)
    assert out.shape == intensity.shape
    # Unsorted variant must give the same values at the same angles.
    perm = rng.permutation(two_theta.size)
    out_shuffled = inst.strip_ka2(two_theta[perm], intensity[perm])
    restore = np.empty_like(out_shuffled)
    restore[perm] = out_shuffled
    assert np.allclose(restore, out, atol=1e-9)


# =========================================================================
# TIER 9: Native GSASIIscriptable API & Universal Restraints
# =========================================================================
def test_tier9_native_residuals_extraction():
    """Verify exact extraction of native agreement indices from hist.residuals."""
    from gsas2_adapter import ensure_fresh_residuals, get_native_histogram_residuals

    class MockHist:
        def __init__(self):
            self.residuals = {
                "wR": 4.52,
                "R": 3.21,
                "wRmin": 1.85,
                "chi2": 5.98,
                "GOF": 2.445,
                "Rbkg": 2.10,
                "wRbkg": 3.40,
            }

    hist = MockHist()
    res = get_native_histogram_residuals(hist)
    assert np.isclose(res["wR"], 4.52)
    assert np.isclose(res["R"], 3.21)
    assert np.isclose(res["wRmin"], 1.85)
    assert np.isclose(res["chi2"], 5.98)
    assert np.isclose(res["GOF"], 2.445)
    assert np.isclose(res["Rbkg"], 2.10)
    assert np.isclose(res["wRbkg"], 3.40)

    # Verify ensure_fresh_residuals returns existing residuals without triggering 0-cycle refinement
    class MockProject:
        def __init__(self, hist):
            self._hist = hist
            self.refinements_called = False

        def histograms(self):
            return [self._hist]

        def do_refinements(self, _ref_list):
            self.refinements_called = True

    gpx = MockProject(hist)
    fresh = ensure_fresh_residuals(gpx, hist)
    assert np.isclose(fresh["wR"], 4.52)
    assert gpx.refinements_called is False

    # Verify 0-cycle fallback when residuals are empty
    empty_hist = MockHist()
    empty_hist.residuals = {}

    class MockProjectZeroCycle:
        def __init__(self, hist):
            self._hist = hist
            self.cycles_set = None

        def histograms(self):
            return [self._hist]

        def get_Controls(self, _name):
            return 3

        def set_Controls(self, _name, val):
            self.cycles_set = val

        def do_refinements(self, _ref_list):
            # 0-cycle evaluation simulates populating hist.residuals
            self._hist.residuals = {"wR": 5.12, "wRmin": 2.01}

    gpx_zero = MockProjectZeroCycle(empty_hist)
    populated = ensure_fresh_residuals(gpx_zero, empty_hist)
    assert gpx_zero.cycles_set == 3  # restored
    assert np.isclose(populated["wR"], 5.12)


def test_tier9_native_compute_mass_fracs():
    """Verify get_project_qpa consumes native ComputeMassFracs() and converts fractions to wt.%."""
    import types

    from gsas2_adapter import get_project_qpa

    class MockHistWithQPA:
        def ComputeMassFracs(self):
            return {
                "SmS": (0.865, 0.004),
                "Sm2O2S": (0.135, 0.003),
            }

    hist = MockHistWithQPA()
    gpx = types.SimpleNamespace(
        phases=lambda: [types.SimpleNamespace(name="SmS"), types.SimpleNamespace(name="Sm2O2S")],
        histograms=lambda: [hist],
    )
    mass_fracs = get_project_qpa(gpx, hist)
    assert mass_fracs["SmS"] == (86.5, 0.4)
    assert mass_fracs["Sm2O2S"] == (13.5, 0.3)


def test_tier9_native_atom_record_properties():
    """Verify G2AtomRecord typed facade over raw list and typed object representations."""
    from stage_gate_rietveld import G2AtomRecord

    # Каноническая структура списка атома GSAS-II (длина 18: индекс 9 - adp_flag 'I', индекс 10 - Uiso)
    raw_atom = [
        "Sm1",
        "Sm",
        "X",
        0.125,
        0.250,
        0.375,
        4.0,
        1.0,
        "P",
        "I",
        0.008,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    rec = G2AtomRecord(raw_atom)

    assert rec.label == "Sm1"
    assert rec.element == "Sm"
    assert rec.refinement_flag == "X"
    assert rec.coordinates == (0.125, 0.250, 0.375)
    assert rec.mult == 4.0
    assert rec.occupancy == 1.0
    assert np.isclose(rec.uiso, 0.008)

    # Mutating via typed setters
    rec.coordinates = (0.200, 0.300, 0.400)
    assert rec.coordinates == (0.200, 0.300, 0.400)
    assert raw_atom[3:6] == [0.200, 0.300, 0.400]

    rec.uiso = 0.015
    assert np.isclose(rec.uiso, 0.015)
    assert np.isclose(raw_atom[10], 0.015)

    rec.occupancy = 0.95
    assert np.isclose(rec.occupancy, 0.95)
    assert np.isclose(raw_atom[7], 0.95)

    rec.label = "Sm_new"
    assert rec.label == "Sm_new"
    assert raw_atom[0] == "Sm_new"


def test_tier9_universal_dist_restraints_builder():
    """Verify dynamic discovery and assignment of 1st coordination sphere geometric restraints."""
    from stage_gate_rietveld import setup_phase_dist_restraints

    ceo2_struct = Structure.from_file(str(CIF_DIR / "CeO2.cif"))

    class MockG2Phase:
        def __init__(self, name="CeO2"):
            self.name = name
            self.cleared = False
            self.restraints = []
            self.weight = 1.0

            # Mock atoms in asymmetric unit / phase
            class MockAtom:
                def __init__(self, label, elem):
                    self.label = label
                    self.element = elem

            self._atoms = [MockAtom("Ce1", "Ce"), MockAtom("O1", "O")]

        def atoms(self):
            return self._atoms

        def clearDistRestraint(self):
            self.cleared = True
            self.restraints.clear()

        def addDistRestraint(self, origin, target, bond, factor=1.15, ESD=0.02):
            self.restraints.append(
                {
                    "origin": [a.label for a in origin],
                    "target": [a.label for a in target],
                    "bond": bond,
                    "factor": factor,
                    "esd": ESD,
                }
            )
            return 1

        def setDistRestraintWeight(self, weight):
            self.weight = weight

    mock_gpx = type("MockGPX", (), {"data": {}})()
    mock_phase = MockG2Phase()

    count = setup_phase_dist_restraints(
        mock_gpx,
        mock_phase,
        structure=ceo2_struct,
        factor=1.15,
        esd=0.02,
        weight=250.0,
    )

    assert count > 0
    assert mock_phase.cleared is True
    assert mock_phase.weight == 250.0
    assert "Restraints" in mock_gpx.data
    assert "CeO2" in mock_gpx.data["Restraints"]["data"]

    # Verify bond distance matches real Ce-O distance in fluorite (~2.343 A)
    assert len(mock_phase.restraints) == 1
    r = mock_phase.restraints[0]
    assert r["origin"] == ["Ce1"]
    assert r["target"] == ["O1"]
    assert np.isclose(r["bond"], 2.343, atol=0.01)
    assert r["factor"] == 1.15
    assert r["esd"] == 0.02


def test_tier9_native_last_fit_results_telemetry():
    """Verify shift_significance_problems and summarize_rule9 evaluate get_LastFitResults() shifts."""
    from stage_gate_rietveld import shift_significance_problems
    from xrd_project import summarize_rule9

    shifts = {
        "0::Acell:0": 0.0001,
        "0::Acell:1": 0.0002,
        "0::AUiso:*": 0.0010,
        ":0:Zero": 0.00005,
    }
    sigmas = {
        "0::Acell:0": 0.0010,  # ratio = 0.10 < 0.15 (PASS)
        "0::Acell:1": 0.0008,  # ratio = 0.25 >= 0.15 (CONVERGING)
        "0::AUiso:*": 0.0002,  # ratio = 5.00 >= 0.15 (CONVERGING)
        ":0:Zero": 0.00050,  # ratio = 0.10 < 0.15 (PASS)
    }

    problems = shift_significance_problems(shifts, sigmas, threshold=0.15)
    assert len(problems) == 2
    assert any("0::Acell:1" in p for p in problems)
    assert any("0::AUiso:*" in p for p in problems)

    total, converged, converging = summarize_rule9(shifts, sigmas)
    assert (total, converged, converging) == (4, 2, 2)


def test_tier9_apply_box_constraints_naming():
    """Verify GSAS-II box constraints use the documented set_Controls keyword convention.

    GSAS-II expects ``set_Controls(control, value=<float>, variable='<name>')``.
    Reversing the last two arguments makes GSAS-II raise
    ``TypeError: argument of type 'float' is not iterable`` and the limit is
    silently dropped, so this test pins the call signature explicitly.
    """
    from stage_gate_rietveld import apply_box_constraints

    class MockPhase:
        def atoms(self):
            return ["at0", "at1"]

    class MockG2ProjectWithControls:
        def __init__(self):
            self.controls: dict[str, dict[str, float]] = {"parmMin": {}, "parmMax": {}}

        def phases(self):
            return [MockPhase()]

        def set_Controls(self, control, value, variable=None):
            assert variable is not None, "GSAS-II requires variable= for parmMin/parmMax"
            assert isinstance(value, float), "GSAS-II requires a float value= for parmMin/parmMax"
            self.controls[control][variable] = value

    gpx = MockG2ProjectWithControls()
    apply_box_constraints(gpx, phase_id=0, hist_id=0)

    mins = gpx.controls["parmMin"]
    maxs = gpx.controls["parmMax"]

    # Verify Uiso thermal parameter boundaries
    assert mins["0::AUiso:0"] == 0.001
    assert maxs["0::AUiso:0"] == 0.15
    assert mins["0::AUiso:1"] == 0.001
    assert maxs["0::AUiso:1"] == 0.15

    # Verify March-Dollase texture boundaries
    assert mins["0:0:MD"] == 0.4
    assert maxs["0:0:MD"] == 2.5

    # Verify histogram sample displacement boundaries
    assert mins[":0:DisplaceX"] == -1000.0
    assert maxs[":0:DisplaceX"] == 1000.0

    # Verify 2theta zero shift boundaries. The mask must be histogram-scoped;
    # the phase-less '::Zero' form does not address the refined parameter.
    assert mins[":0:Zero"] == -0.5
    assert maxs[":0:Zero"] == 0.5

    # Verify axial divergence boundaries: SH/L must stay physical, otherwise the
    # optimizer sharpens every peak and fabricates a low R_wp.
    assert mins[":0:SH/L"] == 0.0
    assert maxs[":0:SH/L"] == 0.05

    # Verify lattice metric-tensor bounds
    assert mins["0::A0"] == 0.0001
    assert maxs["0::A0"] == 1.0


def test_extract_cell_parameters_with_esd_and_dialect_mapping():
    """Verify that extract_cell_parameters handles multi-dialect dicts and extracts ESDs."""
    from gsas2_adapter import extract_cell_parameters

    class MockPhaseWithEsd:
        name = "SmS"

        def get_cell_and_esd(self):
            cell_dict = {
                "length_a": 5.9727,
                "length_b": 5.9727,
                "length_c": 5.9727,
                "angle_alpha": 90.0,
                "angle_beta": 90.0,
                "angle_gamma": 90.0,
                "volume": 213.06,
            }
            esd_dict = {
                "length_a": 0.0004,
                "length_b": 0.0004,
                "length_c": 0.0004,
                "volume": 0.04,
            }
            return cell_dict, esd_dict

    class MockProject:
        def phases(self):
            return [MockPhaseWithEsd()]

    res = extract_cell_parameters(MockProject())
    assert "SmS" in res
    cell = res["SmS"]
    assert cell["a"] == 5.9727
    assert cell["a_esd"] == 0.0004
    assert cell["volume"] == 213.06
    assert cell["volume_esd"] == 0.04


def test_export_phase_cif_safe_native_api(tmp_path):
    """Verify that export_phase_cif_safe calls native phase.export_CIF."""
    from gsas2_adapter import export_phase_cif_safe

    called = []

    class MockPhaseWithExportCIF:
        def export_CIF(self, filename, *_args, **_kwargs):
            called.append(filename)
            Path(filename).write_text("data_test\n_cell_length_a 5.97\n", encoding="utf-8")

    out_cif = tmp_path / "test_phase.cif"
    res_path = export_phase_cif_safe(MockPhaseWithExportCIF(), out_cif)
    assert res_path == out_cif
    assert out_cif.exists()
    assert len(called) == 1
    assert "data_test" in out_cif.read_text(encoding="utf-8")


def _live_mp_or_skip() -> None:
    """Skip only when this machine declares no retrieval environment.

    That is a genuine absence, like an absent PDF-2 installation: the Materials Project
    route is optional and lives in a separate interpreter, so a machine without one cannot
    exercise it on live data. What must not happen is the earlier arrangement, where the
    skip hid an obstacle of our own making -- a client that could not be imported under
    the declared NumPy pin. That obstacle is gone: this interpreter imports the client
    nowhere, and the client's own work is covered by the tests that replace the seam.
    """
    from cif_retriever import resolve_mp_interpreter

    if resolve_mp_interpreter() is None:
        pytest.skip(
            "no retrieval environment declared: set MP_PYTHON to a python executable with "
            "mp-api installed to exercise the live Materials Project route"
        )


def test_materials_project_client_and_query(tmp_path, monkeypatch):
    """Verify MaterialsProjectClient search, conventional cell export, search_cifs and download_cifs_by_ids."""
    from cif_retriever import (
        MaterialsProjectClient,
        download_cifs_by_ids,
        format_search_table,
        search_cifs,
        validate_cif_identity,
    )
    from pymatgen.core import Lattice
    from pymatgen.io.cif import CifWriter

    cache_dir = tmp_path / "mp_cache"

    # 1. Unconfigured client behavior. The client reads a key from the environment or
    # from a .env file above the working directory, so "unconfigured" has to mean that
    # neither address carries one. A real .env on this machine would otherwise make the
    # assertion below read as a defect of the client rather than as a stray configuration.
    monkeypatch.setattr("env_file.find_dotenv", lambda **_kwargs: "")
    monkeypatch.delenv("MP_API_KEY", raising=False)
    unconf_client = MaterialsProjectClient(api_key=None, cache_dir=cache_dir)
    assert unconf_client.is_configured() is False
    assert unconf_client.search(elements=["Sm", "S"]) == []
    assert unconf_client.download_cif("mp-1077") is None

    # Verify auto-configuration via MP_API_KEY environment variable (from .env)
    monkeypatch.setenv("MP_API_KEY", "env_loaded_key_abc")
    env_client = MaterialsProjectClient(cache_dir=cache_dir)
    assert env_client.is_configured() is True
    assert env_client.api_key == "env_loaded_key_abc"

    # 2. Configured client, with the retrieval environment replaced by a responder.
    #
    # What this interpreter still owns is the cache, the file naming and the reading of
    # records; the network belongs to the other environment and is reached through a single
    # seam, cif_retriever.run_mp_query. Replacing that seam tests the client, whereas
    # mocking the client's internals would test only the mock.
    client = MaterialsProjectClient(api_key="mock_test_key_12345", cache_dir=cache_dir)
    assert client.is_configured() is True

    sanitized_cif = str(
        CifWriter(Structure(Lattice.cubic(5.97), ["Sm", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]]))
    )
    records = [
        {
            "material_id": "mp-1077",
            "formula_pretty": "SmS",
            "symmetry": {"symbol": "Fm-3m"},
            "volume": 212.5,
            "energy_above_hull": 0.0,
            "is_stable": True,
            "structure": {
                "lattice": {
                    "a": 5.97,
                    "b": 5.97,
                    "c": 5.97,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                }
            },
        },
        {
            "material_id": "mp-9999",
            "formula_pretty": "Sm2S3",
            "symmetry": {"symbol": "Fm-3m"},
            "volume": 300.0,
            "energy_above_hull": 0.05,
            "is_stable": False,
            "structure": None,
        },
    ]
    sent_requests: list[dict] = []

    def fake_query(request: dict) -> dict:
        sent_requests.append(request)
        if request["action"] == "search":
            return {"records": records}
        return {"cif_text": sanitized_cif, "reduced_formula": "SmS"}

    monkeypatch.setattr("cif_retriever.run_mp_query", fake_query)

    # Test search
    search_res = client.search(elements=["Sm", "S"])
    assert len(search_res) == 2
    assert search_res[0].id == "mp-1077"
    assert search_res[0].formula == "SmS"
    assert search_res[0].is_stable is True
    assert search_res[0].a == 5.97
    assert search_res[0].b == 5.97
    assert search_res[0].c == 5.97
    assert search_res[0].alpha == 90.0
    assert search_res[0].beta == 90.0
    assert search_res[0].gamma == 90.0

    # The request that reached the other environment carries the resolved query and the
    # key: that is the contract between the two processes, and nothing else crosses it.
    assert sent_requests[0]["action"] == "search"
    assert sent_requests[0]["mode"] == "all"
    assert sent_requests[0]["elements"] == ["S", "Sm"]
    assert sent_requests[0]["api_key"] == "mock_test_key_12345"

    # Test download_cif: the text produced in the other environment must land in this
    # client's cache, under a name taken from the formula, and read back as the phase it
    # claims to be. Removing the fractional oxidation states belongs to the bridge, where
    # the structure object exists, and is pinned by the live run below rather than here.
    out_cif = tmp_path / "cifs" / "SmS_mp-1077.cif"
    out_cif.parent.mkdir(parents=True, exist_ok=True)
    downloaded = client.download_cif("mp-1077", out_path=out_cif, formula_hint="SmS")
    assert downloaded is not None
    assert downloaded.exists()

    cif_text = downloaded.read_text(encoding="utf-8")
    assert cif_text == sanitized_cif, "the received CIF must be written verbatim"
    assert (
        validate_cif_identity(downloaded, expected_elements=["Sm", "S"], expected_formula="SmS")
        is True
    )
    assert sent_requests[-1]["action"] == "download"
    assert sent_requests[-1]["material_id"] == "mp-1077"
    assert sent_requests[-1]["symprec"] == 0.01

    # Test search_cifs with limit reporting total_count
    search_data = search_cifs(elements=["Sm", "S"], source="mp", limit=1, mp_client=client)
    assert search_data.total_count == 2
    assert search_data.displayed_count == 1
    assert search_data.limit == 1
    assert len(search_data.results) == 1
    assert search_data.results[0].id == "mp-1077"
    assert search_data.results[0].a == 5.97

    # Test format_search_table
    table = format_search_table(search_data)
    assert "Found 2 total structures" in table
    assert "showing first 1" in table
    assert "limit: 1" in table
    assert "mp-1077" in table
    assert "5.970" in table
    assert "90.0" in table
    assert "alpha" in table

    # Test download_cifs_by_ids
    out_dir = tmp_path / "query_cifs"
    paths = download_cifs_by_ids(
        ids=["mp-1077"],
        out_dir=out_dir,
        mp_client=client,
    )
    assert len(paths) == 1
    assert paths[0].name == "SmS_mp-1077.cif"
    assert paths[0].exists()


def test_cif_retriever_cli_subcommands_and_legacy_rejection(monkeypatch, tmp_path):
    """Verify strict CLI subcommands (search, download) and rejection of legacy options."""
    import sys

    import pytest
    from cif_retriever import main

    # 1. Rejection of deprecated flags
    for bad_flag in [
        "--formula",
        "--max-cifs",
        "--cod-id",
        "--mp-id",
        "--strict-max",
        "--mp-api-key",
    ]:
        monkeypatch.setattr(sys, "argv", ["cif_retriever.py", bad_flag, "value"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0

    # 2. Running CLI without subcommand raises error
    monkeypatch.setattr(sys, "argv", ["cif_retriever.py"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0

    # 3. CLI search subcommand execution with mock COD (using --all, --none, --strict)
    def mock_urlopen(req, *args, **_):
        class MockResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'[{"file": "1538008", "chemname": "Sm S", "a": "5.97", "b": "5.97", "c": "5.97", "alpha": "90", "beta": "90", "gamma": "90", "sg": "F m -3 m"}]'

        return MockResp()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cif_retriever.py",
            "search",
            "--all",
            "Sm,S",
            "--none",
            "C,H",
            "--strict",
            "--source",
            "cod",
        ],
    )
    ret = main()
    assert ret == 0

    # 4. CLI search with --any and --json
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cif_retriever.py",
            "search",
            "--any",
            "Sm,Fe",
            "--source",
            "cod",
            "--limit",
            "1",
            "--json",
        ],
    )
    ret_json = main()
    assert ret_json == 0

    # 5. CLI search rejection of contradictory filter (--all Sm,S --none S)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cif_retriever.py", "search", "--all", "Sm,S", "--none", "S", "--source", "cod"],
    )
    ret_err = main()
    assert ret_err != 0

    # 6. CLI search with --refresh flag
    monkeypatch.setattr(
        sys,
        "argv",
        ["cif_retriever.py", "search", "--all", "Sm,S", "--source", "cod", "--refresh"],
    )
    ret_refresh = main()
    assert ret_refresh == 0


def test_element_filter_logic_and_edge_cases():
    """Verify ElementFilter boolean operations (AND, OR, NOT, STRICT) and edge cases."""
    import pytest
    from cif_retriever import ElementFilter, extract_elements

    # 1. All of (AND)
    f_and = ElementFilter.from_raw(all_elements="Pb,S")
    assert f_and.matches({"Pb", "S"}) is True
    assert f_and.matches({"Pb", "S", "O"}) is True  # PbSO4 in open system
    assert f_and.matches({"Pb", "O"}) is False
    assert f_and.matches({"Zn", "S"}) is False

    # 2. At least one of (OR)
    f_or = ElementFilter.from_raw(any_elements="Fe,Co,Ni")
    assert f_or.matches({"Fe", "O"}) is True
    assert f_or.matches({"Co", "S"}) is True
    assert f_or.matches({"Ti", "O"}) is False

    # 3. None of (NOT / NOR)
    f_not = ElementFilter.from_raw(all_elements="Pb,S", none_elements="C,H,Cl")
    assert f_not.matches({"Pb", "S"}) is True
    assert f_not.matches({"Pb", "S", "O"}) is True
    assert f_not.matches({"Pb", "S", "C"}) is False
    assert f_not.matches({"Pb", "S", "Cl"}) is False

    # 4. Strict mode (Closed system)
    f_strict = ElementFilter.from_raw(all_elements="Pb,S", strict=True)
    assert f_strict.matches({"Pb", "S"}) is True
    assert f_strict.matches({"Pb", "S", "O"}) is False  # PbSO4 rejected in strict closed system
    assert f_strict.matches({"Pb", "S", "Cl"}) is False

    # 5. Strict mode with both all_of and any_of
    f_strict_combo = ElementFilter.from_raw(all_elements="Sm", any_elements="S,Se", strict=True)
    assert f_strict_combo.matches({"Sm", "S"}) is True
    assert f_strict_combo.matches({"Sm", "Se"}) is True
    assert f_strict_combo.matches({"Sm", "S", "Se"}) is True
    assert f_strict_combo.matches({"Sm"}) is False  # Missing any_of
    assert f_strict_combo.matches({"Sm", "S", "O"}) is False  # Contains foreign O

    # 6. Edge cases: Contradiction
    with pytest.raises(ValueError, match="Contradictory filter"):
        ElementFilter.from_raw(all_elements="Pb,S", none_elements="S")

    # 7. Edge cases: Empty filter
    with pytest.raises(ValueError, match="At least one positive inclusion"):
        ElementFilter.from_raw()

    # 8. Edge cases: Unknown IUPAC symbol
    with pytest.raises(ValueError, match="Unknown chemical element 'X'"):
        ElementFilter.from_raw(all_elements="Pb,X")

    # 9. Edge cases: Redundant element in any_of when present in all_of
    f_red = ElementFilter.from_raw(all_elements="Pb,S", any_elements="S,O")
    assert f_red.all_of == frozenset({"Pb", "S"})
    assert f_red.any_of == frozenset({"O"})

    # 10. Heavy hydrogen normalization (D -> H)
    f_d = ElementFilter.from_raw(all_elements="Pb,S", none_elements="D")
    assert f_d.none_of == frozenset({"H"})

    # 11. extract_elements on complex/disordered formulas
    assert extract_elements("Pb5 O4 (S O4)") == {"Pb", "S", "O"}
    assert extract_elements("SmS1.76") == {"Sm", "S"}
    assert extract_elements("(Fe0.8Ni0.2)2O3") == {"Fe", "Ni", "O"}
    assert extract_elements("D2 O") == {"H", "O"}
    assert extract_elements("[Cu(NH3)4]SO4·H2O") == {"Cu", "S", "H", "O", "N"}


def test_search_query_caching_cod_and_mp(tmp_path, monkeypatch):
    """Verify raw query caching for COD and MP, cache HIT, --refresh bypass, corrupted JSON fallback, and TTL."""
    from cif_retriever import (
        CodClient,
        MaterialsProjectClient,
        search_cifs,
    )

    cod_cache = tmp_path / "cod_cache"
    mp_cache = tmp_path / "mp_cache"

    # --- 1. Test COD Search Caching ---
    cod_call_count = 0

    def mock_cod_urlopen(req, *args, **_):
        nonlocal cod_call_count
        cod_call_count += 1

        class MockResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return (
                    b'[{"file": "1538008", "chemname": "Sm S", "a": "5.97", "b": "5.97", "c": "5.97", '
                    b'"alpha": "90", "beta": "90", "gamma": "90", "sg": "F m -3 m", "vol": "212.5"}]'
                )

        return MockResp()

    monkeypatch.setattr("urllib.request.urlopen", mock_cod_urlopen)
    monkeypatch.delenv("COD_HSRDB_PATH", raising=False)
    monkeypatch.setattr("env_file.load_env_file", lambda: None)
    monkeypatch.setattr("hs_rdb.load_env_file", lambda: None)

    cod_client = CodClient(cache_dir=cod_cache, cache_ttl=3600)

    # First call: cache MISS (network call)
    res1 = cod_client.search(elements=["Sm", "S"])
    assert len(res1) == 1
    assert res1[0].a == 5.97
    assert cod_call_count == 1
    assert cod_client.last_cache_info.cached is False
    assert (cod_cache / "searches" / "cod_query_S_Sm.json").exists()

    # Second call: cache HIT (no network call)
    res2 = cod_client.search(elements=["Sm", "S"])
    assert len(res2) == 1
    assert cod_call_count == 1  # Did NOT increment!
    assert cod_client.last_cache_info.cached is True
    assert cod_client.last_cache_info.age == "just now"

    # Third call with refresh=True: cache BYPASS (network call)
    res3 = cod_client.search(elements=["Sm", "S"], refresh=True)
    assert len(res3) == 1
    assert cod_call_count == 2  # Incremented!
    assert cod_client.last_cache_info.cached is False

    # Fourth call with corrupted cache file: graceful fallback to network
    cache_file = cod_cache / "searches" / "cod_query_S_Sm.json"
    cache_file.write_text("{corrupted json", encoding="utf-8")
    res4 = cod_client.search(elements=["Sm", "S"])
    assert len(res4) == 1
    assert cod_call_count == 3  # Fallback called network!

    # --- 2. Test MP Search Caching ---
    # The client's cache policy is the subject here, so the retrieval environment is
    # replaced by a responder and its invocations are counted: a cache hit has to reach the
    # other interpreter zero times.
    mp_call_count = 0
    one_record = [
        {
            "material_id": "mp-1077",
            "formula_pretty": "SmS",
            "symmetry": {"symbol": "Fm-3m"},
            "volume": 212.5,
            "energy_above_hull": 0.0,
            "is_stable": True,
            "structure": {
                "lattice": {
                    "a": 5.97,
                    "b": 5.97,
                    "c": 5.97,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                }
            },
        }
    ]

    def fake_query(_request: dict) -> dict:
        nonlocal mp_call_count
        mp_call_count += 1
        return {"records": one_record}

    monkeypatch.setattr("cif_retriever.run_mp_query", fake_query)

    mp_client = MaterialsProjectClient(api_key="test_key", cache_dir=mp_cache, cache_ttl=3600)

    # First call: cache MISS
    mp_res1 = mp_client.search(elements=["Sm", "S"])
    assert len(mp_res1) == 1
    assert mp_call_count == 1
    assert mp_client.last_cache_info.cached is False
    assert (mp_cache / "searches" / "mp_all_S-Sm.json").exists()

    # Second call: cache HIT
    mp_res2 = mp_client.search(elements=["Sm", "S"])
    assert len(mp_res2) == 1
    assert mp_call_count == 1  # Did NOT increment!
    assert mp_client.last_cache_info.cached is True

    # Third call: refinement with different filter (e.g. none_elements) hits same raw cache!
    mp_search_data = search_cifs(
        elements=["Sm", "S"],
        source="mp",
        mp_client=mp_client,
    )
    assert mp_search_data.total_count == 1
    assert mp_search_data.cache_info["mp"].cached is True
    assert mp_call_count == 1  # Still 1!

    # Fourth call with refresh=True
    mp_res3 = mp_client.search(elements=["Sm", "S"], refresh=True)
    assert len(mp_res3) == 1
    assert mp_call_count == 2
    assert mp_client.last_cache_info.cached is False


def test_mp_bridge_reduces_a_live_document_to_the_record_the_client_reads():
    """The bridge's record shape is the contract between the two processes.

    ``mp_query`` is importable in this environment because it imports ``mp_api`` only inside
    its functions -- which is what makes the contract testable in the very interpreter that
    cannot import the client. The attributes below stand in for a live pydantic document;
    the reduction itself is real.
    """
    import json

    import mp_query

    def document(**fields: object) -> object:
        """Build a stand-in for a live document from its fields.

        Fields are passed by name rather than declared as class attributes, so a static
        analyzer sees them as arguments of a call instead of as assignments nothing reads:
        the bridge reads them through ``getattr``, which no analyzer can follow.
        """
        return type("_Document", (), fields)()

    live = document(
        material_id="mp-1077",
        formula_pretty="SmS",
        symmetry=type("_Symmetry", (), {"symbol": "Fm-3m"})(),
        energy_above_hull=0.0,
        is_stable=True,
        volume=212.5,
        structure=type(
            "_Structure",
            (),
            {
                "lattice": type(
                    "_Lattice",
                    (),
                    {"a": 5.97, "b": 5.97, "c": 5.97, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
                )()
            },
        )(),
    )

    record = mp_query.serialize_document(live)
    assert record["material_id"] == "mp-1077"
    assert record["symmetry"]["symbol"] == "Fm-3m"
    assert record["structure"]["lattice"]["a"] == 5.97
    assert record["structure"]["lattice"]["gamma"] == 90.0
    assert record["is_stable"] is True

    # A document without a structure or a space group must still produce the same keys,
    # because the cache is read field by field and a missing key would read as a defect
    # of the reader rather than as an absence in the record.
    bare = mp_query.serialize_document(
        document(
            material_id="mp-9",
            formula_pretty="S",
            symmetry=None,
            energy_above_hull=1.5,
            is_stable=False,
            volume=None,
            structure=None,
        )
    )
    assert bare["symmetry"]["symbol"] == ""
    assert bare["structure"] is None
    assert bare["volume"] is None
    assert bare["is_stable"] is False
    # Whatever crosses the process boundary has to be JSON, not an object of pymatgen.
    assert json.loads(json.dumps(bare)) == bare
    assert json.loads(json.dumps(record)) == record


def test_mp_route_names_its_obstacle_instead_of_failing_inside_a_third_party(monkeypatch, tmp_path):
    """An unaddressed retrieval environment must be reported by our own code.

    This is the check that replaces the former skip: the obstacle is named here, with the
    variable that would remove it, and the client continues to work from the local cache.
    """
    from cif_retriever import (
        MaterialsProjectClient,
        MaterialsProjectUnavailable,
        resolve_mp_interpreter,
        run_mp_query,
    )
    from loguru import logger as loguru_logger

    monkeypatch.setattr("env_file.find_dotenv", lambda **_kwargs: "")
    monkeypatch.delenv("MP_PYTHON", raising=False)
    assert resolve_mp_interpreter() is None
    with pytest.raises(MaterialsProjectUnavailable, match="MP_PYTHON"):
        run_mp_query({"action": "search"})

    # A variable naming a file that does not exist is configuration being wrong, and is
    # reported as that: passing it to subprocess would raise FileNotFoundError, which says
    # nothing about the address being the problem.
    monkeypatch.setenv("MP_PYTHON", str(tmp_path / "absent-python.exe"))
    assert resolve_mp_interpreter() is None

    monkeypatch.delenv("MP_PYTHON", raising=False)
    client = MaterialsProjectClient(api_key="unused-key", cache_dir=tmp_path / "mp_cache")
    messages: list[str] = []
    sink = loguru_logger.add(messages.append, level="WARNING")
    try:
        assert client.search(elements=["Sm", "S"]) == []
    finally:
        loguru_logger.remove(sink)
    assert any("MP_PYTHON" in message for message in messages), messages


@pytest.mark.network
def _require_mp_transport() -> None:
    """Проверить транспорт до Materials Project до запроса.

    Клиент возвращает пустой список и при отказе сети: без этой проверки
    ``assert results`` сообщает «нет записей» там, где записи никто не получил.
    """
    try:
        with socket.create_connection(("api.materialsproject.org", 443), timeout=10):
            return
    except OSError as exc:
        pytest.fail(
            f"транспорт: api.materialsproject.org недоступен ({exc}); это не ответ «нет записей»"
        )


@pytest.mark.network
def test_mp_live_query_through_the_retrieval_environment(tmp_path):
    """The live route end to end: request out, records back, through that interpreter.

    Skipped only when this machine declares no retrieval environment, which is a genuine
    absence. Where one is declared, this is the run that proves the split works and that
    the reduction of a live document -- impossible to exercise in this interpreter --
    yields records the client can read.
    """
    _live_mp_or_skip()

    from cif_retriever import MaterialsProjectClient

    client = MaterialsProjectClient(cache_dir=tmp_path / "mp_cache")
    if not client.is_configured():
        pytest.skip("no MP_API_KEY configured, so the live route cannot authenticate")

    _require_mp_transport()

    results = client.search(elements=["Sm", "S"], refresh=True)

    assert results, "the live query returned no records"
    assert results[0].id.startswith("mp-")
    assert results[0].formula
