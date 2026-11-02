"""Test suite for analytical candidate phase scoring engine (CandidateScoringEngine)."""

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from cif_retriever import ElementFilter, extract_elements
from pymatgen.core import Lattice, Structure
from search_match import (
    CandidateLines,
    CandidateScoringEngine,
    CandidateScoringResult,
    ProfileProbabilityField,
    apply_anisotropic_bragg_shift,
    apply_bragg_cell_shift,
    format_candidate_table,
    main,
    select_by_elements,
    synthesize_candidate_profile,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CIFS_DIR = FIXTURES_DIR / "cifs"
PATTERNS_DIR = FIXTURES_DIR / "patterns"

# A scoring result with explicit, dull values. Display tests care about columns
# and widths, not about the numbers, so each test replaces only the field it
# examines instead of rebuilding an eighteen-field record by hand.
BASE_RESULT = CandidateScoringResult(
    phase_name="candidate",
    formula="Sm2O3",
    cif_file="candidate",
    structure=None,
    score=42.0,
    matched_lines=3,
    total_lines=4,
    completeness_ml_tl=0.75,
    intensity_match=80.0,
    lattice_shift_da_a=0.0,
    zero_shift=0.0,
    scale_factor=1.0,
    pos_overlap=0.5,
    strain_penalty=1.0,
    missing_penalty=1.0,
    occam_penalty=1.0,
    strong_line_penalty=1.0,
    diagnostic_flags=["EVALUATED"],
)


def make_result(**overrides: Any) -> CandidateScoringResult:
    """Copy the base record with the given fields replaced."""
    return replace(BASE_RESULT, **overrides)


def test_analytical_bragg_dispersion_tan_theta():
    """Verify analytical Bragg peak shift follows -2*tan(theta)*(da/a) and anisotropic axial weights."""
    two_theta = np.array([20.0, 50.0, 80.0])
    da = 0.01  # Lattice expansion
    shifted = apply_bragg_cell_shift(two_theta, delta_a_over_a=da)

    assert np.all(shifted < two_theta)
    shift_20 = abs(shifted[0] - two_theta[0])
    shift_80 = abs(shifted[2] - two_theta[2])
    assert shift_80 > shift_20

    # When da == dc, anisotropic shift must equal isotropic shift identically
    w_c = np.array([0.0, 0.5, 1.0])
    iso_equiv = apply_anisotropic_bragg_shift(two_theta, da, da, w_c)
    assert np.allclose(iso_equiv, shifted)

    # Pure (hk0) vs pure (00l) with opposite axial strains
    aniso = apply_anisotropic_bragg_shift(
        np.array([30.0, 30.0]),
        delta_a_over_a=0.01,
        delta_c_over_c=-0.01,
        c_weights=np.array([0.0, 1.0]),
    )
    assert aniso[0] < 30.0 < aniso[1]


def test_strict_two_theta_bounds_for_tl():
    """Verify candidate reflections outside experimental 2theta range do not inflate TL."""
    engine = CandidateScoringEngine(min_line_intensity=1.0)
    struct = Structure.from_file(str(CIFS_DIR / "TiO2_rutile.cif"))

    # Narrow range: 20 to 45 deg
    _, peaks_narrow, _ = engine.calculate_cif_pattern(struct, two_theta_range=(20.0, 45.0))
    # Wide range: 20 to 90 deg
    _, peaks_wide, _ = engine.calculate_cif_pattern(struct, two_theta_range=(20.0, 90.0))

    assert len(peaks_narrow) < len(peaks_wide)
    assert np.all(peaks_narrow <= 45.0)
    assert np.all(peaks_narrow >= 20.0)


def test_low_symmetry_parasite_rejected_by_ml_tl():
    """Verify that a candidate with hundreds of lines but few matches is penalized by ML/TL."""
    two_theta = np.linspace(15.0, 85.0, 2500)
    # Synthesize clean TiO2 rutile experimental pattern
    engine = CandidateScoringEngine()
    rutile_struct = Structure.from_file(str(CIFS_DIR / "TiO2_rutile.cif"))
    _, r_2th, r_i = engine.calculate_cif_pattern(rutile_struct, (15.0, 85.0))
    y_signal = synthesize_candidate_profile(two_theta, r_2th, r_i)
    y_obs = y_signal * 100.0 + 50.0  # background 50 counts
    sigma = np.sqrt(y_obs)
    y_net = y_signal * 100.0

    # Create artificial low-symmetry structure with huge unit cell and 100+ reflections
    parasite_lattice = Lattice.from_parameters(14.0, 15.0, 16.0, 75.0, 82.0, 68.0)
    parasite_struct = Structure(parasite_lattice, ["Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    candidates = {
        "TiO2_rutile": rutile_struct,
        "Parasite_low_symmetry": parasite_struct,
    }

    results = engine.score_candidates(two_theta, y_net, sigma, candidates)

    assert len(results) >= 2
    top = results[0]
    assert top.phase_name == "TiO2_rutile"
    assert top.score > 85.0
    assert "PERFECT_MATCH" in top.diagnostic_flags

    parasite_res = next(r for r in results if r.phase_name == "Parasite_low_symmetry")
    assert parasite_res.score < top.score / 2.0
    # Parasite must have low completeness flag or ratio < 0.20
    assert parasite_res.completeness_ml_tl < 0.25


def test_veto_triple_peak_noise_floor():
    """Verify that candidate with all top-3 lines in background noise triggers VETO."""
    two_theta = np.linspace(15.0, 85.0, 2500)
    engine = CandidateScoringEngine()
    # Experimental pattern is pure CeO2
    ceo2_struct = Structure.from_file(str(CIFS_DIR / "CeO2.cif"))
    _, c_2th, c_i = engine.calculate_cif_pattern(ceo2_struct, (15.0, 85.0))
    y_signal = synthesize_candidate_profile(two_theta, c_2th, c_i)
    y_obs = y_signal * 100.0 + 30.0
    sigma = np.sqrt(y_obs)
    y_net = y_signal * 100.0

    # ZnO has distinct peaks from CeO2; its top lines fall into noise
    candidates = {
        "CeO2": ceo2_struct,
        "ZnO": CIFS_DIR / "ZnO.cif",
    }

    results = engine.score_candidates(two_theta, y_net, sigma, candidates)
    zno_res = next(r for r in results if r.phase_name == "ZnO")

    assert zno_res.missing_penalty <= 0.10
    assert "VETO: NO_MAJOR_PEAKS" in zno_res.diagnostic_flags
    assert zno_res.score < 15.0


def test_texture_suspect_graceful_handling():
    """Verify that a missing 100% line with another strong line present gets TEXTURE_SUSPECT, not hard veto."""
    two_theta = np.linspace(15.0, 85.0, 2500)
    engine = CandidateScoringEngine()
    rutile_struct = Structure.from_file(str(CIFS_DIR / "TiO2_rutile.cif"))
    _, r_2th, r_i = engine.calculate_cif_pattern(rutile_struct, (15.0, 85.0))

    # Suppress the 100% primary peak (index 0) to simulate extreme preferred orientation
    textured_i = np.copy(r_i)
    textured_i[0] = 0.0  # Zero out the strongest line
    y_signal = synthesize_candidate_profile(two_theta, r_2th, textured_i)
    y_obs = y_signal * 100.0 + 30.0
    sigma = np.sqrt(y_obs)
    y_net = y_signal * 100.0

    results = engine.score_candidates(two_theta, y_net, sigma, {"TiO2_rutile": rutile_struct})
    assert len(results) == 1
    r = results[0]

    # Must NOT have hard VETO
    assert not any("VETO: NO_MAJOR_PEAKS" in f for f in r.diagnostic_flags)
    # Must be flagged as TEXTURE_SUSPECT
    assert any("TEXTURE_SUSPECT" in f for f in r.diagnostic_flags)
    assert r.missing_penalty == 0.50
    # Score is moderately reduced, not crushed to zero
    assert r.score > 25.0


def test_sub_second_fast_strain_optimization():
    """Verify that evaluation of multiple candidates stays under 1.0 s.

    Twenty candidates score in about 0.23 s on an idle machine, so the threshold
    keeps a fourfold margin and does not turn machine load into a false alarm.
    """
    data = np.loadtxt(PATTERNS_DIR / "Mixture_Rutile95_Anatase5.xy")
    two_theta, intensity = data[:, 0], data[:, 1]
    prob = ProfileProbabilityField.from_raw_pattern(two_theta, intensity)

    cifs = {
        "CeO2": CIFS_DIR / "CeO2.cif",
        "TiO2_rutile": CIFS_DIR / "TiO2_rutile.cif",
        "TiO2_anatase": CIFS_DIR / "TiO2_anatase.cif",
        "ZnO": CIFS_DIR / "ZnO.cif",
    }
    # Duplicate pool to test scale
    cifs_pool = {f"{k}_{i}": v for i in range(5) for k, v in cifs.items()}
    assert len(cifs_pool) == 20

    engine = CandidateScoringEngine()
    t0 = time.perf_counter()
    results = engine.score_candidates(
        two_theta,
        prob.y_net,
        prob.sigma,
        cifs_pool,
    )
    duration = time.perf_counter() - t0

    assert len(results) == 20
    assert duration < 1.0  # Sub-second performance for 20 candidates


def test_robust_median_scale_estimation():
    """Verify that scale factor of dominant phase exceeds minor phase."""
    data = np.loadtxt(PATTERNS_DIR / "Mixture_Rutile95_Anatase5.xy")
    two_theta, intensity = data[:, 0], data[:, 1]
    prob = ProfileProbabilityField.from_raw_pattern(two_theta, intensity)

    engine = CandidateScoringEngine()
    results = engine.score_candidates(
        two_theta,
        prob.y_net,
        prob.sigma,
        {
            "TiO2_rutile": CIFS_DIR / "TiO2_rutile.cif",
            "TiO2_anatase": CIFS_DIR / "TiO2_anatase.cif",
        },
    )

    assert len(results) == 2
    r_rutile = next(r for r in results if r.phase_name == "TiO2_rutile")
    r_anatase = next(r for r in results if r.phase_name == "TiO2_anatase")

    assert r_rutile.scale_factor > r_anatase.scale_factor
    assert r_rutile.score > r_anatase.score


def test_candidate_scoring_cli_and_json(capsys):
    """Verify CLI entrypoint with JSON output format."""
    test_args = [
        "search_match.py",
        "--data",
        str(PATTERNS_DIR / "CeO2.xy"),
        "--candidate-cifs",
        str(CIFS_DIR),
        "--json",
    ]
    with patch("sys.argv", test_args):
        rc = main()
    assert rc == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["phase_name"] == "CeO2"
    assert data[0]["score"] > 80.0
    assert "PERFECT_MATCH" in data[0]["diagnostic_flags"]


def test_format_candidate_table_layout():
    """Verify ASCII table formatting has correct header and aligned columns."""
    engine = CandidateScoringEngine()
    struct = Structure.from_file(str(CIFS_DIR / "CeO2.cif"))
    data = np.loadtxt(PATTERNS_DIR / "CeO2.xy")
    prob = ProfileProbabilityField.from_raw_pattern(data[:, 0], data[:, 1])
    res = engine.score_candidates(data[:, 0], prob.y_net, prob.sigma, {"CeO2": struct})

    tbl = format_candidate_table(res)
    assert "CANDIDATE RANKING LIST" in tbl
    assert "CeO2" in tbl
    assert "Rank" in tbl
    assert "Score" in tbl
    assert "ML / TL" in tbl
    assert "Formula" in tbl


def _rule_width(tbl: str) -> int:
    """Width of the table rules, read from the artefact itself and not from a literal."""
    return len(tbl.splitlines()[0])


def _data_rows(tbl: str, count: int) -> list[str]:
    """The data rows, excluding the two rules, the title, the header and the column rule."""
    return tbl.splitlines()[5 : 5 + count]


def test_table_rules_header_and_rows_share_one_width():
    """The flags column is open-ended, so the rule width must follow the content.

    A hard-coded rule drifts as soon as a diagnostic flag grows: the closing
    column then floats and the table stops being a table. The width is therefore
    checked against the longest flag present, never against a literal.
    """
    short = make_result(phase_name="short", diagnostic_flags=["EVALUATED"])
    long_flag = "LATTICE_STRAINED (da/a=-1.70%); RESIDUAL_UNIQUE (2/3)"
    long = make_result(phase_name="long", diagnostic_flags=[long_flag])

    tbl_short = format_candidate_table([short])
    tbl_long = format_candidate_table([short, long])

    for tbl, n_rows in ((tbl_short, 1), (tbl_long, 2)):
        lines = tbl.splitlines()
        width = _rule_width(tbl)
        assert lines[2] == lines[0], "top rule and the rule under the title must match"
        assert len(lines[3]) == width, "header must match the rule width"
        assert len(lines[4]) == width, "column rule must match the rule width"
        rows = _data_rows(tbl, n_rows)
        assert len(rows) == n_rows
        for row in rows:
            assert len(row) == width, f"row width {len(row)} != rule width {width}"
        assert lines[5 + n_rows] == lines[0], "closing rule must match"

    assert long_flag in tbl_long
    assert len(tbl_long.splitlines()[0]) > len(tbl_short.splitlines()[0]), (
        "a longer flag must widen the whole table, not overflow the rule"
    )


def test_table_shows_formula_and_truncates_long_ones():
    """Long card formulas are trimmed in the text table; JSON keeps them whole.

    Card formulas from PDF-2 are long -- "( N H4 )2 ( Mo6 Cl14 ) ( H2 O )" -- and
    an untrimmed column would break the alignment of every row below the header.
    Truncation is a display concern only: the machine-readable payload must keep
    the full string.
    """
    long_formula = "( N H4 )2 ( Mo6 Cl14 ) ( H2 O )"
    result = make_result(phase_name="89-2937", formula=long_formula, cif_file="89-2937")

    tbl = format_candidate_table([result])
    assert long_formula not in tbl, "long formula must be trimmed in the text table"
    assert "( N H4 )2 ( Mo6 Cl14" in tbl

    # The trimmed cell must still respect the row width.
    row = _data_rows(tbl, 1)[0]
    assert len(row) == _rule_width(tbl)

    # The JSON payload keeps the formula in full.
    assert result.to_dict()["formula"] == long_formula


def test_table_marks_missing_formula_without_ragged_row():
    """An absent formula is shown as a dash and must not ragged the row."""
    result = make_result(phase_name="x", formula="", cif_file="x")

    tbl = format_candidate_table([result])
    assert "| - " in tbl
    assert "Diagnostic Flags" in tbl
    assert len(_data_rows(tbl, 1)[0]) == _rule_width(tbl)


def test_demote_unmatched_strong_downweights_absent_strong_lines():
    """A candidate whose own strong lines are absent in the pattern is demoted.

    DemoteUnmatchedStrong is graded over the candidate's STRONG reference lines
    (>= threshold of its own strongest). The true phase, whose strong lines are all
    observed, must keep P_strong = 1.0; a rich-line decoy carrying strong lines
    that do not exist in the experiment must be penalised. The opposite reading
    (demoting candidates that leave strong *observed* peaks unexplained) would
    punish every minor phase and is therefore not the implemented semantics.
    """
    data = np.loadtxt(PATTERNS_DIR / "CeO2.xy")
    two_theta, intensity = data[:, 0], data[:, 1]
    prob = ProfileProbabilityField.from_raw_pattern(two_theta, intensity)

    engine = CandidateScoringEngine()
    results = engine.score_candidates(
        two_theta,
        prob.y_net,
        prob.sigma,
        {
            "CeO2": CIFS_DIR / "CeO2.cif",  # true phase of this pattern
            "TiO2_rutile": CIFS_DIR / "TiO2_rutile.cif",  # strong lines absent here
        },
    )

    by_name = {r.phase_name: r for r in results}
    true = by_name["CeO2"]
    decoy = by_name["TiO2_rutile"]

    assert true.strong_line_penalty == pytest.approx(1.0, abs=1e-6)
    assert decoy.strong_line_penalty < 0.5
    assert decoy.strong_line_penalty < true.strong_line_penalty
    assert decoy.score < true.score
    assert any("RESIDUAL_UNIQUE" in f for f in decoy.diagnostic_flags)


def test_demote_unmatched_strong_can_be_disabled():
    """Weight 0.0 must switch the penalty off entirely (CLI --demote-unmatched-strong)."""
    data = np.loadtxt(PATTERNS_DIR / "CeO2.xy")
    prob = ProfileProbabilityField.from_raw_pattern(data[:, 0], data[:, 1])

    engine = CandidateScoringEngine(demote_unmatched_strong=0.0)
    results = engine.score_candidates(
        data[:, 0], prob.y_net, prob.sigma, {"TiO2_rutile": CIFS_DIR / "TiO2_rutile.cif"}
    )
    assert results[0].strong_line_penalty == 1.0


def test_element_filter_is_reused_not_reimplemented():
    """The element semantics must come from cif_retriever, not from a second copy."""
    import cif_retriever
    import search_match

    assert search_match.ElementFilter is cif_retriever.ElementFilter
    assert search_match.extract_elements is cif_retriever.extract_elements


def test_select_by_elements_without_filter_keeps_everything():
    """An absent filter must not narrow the pool.

    The candidate pool has to cover every thermodynamically plausible product of
    the synthesis, the container and the atmosphere; silently dropping phases is
    prohibited.
    """
    pool = {"a": {"Sm", "S"}, "b": {"Sm", "O"}, "c": {"C"}}
    assert select_by_elements(pool, None) == ["a", "b", "c"]
    assert select_by_elements({}, None) == []


def test_select_by_elements_applies_and_or_not_and_strict():
    """The three-way element filter semantics (AND / OR / NOT / strict) must be honoured."""
    pool = {
        "sm_s": {"Sm", "S"},
        "sm_oxysulfide": {"Sm", "O", "S"},
        "carbon": {"C"},
        "silicate": {"Si", "O"},
        "sm_selenium": {"Sm", "Se"},
    }

    # AND: must contain both Sm and S
    f = ElementFilter.from_raw(all_elements="Sm,S")
    assert set(select_by_elements(pool, f)) == {"sm_s", "sm_oxysulfide"}

    # OR: contains Sm or Se
    f = ElementFilter.from_raw(any_elements="Sm,Se")
    assert set(select_by_elements(pool, f)) == {"sm_s", "sm_oxysulfide", "sm_selenium"}

    # NOT: contains no carbon
    f = ElementFilter.from_raw(all_elements="Sm", none_elements="C")
    assert set(select_by_elements(pool, f)) == {"sm_s", "sm_oxysulfide", "sm_selenium"}

    # STRICT: closed system -- nothing outside Sm and S
    f = ElementFilter.from_raw(all_elements="Sm,S", strict=True)
    assert set(select_by_elements(pool, f)) == {"sm_s"}
    assert "sm_oxysulfide" not in select_by_elements(pool, f)  # oxygen is foreign


def test_extract_elements_handles_pdf2_formula_spacing():
    """PDF-2 formulas separate elements with spaces and use brackets / digits."""
    assert extract_elements("Sm S") == {"Sm", "S"}
    assert extract_elements("( N H4 )2 ( Mo6 Cl14 ) ( H2 O )") == {"N", "H", "Mo", "Cl", "O"}
    assert extract_elements("Ca ( Si F6 ) !2 H2 O") >= {"Ca", "Si", "F", "H", "O"}
    # A chalcogen analogue must not be confused with sulfur.
    assert extract_elements("Sm Se") == {"Sm", "Se"}
    assert "S" not in extract_elements("Sm Se")


def test_scoring_profiles_are_complete_and_distinct():
    """Named profiles must carry every engine parameter and differ where intended."""
    from search_match import SCORING_PROFILES

    assert set(SCORING_PROFILES) == {"ide-all", "minors"}
    keys = {
        "angular_tolerance",
        "min_line_intensity",
        "max_delta_a",
        "max_zero_shift",
        "strong_line_threshold",
        "demote_unmatched_strong",
        "coverage_weight",
    }
    for name, prof in SCORING_PROFILES.items():
        assert set(prof) == keys, f"profile '{name}' is incomplete or over-specified"

    ide, minors = SCORING_PROFILES["ide-all"], SCORING_PROFILES["minors"]
    # A minor phase must not be demoted for leaving the dominant peaks unexplained.
    assert ide["demote_unmatched_strong"] > 0.0
    assert minors["demote_unmatched_strong"] == 0.0
    # Weak reference lines carry the minor phase's evidence.
    assert minors["min_line_intensity"] < ide["min_line_intensity"]
    # Its own strong lines are the only ones worth enforcing.
    assert minors["strong_line_threshold"] > ide["strong_line_threshold"]
    # The coverage exponent is pinned out of the score in every profile. The value itself is
    # still measured and reported for every candidate (see the coverage test below); what was
    # measured on the card series is that the multiplier compresses the scores without
    # separating the physically possible phase -- top-1 unchanged in 11 of 11 and the order
    # inside the top three changed in 1 of 11, there in the wrong direction -- because on the
    # real pool the coverage is nearly the same for all candidates (the winner explains 4 of
    # 10 observed peaks).
    assert ide["coverage_weight"] == 0.0
    assert minors["coverage_weight"] == 0.0


def test_observed_coverage_is_measured_and_applied_only_by_profile():
    """The coverage term must state how much of the pattern a candidate explains.

    Completeness of a candidate's OWN line list is not evidence about the pattern: on the
    Sm-Gd-S series every top-ranked card reached perfect ML/TL while explaining 2-6 of the
    10-20 observed peaks. Two properties are checked, and the second one is why the term is
    separable from the measurement: a report must not state a coverage of 1.0 for a pattern
    that was never examined, so the number is measured even when its weight is zero, and the
    weight decides only whether it enters the score.
    """
    data = np.loadtxt(PATTERNS_DIR / "CeO2.xy")
    two_theta = data[:, 0]
    prob = ProfileProbabilityField.from_raw_pattern(two_theta, data[:, 1])
    pool = {
        "CeO2": CIFS_DIR / "CeO2.cif",
        "TiO2_rutile": CIFS_DIR / "TiO2_rutile.cif",
    }

    measured = data[:, 1]
    applied = CandidateScoringEngine(coverage_weight=0.5)
    by_name = {
        r.phase_name: r
        for r in applied.score_candidates(
            two_theta, prob.y_net, prob.sigma, pool, observed_signal=measured
        )
    }
    true_phase = by_name["CeO2"]
    decoy = by_name["TiO2_rutile"]

    # The denominator is a significant-peak count from the measured signal: the fixture
    # shows nine, of which the true phase accounts for eight. The one it leaves is a weak
    # maximum on the flank of the strongest reflection, which no phase is obliged to
    # explain.
    assert true_phase.observed_peaks_total == 9
    assert true_phase.observed_peaks_explained == 8
    assert true_phase.observed_peak_coverage == pytest.approx(8 / 9, abs=1e-6)
    # A phase whose strong lines are absent explains far less of the same pattern, and the
    # factor follows the measurement in the same order.
    assert decoy.observed_peak_coverage < true_phase.observed_peak_coverage
    assert decoy.coverage_penalty < true_phase.coverage_penalty
    assert true_phase.coverage_penalty < 1.0

    # Without the measured signal the term does not fall back to the clipped field: it stays
    # inert and names the omission. Measuring on the clipped field would have added a ridge
    # maximum to the denominator -- the MAD estimate falls from 3.51 to 0.68 there -- and
    # docked the true phase for a feature that belongs to no phase.
    from loguru import logger as loguru_logger

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="INFO")
    try:
        inert = {
            r.phase_name: r
            for r in applied.score_candidates(two_theta, prob.y_net, prob.sigma, pool)
        }["CeO2"]
    finally:
        loguru_logger.remove(sink_id)
    assert inert.observed_peaks_total == 0
    assert inert.observed_peak_coverage == 1.0
    assert inert.coverage_penalty == 1.0
    assert any("Coverage term inert" in message for message in messages), messages

    off = {
        r.phase_name: r
        for r in CandidateScoringEngine(coverage_weight=0.0).score_candidates(
            two_theta, prob.y_net, prob.sigma, pool, observed_signal=measured
        )
    }["CeO2"]
    # Still measured, no longer applied.
    assert off.observed_peaks_total == true_phase.observed_peaks_total
    assert off.observed_peak_coverage == pytest.approx(true_phase.observed_peak_coverage)
    assert off.coverage_penalty == 1.0


def _pdf2_addressable() -> bool:
    """True when a PDF-2 installation is addressable, however the address was declared.

    The check goes through the reader's own resolver rather than through the environment: an
    address may be exported as a variable or declared in a ``.env`` file, and a test looking
    only at the environment would skip on a machine where the database is reachable -- which
    is the same mistake that once made a sibling test fail rather than skip.
    """
    from pdf2_reader import PDF2NotFoundError, resolve_pdf2_dir

    try:
        resolve_pdf2_dir()
    except PDF2NotFoundError:
        return False
    return True


def test_pdf2_cli_route_consumes_the_reference_pool(tmp_path):
    """The CLI route to the reference card pool must be exercised end to end.

    The suite once stayed green while this route was broken: an import removed from
    ``search_match.py`` left ``--pdf2`` returning rc=1 for all eleven series samples, and no
    test touched the flag. A green suite is therefore not evidence for this route, and this
    test is the evidence: it drives the real CLI with the flag and requires that the first
    pass ran, that cards survived the element criterion, and that a card-shaped result came
    back. It skips only when no installation is addressable, which is a genuine absence
    rather than a hidden obstacle of our own making.
    """
    if not _pdf2_addressable():
        pytest.skip(
            "no addressable PDF-2 installation: point PDF2_DATA_DIR at one to exercise the "
            "card route"
        )
    from loguru import logger as loguru_logger

    out_file = tmp_path / "pdf2_scored.json"
    test_args = [
        "search_match.py",
        "--data",
        str(PATTERNS_DIR / "CeO2.xy"),
        "--pdf2",
        "--any",
        "Ce,O",
        "--strict",
        "--json",
        "--out",
        str(out_file),
    ]

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="INFO")
    try:
        with patch("sys.argv", test_args):
            assert main() == 0
    finally:
        loguru_logger.remove(sink_id)

    # The first pass is the part that must not silently disappear: without it the route
    # still runs, but it reads the entire card stream instead of a few hundred cards.
    assert any("PDF-2 first pass:" in message for message in messages), messages
    assert any("PDF-2 pool:" in message for message in messages), messages

    scored = json.loads(out_file.read_text(encoding="utf-8"))
    assert scored, "the card route produced no scored candidate"
    top = scored[0]
    # A card-shaped record: the name is a PDF number and the card's cell travels with it,
    # which is what tells a real card apart from a CIF-shaped stand-in.
    assert top["cif_file"].count("-") == 1, top["cif_file"]
    assert top["cell"] and top["cell"][0] is not None, top["cell"]
    assert top["observed_peaks_total"] >= 0


def test_cli_profile_minors_is_accepted():
    """`--profile minors` must run end to end and emit a JSON payload."""
    test_args = [
        "search_match.py",
        "--data",
        str(PATTERNS_DIR / "CeO2.xy"),
        "--candidate-cifs",
        str(CIFS_DIR),
        "--profile",
        "minors",
        "--json",
    ]
    with patch("sys.argv", test_args):
        assert main() == 0


def test_rich_line_candidate_does_not_outrank_equal_volume_poor_phase():
    """The observed failure: a 34-line candidate beat a 12-line phase at equal volume.

    Equal cell volumes made the parsimony term blind to line count, so the
    rich-line candidate won on raw matches -- many of which belonged to the other
    phase in the pattern. The complexity term now follows the line count, and the
    residual pass strips credit for peaks an accepted phase already explains.
    """
    engine = CandidateScoringEngine()
    two_theta = np.linspace(15.0, 85.0, 3500)

    # True phase: cubic, few distinct reflections in the window.
    true = Structure(Lattice.cubic(6.0), ["Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    # Second phase supplying extra observed peaks for the decoy to steal.
    filler = Structure(Lattice.cubic(8.0), ["Zn"], [[0, 0, 0]])
    # Decoy: nearly the same volume, triclinic -> far more distinct lines.
    decoy = Structure(
        Lattice.from_parameters(6.0, 6.0, 6.0, 88.0, 91.0, 89.0),
        ["Ti", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    assert abs(decoy.volume - true.volume) / true.volume < 0.02

    _, t_2th, t_i = engine.calculate_cif_pattern(true, (15.0, 85.0))
    _, f_2th, f_i = engine.calculate_cif_pattern(filler, (15.0, 85.0))
    _, d_2th, _ = engine.calculate_cif_pattern(decoy, (15.0, 85.0))
    assert len(d_2th) > 1.5 * len(t_2th), "decoy must have far more lines"

    y_obs = synthesize_candidate_profile(
        two_theta, np.concatenate([t_2th, f_2th]), np.concatenate([t_i, f_i])
    )
    y_obs = y_obs * 100.0 + 30.0
    sigma = np.sqrt(np.maximum(y_obs, 1.0))

    results = engine.score_candidates(
        two_theta, y_obs - 30.0, sigma, {"true": true, "decoy": decoy}
    )
    assert results[0].phase_name == "true", (
        f"rich-line decoy outranked the true phase: "
        f"{results[0].phase_name} {results[0].score:.1f} > {results[1].score:.1f}"
    )


def test_kbeta_satellites_do_not_identify_any_phase():
    """A pattern made only of Cu K-beta satellites must not yield a phase.

    K-beta companions are peaks of non-structural origin (SKILL.md, Rule 13,
    category 2). Against a large reference card pool such a satellite always finds a
    card, so the invariant to protect is that no candidate reaches the
    STRONG_MATCH threshold on a K-beta-only pattern.
    """
    engine = CandidateScoringEngine()
    two_theta = np.linspace(15.0, 85.0, 3500)

    # Take the true K-alpha lines of CeO2 and move each to its K-beta position.
    ceo2 = Structure.from_file(str(CIFS_DIR / "CeO2.cif"))
    _, ka2th, ka_i = engine.calculate_cif_pattern(ceo2, (15.0, 85.0))
    lam_beta_over_alpha = 1.39222 / 1.540596
    arg = lam_beta_over_alpha * np.sin(np.radians(ka2th / 2.0))
    kb2th = np.degrees(2.0 * np.arcsin(np.clip(arg, -1.0, 1.0)))

    y_obs = synthesize_candidate_profile(two_theta, kb2th, ka_i)
    y_obs = y_obs * 100.0 + 30.0
    sigma = np.sqrt(np.maximum(y_obs, 1.0))

    pool = {
        "CeO2": CIFS_DIR / "CeO2.cif",
        "TiO2_rutile": CIFS_DIR / "TiO2_rutile.cif",
        "TiO2_anatase": CIFS_DIR / "TiO2_anatase.cif",
        "ZnO": CIFS_DIR / "ZnO.cif",
    }
    results = engine.score_candidates(two_theta, y_obs - 30.0, sigma, pool)

    assert results, "scoring produced no candidates at all"
    for r in results:
        assert not any("STRONG_MATCH" in f or "PERFECT_MATCH" in f for f in r.diagnostic_flags), (
            f"K-beta satellite identified as a phase: {r.phase_name} {r.score:.1f}"
        )


def _card_from_structure(engine: CandidateScoringEngine, struct: Structure) -> CandidateLines:
    """Build a PDF-2 style candidate whose lines are taken from the structure itself.

    The inverse Bragg expression appears here only to synthesise the card input;
    it does not predict what the engine returns for that card.
    """
    _, ref_2th, ref_i = engine.calculate_cif_pattern(struct, (15.0, 85.0))
    d_spacing = engine.wavelength / (2.0 * np.sin(np.radians(ref_2th / 2.0)))
    return CandidateLines(
        label=struct.composition.reduced_formula,
        formula=struct.composition.reduced_formula,
        volume=struct.volume,
        d_spacing=d_spacing,
        intensity=ref_i,
    )


def test_card_lines_reproduce_the_structure_pattern():
    """Path equivalence: a d-spacing card must return the angles it was built from.

    Cards carry d-spacings and no atomic coordinates, so they reach the engine
    through CandidateLines and Bragg's law. The invariant is that the card route
    reproduces the structure route exactly, which is what makes a card candidate
    admissible into a pool of CIF candidates at all.
    """
    engine = CandidateScoringEngine()
    struct = Structure.from_file(str(CIFS_DIR / "CeO2.cif"))
    _, ref_2th, ref_i = engine.calculate_cif_pattern(struct, (15.0, 85.0))
    card = _card_from_structure(engine, struct)

    _, card_2th, card_i = engine.calculate_lines_pattern(card, (15.0, 85.0))

    assert len(card_2th) == len(ref_2th)
    assert np.allclose(card_2th, ref_2th, atol=1e-9)
    assert np.allclose(card_i, ref_i, atol=1e-9)


def test_card_lines_outside_the_window_are_dropped():
    """The angular window applies to cards exactly as it does to structures."""
    engine = CandidateScoringEngine()
    struct = Structure.from_file(str(CIFS_DIR / "CeO2.cif"))
    card = _card_from_structure(engine, struct)

    _, narrow_2th, _ = engine.calculate_lines_pattern(card, (20.0, 45.0))
    _, wide_2th, _ = engine.calculate_lines_pattern(card, (15.0, 85.0))

    assert len(narrow_2th) < len(wide_2th)
    assert np.all((narrow_2th >= 20.0) & (narrow_2th <= 45.0))


def test_card_candidate_scores_identically_to_its_structure():
    """A card must reach the same verdict as the structure it mirrors.

    The two routes are scored in separate pools on purpose. Putting both records
    of one phase into a single pool triggers the residual pass, which strips
    credit for peaks an accepted phase already explains, so the second record is
    zeroed by design; that behaviour has its own test and must not be re-tested
    here through an accidental zero.
    """
    engine = CandidateScoringEngine()
    struct = Structure.from_file(str(CIFS_DIR / "CeO2.cif"))
    _, ref_2th, ref_i = engine.calculate_cif_pattern(struct, (15.0, 85.0))
    card = _card_from_structure(engine, struct)

    two_theta = np.linspace(15.0, 85.0, 2500)
    y_obs = synthesize_candidate_profile(two_theta, ref_2th, ref_i) * 100.0 + 30.0
    sigma = np.sqrt(y_obs)
    y_net = y_obs - 30.0

    from_structure = engine.score_candidates(two_theta, y_net, sigma, {"structure": struct})[0]
    from_card = engine.score_candidates(two_theta, y_net, sigma, {"card": card})[0]

    assert from_card.score == pytest.approx(from_structure.score, rel=1e-9)
    assert from_card.diagnostic_flags == from_structure.diagnostic_flags
    assert from_card.matched_lines == from_structure.matched_lines
    assert from_card.total_lines == from_structure.total_lines
    assert from_card.volume == pytest.approx(from_structure.volume, rel=1e-9)


def test_matched_peak_index_beyond_the_grid_is_a_miss_not_a_crash():
    """A matched position past the last grid point must count as a miss.

    ``np.searchsorted`` returns the array length for such a position, and the
    residual pass fed that index straight into the array: the two-phase scoring
    died with IndexError on 7 of 17 real patterns instead of reporting the miss.
    """
    from search_match import count_unique_matched_peaks

    exp_two_theta = np.linspace(20.0, 60.0, 2001)
    covering = np.array([30.0, 40.0])
    beyond = len(exp_two_theta)

    assert count_unique_matched_peaks([beyond], exp_two_theta, covering) == 0
    assert count_unique_matched_peaks([0, beyond], exp_two_theta, covering) == 1


def test_difference_curve_field_keeps_the_sign_of_the_residual():
    """The residual field must not be clipped to positive.

    Clipping turns it into a broadly positive surface where every candidate line
    "matches", and completeness then saturates at 1.00 for unrelated phases.
    """
    two_theta = np.linspace(20.0, 60.0, 400)
    y_obs = np.full_like(two_theta, 100.0)
    y_calc = np.full_like(two_theta, 100.0)
    y_calc[(two_theta > 29.0) & (two_theta < 31.0)] = 200.0  # model too tall here
    y_calc[(two_theta > 39.0) & (two_theta < 41.0)] = 0.0  # model too short here

    field = ProfileProbabilityField.from_difference_curve(two_theta, y_obs, y_calc)

    assert field.clamp_positive is False
    assert float(np.min(field.u_field)) < 0.0


def test_isostructural_ambiguity_is_reported_for_a_solid_solution_pair():
    """Two candidates of one structure type at different cells must be flagged.

    Ranking them apart hides the fact that the observed cell lies between them,
    which is how the wrong member of a solid-solution series gets adopted and the
    reflections of the right one are then hunted as an unknown impurity.
    """
    small = Structure.from_spacegroup(
        "I-43d", Lattice.cubic(8.4471), ["Sm"], [[0.125, 0.125, 0.125]]
    )
    large = Structure.from_spacegroup(
        "I-43d", Lattice.cubic(8.5239), ["Sm"], [[0.125, 0.125, 0.125]]
    )

    engine = CandidateScoringEngine()
    two_theta = np.linspace(20.0, 60.0, 3000)
    _struct, peak_2th, peak_i = engine.calculate_cif_pattern(large, (20.0, 60.0))
    y_obs = synthesize_candidate_profile(two_theta, peak_2th, peak_i) * 100.0 + 30.0
    sigma = np.sqrt(np.maximum(y_obs, 1.0))

    results = engine.score_candidates(
        two_theta, y_obs - 30.0, sigma, {"small": small, "large": large}
    )

    assert results, "no candidate scored at all"
    # The engine groups candidates by the space-group token it reads from the
    # structure; both candidates must land in one group. The exact spelling comes
    # from spglib (Ia-3d) rather than from the CIF setting (I-43d), so only the
    # fact of a shared non-empty token is asserted.
    groups = {result.space_group for result in results}
    assert len(groups) == 1 and groups != {""}
    flags = " ".join(flag for result in results for flag in result.diagnostic_flags)
    assert "ISOSTRUCTURAL_AMBIGUITY" in flags


def test_residual_cli_scores_a_difference_curve(tmp_path):
    """``--residual`` must score a three-column difference curve."""
    two_theta = np.linspace(20.0, 60.0, 1200)
    # The far wing of the Gaussian underflows by design; the error state is fixed
    # here so the synthetic pattern does not depend on a session-wide policy.
    with np.errstate(under="ignore"):
        y_obs = 30.0 + 80.0 * np.exp(-0.5 * ((two_theta - 28.5) / 0.12) ** 2)
    y_calc = np.full_like(two_theta, 30.0)
    residual_file = tmp_path / "residual.dat"
    residual_file.write_text(
        "\n".join(f"{t:.4f} {o:.4f} {c:.4f}" for t, o, c in zip(two_theta, y_obs, y_calc)),
        encoding="utf-8",
    )
    out_file = tmp_path / "scored.json"

    test_args = [
        "search_match.py",
        "--residual",
        str(residual_file),
        "--candidate-cifs",
        str(CIFS_DIR),
        "--json",
        "--out",
        str(out_file),
    ]
    # The CLI contract is the subject here, not numpy's error policy: profile
    # synthesis evaluates exp(-large) in the far tails, where an underflow to zero
    # is expected. A session that has switched underflow to "raise" must not make
    # this test fail.
    with np.errstate(under="ignore"), patch("sys.argv", test_args):
        assert main() == 0
    assert out_file.exists()
    scored = json.loads(out_file.read_text(encoding="utf-8"))
    assert scored and "space_group" in scored[0]


def test_export_layout_round_trips_through_the_adapter(tmp_path):
    """A six-column export is read back as a three-column difference curve."""
    from gsas2_adapter import read_difference_curve

    export = tmp_path / "profile_curves.csv"
    export.write_text(
        "two_theta,y_obs,y_calc,y_bkg,y_diff,sigma_obs\n"
        "20.0000,120.00,118.00,110.00,2.00,10.95\n"
        "20.0200,131.00,129.50,110.10,1.50,11.45\n",
        encoding="utf-8",
    )

    table = read_difference_curve(export)

    assert table.shape == (2, 6)
    assert table[0, 0] == pytest.approx(20.0)
    assert table[1, 2] == pytest.approx(129.5)


def test_plain_columns_are_still_accepted(tmp_path):
    """A plain 2theta/Yobs/Ycalc file keeps working (the previous interface)."""
    from gsas2_adapter import read_difference_curve

    plain = tmp_path / "residual.dat"
    plain.write_text("20.0 120.0 118.0\n20.1 121.0 119.0\n", encoding="utf-8")

    table = read_difference_curve(plain)

    assert table.shape == (2, 3)
    assert table[1, 1] == pytest.approx(121.0)


def test_too_few_columns_is_refused(tmp_path):
    """A two-column file is a raw pattern, not a difference curve."""
    from gsas2_adapter import read_difference_curve

    raw = tmp_path / "pattern.xy"
    raw.write_text("20.0 120.0\n20.1 121.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column"):
        read_difference_curve(raw)


def test_two_sources_at_once_are_refused(tmp_path):
    """Exactly one pattern source may be given."""
    pattern = PATTERNS_DIR / "CeO2.xy"
    residual = tmp_path / "residual.dat"
    residual.write_text("20.0 120.0 118.0\n", encoding="utf-8")

    test_args = [
        "search_match.py",
        "--data",
        str(pattern),
        "--residual",
        str(residual),
        "--candidate-cifs",
        str(CIFS_DIR),
    ]
    with patch("sys.argv", test_args):
        assert main() == 2


def test_pdf2_requires_an_element_criterion():
    """The reference card database is never scored unfiltered.

    The check must fire before the database is touched, so this test is deterministic
    whether or not an installation is declared in the environment.
    """
    test_args = [
        "search_match.py",
        "--data",
        str(PATTERNS_DIR / "CeO2.xy"),
        "--candidate-cifs",
        str(CIFS_DIR),
        "--pdf2",
    ]
    with patch("sys.argv", test_args):
        assert main() == 2
