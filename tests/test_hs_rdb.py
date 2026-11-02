"""Unit and integration tests for the COD24_HS4x.hsrdb adapter (hs_rdb.py)."""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest
from cif_retriever import ElementFilter
from hs_rdb import (
    COD_HSRDB_ENV,
    AtomSite,
    CodStructureRecord,
    HsRdbDatabase,
    HsRdbNotFoundError,
    PeakFilter,
    build_crystal_sql_clauses,
    build_element_sql_clauses,
    elements_to_masks,
    extract_cod_id,
    masks_to_elements,
    resolve_hsrdb_path,
    to_signed_int64,
    to_unsigned_int64,
)
from pdf2_reader import CrystallographicFilter


def _hsrdb_is_addressable() -> bool:
    try:
        resolve_hsrdb_path()
    except HsRdbNotFoundError:
        return False
    return True


def test_missing_hsrdb_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(COD_HSRDB_ENV, raising=False)
    monkeypatch.setattr("hs_rdb.load_env_file", lambda: None)
    with pytest.raises(HsRdbNotFoundError):
        resolve_hsrdb_path()

    monkeypatch.setenv(COD_HSRDB_ENV, str(tmp_path / "absent.hsrdb"))
    with pytest.raises(HsRdbNotFoundError):
        resolve_hsrdb_path()


def test_signed_int64_and_element_masks_boundaries() -> None:
    u_bit63 = 1 << 63
    s_bit63 = to_signed_int64(u_bit63)
    assert s_bit63 == -(1 << 63)
    assert to_unsigned_int64(s_bit63) == u_bit63

    # Boundary elements: H (Z=1), Eu (Z=63, sign bit of e1), Gd (Z=64, bit 0 of e2), Lu (Z=71), U (Z=92)
    m = elements_to_masks(["H", "Eu", "Gd", "Lu", "U"])
    assert m.e1 < 0  # Eu sets bit 63
    assert m.e2 > 0
    assert masks_to_elements(m.e1, m.e2) == {"H", "Eu", "Gd", "Lu", "U"}


def test_extract_cod_id_from_comment_and_product_id() -> None:
    assert (
        extract_cod_id("963500134", "Publication title: X\rCOD database code: 3500133")
        == "3500133"
    )
    assert extract_cod_id("961537845", None) == "1537844"


def test_sql_clause_builders() -> None:
    ef = ElementFilter.from_raw(all_elements="Lu,O", none_elements="C", strict=True)
    e_sql = build_element_sql_clauses(ef)
    assert any("e1 &" in c or "e2 &" in c for c in e_sql.clauses)
    assert any("e1 |" in c and "e2 |" in c for c in e_sql.clauses)

    cf = CrystallographicFilter(
        cell_ranges={"a": (10.0, 11.0), "volume": (1100.0, 1150.0)},
        crystal_systems=frozenset({"cubic"}),
        subfiles=frozenset({"inorganic"}),
    )
    c_sql = build_crystal_sql_clauses(cf)
    assert "gi.A >= ?" in c_sql.clauses
    assert "gi.XTLVOL <= ?" in c_sql.clauses
    assert "gi.XTSLSYS IN (?)" in c_sql.clauses
    assert any("SubFileRef" in c for c in c_sql.clauses)


def test_peak_filter_blob_matching() -> None:
    import struct

    pf = PeakFilter(peaks=[29.75], tolerance=0.20, is_two_theta=True, min_intensity=200.0)
    blob_hit = struct.pack("<fH", 3.003, 1000)
    blob_miss = struct.pack("<fH", 2.100, 1000)
    assert pf.matches_blob(blob_hit) is True
    assert pf.matches_blob(blob_miss) is False


def test_record_and_atom_site_contracts() -> None:
    rec = CodStructureRecord(
        pid=1,
        cod_id="1000000",
        product_id="961000001",
        formula="O2 Si",
        name="Quartz",
        mineral_name="Quartz",
        spacegroup="P 32 2 1",
        ita_number=154,
        crystal_system="trigonal",
        a=4.91,
        b=4.91,
        c=5.40,
        alpha=90.0,
        beta=90.0,
        gamma=120.0,
        volume=112.5,
        z=3,
        density_calc=2.65,
        density_meas=2.65,
        rir=3.5,
        temperature_k=293.0,
        pressure_kpa=101.3,
        has_phase=True,
        num_reflections=42,
    )
    assert rec.density_calc == 2.65
    assert rec.density_meas == 2.65
    assert rec.temperature_k == 293.0
    assert rec.pressure_kpa == 101.3
    assert rec.has_phase is True
    assert rec.num_reflections == 42

    site = AtomSite(label="Si1", element="Si", wyckoff="3a", multiplicity=3)
    assert site.multiplicity == 3


@pytest.mark.skipif(not _hsrdb_is_addressable(), reason="no addressable COD24_HS4x.hsrdb")
def test_live_hsrdb_search_and_cif_export(tmp_path: Path) -> None:
    with HsRdbDatabase() as db:
        assert db.count_total() == 511583

        ef = ElementFilter.from_raw(all_elements="Lu,O", strict=True)
        cf = CrystallographicFilter(crystal_systems=frozenset({"cubic"}))
        recs = db.search(element_filter=ef, crystal_filter=cf, limit=5)
        assert len(recs) >= 1
        first = recs[0]
        assert first.cod_id.isdigit()
        assert first.rir is not None and first.rir > 0

        # Resolve by COD ID and verify Phase decoding + CIF export
        resolved_pid = db.pid_for_cod_id(first.cod_id)
        assert resolved_pid == first.pid
        phase_data = db.parse_phase(first.pid)
        assert phase_data is not None and len(phase_data.atoms) > 0
        assert any(site.wyckoff != "" for site in phase_data.atoms.values())

        out_cif = db.export_cif(first.cod_id, out_path=tmp_path / f"cod_{first.cod_id}.cif")
        assert out_cif.is_file()
        doc = gemmi.cif.read_file(str(out_cif))
        block = doc.sole_block()
        assert block.find_value("_cell_length_a") is not None
        assert len(block.find_values("_atom_site_label")) > 0

        cand = db.get_candidate_lines(first.pid)
        assert cand.label == f"cod-{first.cod_id}"
        assert cand.d_spacing.size > 0
        assert len(db.get_literature(first.pid)) >= 0


@pytest.mark.skipif(not _hsrdb_is_addressable(), reason="no addressable COD24_HS4x.hsrdb")
def test_live_hsrdb_subfiles_partition_and_cif_retriever() -> None:
    """Verify subfile counts against official COD24_HS4x.hsrdb specs and test cif_retriever --subfiles."""
    from cif_retriever import search_cifs

    with HsRdbDatabase() as db:
        inorg_cnt = len(
            db.search(
                crystal_filter=CrystallographicFilter(subfiles=frozenset({"inorganic"})),
                limit=None,
            )
        )
        metal_cnt = len(
            db.search(
                crystal_filter=CrystallographicFilter(subfiles=frozenset({"metallic"})),
                limit=None,
            )
        )
        mineral_cnt = len(
            db.search(
                crystal_filter=CrystallographicFilter(subfiles=frozenset({"mineral"})),
                limit=None,
            )
        )
        assert inorg_cnt == 84480
        assert metal_cnt == 9854
        assert mineral_cnt == 15990

    # In Lu-O-C-H system, unfiltered search includes C-H organometallics while --subfiles inorganic excludes them
    ef_lu = ElementFilter.from_raw(all_elements="Lu,O", any_elements="C,H", strict=True)
    res_all = search_cifs(filter=ef_lu, source="cod", limit=0)
    res_inorg = search_cifs(
        filter=ef_lu,
        source="cod",
        limit=0,
        subfiles=frozenset({"inorganic"}),
    )
    assert res_all.total_count > res_inorg.total_count
    assert res_inorg.total_count >= 1


@pytest.mark.skipif(not _hsrdb_is_addressable(), reason="no addressable COD24_HS4x.hsrdb")
def test_live_hsrdb_search_match_cli(tmp_path: Path) -> None:
    """Verify search_match.py --hsrdb --subfiles inorganic end-to-end on CeO2 fixture."""
    import json
    from unittest.mock import patch

    from search_match import main as sm_main

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    out_json = tmp_path / "hsrdb_ceo2.json"
    argv = [
        "search_match.py",
        "--data",
        str(fixtures_dir / "patterns" / "CeO2.xy"),
        "--hsrdb",
        "--elements",
        "Ce,O",
        "--strict",
        "--subfiles",
        "inorganic",
        "--json",
        "--out",
        str(out_json),
    ]
    with patch("sys.argv", argv):
        assert sm_main() == 0

    scored = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(scored) >= 1
    top = scored[0]
    assert top["phase_name"].startswith("cod-")
    assert top["score"] > 75.0


@pytest.mark.skipif(not _hsrdb_is_addressable(), reason="no addressable COD24_HS4x.hsrdb")
def test_live_hsrdb_add_phase_from_cod(tmp_path: Path) -> None:
    """Verify cmd_add_phase --from-cod extracts full structure with atoms from COD24_HS4x.hsrdb into GSAS-II."""
    from types import SimpleNamespace

    from stage_gate_rietveld import ensure_gsas2_environment
    from xrd_project import cmd_add_phase, cmd_init

    if not ensure_gsas2_environment():
        pytest.skip("GSAS-II engine not available")

    from GSASII import GSASIIscriptable as G2sc

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    project_gpx = tmp_path / "cod_import.gpx"
    assert (
        cmd_init(
            SimpleNamespace(
                data=fixtures_dir / "patterns" / "CeO2.xy",
                project=project_gpx,
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
                project=project_gpx,
                cif=None,
                from_card=None,
                from_cod="1537844",
                name="Lu2O3_cod",
                scale=0.01,
                pawley=False,
            )
        )
        == 0
    )

    gpx = G2sc.G2Project(str(project_gpx))
    phases = gpx.phases()
    assert len(phases) == 1
    p = phases[0]
    assert p.name == "Lu2O3_cod"
    assert len(list(p.atoms())) >= 2
    cell_a = float(p.data["General"]["Cell"][1])
    assert 10.35 <= cell_a <= 10.45
