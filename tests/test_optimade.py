"""Unit and integration tests for OPTIMADE API client and crystal structure converter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cif_retriever import ElementFilter
from optimade_client import (
    CellMetrics,
    OptimadeHttpResponse,
    ParsedStructureId,
    StabilityMetrics,
    _compute_cell_parameters,
    _extract_energy_above_hull,
    download_optimade_cif,
    element_filter_to_optimade,
    optimade_resource_to_structure,
    parse_composite_id,
    parse_optimade_record,
    search_optimade,
    structure_to_cif_text,
)


def test_element_filter_to_optimade_translation() -> None:
    # 1. Simple all_of
    f1 = ElementFilter(all_of=frozenset({"Sm", "S"}))
    q1 = element_filter_to_optimade(f1)
    assert q1 == 'elements HAS ALL "S", "Sm"'

    # 2. any_of with supports_has_any=True
    f2 = ElementFilter(all_of=frozenset({"Lu", "O"}), any_of=frozenset({"C", "H"}))
    q2 = element_filter_to_optimade(f2, supports_has_any=True)
    assert q2 == 'elements HAS ALL "Lu", "O" AND elements HAS ANY "C", "H"'

    # 3. any_of with supports_has_any=False (explicit OR)
    q3 = element_filter_to_optimade(f2, supports_has_any=False)
    assert q3 == 'elements HAS ALL "Lu", "O" AND (elements HAS "C" OR elements HAS "H")'

    # 4. none_of exclusion
    f4 = ElementFilter(all_of=frozenset({"Sm", "S"}), none_of=frozenset({"Cl", "F"}))
    q4 = element_filter_to_optimade(f4)
    assert q4 == 'elements HAS ALL "S", "Sm" AND NOT (elements HAS "Cl" OR elements HAS "F")'

    # 5. strict closed system without any_of
    f5 = ElementFilter(all_of=frozenset({"Lu", "O"}), strict=True)
    q5 = element_filter_to_optimade(f5)
    assert q5 == 'elements HAS ALL "Lu", "O" AND nelements = 2'

    # 6. strict closed system with any_of
    f6 = ElementFilter(all_of=frozenset({"Lu", "O"}), any_of=frozenset({"C"}), strict=True)
    q6 = element_filter_to_optimade(f6)
    assert q6 == 'elements HAS ALL "Lu", "O" AND elements HAS ANY "C" AND nelements <= 3'


def test_element_filter_determinism() -> None:
    # Independent of set insertion / hashing order, output must be character-identical
    f_a = ElementFilter(all_of=frozenset(["Sm", "S", "O", "Fe"]))
    f_b = ElementFilter(all_of=frozenset(["Fe", "O", "S", "Sm"]))
    assert element_filter_to_optimade(f_a) == element_filter_to_optimade(f_b)
    assert element_filter_to_optimade(f_a) == 'elements HAS ALL "Fe", "O", "S", "Sm"'


def test_parse_composite_id() -> None:
    p1 = parse_composite_id("optimade-mp-mp-149")
    assert isinstance(p1, ParsedStructureId)
    assert p1.provider_id == "mp"
    assert p1.native_id == "mp-149"

    p2 = parse_composite_id("optimade-alexandria_pbe-agm001003285")
    assert p2.provider_id == "alexandria_pbe"
    assert p2.native_id == "agm001003285"

    p3 = parse_composite_id("optimade-cod-1008317")
    assert p3.provider_id == "cod"
    assert p3.native_id == "1008317"

    p4 = parse_composite_id("mp-149")
    assert p4.provider_id == "mp"
    assert p4.native_id == "mp-149"

    p5 = parse_composite_id("1008317")
    assert p5.provider_id == "cod"
    assert p5.native_id == "1008317"


def test_compute_cell_parameters() -> None:
    # Simple cubic 5x5x5
    cubic_lat = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    metrics = _compute_cell_parameters(cubic_lat)
    assert isinstance(metrics, CellMetrics)
    assert pytest.approx(metrics.a) == 5.0
    assert pytest.approx(metrics.b) == 5.0
    assert pytest.approx(metrics.c) == 5.0
    assert pytest.approx(metrics.alpha) == 90.0
    assert pytest.approx(metrics.beta) == 90.0
    assert pytest.approx(metrics.gamma) == 90.0
    assert pytest.approx(metrics.volume) == 125.0

    # Degenerate matrix returns None
    assert _compute_cell_parameters([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) is None


def test_extract_energy_above_hull() -> None:
    # 1. MP multi-functional stability dict
    mp_attrs = {
        "_mp_stability": {
            "gga": {"energy_above_hull": 0.045},
            "r2scan": {"energy_above_hull": 0.0},
        }
    }
    m1 = _extract_energy_above_hull(mp_attrs)
    assert isinstance(m1, StabilityMetrics)
    assert m1.energy_above_hull == 0.0
    assert m1.is_stable is True

    # 2. Alexandria hull distance
    alex_attrs = {"_alexandria_hull_distance": 0.012}
    m2 = _extract_energy_above_hull(alex_attrs)
    assert m2.energy_above_hull == 0.012
    assert m2.is_stable is False

    # 3. COD (no thermo data)
    m3 = _extract_energy_above_hull({})
    assert m3.energy_above_hull is None
    assert m3.is_stable is None


def test_parse_optimade_record() -> None:
    mock_item = {
        "id": "mp-149",
        "attributes": {
            "chemical_formula_reduced": "Si",
            "space_group_symbol": "F d -3 m",
            "lattice_vectors": [[3.333, 0.0, 1.924], [1.111, 3.142, 1.924], [0.0, 0.0, 3.849]],
            "energy_above_hull": 0.0,
        },
    }
    rec = parse_optimade_record("mp", mock_item)
    assert rec is not None
    assert rec.id == "optimade-mp-mp-149"
    assert rec.formula == "Si"
    assert rec.spacegroup == "F d -3 m"
    assert rec.is_stable is True
    assert rec.a is not None and rec.volume is not None


def test_search_optimade_with_mocks() -> None:
    mock_mp_response = {
        "data": [
            {
                "id": "mp-1102762",
                "attributes": {
                    "chemical_formula_reduced": "AsSSm",
                    "space_group_symbol": "P n m a",
                    "lattice_vectors": [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    "energy_above_hull": 0.0,
                },
            },
            {
                "id": "mp-999999",
                "attributes": {
                    "chemical_formula_reduced": "Fe2O3",  # Mismatches Sm-S filter
                    "space_group_symbol": "R -3 c",
                    "lattice_vectors": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 14.0]],
                },
            },
        ]
    }

    def _mock_get_json(url: str, **kwargs: Any) -> OptimadeHttpResponse:
        if "optimade.materialsproject.org" in url:
            return OptimadeHttpResponse(status=200, data=mock_mp_response)
        return OptimadeHttpResponse(status=200, data={"data": []})

    f = ElementFilter(all_of=frozenset({"Sm", "S"}))
    with patch("optimade_client._http_get_json", side_effect=_mock_get_json):
        results = search_optimade(f, providers=["mp"])
        assert len(results) == 1
        assert results[0].formula == "AsSSm"
        assert results[0].id == "optimade-mp-mp-1102762"


def test_optimade_resource_to_structure_and_cif() -> None:
    # Silicon diamond structure
    si_attrs = {
        "chemical_formula_descriptive": "Si",
        "lattice_vectors": [[3.333573, 0.0, 1.924639], [1.111191, 3.142924, 1.924639], [0.0, 0.0, 3.849278]],
        "cartesian_site_positions": [[-0.0, -0.0, -0.0], [1.111191, 0.785731, 1.924639]],
        "species_at_sites": ["Si", "Si"],
        "species": [{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1.0]}],
    }
    struct = optimade_resource_to_structure(si_attrs)
    assert struct.is_ordered is True
    # Conventional cell of diamond Si has 8 atoms in FCC lattice
    assert len(struct) == 8

    cif_str = structure_to_cif_text(struct)
    assert "_cell_length_a" in cif_str
    assert "_symmetry_space_group_name_H-M" in cif_str
    assert "Fd-3m" in cif_str or "F d -3 m" in cif_str
    assert "227" in cif_str


def test_disordered_alloy_structure_conversion() -> None:
    # Partial occupancy (Sm0.7Gd0.3)S
    disordered_attrs = {
        "chemical_formula_descriptive": "(Sm0.7Gd0.3)S",
        "lattice_vectors": [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]],
        "cartesian_site_positions": [
            [0.0, 0.0, 0.0],
            [2.1, 2.1, 2.1],
        ],
        "species_at_sites": ["Sm_Gd", "S"],
        "species": [
            {"name": "Sm_Gd", "chemical_symbols": ["Sm", "Gd"], "concentration": [0.7, 0.3]},
            {"name": "S", "chemical_symbols": ["S"], "concentration": [1.0]},
        ],
    }
    struct = optimade_resource_to_structure(disordered_attrs)
    assert struct.is_ordered is False
    cif_str = structure_to_cif_text(struct)
    assert "_atom_site_occupancy" in cif_str
    assert "0.7" in cif_str
    assert "0.3" in cif_str


def test_download_optimade_cif_with_mock(tmp_path: Path) -> None:
    mock_payload = {
        "data": {
            "id": "mp-149",
            "attributes": {
                "chemical_formula_reduced": "Si",
                "lattice_vectors": [[3.333573, 0.0, 1.924639], [1.111191, 3.142924, 1.924639], [0.0, 0.0, 3.849278]],
                "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.111191, 0.785731, 1.924639]],
                "species_at_sites": ["Si", "Si"],
                "species": [{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1.0]}],
            },
        }
    }

    with patch("optimade_client._http_get_json", return_value=OptimadeHttpResponse(status=200, data=mock_payload)):
        out_file = download_optimade_cif("optimade-mp-mp-149", out_path=tmp_path)
        assert out_file is not None
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "_cell_length_a" in content
        assert "Si" in content


def test_gsas2_phase_loading_invariant(tmp_path: Path) -> None:
    """Verify that generated CIF passes GSAS-II phase loading invariant."""
    try:
        from stage_gate_rietveld import ensure_gsas2_environment
        if not ensure_gsas2_environment():
            pytest.skip("GSAS-II calculation engine is not installed/discoverable.")
        from GSASII import GSASIIscriptable as G2sc
    except (ImportError, RuntimeError):
        pytest.skip("GSAS-II scriptable library not available in this test runner.")

    # Generate test CIF for Si
    si_attrs = {
        "chemical_formula_descriptive": "Si",
        "lattice_vectors": [[3.333573, 0.0, 1.924639], [1.111191, 3.142924, 1.924639], [0.0, 0.0, 3.849278]],
        "cartesian_site_positions": [[-0.0, -0.0, -0.0], [1.111191, 0.785731, 1.924639]],
        "species_at_sites": ["Si", "Si"],
        "species": [{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1.0]}],
    }
    struct = optimade_resource_to_structure(si_attrs)
    cif_text = structure_to_cif_text(struct)
    cif_path = tmp_path / "Si_standard.cif"
    cif_path.write_text(cif_text, encoding="utf-8")

    gpx_path = tmp_path / "test_project.gpx"
    gpx = G2sc.G2Project(newgpx=str(gpx_path))
    phase = gpx.add_phase(str(cif_path), phasename="Silicon", fmthint="CIF")

    assert phase.name == "Silicon"
    sg_name = phase.data.get("General", {}).get("SGData", {}).get("SpGrp", "")
    assert "F" in sg_name
    assert "d" in sg_name
