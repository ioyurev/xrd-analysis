"""Tests for the PDF-2 reader and its crystallographic filters."""

from __future__ import annotations

import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pdf2_reader import (
    QUALITY_MARKS,
    CrystallographicFilter,
    PDF2Database,
    PDF2NotFoundError,
    PDFCard,
    crystal_system_from_pearson,
)

PDF2_ENV = "PDF2_DATA_DIR"


# ---------------------------------------------------------------------------
# Helpers: synthetic 80-byte records reproducing the format, not the content.
# ---------------------------------------------------------------------------
def make_record(
    data70: str,
    seq: str = " ",
    status: str = "P",
    setno: str = "26",
    cardno: str = "1479",
    cls: str = "C",
    rtype: str = "K",
) -> bytes:
    body = data70.ljust(70)[:70]
    tag = f"{seq}{status}{setno}{cardno}{cls}{rtype}"
    assert len(tag) == 10
    return (body + tag).encode("latin1")


def reflection_row(reflections: list[tuple[str, str, str]]) -> str:
    """Pack up to three (d, I, hkl) triples into one 70-char body, 23 chars each."""
    body = ""
    for d_val, i_val, hkl in reflections[:3]:
        body += f"{d_val:<7}{i_val:<4}{hkl:<12}"
    return body.ljust(70)[:70]


def make_card_bytes(*, setno: str = "26", cardno: str = "1479", cls: str = "C") -> bytes:
    out = b""
    out += make_record("5.962", setno=setno, cardno=cardno, cls=cls, rtype="1")
    out += make_record(
        "Fm-3m      225          4             5.710      211.92",
        setno=setno,
        cardno=cardno,
        cls=cls,
        rtype="3",
    )
    out += make_record("I  A", setno=setno, cardno=cardno, cls=cls, rtype="5")
    out += make_record("Samarium Sulfide", setno=setno, cardno=cardno, cls=cls, rtype="6")
    out += make_record("Sm S", setno=setno, cardno=cardno, cls=cls, rtype="7")
    out += make_record("S Sm", setno=setno, cardno=cardno, cls=cls, rtype="8")
    out += make_record("410442          FZ", setno=setno, cardno=cardno, cls=cls, rtype="B")
    out += make_record("cF   8.00        \\Cl Na\\", setno=setno, cardno=cardno, cls=cls, rtype="A")
    out += make_record(
        "  5.962   5.962   5.962  90.00  90.00  90.00   5.9620",
        setno=setno,
        cardno=cardno,
        cls=cls,
        rtype="E",
    )
    out += make_record(
        reflection_row([("3.44000", " 45 ", "  1  1  1C  "), ("2.98100", "100 ", "  2  0  0C  ")]),
        setno=setno,
        cardno=cardno,
        cls=cls,
        rtype="I",
    )
    out += make_record(
        reflection_row([("2.10800", " 95 ", "  2  2  0C  ")]),
        setno=setno,
        cardno=cardno,
        cls=cls,
        rtype="I",
    )
    out += make_record("I 2.98/X 2.11/X 3.44/5", setno=setno, cardno=cardno, cls=cls, rtype="+")
    out += make_record("I 3.44/5 2.98/X 2.11/X", setno=setno, cardno=cardno, cls=cls, rtype="*")
    out += make_record("", setno=setno, cardno=cardno, cls=cls, rtype="K")
    return out


def build_db(tmp_path: Path, card_blob: bytes, monkeypatch: pytest.MonkeyPatch) -> PDF2Database:
    """Materialise a minimal database, declare it and open it.

    The location is configuration, so the reader is addressed the only way it can
    be addressed: through ``PDF2_DATA_DIR``.
    """
    (tmp_path / "Vaxind").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pdf2.dat").write_bytes(card_blob)
    offsets = [0]
    (tmp_path / "Vaxind" / "offset54.ind").write_bytes(struct.pack(f"<{len(offsets)}I", *offsets))
    monkeypatch.setenv(PDF2_ENV, str(tmp_path))
    db = PDF2Database()
    # The reader warns when the entry count differs from the documented release.
    assert db.card_count == len(offsets)
    return db


# ---------------------------------------------------------------------------
# Crystal system derivation
# ---------------------------------------------------------------------------
def test_crystal_system_from_pearson_symbol():
    """The Pearson symbol's leading letter encodes the crystal system."""
    assert crystal_system_from_pearson("cF   8.00") == "cubic"
    assert crystal_system_from_pearson("mC 132.00") == "monoclinic"
    assert crystal_system_from_pearson("oP  14.00") == "orthorhombic"
    assert crystal_system_from_pearson("tP   6.00") == "tetragonal"
    assert crystal_system_from_pearson("hP   8.00") == "trigonal-hexagonal"
    assert crystal_system_from_pearson("aP   4.00") == "triclinic"
    # Unknown or absent symbols must not be guessed.
    assert crystal_system_from_pearson("") == "unknown"
    assert crystal_system_from_pearson("   ") == "unknown"
    assert crystal_system_from_pearson("cF?") == "cubic"


# ---------------------------------------------------------------------------
# Crystallographic filter (the deliverable of this step)
# ---------------------------------------------------------------------------
_BASE_CARD = PDFCard(
    set_number=26,
    card_number=1479,
    status="P",
    quality_class="C",
    mineral_name="Samarium Sulfide",
    empirical_formula="Sm S",
    structural_formula="S Sm",
    space_group="Fm-3m",
    ita_number=225,
    z_value=4.0,
    cell_a=5.962,
    cell_b=5.962,
    cell_c=5.962,
    cell_alpha=90.0,
    cell_beta=90.0,
    cell_gamma=90.0,
    cell_volume=211.9,
    pearson_symbol="cF   8.00",
    prototype="Cl Na",
)


def _card(**kwargs) -> PDFCard:
    return replace(_BASE_CARD, **kwargs)


def test_filter_without_criteria_matches_everything():
    """An empty filter must not reject anything."""
    empty = CrystallographicFilter()
    assert empty.matches(_card()) is True
    assert empty.matches(_card(cell_a=None, space_group="", pearson_symbol="")) is True


def test_filter_cell_ranges_are_inclusive_and_honour_missing_values():
    """Cell ranges must reject out-of-range cells and cards lacking the value."""
    f = CrystallographicFilter(cell_ranges={"a": (5.90, 6.00)})
    assert f.matches(_card(cell_a=5.962)) is True
    assert f.matches(_card(cell_a=5.90)) is True  # lower bound inclusive
    assert f.matches(_card(cell_a=6.00)) is True  # upper bound inclusive
    assert f.matches(_card(cell_a=6.5)) is False
    assert f.matches(_card(cell_a=None)) is False  # criterion cannot be evaluated

    open_ended = CrystallographicFilter(cell_ranges={"a": (6.00, None)})
    assert open_ended.matches(_card(cell_a=5.962)) is False
    assert open_ended.matches(_card(cell_a=9.0)) is True


def test_filter_space_group_ignores_whitespace():
    """Space-group comparison must be insensitive to the symbol's spacing."""
    f = CrystallographicFilter(space_groups=frozenset({"Fm-3m"}))
    assert f.matches(_card(space_group="Fm-3m")) is True
    assert f.matches(_card(space_group="F m -3 m")) is True
    assert f.matches(_card(space_group="Fd-3m")) is False
    assert f.matches(_card(space_group="")) is False


def test_filter_crystal_system_uses_pearson_symbol():
    """Crystal system is derived deterministically from ITA number/space group, with Pearson fallback."""
    cubic = CrystallographicFilter(crystal_systems=frozenset({"cubic"}))
    assert cubic.matches(_card(pearson_symbol="cF   8.00")) is True
    assert (
        cubic.matches(_card(ita_number=None, space_group="", pearson_symbol="mC 132.00"))
        is False
    )
    assert (
        cubic.matches(_card(ita_number=None, space_group="", pearson_symbol="aP   4.00"))
        is False
    )
    assert cubic.matches(_card(ita_number=None, space_group="", pearson_symbol="")) is False
    tetragonal = CrystallographicFilter(crystal_systems=frozenset({"tetragonal"}))
    assert (
        tetragonal.matches(
            _card(ita_number=139, space_group="I4/mmm", pearson_symbol="h??")
        )
        is True
    )


def test_filter_criteria_are_combined_with_logical_and():
    """A card must satisfy every declared criterion, not just one."""
    f = CrystallographicFilter(
        cell_ranges={"a": (5.90, 6.00)},
        space_groups=frozenset({"Fm-3m"}),
        crystal_systems=frozenset({"cubic"}),
    )
    assert f.matches(_card()) is True
    assert f.matches(_card(cell_a=7.0)) is False
    assert f.matches(_card(space_group="Fd-3m")) is False
    assert f.matches(_card(ita_number=None, space_group="", pearson_symbol="mC 132.00")) is False


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def test_missing_database_is_reported_not_guessed(monkeypatch):
    """With neither address present the reader must refuse, not guess a path.

    The location may legitimately be declared in the environment or in a ``.env`` file,
    so a test of "no address at all" has to remove both. Removing only the variable
    leaves any ``.env`` above the working directory free to supply the path, and the
    reader then opens a real database instead of refusing. That is not a defect of the
    reader: it is the ``.env`` route working. The failure it produces is false, and it
    would push a real reader defect out of sight by making the test unfixable.
    """
    monkeypatch.delenv(PDF2_ENV, raising=False)
    # Addressed by name so the module needs no import here, and so the neutralisation
    # covers whichever .env happens to sit above the working directory.
    monkeypatch.setattr("env_file.find_dotenv", lambda **_kwargs: "")
    with pytest.raises(PDF2NotFoundError):
        PDF2Database()


def test_environment_naming_a_missing_directory_is_reported(monkeypatch, tmp_path):
    """A declared but wrong location is an error, not a silent fallback."""
    monkeypatch.setenv(PDF2_ENV, str(tmp_path / "absent"))
    with pytest.raises(PDF2NotFoundError):
        PDF2Database()


def test_database_location_comes_from_environment(monkeypatch, tmp_path):
    """PDF2_DATA_DIR is the only way to address the installation."""
    with build_db(tmp_path, make_card_bytes(), monkeypatch) as db:
        assert db.db_dir == tmp_path
        assert db.card_count == 1


def test_card_number_lookup_finds_the_card(monkeypatch, tmp_path):
    """A card is addressed by its PDF number, not by a position from a log."""
    with build_db(tmp_path, make_card_bytes(), monkeypatch) as db:
        assert db.index_of("26-1479") == 0


def test_card_number_lookup_rejects_a_non_number(monkeypatch, tmp_path):
    """A malformed argument is reported as such, separately from an absent card."""
    with build_db(tmp_path, make_card_bytes(), monkeypatch) as db:
        with pytest.raises(ValueError, match="not a PDF number"):
            db.index_of("1479")
        with pytest.raises(ValueError, match="absent"):
            db.index_of("99-9999")


def test_synthetic_card_parses_all_expected_fields(monkeypatch, tmp_path):
    """A synthetic card must round-trip the tag line, reflections and ICSD number."""
    db = build_db(tmp_path, make_card_bytes(), monkeypatch)
    try:
        card = db.get_card(0)
        assert card.pdf_number == "26-1479"
        assert card.set_number == 26 and card.card_number == 1479
        assert card.status == "P" and card.quality_class == "C"

        assert card.space_group == "Fm-3m"
        assert card.ita_number == 225
        assert card.z_value == pytest.approx(4.0)
        assert card.cell_a == pytest.approx(5.962)
        assert card.cell_volume == pytest.approx(5.9620)

        assert card.pearson_symbol.startswith("cF")
        assert card.prototype == "Cl Na"
        assert "I" in card.subfiles
        assert card.icsd_number == "410442"
        assert card.empirical_formula == "Sm S"
        assert card.mineral_name == "Samarium Sulfide"
        assert card.structural_formula == "S Sm"

        assert len(card.reflections) == 3
        first = card.reflections[0]
        assert first.d_spacing == pytest.approx(3.44)
        assert first.intensity == pytest.approx(45.0)
        assert first.hkl == (1, 1, 1)

        # Hanawalt order from the '+' record, parsed despite the glued prefix.
        assert card.strongest_d == (2.98, 2.11, 3.44)

        assert card.crystal_system == "cubic"
        assert card.is_deleted is False
    finally:
        db.close()


def test_reflection_layout_is_positional_not_split(monkeypatch, tmp_path):
    """d = [0:7], I = [7:11], hkl = [11:23]; three reflections per record."""
    body = reflection_row([("3.44000", " 45 ", "  1  1  1C  ")])
    assert body[0:7].strip() == "3.44000"
    assert body[7:11].strip() == "45"
    assert body[11:23].replace(" ", "") == "111C"

    db = build_db(tmp_path, make_card_bytes(), monkeypatch)
    try:
        card = db.get_card(0)
        assert card.reflections[0].d_spacing == pytest.approx(3.44)
        assert card.reflections[1].d_spacing == pytest.approx(2.981)
        assert card.reflections[2].d_spacing == pytest.approx(2.108)
    finally:
        db.close()


def test_tag_line_layout():
    """Columns 71-80: counter, status, set, card, class, type."""
    rec = make_record("d=", seq="3", status="A", setno="89", cardno="9114", cls="M", rtype="B")
    assert rec[70:80] == b"3A899114MB"


def test_quality_mark_vocabulary_is_complete():
    """Verify standard quality mark code indices (0..10)."""
    assert QUALITY_MARKS[1] == "qmStar"
    assert QUALITY_MARKS[5] == "qmCalculated"
    assert len(QUALITY_MARKS) == 11
    assert set(QUALITY_MARKS) == set(range(11))


def test_deleted_cards_are_excluded_by_default(monkeypatch, tmp_path):
    """filter_cards must skip deleted cards unless asked for them."""
    deleted = bytearray(make_card_bytes(setno="26", cardno="1480", cls="C"))
    # Every record of a card carries the card's status, not just the terminator.
    for offset in range(0, len(deleted), 80):
        deleted[offset + 71] = ord("D")

    first = make_card_bytes()
    blob = first + bytes(deleted)
    (tmp_path / "Vaxind").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pdf2.dat").write_bytes(blob)
    # The second card starts exactly where the first one ends.
    (tmp_path / "Vaxind" / "offset54.ind").write_bytes(struct.pack("<2I", 0, len(first)))
    monkeypatch.setenv(PDF2_ENV, str(tmp_path))

    with PDF2Database() as db:
        assert len(db.filter_cards()) == 1
        assert len(db.filter_cards(include_deleted=True)) == 2


# ---------------------------------------------------------------------------
# Real database (runs only when the local installation is present)
# ---------------------------------------------------------------------------
def _pdf2_is_addressable() -> bool:
    """True when an installation is addressable, however the address was declared.

    The check must go through the reader's own resolver. An address may be exported as a
    variable or declared in a ``.env`` file, and a test looking only at the environment
    would skip on a machine where the database is reachable -- hiding a working
    configuration route, which is the same mistake that made the sibling test fail.
    """
    from pdf2_reader import PDF2NotFoundError, resolve_pdf2_dir

    try:
        resolve_pdf2_dir()
    except PDF2NotFoundError:
        return False
    return True


@pytest.mark.skipif(not _pdf2_is_addressable(), reason="no addressable PDF-2 installation")
def test_real_card_26_1479_reproduces_its_cell():
    """The SmS card's reflections must follow from its own cell (a/sqrt(3), ...).

    The tolerance is 5e-3 Å, not 2e-3: a card states d-spacings and a refined cell as
    two independent roundings, so they agree only to a few thousandths of an ångström.
    On card 26-1479 the largest disagreement is d(111) = 3.4400 against a/sqrt(3) =
    3.4422. A tolerance below that would fail a correct card, while a parser that
    misplaced a column would still be caught, since its error is in tenths of an Å.
    """
    import math

    with PDF2Database() as db:
        card = db.get_card(db.index_of("26-1479"))
        assert card.pdf_number == "26-1479"
        assert card.cell_a == pytest.approx(5.962, abs=1e-3)

        for refl in card.reflections:
            if len(refl.hkl) < 3:
                continue
            h, k, l = refl.hkl[:3]
            expected = card.cell_a / math.sqrt(h * h + k * k + l * l)
            assert refl.d_spacing == pytest.approx(expected, abs=5e-3)


# ---------------------------------------------------------------------------
# Criteria completed for the reference database: the Pearson symbol itself, and the
# subfiles. Both are compared on a card, so they need no database at all.
# ---------------------------------------------------------------------------
def test_pearson_symbol_matches_letters_exactly_and_count_optionally():
    """The symbol separates cells the crystal system cannot, and an absent count means any.

    ``cF8`` and ``cP1`` are both cubic, so the crystal system alone cannot express "face
    centred"; the count is compared only when the request states one, and unparseable input
    falls back to a literal comparison rather than being accepted.
    """
    from pdf2_reader import pearson_symbol_matches

    stored = "cF   8.00"
    assert pearson_symbol_matches(stored, "cF8") is True
    assert pearson_symbol_matches(stored, "cF") is True  # count not asked for
    assert pearson_symbol_matches(stored, "cF8.0") is True  # same count, written differently
    assert pearson_symbol_matches(stored, "cF4") is False
    assert pearson_symbol_matches(stored, "cP8") is False  # centring letter is the point
    assert pearson_symbol_matches("cP   1.00", "cF") is False
    # Nothing parseable on the left: compare literally, do not accept.
    assert pearson_symbol_matches("Fm-3m", "cF8") is False
    assert pearson_symbol_matches("Fm-3m", "Fm-3m") is True


def test_filter_by_pearson_and_subfiles_are_conjunctive_with_the_rest():
    """Subfiles match on any overlap; every declared criterion narrows the result."""
    from pdf2_reader import CrystallographicFilter

    card = _card(
        pearson_symbol="cF   8.00",
        subfiles=("I", "A", "FIZ"),
        space_group="Fm-3m",
        cell_a=5.962,
    )

    assert CrystallographicFilter(subfiles=frozenset({"I"})).matches(card) is True
    assert CrystallographicFilter(subfiles=frozenset({"I", "O"})).matches(card) is True
    assert CrystallographicFilter(subfiles=frozenset({"O"})).matches(card) is False
    assert CrystallographicFilter(pearson_symbols=frozenset({"cF"})).matches(card) is True
    assert CrystallographicFilter(pearson_symbols=frozenset({"hP"})).matches(card) is False
    # An empty set is a criterion that admits nothing; "do not narrow" is spelled None,
    # which is what the CLI passes when the flag is absent or empty.
    assert CrystallographicFilter(subfiles=frozenset()).matches(card) is False
    assert CrystallographicFilter(subfiles=None).matches(card) is True
    empty = _card(pearson_symbol="", subfiles=())
    assert CrystallographicFilter(pearson_symbols=frozenset({"cF"})).matches(empty) is False
    assert CrystallographicFilter(subfiles=frozenset({"I"})).matches(empty) is False
    # Conjunction with the criteria that already existed.
    both = CrystallographicFilter(
        cell_ranges={"a": (5.9, 6.0)},
        pearson_symbols=frozenset({"cF"}),
        subfiles=frozenset({"I"}),
    )
    assert both.matches(card) is True
    assert (
        both.__class__(cell_ranges={"a": (7.0, 8.0)}, pearson_symbols=frozenset({"cF"})).matches(
            card
        )
        is False
    )


def _write_synthetic_xsdb(root: Path, records: list[tuple[tuple[float, ...], str]]) -> Path:
    """Write a strongest-lines file in the measured layout and return its directory."""
    winind = root / "Winind"
    winind.mkdir(parents=True, exist_ok=True)
    blob = b""
    for hanawalt, card_class in records:
        body = bytearray(b" " * 159)
        for (low, high), value in zip(((142, 147), (147, 152), (152, 157)), hanawalt):
            body[low:high] = f"{value:<5.2f}"[:5].encode("latin1")
        body[158:159] = card_class.encode("latin1")
        blob += bytes(body)
    (winind / "PDF2XSDB.DAT").write_bytes(blob)
    return root


def test_xsdb_index_drops_the_trailing_duplicate_and_matches_in_the_angular_scale(tmp_path: Path):
    """The first pass is answered from the strongest-lines file, in degrees 2theta.

    Built on synthetic records: a golden dump of real cards would carry licensed content
    into the repository and would break whenever the release changes, while the invariant
    being checked is the layout and the matching rule, both of which are ours.
    """
    from pdf2_reader import StrongestLinesIndex

    def to_2theta(d: object) -> object:
        import numpy as np

        values = np.asarray(d, dtype=float)
        return np.degrees(2.0 * np.arcsin(1.540596 / (2.0 * np.maximum(values, 1e-6))))

    strong = (3.44, 2.98, 2.11)
    other = (2.0, 3.0, 1.0)
    root = _write_synthetic_xsdb(
        tmp_path,
        [(strong, "C"), (strong, "O"), (other, "M"), (other, "M")],
    )

    index = StrongestLinesIndex(root)
    assert index.record_count == 3, "the trailing duplicate must not count as a card"
    assert index.d_lines(0) == pytest.approx(strong)
    assert index.quality_class(1) == "O"

    query = to_2theta([3.44, 2.98])
    assert index.survivors(query, to_2theta, 0.20, min_hits=2) == [0, 1]
    # Two query lines cannot satisfy a three-line requirement, which is why the rule is
    # "two of three": requiring all three collapses as soon as one line is missing.
    assert index.survivors(query, to_2theta, 0.20, min_hits=3) == []
    # A criterion about the card class narrows without any card being read.
    assert index.survivors(query, to_2theta, 0.20, allowed_classes=frozenset({"O"})) == [1]
    # Nothing in the pattern where these cards have lines: no survivors, no false positives.
    assert index.survivors(to_2theta([5.0, 4.5]), to_2theta, 0.20) == []
    # The angular window is respected at its edge, not exceeded.
    assert index.survivors(query + 0.15, to_2theta, 0.20) == [0, 1]
    assert index.survivors(query + 0.25, to_2theta, 0.20) == []


def test_xsdb_index_reports_the_pool_bound_it_applies(tmp_path: Path):
    """A capped pool states the survivor count and the bound instead of truncating silently."""
    from pdf2_reader import StrongestLinesIndex

    def to_2theta(d: object) -> object:
        import numpy as np

        values = np.asarray(d, dtype=float)
        return np.degrees(2.0 * np.arcsin(1.540596 / (2.0 * np.maximum(values, 1e-6))))

    strong = (3.44, 2.98, 2.11)
    # Three records, the last repeating the second, exactly as the shipped file ends:
    # the constructor drops that duplicate, so two cards remain.
    root = _write_synthetic_xsdb(tmp_path, [(strong, "C"), (strong, "C"), (strong, "C")])

    index = StrongestLinesIndex(root)
    query = to_2theta([3.44, 2.98])

    assert len(index.survivors(query, to_2theta, 0.20, max_candidates=5)) == 2
    assert index.survivors(query, to_2theta, 0.20, max_candidates=1) == [0]
