"""Interoperability reader for user-supplied ICDD PDF-2 (ASCII 80-column) installations.

Requires a licensed local PDF-2 installation provided by the user via the
``PDF2_DATA_DIR`` environment variable; no database files are bundled with this
package.

Record layout
-------------
``pdf2.dat``            Flat stream of 80-byte ASCII records.
``Vaxind/offset54.ind`` Unsigned 32-bit little-endian byte offsets; entry ``i``
                        is the byte offset of card ``i`` inside ``pdf2.dat``.
``Winind/PDF2XSDB.DAT`` Fixed-width 159-byte summary records; record ``k``
                        describes card ``k`` (the trailing sentinel duplicates
                        the preceding card and is ignored).

Tag line (columns 71-80 of every record, i.e. ``record[70:80]``)
---------------------------------------------------------------
``[70]`` line counter inside a multi-line group; for record type ``1`` it is a
        flag instead.
``[71]`` card status: ``P`` primary, ``A`` alternate, ``D`` deleted.
``[72:74]`` set number, 2 digits.
``[74:78]`` card number, 4 digits. Together with the set this is the PDF number.
``[78]`` card class code.
``[79]`` record type.

Record types (column 80)
------------------------
``1`` cell parameters ``a b c`` + one angle     ``2`` cell standard uncertainties
``3`` space group, ITA number, Z, Dx, V         ``4`` as ``3`` plus figure of merit
``5`` subfile codes + CAS number                ``6`` mineral / chemical name
``7`` empirical formula (elements space-separated)
``8`` structural formula                        ``9`` bibliography
``A`` Pearson symbol and prototype
``B`` comments and collection reference
``C`` cell transformation matrix 3x3            ``D`` reduced cell
``E`` standardized cell ``a b c alpha beta gamma`` + b/a, c/a
``F`` radiation and wavelength                  ``G`` quality metrics, incl. RIR
``H`` counter                                   ``I`` reflections
``K`` card terminator + dates
``*`` strongest lines, sorted by d              ``+`` strongest lines, Hanawalt order

Reflections (type ``I``)
------------------------
Three reflections per record, 23 characters each:
``d = [0:7]``, ``I = [7:11]``, ``hkl = [11:23]``.

Strong-line lists (types ``*`` and ``+``)
-----------------------------------------
``<prefix><d>/<I-code> <d>/<I-code> ...`` within the 70-character data field.
The intensity code is ``round(I/10)`` with 10 written as ``X`` (Roman ten).
"""

from __future__ import annotations

import os
import re
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from env_file import load_env_file
from loguru import logger
from numpy.typing import NDArray

__all__ = [
    "QUALITY_MARKS",
    "CrystallographicFilter",
    "PDF2Database",
    "PDF2NotFoundError",
    "PDFCard",
    "PDFReflection",
    "StrongestLinesIndex",
    "crystal_system_from_pearson",
    "pearson_symbol_matches",
    "resolve_crystal_system",
    "resolve_pdf2_dir",
]

RECORD_LENGTH = 80

#: Standard quality mark codes indexed 0..10.
QUALITY_MARKS: dict[int, str] = {
    0: "qmNone",
    1: "qmStar",
    2: "qmIndexed",
    3: "qmBlank",
    4: "qmDoubtful",
    5: "qmCalculated",
    6: "qmModelled",
    7: "qmFinal",
    8: "qmRietveld",
    9: "qmOther",
    10: "qmHypothetical",
}

PEARSON_SYSTEMS: dict[str, str] = {
    "a": "triclinic",
    "m": "monoclinic",
    "o": "orthorhombic",
    "t": "tetragonal",
    "h": "trigonal-hexagonal",
    "c": "cubic",
}

_ICSD_FZ = re.compile(r"\b(\d{6})\s+FZ\b")
_STRONG_PAIR = re.compile(r"(\d*\.?\d+)/(X|\d+)")
_HKL_INDEX = re.compile(r"-?\d+")


class PDF2NotFoundError(FileNotFoundError):
    """The PDF-2 installation was not located and no path was supplied."""


@dataclass(frozen=True)
class PDFReflection:
    """One Bragg reflection of a card: d-spacing, relative intensity and hkl."""

    d_spacing: float
    intensity: float
    hkl: tuple[int, ...]


@dataclass(frozen=True)
class PDFCard:
    """A single PDF-2 reference pattern."""

    set_number: int
    card_number: int
    status: str
    quality_class: str
    mineral_name: str = ""
    empirical_formula: str = ""
    structural_formula: str = ""
    space_group: str = ""
    ita_number: int | None = None
    z_value: float | None = None
    cell_a: float | None = None
    cell_b: float | None = None
    cell_c: float | None = None
    cell_alpha: float | None = None
    cell_beta: float | None = None
    cell_gamma: float | None = None
    cell_volume: float | None = None
    pearson_symbol: str = ""
    prototype: str = ""
    subfiles: tuple[str, ...] = ()
    icsd_number: str | None = None
    reflections: tuple[PDFReflection, ...] = ()
    strongest_d: tuple[float, ...] = ()

    @property
    def pdf_number(self) -> str:
        return f"{self.set_number:02d}-{self.card_number:04d}"

    @property
    def crystal_system(self) -> str:
        """Crystal system derived deterministically from ITA number, space group, or Pearson symbol."""
        return resolve_crystal_system(self.ita_number, self.space_group, self.pearson_symbol)

    @property
    def is_deleted(self) -> bool:
        return self.status == "D"


def resolve_crystal_system(
    ita_number: int | None,
    space_group: str | None = None,
    pearson_symbol: str | None = None,
) -> str:
    """Deterministically resolve crystal system from ITA number (1..230), gemmi space group, or Pearson fallback."""
    import gemmi

    sg_num = int(ita_number) if ita_number and 1 <= int(ita_number) <= 230 else 0
    if sg_num <= 0 and space_group:
        sg_obj = gemmi.find_spacegroup_by_name(str(space_group).strip())
        if sg_obj is not None:
            sg_num = int(sg_obj.number)

    if 1 <= sg_num <= 230:
        if sg_num <= 2:
            return "triclinic"
        if sg_num <= 15:
            return "monoclinic"
        if sg_num <= 74:
            return "orthorhombic"
        if sg_num <= 142:
            return "tetragonal"
        if sg_num <= 194:
            return "trigonal-hexagonal"
        return "cubic"

    return crystal_system_from_pearson(pearson_symbol or "")


def crystal_system_from_pearson(pearson_symbol: str) -> str:
    """Map a Pearson symbol's leading letter to its crystal system.

    The Pearson symbol encodes the system as a lower-case letter followed by the
    lattice centring: ``cF 8`` is cubic face-centred, ``mC 132`` monoclinic
    C-centred. Unknown or absent symbols map to ``unknown`` rather than being
    guessed.
    """
    text = (pearson_symbol or "").strip()
    if not text:
        return "unknown"
    return PEARSON_SYSTEMS.get(text[0].lower(), "unknown")


#: A Pearson symbol as stored and as asked for: a system letter (lower case), a centring
#: letter (upper case) and an atom count that may carry decimals, as in ``cF   8.00``.
_PEARSON_PATTERN = re.compile(r"^([a-z]?[A-Z])(\d+(?:\.\d+)?)?$")


def pearson_symbol_matches(stored: str, requested: str) -> bool:
    """Compare two Pearson symbols, treating an absent atom count as "any".

    The database stores ``cF   8.00`` for a face-centred cubic cell with eight atoms, while
    a caller writes ``cF8``. The letters are compared exactly -- in this notation the system
    letter is lower case and the centring letter upper case, so the comparison is
    case-sensitive on purpose -- and the count only when the request states one.

    Unparseable input falls back to a literal comparison rather than being accepted: a
    criterion nobody can evaluate is not a criterion that passes.
    """
    stored_text = stored.replace(" ", "")
    requested_text = requested.replace(" ", "")
    stored_parts = _PEARSON_PATTERN.match(stored_text)
    requested_parts = _PEARSON_PATTERN.match(requested_text)
    if stored_parts is None or requested_parts is None:
        return stored_text == requested_text
    if stored_parts.group(1) != requested_parts.group(1):
        return False
    wanted = requested_parts.group(2)
    if wanted is None:
        return True
    have = stored_parts.group(2)
    return have is not None and abs(float(have) - float(wanted)) < 1e-9


@dataclass(frozen=True)
class CrystallographicFilter:
    """Restrictions on cell metrics, space group, crystal system, Pearson symbol, subfiles.

    Restricts candidate cards by unit-cell parameter ranges, space groups,
    crystal systems, Pearson symbols, and subfile codes. Every criterion is
    optional; omitted criteria are not applied. Ranges are inclusive bounds,
    ``None`` meaning unbounded.

    ``subfiles`` matches a card that belongs to **any** of the named subfiles, because one
    card may be listed in several at once and the caller is stating which of them it
    accepts. ``pearson_symbols`` compares the symbol itself, which the crystal system alone
    does not determine: ``cF8`` and ``cP1`` are both cubic, and telling a face-centred cell
    from a primitive one is often the whole point of asking.
    """

    cell_ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    space_groups: frozenset[str] | None = None
    crystal_systems: frozenset[str] | None = None
    pearson_symbols: frozenset[str] | None = None
    subfiles: frozenset[str] | None = None

    def matches(self, card: PDFCard) -> bool:
        """True when the card satisfies every declared criterion (logical AND)."""
        for axis, (lo, hi) in self.cell_ranges.items():
            value = getattr(card, f"cell_{axis}", None)
            if value is None:
                return False
            if lo is not None and value < lo:
                return False
            if hi is not None and value > hi:
                return False

        if self.space_groups is not None:
            if not card.space_group:
                return False
            if card.space_group.replace(" ", "") not in {
                s.replace(" ", "") for s in self.space_groups
            }:
                return False

        if self.crystal_systems is not None and card.crystal_system not in self.crystal_systems:
            return False

        if self.pearson_symbols is not None:
            if not card.pearson_symbol:
                return False
            if not any(
                pearson_symbol_matches(card.pearson_symbol, wanted)
                for wanted in self.pearson_symbols
            ):
                return False

        if self.subfiles is not None:
            if not card.subfiles:
                return False
            if not (self.subfiles & set(card.subfiles)):
                return False

        return True


def resolve_pdf2_dir() -> Path:
    """Return the PDF-2 installation directory declared by the configuration.

    The location is configuration, not a per-call argument: it is read from the
    ``PDF2_DATA_DIR`` environment variable, which may be exported by the caller or
    declared in the ``.env`` file of the working directory. There is deliberately
    no constructor parameter and no CLI flag. An installation addressable three
    ways -- argument, variable, file -- is three addresses, and the reader would
    then answer differently depending on which one a caller happened to use.

    :raises PDF2NotFoundError: when the variable is unset or does not name a
        directory.
    """
    load_env_file()
    env = os.environ.get("PDF2_DATA_DIR")
    if not env:
        raise PDF2NotFoundError(
            "PDF-2 installation not found: PDF2_DATA_DIR is unset. Point it at the "
            "installation directory (holding pdf2.dat and Vaxind/offset54.ind), by "
            "exporting the variable or declaring it in .env."
        )
    resolved = Path(env)
    if not resolved.is_dir():
        raise PDF2NotFoundError(
            f"PDF2_DATA_DIR names {resolved}, which is not an existing directory"
        )
    return resolved


class PDF2Database:
    """Read-only access to a local PDF-2 installation.

    The location comes from ``resolve_pdf2_dir()`` and nowhere else; there is
    deliberately no built-in default, since a hard-coded path would make the tool
    depend on one machine's layout and would silently break everywhere else. When
    the variable is unset the reader raises ``PDF2NotFoundError`` and every PDF-2
    path in the skill stays silent.
    """

    def __init__(self) -> None:
        resolved = resolve_pdf2_dir()

        self.db_dir = resolved
        self.pdf2_path = self.db_dir / "pdf2.dat"
        self.offset_path = self.db_dir / "Vaxind" / "offset54.ind"

        for required in (self.pdf2_path, self.offset_path):
            if not required.is_file():
                raise PDF2NotFoundError(f"Required PDF-2 file is missing: {required}")

        self._offsets = self._load_offsets()
        # The handle is owned by the instance and released by close()/__exit__().
        # Opening it per read would defeat the point of O(1) offset-table seeks.
        # Suppressed rule: SIM115.
        self._handle = open(self.pdf2_path, "rb")  # noqa: SIM115

    def _load_offsets(self) -> list[int]:
        raw = self.offset_path.read_bytes()
        if len(raw) % 4:
            raise ValueError(f"offset54.ind has a truncated tail: {len(raw)} bytes")
        count = len(raw) // 4
        return list(struct.unpack(f"<{count}I", raw))

    @property
    def card_count(self) -> int:
        return len(self._offsets)

    def index_of(self, pdf_number: str) -> int:
        """Return the position of a card in the release, addressed by its PDF number.

        Scans only the 10-byte tag header of each card offset entry.

        :raises ValueError: when the argument is not a PDF number, or the release
            holds no such card.
        """
        set_text, separator, card_text = str(pdf_number).partition("-")  # noqa: PU001
        if not separator or not set_text.isdigit() or not card_text.isdigit():
            raise ValueError(f"not a PDF number: {pdf_number!r}; expected the form '26-1479'")
        wanted = f"{int(set_text):02d}{int(card_text):04d}"

        for index, offset in enumerate(self._offsets):
            self._handle.seek(offset + 70)
            tag = self._handle.read(10).decode("latin1")
            if tag[2:8] == wanted:
                return index
        raise ValueError(f"card {pdf_number} is absent from this release")

    def close(self) -> None:
        self._handle.close()

    # Return type is annotated with the class name directly rather than importing Self.
    # Suppressed rule: PYI034.
    def __enter__(self) -> PDF2Database:  # noqa: PYI034
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _read_records(self, index: int) -> list[bytes]:
        """Read every 80-byte record of one card, stopping on the ``K`` terminator."""
        self._handle.seek(self._offsets[index])
        records: list[bytes] = []
        while True:
            chunk = self._handle.read(RECORD_LENGTH)
            if len(chunk) < RECORD_LENGTH:
                break
            records.append(chunk)
            if chunk[79:80] == b"K":
                break
            if len(records) > 400:
                logger.warning("Card {} exceeded 400 records; stopping early", index)
                break
        return records

    def get_card(self, index: int) -> PDFCard:
        """Parse one card by its zero-based position in the release."""
        if not 0 <= index < len(self._offsets):
            raise IndexError(f"card index {index} out of range 0..{len(self._offsets) - 1}")

        records = self._read_records(index)
        if not records:
            raise ValueError(f"card index {index} produced no records")

        tag = records[0][70:80].decode("latin1")
        set_number = int(tag[2:4])
        card_number = int(tag[4:8])
        status = tag[1]
        quality_class = tag[8]

        fields: dict[str, list[str]] = {}
        reflections: list[PDFReflection] = []
        strongest_d: list[float] = []
        icsd_number: str | None = None

        for record in records:
            record_type = record[79:80].decode("latin1")
            body = record[:70].decode("latin1")
            fields.setdefault(record_type, []).append(body)

            if record_type == "I":
                for slot in range(3):
                    piece = body[slot * 23 : (slot + 1) * 23]
                    if not piece.strip():
                        continue
                    try:
                        d_value = float(piece[0:7].strip().rstrip("*"))
                    except ValueError:
                        continue
                    if d_value <= 0:
                        continue
                    # Intensity codes such as "999*" are markers, not numbers.
                    try:
                        inten = float(piece[7:11].strip().rstrip("*"))
                    except ValueError:
                        inten = 0.0
                    hkl_text = piece[11:23]
                    # The field ends with an optional flag (e.g. "  1  1  1C  "),
                    # so only the first three signed integers are the indices.
                    indices = [int(v) for v in _HKL_INDEX.findall(hkl_text)[:3]]
                    reflections.append(
                        PDFReflection(
                            d_spacing=d_value,
                            intensity=inten,
                            hkl=tuple(indices),
                        )
                    )
            elif record_type == "+" and not strongest_d:
                # The prefix is 0-2 letters and can be glued to the first pair.
                strongest_d = [float(d) for d, _ in _STRONG_PAIR.findall(body)[:3]]
            elif record_type == "B" and icsd_number is None:
                match = _ICSD_FZ.search(body)
                if match:
                    icsd_number = match.group(1)

        cell = self._parse_standard_cell(fields.get("E", [""])[0])
        space_group, ita_number, z_value = self._parse_symmetry(  # noqa: PU001
            fields.get("3", [""])[0]
        )

        return PDFCard(
            set_number=set_number,
            card_number=card_number,
            status=status,
            quality_class=quality_class,
            mineral_name=fields.get("6", [""])[0].strip(),
            empirical_formula=fields.get("7", [""])[0].strip(),
            structural_formula=fields.get("8", [""])[0].strip(),
            space_group=space_group,
            ita_number=ita_number,
            z_value=z_value,
            cell_a=cell[0],
            cell_b=cell[1],
            cell_c=cell[2],
            cell_alpha=cell[3],
            cell_beta=cell[4],
            cell_gamma=cell[5],
            cell_volume=cell[6],
            pearson_symbol=self._parse_pearson(fields.get("A", [""])[0]),
            prototype=self._parse_prototype(fields.get("A", [""])[0]),
            subfiles=tuple(fields.get("5", [""])[0].split()),
            icsd_number=icsd_number,
            reflections=tuple(reflections),
            strongest_d=tuple(strongest_d),
        )

    @staticmethod
    def _parse_symmetry(body: str) -> tuple[str, int | None, float | None]:
        """Type ``3``: space-group symbol, ITA number, Z, Dx, V.

        Parsed by tokens rather than fixed columns because some records carry a
        setting suffix on the ITA number (e.g. ``15C``) while others carry a
        plain integer (``225``).
        """
        tokens = body.split()
        if not tokens:
            return "", None, None
        symbol = tokens[0]

        ita_number: int | None = None
        if len(tokens) > 1:
            digits = re.match(r"(\d+)", tokens[1])
            if digits:
                val = int(digits.group(1))
                if 1 <= val <= 230:
                    ita_number = val

        z_value: float | None = None
        if len(tokens) > 2:
            try:
                z_value = float(tokens[2])
            except ValueError:
                z_value = None

        return symbol, ita_number, z_value

    @staticmethod
    def _parse_standard_cell(body: str) -> tuple[float | None, ...]:
        """Type ``E``: ``a b c alpha beta gamma V b/a c/a``.

        Only the first seven tokens are the standardized cell; the last two are
        axis ratios (a/b and c/b) and are not part of the cell.
        """
        values: list[float | None] = []
        for token in body.split()[:7]:
            try:
                values.append(float(token))
            except ValueError:
                values.append(None)
        values.extend([None] * (7 - len(values)))
        return tuple(values[:7])

    @staticmethod
    def _parse_pearson(body: str) -> str:
        return body[0:8].strip()

    @staticmethod
    def _parse_prototype(body: str) -> str:
        parts = body.split("\\")
        return parts[1].strip() if len(parts) >= 3 else ""

    def iter_cards(self) -> Iterator[PDFCard]:
        """Yield every card in release order. Deleted cards are included."""
        for index in range(self.card_count):
            try:
                yield self.get_card(index)
            except (ValueError, IndexError) as exc:
                logger.warning("Skipping card {}: {}", index, exc)

    def filter_cards(
        self,
        crystal_filter: CrystallographicFilter | None = None,
        include_deleted: bool = False,
    ) -> list[PDFCard]:
        """Return cards satisfying the crystallographic restrictions."""
        accepted: list[PDFCard] = []
        for card in self.iter_cards():
            if card.is_deleted and not include_deleted:
                continue
            if crystal_filter is not None and not crystal_filter.matches(card):
                continue
            accepted.append(card)
        return accepted


#: ``Winind/PDF2XSDB.DAT`` holds one 159-byte summary record per card:
#: ``id[0:6]``, ``name[6:74]``, ``formula[74:141]``, the three Hanawalt
#: d-values at ``[142:147]``, ``[147:152]``, ``[152:157]``, and the card class at ``[158]``.
XSDB_RECORD_LENGTH = 159
XSDB_HANAWALT_COLUMNS = ((142, 147), (147, 152), (152, 157))
XSDB_CLASS_COLUMN = 158


class StrongestLinesIndex:
    """First-pass index of the strongest lines of every card, held in memory.

    Reaching the strongest lines through ``pdf2.dat`` costs a full pass over the card
    stream, so a first pass built that way would scale with the size of the database
    rather than with the size of the answer. The shipped ``Winind/PDF2XSDB.DAT`` carries
    the same three Hanawalt lines per card in one flat file, which makes the whole first
    pass a single read.

    The index answers one question -- which cards have at least ``min_hits`` of their
    strongest lines where the measured pattern has peaks -- and it answers it in the
    angular scale. That is not a detail: an angular window of +-0.20 deg spans a d window
    that varies fortyfold across the usual range (0.0566 A at 17.7 deg against 0.0014 A
    at 100.8 deg), so a fixed d tolerance either loses completeness where d is small or
    admits thousands of false candidates where d is large.

    This is a filter and not an identification: surviving means "worth scoring", and
    nothing here decides which phase is present.
    """

    def __init__(self, data_dir: Path) -> None:
        path = data_dir / "Winind" / "PDF2XSDB.DAT"
        if not path.is_file():
            raise PDF2NotFoundError(f"Required PDF-2 file is missing: {path}")
        raw = path.read_bytes()
        if len(raw) % XSDB_RECORD_LENGTH:
            raise ValueError(
                f"PDF2XSDB.DAT has a truncated tail: {len(raw)} bytes is not a multiple "
                f"of {XSDB_RECORD_LENGTH}"
            )
        self.path = path
        records = len(raw) // XSDB_RECORD_LENGTH
        # The last record repeats the preceding card, exactly as the card stream does, so
        # it is dropped here rather than by every caller in turn.
        self._records = records - 1 if records > 1 else records

        lines = np.full((self._records, len(XSDB_HANAWALT_COLUMNS)), np.nan, dtype=float)
        classes: list[str] = []
        for index in range(self._records):
            start = index * XSDB_RECORD_LENGTH
            record = raw[start : start + XSDB_RECORD_LENGTH]
            for column, (low, high) in enumerate(XSDB_HANAWALT_COLUMNS):
                text = record[low:high].decode("latin1").strip()
                # A d-value is sometimes written as "279." with no fractional part.
                try:
                    value = float(text)
                except ValueError:
                    continue
                if value > 0:
                    lines[index, column] = value
            classes.append(record[XSDB_CLASS_COLUMN : XSDB_CLASS_COLUMN + 1].decode("latin1"))
        self._d = lines
        self._classes = classes

    @property
    def record_count(self) -> int:
        """Number of cards described, the trailing duplicate already excluded."""
        return self._records

    def d_lines(self, index: int) -> tuple[float, ...]:
        """Return the strongest lines of one card, unreadable ones omitted."""
        row = self._d[index]
        return tuple(float(value) for value in row if np.isfinite(value))

    def quality_class(self, index: int) -> str:
        """Return the card's class column, whose semantics are only partly established.

        The shipped release carries eight values (``O X M C T H R A``). Only ``C`` is
        established as "calculated"; a policy of admitting or rejecting the others cannot
        be derived from the data, so this accessor reports the value and leaves the policy
        to the caller's configuration.
        """
        return self._classes[index]

    def survivors(
        self,
        query_two_theta: NDArray[Any],
        to_two_theta: Callable[[NDArray[Any]], NDArray[Any]],
        tolerance_deg: float,
        min_hits: int = 2,
        allowed_classes: frozenset[str] | None = None,
        max_candidates: int | None = None,
    ) -> list[int]:
        """Return the positions of cards that deserve scoring, in card order.

        ``query_two_theta`` holds the measured peak positions to test against, and
        ``to_two_theta`` converts d-spacings to angles: the wavelength belongs to the
        engine, so it is passed in rather than assumed here, and the Bragg relation stays
        stated in one place.

        ``allowed_classes`` defaults to None, which admits every card. The class
        vocabulary is only partly understood, so no rejection policy is imposed by this
        module; a caller that has a policy passes it.

        ``max_candidates`` bounds the pool. Both the total survivor count and the cap are
        logged when truncation occurs. Cards are ordered by number of matched lines with
        the card index as a deterministic tie-breaker.
        """
        query = np.sort(np.asarray(query_two_theta, dtype=float))
        if query.size == 0:
            return []

        angles = np.asarray(to_two_theta(self._d), dtype=float)
        hits = np.zeros(self._records, dtype=np.int32)
        for column in range(angles.shape[1]):
            positions = angles[:, column]
            valid = np.isfinite(positions)
            if not bool(valid.any()):
                continue
            measured = positions[valid]
            where = np.searchsorted(query, measured)
            left = np.clip(where - 1, 0, query.size - 1)
            right = np.clip(where, 0, query.size - 1)
            distance = np.minimum(
                np.abs(query[left] - measured),
                np.abs(query[right] - measured),
            )
            hits[valid] += (distance <= tolerance_deg).astype(np.int32)

        accepted = hits >= min_hits
        if allowed_classes is not None:
            accepted &= np.array(
                [quality in allowed_classes for quality in self._classes], dtype=bool
            )
        positions = np.flatnonzero(accepted)

        if max_candidates is not None and positions.size > max_candidates:
            order = np.lexsort((positions, -hits[positions]))
            kept = positions[order[:max_candidates]]
            logger.warning(
                "PDF-2 first pass left {} cards; keeping the {} with the most matched "
                "lines. A narrower element or class criterion shows the rest.",
                positions.size,
                max_candidates,
            )
            return sorted(int(index) for index in kept)

        return [int(index) for index in positions]
