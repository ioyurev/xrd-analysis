"""Query bridge for the Materials Project route, executed in a second environment.

The Materials Project client cannot share an interpreter with the crystallographic
engine. ``mp_api.client`` imports pyarrow; pyarrow requires NumPy >= 2; the engine is
pinned below 2 because its GSAS-II binaries are built against the NumPy 1.x ABI. Both
ways of forcing them together were measured and both fail: under numpy 2.4.6 the client
does import while twelve GSAS-II workbench tests fail and eight more start skipping, and
pinning ``pyarrow<17`` instead drags mp-api down to 0.45.13, which mismatches
``emmet-core`` 0.87.3.

So the two requirements are kept in two environments, and this module is the bridge. It
is the only file in the skill that imports ``mp_api.client``, and it is executed by the
interpreter named in the ``MP_PYTHON`` variable -- never by the one running the rest of
the skill. The request arrives as JSON on stdin and the answer leaves as JSON on stdout;
the API key travels that way as well rather than in argv, because a command line is
readable by every process on the machine.

Nothing here imports the rest of the skill. The modules beside this file are imported
under the engine's NumPy 1.x pin, while this bridge has to run under NumPy 2, so sharing
them would rebuild the conflict the split exists to remove. That also means the element
filter is resolved by the caller and arrives here as plain arguments: this module must not
carry a second copy of the chemistry.

Run under the retrieval interpreter::

    echo '{"action": "search", "api_key": "...", "mode": "all", "elements": ["Sm", "S"]}' \\
        | <mp_python> mp_query.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

#: Fields requested from the Materials Project summary endpoint. Names must match the
#: live API; they are read generically below.
COMMON_FIELDS: list[str] = [
    "material_id",
    "formula_pretty",
    "symmetry",
    "energy_above_hull",
    "is_stable",
    "volume",
    "structure",
]


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a document that may be a pydantic model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _number(value: Any) -> float | None:
    """Convert a value to float, tolerating the JSON-ish spellings of the API."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def serialize_document(doc: Any) -> dict[str, Any]:
    """Reduce one live summary document to the record shape the caller caches.

    The caller reads records through a tolerant field reader that accepts both attribute
    holders and dicts, and its search cache already stores exactly this shape. Doing the
    reduction here, where the live document exists, keeps one definition of what a record
    is instead of a copy in each environment.
    """
    symmetry = _field(doc, "symmetry", None)
    symbol = ""
    if symmetry:
        symbol = str(_field(symmetry, "symbol", "") or "")

    structure = _field(doc, "structure", None)
    lattice = _field(structure, "lattice", None) if structure is not None else None
    lattice_dict = None
    if lattice is not None:
        lattice_dict = {
            axis: _number(_field(lattice, axis, None))
            for axis in ("a", "b", "c", "alpha", "beta", "gamma")
        }

    return {
        "material_id": str(_field(doc, "material_id", "")),
        "formula_pretty": str(_field(doc, "formula_pretty", "")),
        "symmetry": {"symbol": symbol},
        "volume": _number(_field(doc, "volume", None)),
        "energy_above_hull": _number(_field(doc, "energy_above_hull", 0.0)),
        "is_stable": bool(_field(doc, "is_stable", False)),
        "structure": {"lattice": lattice_dict} if lattice_dict else None,
    }


def _run_with_deadline(fn: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Run one bridge request with a wall-clock limit.

    mp-api issues its HTTP calls without a read timeout, so a stalled connection hangs
    the bridge forever and the caller cannot tell that from a slow query. The limit is
    configuration (``MP_QUERY_TIMEOUT_S``, default 60 s); on expiry the process reports
    the transport obstacle by name and exits with code 3, which the parent turns into a
    named warning instead of a silent empty answer.
    """
    import os
    import threading

    deadline_s = float(os.environ.get("MP_QUERY_TIMEOUT_S", "60"))
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["answer"] = fn(request)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(deadline_s)
    if thread.is_alive():
        print(
            f"mp_query: transport: no answer within {deadline_s:g} s "
            "(stalled connection, not an empty result)",
            file=sys.stderr,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    if "error" in box:
        raise box["error"]
    answer: dict[str, Any] = box["answer"]
    return answer


def search(request: dict[str, Any]) -> dict[str, Any]:
    """Run one summary search and return its records.

    ``mode`` names the query the caller resolved from its own element filter: ``strict``
    searches a fixed chemical system, ``all`` requires every listed element, ``any``
    requires at least one and is issued as one query per element with the results merged.
    """
    from mp_api.client import MPRester

    mode = request["mode"]
    api_key = request["api_key"]
    exclude = request.get("exclude_elements") or None

    documents: list[Any] = []
    with MPRester(api_key=api_key, mute_progress_bars=True) as mpr:
        if mode == "strict":
            documents = mpr.materials.summary.search(
                chemsys=request["chemsys"], fields=COMMON_FIELDS
            )
        elif mode == "all":
            kwargs: dict[str, Any] = {
                "elements": list(request["elements"]),
                "fields": COMMON_FIELDS,
            }
            if exclude:
                kwargs["exclude_elements"] = exclude
            documents = mpr.materials.summary.search(**kwargs)
        elif mode == "any":
            seen: set[str] = set()
            for element in request["any_elements"]:
                kwargs = {"elements": [element], "fields": COMMON_FIELDS}
                if exclude:
                    kwargs["exclude_elements"] = exclude
                for doc in mpr.materials.summary.search(**kwargs):
                    material_id = str(_field(doc, "material_id", "") or "")
                    if material_id and material_id not in seen:
                        seen.add(material_id)
                        documents.append(doc)
        else:
            raise ValueError(f"unknown search mode {mode!r}; expected 'strict', 'all' or 'any'")

    return {"records": [serialize_document(doc) for doc in documents]}


def download(request: dict[str, Any]) -> dict[str, Any]:
    """Fetch one structure and return it as CIF text.

    The text, not a path: the caller owns its cache directory, and a bridge that wrote
    files would put the cache in whichever directory the second interpreter happened to
    run in. Oxidation states are removed and the cell is standardized exactly as the
    caller's cache expects.
    """
    from mp_api.client import MPRester
    from pymatgen.io.cif import CifWriter

    material_id = str(request["material_id"])
    symprec = float(request.get("symprec", 0.01))

    with MPRester(api_key=request["api_key"], mute_progress_bars=True) as mpr:
        structure = mpr.get_structure_by_material_id(material_id, conventional_unit_cell=True)

    structure.remove_oxidation_states()
    return {
        "cif_text": str(CifWriter(structure, symprec=symprec)),
        "reduced_formula": structure.composition.reduced_formula,
    }


def main() -> int:
    """Read one request from stdin, answer on stdout, report failures on stderr."""
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"mp_query: request is not valid JSON: {exc}", file=sys.stderr)
        return 2

    action = request.get("action")
    try:
        if action == "search":
            answer = _run_with_deadline(search, request)
        elif action == "download":
            answer = _run_with_deadline(download, request)
        else:
            print(f"mp_query: unknown action {action!r}", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001 - a bridge reports, the parent decides
        # A third-party failure (network, key, API drift) is reported as a message and a
        # nonzero code. The parent turns it into its own honest warning; a traceback from
        # inside this process would read as a defect of the caller's environment.
        print(f"mp_query: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    json.dump(answer, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
