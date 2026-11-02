"""Crystallographic workbench library (xrd_project.py).

Вход для человека — `xrd_sample.py`: он подставляет пути и имена файлов.
Этот модуль остаётся библиотекой команд без собственного CLI.

Provides an atomic command-line interface for powder diffraction analysis:
- init: Initialize GSAS-II project from experimental pattern with canonical optics in memory.
- add-phase: Import crystal structure CIF into project.
- remove-phase: Safely delete phase from project.
- toggle-phase: Enable or disable phase scale and refinement without deletion.
- status: Inspect project state, exact agreement indices (Rwp, Rp, GoF, DW), QPA, and cells.
- step: Perform one Stage-Gate refinement step with disk-level rollback safety net.
- diagnose: Classify residual peaks and match unindexed reflections against candidate CIFs.
- export: Print the IUCr report to stdout and write refined CIFs, profile CSVs and the plot
  (the machine-readable summary is returned to the caller, not written).
"""

from __future__ import annotations

import io
import json
import math
import os
import platform
import re
import shutil
import sys

# Принудительная инициализация UTF-8 для безопасного вывода греческих символов (Δ, σ, Å) на Windows
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from loguru import logger
from pymatgen.core import Composition, Element, Structure

# Add scripts directory to path for sibling imports
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from constants import PHASE_MICRO_CELL_LOCK_PERCENT, RULE9_SHIFT_SIGMA_RATIO
from gsas2_adapter import (
    KEEP_LISTING_ENV,
    build_scaled_phase_bragg_peaks,
    calculate_profile_metrics,
    discard_gsas_listing,
    ensure_fresh_residuals,
    extract_bragg_reflections,
    extract_cell_parameters,
    extract_phase_scales,
    extract_profile_curves,
    g2_safe_errors,
    get_project_qpa,
    native_phase_r_factors,
    phase_rb_percent,
    phase_rf_percent,
    sync_pawley_reflections_to_hist,
)
from instrument import CanonicalInstrument
from peaks import extract_peaks_robust
from search_match import (
    CandidateScoringEngine,
    ProfileProbabilityField,
    build_element_filter,
    format_candidate_table,
    load_hsrdb_candidates,
    load_pdf2_candidates,
    select_candidates,
)
from stage_gate_rietveld import (
    CANONICAL_STAGES,
    PASS_CONTINUE,
    PASS_EXHAUSTED,
    PASS_PLATEAU,
    PASS_REGRESSED,
    G2AtomRecord,
    PassDecision,
    RefinementStatus,
    StageGateRietveld,
    StageId,
    apply_box_constraints,
    build_stage_recipe,
    clamp_stored_parameters,
    decide_next_pass,
    ensure_gsas2_environment,
    instrumental_lower_bound_report,
    normalize_stage_name,
    reset_stage_flags,
    setup_phase_dist_restraints,
    shift_significance_problems,
)
from xrd_diagnose import (
    diagnose_residuals,
    format_diagnostic_report,
)


def identified_elements(gpx: Any) -> set[str]:
    """Return the element symbols present in the phases already accepted by the model.

    GSAS-II keeps no ``Formula`` in ``General``; the element of every atom sits in the
    ``Atoms`` table at the column named by ``AtomPtrs[1]``. The set is read from the
    phases themselves, not from a table of expected chemistry: a series can contain
    gadolinium in one sample and samarium only in another.
    """
    elements: set[str] = set()
    for phase in gpx.phases():
        atoms = phase.data.get("Atoms") or []
        ptrs = phase.data.get("General", {}).get("AtomPtrs") or []
        type_column = int(ptrs[1]) if len(ptrs) > 1 else 1
        for row in atoms:
            if len(row) <= type_column:
                continue
            found = re.match(r"[A-Z][a-z]?", str(row[type_column]) or "")
            if found:
                elements.add(found.group(0))
    return elements


def partition_candidates(
    matches: list[dict[str, Any]], known_elements: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split scored candidates into chemically possible and new-element ones.

    Scoring ranks by line positions and intensities and is blind to composition: a
    phase carrying gadolinium takes first place in a samarium-sulfide sample. The pool
    must still cover every thermodynamically possible product, so nothing is dropped --
    candidates that introduce an element absent from the accepted phases are returned
    separately, with the element list attached.
    """
    if not known_elements:
        return matches, []
    plausible: list[dict[str, Any]] = []
    novel: list[dict[str, Any]] = []
    for match in matches:
        with suppress(ValueError, TypeError, KeyError, AttributeError):
            match["elements"] = sorted(
                str(el.symbol) for el in Composition(str(match.get("formula") or "")).elements
            )
        match_elements = set(match.get("elements") or [])
        if match_elements and not match_elements <= known_elements:
            novel.append(match)
        else:
            plausible.append(match)
    return plausible, novel


#: Допустимый уход параметра решётки за одну стадию. Выше — это уже другая фаза.
CELL_DRIFT_LIMIT = 0.05


def _refresh_saved_model(gpx: Any) -> None:
    """Пересчитывает Ycalc перед сохранением, чтобы метрики описывали модель.

    После add/remove/toggle фазы сохранённые невязки относятся к прежнему
    состоянию: status, diagnose и export читают их и показывают числа, которых
    в проекте уже нет.
    """
    # SystemExit тоже глушится: при отсутствии свободных параметров GSAS-II
    # завершает процесс, а не возвращает ошибку, и без этого мутация фазы
    # обрывала бы команду уже после успешного изменения модели.
    phases = gpx.phases() if hasattr(gpx, "phases") else []
    if not phases:
        # После удаления последней фазы уточнять нечего, и GSAS-II падает на
        # пустом списке фаз: пересчёт здесь не нужен и не должен ломать команду.
        return
    with suppress(
        SystemExit, RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError
    ):
        hists = gpx.histograms()
        if hists:
            ensure_fresh_residuals(gpx, hists[0], force=True)


def get_sidecar_history_path(project_path: Path) -> Path:
    """Return companion JSON sidecar path for a .gpx project file."""
    return project_path.parent / f"{project_path.stem}_history.json"


def get_checkpoint_path(project_path: Path) -> Path:
    """Return physical disk backup checkpoint path for rollback."""
    return project_path.parent / f"{project_path.stem}_checkpoint.gpx"


#: How many pre-mutation snapshots the checkpoint stack keeps.
MAX_CHECKPOINTS = 5


def get_checkpoint_stack_path(project_path: Path) -> Path:
    """Return the sidecar listing the checkpoint stack of a project."""
    return project_path.parent / f"{project_path.stem}_ckpt.json"


def push_checkpoint(project_path: Path, label: str = "") -> Path | None:
    """Snapshot the project before a mutating command, keeping a bounded stack.

    ``step`` and ``params refine`` keep their own single ``_checkpoint.gpx`` for
    the watchdog rollback, but that file is written *after* a trial phase has
    been added and therefore cannot undo the addition: removing the phase leaves
    the other scales and the background in the local minimum the trial reached.
    The stack below is taken before the mutation, which is what makes a trial
    reversible (add -> refine -> revert).
    """
    if not project_path.exists():
        return None

    stack_file = get_checkpoint_stack_path(project_path)
    try:
        entries = json.loads(stack_file.read_text(encoding="utf-8")) if stack_file.exists() else []
    except (OSError, json.JSONDecodeError):
        entries = []

    index = max((int(entry.get("index", 0)) for entry in entries), default=0) + 1
    snapshot = project_path.parent / f"{project_path.stem}_ckpt_{index:03d}.gpx"
    shutil.copyfile(str(project_path), str(snapshot))
    entries.append(
        {
            "index": index,
            "file": snapshot.name,
            "label": label,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )

    while len(entries) > MAX_CHECKPOINTS:
        dropped = entries.pop(0)
        with suppress(OSError):
            (project_path.parent / str(dropped.get("file", ""))).unlink()

    try:
        stack_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not write checkpoint stack {stack_file}: {exc}")

    logger.info(f"Checkpoint {index} pushed ({label or 'no label'}) before mutation")
    return snapshot


def load_sidecar_history(project_path: Path) -> dict[str, Any]:
    """Safely load sidecar history JSON, returning empty structure if missing (SSOT rule)."""
    p = get_sidecar_history_path(project_path)
    if not p.exists():
        return {
            "project_file": str(project_path.resolve()),
            "created_at": datetime.now(UTC).isoformat(),
            "stages_completed": [],
            "iterations": [],
            "phases": {},
        }
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e_hist:
        logger.warning(f"Could not read sidecar history {p}: {e_hist}")
        return {
            "project_file": str(project_path.resolve()),
            "created_at": datetime.now(UTC).isoformat(),
            "stages_completed": [],
            "iterations": [],
            "phases": {},
        }


def describe_engine() -> dict[str, str]:
    """Возвращает провенанс движка: ревизию GSAS-II и версию интерпретатора.

    Сведения берутся из служебного файла установщика GSAS-II
    (``GSASII/git_verinfo.py``), а не из ``git`` рабочего каталога: файл
    записывается при установке и остаётся верным для тех прогонов, которые
    свершены до следующего обновления. Если файла нет, возвращается
    ``"unknown"`` — отсутствие сведений честнее выдуманного значения.
    """
    provenance = {"python_version": platform.python_version()}
    try:
        from GSASII import git_verinfo as gv
    except ImportError:
        gv = None
    provenance["gsas_commit"] = str(getattr(gv, "git_version", "unknown"))
    provenance["gsas_versiontag"] = str(getattr(gv, "git_versiontag", "unknown"))
    return provenance


def save_sidecar_history(project_path: Path, data: dict[str, Any]) -> None:
    """Safely write sidecar history JSON.

    Каждая запись сопровождается провенансом движка: без него после
    обновления GSAS-II невозможно ответить, какой версией сделан проект.
    """
    p = get_sidecar_history_path(project_path)
    data.update(describe_engine())
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e_hist:
        logger.warning(f"Could not write sidecar history {p}: {e_hist}")


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with ``None`` so that a payload stays valid JSON.

    ``json.dumps`` serializes non-finite floats as the bare tokens ``NaN``,
    ``Infinity`` and ``-Infinity``. Python's own :func:`json.loads` accepts them,
    but strict consumers do not — PowerShell's ``ConvertFrom-Json`` fails with
    "Error parsing NaN value". Non-finite values reach these payloads from
    partially refined projects, where an unrefined parameter has no e.s.d.
    """
    if isinstance(obj, dict):
        return {key: _json_safe(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def print_json(payload: Any) -> None:
    """Write a machine-readable payload to the real stdout as strict JSON.

    ``allow_nan=False`` makes any non-finite value that escaped :func:`_json_safe`
    fail loudly instead of emitting unparsable output. Writes bypass ``sys.stdout``
    because ``--json`` commands redirect it; see :func:`engine_output_to_stderr`.
    """
    text = json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
    sys.__stdout__.write(text + "\n")
    sys.__stdout__.flush()


@contextmanager
def engine_output_to_stderr():
    """Route everything written to stdout (GSAS-II included) into stderr.

    GSAS-II writes notes with plain ``print``, bypassing its own ``G2Print``
    verbosity level: ``GSASIIpath`` reports a missing configuration as
    ``N.B. Configuration file <path> does not exist``. For ``--json`` commands that
    line would land in front of the payload and make it unparsable, so stdout is
    exchanged with stderr for the duration of the command.
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout


#: Строки вывода движка, которые стоит поднять агенту: предупреждения и ошибки.
_ENGINE_NOTE_PATTERN = re.compile(
    r"\b(warn|warning|error|fail(?:ed|ure)?|diverg\w*|singular|exception|traceback)\b",
    re.IGNORECASE,
)

#: RF фазы, выше которого расхождение внутри фазы считается аномалией модели.
PHASE_RF_WARN_PERCENT = 15.0

#: Минимальный допуск дрейфа QPA между стадиями сверх собственной esd, %.
QPA_DRIFT_WARN_PERCENT = 0.1

#: Строки движка, означающие, что уточнение не состоялось. GSAS-II печатает их
#: обычным print и НЕ поднимает исключение: do_refinements возвращается штатно.
_ENGINE_FATAL_PATTERN = re.compile(
    r"(?i)(invalid (cell )?metric tensor|refinement failed|no reflections in data range)"
)


@contextmanager
def capture_engine_notes(context: str):
    """Перехватывает вывод GSAS-II, дедуплицирует предупреждения в лог и возвращает список фатальных строк."""
    fatal_lines: list[str] = []
    buffer = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = buffer
    try:
        yield fatal_lines
    finally:
        sys.stdout = real_stdout
        captured = buffer.getvalue()
        warn_counts: dict[str, int] = {}
        passthrough_lines: list[str] = []
        for line in captured.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("N.B. Configuration file", "gpx file saved as", "Hessian ")):
                continue
            if _ENGINE_FATAL_PATTERN.search(stripped):
                fatal_lines.append(stripped)
            if _ENGINE_NOTE_PATTERN.search(stripped):
                warn_counts[stripped] = warn_counts.get(stripped, 0) + 1
            else:
                passthrough_lines.append(line)
        if passthrough_lines:
            real_stdout.write("\n".join(passthrough_lines) + "\n")
        for msg, cnt in warn_counts.items():
            suffix = f" (x{cnt})" if cnt > 1 else ""
            logger.warning(f"GSAS-II [{context}]: {msg}{suffix}")


def summarize_rule9(shifts: dict[str, float], sigmas: dict[str, float]) -> tuple[int, int, int]:
    """Сводка сходимости Rule 9: всего параметров, в допуске, ещё сходящихся.

    Семантика совпадает с таблицей сдвигов: параметр без положительной ``sigma``
    считается в допуске (PASS), а не сходящимся. Порог берётся из
    :mod:`constants`, чтобы подпись таблицы и расчёт не разъехались.
    """
    total = len(shifts)
    converging = 0
    for parm, value in shifts.items():
        sigma = float(sigmas.get(parm, 0.0))
        if sigma > 0.0 and abs(float(value)) / sigma >= RULE9_SHIFT_SIGMA_RATIO:
            converging += 1
    return total, total - converging, converging


def _report_stage_anomalies(gpx: Any, history: dict[str, Any], stage_name: str) -> None:
    """Печатает только аномалии стадии; отсутствие аномалий = молчание."""
    try:
        reflections = extract_bragg_reflections(gpx)
        rf_by_phase = phase_rf_percent(reflections)
    except (ValueError, KeyError, IndexError, AttributeError, TypeError) as exc:
        logger.debug(f"Could not evaluate per-phase RF: {exc}")
        rf_by_phase = {}
    for phase_name, rf in sorted(rf_by_phase.items()):
        if rf > PHASE_RF_WARN_PERCENT:
            logger.warning(
                f"Phase {phase_name}: RF={rf:.1f}% (> {PHASE_RF_WARN_PERCENT:.0f}%) "
                "at a normal phase weight — suspect the model, not the scale"
            )

    try:
        qpa = get_project_qpa(gpx)
    except (ValueError, KeyError, IndexError, AttributeError, TypeError) as exc:
        logger.debug(f"Could not evaluate QPA drift: {exc}")
        qpa = {}
    previous = history.get("qpa")
    if isinstance(previous, dict):
        for phase_name, pair in qpa.items():
            old = previous.get(phase_name)
            if not isinstance(old, (list, tuple)) or len(old) < 2:
                continue
            drift = abs(float(pair[0]) - float(old[0]))
            # esd может быть не определена (None): тогда допуск строится только на
            # пороге и на той стороне сравнения, у которой σ известна.
            esd_now = float(pair[1]) if pair[1] is not None else None
            esd_then = float(old[1]) if old[1] is not None else None
            tolerance = max(
                [QPA_DRIFT_WARN_PERCENT] + [v for v in (esd_now, esd_then) if v is not None]
            )
            if drift > tolerance:
                logger.warning(
                    f"Phase {phase_name}: QPA drift {drift:.2f}% at stage "
                    f"'{stage_name}' exceeds tolerance {tolerance:.2f}% "
                    "(composition moved, not just the profile)"
                )
    if qpa:
        history["qpa"] = {
            name: [float(value), float(esd) if esd is not None else None]
            for name, (value, esd) in qpa.items()
        }


def sanitize_pattern_data_file(data_path: Path, work_dir: Path) -> Path:
    """Ensure data file is formatted as valid 2-column CSV (2theta, intensity)."""
    if not data_path.exists():
        raise FileNotFoundError(f"Diffraction data file not found: {data_path}")

    # Inspect first few lines
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        sample_lines = [f.readline().strip() for _ in range(10)]

    is_clean_csv = False
    for line in sample_lines:
        if line and not line.startswith(("#", "!", ";", "%", "2theta")):
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    is_clean_csv = True
                    break
                except ValueError:
                    pass

    if is_clean_csv:
        return data_path

    # Convert whitespace or other delimiter to standard CSV
    pts_x = []
    pts_y = []
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith(("#", "!", ";", "%", "2theta")):
                continue
            parts = [p for p in line_str.replace(",", " ").split() if p]
            if len(parts) >= 2:
                try:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                    pts_x.append(x_val)
                    pts_y.append(y_val)
                except ValueError:
                    continue

    if not pts_x:
        raise ValueError(f"Could not parse numeric diffraction points from {data_path}")

    converted_file = work_dir / f"{data_path.stem}_converted.csv"
    np.savetxt(converted_file, np.column_stack([pts_x, pts_y]), delimiter=",", fmt="%.5f")
    logger.info(f"Converted pattern data to clean CSV: {converted_file} ({len(pts_x)} points)")
    return converted_file


# -----------------------------------------------------------------------------
# Subcommand: init
# -----------------------------------------------------------------------------
def cmd_init(args: SimpleNamespace) -> int:
    """Initialize a GSAS-II project from raw diffraction pattern."""
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    data_path = Path(args.data).resolve()
    project_path = Path(args.project).resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_data = sanitize_pattern_data_file(data_path, project_path.parent)

    inst_val = args.instrument
    wavelength_val = args.wavelength
    bkg_terms_val = args.bkg_terms
    # K-alpha2/K-alpha1 intensity ratio: 0.5 is the canonical doublet, 0.0 models
    # a data set whose companion line has already been removed.
    # getattr keeps callers that build the namespace by hand (tests,
    # xrd_pipeline) working without the new flag.
    # None — это «не задано», а не число: берём значение инструмента по умолчанию.
    ka2_ratio_val = float(getattr(args, "ka2_ratio", None) or 0.5)

    # In-memory canonical instrument parameters or custom file
    if str(inst_val).lower() == "canonical":
        inst = CanonicalInstrument(wavelength_ka1=wavelength_val, ka2_ratio=ka2_ratio_val)
        iparams: Any = inst.as_g2_dict()
    else:
        inst_p = Path(inst_val).resolve()
        if not inst_p.exists():
            raise FileNotFoundError(f"Instrument parameter file not found: {inst_p}")
        iparams = str(inst_p)

    gpx = G2sc.G2Project(newgpx=str(project_path))
    hist = gpx.add_powder_histogram(str(cleaned_data), iparams=iparams)

    hist.SampleParameters["Scale"][0] = 1.0
    hist.SampleParameters["Scale"][1] = False

    # Set initial Chebyshev background order
    hist.set_refinements(
        {
            "Background": {
                "type": "chebyschev-1",
                "refine": False,
                "no. coeffs": int(bkg_terms_val),
            }
        }
    )

    gpx.save()

    history = {
        "project_file": str(project_path),
        "data_file": str(data_path),
        "instrument": inst_val,
        "wavelength": wavelength_val,
        "ka2_ratio": ka2_ratio_val,
        "bkg_terms": bkg_terms_val,
        "created_at": datetime.now(UTC).isoformat(),
        "current_r_wp": None,
        "current_gof": None,
        "stages_completed": [],
        "phases": {},
        "iterations": [],
    }
    save_sidecar_history(project_path, history)

    logger.info(f"Project initialized successfully: {project_path}")
    print(
        f"SUCCESS: Project created at {project_path} (histogram points: {len(hist.data['data'][1][0])})"
    )
    return 0


def sanitize_cif_for_gsas2(cif_path: Path, staging_dir: Path) -> Path:
    """Read source CIF, normalize fractional/non-standard oxidation states to neutral symbols,
    and save clean copy in staging_dir without modifying the original user CIF (Immutable Priors rule).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / cif_path.name

    text = cif_path.read_text(encoding="utf-8", errors="replace")

    def _clean_token(match: re.Match[str]) -> str:
        prefix = match.group(1)
        sym = match.group(2)
        if Element.is_valid_symbol(sym):
            return f"{prefix}{sym}"
        return match.group(0)

    pattern1 = re.compile(r"(^|\s)([A-Z][a-z]?)(?:[0-9]+(?:\.[0-9]+)?)?[+-](?=\s|$)")
    pattern2 = re.compile(r"(^|\s)([A-Z][a-z]?)[+-](?:[0-9]+(?:\.[0-9]+)?)?(?=\s|$)")

    cleaned = pattern1.sub(_clean_token, text)
    cleaned = pattern2.sub(_clean_token, cleaned)

    if cleaned != text:
        logger.info(
            f"Sanitized non-standard oxidation states in CIF {cif_path.name} -> {out_path.name}"
        )

    out_path.write_text(cleaned, encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------
# Subcommand: add-phase
# -----------------------------------------------------------------------------
def build_placeholder_cif_from_card(card: Any, out_cif: Path) -> Path:
    """Write a deterministic phase CIF from a reference card's unit cell and space group via gemmi.

    GSAS-II's CIF importer (G2phase_CIF.py) parses the space group directly from
    _space_group_name_H-M_alt / _symmetry_space_group_name_H-M via GSASIIspc.SpcGroup
    without invoking spglib on atomic coordinates. Constructing the CIF document directly
    via gemmi.cif.Document preserves the exact card setting, space group, and axis order
    (a, b, c) with a single placeholder atom at the origin.
    """
    import gemmi

    if card.cell_a is None or card.cell_b is None or card.cell_c is None:
        raise ValueError(
            f"card {card.pdf_number} carries no complete cell, so no phase can be built from it"
        )
    sg_str = str(getattr(card, "space_group", "") or "").strip()
    sg_num = int(card.ita_number) if getattr(card, "ita_number", None) else 0
    if not sg_str and sg_num <= 0:
        raise ValueError(f"card {card.pdf_number} carries no space group")

    formula = card.structural_formula or card.empirical_formula
    try:
        placeholder = next(iter(Composition(formula.replace(" ", "")).elements)).symbol
    except (ValueError, StopIteration) as exc:
        raise ValueError(
            f"card {card.pdf_number} has no readable formula ({formula!r}), so no "
            f"placeholder atom can be chosen: {exc}"
        ) from exc

    sg_obj = gemmi.find_spacegroup_by_name(sg_str) if sg_str else None
    if sg_obj is None and 1 <= sg_num <= 230:
        sg_obj = gemmi.SpaceGroup(sg_num)
    if sg_num <= 0 and sg_obj is not None:
        sg_num = int(sg_obj.number)
    hm_symbol = sg_str.replace("_", "") if sg_str else (sg_obj.short_name() if sg_obj else "P 1")

    a_val = float(card.cell_a)
    b_val = float(card.cell_b)
    c_val = float(card.cell_c)
    alpha_val = float(card.cell_alpha if card.cell_alpha is not None else 90.0)
    beta_val = float(card.cell_beta if card.cell_beta is not None else 90.0)
    gamma_val = float(card.cell_gamma if card.cell_gamma is not None else 90.0)
    uc = gemmi.UnitCell(a_val, b_val, c_val, alpha_val, beta_val, gamma_val)

    logger.info(
        "Card {} has no atomic coordinates; phase built deterministically from its cell "
        "({} {} {}) and space group {} (ITA #{}) with {} as a placeholder atom",
        card.pdf_number,
        card.cell_a,
        card.cell_b,
        card.cell_c,
        hm_symbol,
        sg_num,
        placeholder,
    )

    doc = gemmi.cif.Document()
    block = doc.add_new_block(f"card_{str(card.pdf_number).replace('-', '_')}")
    block.set_pair("_cell_length_a", f"{a_val:.6f}")
    block.set_pair("_cell_length_b", f"{b_val:.6f}")
    block.set_pair("_cell_length_c", f"{c_val:.6f}")
    block.set_pair("_cell_angle_alpha", f"{alpha_val:.4f}")
    block.set_pair("_cell_angle_beta", f"{beta_val:.4f}")
    block.set_pair("_cell_angle_gamma", f"{gamma_val:.4f}")
    block.set_pair("_cell_volume", f"{float(uc.volume):.4f}")
    block.set_pair("_symmetry_space_group_name_H-M", f"'{hm_symbol}'")
    block.set_pair("_space_group_name_H-M_alt", f"'{hm_symbol}'")
    if 1 <= sg_num <= 230:
        block.set_pair("_symmetry_Int_Tables_number", str(sg_num))
        block.set_pair("_space_group_IT_number", str(sg_num))

    loop = block.init_loop(
        "_atom_site_",
        [
            "label",
            "type_symbol",
            "fract_x",
            "fract_y",
            "fract_z",
            "occupancy",
            "U_iso_or_equiv",
        ],
    )
    loop.add_row([f"{placeholder}1", placeholder, "0.000000", "0.000000", "0.000000", "1.0000", "0.0100"])

    out_cif.parent.mkdir(parents=True, exist_ok=True)
    doc.write_file(str(out_cif))
    return out_cif


class PawleyCol(IntEnum):
    """Индексы записи рефлекса в таблице GSAS-II phase.data['Pawley ref']."""

    H = 0
    K = 1
    L = 2
    MULT = 3
    D_SPACING = 4
    REFINE = 5
    INTENSITY = 6
    ESD = 7


def is_pawley_phase(phase: Any) -> bool:
    """Возвращает True, если у фазы активирован нативный режим декомпозиции Поули."""
    pdata = getattr(phase, "data", {})
    if not isinstance(pdata, dict):
        return False
    return bool(pdata.get("General", {}).get("doPawley", False))


def enable_pawley_for_phase(
    phase: Any,
    hist: Any,
    neg_wt: float = 0.05,
    margin_deg: float = 0.5,
) -> int:
    """Детерминированно инициализирует нативный режим Pawley в GSAS-II без эвристических отсечек.

    Генерирует полный список разрешённых симметрией отражений через GSASIIlattice.GenHLaue
    и GSASIIspc.GenHKLf, активирует все рефлексы внутри фактического диапазона измерения
    гистограммы [two_theta_min, two_theta_max] и фиксирует Scale = [1.0, False].
    """
    from GSASII import GSASIIlattice as G2lat
    from GSASII import GSASIIspc as G2spc

    pdata = phase.data["General"]
    sgdata = pdata["SGData"]
    cell = pdata["Cell"][1:7]

    inst_raw = hist.data.get("Instrument Parameters", [{}])[0]
    lam_rec = inst_raw.get("Lam1") or inst_raw.get("Lam") or [1.540596, 1.540596]
    lam = float(lam_rec[1] if len(lam_rec) > 1 else lam_rec[0])

    two_theta_min = 10.0
    two_theta_max = 80.0
    with suppress(ValueError, KeyError, IndexError, AttributeError, TypeError):
        curves = extract_profile_curves(hist)
        if len(curves.two_theta) > 0:
            two_theta_min = float(np.min(curves.two_theta))
            two_theta_max = float(np.max(curves.two_theta))

    sin_max = np.sin(np.radians(min(two_theta_max + float(margin_deg), 175.0) / 2.0))
    effective_dmin = lam / (2.0 * max(sin_max, 1e-4))

    pdata["doPawley"] = True
    pdata["Pawley ref"] = True
    pdata["Pawley dmin"] = effective_dmin
    pdata["Pawley neg wt"] = float(neg_wt)

    metric_tensor = G2lat.cell2A(cell)
    hkl_d_list = G2lat.GenHLaue(effective_dmin, sgdata, metric_tensor)

    pawley_peaks: list[list[Any]] = []
    active_count = 0

    for row in hkl_d_list:
        h_idx = int(row[0])
        k_idx = int(row[1])
        l_idx = int(row[2])
        d_val = float(row[3])
        hkl_f = G2spc.GenHKLf([h_idx, k_idx, l_idx], sgdata)
        ext = bool(hkl_f[0])
        mul = int(hkl_f[1])
        if ext:
            continue
        mul *= 2
        sin_th = lam / (2.0 * max(d_val, 1e-6))
        if sin_th <= 1.0:
            two_th = float(np.degrees(2.0 * np.arcsin(sin_th)))
            if two_theta_min <= two_th <= two_theta_max:
                pawley_peaks.append([h_idx, k_idx, l_idx, mul, d_val, True, 100.0, 1.0])
                active_count += 1
            else:
                pawley_peaks.append([h_idx, k_idx, l_idx, mul, d_val, False, 0.0, 1.0])
        else:
            pawley_peaks.append([h_idx, k_idx, l_idx, mul, d_val, False, 0.0, 1.0])

    phase.data["Pawley ref"] = pawley_peaks

    hist_name = getattr(hist, "name", "")
    if "Histograms" in phase.data and hist_name in phase.data["Histograms"]:
        hap = phase.data["Histograms"][hist_name]
        hap["Scale"] = [1.0, False]
        if (
            "Size" in hap
            and isinstance(hap["Size"], list)
            and len(hap["Size"]) >= 2
            and isinstance(hap["Size"][1], list)
            and float(hap["Size"][1][0]) >= 0.5
        ):
            hap["Size"][1][0] = 0.030
        if (
            "Mustrain" in hap
            and isinstance(hap["Mustrain"], list)
            and len(hap["Mustrain"]) >= 2
            and isinstance(hap["Mustrain"][1], list)
        ):
            hap["Mustrain"][1][0] = 0.0

    return active_count


def clamp_pawley_reflections(gpx: Any, lock_nonpositive: bool = False) -> int:
    """Проецирует отрицательные интенсивности Pawley на неотрицательный конус.

    На промежуточных стадиях (lock_nonpositive=False) удерживает интенсивность >= 5.0
    с сохранением refine=True. На финальной стадии (lock_nonpositive=True) обнуляет
    неположительные рефлексы и снимает refine=False, предотвращая деление на ноль
    в GSASIIstrMath (dMdpw = dervDict['int'] / refl[9+im]).
    """
    clamped_total = 0
    for phase in gpx.phases():
        if not is_pawley_phase(phase):
            continue
        for pk in phase.data.get("Pawley ref", []):
            if not bool(pk[PawleyCol.REFINE]):
                continue
            inten = float(pk[PawleyCol.INTENSITY])
            if lock_nonpositive and inten <= 0.0:
                pk[PawleyCol.INTENSITY] = 0.0
                pk[PawleyCol.REFINE] = False
                clamped_total += 1
            elif not lock_nonpositive and inten <= 0.0:
                pk[PawleyCol.INTENSITY] = 5.0
                clamped_total += 1
    return clamped_total


def cmd_add_phase(args: SimpleNamespace) -> int:
    """Add a crystallographic phase from CIF, or from a reference card, into the project."""
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    # A trial addition must be reversible: the snapshot is taken before the
    # phase enters the project, so `revert` can undo the addition and the
    # refinement that followed it.
    from_card = getattr(args, "from_card", None)
    from_cod = getattr(args, "from_cod", None)
    sources_given = [s for s in (args.cif, from_card, from_cod) if s is not None]
    if len(sources_given) != 1:
        logger.error("Give exactly one of --cif, --from-card, or --from-cod")
        return 1

    label_src = from_card or (f"cod_{from_cod}" if from_cod else Path(args.cif).stem)
    label = f"add-phase {args.name or label_src}"
    push_checkpoint(project_path, label=label)

    if from_card:
        try:
            from pdf2_reader import PDF2Database

            with PDF2Database() as db:
                card = db.get_card(db.index_of(str(from_card)))
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            logger.error("Cannot read reference card {}: {}", from_card, exc)
            return 1
        try:
            cif_path = build_placeholder_cif_from_card(
                card, project_path.parent / ".staging_cifs" / f"card_{from_card}.cif"
            )
        except ValueError as exc:
            logger.error("{}", exc)
            return 1
    elif from_cod:
        try:
            from hs_rdb import HsRdbDatabase

            staging_target = project_path.parent / ".staging_cifs" / f"cod_{from_cod}.cif"
            with HsRdbDatabase() as db:
                cif_path = db.export_cif(str(from_cod), out_path=staging_target, by_cod_id=True)
        except (ImportError, KeyError, ValueError, OSError, RuntimeError) as exc:
            logger.error("Cannot extract COD structure {}: {}", from_cod, exc)
            return 1
    else:
        cif_path = Path(args.cif).resolve()
        if not cif_path.exists():
            logger.error(f"CIF file not found: {cif_path}")
            return 1

    gpx = G2sc.G2Project(str(project_path))
    phase_name = args.name.strip() if args.name else cif_path.stem

    # Check existing phases to avoid duplicate names
    existing_names = [p.name for p in gpx.phases()]
    if phase_name in existing_names:
        logger.warning(f"Phase '{phase_name}' already exists in project. Appending suffix.")
        phase_name = f"{phase_name}_{len(existing_names) + 1}"

    # Normalize CIF to staging directory without modifying original user CIF
    staging_dir = project_path.parent / ".staging_cifs"
    staging_cif = sanitize_cif_for_gsas2(cif_path, staging_dir)

    p = gpx.add_phase(
        str(staging_cif), histograms=gpx.histograms(), phasename=phase_name, fmthint="CIF"
    )

    use_pawley = bool(getattr(args, "pawley", False) or from_card)

    if from_card:
        import gemmi

        held = str(p.data.get("General", {}).get("SGData", {}).get("SpGrp", ""))
        wanted = str(card.space_group)
        held_sg = gemmi.find_spacegroup_by_name(held)
        wanted_num = int(card.ita_number) if getattr(card, "ita_number", None) else (
            int(gemmi.find_spacegroup_by_name(wanted).number)
            if gemmi.find_spacegroup_by_name(wanted) is not None
            else -1
        )
        held_num = int(held_sg.number) if held_sg is not None else -2
        if (
            held.replace(" ", "").lower() != wanted.replace(" ", "").lower()
            and held_num != wanted_num
        ):
            logger.error(
                "A phase built from card {} entered the project as '{}' instead of '{}', so "
                "the model would describe a different symmetry than the card states. The "
                "addition is refused and nothing is written.",
                from_card,
                held,
                wanted,
            )
            return 1

    if use_pawley:
        # First run 1 cycle with doPawley=False so GSAS-II populates RefList
        # (required both to verify the cell/symmetry and to supply Bragg ticks for plots).
        failure = ""
        try:
            gpx.set_Controls("cycles", 1)
            gpx.refine(makeBack=False)
        except Exception as exc:  # noqa: BLE001 - the engine reports a bare string message
            failure = str(exc)
        reflection_lists = gpx.histograms()[0].data.get("Reflection Lists", {})
        built = reflection_lists.get(phase_name) or {}
        reflist = built.get("RefList")
        if failure or reflist is None or len(reflist) == 0:
            logger.error(
                "Phase '{}' cannot be used by the engine: it built no reflection list "
                "for it ({}). The addition is refused and nothing is written.",
                phase_name,
                failure or f"reflection lists hold {sorted(reflection_lists) or 'nothing'}",
            )
            return 1

    init_scale = 1.0 if use_pawley else (float(args.scale) if args.scale is not None else 1e-2)
    for h in gpx.histograms():
        if use_pawley:
            active_cnt = enable_pawley_for_phase(p, h)
            sync_pawley_reflections_to_hist(gpx, h)
            logger.info(
                "Phase '{}' configured for native Pawley refinement ({} active reflections)",
                phase_name,
                active_cnt,
            )
        else:
            p.set_HAP_refinements({"Scale": True})
            try:
                if "Histograms" in p.data and h.name in p.data["Histograms"]:
                    p.data["Histograms"][h.name]["Scale"][0] = init_scale
            except (KeyError, IndexError, TypeError, AttributeError) as e_hap:
                logger.debug(f"Could not set initial HAP scale: {e_hap}")

    # The new phase enters with GSAS-II's default profile width, which can be far from what
    # the phases already in the project refined to. A phase whose calculated peaks are much
    # narrower than the observed ones cannot match them: it takes a small weight, the
    # residuals stay large, and the result then reads as a property of the sample rather
    # than of the starting model. Measured on the Sm-S workbench: setting the width alone
    # moved R_wp from 9.23 to 8.35 % and the minority weight from 3.81 to 8.78 %. Both
    # values are named here instead of leaving the default silent.
    histograms = gpx.histograms()
    if histograms:
        new_size: float | None = None
        reference_sizes: list[float] = []
        for ph in gpx.phases():
            hap = ph.data.get("Histograms", {}).get(histograms[0].name, {})
            record = hap.get("Size") if isinstance(hap, dict) else None
            if not (isinstance(record, (list, tuple)) and len(record) >= 2):
                continue
            with suppress(TypeError, ValueError, IndexError):
                value = float(record[1][0])
                if ph.name == phase_name:
                    new_size = value
                elif value > 0:
                    reference_sizes.append(value)
        if new_size and reference_sizes:
            reference = sorted(reference_sizes)[len(reference_sizes) // 2]
            if new_size > 2 * reference or new_size < reference / 2:
                logger.warning(
                    "Phase '{}' enters with crystallite size {:.4f} um, while the phase(s) "
                    "already in the project sit at {:.4f} um. Peaks calculated much "
                    "narrower than the observed ones cannot match them, so this phase will "
                    "take a small weight and R_wp will suffer. Set the width before "
                    "refining: `params dump --profile-start-size {:.4f}`, or edit "
                    "profile.size of this phase in the dumped document.",
                    phase_name,
                    new_size,
                    reference,
                    reference,
                )

    _refresh_saved_model(gpx)
    gpx.save()

    history = load_sidecar_history(project_path)
    history["phases"][phase_name] = {
        "cif_file": str(cif_path),
        "added_at": datetime.now(UTC).isoformat(),
        "enabled": True,
        "initial_scale": init_scale,
    }
    history["iterations"].append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": f"add-phase: {phase_name}",
            "cif": str(cif_path),
        }
    )
    save_sidecar_history(project_path, history)

    logger.info(f"Phase '{phase_name}' added to {project_path}")
    print(f"SUCCESS: Added phase '{phase_name}' from {cif_path.name}")
    return 0


# -----------------------------------------------------------------------------
# Subcommand: remove-phase
# -----------------------------------------------------------------------------
def cmd_remove_phase(args: SimpleNamespace) -> int:
    """Safely remove a phase from the project."""
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    push_checkpoint(project_path, label=f"remove-phase {args.phase_name}")

    gpx = G2sc.G2Project(str(project_path))
    phases = gpx.phases()

    target_name = args.phase_name.strip()
    target_phase = None
    for p in phases:
        if p.name.lower() == target_name.lower():
            target_phase = p
            break

    if target_phase is None:
        logger.error(f"Phase '{target_name}' not found in project {project_path}")
        return 1

    p_actual_name = target_phase.name
    target_id = target_phase.ranId

    # 1. Remove from self.names list under 'Phases'
    for obj in gpx.names:
        if obj and obj[0] == "Phases":
            if target_id is not None and target_id in obj:
                obj.remove(target_id)
            elif target_phase in obj:
                obj.remove(target_phase)
            elif p_actual_name in obj:
                obj.remove(p_actual_name)

    # 2. Remove from gpx.data dictionary
    if target_id is not None and target_id in gpx.data:
        gpx.data.pop(target_id, None)
    if "Phases" in gpx.data and p_actual_name in gpx.data["Phases"]:
        gpx.data["Phases"].pop(p_actual_name, None)

    # 3. Clean up histograms
    for h in gpx.histograms():
        refl_dict = h.data.get("Reflection Lists", {})
        refl_dict.pop(p_actual_name, None)
        if target_id is not None:
            refl_dict.pop(str(target_id), None)
        ph_dict = h.data.get("Phases", {})
        ph_dict.pop(p_actual_name, None)
        if target_id is not None:
            ph_dict.pop(str(target_id), None)

    gpx.index_ids()

    _refresh_saved_model(gpx)
    gpx.save()

    history = load_sidecar_history(project_path)
    if p_actual_name in history.get("phases", {}):
        history["phases"].pop(p_actual_name, None)
    history["iterations"].append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": f"remove-phase: {p_actual_name}",
        }
    )
    save_sidecar_history(project_path, history)

    logger.info(f"Phase '{p_actual_name}' removed from {project_path}")
    print(f"SUCCESS: Removed phase '{p_actual_name}'")
    return 0


# -----------------------------------------------------------------------------
# Subcommand: toggle-phase
# -----------------------------------------------------------------------------
def cmd_toggle_phase(args: SimpleNamespace) -> int:
    """Enable or disable a phase in refinement without deleting it."""
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    push_checkpoint(project_path, label=f"toggle-phase {args.phase_name}")

    gpx = G2sc.G2Project(str(project_path))
    phases = gpx.phases()

    target_name = args.phase_name.strip()
    target_phase = None
    for p in phases:
        if p.name.lower() == target_name.lower():
            target_phase = p
            break

    if target_phase is None:
        logger.error(f"Phase '{target_name}' not found in project.")
        return 1

    enable = args.enable if args.enable is not None else not args.disable

    hist_names = [h.name for h in gpx.histograms()]
    if not enable:
        # Disable: lock scale to 0.0 and clear refinements
        target_phase.set_HAP_refinements({"Scale": False})
        target_phase.set_refinements({"Cell": False})
        for h_name in hist_names:
            try:
                target_phase.data["Histograms"][h_name]["Scale"][0] = 0.0
                target_phase.data["Histograms"][h_name]["Scale"][1] = False
            except (KeyError, IndexError, TypeError, AttributeError) as e_dis:
                logger.debug(f"Could not lock scale to 0.0: {e_dis}")
        action_str = f"Disabled phase '{target_phase.name}' (Scale=0.0, refinement locked)"
    else:
        # Enable: unlock scale with small initial positive fraction
        target_phase.set_HAP_refinements({"Scale": True})
        for h_name in hist_names:
            try:
                cur_sc = float(target_phase.data["Histograms"][h_name]["Scale"][0])
                if cur_sc <= 0.0:
                    target_phase.data["Histograms"][h_name]["Scale"][0] = 0.001
                target_phase.data["Histograms"][h_name]["Scale"][1] = True
            except (KeyError, IndexError, TypeError, AttributeError) as e_en:
                logger.debug(f"Could not enable scale: {e_en}")
        action_str = f"Enabled phase '{target_phase.name}'"

    # После смены фаз Ycalc описывает прежнюю модель; без пересчёта status и
    # diagnose показывают метрику до правки (наблюдалось 18.73 % после отключения фазы).
    _refresh_saved_model(gpx)

    gpx.save()

    history = load_sidecar_history(project_path)
    if target_phase.name in history.get("phases", {}):
        history["phases"][target_phase.name]["enabled"] = enable
    history["iterations"].append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action_str,
        }
    )
    save_sidecar_history(project_path, history)

    logger.info(action_str)
    print(f"SUCCESS: {action_str}")
    return 0


# -----------------------------------------------------------------------------
# Subcommand: revert
# -----------------------------------------------------------------------------
def cmd_revert(args: SimpleNamespace) -> int:
    """Restore a project from its checkpoint stack.

    This is the undo of a trial mutation. ``remove-phase`` cannot serve as one:
    it takes the phase out but leaves every other parameter where the trial put
    it, so the metric does not return to its previous value.
    """
    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    stack_file = get_checkpoint_stack_path(project_path)
    if not stack_file.exists():
        logger.error(f"No checkpoint stack for {project_path.name}")
        return 1

    try:
        entries = json.loads(stack_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(f"Cannot read checkpoint stack {stack_file}: {exc}")
        return 1
    if not entries:
        logger.error("Checkpoint stack is empty")
        return 1

    if args.to is None:
        chosen = entries[-1]
    else:
        chosen = next((e for e in entries if int(e.get("index", -1)) == args.to), None)
        if chosen is None:
            available = [e.get("index") for e in entries]
            logger.error(f"Checkpoint {args.to} not found; available: {available}")
            return 1

    snapshot = project_path.parent / str(chosen.get("file", ""))
    if not snapshot.exists():
        logger.error(f"Checkpoint file is missing: {snapshot}")
        return 1

    shutil.copyfile(str(snapshot), str(project_path))

    restored_index = int(chosen.get("index", 0))
    reverted = [e for e in entries if int(e.get("index", 0)) >= restored_index]
    remaining = [e for e in entries if int(e.get("index", 0)) < restored_index]
    for entry in reverted:
        with suppress(OSError):
            (project_path.parent / str(entry.get("file", ""))).unlink()
    try:
        stack_file.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not update checkpoint stack {stack_file}: {exc}")

    # The journal is a log of what was done, so the revert is recorded in it
    # rather than silently erased.
    history = load_sidecar_history(project_path)
    history["iterations"].append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": f"revert-to-checkpoint-{restored_index}",
        }
    )
    save_sidecar_history(project_path, history)

    logger.info(f"Reverted {project_path.name} to checkpoint {restored_index}")
    print(f"SUCCESS: Reverted to checkpoint {restored_index} ({chosen.get('label') or 'no label'})")
    return 0


# -----------------------------------------------------------------------------
# Subcommand: status
# -----------------------------------------------------------------------------
def cmd_status(args: SimpleNamespace) -> int:
    """Inspect full project state directly from .gpx SSOT."""
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    gpx = G2sc.G2Project(str(project_path))
    phases = gpx.phases()
    histograms = gpx.histograms()

    # Load sidecar history for iteration logs, but do not fail if missing (SSOT rule)
    history = load_sidecar_history(project_path)

    vary_list = []
    try:
        vary_list = gpx.data.get("Controls", {}).get("VaryList", [])
    except (KeyError, AttributeError):
        pass

    scales = extract_phase_scales(gpx)
    cells = extract_cell_parameters(gpx)

    # QPA via canonical SSOT: get_project_qpa() (native ComputeMassFracs + Hill-Howard fallback with Z*M)
    mass_fracs_dict: dict[str, tuple[float, float]] = get_project_qpa(
        gpx, histograms[0] if histograms else None
    )

    # Agreement indices calculated directly from profile curves
    metrics_data: dict[str, Any] = {}
    if histograms:
        try:
            curves = extract_profile_curves(histograms[0])
            m = calculate_profile_metrics(curves, n_free_parameters=max(1, len(vary_list)))
            metrics_data = {
                "r_wp_percent": m.r_wp_percent,
                "r_p_percent": m.r_p_percent,
                "r_wp_over_r_p_ratio": m.r_wp_over_r_p_ratio,
                "gof": m.gof,
                "chi_square": m.reduced_chi2,
                "durbin_watson": m.durbin_watson,
                "data_points": m.n_data_points,
                "free_parameters_count": len(vary_list),
            }
        except (ValueError, KeyError, IndexError, AttributeError) as e_prof:
            metrics_data = {"error": str(e_prof)}

    phases_info = []
    for p in phases:
        p_name = p.name
        sg = "Unknown"
        try:
            sg = p.data.get("General", {}).get("SGData", {}).get("SpGrp", "Unknown")
        except (KeyError, AttributeError):
            pass

        sc = scales.get(p_name, 0.0)
        is_enabled = sc > 1e-9
        mf_tuple = mass_fracs_dict.get(p_name, (0.0, 0.0))

        phases_info.append(
            {
                "name": p_name,
                "space_group": sg,
                "enabled": is_enabled,
                "scale": sc,
                "weight_pct": round(float(mf_tuple[0]), 2),
                "weight_pct_esd": (
                    round(float(mf_tuple[1]), 2) if mf_tuple[1] is not None else None
                ),
                "cell": cells.get(p_name, {}),
            }
        )

    status_data = {
        "project": str(project_path),
        "phases_count": len(phases),
        "histograms_count": len(histograms),
        "metrics": metrics_data,
        "phases": phases_info,
        "vary_list_count": len(vary_list),
        "last_stage_completed": history.get("last_stage"),
        "iteration_history_count": len(history.get("iterations", [])),
        "iteration_history": history.get("iterations", []),
    }

    if args.json:
        print_json(status_data)
    else:
        print("=" * 65)
        print(f"GSAS-II Project Status: {project_path.name}")
        print("=" * 65)
        m = status_data["metrics"]
        if "r_wp_percent" in m:
            print(
                f"R_wp: {m['r_wp_percent']}% | R_p: {m['r_p_percent']}% | "
                f"chi^2: {float(m['chi_square']):.2f} | GoF: {m['gof']} | DW: {m['durbin_watson']}"
            )
            print(
                f"Data points: {m['data_points']} | Refined parameters: {m['free_parameters_count']}"
            )
        else:
            print("Agreement metrics: Not yet refined")
        print(f"\nPhases ({len(phases_info)}):")
        for p in phases_info:
            c = p["cell"]
            c_str = f"a={c.get('a', 0):.4f}" if "a" in c else ""
            if "c" in c and c.get("c") != c.get("a"):
                c_str += f", c={c.get('c', 0):.4f}"
            status_tag = "ACTIVE" if p["enabled"] else "DISABLED"
            print(
                f"  - [{status_tag}] {p['name']} ({p['space_group']}): QPA={p['weight_pct']}% (±{p['weight_pct_esd']}%) | Scale={p['scale']:.4e} | {c_str}"
            )
        # Фазовые R-факторы: R_F совпадает с движком, R_B(lin) — наш расчёт,
        # Rf^2 движка считается по RMS-формуле и печатается отдельной колонкой.
        if histograms:
            try:
                reflections = extract_bragg_reflections(histograms[0])
                rf_by_phase = phase_rf_percent(reflections)
                rb_by_phase = phase_rb_percent(reflections)
                id_to_name = {str(i): ph.name for i, ph in enumerate(phases)}
                native_by_phase = {
                    id_to_name.get(k, k): v
                    for k, v in native_phase_r_factors(histograms[0]).items()
                }
                print("Phase R-factors:")
                for name in sorted(rf_by_phase):
                    rf_v = rf_by_phase.get(name)
                    rb_v = rb_by_phase.get(name)
                    rf2_v = native_by_phase.get(name, {}).get("Rf^2")
                    tail = f" Rf^2(GSAS)={rf2_v:.2f}%" if rf2_v is not None else ""
                    print(f"  - {name}: R_F={rf_v:.2f}% R_B(lin)={rb_v:.2f}%{tail}")
            except (ValueError, KeyError, IndexError, AttributeError, TypeError) as e_rf:
                logger.debug(f"Phase R-factors unavailable: {e_rf}")

        print("=" * 65)

    return 0


# -----------------------------------------------------------------------------
# Subcommand: step
# -----------------------------------------------------------------------------
def _snapshot_project_state(gpx: Any, hist: Any) -> dict[str, float]:
    """Снимает словарь ключевых параметров проекта (DisplaceX, Zero, scale, a, b, c, MD)."""
    state_dict: dict[str, float] = {}
    try:
        state_dict["DisplaceX"] = float(hist.SampleParameters.get("DisplaceX", [0.0, False])[0])
        state_dict["Zero"] = float(hist.InstrumentParameters.get("Zero", [0.0, False])[0])
        cells = extract_cell_parameters(gpx)
        scales = extract_phase_scales(gpx)
        for p in gpx.phases():
            p_name = p.name
            state_dict[f"{p_name}_scale"] = scales.get(p_name, 0.0)
            for ax in ("a", "b", "c"):
                if ax in cells.get(p_name, {}):
                    state_dict[f"{p_name}_{ax}"] = cells[p_name][ax]
            if hist.name in p.data.get("Histograms", {}):
                hap = p.data["Histograms"][hist.name]
                if "Pref.Ori." in hap:
                    state_dict[f"{p_name}_MD"] = float(hap["Pref.Ori."][1])
    except (ValueError, KeyError, IndexError, AttributeError, TypeError) as exc:
        logger.debug(f"Could not snapshot project parameters: {exc}")
    return state_dict


def _run_stage_in_project(
    gpx: Any,
    hist: Any,
    stage_name: str,
    args: SimpleNamespace,
    project_path: Path,
    ckpt_path: Path,
) -> dict[str, Any] | None:
    """Execute a single refinement stage on loaded GPX project and commit or rollback."""
    # Measure baseline R_wp before step
    prev_r_wp = 99.0
    try:
        prev_curves = extract_profile_curves(hist)
        prev_m = calculate_profile_metrics(prev_curves)
        prev_r_wp = prev_m.r_wp_percent
    except (ValueError, KeyError, IndexError, AttributeError) as e_prev:
        logger.debug(f"Could not measure baseline R_wp: {e_prev}")

    prev_state_dict = _snapshot_project_state(gpx, hist)

    # Set optimization cycles
    target_cycles = int(args.cycles)
    gpx.set_Controls("cycles", target_cycles)
    gpx.data.setdefault("Controls", {}).setdefault("data", {})["max cyc"] = target_cycles

    # Ensure all atomic Uiso are within physical bounds before stage setup
    for p in gpx.phases():
        for at in p.atoms():
            rec = G2AtomRecord(at)
            if rec.uiso < 0.001 or rec.uiso > 0.15:
                rec.uiso = 0.01

    # 2. Снять флаги предыдущих стадий: стадия освобождает только свои параметры.
    if hasattr(hist, "clear_refinements"):
        reset_stage_flags(gpx, hist)

    has_pawley = any(is_pawley_phase(p) for p in gpx.phases())
    for p in gpx.phases():
        if is_pawley_phase(p):
            p.data["General"]["Pawley ref"] = True
            if hist.name in p.data.get("Histograms", {}):
                p.data["Histograms"][hist.name]["Scale"] = [1.0, False]

    # На стадиях ячейки (3), профиля (7) и текстуры (8) масштабные факторы активных
    # фаз остаются свободными, чтобы МНК не компенсировал сдвиг и форму пиков
    # за счёт замороженного масштаба и сохранял ковариацию масштабов для QPA e.s.d.
    if stage_name in (
        StageId.UNIT_CELL,
        StageId.SAMPLE_PROFILE_XY,
        StageId.PREFERRED_ORIENTATION,
    ):
        active_scales = extract_phase_scales(gpx)
        for p in gpx.phases():
            if not is_pawley_phase(p) and active_scales.get(p.name, 0.0) > 1e-9:
                p.set_HAP_refinements({"Scale": True})

    # 3. Build declarative Stage-Gate refinement recipe via canonical SSOT
    recipe: dict[str, Any] = {}

    if stage_name == "1_Scale_and_Background":
        hist.SampleParameters["Scale"][0] = 1.0
        hist.SampleParameters["Scale"][1] = False
        for p in gpx.phases():
            if not is_pawley_phase(p):
                p.set_HAP_refinements({"Scale": True})
        bkg_terms_val = args.bkg_terms
        recipe = build_stage_recipe(stage_name, bkg_terms=int(bkg_terms_val))
        if has_pawley:
            recipe.pop("Scale", None)

    elif stage_name == "4_Atomic_Coordinates":
        # Setup universal distance restraints for each active phase
        history_snap = load_sidecar_history(project_path)
        for p in gpx.phases():
            p_struct = None
            cif_p = history_snap.get("phases", {}).get(p.name, {}).get("cif_file")
            if cif_p and Path(cif_p).exists():
                with suppress(RuntimeError, ValueError, KeyError, AttributeError, OSError):
                    p_struct = Structure.from_file(cif_p)
            setup_phase_dist_restraints(gpx, p, p_struct)
        recipe = build_stage_recipe(stage_name)

    elif stage_name == "9_Occupancy":
        # Заселение освобождается не по «катионности» элемента (это свойство
        # химии, а не позиции), а по состоянию модели: там, где она сама уже
        # допускает неполное заполнение. Всё остальное — только по явному выбору.
        explicit = getattr(args, "occupancy_atoms", None)
        selected: list[tuple[Any, Any]] = []
        if explicit:
            wanted = {item.strip().lower() for item in str(explicit).split(",") if item.strip()}
            for p in gpx.phases():
                for atom in p.atoms():
                    if f"{p.name}:{atom.label}".lower() in wanted:
                        selected.append((p, atom))
            if not selected:
                logger.error(f"9_Occupancy: ни один из указанных атомов не найден: {explicit}")
                return None
        else:
            for p in gpx.phases():
                for atom in p.atoms():
                    if float(atom.occupancy) < 0.999:
                        selected.append((p, atom))
        for p in gpx.phases():
            for atom in p.atoms():
                atom.refinement_flags = ""
        for p, atom in selected:
            atom.refinement_flags = "F"
        if not selected:
            logger.error(
                "9_Occupancy: в модели нет позиций с заселением < 1 и атомы не указаны явно. "
                'Назовите их: --occupancy-atoms "Phase:Label".'
            )
            return None
        logger.info(
            "9_Occupancy refines occupancy of: "
            + ", ".join(f"{p.name}:{atom.label}" for p, atom in selected)
        )
        recipe = {}

    elif stage_name == "7_Sample_Profile_XY":
        # Decouple: fix atomic coordinates and thermal parameters to prevent profile-thermal Hessian singularities
        for p in gpx.phases():
            p.set_refinements({"Atoms": {"all": ""}})
        if has_pawley:
            for p in gpx.phases():
                if is_pawley_phase(p):
                    p.data["General"]["Pawley ref"] = False
                if hist.name in p.data.get("Histograms", {}):
                    hap = p.data["Histograms"][hist.name]
                    if "Size" in hap and len(hap["Size"]) >= 3:
                        if float(hap["Size"][1][0]) >= 0.200 or float(hap["Size"][1][0]) < 0.015:
                            hap["Size"][1][0] = 0.030
                        hap["Size"][2][0] = True
                    if "Mustrain" in hap and len(hap["Mustrain"]) >= 3:
                        hap["Mustrain"][1][0] = 0.0
                        hap["Mustrain"][2][0] = False
            recipe = {"Instrument Parameters": ["SH/L"]} if bool(args.refine_axial) else {}
        else:
            recipe = build_stage_recipe(
                stage_name,
                refine_axial=bool(args.refine_axial),
            )
        # Uniaxial crystallite size for platy/needle phases (project-level HAP mutation)
        if args.uniaxial_size:
            for u_item in args.uniaxial_size:
                try:
                    p_name, hkl_str = u_item.split(":")  # noqa: PU001
                    axis = [int(x.strip()) for x in hkl_str.split(",")]
                    target_p = None
                    for p in gpx.phases():
                        if p.name.lower() == p_name.lower().strip():
                            target_p = p
                            break
                    if target_p and hist.name in target_p.data.get("Histograms", {}):
                        hap = target_p.data["Histograms"][hist.name]
                        hap["Size"] = ["uniaxial", [0.1, 0.1], [True, True], axis]
                        logger.info(
                            f"Activated uniaxial crystallite size along {axis} for phase {target_p.name}"
                        )
                except (ValueError, KeyError, IndexError, AttributeError, TypeError) as e_uni:
                    logger.warning(f"Could not apply uniaxial size '{u_item}': {e_uni}")

    elif stage_name == "8_Preferred_Orientation":
        init_md = args.init_md
        # Decouple: disable unconstrained atomic displacement refinement
        for p in gpx.phases():
            p.set_refinements({"Atoms": {"all": ""}})
            for at in p.atoms():
                rec = G2AtomRecord(at)
                if rec.uiso < 0.001 or rec.uiso > 0.5:
                    rec.uiso = 0.01
                rec.refinement_flag = ""

        # Support multiple texture axes simultaneously
        target_axes_map: dict[str, list[int]] = {}
        raw_axes = list(args.texture_axis)
        for tax in raw_axes:
            try:
                p_part, hkl_part = tax.split(":")  # noqa: PU001
                target_axes_map[p_part.lower().strip()] = [
                    int(x.strip()) for x in hkl_part.split(",")
                ]
            except (ValueError, KeyError, IndexError, AttributeError, TypeError) as e_tax:
                logger.warning(f"Could not parse texture axis '{tax}': {e_tax}")

        if target_axes_map:
            for p in gpx.phases():
                p_norm = p.name.lower().strip()
                if hist.name in p.data.get("Histograms", {}):
                    hap = p.data["Histograms"][hist.name]
                    if p_norm in target_axes_map:
                        axis = target_axes_map[p_norm]
                        hap["Pref.Ori."][3] = axis
                        if hap["Pref.Ori."][1] == 1.0:
                            hap["Pref.Ori."][1] = float(init_md)
                        hap["Pref.Ori."][2] = True
                        logger.info(
                            f"Enabled March-Dollase on {p.name} along {axis} (init MD={init_md})"
                        )
                    else:
                        hap["Pref.Ori."][2] = False
            recipe = {}  # already configured directly on target phases
        else:
            recipe = build_stage_recipe(stage_name)

    elif stage_name == "3_Unit_Cell":
        qpa_now = get_project_qpa(gpx, hist)
        unlocked_phases: list[str] = []
        locked_phases: list[str] = []
        for p in gpx.phases():
            if is_pawley_phase(p):
                p.data["General"]["Pawley ref"] = False
            wt_pct = float(qpa_now.get(p.name, (100.0, None))[0])
            if (
                is_pawley_phase(p)
                or len(gpx.phases()) == 1
                or wt_pct >= PHASE_MICRO_CELL_LOCK_PERCENT
            ):
                p.set_refinements({"Cell": True})
                unlocked_phases.append(p.name)
            else:
                p.set_refinements({"Cell": False})
                locked_phases.append(f"{p.name} ({wt_pct:.2f}%)")
        if locked_phases:
            logger.info(
                "3_Unit_Cell locked unit cell for microphase(s) (< {} wt%): {}",
                PHASE_MICRO_CELL_LOCK_PERCENT,
                ", ".join(locked_phases),
            )
        if not unlocked_phases and gpx.phases():
            gpx.phases()[0].set_refinements({"Cell": True})
        recipe = {}

    elif stage_name == "12_Final_Scale" and has_pawley:
        for p in gpx.phases():
            if not is_pawley_phase(p):
                p.set_HAP_refinements({"Scale": True})
        recipe = {}
    else:
        recipe = build_stage_recipe(
            stage_name,
            refine_displacement=bool(args.refine_displacement),
        )

    # Parameter injection escape hatch via --override-json
    if args.override_json:
        try:
            overrides = json.loads(args.override_json)
            # 1. Phase-level overrides: {"phases": {"SmS": {"Pref.Ori.": [...]}}}
            for p_name, p_params in overrides.get("phases", {}).items():
                target_p = None
                for p in gpx.phases():
                    if p.name.lower() == p_name.lower().strip():
                        target_p = p
                        break
                if target_p:
                    if "Histograms" in target_p.data and hist.name in target_p.data["Histograms"]:
                        hap = target_p.data["Histograms"][hist.name]
                        for k, v in p_params.items():
                            hap[k] = v
                            logger.info(
                                f"Injected phase override: {target_p.name} -> HAP['{k}'] = {v}"
                            )
                    if "General" in target_p.data:
                        for k, v in p_params.get("General", {}).items():
                            target_p.data["General"][k] = v
                            logger.info(
                                f"Injected General override: {target_p.name} -> General['{k}'] = {v}"
                            )
            # 2. Histogram-level overrides: {"histogram": {"SampleParameters": ...}}
            if "histogram" in overrides:
                for k, v in overrides["histogram"].items():
                    if k in hist.data:
                        hist.data[k] = v
                    else:
                        setattr(hist, k, v)
                    logger.info(f"Injected histogram override: {k} = {v}")
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError, ValueError) as e_ovr:
            logger.warning(f"Could not apply --override-json: {e_ovr}")

    # Apply native Box Constraints (parmMin, parmMax)
    from stage_gate_rietveld import set_param_limit

    for s_idx, s_phase in enumerate(gpx.phases()):
        apply_box_constraints(gpx, phase_id=s_idx, hist_id=0)
        if is_pawley_phase(s_phase):
            set_param_limit(gpx, "parmMin", f"{s_idx}:0:Size;i", 0.015)
            set_param_limit(gpx, "parmMax", f"{s_idx}:0:Size;i", 0.150)

    # 3. Execute refinement via native do_refinements
    diverged = False
    fail_reason = ""
    last_fit_results = None
    fatal_lines: list[str] = []
    try:
        with capture_engine_notes(f"step:{stage_name}") as fatal_lines:
            if stage_name == StageId.FINAL_SCALE and has_pawley:
                for p in gpx.phases():
                    if is_pawley_phase(p):
                        p.data["General"]["Pawley ref"] = True
                        p.set_refinements({"Cell": False})
                        if hist.name in p.data.get("Histograms", {}):
                            hap = p.data["Histograms"][hist.name]
                            if "Size" in hap and len(hap["Size"]) >= 3:
                                hap["Size"][1][0] = float(
                                    np.clip(float(hap["Size"][1][0]), 0.015, 0.150)
                                )
                                hap["Size"][2][0] = False
                            if "Mustrain" in hap and len(hap["Mustrain"]) >= 3:
                                hap["Mustrain"][1][0] = 0.0
                                hap["Mustrain"][2][0] = False
                gpx.set_Controls("cycles", 5)
                gpx.refine(makeBack=False)
                extract_cell_parameters(gpx)
                sync_pawley_reflections_to_hist(gpx, hist)
            else:
                gpx.do_refinements([{"set": recipe}])
                extract_cell_parameters(gpx)
                if has_pawley:
                    for p in gpx.phases():
                        if is_pawley_phase(p) and hist.name in p.data.get("Histograms", {}):
                            hap = p.data["Histograms"][hist.name]
                            if "Size" in hap and len(hap["Size"]) >= 2:
                                hap["Size"][1][0] = float(
                                    np.clip(float(hap["Size"][1][0]), 0.015, 0.150)
                                )
                    if stage_name not in (StageId.UNIT_CELL, StageId.SAMPLE_PROFILE_XY):
                        clamp_pawley_reflections(gpx, lock_nonpositive=False)
                    sync_pawley_reflections_to_hist(gpx, hist)
        with suppress(*g2_safe_errors()):
            last_fit_results = gpx.get_LastFitResults()
    except Exception as e_refine:  # noqa: BLE001
        diverged = True
        fail_reason = f"Refinement exception: {e_refine}"

    engine_error = fatal_lines[0] if fatal_lines else ""
    with suppress(OSError):
        engine_listing = project_path.with_suffix(".lst")
        if not engine_error and engine_listing.exists():
            engine_text = engine_listing.read_text(encoding="utf-8", errors="replace")
            for marker in (
                "Invalid metric tensor",
                "Invalid cell metric tensor",
                "Refinement failed",
                "no reflections in data range",
            ):
                if marker in engine_text:
                    engine_error = marker
                    break
    if engine_error and not diverged:
        diverged = True
        fail_reason = f"engine reported '{engine_error}' during refinement"
        # Один и тот же отказ встречается, когда файл проекта уже повреждён: движок
        # отказывает на первой же итерации любой стадии, называя фазу, чья ячейка
        # при этом нормальна. Пересборка из data/ и CIF — единственный измеренно
        # работающий выход, и он должен быть назван здесь, а не искаться заново.
        if "metric tensor" in engine_error.lower():
            logger.error(
                "Отказ метрического тензора на первой итерации обычно означает "
                "повреждённый файл проекта, а не испорченную ячейку: проверьте "
                "`status` и, если любая стадия падает так же, пересоберите проект — "
                "`init --data <измерение>` и `add-phase` для каждой фазы, затем "
                "последовательность 1,3,5,7 до плато и финальная стадия 1."
            )

    # Следы самого уточнения: сколько переменных удержал фит и создал ли он
    # ковариацию. GSAS-II возвращает пустую covMatrix и при нулевом числе
    # переменных, и при вырожденной нормальной матрице (GSASIIstrMain.RefineCore),
    # поэтому «стадия прошла» без этих двух чисел ничего не значит.
    varied_parameters = 0
    if (
        isinstance(last_fit_results, (tuple, list))
        and len(last_fit_results) >= 2
        and isinstance(last_fit_results[1], dict)
    ):
        varied_parameters = len(last_fit_results[1])
    cov_raw = (gpx.data.get("Covariance") or {}).get("data", {}).get("covMatrix")
    covariance_present = cov_raw is not None and len(getattr(cov_raw, "shape", (0,))) > 0
    flagged_corrs, cond_num, is_cond_singular = (  # noqa: PU001
        StageGateRietveld().audit_project_covariance(gpx)
        if covariance_present
        else ([], 1.0, False)
    )
    if flagged_corrs:
        logger.warning(
            "Stage '{}': high parameter correlations (|rho| > 0.85): {}",
            stage_name,
            "; ".join(flagged_corrs[:5]),
        )
    if is_cond_singular and np.isfinite(cond_num):
        logger.warning(
            "Stage '{}': ill-conditioned covariance matrix (kappa = {:.2e} > 1e5)",
            stage_name,
            cond_num,
        )
    refinement_anomaly = ""
    if not diverged:
        if varied_parameters == 0:
            refinement_anomaly = (
                "уточнение не удержало ни одной переменной: R_wp описывает расчёт "
                "модели, а не подгонку"
            )
        elif not covariance_present:
            refinement_anomaly = (
                "ковариационная матрица не создана: нормальная матрица вырождена, "
                "esd параметров не определены"
            )

    # 4. Check convergence guards
    new_r_wp = 99.0
    new_r_p = 99.0
    new_gof = 1.0
    new_chi2 = 1.0
    new_dw = 2.0
    new_r_exp = 1.0
    new_r_ratio = 1.0
    if not diverged:
        try:
            new_curves = extract_profile_curves(hist)
            vary_cnt = len(gpx.data.get("Controls", {}).get("VaryList", []))
            new_m = calculate_profile_metrics(new_curves, n_free_parameters=max(1, vary_cnt))
            new_r_wp = new_m.r_wp_percent
            new_r_p = new_m.r_p_percent
            new_gof = new_m.gof
            new_chi2 = new_m.reduced_chi2
            new_dw = new_m.durbin_watson
            new_r_exp = new_m.r_exp_percent
            new_r_ratio = new_m.r_wp_over_r_exp_ratio

            # Divergence guard: R_wp exploded or NaN
            if np.isnan(new_r_wp) or (prev_r_wp < 90.0 and new_r_wp > prev_r_wp * 1.30):
                diverged = True
                fail_reason = f"R_wp exploded from {prev_r_wp:.2f}% to {new_r_wp:.2f}%"

            # Check unphysical lattice drift
            cells = extract_cell_parameters(gpx)
            for p_name, cell_dict in cells.items():
                if "a" in cell_dict and (cell_dict["a"] < 1.5 or cell_dict["a"] > 50.0):
                    diverged = True
                    fail_reason = (
                        f"Unphysical lattice parameter a={cell_dict['a']:.3f} in phase {p_name}"
                    )
                    break

            # Порог 5 %, а не 2 %: измерено, что фаза, взятая из CIF другого
            # образца, законно уточняется на 3 % (Sm5(SiO4)3O: c 7.0571 -> 6.8477),
            # и порог 2 % откатывал такую стадию как деградацию. Деградация же,
            # которую надо ловить, давала +15 % (c 6.6968 -> 7.7183).
            #
            # Ячейка, уехавшая на проценты, — это уже другая фаза, а не уточнение.
            # Измерено на пересобранном образце: у минорной фазы (3.85 масс. %)
            # ячейка ушла с a=3.8811, c=6.6968 на a=3.7281, c=7.7183 (-4 %, +15 %),
            # после чего её доля выросла с 3.85 до 34 %, а R_wp застрял на 7.38 %
            # вместо 5.68 %. Данные не ограничивают ячейку примеси, и оптимизатор
            # использует её как убежище. Такой шаг откатывается с названной причиной.
            if not diverged and stage_name == "3_Unit_Cell":
                for p_name, cell_dict in cells.items():
                    for axis in ("a", "c"):
                        before = prev_state_dict.get(f"{p_name}_{axis}")
                        after = cell_dict.get(axis)
                        if before is None or after is None or before <= 0:
                            continue
                        drift = abs(after - before) / before
                        if drift > CELL_DRIFT_LIMIT:
                            diverged = True
                            fail_reason = (
                                f"Cell of phase {p_name} moved {drift * 100:.1f}% along {axis} "
                                f"({before:.4f} -> {after:.4f}); a refinement step this large "
                                "is a different phase, not a better model"
                            )
                            break
                    if diverged:
                        break

            # Clamp post-refinement Uiso to positive physical bound
            for p in gpx.phases():
                for at in p.atoms():
                    rec = G2AtomRecord(at)
                    if rec.uiso < 0.001 or rec.uiso > 0.15:
                        logger.warning(
                            f"Unphysical post-refinement Uiso={rec.uiso:.4f} for {p.name}:{rec.label}, resetting to 0.01"
                        )
                        rec.uiso = 0.01
                # Профильные величины тоже проверяются после уточнения, а не только
                # до него: граница ограничивает следующий шаг, но шаг может уйти за
                # неё. Измерено: у фазы осталось Mustrain = -83596.8 при Size на
                # нижней границе, и модель застряла на 7.38 % вместо 5.68 %.
                clamped = clamp_stored_parameters(gpx, p, hist)
                if clamped:
                    logger.warning(
                        f"Stage '{stage_name}': unphysical stored value(s) clamped for {p.name}: "
                        + ", ".join(clamped)
                    )

            # Report reflections sharper than the canonical hardware limit (Stage 7).
            # The report is shared with the declarative params path, which used to
            # skip this check entirely.
            if stage_name == "7_Sample_Profile_XY":
                for message in instrumental_lower_bound_report(hist):
                    logger.warning(message)
        except (ValueError, KeyError, IndexError, AttributeError) as e_guard:
            diverged = True
            fail_reason = f"Guard check error: {e_guard}"

    # 5. Handle rollback or commit
    history = load_sidecar_history(project_path)
    if diverged and not args.force:
        logger.warning(
            f"Stage '{stage_name}' diverged ({fail_reason}). Rolling back from disk checkpoint."
        )
        shutil.copyfile(str(ckpt_path), str(project_path))
        history["iterations"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "stage": stage_name,
                "status": str(RefinementStatus.REVERTED),
                "reason": fail_reason,
                "r_wp_before": prev_r_wp,
            }
        )
        save_sidecar_history(project_path, history)
        print(
            json.dumps(
                {
                    "status": str(RefinementStatus.REVERTED),
                    "stage": stage_name,
                    "reason": fail_reason,
                    "r_wp": prev_r_wp,
                },
                indent=2,
            )
        )
        listing = project_path.with_suffix(".lst")
        if listing.exists():
            print(f"GSAS-II listing kept for diagnostics: {listing}", file=sys.stderr)
        return None

    # Аномалии стадии: только превышение порогов, без повтора метрик
    _report_stage_anomalies(gpx, history, stage_name)

    post_state_dict = _snapshot_project_state(gpx, hist)

    # Success: commit changes to disk and update checkpoint to current state
    gpx.save()

    # Метрика обязана описывать то, что записано на диск. Отчёт, снятый до
    # сохранения, расходился с сохранённым проектом (17.69 против 19.03 %):
    # агент читал его как ворота принятия.
    try:
        from GSASII import GSASIIscriptable as _G2sc

        verify_gpx = _G2sc.G2Project(str(project_path))
        verify_hist = verify_gpx.histograms()[0]
        verify_m = calculate_profile_metrics(extract_profile_curves(verify_hist))
        if abs(verify_m.r_wp_percent - new_r_wp) > 0.05:
            logger.warning(
                f"Stage '{stage_name}': отчёт дал R_wp={new_r_wp:.2f}%, "
                f"сохранённый проект даёт {verify_m.r_wp_percent:.2f}%; "
                "дальше используется значение с диска"
            )
            new_r_wp = verify_m.r_wp_percent
            new_r_p = verify_m.r_p_percent
            new_gof = verify_m.gof
            new_chi2 = verify_m.reduced_chi2
            new_dw = verify_m.durbin_watson
            new_r_exp = verify_m.r_exp_percent
            new_r_ratio = verify_m.r_wp_over_r_exp_ratio
    except (ValueError, KeyError, IndexError, AttributeError, TypeError, OSError) as e_verify:
        logger.debug(f"Post-save metric verification skipped: {e_verify}")

    shutil.copyfile(str(project_path), str(ckpt_path))
    if refinement_anomaly:
        logger.warning(
            f"Stage '{stage_name}': {refinement_anomaly} "
            f"(переменных {varied_parameters}, ковариация "
            f"{'есть' if covariance_present else 'нет'})"
        )
        # Листинг при аномалии — единственный документ того, что произошло в
        # движке (Maximum shift/esd, singular matrix, Bad parameter).
        listing = project_path.with_suffix(".lst")
        if listing.exists():
            logger.warning(f"GSAS-II listing kept for diagnostics: {listing}")
    else:
        # Листинг GSAS-II — побочный продукт: метрики уже сняты из объектов проекта.
        discard_gsas_listing(project_path)

    history["last_stage"] = stage_name
    history["current_r_wp"] = new_r_wp
    history["current_gof"] = new_gof
    if stage_name not in history.get("stages_completed", []):
        history.setdefault("stages_completed", []).append(stage_name)

    # Rule 9 считается до записи истории: критерий попадает в тот же артефакт,
    # который затем читает экспорт, а не только в печать таблицы.
    rule9_problems: list[str] = []
    if (
        last_fit_results
        and isinstance(last_fit_results, (tuple, list))
        and len(last_fit_results) >= 2
    ):
        rule9_problems = shift_significance_problems(
            last_fit_results[0], last_fit_results[1], RULE9_SHIFT_SIGMA_RATIO
        )

    history["iterations"].append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": stage_name,
            "status": "SUCCESS",
            "r_wp": new_r_wp,
            "delta_r_wp": round(prev_r_wp - new_r_wp, 3) if prev_r_wp < 90.0 else 0.0,
            "r_exp": round(new_r_exp, 2),
            "r_p": round(new_r_p, 2),
            "chi_square": new_chi2,
            "r_wp_over_r_exp": round(new_r_ratio, 2),
            "gof": new_gof,
            "durbin_watson": new_dw,
            "rule9_problems": rule9_problems,
            "varied_parameters": varied_parameters,
            "covariance_present": covariance_present,
            "correlations_flagged": flagged_corrs,
            "condition_number": round(cond_num, 2) if np.isfinite(cond_num) else None,
            "refinement_anomaly": refinement_anomaly or None,
        }
    )
    save_sidecar_history(project_path, history)

    # Print parameter delta shifts table to stderr to keep stdout a clean JSON stream
    print(f"\n--- Parameter Telemetry & Shifts [{stage_name}] ---", file=sys.stderr)
    if (
        last_fit_results
        and isinstance(last_fit_results, (tuple, list))
        and len(last_fit_results) >= 2
    ):
        shifts_dict, sigmas_dict = last_fit_results[0], last_fit_results[1]
        print(
            f"{'Parameter':<32} | {'Shift (Δ)':>14} | {'ESD (σ)':>14} | {'|Δ|/σ':>10} | {f'Rule 9 (<{RULE9_SHIFT_SIGMA_RATIO})':>15}",
            file=sys.stderr,
        )
        print("-" * 92, file=sys.stderr)
        pwl_count = 0
        pwl_max_ratio = 0.0
        for parm in sorted(shifts_dict.keys()):
            sh = float(shifts_dict[parm])
            sig = float(sigmas_dict.get(parm, 0.0))
            ratio = abs(sh) / max(1e-12, sig) if sig > 0 else 0.0
            if ":PWLref:" in parm:
                pwl_count += 1
                pwl_max_ratio = max(pwl_max_ratio, ratio)
                continue
            status_r9 = (
                "PASS" if abs(sh) / max(1e-12, sig) < RULE9_SHIFT_SIGMA_RATIO else "CONVERGING"
            )
            print(
                f"{parm:<32} | {sh:>+14.6f} | {sig:>14.6f} | {ratio:>10.3f} | {status_r9:>15}",
                file=sys.stderr,
            )
        if pwl_count > 0:
            pwl_status = "PASS" if pwl_max_ratio < RULE9_SHIFT_SIGMA_RATIO else "CONVERGING"
            pwl_label = f"PWLref ({pwl_count} reflections)"
            print(
                f"{pwl_label:<32} | {'(summary)':>14} | {'(max ratio)':>14} | {pwl_max_ratio:>10.3f} | {pwl_status:>15}",
                file=sys.stderr,
            )
        print("-" * 92 + "\n", file=sys.stderr)
        rule9_problems = shift_significance_problems(
            shifts_dict, sigmas_dict, RULE9_SHIFT_SIGMA_RATIO
        )
        total, converged, converging = summarize_rule9(  # noqa: PU001
            shifts_dict, sigmas_dict
        )
        if rule9_problems:
            logger.warning(
                f"Rule 9: {len(rule9_problems)} parameter(s) not confirmed by data "
                f"in stage '{stage_name}'; first: {rule9_problems[0]}"
            )
        print(
            f"Convergence: {converged}/{total} parameters within Rule 9 "
            f"(<{RULE9_SHIFT_SIGMA_RATIO}); {converging} still CONVERGING",
            file=sys.stderr,
        )
    else:
        print(
            f"{'Parameter':<28} | {'Before':>12} | {'After':>12} | {'Shift (Δ)':>12}",
            file=sys.stderr,
        )
        print("-" * 72, file=sys.stderr)
        for k in sorted(set(prev_state_dict.keys()) | set(post_state_dict.keys())):
            v_pre = prev_state_dict.get(k, 0.0)
            v_post = post_state_dict.get(k, 0.0)
            delta = v_post - v_pre
            if abs(delta) > 1e-7 or k in ("DisplaceX", "Zero") or "scale" in k:
                print(
                    f"{k:<28} | {v_pre:>12.6f} | {v_post:>12.6f} | {delta:>+12.6f}", file=sys.stderr
                )
        print("-" * 72 + "\n", file=sys.stderr)

    logger.info(
        f"Stage '{stage_name}' completed: R_wp={new_r_wp:.2f}%, R_exp={new_r_exp:.2f}%, GoF={new_gof:.2f}"
    )
    result_dict = {
        "status": "SUCCESS",
        "stage": stage_name,
        "r_wp_percent": new_r_wp,
        "delta_r_wp": round(prev_r_wp - new_r_wp, 3) if prev_r_wp < 90.0 else 0.0,
        "r_exp_percent": round(new_r_exp, 2),
        "r_p_percent": round(new_r_p, 2),
        "r_wp_over_r_exp": round(new_r_ratio, 2),
        "gof": new_gof,
        "chi_square": new_chi2,
        "durbin_watson": new_dw,
    }
    print(json.dumps(result_dict, indent=2))
    return result_dict


def cmd_step(args: SimpleNamespace) -> int:
    """Execute a single Stage-Gate refinement step or sequence with disk-level rollback."""
    if getattr(args, "keep_gsas_listing", False):
        os.environ[KEEP_LISTING_ENV] = "1"
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    if args.sequence:
        raw_stages = [s.strip() for s in args.sequence.split(",") if s.strip()]
        stages_to_run = [normalize_stage_name(s) for s in raw_stages]
        for st in stages_to_run:
            if st not in CANONICAL_STAGES:
                logger.error(f"Invalid stage '{st}' in sequence. Valid stages: {CANONICAL_STAGES}")
                return 1
    else:
        st_name = normalize_stage_name(args.stage)
        if st_name not in CANONICAL_STAGES:
            logger.error(f"Invalid stage '{args.stage}'. Valid stages: {CANONICAL_STAGES}")
            return 1
        stages_to_run = [st_name]

    # 1. Physical disk backup checkpoint before optimization
    ckpt_path = get_checkpoint_path(project_path)
    shutil.copyfile(str(project_path), str(ckpt_path))
    logger.debug(f"Created physical disk checkpoint: {ckpt_path}")

    gpx = G2sc.G2Project(str(project_path))
    hists = gpx.histograms()
    if not hists:
        logger.error("Project contains no histograms.")
        return 1
    hist = hists[0]

    # 2. Решение о числе проходов. Маршрут (--sequence) по умолчанию идёт до
    #    плато: один проход — это одна итерация метода, а не сходимость. Одиночная
    #    стадия (--stage) означает буквально «освободи и уточни», поэтому для неё
    #    умолчание — один проход.
    explicit_passes = getattr(args, "passes", None)
    if explicit_passes is not None:
        max_passes = max(1, int(explicit_passes))
        until_plateau = False
    elif len(stages_to_run) > 1:
        max_passes = max(1, int(getattr(args, "max_passes", 5) or 5))
        # Флаг по умолчанию None означает «не задан», а не «выключено»: иначе
        # умолчание для последовательности не работало бы вовсе.
        requested = getattr(args, "until_plateau", None)
        until_plateau = True if requested is None else bool(requested)
    else:
        max_passes = 1
        until_plateau = False
    plateau_delta = float(getattr(args, "plateau_delta", 0.1) or 0.1)

    # Лучшее найденное состояние хранится отдельным файлом: «до плато» без памяти
    # лучшего фиксирует последнюю точку, а не лучшую (измерено: 8.17 -> 9.24 %).
    best_path = project_path.parent / f"{project_path.stem}_best.gpx"
    best_r_wp: float | None = None
    history: list[float] = []
    decision = PASS_CONTINUE

    for pass_index in range(1, max_passes + 1):
        for stage_name in stages_to_run:
            res = _run_stage_in_project(gpx, hist, stage_name, args, project_path, ckpt_path)
            if res is None:
                return 1
        if not isinstance(res, dict):
            logger.error(f"Маршрут остановлен: стадия '{stage_name}' не дала результата")
            return 1
        current = float(res.get("r_wp_percent", 0.0))
        history.append(current)
        if best_r_wp is None or current < best_r_wp:
            best_r_wp = current
            shutil.copyfile(str(project_path), str(best_path))
        if not until_plateau:
            break
        decision, _best_index = decide_next_pass(  # noqa: PU001
            history, plateau_delta
        )
        if decision != PASS_CONTINUE:
            break
        if pass_index == max_passes:
            decision = PASS_EXHAUSTED

    if until_plateau:
        if decision in (PASS_PLATEAU, PASS_REGRESSED, PASS_EXHAUSTED) and best_path.exists():
            shutil.copyfile(str(best_path), str(project_path))
            logger.info(
                f"Маршрут остановлен ({decision}): проходы {history}, "
                f"принято лучшее состояние R_wp={best_r_wp}%"
            )
        # Промежуточный снимок лучшего состояния убирается в любом случае: после
        # восстановления он дублирует проект, а до восстановления — уже не нужен.
        if best_path.exists():
            best_path.unlink()

    # Каждый проход — отдельная запись истории: без неё Rule 9 считается по
    # последнему проходу без контекста, и «сошлось» нечем проверить.
    history_doc = load_sidecar_history(project_path)
    route_decision = str(decision if until_plateau else PassDecision.SINGLE_PASS)
    history_doc["iterations"].append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": f"route:{len(stages_to_run)} stages",
            "status": str(RefinementStatus.SUCCESS),
            "passes": len(history),
            "route_r_wp_per_pass": [round(value, 3) for value in history],
            "route_decision": route_decision,
            "route_best_r_wp": round(best_r_wp, 3) if best_r_wp is not None else None,
        }
    )
    save_sidecar_history(project_path, history_doc)
    if len(history) > 1:
        print(
            json.dumps(
                {
                    "status": str(RefinementStatus.SUCCESS),
                    "passes": len(history),
                    "decision": route_decision,
                    "r_wp_per_pass": [round(value, 3) for value in history],
                    "best_r_wp": round(best_r_wp, 3) if best_r_wp is not None else None,
                },
                indent=2,
            )
        )

    if ckpt_path.exists():
        ckpt_path.unlink()  # remove checkpoint when all stages complete successfully

    if len(stages_to_run) > 1:
        logger.info(f"Refinement sequence completed: {stages_to_run}, passes={len(history)}")

    return 0


# -----------------------------------------------------------------------------
# Subcommand: diagnose
# -----------------------------------------------------------------------------
def format_recommended_actions(recommended_actions: list[dict[str, Any]]) -> str:
    """Render the mandatory next steps, printing a command only where one exists.

        A reference card is not a structure, so its action deliberately carries no command:
        ``add-phase`` pointed at a card number would fail, and promising it would be worse
        than saying what has to happen first. Reading ``act["command"]`` unconditionally
    turned that honest omission into a ``KeyError`` raised after the whole report had
        already been printed, so a successful diagnosis exited nonzero.
    """
    lines: list[str] = []
    for act in recommended_actions:
        action_type = act.get("action_type", "ACTION")
        rationale = act.get("rationale", "")
        lines.append(f"  * [{action_type}] {rationale}")
        command = act.get("command")
        if command:
            lines.append(f"    Command: {command}")
    return "\n".join(lines)


def cmd_diagnose(args: SimpleNamespace) -> int:
    """Classify residual symptoms and evaluate candidate CIFs against unindexed peaks."""
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    gpx = G2sc.G2Project(str(project_path))
    hists = gpx.histograms()
    if not hists:
        logger.error("Project contains no histograms.")
        return 1
    hist = hists[0]

    curves = extract_profile_curves(hist)
    refls = extract_bragg_reflections(hist)
    scales = extract_phase_scales(gpx)

    phase_bragg_dict = build_scaled_phase_bragg_peaks(curves, refls, scales)
    observed_peaks = extract_peaks_robust(curves.two_theta, curves.y_obs).two_theta
    diag_res = diagnose_residuals(
        observed_peaks_2theta=observed_peaks,
        two_theta=curves.two_theta,
        y_obs=curves.y_obs,
        y_calc=curves.y_calc,
        wavelength=args.wavelength,
        phase_bragg_peaks=phase_bragg_dict,
        y_bkg=curves.y_bkg,
        angular_tolerance=args.angular_tolerance,
        snr_threshold=args.snr_threshold,
    )
    true_unindexed = [p.to_dict() for p in diag_res.true_unindexed_reflections]
    intensity_defects = [p.to_dict() for p in diag_res.intensity_mismatch_reflections]
    symptoms = [s.to_dict() for s in diag_res.symptoms]

    # --- Candidate pool: local CIFs and, when the environment declares one, the
    # reference card database. The pool is their union, because both sources answer
    # the same question -- which phase explains this residual -- and a caller should
    # not have to choose between them.
    candidate_matches: list[dict[str, Any]] = []
    new_element_candidates: list[dict[str, Any]] = []
    candidate_results: list[Any] = []
    candidates_dict: dict[str, Any] = {}
    if args.candidate_cifs:
        cif_pool_path = Path(args.candidate_cifs).resolve()
        cif_files: list[Path] = []
        if cif_pool_path.is_dir():
            cif_files = sorted(cif_pool_path.glob("*.cif"))
        elif cif_pool_path.is_file():
            cif_files = [cif_pool_path]
        candidates_dict.update({cf.stem: cf for cf in cif_files})

    try:
        elem_filter = build_element_filter(
            all_elements=args.elements,
            any_elements=args.any,
            none_elements=args.none,
            strict=args.strict,
        )
    except ValueError as exc:
        logger.error("Invalid element filter: {}", exc)
        return 2

    # The reference pool is requested explicitly and is narrowed while the records are
    # still cards.
    if args.pdf2:
        if elem_filter is None:
            logger.error(
                "--pdf2 scores the reference card database; give an "
                "element criterion (--elements / --any / --none) so the pool is narrowed "
                "while the records are still cards"
            )
            return 2
        unindexed_sorted = sorted(
            true_unindexed,
            key=lambda item: float(item.get("residual_height", 0.0)),
            reverse=True,
        )[:3]
        query_peaks = (
            np.sort(np.asarray([float(item["2theta"]) for item in unindexed_sorted], dtype=float))
            if unindexed_sorted
            else None
        )
        raw_sf_pdf = getattr(args, "subfiles", None)
        sf_set_pdf = (
            frozenset(s.strip() for s in str(raw_sf_pdf).split(",") if s.strip())
            if raw_sf_pdf
            else None
        )
        try:
            candidates_dict.update(
                load_pdf2_candidates(
                    elem_filter,
                    query_two_theta=query_peaks,
                    tolerance_deg=float(args.angular_tolerance),
                    min_hits=min(2, len(unindexed_sorted)) if unindexed_sorted else 2,
                    wavelength=float(args.wavelength),
                    subfiles=sf_set_pdf,
                )
            )
        except RuntimeError as exc:
            logger.error("{}", exc)
            return 2

    if getattr(args, "hsrdb", False):
        if elem_filter is None:
            logger.error(
                "--hsrdb scores the COD HSRDB database; give an "
                "element criterion (--elements / --any / --none)"
            )
            return 2
        raw_sf_hs = getattr(args, "subfiles", None)
        sf_set_hs = (
            frozenset(s.strip() for s in str(raw_sf_hs).split(",") if s.strip())
            if raw_sf_hs
            else None
        )
        unindexed_sorted_hs = sorted(
            true_unindexed,
            key=lambda item: float(item.get("residual_height", 0.0)),
            reverse=True,
        )[:3]
        query_peaks_hs = (
            np.sort(
                np.asarray([float(item["2theta"]) for item in unindexed_sorted_hs], dtype=float)
            )
            if unindexed_sorted_hs
            else None
        )
        try:
            candidates_dict.update(
                load_hsrdb_candidates(
                    elem_filter,
                    query_two_theta=query_peaks_hs,
                    tolerance_deg=float(args.angular_tolerance),
                    min_hits=min(2, len(unindexed_sorted_hs)) if unindexed_sorted_hs else 2,
                    wavelength=float(args.wavelength),
                    subfiles=sf_set_hs,
                )
            )
        except RuntimeError as exc:
            logger.error("{}", exc)
            return 2

    candidates_dict = select_candidates(candidates_dict, elem_filter)

    if candidates_dict:
        engine = CandidateScoringEngine(
            wavelength=args.wavelength,
            angular_tolerance=args.angular_tolerance,
        )
        prob_field = ProfileProbabilityField.from_difference_curve(
            two_theta=curves.two_theta,
            y_obs=curves.y_obs,
            y_calc=curves.y_calc,
        )
        candidate_results = engine.score_candidates(
            two_theta=curves.two_theta,
            y_net=prob_field.y_net,
            sigma=prob_field.sigma,
            candidate_structures=candidates_dict,
        )
        candidate_matches = [cr.to_dict() for cr in candidate_results]
        # Ранжирование учитывает положение линий и интенсивности, но не состав:
        # фаза с посторонним элементом (YbS в пробе Sm–S) получает первое место
        # наравне с возможной. Делим, а не выбрасываем.
        known_elements = identified_elements(gpx)
        candidate_matches, new_element_candidates = partition_candidates(  # noqa: PU001
            candidate_matches, known_elements
        )
        if new_element_candidates:
            introduced = sorted(
                {el for match in new_element_candidates for el in (match.get("elements") or [])}
                - known_elements
            )
            logger.warning(
                f"Candidate pool: {len(new_element_candidates)} phase(s) introduce element(s) "
                f"absent from the accepted phases ({', '.join(introduced)}); "
                "reported separately as new_element_candidates"
            )

    vary_cnt = len(gpx.data.get("Controls", {}).get("VaryList", []))
    metrics = calculate_profile_metrics(curves, n_free_parameters=max(1, vary_cnt))

    # Build mandatory recommended actions
    recommended_actions: list[dict[str, Any]] = []
    if candidate_matches:
        vetted = [
            cm
            for cm in candidate_matches
            if not any("VETO" in f for f in cm.get("diagnostic_flags", []))
            and cm.get("score", 0.0) >= 30.0
        ]
        top_match = vetted[0] if vetted else candidate_matches[0]
        flags_desc = "; ".join(top_match.get("diagnostic_flags", []))
        score_desc = (
            f"Candidate phase {top_match['phase_name']} has Score {top_match['score']}% "
            f"(ML/TL={top_match['matched_lines']}/{top_match['total_lines']}, "
            f"flags: {flags_desc})."
        )
        match_file = Path(str(top_match["cif_file"]))
        if match_file.suffix.lower() == ".cif" and match_file.is_file():
            recommended_actions.append(
                {
                    "action_type": "ADD_CANDIDATE_PHASE",
                    "phase": top_match["phase_name"],
                    "command": f'uv run py {Path(__file__).name} add-phase "{project_path}" "{match_file}"',
                    "rationale": score_desc,
                }
            )
        else:
            # A reference card is not a structure: it carries d-spacings and an
            # ICSD/COD code, so it cannot be added as a phase directly. Saying so is
            # the honest recommendation; a command that pointed add-phase at a card
            # number would fail.
            recommended_actions.append(
                {
                    "action_type": "OBTAIN_STRUCTURE_FOR_CARD",
                    "phase": top_match["phase_name"],
                    "card": top_match["cif_file"],
                    "rationale": (
                        f"{score_desc} The match is reference card {top_match['cif_file']} "
                        "with its own unit cell and ICSD/COD code; obtain the structure "
                        "recorded on that card before adding the phase."
                    ),
                }
            )
    elif true_unindexed:
        recommended_actions.append(
            {
                "action_type": "EXPAND_SEARCH_POOL",
                "command": "uv run py .agents/skills/xrd-analysis/scripts/cif_retriever.py --elements <elements> --out-dir candidate_cifs",
                "rationale": f"Found {len(true_unindexed)} true unindexed peak(s) without match in local CIF pool.",
            }
        )

    symptom_codes = {s.get("code") for s in symptoms}

    # 1. Derivative residual (zero shift / cell mismatch)
    if "SYM_DERIVATIVE_RESIDUAL" in symptom_codes:
        recommended_actions.append(
            {
                "action_type": "REFINE_ZERO_SHIFT_AND_CELL",
                "stage": "2_Zero_Shift",
                "command": f'uv run py {Path(__file__).name} step "{project_path}" --stage 2_Zero_Shift',
                "rationale": "Derivative shape (S-curve) residual detected at Bragg reflections: refine zero shift and lattice parameters.",
            }
        )

    # 2. Background wiggle / ripple
    if "SYM_BACKGROUND_WIGGLE" in symptom_codes:
        recommended_actions.append(
            {
                "action_type": "REDUCE_BACKGROUND_COMPLEXITY",
                "stage": "1_Scale_and_Background",
                "command": f'uv run py {Path(__file__).name} step "{project_path}" --stage 1_Scale_and_Background --bkg-terms 3',
                "rationale": "Background ripple/overfitting detected: baseline has excessive inflections. Reduce Chebyshev polynomial terms.",
            }
        )

    # 3. Intensity mismatch / Preferred orientation
    if intensity_defects or "SYM_INTENSITY_MISMATCH" in symptom_codes:
        defect_phases = [
            d.get("matched_phase") for d in intensity_defects if d.get("matched_phase")
        ]
        proj_phases = gpx.phases()
        top_defect_phase = (
            Counter(defect_phases).most_common(1)[0][0]
            if defect_phases
            else (proj_phases[0].name if proj_phases else "Phase")
        )
        recommended_actions.append(
            {
                "action_type": "REFINE_PREFERRED_ORIENTATION",
                "phase": top_defect_phase,
                "stage": "8_Preferred_Orientation",
                "command": f'uv run py {Path(__file__).name} step "{project_path}" --stage 8_Preferred_Orientation --texture-axis "{top_defect_phase}:0,0,1" --init-md 0.95',
                "rationale": f"Severe intensity mismatch ({len(intensity_defects)} reflection(s)) detected on phase {top_defect_phase}. Refine preferred orientation (March-Dollase).",
            }
        )

    if metrics.durbin_watson < 1.0 and metrics.r_wp_percent > 10.0:
        recommended_actions.append(
            {
                "action_type": "REFINE_SAMPLE_PROFILE",
                "stage": "7_Sample_Profile_XY",
                "command": f'uv run py {Path(__file__).name} step "{project_path}" --stage 7_Sample_Profile_XY',
                "rationale": f"Low Durbin-Watson statistic ({metrics.durbin_watson:.2f}) indicates severe peak shape / profile mismatch.",
            }
        )

    # Native GSAS-II gradient & worst-fit parameter analysis.
    # Opt-in: ComputeWorstFit() is slow enough to hang on multiphase projects, so it runs
    # only under --gradients; the residual classification above stays instant.
    worst_fit_params: list[dict[str, Any]] = []
    if args.gradients and len(gpx.phases()) > 0:
        with suppress(*g2_safe_errors()):
            wf_res = gpx.ComputeWorstFit()
            if wf_res and isinstance(wf_res, (list, tuple)) and len(wf_res) >= 2:
                keys, derivs_dict = wf_res[0], wf_res[1]
                if isinstance(keys, (list, tuple)) and isinstance(derivs_dict, dict):
                    for k in keys[:5]:
                        d_val = derivs_dict.get(k)
                        grad = (
                            float(d_val[1])
                            if isinstance(d_val, (list, tuple)) and len(d_val) > 1
                            else (float(d_val) if isinstance(d_val, (int, float)) else 0.0)
                        )
                        worst_fit_params.append({"parameter": str(k), "gradient": grad})

    diagnosis_output = {
        "project": str(project_path),
        "r_wp_percent": metrics.r_wp_percent,
        "r_p_percent": metrics.r_p_percent,
        "gof": metrics.gof,
        "chi_square": metrics.reduced_chi2,
        "durbin_watson": metrics.durbin_watson,
        "symptoms": symptoms,
        "true_unindexed_count": len(true_unindexed),
        "intensity_defects_count": len(intensity_defects),
        "true_unindexed_peaks": true_unindexed,
        "intensity_defects": intensity_defects,
        "candidate_phase_matches": candidate_matches,
        "new_element_candidates": new_element_candidates,
        "recommended_actions": recommended_actions,
        "gradient_worst_fit": worst_fit_params,
    }

    if args.json:
        print_json(diagnosis_output)
    else:
        print("=" * 65)
        print(f"Rietveld Residual Diagnosis: {project_path.name}")
        print("=" * 65)
        print(
            f"R_wp: {metrics.r_wp_percent}% | R_p: {metrics.r_p_percent}% | "
            f"chi^2: {metrics.reduced_chi2:.2f} | GoF: {metrics.gof} | DW: {metrics.durbin_watson}"
        )
        print(f"True unindexed impurity peaks (> {args.snr_threshold}σ): {len(true_unindexed)}")
        for up in true_unindexed:
            print(
                f"  - 2θ = {up['2theta']}° (d = {up['d_spacing_angstrom']} Å, significance = {up['significance_sigma']}σ)"
            )

        if worst_fit_params:
            print("\nTop Gradient Parameters (ComputeWorstFit):")
            for wp in worst_fit_params:
                print(f"  * {wp['parameter']}: |grad| = {wp['gradient']:.4e}")

        if symptoms:
            print("\nDetected Crystallographic Symptoms:")
            for s in symptoms:
                print(
                    f"  * [{s.get('severity', 'INFO')}] {s.get('name', '')} ({s.get('code', '')}): {s.get('description', '')}"
                )

        rf_by_phase = phase_rf_percent(refls)
        rb_by_phase = phase_rb_percent(refls)
        if rf_by_phase or rb_by_phase:
            print("\nPhase R-factors:")
            print(f"  {'phase':<24}{'R_F,%':>9}{'R_B,%':>9}")
            for phase_name in sorted(set(rf_by_phase) | set(rb_by_phase)):
                rf_val = rf_by_phase.get(phase_name, float("nan"))
                rb_val = rb_by_phase.get(phase_name, float("nan"))
                print(f"  {phase_name:<24}{rf_val:>9.2f}{rb_val:>9.2f}")

        if candidate_results:
            print("\n" + format_candidate_table(candidate_results))
        elif candidate_matches:
            print("\nRecommended Candidate Phases from Pool:")
            for cm in candidate_matches:
                print(
                    f"  * {cm['phase_name']} ({cm['formula']}): explains {cm['explained_peaks_count']} peaks (2θ: {cm['explained_2thetas']})"
                )
                print(
                    f'    Command: uv run py {Path(__file__).name} add-phase "{project_path}" "{cm["cif_file"]}"'
                )

        if recommended_actions:
            print("\nRecommended Actions (Mandatory for Agent Iteration):")
            print(format_recommended_actions(recommended_actions))
        print("=" * 65)

    if args.out:
        out_f = Path(args.out).resolve()
        out_f.parent.mkdir(parents=True, exist_ok=True)
        with open(out_f, "w", encoding="utf-8") as f:
            if args.json or out_f.suffix.lower() == ".json":
                json.dump(
                    _json_safe(diagnosis_output), f, indent=2, ensure_ascii=False, allow_nan=False
                )
            else:
                f.write(format_diagnostic_report(diag_res))
        logger.info(f"Diagnosis written to {out_f}")

    return 0


# -----------------------------------------------------------------------------
# Subcommand: export
# -----------------------------------------------------------------------------
def cmd_export(args: SimpleNamespace) -> Any:
    """Тонкая обёртка: экспорт живёт в export.py и возвращает ExportSummary либо код ошибки."""
    from export import run_export

    return run_export(args)


def cmd_params(args: SimpleNamespace) -> int:
    """Handle declarative Refinement-as-Code parameter management via params.json."""
    if getattr(args, "keep_gsas_listing", False):
        os.environ[KEEP_LISTING_ENV] = "1"
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc
    from rietveld_params import (
        dump_project_params,
        format_parameter_table,
        refine_project_with_params,
    )

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    gpx = G2sc.G2Project(str(project_path))

    if args.action == "dump":
        out_path = (
            Path(args.out).resolve()
            if args.out
            else project_path.parent / f"{project_path.stem}_params.json"
        )
        doc = dump_project_params(gpx, hist_idx=0, preset=args.preset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(doc), f, indent=2, ensure_ascii=False, allow_nan=False)
        logger.info(f"Dumped project parameters to {out_path}")
        print("\n" + format_parameter_table(doc))
        print(f"\nSUCCESS: Dumped parameters to {out_path}")
        return 0

    elif args.action == "refine":
        params_file = (
            Path(args.params).resolve()
            if args.params
            else project_path.parent / f"{project_path.stem}_params.json"
        )
        if not params_file.exists():
            logger.error(f"Parameters file not found: {params_file}")
            return 1
        with open(params_file, "r", encoding="utf-8") as f:
            doc = json.load(f)

        # Deliberate profile restart. A profile can sit in a dead local minimum
        # (all broadening absorbed as microstrain while the size sits at its
        # ceiling), and the way out is to restart it from a physical value
        # instead of editing the JSON by hand.
        start_size = getattr(args, "profile_start_size", None)
        mustrain_cap_pct = getattr(args, "mustrain_max_percent", None)
        if start_size is not None or mustrain_cap_pct is not None:
            for phase_name, phase_cfg in (doc.get("phases") or {}).items():
                profile_cfg = phase_cfg.get("profile", {})
                if start_size is not None and "size" in profile_cfg:
                    profile_cfg["size"]["value"] = float(start_size)
                    logger.info(f"Profile start: {phase_name} size = {start_size} um")
                if mustrain_cap_pct is not None and "mustrain" in profile_cfg:
                    profile_cfg["mustrain"]["max"] = float(mustrain_cap_pct) * 10000.0
                    logger.info(f"Profile cap: {phase_name} microstrain <= {mustrain_cap_pct} %")

        cycles = int(args.cycles)
        with capture_engine_notes("params:refine"):
            updated_doc, diverged, fail_reason = refine_project_with_params(  # noqa: PU001
                gpx=gpx,
                params_dict=doc,
                cycles=cycles,
                project_path=project_path,
                sequential_scales=bool(args.sequential_scales),
            )

        if diverged:
            logger.error(f"Refinement diverged: {fail_reason}")
            print(f"FAILED: Refinement diverged ({fail_reason})")
            return 1

        # Успешное уточнение не оставляет листинг; при расхождении он сохранён выше.
        discard_gsas_listing(project_path)

        out_file = Path(args.out).resolve() if args.out else params_file
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(_json_safe(updated_doc), f, indent=2, ensure_ascii=False, allow_nan=False)

        print("\n" + format_parameter_table(updated_doc))
        print(f"\nSUCCESS: Refinement completed ({cycles} cycles). Parameters saved to {out_file}")
        return 0

    return 1


# -----------------------------------------------------------------------------
# CLI Dispatcher
# -----------------------------------------------------------------------------
