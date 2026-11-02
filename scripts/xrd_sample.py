"""Sample-level workspace: one sample is one folder.

Скилл работает с одним образцом за вызов. Образец — это папка с фиксированной
раскладкой; агент называет только операции, все пути и имена файлов вычисляет
этот модуль. Поэтому:

* микроменеджмент имён и структуры снят с агента;
* правила игнорирования описывают роли каталогов (`projects/`, `artifacts/`),
  а не типы файлов, и не меняются при смене состояния дифрактограммы;
* серии и манифестов с перечнем образцов нет: одна папка — один образец.

Раскладка папки образца::

    <sample>/
      data/          исходные измерения         — отслеживается git
      params.json    рецепт уточнения           — отслеживается git
      notes.md       выводы по образцу          — отслеживается git
      projects/      проект GSAS-II и сайдкары  — игнорируется
      artifacts/     кривые, CIF, PNG, листинги — игнорируется

Точка входа — ``python scripts/xrd_sample.py <команда>``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from constants import DEFAULT_WAVELENGTH_KA1
from loguru import logger

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DATA_DIR = "data"
PROJECTS_DIR = "projects"
ARTIFACTS_DIR = "artifacts"
PARAMS_FILE = "params.json"
NOTES_FILE = "notes.md"

#: Допуск сверки метрик при приёме существующего набора файлов, процент R_wp.
ADOPT_RWP_TOLERANCE = 0.5


@dataclass(frozen=True)
class SampleLayout:
    """Вычисленные пути одного образца: единственное место, знающее раскладку."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / DATA_DIR

    @property
    def projects_dir(self) -> Path:
        return self.root / PROJECTS_DIR

    @property
    def artifacts_dir(self) -> Path:
        return self.root / ARTIFACTS_DIR

    @property
    def params(self) -> Path:
        return self.root / PARAMS_FILE

    @property
    def notes(self) -> Path:
        return self.root / NOTES_FILE

    @property
    def project(self) -> Path:
        """Проект GSAS-II. Имя совпадает с именем папки образца."""
        return self.projects_dir / f"{self.root.name}.gpx"

    @property
    def history(self) -> Path:
        return self.projects_dir / f"{self.root.name}_history.json"

    @property
    def diagnosis(self) -> Path:
        return self.projects_dir / f"{self.root.name}_diag.json"

    def prepare(self) -> None:
        """Создаёт рабочие каталоги; вызывается только путями init/adopt."""
        for directory in (self.data_dir, self.projects_dir, self.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_sample(self) -> None:
        """Отказывает, если каталог не выглядит папкой образца."""
        markers = [self.data_dir, self.projects_dir, self.params]
        if not any(marker.exists() for marker in markers):
            raise SystemExit(
                f"[Ошибка] {self.root} не является папкой образца: нет ни "
                f"{DATA_DIR}/, ни {PROJECTS_DIR}/, ни {PARAMS_FILE}. "
                "Запустите команду в папке образца или укажите --dir."
            )


def open_sample(directory: str | Path | None = None) -> SampleLayout:
    """Возвращает раскладку образца для каталога (по умолчанию — текущего)."""
    root = Path(directory or Path.cwd()).resolve()
    if not root.exists():
        raise SystemExit(f"[Ошибка] Каталог образца не найден: {root}")
    return SampleLayout(root=root)


def _wavelength(layout: SampleLayout) -> float:
    """Длина волны из рецепта; иначе значение по умолчанию движка."""
    if not layout.params.exists():
        return DEFAULT_WAVELENGTH_KA1
    try:
        document = json.loads(layout.params.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_WAVELENGTH_KA1
    for key in ("wavelength", "wavelength_ka1"):
        value = document.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    histogram = document.get("histogram")
    if isinstance(histogram, dict):
        value = histogram.get("wavelength")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return DEFAULT_WAVELENGTH_KA1


# ---------------------------------------------------------------------------
# Команды: пути подставляет resolve-слой, агент передаёт только смысл.
# ---------------------------------------------------------------------------


def sample_init(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    data = Path(args.data).resolve()
    if not data.exists():
        logger.error(f"Measurement file not found: {data}")
        return 1
    layout.prepare()
    return core.cmd_init(
        SimpleNamespace(
            data=data,
            project=layout.project,
            instrument=args.instrument,
            wavelength=args.wavelength or _wavelength(layout),
            bkg_terms=args.bkg_terms,
            ka2_ratio=args.ka2_ratio,
        )
    )


def sample_add_phase(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    return core.cmd_add_phase(
        SimpleNamespace(
            project=layout.project,
            cif=Path(args.cif).resolve() if args.cif else None,
            from_card=args.from_card,
            from_cod=args.from_cod,
            name=args.name,
            scale=args.scale,
            pawley=bool(args.pawley),
        )
    )


def sample_remove_phase(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    return core.cmd_remove_phase(SimpleNamespace(project=layout.project, phase_name=args.phase))


def sample_toggle_phase(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    return core.cmd_toggle_phase(
        SimpleNamespace(
            project=layout.project,
            phase_name=args.phase,
            enable=bool(args.enable),
            disable=bool(args.disable),
        )
    )


def sample_step(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    return core.cmd_step(
        SimpleNamespace(
            project=layout.project,
            stage=args.stage,
            sequence=args.sequence,
            bkg_terms=args.bkg_terms,
            cycles=args.cycles,
            refine_displacement=bool(args.refine_displacement),
            refine_axial=bool(args.refine_axial),
            texture_axis=list(args.texture_axis),
            uniaxial_size=list(args.uniaxial_size),
            init_md=args.init_md,
            occupancy_atoms=args.occupancy_atoms,
            override_json=args.override_json,
            force=bool(args.force),
            passes=getattr(args, "passes", None),
            until_plateau=getattr(args, "until_plateau", None),
            max_passes=getattr(args, "max_passes", 5),
            plateau_delta=getattr(args, "plateau_delta", 0.1),
            keep_gsas_listing=bool(getattr(args, "keep_gsas_listing", False)),
        )
    )


def sample_params(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    refine = args.action == "refine"
    return core.cmd_params(
        SimpleNamespace(
            action=args.action,
            project=layout.project,
            params=layout.params if refine else None,
            out=None if refine else layout.params,
            preset=args.preset,
            cycles=args.cycles if args.cycles is not None else 5,
            sequential_scales=bool(getattr(args, "sequential_scales", False)),
            profile_start_size=args.profile_start_size,
            mustrain_max_percent=args.mustrain_max_percent,
            keep_gsas_listing=bool(getattr(args, "keep_gsas_listing", False)),
        )
    )


def sample_status(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    return core.cmd_status(SimpleNamespace(project=layout.project, json=bool(args.json)))


def sample_diagnose(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    return core.cmd_diagnose(
        SimpleNamespace(
            project=layout.project,
            out=layout.diagnosis if args.out else None,
            candidate_cifs=Path(args.candidate_cifs).resolve() if args.candidate_cifs else None,
            elements=args.elements,
            any=args.any,
            none=args.none,
            strict=bool(args.strict),
            pdf2=bool(getattr(args, "pdf2", False)),
            hsrdb=bool(getattr(args, "hsrdb", False)),
            subfiles=getattr(args, "subfiles", None),
            angular_tolerance=float(args.angular_tolerance),
            snr_threshold=float(args.snr_threshold),
            wavelength=args.wavelength or _wavelength(layout),
            gradients=bool(args.gradients),
            json=bool(getattr(args, "json", False)),
        )
    )


def sample_export(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    layout.artifacts_dir.mkdir(parents=True, exist_ok=True)
    result = core.cmd_export(
        SimpleNamespace(
            project=layout.project,
            out_dir=layout.artifacts_dir,
            plot=None,
            wavelength=args.wavelength or _wavelength(layout),
            gii_threshold=args.gii_threshold,
            dpi=args.dpi,
            strict=bool(args.strict),
            allow_incomplete=bool(args.allow_incomplete),
            json=bool(args.json),
        )
    )
    # cmd_export возвращает сводку либо код ошибки; CLI уже напечатал отчёт сам.
    return result if isinstance(result, int) else 0


def sample_revert(layout: SampleLayout, args: argparse.Namespace) -> int:
    import xrd_project as core

    return core.cmd_revert(SimpleNamespace(project=layout.project, to=args.to))


#: Расширения, которые скилл читает как текст измерения. Приборный контейнер
#: `.raw` в этот список не входит: его разбор скилл не выполняет.
TEXT_MEASUREMENT_SUFFIXES: tuple[str, ...] = (".asc", ".xy", ".txt", ".dat")


def measurement_file(layout: SampleLayout, explicit: str | None) -> Path | None:
    """Возвращает измерение образца: явное, единственное текстовое или единственное.

    Если в `data/` лежит несколько файлов, выбор не угадывается: два измерения —
    это два образца, и смешивать их молча нельзя. Но архивный `.raw` рядом с его
    текстовой копией — не второе измерение: измерительно это один и тот же скан,
    и предпочтение отдаётся тексту, который скилл умеет читать.
    """
    if explicit:
        path = Path(explicit).resolve()
        return path if path.exists() else None
    if not layout.data_dir.is_dir():
        return None
    candidates = sorted(item for item in layout.data_dir.iterdir() if item.is_file())
    if len(candidates) == 1:
        return candidates[0]
    text_candidates = [
        item for item in candidates if item.suffix.lower() in TEXT_MEASUREMENT_SUFFIXES
    ]
    return text_candidates[0] if len(text_candidates) == 1 else None


def sample_pipeline(layout: SampleLayout, args: argparse.Namespace) -> int:
    """Полный прогон поиска фаз и уточнения для одного образца."""
    from dataclasses import asdict

    from xrd_pipeline import run_pipeline

    data = measurement_file(layout, args.data)
    if data is None:
        logger.error("Измерение не выбрано: положите один файл в data/ или укажите --data")
        return 1
    layout.artifacts_dir.mkdir(parents=True, exist_ok=True)
    elements = [item.strip() for item in args.elements.split(",")] if args.elements else None
    report = run_pipeline(
        data_path=data,
        cif_dir=Path(args.cif_dir).resolve() if args.cif_dir else None,
        local_cif_dir=None,
        elements=elements,
        plot_path=None,
        output_dir=layout.artifacts_dir,
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    return 0


def phase_name_from_refined_cif(filename: str, sample: str) -> str:
    """Вернуть имя фазы из имени выгруженного CIF.

    ``export`` пишет ``<образец>_<фаза>_refined.cif``, а повторный экспорт уже
    выгруженной фазы даёт ``..._refined_refined.cif``. Имя для проекта — это имя
    фазы, без префикса образца и без служебных суффиксов: иначе после пересборки
    проект получит технический префикс в имени фазы и в легенде рисунка.
    """
    stem = Path(filename).stem.removeprefix(f"{sample}_")
    while stem.endswith("_refined"):
        stem = stem[: -len("_refined")]
    return stem


def sample_rebuild(layout: SampleLayout, args: argparse.Namespace) -> int:
    """Пересобрать проект образца из измерения и выгруженных CIF.

    Единственный измеренно работающий выход из проекта, который перестал
    уточняться: движок отказывает на первой итерации любой стадии с
    ``Invalid cell metric tensor``, называя фазу с нормальной ячейкой. Причина в
    состоянии файла, а не в модели, поэтому пересборка — лечение, а не обход.

    Старое состояние сохраняется рядом как ``<проект>.before_rebuild.gpx``: это
    не резервная копия «на всякий случай», а доказательство того, что было до.
    """
    import xrd_project as core

    data = measurement_file(layout, args.data)
    if data is None:
        logger.error("Измерение не выбрано: положите один файл в data/ или укажите --data")
        return 1

    sample = layout.root.name
    if args.cif:
        cif_paths = [Path(item).resolve() for item in args.cif]
    else:
        cif_paths = sorted(layout.artifacts_dir.glob(f"{sample}_*_refined.cif"))
    cif_paths = [path for path in cif_paths if path.is_file()]
    if not cif_paths:
        logger.error(
            f"Не найдено ни одного CIF фазы: искали {sample}_*_refined.cif в "
            f"{layout.artifacts_dir} или укажите --cif"
        )
        return 1

    layout.prepare()
    if layout.project.exists():
        backup = layout.project.with_suffix(".before_rebuild.gpx")
        shutil.copyfile(layout.project, backup)
        logger.info(f"Прежнее состояние проекта сохранено: {backup.name}")

    rc = core.cmd_init(
        SimpleNamespace(
            data=data,
            project=layout.project,
            instrument=args.instrument,
            wavelength=args.wavelength or _wavelength(layout),
            bkg_terms=args.bkg_terms,
            ka2_ratio=None,
        )
    )
    if rc != 0:
        return rc

    for cif_path in cif_paths:
        name = phase_name_from_refined_cif(cif_path.name, sample)
        rc = core.cmd_add_phase(
            SimpleNamespace(
                project=layout.project,
                cif=cif_path,
                from_card=None,
                name=name,
                scale=None,
            )
        )
        if rc != 0:
            logger.error(f"Фаза '{name}' не добавлена; пересборка остановлена")
            return rc

    print(
        "SUCCESS: проект пересобран из измерения и "
        f"{len(cif_paths)} фаз(ы). Дальше: step --sequence 1,3,5,7 до плато, "
        "затем финальная step --stage 1 и export."
    )
    return 0


def sample_adopt(layout: SampleLayout, args: argparse.Namespace) -> int:
    """Принимает существующий набор файлов образца в раскладку с проверкой.

    Перенос без проверки был бы переименованием вслепую: проект с другим
    содержимым занял бы место исходного. Поэтому источник и принятая копия
    перезагружаются, и их невязки сравниваются между собой — не с записью в
    истории, которая может относиться к другой стадии уточнения.
    """
    source_project = Path(args.gpx).resolve()
    if not source_project.exists():
        logger.error(f"Project file not found: {source_project}")
        return 1
    data_source = Path(args.data).resolve() if args.data else None
    if data_source is not None and not data_source.exists():
        logger.error(f"Measurement file not found: {data_source}")
        return 1

    layout.prepare()
    if layout.project.exists() and not args.force:
        logger.error(f"Sample already has a project: {layout.project}. Use --force to replace it.")
        return 1

    if data_source is not None:
        shutil.copyfile(data_source, layout.data_dir / data_source.name)
    shutil.copyfile(source_project, layout.project)

    params_source = Path(args.params).resolve() if args.params else None
    if params_source is not None and params_source.exists():
        shutil.copyfile(params_source, layout.params)

    history_source = source_project.with_name(f"{source_project.stem}_history.json")
    if history_source.exists():
        shutil.copyfile(history_source, layout.history)

    from gsas2_adapter import ensure_fresh_residuals
    from stage_gate_rietveld import ensure_gsas2_environment

    if not ensure_gsas2_environment():
        logger.error(
            "GSAS-II engine is not available: adoption cannot be verified; nothing accepted."
        )
        return 1
    from GSASII import GSASIIscriptable as G2sc

    def residual_wr(project: Path) -> float | None:
        project_obj = G2sc.G2Project(str(project))
        hists = project_obj.histograms()
        if not hists:
            return None
        values = ensure_fresh_residuals(project_obj, hists[0])
        wr = values.get("wR")
        return float(wr) if wr is not None else None

    source_wr = residual_wr(source_project)
    adopted_wr = residual_wr(layout.project)
    if source_wr is None or adopted_wr is None:
        logger.error("Adoption rejected: project holds no powder histogram.")
        return 1
    if abs(source_wr - adopted_wr) > ADOPT_RWP_TOLERANCE:
        logger.error(
            f"Adoption rejected: source gives wR={source_wr:.2f}%, "
            f"adopted copy gives {adopted_wr:.2f}%."
        )
        return 1
    measured = adopted_wr
    logger.info(
        f"Adopted {layout.root.name}: project={layout.project.name}, wR={float(measured):.2f}%"
        if measured is not None
        else f"Adopted {layout.root.name}"
    )
    print(f"SUCCESS: sample '{layout.root.name}' adopted into {layout.root}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xrd_sample",
        description=(
            "Работа с одним образцом: одна папка — один образец. Пути и имена "
            "файлов вычисляются раскладкой, агент называет только операцию."
        ),
    )
    parser.add_argument("--dir", default=None, help="Папка образца (по умолчанию текущая)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Создать проект образца из измерения")
    p_init.add_argument("--data", required=True, help="Файл измерения (.raw/.asc/.xy)")
    p_init.add_argument("--instrument", default="canonical")
    p_init.add_argument("--wavelength", type=float, default=None)
    p_init.add_argument("--bkg-terms", type=int, default=6)
    p_init.add_argument("--ka2-ratio", dest="ka2_ratio", type=float, default=None)

    p_add = subparsers.add_parser("add-phase", help="Добавить фазу в проект")
    p_add.add_argument("--cif", default=None)
    p_add.add_argument("--from-card", default=None)
    p_add.add_argument("--from-cod", dest="from_cod", default=None)
    p_add.add_argument("--name", default=None)
    p_add.add_argument("--scale", type=float, default=None)
    p_add.add_argument(
        "--pawley",
        action="store_true",
        help="Активировать нативную бесструктурную декомпозицию Поули (doPawley)",
    )

    p_rem = subparsers.add_parser("remove-phase", help="Удалить фазу из проекта")
    p_rem.add_argument("--phase", required=True)

    p_tog = subparsers.add_parser("toggle-phase", help="Включить или выключить фазу")
    p_tog.add_argument("--phase", required=True)
    group = p_tog.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true")
    group.add_argument("--disable", action="store_true")

    p_step = subparsers.add_parser("step", help="Выполнить стадию уточнения")
    step_group = p_step.add_mutually_exclusive_group(required=True)
    step_group.add_argument("--stage", default=None)
    step_group.add_argument("--sequence", default=None)
    p_step.add_argument("--cycles", type=int, default=5)
    p_step.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Фиксированное число проходов маршрута (отменяет умолчание «до плато»)",
    )
    p_step.add_argument(
        "--until-plateau",
        dest="until_plateau",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Идти проходами до плато: по умолчанию включено для --sequence и "
            "выключено для одиночной --stage"
        ),
    )
    p_step.add_argument(
        "--max-passes",
        dest="max_passes",
        type=int,
        default=5,
        help="Предел проходов; его достижение называется пределом, а не сходимостью",
    )
    p_step.add_argument(
        "--plateau-delta",
        dest="plateau_delta",
        type=float,
        default=0.1,
        help="Порог улучшения R_wp, ниже которого проход считается плато, %%",
    )
    p_step.add_argument("--bkg-terms", dest="bkg_terms", type=int, default=6)
    p_step.add_argument("--init-md", dest="init_md", type=float, default=None)
    p_step.add_argument(
        "--occupancy-atoms",
        dest="occupancy_atoms",
        default=None,
        help='Атомы для стадии 9: "PhaseA:M1,PhaseB:M2" (по умолчанию — позиции с заселением < 1)',
    )
    p_step.add_argument("--refine-displacement", dest="refine_displacement", action="store_true")
    p_step.add_argument("--refine-axial", dest="refine_axial", action="store_true")
    p_step.add_argument("--texture-axis", dest="texture_axis", action="append", default=[])
    p_step.add_argument("--uniaxial-size", dest="uniaxial_size", action="append", default=[])
    p_step.add_argument("--override-json", dest="override_json", default=None)
    p_step.add_argument("--force", action="store_true")

    p_params = subparsers.add_parser("params", help="Работа с рецептом уточнения")
    p_params.add_argument("action", choices=["dump", "refine"])
    p_params.add_argument("--preset", default=None)
    p_params.add_argument("--cycles", type=int, default=None)
    p_params.add_argument("--sequential-scales", dest="sequential_scales", action="store_true")
    p_params.add_argument(
        "--profile-start-size", dest="profile_start_size", type=float, default=None
    )
    p_params.add_argument(
        "--mustrain-max-percent", dest="mustrain_max_percent", type=float, default=None
    )

    p_status = subparsers.add_parser("status", help="Метрики и вердикт образца")
    p_status.add_argument("--json", action="store_true")

    p_diag = subparsers.add_parser("diagnose", help="Разбор остатка и кандидатов")
    p_diag.add_argument("--out", action="store_true", help="Сохранить диагностику в projects/")
    p_diag.add_argument("--candidate-cifs", dest="candidate_cifs", default=None)
    p_diag.add_argument("--pdf2", action="store_true", help="Оценить базу карточек PDF-2")
    p_diag.add_argument("--hsrdb", action="store_true", help="Оценить базу структур COD HSRDB")
    p_diag.add_argument(
        "--subfiles",
        default=None,
        help="Фильтр подфайлов БД (например, inorganic,mineral,metallic)",
    )
    p_diag.add_argument("--elements", default=None)
    p_diag.add_argument("--any", default=None)
    p_diag.add_argument("--none", default=None)
    p_diag.add_argument("--strict", action="store_true")
    p_diag.add_argument("--angular-tolerance", dest="angular_tolerance", type=float, default=0.20)
    p_diag.add_argument("--snr-threshold", dest="snr_threshold", type=float, default=3.0)
    p_diag.add_argument("--gradients", action="store_true")
    p_diag.add_argument("--wavelength", type=float, default=None)
    p_diag.add_argument("--json", action="store_true")

    p_export = subparsers.add_parser("export", help="Печать отчёта и выгрузка артефактов")
    p_export.add_argument("--json", action="store_true", help="Напечатать сводку JSON в stdout")
    p_export.add_argument("--wavelength", type=float, default=None)
    p_export.add_argument("--dpi", type=int, default=600)
    p_export.add_argument("--gii-threshold", dest="gii_threshold", type=float, default=None)
    p_export.add_argument("--strict", action="store_true")
    p_export.add_argument("--allow-incomplete", dest="allow_incomplete", action="store_true")

    p_revert = subparsers.add_parser("revert", help="Откат к контрольной точке")
    p_revert.add_argument("--to", type=int, default=None)

    p_pipe = subparsers.add_parser("pipeline", help="Поиск фаз и уточнение полным прогоном")
    p_pipe.add_argument(
        "--data", default=None, help="Измерение (по умолчанию — единственный файл в data/)"
    )
    p_pipe.add_argument("--cif-dir", dest="cif_dir", default=None)
    p_pipe.add_argument("--elements", default=None)

    p_rebuild = subparsers.add_parser(
        "rebuild", help="Пересобрать проект из измерения и выгруженных CIF"
    )
    p_rebuild.add_argument("--data", default=None)
    p_rebuild.add_argument("--cif", action="append", default=[])
    p_rebuild.add_argument("--instrument", default="canonical")
    p_rebuild.add_argument("--wavelength", type=float, default=None)
    p_rebuild.add_argument("--bkg-terms", dest="bkg_terms", type=int, default=6)

    p_adopt = subparsers.add_parser("adopt", help="Принять существующий набор файлов образца")
    p_adopt.add_argument("--gpx", required=True, help="Существующий проект GSAS-II")
    p_adopt.add_argument("--data", default=None, help="Существующее измерение")
    p_adopt.add_argument("--params", default=None, help="Существующий рецепт")
    p_adopt.add_argument("--force", action="store_true", help="Заменить проект образца")

    return parser


COMMANDS = {
    "init": sample_init,
    "add-phase": sample_add_phase,
    "remove-phase": sample_remove_phase,
    "toggle-phase": sample_toggle_phase,
    "step": sample_step,
    "params": sample_params,
    "status": sample_status,
    "diagnose": sample_diagnose,
    "export": sample_export,
    "pipeline": sample_pipeline,
    "revert": sample_revert,
    "adopt": sample_adopt,
    "rebuild": sample_rebuild,
}


def main(argv: list[str] | None = None) -> int:
    import os

    log_level = os.environ.get("XRD_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    args = build_parser().parse_args(argv)
    layout = open_sample(args.dir)
    # init и adopt создают папку образца; остальные требуют существующей раскладки.
    if args.command in {"init", "adopt"}:
        layout.prepare()
    else:
        layout.ensure_sample()
    if getattr(args, "json", False):
        import xrd_project as core

        # GSAS-II пишет «N.B. Configuration file ...» обычным print в stdout и ломает
        # машинный вывод; на время --json stdout уводится в stderr.
        with core.engine_output_to_stderr():
            return COMMANDS[args.command](layout, args)
    return COMMANDS[args.command](layout, args)


if __name__ == "__main__":
    raise SystemExit(main())
