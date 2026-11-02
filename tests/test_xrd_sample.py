"""Tests for the sample-level workspace layer (xrd_sample.py).

Слой отвечает за одно: превратить «образец» в набор путей. Поэтому тесты
проверяют не расчёты, а раскладку и отказы — то, из-за чего агент больше не
выбирает имена файлов.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from xrd_sample import (
    SampleLayout,
    _wavelength,
    build_parser,
    measurement_file,
    open_sample,
    sample_adopt,
    sample_init,
)


def test_paths_derive_from_folder_name(tmp_path: Path):
    root = tmp_path / "sample_A"
    root.mkdir()
    layout = open_sample(root)

    assert layout.project == root / "projects" / "sample_A.gpx"
    assert layout.history == root / "projects" / "sample_A_history.json"
    assert layout.diagnosis == root / "projects" / "sample_A_diag.json"
    assert layout.params == root / "params.json"
    assert layout.artifacts_dir == root / "artifacts"


def test_prepare_creates_only_working_folders(tmp_path: Path):
    layout = SampleLayout(root=tmp_path / "S1")
    layout.prepare()

    assert (tmp_path / "S1" / "data").is_dir()
    assert (tmp_path / "S1" / "projects").is_dir()
    assert (tmp_path / "S1" / "artifacts").is_dir()
    # Рецепт и заметки создаёт человек или params dump, а не раскладка.
    assert not layout.params.exists()


def test_foreign_folder_is_refused(tmp_path: Path):
    with pytest.raises(SystemExit):
        open_sample(tmp_path).ensure_sample()


def test_sample_folder_is_accepted_once_data_exists(tmp_path: Path):
    layout = SampleLayout(root=tmp_path)
    (tmp_path / "data").mkdir()
    layout.ensure_sample()  # не бросает


def test_wavelength_comes_from_recipe(tmp_path: Path):
    layout = SampleLayout(root=tmp_path)
    layout.params.write_text(json.dumps({"wavelength": 1.7889}), encoding="utf-8")
    assert _wavelength(layout) == 1.7889


def test_wavelength_falls_back_when_recipe_is_absent(tmp_path: Path):
    assert _wavelength(SampleLayout(root=tmp_path)) == 1.5406


def test_init_refuses_missing_measurement_and_creates_nothing(tmp_path: Path):
    layout = SampleLayout(root=tmp_path / "S2")
    args = argparse.Namespace(
        data=str(tmp_path / "missing.raw"),
        instrument="canonical",
        wavelength=None,
        bkg_terms=6,
    )

    assert sample_init(layout, args) == 1
    assert not (tmp_path / "S2" / "projects").exists()


def test_adopt_refuses_missing_project(tmp_path: Path):
    layout = SampleLayout(root=tmp_path / "S3")
    args = argparse.Namespace(
        gpx=str(tmp_path / "missing.gpx"), data=None, params=None, force=False
    )

    assert sample_adopt(layout, args) == 1
    assert not (tmp_path / "S3" / "projects").exists()


def test_adopt_does_not_overwrite_silently(tmp_path: Path):
    root = tmp_path / "S4"
    layout = SampleLayout(root=root)
    layout.prepare()
    layout.project.write_bytes(b"existing")
    source = tmp_path / "source.gpx"
    source.write_bytes(b"new")
    args = argparse.Namespace(gpx=str(source), data=None, params=None, force=False)

    assert sample_adopt(layout, args) == 1
    assert layout.project.read_bytes() == b"existing"


def test_measurement_is_found_only_when_unambiguous(tmp_path: Path):
    layout = SampleLayout(root=tmp_path)
    layout.data_dir.mkdir()
    (layout.data_dir / "a.raw").write_text("x", encoding="utf-8")
    assert measurement_file(layout, None) == layout.data_dir / "a.raw"

    # Два измерения — два образца: выбор не угадывается.
    (layout.data_dir / "b.raw").write_text("y", encoding="utf-8")
    assert measurement_file(layout, None) is None


def test_explicit_measurement_must_exist(tmp_path: Path):
    layout = SampleLayout(root=tmp_path)
    assert measurement_file(layout, str(tmp_path / "missing.raw")) is None


@pytest.mark.parametrize(
    ("argv", "attribute"),
    [
        (["init", "--data", "x.raw"], "wavelength"),
        (["diagnose"], "wavelength"),
        (["export"], "wavelength"),
        (["step", "--stage", "2_Zero_Shift"], "bkg_terms"),
        (["step", "--stage", "2_Zero_Shift"], "init_md"),
    ],
)
def test_handler_reads_only_declared_arguments(argv: list[str], attribute: str):
    """Флаг, который читает обработчик, должен быть объявлен его парсером.

    `diagnose` читал `args.wavelength`, но `--wavelength` не объявлял: команда
    падала с AttributeError до всякого расчёта, и диагностика образца была
    недоступна.
    """
    assert hasattr(build_parser().parse_args(argv), attribute)
    # Отрицательный контроль: у команды без этого флага атрибута быть не должно,
    # иначе проверка проходит независимо от объявления.
    assert not hasattr(build_parser().parse_args(["status"]), attribute)


def _args_attributes(handler: object) -> set[str]:
    """Имена, которые обработчик читает как `args.<name>`.

    `getattr(args, "x", default)` сюда не попадает: там имя — строка-константа,
    а не атрибут, и отсутствие такого поля обработчик переживает.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(handler))  # type: ignore[arg-type]
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }


def _value_for(action: argparse.Action) -> str:
    if action.choices:
        return str(action.choices[0])
    if action.type in (int, float):
        return "1"
    return "value"


def _append_action(argv: list[str], action: argparse.Action) -> None:
    if action.option_strings:
        argv.append(action.option_strings[0])
        if action.nargs != 0:
            argv.append(_value_for(action))
    else:
        argv.append(_value_for(action))


def _minimal_argv(subparser: argparse.ArgumentParser) -> list[str]:
    """Собирает командную строку из объявленных обязательных аргументов."""
    argv: list[str] = []
    for action in subparser._actions:
        if action.required and action.dest != "help":
            _append_action(argv, action)
    # Взаимно исключающая группа (--enable | --disable) объявляет обязательность на
    # группе, а не на действии, поэтому в общем цикле её элементы выглядят необязательными.
    for group in subparser._mutually_exclusive_groups:
        if not group.required:
            continue
        options = {opt for action in group._group_actions for opt in action.option_strings}
        if options & set(argv):
            continue
        _append_action(argv, group._group_actions[0])
    return argv


def test_every_handler_reads_only_declared_arguments():
    """Обработчик не может читать флаг, которого его парсер не объявил.

    Дрейф парсера и обработчика — не гипотеза: `diagnose` читал `args.wavelength`,
    `step` — `args.bkg_terms` и `args.init_md`, и обе команды падали с
    AttributeError до всякого расчёта. Проверка идёт по всем командам сразу,
    поэтому следующий забытый флаг ловится здесь, а не на образце.
    """
    import xrd_sample as skill

    parser = skill.build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    problems: list[str] = []
    for name, subparser in subparsers.choices.items():
        namespace = subparser.parse_args(_minimal_argv(subparser))
        missing = _args_attributes(skill.COMMANDS[name]) - set(vars(namespace))
        if missing:
            problems.append(f"{name}: {sorted(missing)}")
    assert not problems, "обработчики читают необъявленные аргументы: " + "; ".join(problems)


def test_phase_name_from_refined_cif_strips_prefix_and_suffixes():
    """Имя фазы из имени CIF — без префикса образца и без служебных суффиксов."""
    from xrd_sample import phase_name_from_refined_cif

    sample = "sample_A"
    assert phase_name_from_refined_cif(f"{sample}_SmS_refined.cif", sample) == "SmS"
    assert phase_name_from_refined_cif(f"{sample}_Sm2O2S_refined_refined.cif", sample) == "Sm2O2S"
    # Чужой префикс не трогаем: имя принадлежит другой фазе, а не этому образцу.
    assert phase_name_from_refined_cif("SmS.cif", sample) == "SmS"
    assert phase_name_from_refined_cif("other_SmS_refined.cif", sample) == "other_SmS"


def test_rebuild_refuses_without_measurement_or_cifs(tmp_path: Path):
    """Без измерения и без CIF пересборка не начинается и ничего не создаёт."""
    from xrd_sample import sample_rebuild

    layout = SampleLayout(root=tmp_path / "S5")
    args = argparse.Namespace(
        data=None, cif=[], instrument="canonical", wavelength=None, bkg_terms=6
    )

    assert sample_rebuild(layout, args) == 1
    assert not (tmp_path / "S5" / "projects").exists() or not list(
        (tmp_path / "S5" / "projects").glob("*.gpx")
    )


def test_archive_and_its_text_copy_are_one_measurement(tmp_path: Path):
    """`.raw` рядом с его текстовой копией — один скан, а не два образца.

    Скилл не читает приборный контейнер, поэтому при наличии текста выбирается
    текст. Если текстовых файлов больше одного, выбор по-прежнему не угадывается.
    """
    from xrd_sample import measurement_file

    layout = SampleLayout(root=tmp_path)
    layout.data_dir.mkdir()
    (layout.data_dir / "S.raw").write_text("RAW1.01", encoding="utf-8")
    (layout.data_dir / "S.txt.asc").write_text("20.00 100\n", encoding="utf-8")

    assert measurement_file(layout, None) == layout.data_dir / "S.txt.asc"

    (layout.data_dir / "S2.xy").write_text("20.00 100\n", encoding="utf-8")
    assert measurement_file(layout, None) is None
