"""Publication-quality Rietveld figure, drawn by this skill from a GSAS-II project.

GSAS-II supplies the numbers, not the picture: the profile arrays are read with
``extract_profile_curves`` (``hist.data["data"][1]``) and the reflections with
``extract_bragg_reflections`` (``hist.data["Reflection Lists"]``). The figure is built
with matplotlib on the Agg backend, because the engine's own plotter opens a wxPython
window and cannot run headless.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


class _InterceptHandler(logging.Handler):
    """Мост stdlib logging → loguru для неймспейса xrdplot.

    Библиотека пишет в стандартный логгер ``xrdplot`` и объявляет NullHandler, то есть
    по умолчанию молчит — так и должно быть для библиотеки. Куда направить её записи,
    решает вызывающее приложение, и здесь этим приложением является скилл: его логи уже
    идут через loguru в stderr.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Поднимаемся к кадру, который вызвал логирование, а не к кадру logging.
        frame = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Глубина кадра сама по себе ненадёжна: в разных версиях logging число
        # служебных кадров разное, и запись подписывается «logging:callHandlers».
        # Поэтому источник берётся из самой записи: имя логгера, модуль, функция, строка.
        logger.opt(depth=depth, exception=record.exc_info).patch(
            lambda patched: patched.update(
                name=record.name,
                module=record.module,
                function=record.funcName,
                line=record.lineno,
            )
        ).log(level, record.getMessage())


def _install_xrdplot_logging() -> None:
    """Один раз подключает мост; повторный импорт не добавляет второй обработчик."""
    package_logger = logging.getLogger("xrdplot")
    if not any(isinstance(handler, _InterceptHandler) for handler in package_logger.handlers):
        package_logger.addHandler(_InterceptHandler())
        package_logger.setLevel(logging.INFO)
        # propagate выключен: иначе сообщение уйдёт ещё и в root-логгер приложения.
        package_logger.propagate = False


_install_xrdplot_logging()

try:
    from stage_gate_rietveld import ensure_gsas2_environment
except ImportError:
    from scripts.stage_gate_rietveld import ensure_gsas2_environment

__all__ = [
    "plot_gpx_project",
    "plot_gsas2_histogram",
]


def plot_gsas2_histogram(
    hist: Any,
    out_path: Path,
    title: str | None = None,
    dpi: int = 600,
    gpx: Any | None = None,
) -> Path:
    """Render a publication-grade Rietveld plot from a GSAS-II histogram or ProfileCurves.

    The picture is built by ``xrdplot``: this function only translates the engine's
    numbers into that package's model (``ProfileCurves`` and Bragg reflections into a
    ``Diffractogram``) and saves the resulting figure. One renderer therefore serves
    both this skill and standalone publication figure scripts.

    ``hist`` can be:
    - A GSAS-II ``G2PwdrData`` histogram object (live project or loaded .gpx).
    - A typed :class:`gsas2_adapter.ProfileCurves` for direct array-based plots
      (e.g., in tests or post-processing), without requiring a live GSAS-II project.
    """
    from gsas2_adapter import (
        ProfileCurves,
        calculate_profile_metrics,
        extract_bragg_reflections,
        extract_profile_curves,
    )
    from xrdplot import (
        Diffractogram,
        Phase,
        PhaseParams,
        RefinementStats,
        Settings,
        ValueWithError,
        build_figure,
    )

    curves = hist if isinstance(hist, ProfileCurves) else extract_profile_curves(hist)
    two_theta = np.asarray(curves.two_theta, dtype=float)

    by_phase: dict[str, list[float]] = {}
    if not isinstance(hist, ProfileCurves):
        for reflection in extract_bragg_reflections(hist):
            if two_theta.min() <= reflection.two_theta <= two_theta.max():
                by_phase.setdefault(reflection.phase_name, []).append(reflection.two_theta)
    # R_B и массовые доли берутся только при наличии проекта: без него их взять
    # неоткуда, и подпись фазы остаётся без чисел, а не с выдуманными.
    weights: dict[str, tuple[float, float]] = {}
    rb_values: dict[str, float] = {}
    if gpx is not None and not isinstance(hist, ProfileCurves):
        from gsas2_adapter import get_project_qpa, phase_rb_percent

        weights = get_project_qpa(gpx, hist)
        rb_values = phase_rb_percent(extract_bragg_reflections(hist))

    phases = []
    for name, angles in by_phase.items():
        params = None
        if name in weights or name in rb_values:
            weight, _esd = weights.get(name, (None, None))  # type: ignore[misc]  # noqa: PU001
            params = PhaseParams(
                RBragg=ValueWithError(rb_values[name]) if name in rb_values else None,
                Weight_Percentage=ValueWithError(weight) if weight is not None else None,
            )
        phases.append(Phase(phase_id=name, angles=np.array(sorted(set(angles))), params=params))

    stats = None
    if not isinstance(hist, ProfileCurves):
        metrics = calculate_profile_metrics(curves)
        stats = RefinementStats(
            Rwp=ValueWithError(metrics.r_wp_percent),
            Rp=ValueWithError(metrics.r_p_percent),
            Rexp=ValueWithError(metrics.r_exp_percent),
            GOF=ValueWithError(metrics.gof),
        )

    diffractogram = Diffractogram(
        x=two_theta,
        i_obs=np.asarray(curves.y_obs, dtype=float),
        i_calc=np.asarray(curves.y_calc, dtype=float),
        i_bkg=np.asarray(curves.y_bkg, dtype=float),
        phases=phases,
        stats=stats,
        title=title,
        sigma=np.asarray(curves.sigma_obs, dtype=float),
    )
    settings = Settings(
        dpi=dpi,
        panel_size=(7.0, 3.5),
        # Одна панель — буквенная метка (a) лишняя; R_B и доли фаз включаем.
        show_panel_labels=False,
        show_rb=True,
        show_weight=True,
        show_diff=True,
        show_bragg=bool(phases),
        show_sigma_band=True,
        xlim=(float(two_theta.min()), float(two_theta.max())),
        # Отчёт отключён: скилл возвращает рисунок и не оставляет рядом с ним
        # markdown/docx. Отчёт нужен там, где его читает человек или сборщик публикации.
        report_path=None,
        save_docx=False,
    )
    figure = build_figure([diffractogram], settings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"Refinement plot saved to {out_path} ({dpi} DPI)")
    return out_path


def plot_gpx_project(
    gpx_path: Path,
    out_path: Path,
    dpi: int = 600,
) -> Path:
    """Load a GSAS-II .gpx project Headless and render the refinement plot."""
    if not ensure_gsas2_environment():
        raise RuntimeError("GSAS-II environment could not be initialized.")

    from GSASII import GSASIIscriptable as G2sc

    gpx = G2sc.G2Project(str(gpx_path))
    histograms = gpx.histograms()
    if not histograms:
        raise ValueError(f"No histograms found in project {gpx_path}")

    # Plot first powder histogram
    hist = histograms[0]
    return plot_gsas2_histogram(
        hist,
        out_path,
        title=f"Rietveld Refinement - {gpx_path.stem}",
        dpi=dpi,
        gpx=gpx,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="XRD Native GSAS-II Publication Plotter")
    parser.add_argument("gpx_file", type=Path, help="Path to GSAS-II .gpx project file")
    parser.add_argument(
        "--out", type=Path, default=Path("rietveld_plot.png"), help="Output image file (.png/.pdf)"
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution (default: 600)")
    args = parser.parse_args()

    if not args.gpx_file.exists():
        logger.error(f"GPX file not found: {args.gpx_file}")
        return 1

    try:
        plot_gpx_project(args.gpx_file, args.out, dpi=args.dpi)
        return 0
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Plotting failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
