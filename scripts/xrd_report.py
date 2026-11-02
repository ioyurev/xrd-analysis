"""Deterministic, truthful crystallographic Markdown report generator adhering to IUCr standards.

Guarantees 100% mathematical consistency between numerical calculation logs
(PipelineReport, StageGateOutput, residual diagnosis) and human-readable text.
Eliminates rhetorical LLM optimism or false "zero unindexed peaks" claims.
Reports phase-specific R_B / R_F, Durbin-Watson statistic, standard uncertainties (e.s.d.),
and physical microstructure parameters (crystallite size, microstrain).
"""

from __future__ import annotations

import io
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from export import ExportSummary
    from xrd_diagnose import ResidualDiagnosisReport


def format_val_esd(val: float, esd: float | None, default_decimals: int = 4) -> str:
    """Format parameter value with standard uncertainty (e.s.d.) according to IUCr conventions:
    e.g. 5.97271 with esd 0.00042 -> '5.9727(4)'.
    """
    if esd is None or esd <= 0.0 or np.isnan(esd):
        return f"{val:.{default_decimals}f}"
    order = int(np.floor(np.log10(esd)))
    dec_needed = max(0, -order)
    esd_int = round(esd * (10**dec_needed))
    return f"{val:.{dec_needed}f}({esd_int})"


def generate_honest_markdown_report(
    summary: ExportSummary,
    diagnosis: ResidualDiagnosisReport,
    out_path: Path | None = None,
) -> str:
    """Generate a rigorous crystallographic report strictly bound to typed contracts and IUCr standards."""
    r_wp = float(summary.r_wp_percent)
    r_p = float(summary.r_p_percent)
    chi2 = float(summary.chi_square)
    status = str(summary.status)

    r_ratio = float(r_wp / r_p) if r_p > 0 else 1.0
    dw_stat = float(summary.durbin_watson)

    true_unindexed = diagnosis.true_unindexed_reflections
    intensity_defects = diagnosis.intensity_mismatch_reflections
    true_unindexed_count = len(true_unindexed)
    intensity_defects_count = len(intensity_defects)

    profile_csv = summary.profile_curves_csv
    bragg_csv = summary.bragg_reflections_csv

    wr_bkg_str = (
        f"{float(summary.wr_bkg_percent):.2f}%" if summary.wr_bkg_percent is not None else "—"
    )
    gii_str = f"{float(summary.gii_value):.4f} v.u." if summary.gii_value is not None else "—"

    dw_assessment = (
        "Идеальный белый шум (автокорреляция отсутствует)"
        if 1.7 <= dw_stat <= 2.3
        else (
            "Положительная сериальная автокорреляция (дефекты формы хвостов пиков или фона)"
            if dw_stat < 1.7
            else "Отрицательная автокорреляция (высокочастотные осцилляции)"
        )
    )

    ratio_assessment = (
        "Норма (равномерное качество подгонки)"
        if r_ratio <= 1.40
        else "Повышенное расхождение: подгонка ухудшена на слабом фоне / в хвостах пиков"
    )

    sigma_bg = float(diagnosis.noise_sigma)
    i_max_net = float(diagnosis.i_max_net)
    if i_max_net > 0 and sigma_bg > 0:
        lod_pct = (3.0 * sigma_bg / i_max_net) * 100.0
    else:
        lod_pct = float(diagnosis.lod_3sigma_percent)
    lod_pct = max(0.05, min(10.0, lod_pct))

    lines: list[str] = [
        "# Отчет о полнопрофильном анализе дифрактограммы (IUCr Compliant XRD Report)",
        "",
        "## 1. Сводные метрики сходимости и качества (IUCr Standards)",
        "",
        f"- **Статус пайплайна**: `{status}`",
        f"- **Взвешенный профильный фактор ($R_{{wp}}$)**: **{r_wp:.2f}%**",
        f"- **Профильный фактор ($R_p$)**: **{r_p:.2f}%**",
        f"- **Фактор профиля без фона ($wR_{{bkg}}$)**: **{wr_bkg_str}**",
        f"- **Отношение $R_{{wp}} / R_p$**: **{r_ratio:.2f}** ({ratio_assessment})",
        f"- **Фактор качества подгонки ($\\text{{GoF}} = R_{{wp}} / R_{{\\text{{exp}}}}$)**: **{float(summary.gof):.2f}**",
        f"- **Приведённый хи-квадрат ($\\chi^2_\\nu = \\text{{GoF}}^2$)**: **{chi2:.2f}**",
        f"- **Статистика Дарбина — Уотсона ($d$)**: **{dw_stat:.2f}** ({dw_assessment})",
        f"- **Индекс нестабильности структуры (GII)**: **{gii_str}**",
        f"- **Предел обнаружения примесных фаз (IUPAC $3\\sigma$ LOD)**: **~{lod_pct:.2f} масс. %**",
        f"- **Истинные неиндексированные пики примесей**: **{true_unindexed_count}**",
        f"- **Дефекты интенсивности/текстуры известных фаз**: **{intensity_defects_count}**",
        "",
        "## 2. Кристаллографическая оценка качества",
        "",
    ]

    if true_unindexed_count == 0 and r_wp <= 10.0:
        verdict = (
            f"Отличное согласие ($R_{{wp}} = {r_wp:.2f}\\%$, $\\chi^2 = {chi2:.2f}$). "
            "Все дифракционные максимумы эксперимента полностью индексированы в рамках выбранного пула фаз."
        )
    elif true_unindexed_count == 0 and r_wp > 10.0:
        verdict = (
            f"Качественная фазовая идентификация завершена: все дифракционные пики эксперимента "
            f"отнесены к фазам пула (0 истинных неиндексированных рефлексов). "
            f"Повышенное значение $R_{{wp}} = {r_wp:.2f}\\%$ ($\\chi^2 = {chi2:.2f}$) обусловлено "
            f"несовершенством моделирования профиля и интенсивностей (выявлено {intensity_defects_count} "
            "пиков с невязкой интенсивности из-за текстуры, нестехиометрии или замороженных координат)."
        )
    else:
        verdict = (
            f"Неполная компенсация: обнаружено {true_unindexed_count} истинных неиндексированных рефлексов "
            f"(отсутствуют в брэгговских списках идентифицированных фаз). "
            f"Значение $R_{{wp}} = {r_wp:.2f}\\%$ ($\\chi^2 = {chi2:.2f}$). "
            "Требуется расширение поискового пула кандидатов для выявления второстепенных примесей."
        )
    lines.append(verdict)
    lines.append("")

    lines.append("## 3. Количественный фазовый анализ и характеристики фаз")
    lines.append("")

    if summary.phases:
        lines.append(
            "| Фаза | Массовая доля W (масс. %) | $R_B$ (Bragg) | $R_F$ (Structure) | ОКР $\\langle L \\rangle$ (нм) | Микронапряжения $\\varepsilon$ (%) | Масштаб S |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for ph_name, pinfo in summary.phases.items():
            w_pct = pinfo.weight_fraction_percent
            scale_val = pinfo.scale
            rb_str = f"{pinfo.r_bragg_percent:.2f}%" if pinfo.r_bragg_percent is not None else "—"
            rf_str = f"{pinfo.r_f_percent:.2f}%" if pinfo.r_f_percent is not None else "—"
            micro = pinfo.microstructure
            size_str = (
                f"{micro.crystallite_size_nm:.1f}" if micro.crystallite_size_nm is not None else "—"
            )
            strain_str = (
                f"{micro.microstrain_percent:.4f}%"
                if micro.microstrain_percent is not None
                else "—"
            )
            lines.append(
                f"| **{ph_name}** | {w_pct:.2f}% | {rb_str} | {rf_str} | {size_str} | {strain_str} | {scale_val:.4e} |"
            )
    else:
        lines.append("*Количественные доли фаз не определены или уточнение не проводилось.*")
    lines.append("")

    has_cells = any(bool(p.cell_parameters) for p in summary.phases.values())
    if has_cells:
        lines.append(
            "### 3.1. Параметры элементарных ячеек со стандартными неопределенностями (e.s.d.)"
        )
        lines.append("")
        lines.append("| Фаза | $a$ (Å) | $b$ (Å) | $c$ (Å) | Объем ячейки $V$ (Å³) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for ph_name, pinfo in summary.phases.items():
            c_dict = pinfo.cell_parameters
            if not c_dict:
                continue
            a_fmt = format_val_esd(c_dict.get("a", 0.0), c_dict.get("a_esd"))
            b_fmt = format_val_esd(c_dict.get("b", 0.0), c_dict.get("b_esd"))
            c_fmt = format_val_esd(c_dict.get("c", 0.0), c_dict.get("c_esd"))
            v_fmt = format_val_esd(
                c_dict.get("volume", 0.0), c_dict.get("volume_esd"), default_decimals=2
            )
            beta_val = float(c_dict.get("beta", 90.0) or 90.0)
            if abs(beta_val - 90.0) > 1e-3:
                beta_fmt = format_val_esd(beta_val, c_dict.get("beta_esd"), default_decimals=2)
                c_fmt = f"{c_fmt} (β={beta_fmt}°)"
            lines.append(f"| **{ph_name}** | {a_fmt} | {b_fmt} | {c_fmt} | {v_fmt} |")
        lines.append("")

    lines.append(
        "### 3.2. Ограничения фазового анализа и технологический дисклеймер (IUPAC LOD & LECO)"
    )
    lines.append("")
    lines.append(
        f"- **Порог обнаружения примесных фаз по фону (IUPAC $3\\sigma$)**: **~{lod_pct:.2f} масс. %** "
        f"(оценка по $\\sigma_{{\\text{{bg}}}} = {sigma_bg:.1f}$ при пиковой интенсивности $I_{{\\max,\\text{{net}}}} = {i_max_net:.1f}$)."
    )

    has_oxide = any(
        ("o" in ph.lower() and "co" not in ph.lower())
        or "oxide" in ph.lower()
        or "oxysulfide" in ph.lower()
        for ph in summary.phases
    )
    if not has_oxide:
        lines.append("")
        lines.append(
            "> [!WARNING] **Предел обнаружения и примеси лёгких элементов:**\n"
            f"> Предел обнаружения кристаллических примесей методом порошковой дифрактометрии составляет **~{lod_pct:.2f} масс. %**. "
            "Отсутствие дифракционных максимумов вторичных оксидных и гетероанионных фаз "
            "подтверждает их отсутствие лишь в пределах чувствительности метода ($C < C_{\\text{LOD}}$). "
            "РФА не чувствителен к лёгким примесным элементам (O, N, H, C), растворённым в кристаллической решётке ниже предела растворимости "
            "либо локализованным в рентгеноаморфных межзёренных прослойках; для их строгой верификации требуется прямой элементный анализ (например, метод восстановительного плавления в токе инертного газа)."
        )
    lines.append("")

    lines.append("## 4. Диагностика невязок профиля")
    lines.append("")

    if true_unindexed:
        lines.append("### 4.1. Истинные неиндексированные рефлексы (Кандидаты в примеси)")
        lines.append("")
        lines.append(
            "Эти пики удалены от брэгговских линий идентифицированных фаз более чем на $\\pm 0.20^\\circ$:"
        )
        lines.append("")
        lines.append(
            "| $2\\theta$ (град) | $d$-расстояние (Å) | Амплитуда невязки $\\Delta Y$ | Значимость ($\\sigma$) |"
        )
        lines.append("| :---: | :---: | :---: | :---: |")
        for u in true_unindexed:
            lines.append(
                f"| {u.two_theta:.3f} | {u.d_spacing_angstrom:.4f} | "
                f"{u.residual_height:.1f} | {u.significance_sigma:.1f}$\\sigma$ |"
            )
        lines.append("")
    else:
        lines.append("### 4.1. Истинные неиндексированные рефлексы")
        lines.append("")
        lines.append("*Истинных неиндексированных пиков примесей не обнаружено.*")
        lines.append("")

    if intensity_defects:
        lines.append("### 4.2. Дефекты интенсивности и текстуры известных фаз")
        lines.append("")
        lines.append(
            "Эти пики совпадают с брэгговскими положениями известных фаз, но имеют невязку интенсивности:"
        )
        lines.append("")
        lines.append(
            "| $2\\theta$ (град) | Фаза | Брэгг $2\\theta$ | $\\Delta 2\\theta$ | Невязка | Значимость | Тип дефекта |"
        )
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :--- |")
        for d in intensity_defects:
            lines.append(
                f"| {d.two_theta:.3f} | **{d.matched_phase or '—'}** | "
                f"{(d.bragg_2theta or 0.0):.3f} | {(d.delta_2theta or 0.0):+.3f} | "
                f"{d.residual_height:.1f} | {d.significance_sigma:.1f}$\\sigma$ | "
                f"{d.defect_type} |"
            )
        lines.append("")
    else:
        lines.append("### 4.2. Дефекты интенсивности и текстуры известных фаз")
        lines.append("")
        lines.append("*Систематических дефектов интенсивности известных фаз не обнаружено.*")
        lines.append("")

    if profile_csv or bragg_csv:
        lines.append("## 5. Файлы числового экспорта (Origin / Excel / Prism)")
        lines.append("")
        lines.append(
            "Для построения графиков в сторонних пакетах сформированы структурированные CSV-таблицы:"
        )
        if profile_csv:
            lines.append(
                f"- **Профильные кривые ($2\\theta, Y_{{\\text{{obs}}}}, Y_{{\\text{{calc}}}}, Y_{{\\text{{bkg}}}}, \\Delta Y, \\sigma$)**: `{profile_csv}`"
            )
        if bragg_csv:
            lines.append(f"- **Штрихи Брэгга и интегральные интенсивности**: `{bragg_csv}`")
        lines.append("")

    symptoms = diagnosis.symptoms
    actions = diagnosis.recommended_actions
    if symptoms or actions:
        sec_num = 6 if (profile_csv or bragg_csv) else 5
        lines.append(f"## {sec_num}. Рекомендации по дальнейшему уточнению")
        lines.append("")
        if symptoms:
            lines.append("### Выявленные симптомы:")
            for s in symptoms:
                lines.append(f"- **[{s.severity}] {s.name}**: {s.description}")
            lines.append("")
        if actions:
            lines.append("### Рекомендуемые действия:")
            for idx, a in enumerate(actions, 1):
                lines.append(f"{idx}. {a}")
            lines.append("")

    content = "\n".join(lines)
    if out_path:
        out_path = Path(out_path)
        out_path.write_text(content, encoding="utf-8")
        logger.info(f"Report written to {out_path}")

    return content


# CLI этого модуля удалён: отчёт печатает `xrd_sample export`, файл при
# необходимости создаёт вызывающий. Функция выше остаётся библиотечной.
