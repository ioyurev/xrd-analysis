"""Экспорт уточнённого проекта: артефакты, вердикт и отчёт.

Функция вынесена из монолита `xrd_project.py`: она отвечает только за выгрузку
и печать, а числа берёт из одного словаря сводки, а не собирает их повторно.
"""

from __future__ import annotations

import io
import json
import sys

# Принудительная инициализация UTF-8 для безопасного вывода греческих символов (Δ, σ, Å) на Windows
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from constants import DEFAULT_GII_REJECT_THRESHOLD, DEFAULT_WAVELENGTH_KA1
from gatekeeper import CrystallochemicalGatekeeper, GiiStatus

# Add scripts directory to path for sibling imports
from gsas2_adapter import (
    build_scaled_phase_bragg_peaks,
    calculate_profile_metrics,
    ensure_fresh_residuals,
    export_phase_cif_safe,
    extract_bragg_reflections,
    extract_cell_parameters,
    extract_phase_scales,
    extract_profile_curves,
    get_project_qpa,
    phase_rb_percent,
    phase_rf_percent,
    sync_pawley_reflections_to_hist,
    write_bragg_reflections_csv,
)
from loguru import logger
from peaks import extract_peaks_robust
from pymatgen.core.structure import Structure
from stage_gate_rietveld import (
    ADVANCED_STAGES,
    RefinementStatus,
    SuccessGateInputs,
    ensure_gsas2_environment,
    evaluate_success_gates,
    normalize_stage_name,
    undefined_esd_problems,
)
from xrd_diagnose import (
    DiagnosticSymptom,
    MicrostructureMetrics,
    ResidualPeak,
    diagnose_residuals,
)
from xrd_plot import plot_gsas2_histogram
from xrd_project import (
    capture_engine_notes,
    is_pawley_phase,
    load_sidecar_history,
    print_json,
)
from xrd_report import generate_honest_markdown_report


@dataclass(frozen=True)
class PhaseExportSummary:
    """Typed export summary for a single crystalline phase."""

    weight_fraction_percent: float
    weight_fraction_esd: float | None
    scale: float
    cell_parameters: dict[str, float]
    microstructure: MicrostructureMetrics
    r_bragg_percent: float | None = None
    r_f_percent: float | None = None
    space_group: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "space_group": self.space_group,
            "weight_fraction_percent": self.weight_fraction_percent,
            "weight_fraction_esd": self.weight_fraction_esd,
            "scale": self.scale,
            "cell_parameters": dict(self.cell_parameters),
            "microstructure": self.microstructure.to_dict(),
        }
        if self.r_bragg_percent is not None:
            payload["r_bragg_percent"] = self.r_bragg_percent
        if self.r_f_percent is not None:
            payload["r_f_percent"] = self.r_f_percent
        return payload


@dataclass(frozen=True)
class InstrumentSummary:
    """Typed summary of histogram instrument parameters."""

    zero_deg: float | None = None
    sh_over_l: float | None = None
    ka2_ratio: float | None = None
    u: float | None = None
    v: float | None = None
    w: float | None = None
    wavelength_ka1: float | None = None
    wavelength_ka2: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "zero_deg": self.zero_deg,
            "sh_over_l": self.sh_over_l,
            "ka2_ratio": self.ka2_ratio,
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "wavelength_ka1": self.wavelength_ka1,
            "wavelength_ka2": self.wavelength_ka2,
        }


@dataclass(frozen=True)
class SampleSummary:
    """Typed summary of histogram sample parameters."""

    displace_x_um: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {"displace_x_um": self.displace_x_um}


@dataclass(frozen=True)
class BackgroundSummary:
    """Typed summary of Chebyshev background model."""

    bkg_type: str | None = None
    refine: bool | None = None
    n_terms: int = 0
    coeffs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.bkg_type,
            "refine": self.refine,
            "n_terms": self.n_terms,
            "coeffs": list(self.coeffs),
        }


@dataclass(frozen=True)
class ExportSummary:
    """Complete typed data contract of a Rietveld project export."""

    project: str
    status: RefinementStatus
    r_wp_percent: float
    r_p_percent: float
    chi_square: float
    phase_rb_percent: dict[str, float]
    r_exp_percent: float
    r_wp_over_r_exp: float
    wr_bkg_percent: float | None
    gof: float
    durbin_watson: float
    gii_value: float | None
    gii_status: GiiStatus
    free_parameters: int
    true_unindexed_peaks: list[ResidualPeak]
    intensity_mismatch_peaks: list[ResidualPeak]
    symptoms: list[DiagnosticSymptom]
    phases: dict[str, PhaseExportSummary]
    microstructure: dict[str, MicrostructureMetrics]
    instrument: InstrumentSummary
    sample: SampleSummary
    background: BackgroundSummary
    unmet_criteria: list[str]
    exportable: bool
    history: list[dict[str, Any]]
    rejection_reason: str | None = None
    profile_curves_csv: str | None = None
    bragg_reflections_csv: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project": self.project,
            "status": str(self.status),
            "r_wp_percent": self.r_wp_percent,
            "r_p_percent": self.r_p_percent,
            "chi_square": self.chi_square,
            "phase_rb_percent": dict(self.phase_rb_percent),
            "r_exp_percent": self.r_exp_percent,
            "r_wp_over_r_exp": self.r_wp_over_r_exp,
            "wr_bkg_percent": self.wr_bkg_percent,
            "gof": self.gof,
            "durbin_watson": self.durbin_watson,
            "gii_value": self.gii_value,
            "gii_status": str(self.gii_status),
            "free_parameters": self.free_parameters,
            "true_unindexed_peaks": [p.to_dict() for p in self.true_unindexed_peaks],
            "intensity_mismatch_peaks": [p.to_dict() for p in self.intensity_mismatch_peaks],
            "symptoms": [s.to_dict() for s in self.symptoms],
            "phases": {k: v.to_dict() for k, v in self.phases.items()},
            "microstructure": {k: v.to_dict() for k, v in self.microstructure.items()},
            "instrument": self.instrument.to_dict(),
            "sample": self.sample.to_dict(),
            "background": self.background.to_dict(),
            "unmet_criteria": list(self.unmet_criteria),
            "exportable": self.exportable,
            "history": list(self.history),
        }
        if self.rejection_reason:
            payload["rejection_reason"] = self.rejection_reason
        return payload


def _esd_or_none(value: float | None) -> float | None:
    """Вернуть esd или None, если она не определена.

    Нулевая и отрицательная σ означает отсутствие ковариации, а не идеальную
    точность: в сводке она должна быть ``null``, чтобы потребитель не напечатал
    голое число вместо неопределённости.
    """
    if value is None:
        return None
    number = float(value)
    return round(number, 2) if number > 0.0 else None


def run_export(args: SimpleNamespace) -> ExportSummary | int:
    """Export refined CIFs, publication plot, profile CSVs, and honest reports."""
    if not ensure_gsas2_environment():
        logger.error("GSAS-II calculation engine is not available.")
        return 1

    from GSASII import GSASIIscriptable as G2sc

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        logger.error(f"Project file not found: {project_path}")
        return 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = project_path.stem

    gpx = G2sc.G2Project(str(project_path))
    phases = gpx.phases()
    hists = gpx.histograms()
    if not hists:
        logger.error("Project contains no histograms.")
        return 1
    hist = hists[0]
    has_pawley = any(is_pawley_phase(p) for p in phases)
    if has_pawley:
        sync_pawley_reflections_to_hist(gpx, hist)
        with capture_engine_notes("export"):
            gpx.save()

    # 1. Export Refined CIFs with updated coordinates
    cif_exported_count = 0
    for p in phases:
        p_name = p.name
        cif_target = out_dir / f"{stem}_{p_name}_refined.cif"
        res = export_phase_cif_safe(p, cif_target)
        if res.exists() and res.stat().st_size > 0:
            cif_exported_count += 1
            logger.info(f"Exported refined CIF: {cif_target}")

    # 2. Export profile curves & reflections CSVs
    curves_csv = out_dir / "profile_curves.csv"
    refl_csv = out_dir / "bragg_reflections.csv"

    # Нативный hist.Export(..., 'histogram CSV') здесь больше не вызывается: измерено, что он
    # пишет не таблицу кривых, а дамп движка со своими блоками метаданных (первая строка
    # "Histogram, PWDR ...", далее "Instparm: Type, PXC"; 2031 строка против 2001 у адаптера),
    # и ни один потребитель его не читает. Единственная кривая таблицы для потребителей —
    # profile_curves.csv ниже; второй файл в 26 каталогах дублировал данные (2.33 МБ) и ничего не проверял.

    # Verified fallback via gsas2_adapter
    curves = extract_profile_curves(hist)
    np.savetxt(
        curves_csv,
        np.column_stack(
            [
                curves.two_theta,
                curves.y_obs,
                curves.y_calc,
                curves.y_bkg,
                curves.y_diff,
                curves.sigma_obs,
            ]
        ),
        delimiter=",",
        header="two_theta,y_obs,y_calc,y_bkg,y_diff,sigma_obs",
        comments="",
        fmt="%.5f",
    )

    refls = extract_bragg_reflections(hist)
    write_bragg_reflections_csv(refls, refl_csv)

    # 3. Export publication plot (600 DPI)
    plot_target = Path(args.plot).resolve() if args.plot else (out_dir / f"{stem}_rietveld.png")
    plot_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        plot_gsas2_histogram(
            hist,
            plot_target,
            title=f"Rietveld Refinement - {stem}",
            dpi=args.dpi,
            gpx=gpx,
        )
    except (RuntimeError, ValueError, KeyError, AttributeError, OSError, TypeError) as e_plot:
        logger.warning(f"Could not generate plot: {e_plot}")

    # 4. QPA via canonical SSOT: get_project_qpa() (native ComputeMassFracs + Hill-Howard Z*M fallback)
    scales = extract_phase_scales(gpx)
    cells = extract_cell_parameters(gpx)
    mass_fracs: dict[str, tuple[float, float | None]] = get_project_qpa(gpx, hist)

    vary_cnt = len(gpx.data.get("Controls", {}).get("VaryList", []))
    metrics = calculate_profile_metrics(curves, n_free_parameters=max(1, vary_cnt))
    history = load_sidecar_history(project_path)
    iterations = history.get("iterations") or []
    rule9_problems = iterations[-1].get("rule9_problems") if iterations else None
    rule9_measured = rule9_problems is not None

    # 5. Extract fresh residuals via native API
    fresh_residuals = ensure_fresh_residuals(gpx, hist)
    wr_bkg = fresh_residuals.get("wRb")

    # 6. Crystallochemical Gatekeeper audit on exported CIFs
    max_gii = 0.0
    gii_status = GiiStatus.PASSED
    is_crystallochemically_rejected = False
    rejection_reason = None

    def pauling_x(specie: Any) -> float:
        """Pauling electronegativity of a site species, with None/NaN (unknown) mapped to 0.0."""
        value = specie.X
        if value is None:
            return 0.0
        value = float(value)
        return value if np.isfinite(value) else 0.0

    for p in phases:
        if is_pawley_phase(p):
            continue
        p_name = p.name
        cif_target = out_dir / f"{stem}_{p_name}_refined.cif"
        if cif_target.exists():
            try:
                st = Structure.from_file(str(cif_target))

                # Domain-neutral electronegativity evaluation (Rule 8)
                anion_elements = [sp for sp in st.species if pauling_x(sp) >= 2.0]
                is_pure_ionic_oxide_halide = (
                    all(pauling_x(a) >= 3.0 for a in anion_elements) if anion_elements else False
                )
                custom_gii = args.gii_threshold
                effective_gii_limit = (
                    float(custom_gii)
                    if custom_gii is not None
                    else (DEFAULT_GII_REJECT_THRESHOLD if is_pure_ionic_oxide_halide else 0.65)
                )

                gatekeeper = CrystallochemicalGatekeeper(gii_reject_threshold=effective_gii_limit)
                audit = gatekeeper.audit_structure(st)
                gii_val = audit.gii_value
                if gii_val is not None and gii_val > max_gii:
                    max_gii = gii_val

                if not audit.is_valid:
                    is_crystallochemically_rejected = True
                    rejection_reason = (
                        "; ".join(audit.errors)
                        if audit.errors
                        else f"Crystallochemical validation failed for {p_name}."
                    )
                    gii_status = GiiStatus.FAIL
                elif audit.gii_status == GiiStatus.COVALENT_RELAXED:
                    if gii_status != GiiStatus.FAIL:
                        gii_status = GiiStatus.COVALENT_RELAXED
                    logger.info(
                        f"Phase {p_name} validated as COVALENT_RELAXED "
                        f"(GII={gii_val:.3f} v.u. <= {effective_gii_limit:.2f} v.u.)"
                    )
            except (
                ValueError,
                KeyError,
                AttributeError,
                OSError,
                TypeError,
                RuntimeError,
            ) as e_audit:
                logger.debug(f"Gatekeeper audit failed for {cif_target}: {e_audit}")

    # Determine status (Hard Gatekeeper block on GII / bond collapse)
    if is_crystallochemically_rejected:
        status_val = RefinementStatus.REJECTED_BY_GATEKEEPER
        logger.error(f"Gatekeeper rejection: {rejection_reason}")
        print(f"ERROR: Crystallochemical instability rejection: {rejection_reason}")
    else:
        # Provisional. The full Stage-Gate success gate set (SKILL.md Rule 9) is
        # evaluated below, once the microstructure and the residual diagnosis
        # exist. A single R_wp threshold is not a convergence criterion: a model
        # at R_wp = 12 % with 1 % microstrain and serial residuals is not
        # converged, and certifying it as such hides the defect the gates exist
        # to catch.
        status_val = RefinementStatus.NEEDS_REFINEMENT
    unmet_criteria: list[str] = []
    exportable = bool(metrics.r_wp_percent <= 10.0)

    # Фазовые R считаются здесь, а не только в воротах: сводка по фазам без них
    # печатает null там, где величина измерена (измерено: phase_rb_percent на
    # верхнем уровне содержит 8.51 и 34.10, а у фаз стояло None).
    gate_reflections = extract_bragg_reflections(hist)
    rb_phase = phase_rb_percent(gate_reflections)
    rf_phase = phase_rf_percent(gate_reflections)

    # Build per-phase summary, microstructure and R-factors
    phases_summary: dict[str, PhaseExportSummary] = {}
    microstructure_dict: dict[str, MicrostructureMetrics] = {}
    for p in phases:
        r_vals = p.data.get("General", {}).get("Rvalues", {})
        r_b_val = (
            float(r_vals["RF^2"])
            if "RF^2" in r_vals
            else (float(r_vals["RB"]) if "RB" in r_vals else None)
        )
        r_f_val = float(r_vals["RF"]) if "RF" in r_vals else None

        micro_entry = MicrostructureMetrics()
        with suppress(RuntimeError, ValueError, KeyError, AttributeError, IndexError, TypeError):
            hap = p.getHAPvalues(hist.name)
            size_nm = None
            strain_pct = None
            if "Size" in hap and len(hap["Size"]) > 1 and len(hap["Size"][1]) > 0:
                s_um = float(hap["Size"][1][0])
                if s_um > 0:
                    size_nm = s_um * 1000.0  # um to nm
            if "Mustrain" in hap and len(hap["Mustrain"]) > 1 and len(hap["Mustrain"][1]) > 0:
                m_raw = float(hap["Mustrain"][1][0])
                if m_raw > 0:
                    strain_pct = m_raw / 10000.0  # microstrain to percent

            if size_nm is not None or strain_pct is not None:
                lam_ang = DEFAULT_WAVELENGTH_KA1
                lam_entry = hist.InstrumentParameters.get("Lam", [DEFAULT_WAVELENGTH_KA1])
                if len(lam_entry) > 0:
                    lam_ang = float(lam_entry[0])
                lam_nm = lam_ang / 10.0
                x_deg = (
                    round(float((180.0 * 0.90 * lam_nm) / (np.pi * max(size_nm, 1e-3))), 4)
                    if size_nm is not None
                    else None
                )
                y_deg = (
                    round(float((strain_pct * 720.0) / (100.0 * np.pi)), 4)
                    if strain_pct is not None
                    else None
                )
                micro_entry = MicrostructureMetrics(
                    crystallite_size_nm=round(size_nm, 2) if size_nm is not None else None,
                    microstrain_percent=round(strain_pct, 4) if strain_pct is not None else None,
                    x_lorentzian_deg=x_deg,
                    y_lorentzian_deg=y_deg,
                )
        microstructure_dict[p.name] = micro_entry

        r_b_final = r_b_val if r_b_val is not None else rb_phase.get(p.name)
        r_f_final = r_f_val if r_f_val is not None else rf_phase.get(p.name)
        sg_str = str(p.data.get("General", {}).get("SGData", {}).get("SpGrp", "")).strip()
        phases_summary[p.name] = PhaseExportSummary(
            weight_fraction_percent=round(mass_fracs.get(p.name, (0.0, 0.0))[0], 2),
            weight_fraction_esd=_esd_or_none(mass_fracs.get(p.name, (0.0, None))[1]),
            scale=scales.get(p.name, 0.0),
            cell_parameters=cells.get(p.name, {}),
            microstructure=micro_entry,
            r_bragg_percent=r_b_final,
            r_f_percent=r_f_final,
            space_group=sg_str,
        )

    # 6.5. Honest physical residual diagnostic (Yobs - Ycalc)
    phase_bragg_dict = build_scaled_phase_bragg_peaks(curves, refls, scales)

    # Наблюдаемые пики передаются в диагностику: без них максимум остатка там, где
    # в измерении пика нет, считался бы неиндицированным отражением.
    observed_peaks = extract_peaks_robust(curves.two_theta, curves.y_obs).two_theta
    diag_res = diagnose_residuals(
        observed_peaks_2theta=observed_peaks,
        two_theta=curves.two_theta,
        y_obs=curves.y_obs,
        y_calc=curves.y_calc,
        wavelength=args.wavelength,
        phase_bragg_peaks=phase_bragg_dict,
        y_bkg=curves.y_bkg,
        angular_tolerance=0.20,
        snr_threshold=3.0,
        min_peak_snr=3.0,
    )

    # 6.6. Full Stage-Gate success gate (SKILL.md Rule 9), evaluated by the module
    # that owns the protocol, so the criteria stay testable without a project.
    # Погрешность доли существует только там, где масштабный фактор был в последнем
    # уточнении: ComputeMassFracs берёт σ из ковариации, а если масштабы не входили
    # в varyList, σ равна нулю. Это отсутствие измерения, и его надо назвать.
    qpa_esd_reason: str | None = None
    if not has_pawley and mass_fracs and all(esd is None for _pct, esd in mass_fracs.values()):
        # Две разные причины, и путать их нельзя: у однофазного образца доля равна
        # 100 % по определению — погрешности не существует в принципе, а не потому,
        # что её не измерили.
        if len(mass_fracs) == 1:
            qpa_esd_reason = (
                "esd массовой доли не существует: образец однофазный, доля 100 % по определению"
            )
        else:
            qpa_esd_reason = (
                "esd массовых долей не определена: масштабные факторы не входили в "
                "последнее уточнение — завершите маршрут стадией, где масштабы свободны"
            )

    if not is_crystallochemically_rejected:
        unmet_criteria = evaluate_success_gates(
            SuccessGateInputs(
                r_wp_percent=float(metrics.r_wp_percent),
                r_p_percent=float(metrics.r_p_percent),
                phase_weights_percent={
                    name: float(pct) for name, (pct, _esd) in mass_fracs.items()
                },
                rb_by_phase=rb_phase,
                rf_by_phase=rf_phase,
                rule9_problems=rule9_problems,
                rule9_measured=rule9_measured,
                esd_problems=undefined_esd_problems(
                    dict(
                        zip(
                            gpx.data.get("Covariance", {}).get("varyList", []),
                            gpx.data.get("Covariance", {}).get("sig", []),
                        )
                    )
                ),
                durbin_watson=float(metrics.durbin_watson),
                max_microstrain_percent=max(
                    (float(v.microstrain_percent or 0.0) for v in microstructure_dict.values()),
                    default=0.0,
                ),
                max_crystallite_size_nm=max(
                    (float(v.crystallite_size_nm or 0.0) for v in microstructure_dict.values()),
                    default=0.0,
                ),
                unindexed_count=diag_res.true_unindexed_count,
                qpa_esd_reason=qpa_esd_reason,
            )
        )
        status_val = (
            RefinementStatus.CONVERGED if not unmet_criteria else RefinementStatus.NEEDS_REFINEMENT
        )
        if unmet_criteria:
            logger.info(
                "Export status {}; unmet criteria: {}", status_val, "; ".join(unmet_criteria)
            )

    # 6.7. Histogram-level parameters that the verdict and any comparison with a
    # reference model depend on. The K-alpha2 ratio is reported under its own
    # name: the state belongs to the data set and is read from the project rather
    # than inferred.
    inst_raw = hist.data.get("Instrument Parameters", [{}])[0]

    def _inst_value(name: str) -> float | None:
        rec = inst_raw.get(name) if isinstance(inst_raw, dict) else None
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            try:
                return float(rec[1])
            except (TypeError, ValueError):
                return None
        return None

    bkg_raw = hist.data.get("Background", [[None, False, 0]])[0]
    n_bkg = int(bkg_raw[2]) if len(bkg_raw) > 2 else 0
    disp_raw = hist.data.get("Sample Parameters", {}).get("DisplaceX", [None, False])
    instrument_summary = InstrumentSummary(
        zero_deg=_inst_value("Zero"),
        sh_over_l=_inst_value("SH/L"),
        ka2_ratio=_inst_value("I(L2)/I(L1)"),
        u=_inst_value("U"),
        v=_inst_value("V"),
        w=_inst_value("W"),
        wavelength_ka1=_inst_value("Lam1") or _inst_value("Lam"),
        wavelength_ka2=_inst_value("Lam2"),
    )
    sample_summary = SampleSummary(
        displace_x_um=float(disp_raw[0])
        if isinstance(disp_raw, (list, tuple)) and disp_raw[0] is not None
        else None,
    )
    background_summary = BackgroundSummary(
        bkg_type=str(bkg_raw[0]) if len(bkg_raw) > 0 else None,
        refine=bool(bkg_raw[1]) if len(bkg_raw) > 1 else None,
        n_terms=n_bkg,
        coeffs=[float(x) for x in bkg_raw[3 : 3 + n_bkg]] if len(bkg_raw) > 3 else [],
    )

    summary = ExportSummary(
        project=str(project_path),
        status=status_val,
        r_wp_percent=metrics.r_wp_percent,
        r_p_percent=metrics.r_p_percent,
        chi_square=metrics.reduced_chi2,
        phase_rb_percent=rb_phase,
        r_exp_percent=round(metrics.r_exp_percent, 2),
        r_wp_over_r_exp=round(metrics.r_wp_over_r_exp_ratio, 2),
        wr_bkg_percent=wr_bkg,
        gof=metrics.gof,
        durbin_watson=metrics.durbin_watson,
        gii_value=round(max_gii, 4) if max_gii > 0.0 else None,
        gii_status=gii_status,
        free_parameters=vary_cnt,
        true_unindexed_peaks=diag_res.true_unindexed_reflections,
        intensity_mismatch_peaks=diag_res.intensity_mismatch_reflections,
        symptoms=diag_res.symptoms,
        phases=phases_summary,
        microstructure=microstructure_dict,
        instrument=instrument_summary,
        sample=sample_summary,
        background=background_summary,
        unmet_criteria=unmet_criteria,
        exportable=exportable,
        history=history.get("iterations", []),
        rejection_reason=rejection_reason,
        profile_curves_csv=str(curves_csv),
        bragg_reflections_csv=str(refl_csv),
    )

    summary_dict = summary.to_dict()
    summary_target = out_dir / "summary.json"
    summary_target.write_text(
        json.dumps(summary_dict, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if getattr(args, "json", False):
        print_json(summary_dict)

    report_content = generate_honest_markdown_report(
        summary=summary,
        diagnosis=diag_res,
    )
    # 8. Отчёт печатается, а не сохраняется: файл создаёт вызывающий, если он нужен.
    print(report_content)

    logger.info(f"Export completed to {out_dir} (status: {status_val})")
    last_stage = history.get("last_stage", "")
    is_advanced_stage = (
        normalize_stage_name(str(last_stage)) in ADVANCED_STAGES if last_stage else False
    )
    is_converged = status_val in (RefinementStatus.PASS, RefinementStatus.CONVERGED)

    if not (is_converged and is_advanced_stage):
        if not args.allow_incomplete:
            print("\n" + "!" * 70)
            print(">>> EXPORT_WARNING: PRELIMINARY_UNREFINED")
            print(
                f"Project status is '{status_val}', last completed stage is '{last_stage}' (R_wp = {metrics.r_wp_percent:.2f}%)."
            )
            print("The model has not satisfied all convergence criteria or completed Stage 7.")
            print("Pass --allow-incomplete to suppress this notice.")
            print("!" * 70 + "\n")
        if args.strict:
            logger.error(
                f"Export failed under --strict: status is '{status_val}', last stage is '{last_stage}'."
            )
            return 1

    print(
        f"SUCCESS: Exported {cif_exported_count} CIFs, profile curves, Bragg reflections "
        f"and plot to {out_dir}"
    )
    return summary
