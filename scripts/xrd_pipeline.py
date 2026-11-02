"""CLI entry point for the autonomous XRD crystallographic analysis pipeline."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from loguru import logger
from pymatgen.core import Structure

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export import ExportSummary
from gatekeeper import GiiStatus
from instrument import CanonicalInstrument
from search_match import CandidateScoringEngine, ProfileProbabilityField
from stage_gate_rietveld import (
    RefinementConfig,
    RefinementStatus,
    StageGateRietveld,
    ensure_gsas2_environment,
)
from xrd_diagnose import MicrostructureMetrics
from xrd_inspect import PatternInspectionReport, inspect_powder_pattern, load_xy_data
from xrd_project import cmd_add_phase, cmd_export, cmd_init, cmd_step


@dataclass(frozen=True)
class IdentifiedPhase:
    """Strongly typed descriptor of an identified crystalline phase."""

    phase: str
    coverage: float
    matched_peaks: int
    total_peaks: int
    estimated_fraction: float
    residual_explained: float
    cell_delta_a: float = 0.0
    weight_fraction_wt: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "coverage": self.coverage,
            "matched_peaks": self.matched_peaks,
            "total_peaks": self.total_peaks,
            "estimated_fraction": self.estimated_fraction,
            "residual_explained": self.residual_explained,
            "cell_delta_a": self.cell_delta_a,
            "weight_fraction_wt": self.weight_fraction_wt,
        }


@dataclass(frozen=True)
class PipelineReport:
    """Strongly typed data contract for the entire pipeline run."""

    status: str
    data_file: str
    data_points: int
    two_theta_range: tuple[float, float]
    phases_identified: list[IdentifiedPhase]
    audit_passed: bool
    stage_gate_protocol: list[str]
    stage_results: list[dict] = field(default_factory=list)
    refined_cif: str | None = None
    plot_image: str | None = None
    pre_inspection: PatternInspectionReport | None = None
    gii_status: GiiStatus | str | None = None
    gii_value: float | None = None
    qpa_weights_wt: dict[str, float] = field(default_factory=dict)
    r_p: float = 0.0
    durbin_watson: float = 2.0
    phase_r_factors: dict[str, dict[str, float]] = field(default_factory=dict)
    cell_parameters_esd: dict[str, dict[str, float]] = field(default_factory=dict)
    microstructure: dict[str, MicrostructureMetrics] = field(default_factory=dict)
    profile_curves_path: str | None = None
    bragg_reflections_path: str | None = None
    true_unindexed_peaks_count: int = 0
    intensity_defect_peaks_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "data_file": self.data_file,
            "data_points": self.data_points,
            "two_theta_range": [self.two_theta_range[0], self.two_theta_range[1]],
            "phases_identified": [p.to_dict() for p in self.phases_identified],
            "audit_passed": self.audit_passed,
            "stage_gate_protocol": list(self.stage_gate_protocol),
            "stage_results": list(self.stage_results),
            "refined_cif": self.refined_cif,
            "plot_image": self.plot_image,
            "pre_inspection": (
                self.pre_inspection.to_dict() if self.pre_inspection is not None else None
            ),
            "gii_status": str(self.gii_status) if self.gii_status is not None else None,
            "gii_value": self.gii_value,
            "qpa_weights_wt": dict(self.qpa_weights_wt),
            "r_p": self.r_p,
            "durbin_watson": self.durbin_watson,
            "phase_r_factors": dict(self.phase_r_factors),
            "cell_parameters_esd": dict(self.cell_parameters_esd),
            "microstructure": {k: v.to_dict() for k, v in self.microstructure.items()},
            "profile_curves_path": self.profile_curves_path,
            "bragg_reflections_path": self.bragg_reflections_path,
            "true_unindexed_peaks_count": self.true_unindexed_peaks_count,
            "intensity_defect_peaks_count": self.intensity_defect_peaks_count,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def run_pipeline(
    data_path: Path,
    cif_dir: Path | None = None,
    local_cif_dir: Path | None = None,
    elements: list[str] | None = None,
    plot_path: Path | None = None,
    output_dir: Path | None = None,
    config: RefinementConfig | None = None,
) -> PipelineReport:
    """Execute the thin-glue XRD analysis pipeline."""
    out_dir = output_dir or (Path(__file__).resolve().parent.parent / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []
    errors_list: list[str] = []
    phases_identified: list[IdentifiedPhase] = []
    stage_results_dicts: list[dict] = []

    # Optional local CIF pool harvesting
    if local_cif_dir and local_cif_dir.exists():
        try:
            from cif_retriever import harvest_local_cifs

            cif_pool_dir = out_dir / "cifs"
            harvest_local_cifs(local_cif_dir, cif_pool_dir, elements=elements)
            cif_dir = cif_pool_dir
        except (ImportError, OSError, ValueError) as err:
            warnings_list.append(f"Local CIF harvesting warning: {err}")

    # 1. Load data & Apply canonical instrument bounds
    instrument = CanonicalInstrument()
    logger.debug(
        f"Initialized canonical instrument profile (U={instrument.u_inst}, V={instrument.v_inst}, W={instrument.w_inst})"
    )
    two_theta, intensity = load_xy_data(data_path)  # noqa: PU001
    data_points = len(two_theta)
    two_theta_range = (float(np.min(two_theta)), float(np.max(two_theta)))

    # Pre-inspection of experimental pattern
    try:
        pre_insp = inspect_powder_pattern(two_theta, intensity)
    except (ValueError, OSError, RuntimeError) as e:
        logger.debug(f"Pre-inspection note: {e}")
        pre_insp = None

    # 2. Search-Match against candidate CIF pool
    identified_structures: list[Structure] = []
    candidate_cif_map: dict[str, Path] = {}
    if cif_dir and cif_dir.exists():
        engine = CandidateScoringEngine()
        candidate_files = list(cif_dir.glob("*.cif"))
        candidates: dict[str, Structure] = {}

        for cf in candidate_files:
            try:
                struct = Structure.from_file(str(cf))
                if elements:
                    struct_elems = {s.symbol for s in struct.composition.elements}
                    if not struct_elems.issubset(set(elements)):
                        continue
                candidates[cf.stem] = struct
                candidate_cif_map[cf.stem] = cf
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                warnings_list.append(f"Failed to parse {cf.name}: {e}")

        if candidates:
            prob_field = ProfileProbabilityField.from_raw_pattern(two_theta, intensity)
            ranked = engine.score_candidates(
                two_theta=two_theta,
                y_net=prob_field.y_net,
                sigma=prob_field.sigma,
                candidate_structures=candidates,
                observed_signal=intensity,
            )

            # Deterministic selection for autonomous pipeline:
            # 1. Primary candidate with Score >= 50%
            # 2. Secondary candidate if Score >= 35%, no VETO, and >= 2 unique reflections
            selected_results = []
            if (
                ranked
                and ranked[0].score >= 50.0
                and not any("VETO" in f for f in ranked[0].diagnostic_flags)
            ):
                selected_results.append(ranked[0])
                first_peaks = np.array(ranked[0].explained_2thetas)

                for cand in ranked[1:]:
                    if len(selected_results) >= 2:
                        break
                    if cand.score < 35.0 or any("VETO" in f for f in cand.diagnostic_flags):
                        continue
                    unique_peaks = [
                        p
                        for p in cand.explained_2thetas
                        if len(first_peaks) == 0 or np.min(np.abs(first_peaks - p)) > 0.20
                    ]
                    if len(unique_peaks) >= 2:
                        selected_results.append(cand)

            total_scale = sum(r.scale_factor for r in selected_results)
            for r in selected_results:
                frac = r.scale_factor / total_scale if total_scale > 0 else 1.0
                phases_identified.append(
                    IdentifiedPhase(
                        phase=r.phase_name,
                        coverage=round(r.completeness_ml_tl, 3),
                        matched_peaks=r.matched_lines,
                        total_peaks=r.total_lines,
                        estimated_fraction=round(frac, 3),
                        residual_explained=round(r.pos_overlap, 3),
                        cell_delta_a=round(r.lattice_shift_da_a, 4),
                    )
                )
                identified_structures.append(r.structure)

    # 3. Stage-Gate Rietveld refinement execution via xrd_project workbench
    project_gpx = out_dir / "project.gpx"
    target_plot_path = plot_path or (out_dir / "rietveld_plot.png")
    bkg_terms = config.bkg_chebyshev_terms if config else 6
    stage_results_dicts: list[dict] = []

    has_engine = ensure_gsas2_environment()
    if has_engine and phases_identified:
        # Команды вызываются типизированно: аргументы — объект, а не строка
        # командной строки, поэтому поля называются явно.

        # Step A: init
        cmd_init(
            SimpleNamespace(
                data=data_path,
                project=project_gpx,
                instrument="canonical",
                wavelength=instrument.wavelength_ka1,
                bkg_terms=bkg_terms,
                ka2_ratio=None,
            )
        )

        # Step B: add-phase
        for iphase in phases_identified:
            cif_file = candidate_cif_map.get(iphase.phase) or (
                cif_dir / f"{iphase.phase}.cif" if cif_dir else None
            )
            if cif_file and cif_file.exists():
                cmd_add_phase(
                    SimpleNamespace(
                        project=project_gpx,
                        cif=cif_file,
                        from_card=None,
                        name=iphase.phase,
                        scale=max(0.01, iphase.estimated_fraction),
                    )
                )

        # Step C: step (все стадии Stage-Gate)
        tex_axes: list[str] = []
        if config and config.texture_axes:
            tex_axes = [
                f"{p}:{','.join(str(x) for x in hkl)}" for p, hkl in config.texture_axes.items()
            ]

        refine_displacement = config.refine_sample_displacement if config else True
        refine_axial = config.refine_axial_divergence if config else False

        for stage_name in StageGateRietveld.STAGES:
            axes = tex_axes if stage_name == "8_Preferred_Orientation" else []
            step_rc = cmd_step(
                SimpleNamespace(
                    project=project_gpx,
                    stage=stage_name,
                    sequence=None,
                    bkg_terms=bkg_terms,
                    cycles=5,
                    refine_displacement=bool(refine_displacement),
                    refine_axial=bool(refine_axial),
                    texture_axis=list(axes),
                    uniaxial_size=[],
                    init_md=0.95,
                    override_json=None,
                    force=False,
                    keep_gsas_listing=False,
                )
            )
            stage_results_dicts.append(
                {
                    "stage_name": stage_name,
                    "status": str(
                        RefinementStatus.SUCCESS if step_rc == 0 else RefinementStatus.FAILED
                    ),
                    "notes": f"Workbench step: {stage_name}",
                }
            )

        # Step D: export печатает отчёт и возвращает сводку.
        summary_data = cmd_export(
            SimpleNamespace(
                project=project_gpx,
                out_dir=out_dir,
                plot=target_plot_path,
                wavelength=instrument.wavelength_ka1,
                gii_threshold=None,
                dpi=600,
                strict=False,
                allow_incomplete=False,
                json=False,
            )
        )
    else:
        stage_results_dicts = [
            {
                "stage_name": s,
                "status": "SKIPPED",
                "notes": "GSAS-II engine required" if not has_engine else "No phases identified",
            }
            for s in StageGateRietveld.STAGES
        ]

    summary_obj: ExportSummary | None = (
        summary_data if isinstance(summary_data, ExportSummary) else None
    )

    qpa_weights_wt: dict[str, float] = (
        {ph_k: float(ph_v.weight_fraction_percent) for ph_k, ph_v in summary_obj.phases.items()}
        if summary_obj is not None
        else {}
    )

    if qpa_weights_wt:
        updated_phases = []
        for p in phases_identified:
            wt = qpa_weights_wt.get(p.phase)
            updated_phases.append(
                IdentifiedPhase(
                    phase=p.phase,
                    coverage=p.coverage,
                    matched_peaks=p.matched_peaks,
                    total_peaks=p.total_peaks,
                    estimated_fraction=p.estimated_fraction,
                    residual_explained=p.residual_explained,
                    cell_delta_a=p.cell_delta_a,
                    weight_fraction_wt=wt,
                )
            )
        phases_identified = updated_phases

    profile_curves_path = out_dir / "profile_curves.csv"
    bragg_reflections_path = out_dir / "bragg_reflections.csv"

    refined_cif_files = sorted(out_dir.glob("*_refined.cif"))
    refined_cif_path = refined_cif_files[0] if refined_cif_files else out_dir / "refined.cif"

    raw_status = summary_obj.status if summary_obj is not None else RefinementStatus.SUCCESS
    audit_passed = raw_status != RefinementStatus.REJECTED_BY_GATEKEEPER
    pipeline_status = (
        RefinementStatus.REJECTED_BY_GATEKEEPER
        if raw_status == RefinementStatus.REJECTED_BY_GATEKEEPER
        else (
            RefinementStatus.SUCCESS
            if (has_engine and phases_identified)
            else RefinementStatus.SKIPPED
        )
    )

    report = PipelineReport(
        status=pipeline_status,
        data_file=str(data_path),
        data_points=data_points,
        two_theta_range=two_theta_range,
        phases_identified=phases_identified,
        audit_passed=audit_passed,
        stage_gate_protocol=StageGateRietveld.STAGES,
        stage_results=stage_results_dicts,
        refined_cif=str(refined_cif_path)
        if refined_cif_path.exists()
        else (str(refined_cif_files[0]) if refined_cif_files else None),
        plot_image=str(target_plot_path) if target_plot_path.exists() else None,
        pre_inspection=pre_insp,
        gii_status=summary_obj.gii_status if summary_obj is not None else None,
        gii_value=summary_obj.gii_value if summary_obj is not None else None,
        qpa_weights_wt=qpa_weights_wt,
        r_p=float(summary_obj.r_p_percent) if summary_obj is not None else 0.0,
        durbin_watson=float(summary_obj.durbin_watson) if summary_obj is not None else 2.0,
        phase_r_factors={
            k: {
                "r_f_percent": float(v.r_f_percent or 0.0),
                "r_bragg_percent": float(v.r_bragg_percent or 0.0),
            }
            for k, v in (summary_obj.phases.items() if summary_obj is not None else ())
        },
        cell_parameters_esd={
            k: dict(v.cell_parameters)
            for k, v in (summary_obj.phases.items() if summary_obj is not None else ())
        },
        microstructure=dict(summary_obj.microstructure) if summary_obj is not None else {},
        profile_curves_path=str(profile_curves_path) if profile_curves_path.exists() else None,
        bragg_reflections_path=str(bragg_reflections_path)
        if bragg_reflections_path.exists()
        else None,
        true_unindexed_peaks_count=len(summary_obj.true_unindexed_peaks)
        if summary_obj is not None
        else 0,
        intensity_defect_peaks_count=len(summary_obj.intensity_mismatch_peaks)
        if summary_obj is not None
        else 0,
        warnings=warnings_list,
        errors=errors_list,
    )

    return report
