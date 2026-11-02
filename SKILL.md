---
name: xrd-analysis
description: Crystallographic analysis of powder X-ray diffraction data — search-match phase identification, Le Bail whole-pattern decomposition, Stage-Gate Rietveld refinement, and validation of a crystal-chemical model. Use this skill for any work with a measured diffractogram even if the user does not say XRD and asks to match this pattern, fit the lattice, check whether the sample is single-phase, explain extra peaks, or compare diffraction with thermodynamics or microscopy. Not for single-crystal structure solution or neutron data unless the user asks to treat it as a powder pattern.
license: BSD-3-Clause
compatibility: Requires Python 3.11+, uv, diffpy-srfit, cctbx-base, pymatgen, gemmi, spglib, scipy, xrdplot (figure rendering); the Materials Project client is optional and runs in a separate environment named by MP_PYTHON
---

# XRD Crystallographic Analysis Skill

Use this skill to perform end-to-end crystallographic analysis of powder X-ray diffraction patterns:
1. Pre-inspect raw scans for baseline curvature, signal-to-noise ratio, peak widths, and amorphous humps.
2. Build a chemically complete candidate pool from synthesis provenance and run analytical Search-Match against local CIFs, the local COD HSRDB database (`COD_HSRDB_PATH`), Materials Project (`MP_API_KEY`), and ICDD PDF-2 (`PDF2_DATA_DIR`).
3. Execute step-by-step Stage-Gate Rietveld refinement (or structureless Pawley decomposition when atomic coordinates are absent) with automatic divergence rollback.
4. Validate refined models against IUCr agreement gates and crystallochemical criteria (Bond Valence Sum, Global Instability Index GII, periodic hard-sphere limits, and ITA Vol. A symmetry).

## Reference files — load on demand

Load only the reference file needed for the current step (all relative paths resolve against `.agents/skills/xrd-analysis/`):

| Task or question | Load this file |
|---|---|
| 12-stage parameter release sequence, mathematical models (Chebyshev, $B_{\text{iso}}$, Scherrer/Stokes-Wilson) & IUCr acceptance gates | `references/stage-gate-protocol.md` |
| Difference-curve symptom diagnosis ($Y_{\text{obs}} - Y_{\text{calc}}$) & catalog of non-structural peaks ($K\beta$, tube lines, holder) | `references/rietveld-troubleshooting.md` |
| Synthesis provenance rules: volatility series, container thermodynamics ($\Delta G_f^\circ$), gas sorption & Vegard's law | `references/chemical-heuristics.md` |
| Ulrich Müller inorganic structural prototypes atlas (close packings, perovskite $t$-factor, spinels, Zintl phases) | `references/structural-prototypes.md` |
| Crystallochemical validation gates (BVS, GII thresholds, periodic hard-sphere distances, ITA standardization) | `references/crystallochemical-rules.md` |
| Low-level `GSASIIscriptable` API, recipe dictionaries & engine gotchas (load only when using `--override-json` or diagnosing engine errors) | `references/gsas2-scriptable-api.md` |

---

## Core Principles & Guardrails

### 1. Identification Completeness Precedes Metric Quality
Full indexing of observed peaks is a **precondition** for quantitative refinement, not its byproduct. While any observed peak above the detection threshold remains unexplained, refinement is used only to align peak positions and widths (Stages 1–3, 7), and agreement metrics ($R_{wp}, R_p, \text{GoF}, R_B, R_F$) are not treated as the objective.
- Indexing is closed only when every observed peak above the detection threshold is assigned to one of three categories:
  1. **Crystalline phase** — peak position matches within angular tolerance **and** relative intensity is physically consistent with the phase;
  2. **Non-structural peak** — attributed to a **named physical mechanism** (residual $\text{Cu}\,K\beta$, X-ray tube contaminant line $\text{W}\,L\alpha$, sample holder/substrate reflection, escape peak; see `references/rietveld-troubleshooting.md` §9) and verified by position and intensity ratio;
  3. **Unattributed reflection** — explicitly reported as unindexed with a physical hypothesis; the model is marked incomplete and quantitative results are labeled preliminary.
- A synthetic unit cell obtained by indexing or structureless fitting (Pawley / Le Bail) without an established chemical formula and published crystal-chemical prototype counts as an unattributed reflection, never as a completed phase identification.

### 2. Candidate Pool Completeness from Synthesis Provenance
Never narrow the candidate pool arbitrarily before analysis:
- Formulate the elemental and structural pool from the actual synthesis description using `references/chemical-heuristics.md` and `references/structural-prototypes.md` (precursors, adjacent phase-diagram compounds, atmospheric sorption of $\text{O, C, H}$, and container-reaction products when thermodynamically favored by $\Delta G_f^\circ$).
- Container material is a fact of the specific sample's synthesis route; never transplant crucible/ampoule materials from a different synthesis route.
- Only add a candidate phase to the active GSAS-II project if at least one of its major reflections is present in the observed peak table (`extract_peaks_robust`). Adding a phase with zero observed lines leaves its unit cell unconstrained and causes solver collapse.
- Retain a minor phase ($W_p < 0.5\text{ wt.}\%$) if and only if it is the sole explanation for observed reflections; otherwise remove it when its weight converges to zero without improving $R_{wp}$.

### 3. Stage-Gate Parameter Release & Guardrails
Release parameters strictly in physical order (`references/stage-gate-protocol.md`):
- **Stage 1 (`1_Scale_and_Background`):** Phase scales + Chebyshev background (4–6 terms; avoid $> 8$ terms to prevent background wiggle).
- **Stage 2 (`2_Zero_Shift`):** Detector `Zero` ($|\Delta 2\theta| \le 0.5^\circ$; constant offset) or specimen displacement `DisplaceX` ($\cos\theta$ law, $\pm 1000\ \mu\text{m}$).
- **Stage 3 (`3_Unit_Cell`):** Unit cell parameters ($a, b, c, \alpha, \beta, \gamma$, step drift $\le 5\%$). For trace microphases ($< 2.0\text{ wt.}\%$), keep the unit cell locked to the reference CIF to prevent metric tensor singularity.
- **Stage 4 (`4_Atomic_Coordinates`):** Fractional coordinates $(x, y, z)$ with automatic first-coordination-sphere bond distance restraints (`addDistRestraint`).
- **Stages 5–6 (`5_Overall_Displacement`, `6_Group_B_iso`):** Isotropic thermal displacements ($U_{\text{iso}} \in [0.001, 0.15]\ \text{Å}^2$, enforcing $B_{\text{cation}} \le B_{\text{anion}}$).
- **Stage 7 (`7_Sample_Profile_XY`):** Sample crystallite size (`Size`) and microstrain (`Mustrain`), optionally axial divergence `SH/L` ($\in [0, 0.05]$).
- **Stage 8 (`8_Preferred_Orientation`):** March-Dollase texture along a single cleavage axis ($r \in [0.7, 1.3]$).
- **Stage 9 (`9_Occupancy`):** Site occupancies ($F \in [0, 1]$) for partially occupied sites or `--occupancy-atoms`.
- **Stage 10 (`10_Instrument_Profile`):** Instrumental Caglioti terms ($U, V, W$), never sharpened below the canonical Bragg-Brentano optics floor ($\text{FWHM}_{\text{inst}} \approx 0.035\text{--}0.050^\circ$).
- **Stage 11 (`11_Absorption_Extinction`):** Sample absorption and phase extinction ($\ge 0$).
- **Stage 12 (`12_Final_Scale`):** Phase scales only (background locked) so the final covariance matrix provides defined standard uncertainties (e.s.d.) for QPA mass fractions.

### 4. Iterative Refinement Loop & Stopping Criteria
After each refinement pass, inspect the model state (`params dump` / `status`), run `diagnose`, and execute its `recommended_actions` until one of four gates triggers:
1. **Success Gate (`CONVERGED`):**
   - $R_{wp} \le 10.0\%$ and $R_p \le 10.0\%$;
   - Phase Bragg factors: dominant phase $R_B \le 10.0\%, R_F \le 6.0\%$; subdominant phases ($> 5\text{ wt.}\%$) $R_B \le 15.0\%, R_F \le 8.0\%$ (minor phases $\le 5\text{ wt.}\%$ are reported informatively);
   - Zero true unindexed reflections ($> 3\sigma$);
   - Parameter convergence: $\max |\Delta / \sigma| < 0.15$ (Rule 9) with defined positive e.s.d.s;
   - Physical profile bounds: crystallite size $< 900\text{ nm}$, microstrain $\varepsilon < 0.8\%$, parameter correlations $|\rho| \le 0.85$, condition number $\kappa(H) \le 10^5$;
   - Durbin-Watson statistic $d \in [1.5, 2.5]$;
   - Crystallochemical gatekeeper: $\text{GII} \le 0.20\text{ v.u.}$ for ionic oxides/halides ($\le 0.65\text{ v.u.}$ for covalent chalcogenides/pnictides/intermetallics) and no interatomic bond collapse ($d \ge 0.65(r_i + r_j)$).
2. **Plateau Gate:** If adding a parameter or trial modification improves $R_{wp}$ by $< 0.3\%$ (and does not explain previously unattributed observed peaks), revert the change and stop.
3. **Overfitting / Divergence Gate:** If a step causes $R_{wp}$ regression, $|\rho| > 0.85$, $\kappa(H) > 10^5$, or unphysical cell drift ($> 5\%$), roll back to the pre-step checkpoint.
4. **Data-Bound / Exhaustion Gate:** When counting statistics or severe multiphase nanosize overlap bound the achievable fit, report all unmet criteria explicitly and state the IUPAC $3\sigma$ limit of detection (typically $0.5\text{--}1.5\text{ wt.}\%$ for laboratory $\text{Cu}\,K\alpha$ XRD).

### 5. Operational Integrity Rules
- **One Sample, One Analysis (No Batch Processing):** Each diffractogram is analyzed individually in its own sample directory. Never apply a single batch recipe across a series without per-sample phase selection and residual verification.
- **Declared Document Wins:** In declarative mode (`params dump` $\to$ edit `params.json` $\to$ `params refine`), `params.json` is the single source of truth for refinement flags and bounds. Any temporary lock during sequential scale factorization must restore the declared flags before completing.
- **No Hand-Written CIFs or Surrogate Solvers:** Never fabricate CIF text by string concatenation (use `cif_retriever.py analog` or `add-phase --from-cod` / `--from-card`) and never replace GSAS-II with custom least-squares peak fitters.
- **Read-Only Skill Code:** Never modify `.agents/skills/xrd-analysis/scripts/*.py` during an analysis task. Pass custom settings via CLI flags (`--override-json`, `--texture-axis`, `params.json`) and report any engine/skill defects in the final report.

---

## CLI Workflow & Tools

All commands are invoked via `uv run py .agents/skills/xrd-analysis/scripts/<tool>.py ...`.

### 1. Pre-Inspect Raw Diffraction Pattern (`xrd_inspect.py`)
```bash
uv run py .agents/skills/xrd-analysis/scripts/xrd_inspect.py data/sample.xy --json
```
Reports $2\theta$ range, step size, SNR, low/high-angle FWHM, crystallinity grade, and amorphous background humps.

### 2. Search, Download & Generate Isostructural Analog CIFs (`cif_retriever.py`)
Configures database access from `.env`:
- `COD_HSRDB_PATH`: local SQLite archive of COD structures (`.hsrdb`, used automatically offline for `--source cod` / `all`, with transparent fallback to COD REST API);
- `MP_API_KEY` + `MP_PYTHON`: Materials Project API key and retrieval Python interpreter.

```bash
# 1. Search structures by 3-way element filter (--all, --any, --none, --strict, --subfiles):
uv run py .agents/skills/xrd-analysis/scripts/cif_retriever.py search \
  --all "Lu,O" --any "C,N,H" --strict --subfiles inorganic --source cod --limit 30

# 2. Download CIFs by explicit IDs (cod-<id> or mp-<id>):
uv run py .agents/skills/xrd-analysis/scripts/cif_retriever.py download \
  --ids "cod-1537844,mp-1269" --out-dir ./cifs

# 3. Generate an isostructural analog (by Shannon radii or direct d-spacing scaling):
uv run py .agents/skills/xrd-analysis/scripts/cif_retriever.py analog \
  --prototype cifs/Ho2OH4CO3.cif --substitute "Ho:Lu" --anion O --out-cif cifs/Lu2OH4CO3.cif

uv run py .agents/skills/xrd-analysis/scripts/cif_retriever.py analog \
  --prototype cifs/Phase.cif --scale-to-d 3.003 --hkl "2,2,2" --out-cif cifs/Phase_scaled.cif

# 4. Harvest and filter an existing local directory of CIFs:
uv run py .agents/skills/xrd-analysis/scripts/cif_retriever.py harvest \
  --local-dir ./raw_cifs --all "Sm,S" --strict --out-dir ./cifs
```

### 3. Analytical Search-Match (`search_match.py`)
Scores candidate phases against a raw scan (`--data`), a GSAS-II project's difference curve (`--project`), or an exported curve file (`--residual`).
- `--candidate-cifs <dir>`: score local CIF files;
- `--hsrdb`: score the local COD database (`COD_HSRDB_PATH` in `.env`);
- `--pdf2`: score the local user-supplied ICDD PDF-2 card database (`PDF2_DATA_DIR` in `.env`).

```bash
# Score local CIFs + local COD HSRDB against a raw pattern:
uv run py .agents/skills/xrd-analysis/scripts/search_match.py \
  --data data/sample.xy --candidate-cifs ./cifs --hsrdb \
  --elements "Lu,O" --any "C,N,H" --strict --profile ide-all --json

# Score PDF-2 and COD HSRDB against the residual of a GSAS-II project for minor phases:
uv run py .agents/skills/xrd-analysis/scripts/search_match.py \
  --project projects/sample.gpx --hsrdb --pdf2 \
  --any "Lu,Yb,Er,Si,O" --strict --profile minors --json
```

### 4. Sample Workspace & Stage-Gate Refinement (`xrd_sample.py`)
One sample is one folder (`--dir <sample_dir>`, defaulting to current working directory) with layout `data/`, `params.json`, `notes.md`, `projects/`, and `artifacts/`:

```bash
# 1. Initialize project from raw measurement (.xy / .asc / .txt / .dat):
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py init --data data/scan.asc --bkg-terms 6

# 2. Add phases (from CIF, from local COD HSRDB by COD ID, or from PDF-2 card; add --pawley for Pawley mode):
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py add-phase --cif cifs/Lu2O3.cif --name Lu2O3
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py add-phase --from-cod 1537844 --name Lu2O3
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py add-phase --from-card 78-0901 --name Phase1 --pawley

# 3. Toggle or remove phases, or revert to a pre-mutation checkpoint:
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py toggle-phase --phase Phase1 --disable
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py remove-phase --phase Phase1
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py revert

# 4. Run single Stage-Gate step or multi-stage sequence (runs until plateau by default):
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py step --stage 1_Scale_and_Background
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py step --sequence "1,2,3,5,7,12" --cycles 5
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py step --stage 8_Preferred_Orientation --texture-axis "Lu2O3:0,0,1"

# 5. Declarative Refinement-as-Code (params.json):
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py params dump --preset 3
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py params refine --cycles 5 --sequential-scales

# 6. Inspect status and run residual diagnostics:
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py status
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py diagnose --candidate-cifs ./cifs --hsrdb --any "Lu,O,C,N,H" --strict --out

# 7. Export refined CIFs, profile_curves.csv, bragg_reflections.csv, 600 DPI plot & IUCr report:
uv run py .agents/skills/xrd-analysis/scripts/xrd_sample.py export
```
