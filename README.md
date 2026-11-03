# xrd-analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23892876.svg)](https://doi.org/10.5281/zenodo.23892876)

[Русская версия (README.ru.md)](README.ru.md)

**Autonomous Crystallographic Analysis and Stage-Gate Rietveld / Pawley Refinement Skill**

`xrd-analysis` is an Agent Skill (`SKILL.md`) and Python crystallographic workbench built on top of the open-source [GSAS-II](https://github.com/AdvancedPhotonSource/GSAS-II) calculation engine (`GSASIIscriptable`), `pymatgen`, `gemmi` and `spglib`.

- **Primary repository (GitFlic):** [https://gitflic.ru/project/i-o-yurev/xrd-analysis](https://gitflic.ru/project/i-o-yurev/xrd-analysis)
- **Mirror repository (GitHub):** [https://github.com/ioyurev/xrd-analysis](https://github.com/ioyurev/xrd-analysis)

## Key Capabilities

- **Raw Pattern Pre-Inspection (`scripts/xrd_inspect.py`):** Evaluates $2\theta$ range, step size, signal-to-noise ratio (SNR), low/high-angle FWHM, crystallinity grade, and amorphous background humps.
- **Structure Retrieval & Isostructural Analog Generation (`scripts/cif_retriever.py`):** Queries the Crystallography Open Database (COD) and Materials Project, and generates isostructural analogs via Shannon ionic radii or direct $d_{hkl}$ metric-tensor scaling while preserving space-group symmetry via `gemmi`.
- **Analytical Search-Match (`scripts/search_match.py`):** Multi-component profile scoring against local CIF pools, local COD SQLite archives (`COD_HSRDB_PATH`), and user-licensed ICDD PDF-2 installations (`PDF2_DATA_DIR`).
- **Stateful Sample Workbench (`scripts/xrd_sample.py`):** Manages one sample per directory (`data/`, `params.json`, `projects/`, `artifacts/`) with 12-stage Stage-Gate Rietveld refinement, structureless Pawley decomposition (`--pawley`), declarative `params.json` refinement-as-code, automatic divergence rollback, residual diagnostics (including $\text{Cu}\,K\beta$ satellite classification), and IUCr / Bond-Valence (GII) validation gates.

## Installation

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python scripts/install_gsas2.py
```

### Optional Environment Configuration (`.env`)

Database integrations are configured via environment variables or a `.env` file in the working directory (no proprietary databases are bundled with this repository):

```ini
# Optional: local SQLite archive of COD structures
COD_HSRDB_PATH=/path/to/COD24_HS4x.hsrdb

# Optional: user-licensed local ICDD PDF-2 installation directory
PDF2_DATA_DIR=/path/to/PDF2
```

## Quick Start

See [`SKILL.md`](SKILL.md) for the complete operational protocol and CLI reference:

```bash
# 1. Initialize sample workspace from powder diffraction scan
uv run python scripts/xrd_sample.py --dir ./my_sample init --data data/scan.xy --bkg-terms 6

# 2. Add crystalline phase (from CIF, local COD archive, or PDF-2 card)
uv run python scripts/xrd_sample.py --dir ./my_sample add-phase --cif cifs/CeO2.cif --name CeO2

# 3. Run Stage-Gate refinement sequence
uv run python scripts/xrd_sample.py --dir ./my_sample step --sequence "1,2,3,5,7,12" --cycles 5

# 4. Inspect status, diagnose residuals, and export results (CIFs, CSVs, 600 DPI figure, IUCr report)
uv run python scripts/xrd_sample.py --dir ./my_sample status
uv run python scripts/xrd_sample.py --dir ./my_sample diagnose --out
uv run python scripts/xrd_sample.py --dir ./my_sample export
```

## License & Citation

Released under the [BSD 3-Clause License](LICENSE) ([Russian translation](LICENSE.ru)). If you use `xrd-analysis` in academic work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff) alongside the underlying crystallographic software and open databases:

- **xrd-analysis:** Yurev, I. O. (2026). *xrd-analysis: Autonomous Crystallographic Analysis and Stage-Gate Rietveld/Pawley Refinement Skill* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23892876
- **GSAS-II:** Toby, B. H. & Von Dreele, R. B. (2013). *GSAS-II: the genesis of a modern open-source all purpose crystallography software package*. Journal of Applied Crystallography, 46, 544–549. https://doi.org/10.1107/S0021889813003531
- **Crystallography Open Database (COD):**
  - Gražulis, S., Daškevič, A., Merkys, A., Chateigner, D., Lutterotti, L., Quirós, M., Serebryanaya, N. R., Moeck, P., Downs, R. T. & Le Bail, A. (2012). *Crystallography Open Database (COD): an open-access collection of crystal structures and platform for world-wide collaboration*. Nucleic Acids Research, 40, D420–D427. https://doi.org/10.1093/nar/gkr900
  - Gražulis, S., Chateigner, D., Downs, R. T., Yokochi, A. F. T., Quirós, M., Lutterotti, L., Manakova, E., Butkus, J., Moeck, P. & Le Bail, A. (2009). *Crystallography Open Database — an open-access collection of crystal structures*. Journal of Applied Crystallography, 42, 726–729. https://doi.org/10.1107/S0021889809016690
- **AMCSD (when using mineral subsets in COD):** Downs, R. T. & Hall-Wallace, M. (2003). *The American Mineralogist Crystal Structure Database*. American Mineralogist, 88, 247–250.

## Trademark Disclaimer

PDF-2 and ICDD are registered trademarks of the International Centre for Diffraction Data. This software is an independent interoperability tool, bundles no proprietary database files, and is not endorsed by, sponsored by, or affiliated with ICDD.
