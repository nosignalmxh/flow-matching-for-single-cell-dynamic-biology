# Flow Matching for Dynamic Biology

Teaching code for a paper-facing tutorial on flow matching for time-resolved single-cell snapshots.

This tree now uses the v2 paper workflow at the repository root. The migrated
notebooks and generated artifacts have replaced the older notebook/result set.

## Layout

    notebooks/   v2 paper-facing notebooks
    scripts/     v2 runners and notebook builders
    src/         shared modules imported by notebooks and scripts
    figures/     generated paper figures
    outputs/     generated run summaries, executed notebooks, and caches
    tables/      generated paper tables
    configs/     YAML configs retained for reusable examples
    data/        reusable datasets, organized by dataset rather than chapter
    tests/       lightweight migration and artifact sanity tests

## Data

Reusable data stays under `data/`. The current v2 workflow uses the EB
time-course assets, sci-Plex A549 assets, LINCS compound metadata, and selected
toy assets through the shared `src/` loaders.

## Design Rules

- Keep training loops explicit and readable.
- Avoid production abstractions, experiment managers, and deep inheritance.
- Keep paper claims tied to generated artifacts in `figures/`, `tables/`, and
  `outputs/`.

## Environment

Create the project GPU environment from the checked-in conda file:

```bash
conda env create -f environment.yml
conda activate fmdb
python -m ipykernel install --user --name fmdb --display-name fmdb
```

The environment name is defined in `environment.yml` as `fmdb`. The file pins
PyTorch with the CUDA 12.4 runtime (`pytorch-cuda=12.4`), which is intended for
running the notebook validations on NVIDIA GPUs with a compatible host driver.
After activation, a quick CUDA sanity check is:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Quick Smoke Test

```bash
python -m pytest -q
```

## How to reproduce paper figures

Steps in order:
1. `conda env create -f environment.yml` (creates the GPU-enabled `fmdb` env)
2. `conda activate fmdb`
3. `python -m ipykernel install --user --name fmdb --display-name fmdb`
4. `python -m pytest -q` (helper smoke tests)
5. Run notebooks in dependency order (see notebooks/INDEX.md):
   ```bash
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter2_distribution_transport.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter3_1_flow_matching_from_scratch.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter3_2_eb_flow_matching.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter3_3_eb_ablations.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter4_1_coupling_geometry.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter4_2_state_space_assumptions.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter4_3_sampling_depth.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter5_1_timecourse_suite.ipynb
   QUICK_MODE=0 jupyter nbconvert --to notebook --execute notebooks/chapter5_2_perturbation_sciplex.ipynb
   ```
6. Generated figures/tables land in `figures/`, `tables/`, `outputs/`.

Notes:
- `QUICK_MODE=1` (default) is the fast smoke run; full mode (`QUICK_MODE=0`) produces paper-grade figures.
- `SMOKE_MODE=1` is even smaller and intended for CI only.
- Each notebook is independently runnable as long as its upstream chapter artifacts/caches exist (see notebooks/INDEX.md).
