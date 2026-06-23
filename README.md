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

## Quick Smoke Test

    python -m pytest -q

The base conda environment used for the current v2 artifacts is
`/home/xmabs/anaconda3/bin/python`.

## How to reproduce paper figures

Steps in order:
1. `conda env create -f environment.yml && conda activate flow_matching_db`
2. `python -m ipykernel install --user --name flow_matching_db --display-name flow_matching_db`
3. `python -m pytest -q` (helper smoke tests)
4. Run notebooks in dependency order (see notebooks/INDEX.md):
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
5. Generated figures/tables land in `figures/`, `tables/`, `outputs/`.

Notes:
- `QUICK_MODE=1` (default) is the fast smoke run; full mode (`QUICK_MODE=0`) produces paper-grade figures.
- `SMOKE_MODE=1` is even smaller and intended for CI only.
- Each notebook is independently runnable as long as its upstream chapter artifacts/caches exist (see notebooks/INDEX.md).
