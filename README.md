# Flow Matching for Single-Cell Dynamic Biology

Teaching code for a paper-facing tutorial on flow matching for time-resolved single-cell snapshots.

The repository is organized around runnable notebooks. Generated figures, tables,
and output summaries are not tracked by Git; rerun the notebooks to regenerate them locally.

## Layout

    notebooks/   tutorial notebooks, ordered by chapter
    scripts/     runners and notebook builders
    src/         shared modules imported by notebooks and scripts
    configs/     YAML configs for reusable examples
    data/        small reusable datasets, organized by dataset

Running the notebooks regenerates `figures/`, `tables/`, and `outputs/` locally;
these directories hold reproducible artifacts rather than source files.

## Data

Reusable data stays under `data/`, organized by dataset. The tutorial draws on
the EB time-course, sci-Plex A549, LINCS compound metadata, and toy
branching-snapshot assets, accessed through the shared `src/` loaders. The small
data files required by the notebooks are tracked; large raw downloads and local
caches are not.

## Design Rules

- Keep training loops explicit and readable.
- Avoid production abstractions, experiment managers, and deep inheritance.
- Keep generated artifacts in `figures/`, `tables/`, and `outputs/`; these
  directories are reproducible outputs rather than source files.

## Environment

Create the project environment and register the notebook kernel:

```bash
conda env create -f environment.yml
conda activate fmdb
python -m ipykernel install --user --name fmdb --display-name fmdb
```

The environment name (`fmdb`) and GPU runtime are defined in `environment.yml`.

## Running

Open the notebooks in chapter order in Jupyter and run all cells.