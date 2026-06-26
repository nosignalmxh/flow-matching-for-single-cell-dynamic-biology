# Flow Matching for Single-Cell Dynamic Biology

Teaching code for the paper **Flow Matching for Single-Cell Dynamic Biology**.

## Repository Structure

```text
notebooks/
  INDEX.md
  chapter2_distribution_transport.ipynb
  chapter3_1_flow_matching_from_scratch.ipynb
  chapter3_2_eb_flow_matching.ipynb
  chapter3_3_eb_ablations.ipynb
  chapter4_1_coupling_geometry.ipynb
  chapter4_2_state_space_assumptions.ipynb
  chapter4_3_sampling_depth.ipynb
  chapter5_1_timecourse_suite.ipynb
  chapter5_2_perturbation_sciplex.ipynb

src/
  core/           flow-matching losses, models, sampling, and OT helpers
  data/           dataset loaders and preprocessing utilities
  evaluation/     metrics and representation readouts
  experiments/    reusable experiment routines used by notebooks
  visualization/  plotting helpers for tutorial and paper figures

configs/          YAML configs for reusable examples
data/             small tracked tutorial datasets and metadata
scripts/          command-line helpers for selected chapter outputs
tests/            structural and helper tests for notebooks and src modules
archive/          retired notebooks and historical build artifacts
```

Running the notebooks regenerates local `figures/`, `tables/`, and `outputs/` directories. These directories contain reproducible artifacts and caches rather than source files.

## Notebooks

- Start with `notebooks/chapter2_distribution_transport.ipynb` for distribution transport, couplings, path construction, and solver-in-loop diagnostics.
- Follow `chapter3_1` to `chapter3_3` for flow matching from scratch, EB flow matching, and EB ablation studies.
- Follow `chapter4_1` to `chapter4_3` for coupling geometry, state-space assumptions, PC-versus-PHATE diagnostics, manifold-aware paths, and sampling-depth boundaries.
- Follow `chapter5_1` and `chapter5_2` for time-course and perturbation-response examples.

For the full dependency order and expected outputs, see `notebooks/INDEX.md`.

## Environment

Create the conda environment and register the notebook kernel:

```bash
conda env create -f environment.yml
conda activate fmdb
python -m ipykernel install --user --name fmdb --display-name fmdb
```

The environment uses Python 3.10, PyTorch with CUDA 12.4, and the scientific Python packages listed in `environment.yml`.

## Data

Small tutorial assets are kept under `data/`, including toy branching snapshots, EB time-course data, sci-Plex A549 inputs, and LINCS compound metadata. Large raw downloads, local caches, and regenerated artifacts are intentionally kept out of Git.

## Reproducing Outputs

Open the notebooks in Jupyter and run them in chapter order:

```text
chapter2
  -> chapter3_1 -> chapter3_2 -> chapter3_3
                 -> chapter4_1 -> chapter4_2 -> chapter4_3
                            -> chapter5_1 -> chapter5_2
```

Most notebooks can be run interactively. Chapter 4 and Chapter 5 notebooks may take longer in full mode because they train models, compute optimal-transport couplings, or regenerate cached diagnostics.

## Development Checks

Run the focused tests from the repository root:

```bash
pytest tests
```

The tests check notebook structure, helper contracts, plotting/display assumptions, and selected reusable modules.
