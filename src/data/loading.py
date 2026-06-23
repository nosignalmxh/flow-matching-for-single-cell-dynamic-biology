from __future__ import annotations

from dataclasses import dataclass
import gzip
import warnings
from pathlib import Path
from typing import Callable
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .samplers import SnapshotDataset
from .toy import make_branching_snapshot_toy, make_y_branching_snapshots
from ..utils import ensure_dir
from ..artifacts import sample_rows, save_json
from ..evaluation.representations import (
    fit_pca_state_space,
    program_index_dict,
    readout_program_scores_from_matrix,
    standardize_train_space,
)


CH02_TIMEPOINTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CH01_REQUIRED_COLUMNS = ["cell_id", "state_1", "state_2", "time", "condition", "fate_label"]
CH02_OBS_COLUMNS = CH01_REQUIRED_COLUMNS + ["batch", "donor", "replicate"]
TOY_DATA_DIR = Path("data/toy_branching_snapshots")
OBSERVED_2D_SNAPSHOTS = TOY_DATA_DIR / "observed_2d_snapshots.csv"
CONDITIONED_SNAPSHOT_TABLE = TOY_DATA_DIR / "conditioned_snapshot_table.csv"
PSEUDOCOUNT_ADATA = TOY_DATA_DIR / "branching_toy_pseudocounts.h5ad"
TRAJECTORYNET_EB_SOURCE = Path("../baselines/trajectorynet/data/eb_velocity_v5.npz")
TRAJECTORYNET_EB_COPIED = Path("data/trajectorynet_eb/eb_velocity_v5.npz")
SCIPLEX3_A549_DIR = Path("data/sciplex3_a549")
CHEMCPA_LINCS_SMILES_DIR = Path("data/chemcpa_lincs_smiles")
CELLOT_PROCESSED_DATASETS_URL = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams/"
    "7c5fe615-a6fa-4464-8ae7-4482a02040db/content"
)
LINCS_PERT_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo.txt.gz"
)
SCPERT_SCIPLEX3_URL = "https://zenodo.org/api/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad/content"


@dataclass
class SciplexPreparedData:
    adata: object
    metadata: pd.DataFrame
    cell_counts: pd.DataFrame
    paths: dict[str, str]
    summary: dict


@dataclass
class SciplexPCAStateTable:
    X_pca: np.ndarray
    metadata: pd.DataFrame
    hvg_genes: list[str]
    pca_explained_variance_ratio: list[float]
    train_mean: np.ndarray
    train_std: np.ndarray


@dataclass
class LincsSmilesCorpus:
    smiles: list[str]
    frame: pd.DataFrame
    path: Path
    n_invalid: int


@dataclass
class RDKit2DResult:
    features: np.ndarray
    raw_features: np.ndarray
    external_mean: np.ndarray
    external_std: np.ndarray
    feature_names: list[str]
    diagnostics: dict


def _readable_shape(shape) -> str:
    return "(" + ", ".join(str(int(x)) for x in shape) + ")"


def _require_anndata():
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "anndata is required for Chapter 2 AnnData assets. Install the project environment first."
        ) from exc
    return ad


def _validate_snapshot_frame(frame: pd.DataFrame, required_columns=CH01_REQUIRED_COLUMNS) -> None:
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"snapshot table is missing required columns: {missing}")


def load_ch01_snapshot_table(
    path: str | Path = OBSERVED_2D_SNAPSHOTS,
) -> pd.DataFrame:
    """Load the Chapter 1 observed snapshot table, with a toy fallback."""
    path = Path(path)
    if not path.exists() and path == OBSERVED_2D_SNAPSHOTS:
        legacy_path = Path("data/ch01/ch01_branching_snapshot_toy.csv")
        if legacy_path.exists():
            path = legacy_path
    if path.exists():
        frame = pd.read_csv(path)
    else:
        warnings.warn(
            f"{path} was not found; generating a deterministic Chapter 1 toy fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = make_branching_snapshot_toy(n_cells=4000, timepoints=CH02_TIMEPOINTS, seed=42).to_frame()

    _validate_snapshot_frame(frame)
    frame = frame[CH01_REQUIRED_COLUMNS].copy()
    frame["time"] = frame["time"].astype(float)
    return frame


def _subsample_balanced_by_time(frame: pd.DataFrame, n_cells: int | None, seed: int) -> pd.DataFrame:
    if n_cells is None or len(frame) <= n_cells:
        return frame.copy()

    times = sorted(frame["time"].unique())
    base = n_cells // len(times)
    remainder = n_cells % len(times)
    pieces = []
    for i, timepoint in enumerate(times):
        subset = frame[frame["time"] == timepoint]
        target = min(len(subset), base + (1 if i < remainder else 0))
        pieces.append(subset.sample(n=target, random_state=seed + i, replace=False))
    return pd.concat(pieces, axis=0).sort_values(["time", "cell_id"]).reset_index(drop=True)


def _rewrite_cell_ids(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    frame["cell_id"] = [f"{prefix}_cell_{i:06d}" for i in range(len(frame))]
    frame["condition"] = prefix
    return frame


def _add_experimental_metadata(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add batch, donor, and replicate without making batch identical to time."""
    rng = np.random.default_rng(seed)
    frame = frame.copy().reset_index(drop=True)
    times = sorted(frame["time"].unique())
    batches = np.asarray(["batch_A", "batch_B", "batch_C"], dtype=object)
    donors = np.asarray(["donor_1", "donor_2", "donor_3", "donor_4"], dtype=object)

    batch_values = []
    donor_values = []
    replicate_values = []
    for _, row in frame.iterrows():
        time_idx = times.index(row["time"])
        condition_shift = 1 if row["condition"] == "perturbed" else 0
        weights = np.array([0.50, 0.30, 0.20], dtype=float)
        weights = np.roll(weights, (time_idx + condition_shift) % len(weights))
        weights = 0.82 * weights + 0.18 / len(weights)
        weights = weights / weights.sum()
        batch_values.append(rng.choice(batches, p=weights))

        donor_weights = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)
        donor_weights = np.roll(donor_weights, (time_idx // 2 + condition_shift) % len(donor_weights))
        donor_weights = donor_weights / donor_weights.sum()
        donor_values.append(rng.choice(donors, p=donor_weights))
        replicate_values.append(f"replicate_{1 + ((time_idx + condition_shift + rng.integers(0, 2)) % 3)}")

    frame["batch"] = np.asarray(batch_values, dtype=object)
    frame["donor"] = np.asarray(donor_values, dtype=object)
    frame["replicate"] = np.asarray(replicate_values, dtype=object)
    return frame


def make_ch02_conditioned_snapshot_table(
    control_path: str | Path = OBSERVED_2D_SNAPSHOTS,
    n_cells_per_condition: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create the Chapter 2 two-condition snapshot table from Chapter 1 geometry."""
    control = load_ch01_snapshot_table(control_path)
    control = _subsample_balanced_by_time(control, n_cells_per_condition, seed=seed)
    control = _rewrite_cell_ids(control, "control")

    n_perturbed = len(control) if n_cells_per_condition is None else n_cells_per_condition
    perturbed = make_branching_snapshot_toy(
        n_cells=n_perturbed,
        timepoints=CH02_TIMEPOINTS,
        condition_effect="more_rare_fate",
        seed=seed + 1,
    ).to_frame()
    perturbed = perturbed[CH01_REQUIRED_COLUMNS].copy()
    perturbed = _rewrite_cell_ids(perturbed, "perturbed")

    frame = pd.concat([control, perturbed], axis=0, ignore_index=True)
    frame["time"] = frame["time"].astype(float)
    frame = _add_experimental_metadata(frame, seed=seed)
    return frame[CH02_OBS_COLUMNS].copy()


def _gene_programs(n_genes: int) -> np.ndarray:
    programs = np.full(n_genes, "background", dtype=object)
    blocks = [
        (0, 20, "progenitor_trunk"),
        (20, 40, "transition"),
        (40, 60, "major_fate"),
        (60, 80, "rare_fate"),
        (80, n_genes, "background"),
    ]
    for start, stop, name in blocks:
        if start < n_genes:
            programs[start : min(stop, n_genes)] = name
    return programs


def make_pseudocount_anndata(
    frame: pd.DataFrame,
    n_genes: int = 120,
    seed: int = 42,
):
    """Generate a deterministic count-like AnnData object tied to toy snapshot coordinates."""
    if n_genes < 20:
        raise ValueError("n_genes must be at least 20 for Chapter 2 pseudo-gene programs")
    _validate_snapshot_frame(frame)
    ad = _require_anndata()

    frame = frame.copy().reset_index(drop=True)
    if not {"batch", "donor", "replicate"}.issubset(frame.columns):
        frame = _add_experimental_metadata(frame, seed=seed)

    rng = np.random.default_rng(seed)
    n_cells = len(frame)
    state_1 = frame["state_1"].to_numpy(dtype=float)
    state_2 = frame["state_2"].to_numpy(dtype=float)
    time = frame["time"].to_numpy(dtype=float)
    fate = frame["fate_label"].to_numpy(dtype=object)
    condition = frame["condition"].to_numpy(dtype=object)
    programs = _gene_programs(n_genes)

    trunk = np.clip(1.05 - time, 0.0, 1.0)
    transition = np.exp(-((time - 0.50) ** 2) / (2 * 0.16**2))
    major = ((fate == "major").astype(float) * (0.35 + time)) + np.clip(state_2, 0, None)
    rare = ((fate == "rare").astype(float) * (0.40 + time)) + np.clip(-state_2, 0, None)
    perturbed = (condition == "perturbed").astype(float)
    rare = rare * (1.0 + 0.50 * perturbed)
    major = major * (1.0 - 0.10 * perturbed)
    background_signal = 0.18 + 0.04 * np.sin(2.5 * state_1)

    program_signal = {
        "progenitor_trunk": trunk,
        "transition": transition,
        "major_fate": major,
        "rare_fate": rare,
        "background": background_signal,
    }

    gene_baseline = rng.gamma(shape=1.4, scale=0.55, size=n_genes) + 0.08
    gene_loading = rng.uniform(0.75, 1.35, size=n_genes)
    mean = np.empty((n_cells, n_genes), dtype=float)
    for gene_idx, program in enumerate(programs):
        signal = program_signal[str(program)]
        local_noise = 0.04 * np.cos((gene_idx + 1) * state_1) + 0.03 * np.sin((gene_idx + 3) * state_2)
        mean[:, gene_idx] = gene_baseline[gene_idx] + gene_loading[gene_idx] * np.clip(signal + local_noise, 0, None)

    library_size = rng.lognormal(mean=np.log(1.0), sigma=0.32, size=n_cells)
    donor_boost = frame["donor"].astype(str).map({"donor_1": 1.06, "donor_2": 0.98, "donor_3": 1.02, "donor_4": 0.94})
    donor_boost = donor_boost.fillna(1.0).to_numpy(dtype=float)
    mean = mean * library_size[:, None] * donor_boost[:, None] * 1.85
    mean = np.clip(mean, 0.01, None)
    counts = rng.poisson(mean).astype(np.int32)

    obs = frame[["cell_id", "time", "condition", "fate_label", "batch", "donor", "replicate"]].copy()
    obs.index = obs["cell_id"].astype(str).to_numpy()
    var = pd.DataFrame(
        {
            "gene_id": [f"G{i:03d}" for i in range(n_genes)],
            "program": programs.astype(str),
        }
    )
    var.index = var["gene_id"].to_numpy()

    adata = ad.AnnData(X=counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    adata.obsm["X_toy_state"] = frame[["state_1", "state_2"]].to_numpy(dtype=np.float32)
    adata.uns["data_note"] = (
        "Toy-derived pseudo-count demo for Chapter 2. Counts are simulated from the "
        "Chapter 1 branching snapshot geometry and are not real biological measurements."
    )
    return adata


def metadata_summary_table(adata_or_frame) -> pd.DataFrame:
    if hasattr(adata_or_frame, "obs"):
        frame = adata_or_frame.obs
    else:
        frame = pd.DataFrame(adata_or_frame)

    rows = []
    roles = {
        "cell_id": ("obs", "sample identifier", False),
        "time": ("obs", "snapshot index", True),
        "condition": ("obs", "condition-indexed snapshot", True),
        "fate_label": ("obs", "diagnostic downstream annotation", False),
        "batch": ("obs", "experimental design metadata", False),
        "donor": ("obs", "experimental design metadata", False),
        "replicate": ("obs", "experimental design metadata", False),
    }
    for field, (location, role, defines_snapshot) in roles.items():
        if field not in frame.columns:
            continue
        series = frame[field]
        rows.append(
            {
                "field": field,
                "location": location,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique()),
                "role": role,
                "defines_snapshot": bool(defines_snapshot),
            }
        )
    return pd.DataFrame(rows)


def time_condition_counts_table(frame_or_adata) -> pd.DataFrame:
    if hasattr(frame_or_adata, "obs"):
        frame = frame_or_adata.obs.copy()
    else:
        frame = pd.DataFrame(frame_or_adata).copy()
    if "fate_label" not in frame.columns:
        frame["fate_label"] = "unlabeled"
    grouped = (
        frame.groupby(["time", "condition"], observed=False)["fate_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["major", "rare"]:
        if label not in grouped.columns:
            grouped[label] = 0
    grouped["n_cells"] = grouped.drop(columns=["time", "condition"]).sum(axis=1)
    grouped["n_major"] = grouped["major"].astype(int)
    grouped["n_rare"] = grouped["rare"].astype(int)
    grouped["rare_fraction"] = grouped["n_rare"] / grouped["n_cells"].clip(lower=1)
    return grouped[["time", "condition", "n_cells", "n_major", "n_rare", "rare_fraction"]].sort_values(
        ["condition", "time"]
    )


def save_ch02_demo_data(
    data_dir: str | Path = TOY_DATA_DIR,
    output_dir: str | Path = "outputs/ch02",
    quick_mode: bool = True,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate and save the Chapter 2 toy-first AnnData assets."""
    data_dir = ensure_dir(data_dir)
    output_dir = ensure_dir(output_dir)
    n_cells_per_condition = 2000 if quick_mode else 4000
    frame = make_ch02_conditioned_snapshot_table(
        n_cells_per_condition=n_cells_per_condition,
        seed=seed,
    )
    adata = make_pseudocount_anndata(frame, n_genes=120, seed=seed)

    snapshot_data_path = data_dir / "conditioned_snapshot_table.csv"
    snapshot_output_path = output_dir / "ch02_snapshot_table.csv"
    adata_path = data_dir / "branching_toy_pseudocounts.h5ad"
    metadata_path = output_dir / "table02_01_metadata_summary.csv"
    counts_path = output_dir / "table02_03_time_condition_counts.csv"

    frame.to_csv(snapshot_data_path, index=False)
    frame.to_csv(snapshot_output_path, index=False)
    adata.write_h5ad(adata_path)
    metadata_summary_table(adata).to_csv(metadata_path, index=False)
    time_condition_counts_table(adata).to_csv(counts_path, index=False)

    return {
        "snapshot_csv": snapshot_data_path,
        "snapshot_output_csv": snapshot_output_path,
        "adata_h5ad": adata_path,
        "metadata_summary_csv": metadata_path,
        "time_condition_counts_csv": counts_path,
    }


def load_demo_timecourse(
    path: str | Path = PSEUDOCOUNT_ADATA,
    fallback_to_generate: bool = True,
    quick_mode: bool = True,
):
    path = Path(path)
    ad = _require_anndata()
    if not path.exists() and path == PSEUDOCOUNT_ADATA:
        legacy_path = Path("data/ch02/ch02_branching_toy_count_adata.h5ad")
        if legacy_path.exists():
            path = legacy_path
    if path.exists():
        return ad.read_h5ad(path)
    if fallback_to_generate:
        paths = save_ch02_demo_data(data_dir=path.parent, quick_mode=quick_mode)
        return ad.read_h5ad(paths["adata_h5ad"])
    raise FileNotFoundError(path)


def adata_to_snapshot_dataset(
    adata,
    representation_key: str = "X_pca",
    time_key: str = "time",
    condition_key: str | None = "condition",
    label_key: str | None = "fate_label",
) -> SnapshotDataset:
    if representation_key == "X":
        X = np.asarray(adata.X)
    elif representation_key in adata.obsm:
        X = np.asarray(adata.obsm[representation_key])
    else:
        raise KeyError(
            f"representation_key={representation_key!r} was not found. Use 'X' or one of {list(adata.obsm.keys())}."
        )

    obs = adata.obs
    time = obs[time_key].to_numpy()
    condition = obs[condition_key].to_numpy() if condition_key is not None and condition_key in obs else None
    labels = obs[label_key].to_numpy() if label_key is not None and label_key in obs else None
    cell_id = obs["cell_id"].to_numpy() if "cell_id" in obs else obs.index.to_numpy()
    split = obs["split"].to_numpy() if "split" in obs else None
    return SnapshotDataset(
        X=np.asarray(X, dtype=np.float32),
        time=time,
        condition=condition,
        labels=labels,
        cell_id=cell_id,
        split=split,
    )


def load_trajectorynet_eb_npz(
    path: str | Path = "../baselines/trajectorynet/data/eb_velocity_v5.npz",
    embedding: str = "pcs",
    max_dim: int = 20,
    max_cells: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, SnapshotDataset]:
    """Optionally load the TrajectoryNet EB npz as a real-like embedding demo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    if embedding not in data:
        raise KeyError(f"{embedding!r} not found in {path}; available keys: {sorted(data.files)}")
    X = np.asarray(data[embedding], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{embedding!r} must be a 2D array")
    X = X[:, : min(max_dim, X.shape[1])]
    labels = np.asarray(data["sample_labels"] if "sample_labels" in data else np.zeros(X.shape[0]), dtype=object)

    rng = np.random.default_rng(seed)
    if max_cells is not None and X.shape[0] > max_cells:
        idx = np.sort(rng.choice(X.shape[0], size=max_cells, replace=False))
        X = X[idx]
        labels = labels[idx]
    time = labels.astype(str)
    cell_id = np.asarray([f"trajectorynet_eb_cell_{i:06d}" for i in range(X.shape[0])], dtype=object)
    dataset = SnapshotDataset(X=X, time=time, condition=np.full(X.shape[0], "eb_npz"), cell_id=cell_id)

    rows = []
    for t, n in pd.Series(time).value_counts().sort_index().items():
        rows.append(
            {
                "source": str(path),
                "embedding": embedding,
                "n_cells": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "time": t,
                "n_cells_at_time": int(n),
            }
        )
    return pd.DataFrame(rows), dataset


def copy_trajectorynet_eb_to_data(
    source_path: str | Path = TRAJECTORYNET_EB_SOURCE,
    target_path: str | Path = TRAJECTORYNET_EB_COPIED,
    overwrite: bool = False,
) -> Path:
    """Copy the EB npz from baselines into the tutorial data directory."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    if target_path.exists() and not overwrite:
        return target_path
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    ensure_dir(target_path.parent)
    shutil.copy2(source_path, target_path)
    return target_path


def load_eb_timecourse_for_ch03(
    path: str | Path = TRAJECTORYNET_EB_COPIED,
    cost_embedding: str = "pcs",
    plot_embedding: str = "phate",
    n_cost_dims: int = 20,
    max_cells_per_time: int = 900,
    seed: int = 42,
) -> dict:
    """Load EB as the Chapter 3 main dataset.

    The loader intentionally ignores velocity-like arrays such as
    ``pcs_delta`` and ``delta_embedding``. Chapter 3 treats the data as
    unpaired, time-indexed snapshot embeddings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy it first from "
            "../baselines/trajectorynet/data/eb_velocity_v5.npz to data/trajectorynet_eb/."
        )
    data = np.load(path, allow_pickle=True)
    missing = [key for key in [cost_embedding, plot_embedding, "sample_labels"] if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required EB keys: {missing}; available keys: {sorted(data.files)}")

    X_cost_full = np.asarray(data[cost_embedding], dtype=np.float32)
    X_plot_full = np.asarray(data[plot_embedding], dtype=np.float32)
    time_full = np.asarray(data["sample_labels"])
    if X_cost_full.ndim != 2:
        raise ValueError(f"{cost_embedding!r} must be a 2D array")
    if X_plot_full.ndim != 2 or X_plot_full.shape[1] < 2:
        raise ValueError(f"{plot_embedding!r} must be a 2D array with at least two columns")
    if X_cost_full.shape[0] != X_plot_full.shape[0] or X_cost_full.shape[0] != time_full.shape[0]:
        raise ValueError("pcs, phate, and sample_labels must have matching cell counts")

    n_cost_dims = min(int(n_cost_dims), X_cost_full.shape[1])
    X_cost_full = X_cost_full[:, :n_cost_dims]
    X_plot_full = X_plot_full[:, :2]

    full_counts = (
        pd.Series(time_full.astype(str), name="time")
        .value_counts()
        .sort_index()
        .rename_axis("time")
        .reset_index(name="n_cells")
    )
    full_counts["source"] = str(path)
    full_counts["cost_embedding"] = cost_embedding
    full_counts["plot_embedding"] = plot_embedding
    full_counts["n_cost_dims"] = int(n_cost_dims)
    full_counts = full_counts[["source", "cost_embedding", "plot_embedding", "n_cost_dims", "time", "n_cells"]]

    rng = np.random.default_rng(seed)
    selected = []
    for timepoint in sorted(np.unique(time_full.astype(str)).tolist()):
        idx = np.flatnonzero(time_full.astype(str) == timepoint)
        if max_cells_per_time is not None and len(idx) > max_cells_per_time:
            idx = np.sort(rng.choice(idx, size=int(max_cells_per_time), replace=False))
        selected.append(idx)
    selected_idx = np.concatenate(selected) if selected else np.array([], dtype=int)
    selected_idx = np.sort(selected_idx)

    time_sub = time_full[selected_idx].astype(str)
    cell_ids = np.asarray([f"trajectorynet_eb_cell_{i:06d}" for i in selected_idx], dtype=object)
    return {
        "X_cost": X_cost_full[selected_idx],
        "X_plot": X_plot_full[selected_idx],
        "time": time_sub,
        "cell_id": cell_ids,
        "source_path": str(path),
        "full_counts_by_time": full_counts,
        "cost_embedding": cost_embedding,
        "plot_embedding": plot_embedding,
        "n_cost_dims": int(n_cost_dims),
        "selected_indices": selected_idx,
    }


def load_eb_pair_data_for_ch04(
    path: str | Path = TRAJECTORYNET_EB_COPIED,
    source_time: str = "1",
    target_time: str = "2",
    n_cost_dims: int = 20,
    max_cells_per_time: int = 900,
    train_state: str = "phate",
    standardize_state: bool = True,
    seed: int = 42,
) -> dict:
    """Load EB source/target snapshots for Chapter 4.

    The returned state coordinates are used by the tutorial velocity model.
    Cost coordinates remain PCA-based so the chosen coupling can reuse the
    Chapter 3 data logic.
    """
    train_state = str(train_state).lower()
    if train_state not in {"phate", "plot"}:
        raise ValueError("train_state must be 'phate' for the Chapter 4 2D tutorial")

    eb = load_eb_timecourse_for_ch03(
        path=path,
        cost_embedding="pcs",
        plot_embedding="phate",
        n_cost_dims=n_cost_dims,
        max_cells_per_time=max_cells_per_time,
        seed=seed,
    )
    time = eb["time"].astype(str)
    available = sorted(pd.Series(time).unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    source_time = str(source_time)
    target_time = str(target_time)
    missing = [label for label in [source_time, target_time] if label not in set(available)]
    if missing:
        raise ValueError(
            "Requested EB time label(s) not found: "
            f"{missing}. Available labels are {available}."
        )

    idx0 = np.flatnonzero(time == source_time)
    idx1 = np.flatnonzero(time == target_time)
    X0_plot = np.asarray(eb["X_plot"][idx0], dtype=np.float32)
    X1_plot = np.asarray(eb["X_plot"][idx1], dtype=np.float32)
    X0_cost = np.asarray(eb["X_cost"][idx0], dtype=np.float32)
    X1_cost = np.asarray(eb["X_cost"][idx1], dtype=np.float32)

    state_mean = np.zeros(2, dtype=np.float32)
    state_std = np.ones(2, dtype=np.float32)
    if standardize_state:
        combined = np.vstack([X0_plot, X1_plot]).astype(np.float32)
        state_mean = combined.mean(axis=0).astype(np.float32)
        state_std = combined.std(axis=0).astype(np.float32)
        state_std = np.where(state_std < 1e-6, 1.0, state_std).astype(np.float32)
        X0_state = (X0_plot - state_mean) / state_std
        X1_state = (X1_plot - state_mean) / state_std
    else:
        X0_state = X0_plot.copy()
        X1_state = X1_plot.copy()

    return {
        "X0_state": np.asarray(X0_state, dtype=np.float32),
        "X1_state": np.asarray(X1_state, dtype=np.float32),
        "X0_cost": X0_cost,
        "X1_cost": X1_cost,
        "X0_plot": X0_plot,
        "X1_plot": X1_plot,
        "source_time": source_time,
        "target_time": target_time,
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "full_counts_by_time": eb["full_counts_by_time"],
        "source_path": eb["source_path"],
        "selected_indices_source": eb["selected_indices"][idx0],
        "selected_indices_target": eb["selected_indices"][idx1],
        "cost_embedding": eb["cost_embedding"],
        "plot_embedding": eb["plot_embedding"],
        "train_state": train_state,
        "standardize_state": bool(standardize_state),
    }


def load_demo_h5ad(path: str | Path = "data/demo_timecourse.h5ad", fallback_to_toy: bool = True):
    """Backward-compatible Chapter 1 helper retained for older notebooks."""
    path = Path(path)
    if path.exists():
        try:
            ad = _require_anndata()
            return ad.read_h5ad(path)
        except Exception as exc:
            if not fallback_to_toy:
                raise exc
    if fallback_to_toy:
        return make_y_branching_snapshots(n_cells=2000, seed=42)
    raise FileNotFoundError(path)


def _download_file(url: str, path: str | Path, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    ensure_dir(path.parent)
    with urllib.request.urlopen(url, timeout=60) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return path


def _dense_array(x) -> np.ndarray:
    try:
        from scipy import sparse

        if sparse.issparse(x):
            return np.asarray(x.toarray())
    except Exception:
        pass
    if hasattr(x, "A"):
        return np.asarray(x.A)
    return np.asarray(x)


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _slug_cell_id(value: object, fallback: int) -> str:
    text = str(value) if value is not None else ""
    text = text.strip()
    if not text:
        text = f"cell_{fallback:06d}"
    out = []
    for char in text:
        out.append(char if char.isalnum() or char in {"_", "-", "."} else "_")
    return "".join(out)


def _parse_compound_dose_name(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    compounds = []
    doses = []
    for raw in values.astype(str):
        text = raw.strip()
        parts = [part for part in text.replace(":", "_").replace("|", "_").split("_") if part != ""]
        dose = np.nan
        compound_parts = parts[:]
        for i in range(len(parts) - 1, -1, -1):
            try:
                dose = float(parts[i])
                compound_parts = parts[:i] + parts[i + 1 :]
                break
            except ValueError:
                continue
        compounds.append("_".join(compound_parts) if compound_parts else text)
        doses.append(dose)
    return pd.Series(compounds, index=values.index), pd.Series(doses, index=values.index, dtype=float)


def _normalized_drug_key(value: object) -> str:
    import re

    text = str(value).strip().lower()
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_sciplex_obs(obs: pd.DataFrame, smiles_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Normalize common sci-Plex/A549 obs schemas without fabricating drug-dose metadata."""
    frame = obs.copy()
    compound_col = _first_existing_column(
        frame,
        [
            "compound",
            "drug",
            "perturbation_name",
            "condition",
            "cov_drug",
            "cov_drug_dose_name",
            "product_name",
            "perturbation",
            "cov_drug_name",
        ],
    )
    dose_col = _first_existing_column(
        frame,
        ["dose", "dose_val", "dose_value", "perturbation_value", "drug_dose", "pert_dose", "dose_um", "dosage"],
    )
    cell_line_col = _first_existing_column(frame, ["cell_line", "cell_type", "celltype", "cell", "split_cell"])
    smiles_col = _first_existing_column(frame, ["SMILES", "smiles", "canonical_smiles"])

    if compound_col is None:
        raise ValueError(
            "Could not infer a compound column from sci-Plex obs. "
            "Expected one of compound/drug/perturbation_name/condition/cov_drug/cov_drug_dose_name."
        )
    parsed_dose = None
    if str(compound_col).lower() == "cov_drug_dose_name":
        parsed_compound, parsed_dose = _parse_compound_dose_name(frame[compound_col])
        frame["compound"] = parsed_compound.astype(str)
    else:
        frame["compound"] = frame[compound_col].astype(str)
    if dose_col is None:
        if parsed_dose is not None and parsed_dose.notna().any():
            frame["dose"] = parsed_dose.fillna(0.0).astype(float)
        else:
            raise ValueError(
                "sci-Plex obs has no explicit dose column and dose could not be parsed from cov_drug_dose_name; "
                "no treated drug-dose compounds can be recovered."
            )
    else:
        frame["dose"] = pd.to_numeric(frame[dose_col], errors="coerce").fillna(0.0).astype(float)

    frame["compound"] = frame["compound"].astype(str).str.strip()
    lower = frame["compound"].str.lower()
    vehicle_names = {"control", "dmso", "vehicle", "ctrl", "untreated", "mock", "none"}
    frame["is_vehicle"] = lower.isin(vehicle_names) | (frame["dose"].to_numpy(dtype=float) == 0.0)
    frame.loc[frame["is_vehicle"], "compound"] = "DMSO"
    if smiles_col is not None:
        frame["SMILES"] = frame[smiles_col].astype(str)
    else:
        frame["SMILES"] = ""
    if smiles_map:
        direct = frame["compound"].astype(str).str.strip().str.lower().map(smiles_map)
        normalized = frame["compound"].map(lambda x: smiles_map.get(_normalized_drug_key(x)))
        mapped = direct.where(direct.notna(), normalized)
        frame.loc[mapped.notna(), "SMILES"] = mapped[mapped.notna()].astype(str)
    frame.loc[frame["is_vehicle"] & frame["SMILES"].isin(["", "nan", "None"]), "SMILES"] = "CS(C)=O"
    if cell_line_col is None:
        frame["cell_line"] = ""
    else:
        frame["cell_line"] = frame[cell_line_col].astype(str)
    frame["log_dose"] = np.log1p(np.clip(frame["dose"].to_numpy(dtype=float), 0.0, None))
    frame["cell_id"] = [_slug_cell_id(idx, i) for i, idx in enumerate(frame.index)]
    treated_k = int(frame.loc[~frame["is_vehicle"], "compound"].nunique())
    if treated_k == 0:
        raise ValueError("sci-Plex obs normalization found no treated drug-dose compounds.")
    return frame


def _normalize_sciplex_metadata(obs: pd.DataFrame, smiles_map: dict[str, str] | None = None) -> pd.DataFrame:
    return normalize_sciplex_obs(obs, smiles_map=smiles_map)


def _cell_counts_by_compound_dose(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby(["compound", "dose", "is_vehicle"], observed=False)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["is_vehicle", "compound", "dose"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _load_lincs_name_smiles_map(cache_dir: str | Path, download: bool = True) -> dict[str, str]:
    cache_dir = ensure_dir(cache_dir)
    raw_path = cache_dir / "GSE92742_Broad_LINCS_pert_info.txt"
    if not raw_path.exists():
        load_lincs_smiles_corpus(cache_dir=cache_dir, download=download)
    if not raw_path.exists():
        return {}
    frame = pd.read_csv(raw_path, sep="\t", usecols=lambda col: col in {"pert_iname", "canonical_smiles"})
    if "pert_iname" not in frame.columns or "canonical_smiles" not in frame.columns:
        return {}
    raw_smiles = frame["canonical_smiles"].astype(str).str.strip()
    valid = ~raw_smiles.str.lower().isin(["", "-666", "restricted", "nan", "none"])
    out: dict[str, str] = {}
    for name, smiles in zip(frame.loc[valid, "pert_iname"].astype(str), raw_smiles[valid]):
        out[str(name).strip().lower()] = str(smiles)
        out[_normalized_drug_key(name)] = str(smiles)
    return out


def _subset_group_indices(metadata: pd.DataFrame, max_per_group: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in metadata.groupby(["compound", "dose", "is_vehicle"], observed=False):
        idx = group["_row_pos"].to_numpy(dtype=int)
        if len(idx) > int(max_per_group):
            idx = rng.choice(idx, size=int(max_per_group), replace=False)
        selected.append(np.sort(idx))
    return np.sort(np.concatenate(selected))


def _select_sparse_hvg(X, top_n: int) -> np.ndarray:
    try:
        from scipy import sparse

        if sparse.issparse(X):
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            var = mean_sq - mean**2
        else:
            var = np.var(np.asarray(X), axis=0)
    except Exception:
        var = np.var(_dense_array(X), axis=0)
    top_n = min(int(top_n), int(len(var)))
    return np.sort(np.argsort(var)[-top_n:])


def _prepare_scipert_sciplex3_a549_subset(
    source_path: Path,
    data_dir: Path,
    lincs_smiles_dir: str | Path,
    hvg_top_n: int,
    download: bool,
    seed: int,
):
    ad = _require_anndata()
    smiles_map = _load_lincs_name_smiles_map(lincs_smiles_dir, download=download)
    backed = ad.read_h5ad(source_path, backed="r")
    obs = backed.obs.copy()
    metadata = normalize_sciplex_obs(obs, smiles_map=smiles_map)
    a549_mask = metadata["cell_line"].astype(str).str.upper().eq("A549")
    if int(a549_mask.sum()) == 0:
        raise ValueError(f"scPerturb sci-Plex3 file {source_path} contains no A549 cells")
    metadata = metadata.loc[a549_mask].copy()
    metadata["_row_pos"] = np.flatnonzero(a549_mask.to_numpy())

    smiles_text = metadata["SMILES"].astype(str).str.strip().str.lower()
    treated = metadata.loc[(~metadata["is_vehicle"]) & (~smiles_text.isin(["", "nan", "none", "-666", "restricted"]))].copy()
    counts = treated.groupby(["compound", "dose"], observed=False).size().reset_index(name="n_cells")
    complete = (
        counts[counts["dose"].isin([10.0, 100.0, 1000.0, 10000.0])]
        .groupby("compound", observed=False)
        .agg(n_doses=("dose", "nunique"), min_cells=("n_cells", "min"))
        .reset_index()
    )
    eligible = complete.loc[(complete["n_doses"] >= 4) & (complete["min_cells"] >= 200), "compound"].astype(str).tolist()
    selected_compounds = sorted(eligible)[:8]
    if len(selected_compounds) < 4:
        raise ValueError(
            "Real scPerturb sci-Plex3 A549 data did not yield at least four compounds with complete 4-dose ladder, "
            "LINCS SMILES, and >=200 cells per dose."
        )
    subset_rule = (
        "A549 only; compounds with LINCS SMILES, doses 10/100/1000/10000 nM, "
        ">=200 cells per compound-dose; sorted by compound name; first 8 compounds; "
        "at most 250 cells per compound-dose and 250 vehicle cells sampled with fixed seed."
    )
    keep_meta = metadata.loc[metadata["is_vehicle"] | metadata["compound"].isin(selected_compounds)].copy()
    selected_rows = []
    vehicle = keep_meta.loc[keep_meta["is_vehicle"]].copy()
    if len(vehicle) > 1000:
        selected_rows.append(vehicle.sample(n=1000, random_state=seed))
    else:
        selected_rows.append(vehicle)
    selected_rows.append(keep_meta.loc[~keep_meta["is_vehicle"]])
    keep_meta = pd.concat(selected_rows, axis=0).copy()
    keep_positions = _subset_group_indices(keep_meta, max_per_group=250, seed=seed)
    final_meta = metadata.loc[metadata["_row_pos"].isin(keep_positions)].copy()
    final_meta = final_meta.sort_values("_row_pos")
    row_positions = final_meta["_row_pos"].to_numpy(dtype=int)

    adata_rows = backed[row_positions, :].to_memory()
    keep_genes = _select_sparse_hvg(adata_rows.X, hvg_top_n)
    adata = backed[row_positions, keep_genes].to_memory()
    final_meta = final_meta.drop(columns=["_row_pos"])
    final_meta.index = adata.obs_names.astype(str)
    final_meta["cell_id"] = [_slug_cell_id(idx, i) for i, idx in enumerate(final_meta.index)]
    adata.obs = final_meta.copy()
    backed.file.close()
    return adata, final_meta, {
        "source": "scPerturb SrivatsanTrapnell2020_sciplex3.h5ad",
        "source_url": SCPERT_SCIPLEX3_URL,
        "is_synthetic": False,
        "obs_schema_used": {
            "cell_line": "cell_line",
            "compound": "perturbation",
            "dose": "dose_value",
            "vehicle": "perturbation == control or dose_value == 0",
            "smiles": "LINCS GSE92742 pert_iname/canonical_smiles map",
        },
        "subset_rule": subset_rule,
        "compound_list": selected_compounds,
    }


def _maybe_filter_a549(adata):
    obs = adata.obs
    cell_col = _first_existing_column(obs, ["cell_type", "cell_line", "cell_id", "cell"])
    if cell_col is None:
        return adata
    values = obs[cell_col].astype(str)
    mask = values.str.upper().eq("A549")
    if int(mask.sum()) == 0:
        return adata
    return adata[mask.to_numpy()].copy()


def _extract_cellot_sciplex_zip(zip_path: Path, target_dir: Path) -> dict[str, Path]:
    ensure_dir(target_dir)
    wanted = {
        "hvg.h5ad": target_dir / "hvg.h5ad",
        "hvg-top1k.h5ad": target_dir / "hvg-top1k.h5ad",
        "hvg-top1k-train-only.h5ad": target_dir / "hvg-top1k-train-only.h5ad",
    }
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for filename, output_path in wanted.items():
            matches = [name for name in names if name.endswith(f"scrna-sciplex3/{filename}") or name.endswith(filename)]
            if not matches:
                continue
            with archive.open(matches[0]) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted[filename] = output_path
    return extracted


def _generate_synthetic_sciplex_a549(path: Path, seed: int = 42):
    ad = _require_anndata()
    rng = np.random.default_rng(seed)
    compounds = {
        "Belinostat": "CC1=C(C=C(C=C1)NC(=O)CCCCCCC(=O)NO)C",
        "Vorinostat": "C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO",
        "Trametinib": "CC1=C(C(=NC=C1)NC(=O)C2=CC=C(C=C2)F)C",
        "Dacinostat": "CC(C)C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO",
    }
    doses = [0.01, 0.1, 1.0, 10.0]
    rows = []
    X_rows = []
    n_genes = 1200
    base = rng.normal(0.0, 0.3, size=n_genes)
    for i in range(180):
        rows.append({"compound": "DMSO", "dose": 0.0, "is_vehicle": True, "SMILES": "CS(C)=O"})
        X_rows.append(base + rng.normal(0.0, 0.7, size=n_genes))
    for c_idx, (compound, smiles) in enumerate(compounds.items()):
        direction = rng.normal(0.0, 1.0, size=n_genes)
        direction = direction / np.clip(np.linalg.norm(direction), 1e-12, None)
        for dose in doses:
            strength = (0.25 + 0.15 * c_idx) * np.log1p(dose)
            n = 55 if dose < 10 else 45
            for _ in range(n):
                rows.append({"compound": compound, "dose": dose, "is_vehicle": False, "SMILES": smiles})
                X_rows.append(base + strength * direction * 8.0 + rng.normal(0.0, 0.7, size=n_genes))
    obs = pd.DataFrame(rows)
    obs.index = [f"synthetic_a549_{i:05d}" for i in range(len(obs))]
    obs["cell_type"] = "A549"
    obs["log_dose"] = np.log1p(obs["dose"].astype(float))
    var = pd.DataFrame(index=[f"gene_{i:04d}" for i in range(n_genes)])
    adata = ad.AnnData(X=np.asarray(X_rows, dtype=np.float32), obs=obs, var=var)
    ensure_dir(path.parent)
    adata.write_h5ad(path)
    return adata


def load_or_prepare_sciplex3_a549(
    data_dir: str | Path = SCIPLEX3_A549_DIR,
    lincs_smiles_dir: str | Path = CHEMCPA_LINCS_SMILES_DIR,
    download: bool = True,
    hvg_top_n: int = 1000,
    overwrite: bool = False,
    synthetic_if_missing: bool = False,
    seed: int = 42,
) -> SciplexPreparedData:
    """Load/cache the CellOT sci-Plex 3 A549 subset and basic Chapter 5 metadata.

    This function prepares data only. Split-specific PCA is handled by
    ``make_sciplex_pca_state_table`` so held-out cells never affect PCA or
    per-PC standardization.
    """
    ad = _require_anndata()
    data_dir = ensure_dir(data_dir)
    raw_dir = ensure_dir(data_dir / "raw")
    prepared_path = data_dir / "sciplex3_a549_hvg_top1000.h5ad"
    metadata_path = data_dir / "metadata.csv"
    counts_path = data_dir / "compound_dose_cell_counts.csv"
    summary_path = data_dir / "data_cache_summary.json"

    if prepared_path.exists() and not overwrite:
        adata = ad.read_h5ad(prepared_path)
        summary = pd.read_json(summary_path, typ="series").to_dict() if summary_path.exists() else {}
        is_cached_synthetic = bool(summary.get("is_synthetic", False)) or "synthetic" in str(summary.get("source", "")).lower()
        if is_cached_synthetic and not synthetic_if_missing:
            overwrite = True
        else:
            metadata = pd.read_csv(metadata_path, index_col=0) if metadata_path.exists() else _normalize_sciplex_metadata(adata.obs)
            counts = pd.read_csv(counts_path) if counts_path.exists() else _cell_counts_by_compound_dose(metadata)
            cached_k = int(metadata.loc[~metadata["is_vehicle"], "compound"].nunique()) if "is_vehicle" in metadata else 0
            if cached_k == 0:
                if synthetic_if_missing:
                    overwrite = True
                else:
                    raise ValueError(
                        f"Cached sci-Plex file at {prepared_path} has no treated compounds after metadata parsing. "
                        "Remove/rebuild the cache or pass synthetic_if_missing=True for a pipeline smoke run."
                    )
            else:
                if summary_path.exists():
                    summary["cache_status"] = "cached"
                else:
                    summary = {
                        "source": "cached",
                        "source_url": "",
                        "is_synthetic": False,
                        "adata_path": str(prepared_path),
                        "n_cells": int(adata.n_obs),
                        "n_genes": int(adata.n_vars),
                        "K_compounds": int(cached_k),
                    }
                return SciplexPreparedData(
                    adata=adata,
                    metadata=metadata,
                    cell_counts=counts,
                    paths={
                        "adata": str(prepared_path),
                        "metadata": str(metadata_path),
                        "cell_counts": str(counts_path),
                        "summary": str(summary_path),
                    },
                    summary=summary,
                )
    source_candidates = [
        raw_dir / "SrivatsanTrapnell2020_sciplex3.h5ad",
        data_dir / "hvg.h5ad",
        data_dir / "hvg-top1k.h5ad",
        data_dir / "hvg-top1k-train-only.h5ad",
        Path("../baselines/cellot/datasets/scrna-sciplex3/hvg.h5ad"),
        Path("../baselines/cellot/datasets/scrna-sciplex3/hvg-top1k.h5ad"),
        Path("../baselines/cellot/datasets/scrna-sciplex3/hvg-top1k-train-only.h5ad"),
    ]
    source_path = next((path for path in source_candidates if Path(path).exists()), None)
    download_error = None
    if source_path is None and download:
        try:
            scpert_path = raw_dir / "SrivatsanTrapnell2020_sciplex3.h5ad"
            _download_file(SCPERT_SCIPLEX3_URL, scpert_path, overwrite=False)
            source_path = scpert_path
        except Exception as exc:  # pragma: no cover - network-dependent
            download_error = str(exc)
    if source_path is None and download:
        try:
            zip_path = raw_dir / "processed_datasets_all.zip"
            _download_file(CELLOT_PROCESSED_DATASETS_URL, zip_path, overwrite=False)
            extracted = _extract_cellot_sciplex_zip(zip_path, data_dir)
            source_path = extracted.get("hvg.h5ad") or extracted.get("hvg-top1k.h5ad") or extracted.get("hvg-top1k-train-only.h5ad")
        except Exception as exc:  # pragma: no cover - network-dependent
            download_error = str(exc)

    if source_path is None:
        if synthetic_if_missing:
            adata = _generate_synthetic_sciplex_a549(prepared_path, seed=seed)
            source_note = "synthetic_fallback"
        else:
            raise FileNotFoundError(
                "Could not find or download CellOT sci-Plex 3 hvg-top1k data. "
                f"Checked: {[str(p) for p in source_candidates]}. "
                f"Download error: {download_error}"
            )
    else:
        if Path(source_path).name == "SrivatsanTrapnell2020_sciplex3.h5ad":
            adata, metadata, source_info = _prepare_scipert_sciplex3_a549_subset(
                Path(source_path),
                data_dir=data_dir,
                lincs_smiles_dir=lincs_smiles_dir,
                hvg_top_n=hvg_top_n,
                download=download,
                seed=seed,
            )
            source_note = source_info["source"]
        else:
            adata = ad.read_h5ad(source_path)
            adata = _maybe_filter_a549(adata)
            metadata = _normalize_sciplex_metadata(adata.obs)
            source_note = str(source_path)
            source_info = {
                "source": source_note,
                "source_url": CELLOT_PROCESSED_DATASETS_URL,
                "is_synthetic": False,
                "obs_schema_used": {},
                "subset_rule": "CellOT local h5ad filtered to A549 if a cell-line column is present.",
                "compound_list": sorted(metadata.loc[~metadata["is_vehicle"], "compound"].astype(str).unique().tolist()),
            }

    parsed_k = int(metadata.loc[~metadata["is_vehicle"], "compound"].nunique())
    if parsed_k == 0:
        if synthetic_if_missing:
            adata = _generate_synthetic_sciplex_a549(prepared_path, seed=seed)
            metadata = _normalize_sciplex_metadata(adata.obs)
            source_note = f"synthetic_fallback_invalid_source:{source_note}"
            source_info = {
                "source": source_note,
                "source_url": "",
                "is_synthetic": True,
                "obs_schema_used": {},
                "subset_rule": "synthetic smoke fallback",
                "compound_list": sorted(metadata.loc[~metadata["is_vehicle"], "compound"].astype(str).unique().tolist()),
            }
        else:
            raise ValueError(
                f"Source sci-Plex file {source_note} has no treated drug-dose compounds after metadata parsing. "
                "This is not a usable sci-Plex 3 A549 perturbation cache."
            )
    adata.obs = metadata.copy()
    if int(hvg_top_n) > 0 and adata.n_vars > int(hvg_top_n):
        # Prefer CellOT's hvg-top1k if present; this variance fallback keeps raw
        # downloads bounded but split-specific HVG is still refit downstream.
        X = _dense_array(adata.X).astype(np.float32)
        var = np.var(X, axis=0)
        keep = np.sort(np.argsort(var)[-int(hvg_top_n) :])
        adata = adata[:, keep].copy()

    metadata = _normalize_sciplex_metadata(adata.obs)
    counts = _cell_counts_by_compound_dose(metadata)
    ensure_dir(prepared_path.parent)
    adata.write_h5ad(prepared_path)
    metadata.to_csv(metadata_path)
    counts.to_csv(counts_path, index=False)

    smiles_text = metadata.loc[~metadata["is_vehicle"], "SMILES"].astype(str).str.strip().str.lower()
    missing_smiles = int(smiles_text.isin(["", "nan", "none", "-666", "restricted"]).sum())
    summary = {
        "source": source_info.get("source", source_note),
        "source_url": source_info.get("source_url", ""),
        "download_url": source_info.get("source_url", CELLOT_PROCESSED_DATASETS_URL),
        "download_error": download_error,
        "is_synthetic": bool(source_info.get("is_synthetic", False)),
        "adata_path": str(prepared_path),
        "metadata_path": str(metadata_path),
        "cell_counts_path": str(counts_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "K_compounds": int(metadata.loc[~metadata["is_vehicle"], "compound"].nunique()),
        "compound_list": sorted(metadata.loc[~metadata["is_vehicle"], "compound"].astype(str).unique().tolist()),
        "vehicle_count": int(metadata["is_vehicle"].sum()),
        "dose_values": sorted(map(float, metadata.loc[~metadata["is_vehicle"], "dose"].dropna().unique().tolist())),
        "missing_smiles_count": missing_smiles,
        "hvg_top_n": int(hvg_top_n),
        "obs_schema_used": source_info.get("obs_schema_used", {}),
        "subset_rule": source_info.get("subset_rule", ""),
    }
    pd.Series(summary).to_json(summary_path, indent=2)
    load_lincs_smiles_corpus(cache_dir=lincs_smiles_dir, download=download)
    return SciplexPreparedData(
        adata=adata,
        metadata=metadata,
        cell_counts=counts,
        paths={
            "adata": str(prepared_path),
            "metadata": str(metadata_path),
            "cell_counts": str(counts_path),
            "summary": str(summary_path),
        },
        summary=summary,
    )


def make_sciplex_split(
    split_name: str,
    metadata: pd.DataFrame,
    test_fraction: float = 0.2,
    heldout_compound: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create Chapter 5 Split A/B/C metadata without leaking held-out groups."""
    frame = _normalize_sciplex_metadata(metadata)
    frame["split"] = "train"
    name = str(split_name).lower().replace("-", "_").replace(" ", "_")
    rng = np.random.default_rng(seed)

    if name in {"a", "split_a", "random", "random_sanity"}:
        frame["split_name"] = "split_a_random"
        for idx in frame.groupby(["compound", "dose", "is_vehicle"], observed=False).indices.values():
            idx = np.asarray(idx, dtype=int)
            if len(idx) <= 1:
                continue
            n_test = max(1, int(round(len(idx) * float(test_fraction))))
            n_test = min(n_test, len(idx) - 1)
            test_idx = rng.choice(idx, size=n_test, replace=False)
            frame.iloc[test_idx, frame.columns.get_loc("split")] = "test"
    elif name in {"b", "split_b", "heldout_highest_dose", "highest_dose"}:
        frame["split_name"] = "split_b_heldout_highest_dose"
        treated = frame[~frame["is_vehicle"]]
        for compound, group in treated.groupby("compound", observed=False):
            highest = float(group["dose"].max())
            frame.loc[(frame["compound"] == compound) & (~frame["is_vehicle"]) & (frame["dose"] == highest), "split"] = "test"
    elif name in {"c", "split_c", "heldout_compound", "compound"}:
        if heldout_compound is None:
            raise ValueError("heldout_compound is required for Split C")
        frame["split_name"] = "split_c_heldout_compound"
        frame.loc[(frame["compound"] == str(heldout_compound)) & (~frame["is_vehicle"]), "split"] = "test"
    else:
        raise ValueError(f"Unknown sci-Plex split_name={split_name!r}")

    frame["is_train"] = frame["split"].eq("train")
    frame["is_test"] = frame["split"].eq("test")
    return frame


def make_sciplex_pca_state_table(
    adata,
    split_metadata: pd.DataFrame,
    n_pcs: int = 30,
    hvg_top_n: int = 1000,
    eps: float = 1e-6,
) -> SciplexPCAStateTable:
    """Fit HVG, PCA, and per-PC standardization on training cells only."""
    from sklearn.decomposition import PCA

    metadata = _normalize_sciplex_metadata(split_metadata)
    if "split" not in split_metadata.columns:
        raise ValueError("split_metadata must contain a split column")
    metadata["split"] = split_metadata["split"].to_numpy()
    train_mask = metadata["split"].to_numpy() == "train"
    if int(train_mask.sum()) < 2:
        raise ValueError("At least two training cells are required for split-aware PCA")

    X = _dense_array(adata.X).astype(np.float32)
    if X.shape[0] != len(metadata):
        raise ValueError("adata.n_obs must match split_metadata rows")
    if np.any(~np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    hvg_top_n = min(int(hvg_top_n), X.shape[1])
    if hvg_top_n > 0 and hvg_top_n < X.shape[1]:
        var = np.var(X[train_mask], axis=0)
        keep = np.sort(np.argsort(var)[-hvg_top_n:])
    else:
        keep = np.arange(X.shape[1])
    X_hvg = X[:, keep]
    n_components = min(int(n_pcs), X_hvg.shape[1], int(train_mask.sum()) - 1)
    if n_components <= 0:
        raise ValueError("n_pcs must be positive after accounting for training cells and genes")
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    pca.fit(X_hvg[train_mask])
    scores = pca.transform(X_hvg).astype(np.float32)
    mean = scores[train_mask].mean(axis=0).astype(np.float32)
    std = scores[train_mask].std(axis=0).astype(np.float32)
    std = np.where(std < float(eps), 1.0, std).astype(np.float32)
    scores = ((scores - mean) / std).astype(np.float32)
    if n_components < int(n_pcs):
        pad = np.zeros((scores.shape[0], int(n_pcs) - n_components), dtype=np.float32)
        scores = np.hstack([scores, pad])
        evr = list(map(float, pca.explained_variance_ratio_)) + [0.0] * (int(n_pcs) - n_components)
        mean = np.r_[mean, np.zeros(int(n_pcs) - n_components, dtype=np.float32)]
        std = np.r_[std, np.ones(int(n_pcs) - n_components, dtype=np.float32)]
    else:
        evr = list(map(float, pca.explained_variance_ratio_))

    var_names = np.asarray(getattr(adata, "var_names", [f"gene_{i}" for i in range(X.shape[1])])).astype(str)
    hvg_genes = var_names[keep].tolist()
    return SciplexPCAStateTable(
        X_pca=scores.astype(np.float32),
        metadata=metadata.copy(),
        hvg_genes=hvg_genes,
        pca_explained_variance_ratio=evr,
        train_mean=mean.astype(np.float32),
        train_std=std.astype(np.float32),
    )


def load_lincs_smiles_corpus(
    cache_dir: str | Path = CHEMCPA_LINCS_SMILES_DIR,
    download: bool = True,
    overwrite: bool = False,
) -> LincsSmilesCorpus:
    """Load/cache the chemCPA LINCS SMILES corpus used for external RDKit2D normalization."""
    cache_dir = ensure_dir(cache_dir)
    csv_path = cache_dir / "lincs_smiles.csv"
    raw_path = cache_dir / "GSE92742_Broad_LINCS_pert_info.txt"
    if not csv_path.exists() or overwrite:
        if not raw_path.exists() or overwrite:
            if not download:
                raise FileNotFoundError(csv_path)
            gz_path = cache_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz"
            _download_file(LINCS_PERT_INFO_URL, gz_path, overwrite=overwrite)
            with gzip.open(gz_path, "rb") as src, raw_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        frame = pd.read_csv(raw_path, sep="\t")
        smiles_col = _first_existing_column(frame, ["canonical_smiles", "SMILES", "smiles"])
        if smiles_col is None:
            raise ValueError(f"No SMILES column found in {raw_path}")
        cols = [col for col in ["pert_id", smiles_col] if col in frame.columns]
        out = frame[cols].copy()
        if smiles_col != "canonical_smiles":
            out = out.rename(columns={smiles_col: "canonical_smiles"})
        out.to_csv(csv_path, index=False)
    frame = pd.read_csv(csv_path)
    smiles_col = _first_existing_column(frame, ["canonical_smiles", "SMILES", "smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {csv_path}")
    raw = frame[smiles_col].astype(str)
    invalid = raw.str.lower().isin(["", "-666", "restricted", "nan", "none"])
    smiles = raw[~invalid].drop_duplicates().tolist()
    return LincsSmilesCorpus(smiles=smiles, frame=frame, path=csv_path, n_invalid=int(invalid.sum()))


def _make_rdkit2d_generator():
    try:
        from descriptastorus.descriptors import rdDescriptors
    except ImportError as exc:
        raise ImportError(
            "descriptastorus with RDKit support is required for RDKit2D descriptors. "
            "Install descriptastorus/rdkit or pass an injected generator for tests."
        ) from exc
    return rdDescriptors.RDKit2D()


def _descriptor_names(generator, n_features: int) -> list[str]:
    columns = getattr(generator, "columns", None)
    if columns is None:
        return [f"rdkit2d_{i:03d}" for i in range(int(n_features))]
    names = []
    for col in list(columns)[1:]:
        if isinstance(col, tuple):
            names.append(str(col[0]))
        else:
            names.append(str(col))
    if len(names) != int(n_features):
        names = [f"rdkit2d_{i:03d}" for i in range(int(n_features))]
    return names


def _process_smiles_descriptor(generator, smiles: str, n_features: int | None = None) -> tuple[np.ndarray | None, bool, int]:
    try:
        values = list(generator.process(str(smiles)))
        if len(values) == 0:
            return None, False, 0
        success = bool(values[0])
        arr = np.asarray(values[1:], dtype=np.float32)
        nan_inf = int((~np.isfinite(arr)).sum())
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if n_features is not None and arr.shape[0] != int(n_features):
            fixed = np.zeros(int(n_features), dtype=np.float32)
            fixed[: min(len(arr), int(n_features))] = arr[: min(len(arr), int(n_features))]
            arr = fixed
        return arr, bool(success), nan_inf
    except Exception:
        if n_features is None:
            return None, False, 0
        return np.zeros(int(n_features), dtype=np.float32), False, 0


def compute_rdkit2d_with_external_norm(
    smiles: list[str] | pd.Series | np.ndarray,
    external_smiles: list[str] | pd.Series | np.ndarray,
    generator=None,
    eps: float = 1e-8,
) -> RDKit2DResult:
    """Compute RDKit2D descriptors and standardize with an external LINCS corpus."""
    smiles = [str(s) for s in list(smiles)]
    external_smiles = [str(s) for s in list(external_smiles)]
    backend = "injected" if generator is not None else "descriptastorus"
    generator = _make_rdkit2d_generator() if generator is None else generator

    n_features = None
    external_raw = []
    external_failures = 0
    nan_inf_count = 0
    for smi in external_smiles:
        arr, success, n_bad = _process_smiles_descriptor(generator, smi, n_features=n_features)
        if arr is None:
            external_failures += 1
            continue
        if n_features is None:
            n_features = int(arr.shape[0])
        if not success:
            external_failures += 1
        nan_inf_count += int(n_bad)
        external_raw.append(arr)
    if n_features is None or len(external_raw) == 0:
        raise ValueError("No valid external SMILES descriptors were available for RDKit2D normalization")
    external_matrix = np.vstack(external_raw).astype(np.float32)
    mean = external_matrix.mean(axis=0).astype(np.float32)
    std = external_matrix.std(axis=0).astype(np.float32)
    std_too_small = std < float(eps)
    std = np.where(std_too_small, 1.0, std).astype(np.float32)

    raw_rows = []
    failures = 0
    for smi in smiles:
        arr, success, n_bad = _process_smiles_descriptor(generator, smi, n_features=n_features)
        if arr is None:
            arr = np.zeros(int(n_features), dtype=np.float32)
            success = False
        if not success:
            failures += 1
        nan_inf_count += int(n_bad)
        raw_rows.append(arr)
    raw_features = np.vstack(raw_rows).astype(np.float32)
    features = ((raw_features - mean) / std).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    diagnostics = {
        "descriptor_backend": backend,
        "D_RDKit": int(features.shape[1]),
        "n_query_smiles": int(len(smiles)),
        "n_external_smiles": int(len(external_smiles)),
        "smiles_failure_count": int(failures),
        "external_smiles_failure_count": int(external_failures),
        "nan_inf_count": int(nan_inf_count),
        "std_too_small_count": int(std_too_small.sum()),
        "std_eps": float(eps),
    }
    return RDKit2DResult(
        features=features,
        raw_features=raw_features,
        external_mean=mean,
        external_std=std,
        feature_names=_descriptor_names(generator, features.shape[1]),
        diagnostics=diagnostics,
    )


# Chapter 4 manifold data helpers.
def load_eb_data(
    path: str | Path,
    *,
    source_time: str = "1",
    target_time: str = "2",
    out_dir: str | Path | None = None,
    max_cells_per_time: int | None = None,
    seed: int = 42,
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)
    pcs_full = np.asarray(z["pcs"], dtype=np.float32)
    phate = np.asarray(z["phate"], dtype=np.float32)[:, :2]
    labels = np.asarray(z["sample_labels"]).astype(str)
    pcs20_raw = pcs_full[:, :20].astype(np.float32)
    mean = pcs20_raw.mean(axis=0)
    std = pcs20_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    pcs20 = ((pcs20_raw - mean) / std).astype(np.float32)
    available = sorted(np.unique(labels).tolist(), key=lambda s: int(s) if str(s).isdigit() else str(s))
    if str(source_time) not in available or str(target_time) not in available:
        raise ValueError(f"Requested bridge {source_time}->{target_time}; available labels: {available}")
    idx_source_all = np.flatnonzero(labels == str(source_time))
    idx_target_all = np.flatnonzero(labels == str(target_time))
    if max_cells_per_time is not None:
        idx_source = idx_source_all[sample_rows(len(idx_source_all), max_cells_per_time, seed=seed)]
        idx_target = idx_target_all[sample_rows(len(idx_target_all), max_cells_per_time, seed=seed + 1)]
    else:
        idx_source, idx_target = idx_source_all, idx_target_all
    counts = pd.Series(labels, name="time").value_counts().sort_index().rename_axis("time").reset_index(name="n_cells")
    summary = {
        "source_path": str(path),
        "available_keys": keys,
        "pcs_shape_actual": list(pcs_full.shape),
        "pc20_shape_used": list(pcs20.shape),
        "phate_shape": list(phate.shape),
        "sample_label_values": counts.to_dict(orient="records"),
        "source_time": str(source_time),
        "target_time": str(target_time),
        "n_source_full": int(len(idx_source_all)),
        "n_target_full": int(len(idx_target_all)),
        "n_source_used": int(len(idx_source)),
        "n_target_used": int(len(idx_target)),
        "training_space": "standardized PC-20 from pcs[:, :20]",
        "ot_cost_space": "standardized PC-20 unless Exp 7 contrastive diagnostic",
        "display_space": "PHATE 2D only for visualization",
        "distributional_metrics_space": "standardized PC-20",
        "standardization": "mean/std fit on all EB snapshots in PC-20",
        "adaptation_note": "Input pcs has 100 columns; this chapter uses the first 20 PCs as PC-20.",
    }
    if out_dir is not None:
        save_json(Path(out_dir) / "eb_data_summary.json", summary)
    return {
        "keys": keys,
        "pcs20_all": pcs20,
        "pcs20_raw_all": pcs20_raw,
        "phate_all": phate,
        "labels": labels,
        "counts": counts,
        "pc_mean": mean,
        "pc_std": std,
        "idx_source": idx_source,
        "idx_target": idx_target,
        "X0_pc": pcs20[idx_source],
        "X1_pc": pcs20[idx_target],
        "X0_phate": phate[idx_source],
        "X1_phate": phate[idx_target],
        "source_time": str(source_time),
        "target_time": str(target_time),
        "summary": summary,
    }

def load_toy_snapshots(path: str | Path) -> pd.DataFrame:
    """Load toy snapshot CSVs with a numeric time column."""
    frame = pd.read_csv(path)
    frame["time"] = frame["time"].astype(float)
    return frame

def load_toy_expression_representations(path: str | Path, anndata_module=None) -> dict:
    """Load toy pseudocount representations used by the Chapter 4.2 state-space diagnostic."""
    if anndata_module is None:
        try:
            import anndata as anndata_module
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("anndata is required for toy pseudocount representation experiment") from exc
    adata = anndata_module.read_h5ad(path)
    X_counts = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    lib = np.maximum(X_counts.sum(axis=1, keepdims=True), 1.0)
    X_log = np.log1p(X_counts / lib * 1e4).astype(np.float32)
    pca_state = fit_pca_state_space(X_log, n_components=30, seed=42)
    programs = program_index_dict(adata, program_key="program", include_background=False)
    program_scores, program_names = readout_program_scores_from_matrix(X_log, programs)
    obs = adata.obs.reset_index(drop=True).copy()
    obs["time"] = obs["time"].astype(float)
    idx0 = np.flatnonzero(obs["time"].to_numpy() == obs["time"].min())
    idx1 = np.flatnonzero(obs["time"].to_numpy() == obs["time"].max())
    X0_pca_raw, X1_pca_raw = pca_state["coords"][idx0], pca_state["coords"][idx1]
    X0_prog_raw, X1_prog_raw = program_scores[idx0], program_scores[idx1]
    X0_pca, X1_pca, pca_std = standardize_train_space(X0_pca_raw, X1_pca_raw)
    X0_prog, X1_prog, prog_std = standardize_train_space(X0_prog_raw, X1_prog_raw)
    return {
        "adata": adata,
        "obs": obs,
        "X_log": X_log,
        "pca_state": pca_state,
        "programs": programs,
        "program_names": program_names,
        "program_scores": program_scores,
        "idx0": idx0,
        "idx1": idx1,
        "X0_pca": X0_pca,
        "X1_pca": X1_pca,
        "X0_prog": X0_prog,
        "X1_prog": X1_prog,
        "X0_prog_raw": X0_prog_raw,
        "X1_prog_raw": X1_prog_raw,
        "pca_std": pca_std,
        "prog_std": prog_std,
        "X0_viz": np.asarray(adata.obsm["X_toy_state"])[idx0].astype(np.float32),
        "X1_viz": np.asarray(adata.obsm["X_toy_state"])[idx1].astype(np.float32),
        "labels1": obs.iloc[idx1]["fate_label"].astype(str).to_numpy(),
    }

def fit_pc_to_phate_mapper(pcs, phate, n_neighbors: int = 15) -> Callable[[np.ndarray], np.ndarray]:
    from sklearn.neighbors import KNeighborsClassifier

    pcs = np.asarray(pcs, dtype=np.float32)
    phate = np.asarray(phate, dtype=np.float32)[:, :2]
    knn = KNeighborsClassifier(n_neighbors=min(int(n_neighbors), len(pcs)), weights="distance")
    knn.fit(pcs, np.arange(len(pcs)))

    def pc_to_phate(points_pc):
        points_pc = np.asarray(points_pc, dtype=np.float32)
        dist, ind = knn.kneighbors(points_pc, return_distance=True)
        w = 1.0 / np.clip(dist, 1e-6, None)
        w = w / w.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkd->nd", w, phate[ind])

    return pc_to_phate
