from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .artifacts import display_saved_figure, save_csv, save_figure as save_artifact_figure, save_json, sample_rows
from .representations import (
    fit_pca_state_space,
    program_index_dict,
    readout_program_scores_from_matrix,
    standardize_train_space,
)


CH04_PALETTE = {
    "source": "#4C78A8",
    "target": "#F58518",
    "random": "#8E8E8E",
    "ot": "#008A7A",
    "reflow1": "#5369A6",
    "reflow2": "#B279A2",
    "rare": "#D95F02",
    "major": "#2C7FB8",
    "program": "#54A24B",
    "diagnostic": "#E45756",
}

METHOD_LABELS = {
    "random_cfm": "Random CFM",
    "ot_cfm": "OT-CFM",
    "reflow_1": "Reflow 1",
    "reflow_2": "Reflow 2",
    "random": "Random CFM",
    "ot": "OT-CFM",
    "reflow1": "Reflow 1",
    "reflow2": "Reflow 2",
}

METHOD_COLORS = {
    "random_cfm": CH04_PALETTE["random"],
    "ot_cfm": CH04_PALETTE["ot"],
    "reflow_1": CH04_PALETTE["reflow1"],
    "reflow_2": CH04_PALETTE["reflow2"],
    "random": CH04_PALETTE["random"],
    "ot": CH04_PALETTE["ot"],
    "reflow1": CH04_PALETTE["reflow1"],
    "reflow2": CH04_PALETTE["reflow2"],
}

REP_PAIR_QUANTILES = (0.50, 0.75, 0.90, 0.95)
REP_TRAJ_QUANTILES = (0.35, 0.55, 0.75, 0.90)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method).replace("_", " "))


def method_color(method: str) -> str:
    return METHOD_COLORS.get(str(method), "0.35")


def save_ch04_figure(fig, fig_dir: str | Path, filename: str, close: bool = True) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    path = save_artifact_figure(fig, fig_dir, filename, write_pdf=False)
    if close:
        plt.close(fig)
    return path


def display_ch04_figure(fig_dir: str | Path, filename: str, width: int = 900) -> Path:
    return display_saved_figure(Path(fig_dir) / filename, width=width)


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


def fate_conditioned_plan(X0, X1, source_labels, target_labels, source_labels_for_plan=None) -> np.ndarray:
    """Build a row-balanced empirical plan constrained by matched fate labels."""
    source_labels = np.asarray(source_labels).astype(str)
    target_labels = np.asarray(target_labels).astype(str)
    labels_for_plan = source_labels if source_labels_for_plan is None else np.asarray(source_labels_for_plan).astype(str)
    pi = np.zeros((len(X0), len(X1)), dtype=float)
    X0 = np.asarray(X0, dtype=np.float32)
    X1 = np.asarray(X1, dtype=np.float32)
    for i, lab in enumerate(labels_for_plan):
        cols = np.flatnonzero(target_labels == lab)
        if cols.size == 0:
            cols = np.arange(len(X1))
        d2 = np.sum((X1[cols] - X0[i]) ** 2, axis=1)
        scale = max(float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0, 1e-6)
        w = np.exp(-d2 / scale)
        w = w / np.clip(w.sum(), 1e-12, None)
        pi[i, cols] = w / len(X0)
    return pi / np.clip(pi.sum(), 1e-12, None)


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


def build_artifact_manifest(
    *,
    project_root: str | Path,
    fig_dir: str | Path,
    out_dir: str | Path,
    run_config: Mapping[str, object],
    expected_figures: Iterable[str | Path],
    expected_tables: Iterable[str | Path],
    dependency_files: Iterable[str | Path] = (),
) -> pd.DataFrame:
    project_root = Path(project_root)
    fig_dir = Path(fig_dir)
    out_dir = Path(out_dir)
    rows = []
    for key, value in run_config.items():
        rows.append({"artifact": f"RUN_CONFIG:{key}={value}", "kind": "run_config", "exists": True, "bytes": 0})
    for name in expected_figures:
        p = fig_dir / name
        rows.append({"artifact": str(p.relative_to(project_root)), "kind": "figure", "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0})
    for name in expected_tables:
        p = out_dir / name
        rows.append({"artifact": str(p.relative_to(project_root)), "kind": "table_or_json", "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0})
    for path in dependency_files:
        p = Path(path)
        rows.append({"artifact": str(p.relative_to(project_root)), "kind": "dependency", "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0})
    return pd.DataFrame(rows)


def write_artifact_manifest(manifest_path: str | Path, manifest: pd.DataFrame) -> Path:
    return save_csv(manifest_path, manifest)
