from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..artifacts import display_saved_figure, save_figure_formats
from ..evaluation.graph_paths import extract_path_indices
from ..core.ot import sample_from_plan


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

NEW_SMALL_FIGURES = [
    "fig4_2_toy_pca30_representative_pairs.png",
    "fig4_2_toy_program4_representative_pairs.png",
    "fig4_2_toy_representation_coupling_summary.png",
    "fig4_2_eb_pc20_coupling_representative_pairs.png",
    "fig4_2_eb_phate_diagnostic_coupling_representative_pairs.png",
    "fig4_2_eb_pc_vs_phate_distance_summary.png",
    "fig4_2_state_space_model_readout_summary.png",
    "fig4_3_toy_single_pair_chord_vs_graph_path.png",
    "fig4_3_eb_chord_vs_graph_matched_examples.png",
    "fig4_3_eb_density_radius_delta.png",
    "fig4_3_eb_knn_radius_delta.png",
    "fig4_3_eb_off_manifold_positive_fraction.png",
]

SMALL_FIGURE_DEPENDENCIES = [
    "outputs/ch04/table4_3_representation_coupling_diagnostics.csv",
    "outputs/ch04/table4_4_state_space_model_metrics.csv",
    "outputs/ch04/table4_5_eb_representation_coupling_diagnostics.csv",
    "outputs/ch04/exp8_off_manifold_stats.csv",
    "outputs/ch04/exp8_pair_selection_diagnostics.csv",
    "outputs/ch04/exp8_eb_off_manifold_stats.csv",
    "outputs/ch04/exp8_eb_pair_selection_diagnostics.csv",
    "outputs/ch04/cache/exp1_eb_couplings.npz",
    "outputs/ch04/cache/exp6_toy_pca30_d30_steps1500_batch256_seed70_model.pt",
    "outputs/ch04/cache/exp6_toy_program4_d4_steps1500_batch256_seed70_model.pt",
]


def save_small_figure(fig, fig_dir: str | Path, filename: str, *, display: bool = True) -> Path:
    paths = save_figure_formats(fig, fig_dir, Path(filename).stem, formats=(Path(filename).suffix.lstrip(".") or "png",), dpi=220, close=True)
    path = paths[0]
    if display:
        display_saved_figure(path)
    return path


def fmt_int(value) -> str:
    return f"{float(value):,.0f}"


def fmt_float(value, ndigits: int = 2) -> str:
    return f"{float(value):.{int(ndigits)}f}"


def axis_limits(*arrays, pad: float = 0.06):
    pts = np.vstack([np.asarray(a, dtype=float).reshape(-1, 2) for a in arrays if len(a)])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return (lo[0] - pad * span[0], hi[0] + pad * span[0]), (lo[1] - pad * span[1], hi[1] + pad * span[1])


def representative_sources_from_plan(pi, X0_metric, X1_metric, quantiles=(0.50, 0.75, 0.90, 0.95), seed: int = 217, n_sample: int = 8192) -> pd.DataFrame:
    idx0, idx1 = sample_from_plan(pi, int(n_sample), seed=int(seed))
    distances = np.linalg.norm(np.asarray(X1_metric)[idx1] - np.asarray(X0_metric)[idx0], axis=1)
    rows = []
    used_sources = set()
    used_pairs = set()
    for q in quantiles:
        target_distance = float(np.quantile(distances, float(q)))
        candidates = np.argsort(np.abs(distances - target_distance))
        chosen = None
        for k in candidates:
            key = (int(idx0[k]), int(idx1[k]))
            if int(idx0[k]) not in used_sources and key not in used_pairs:
                chosen = int(k)
                break
        if chosen is None:
            chosen = int(candidates[0])
        used_sources.add(int(idx0[chosen]))
        used_pairs.add((int(idx0[chosen]), int(idx1[chosen])))
        rows.append(
            {
                "quantile": float(q),
                "source_idx": int(idx0[chosen]),
                "sampled_target_idx": int(idx1[chosen]),
                "sampled_metric_distance": float(distances[chosen]),
            }
        )
    return pd.DataFrame(rows)


def highest_mass_targets(pi, source_idx):
    pi = np.asarray(pi, dtype=float)
    return np.array([int(np.argmax(pi[int(i)])) for i in source_idx], dtype=int)


def plot_representative_endpoint_pairs(
    display0,
    display1,
    pair_table,
    *,
    fig_dir: str | Path,
    filename: str,
    title: str,
    metric_note: str,
    line_color: str,
    xlim=None,
    ylim=None,
    source_label: str = "source cells",
    target_label: str = "target cells",
    palette: Mapping[str, str] = CH04_PALETTE,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    display0 = np.asarray(display0, dtype=float)
    display1 = np.asarray(display1, dtype=float)
    ax.scatter(display0[:, 0], display0[:, 1], s=8, color=palette["source"], alpha=0.10, linewidths=0, label=source_label)
    ax.scatter(display1[:, 0], display1[:, 1], s=8, color=palette["target"], alpha=0.10, linewidths=0, label=target_label)
    for n, row in pd.DataFrame(pair_table).reset_index(drop=True).iterrows():
        start = display0[int(row["source_idx"])]
        end = display1[int(row["target_idx"])]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=line_color, alpha=0.88, linewidth=1.45, label="representative endpoint link" if n == 0 else None)
        ax.scatter(start[0], start[1], s=42, marker="o", facecolor="white", edgecolor=palette["source"], linewidth=1.0, zorder=4)
        ax.scatter(end[0], end[1], s=48, marker="^", facecolor="white", edgecolor=palette["target"], linewidth=1.0, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("toy state 1" if "toy" in filename else "PHATE 1 (display only)")
    ax.set_ylabel("toy state 2" if "toy" in filename else "PHATE 2 (display only)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.02, 0.02, metric_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.25")
    ax.legend(frameon=False, loc="upper right", handlelength=1.5, borderpad=0.2)
    return save_small_figure(fig, fig_dir, filename)


def load_eb_off_manifold_differences(out_dir: str | Path) -> pd.DataFrame:
    stats = pd.read_csv(Path(out_dir) / "exp8_eb_off_manifold_stats.csv")
    cols = [
        "pair_id",
        "t",
        "straight_minus_graph_density_percentile",
        "straight_minus_graph_knn_radius",
        "selected_for_main_figure",
        "connected_pair",
        "used_fallback",
    ]
    return stats[cols].drop_duplicates(["pair_id", "t"]).copy()


def plot_delta_distribution(
    diff,
    value_col: str,
    ylabel: str,
    *,
    fig_dir: str | Path,
    filename: str,
    mean_ndigits: int = 2,
    t_values: Iterable[float] = (0.25, 0.50, 0.75),
    palette: Mapping[str, str] = CH04_PALETTE,
) -> Path:
    import matplotlib.pyplot as plt

    diff = pd.DataFrame(diff)
    t_values = tuple(float(t) for t in t_values)
    data = [diff.loc[np.isclose(diff["t"], t), value_col].to_numpy(dtype=float) for t in t_values]
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    bp = ax.boxplot(data, positions=np.arange(len(t_values)), widths=0.48, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#D6E4F0")
        patch.set_edgecolor("0.25")
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("0.10")
        median.set_linewidth(1.2)
    rng = np.random.default_rng(431)
    mean_values = []
    for x, vals in enumerate(data):
        shown = vals if len(vals) <= 160 else rng.choice(vals, size=160, replace=False)
        jitter = rng.normal(0.0, 0.035, size=len(shown))
        ax.scatter(np.full(len(shown), x) + jitter, shown, s=8, color="0.35", alpha=0.25, linewidths=0, zorder=2)
        mean_val = float(np.mean(vals))
        mean_values.append(mean_val)
        ax.scatter([x], [mean_val], s=42, marker="D", color=palette["diagnostic"], edgecolor="white", linewidth=0.5, zorder=4)
    ax.axhline(0, color="0.15", linewidth=0.9, linestyle="--")
    ax.set_xticks(np.arange(len(t_values)), [str(t) for t in t_values])
    ax.set_xlabel("intermediate time t", labelpad=24)
    ax.set_ylabel(ylabel)
    ax.set_title("Straight chord minus graph path")
    ax.text(0.02, 0.98, "positive: chord farther from observed PC-20 support", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="0.25")
    mean_row_y = -0.19
    ax.text(
        -0.52,
        mean_row_y,
        "mean:",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
        clip_on=False,
    )
    for x, mean_val in enumerate(mean_values):
        ax.text(
            x,
            mean_row_y,
            f"{mean_val:.{int(mean_ndigits)}f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color=palette["diagnostic"],
            clip_on=False,
        )
    ax.scatter([], [], s=42, marker="D", color=palette["diagnostic"], edgecolor="white", linewidth=0.5, label="mean")
    ax.legend(frameon=False, loc="upper right", fontsize=7, handletextpad=0.3)
    ax.grid(axis="y", alpha=0.18)
    fig.subplots_adjust(bottom=0.30)
    return save_small_figure(fig, fig_dir, filename)


def choose_eb_path_example_ids(out_dir: str | Path, max_examples: int = 4) -> np.ndarray:
    out_dir = Path(out_dir)
    diff = load_eb_off_manifold_differences(out_dir)
    diag = pd.read_csv(out_dir / "exp8_eb_pair_selection_diagnostics.csv")
    pool = diff.merge(
        diag[["pair_id", "selected_for_main_figure", "connected_pair", "used_fallback"]],
        on="pair_id",
        how="left",
        suffixes=("", "_diag"),
    )
    for col in ["selected_for_main_figure", "connected_pair", "used_fallback"]:
        diag_col = f"{col}_diag"
        if diag_col in pool:
            pool[col] = pool[col].fillna(pool[diag_col])
    pool = pool[(pool["connected_pair"].astype(bool)) & (~pool["used_fallback"].astype(bool))]
    selected = pool[pool["selected_for_main_figure"].astype(bool)].copy()
    if selected.empty:
        selected = pool.copy()
    ranked = selected.groupby("pair_id").agg(
        mean_density_delta=("straight_minus_graph_density_percentile", "mean"),
        mean_knn_delta=("straight_minus_graph_knn_radius", "mean"),
    ).reset_index()
    ranked = ranked.sort_values(["mean_density_delta", "mean_knn_delta", "pair_id"], ascending=[False, False, True])
    return ranked.head(int(max_examples))["pair_id"].to_numpy(dtype=int)


def reconstruct_eb_graph_paths(pair_rows, reference_pc20, X0_pc, X1_pc, *, default_k_graph: int = 30) -> dict[int, np.ndarray]:
    from scipy.sparse.csgraph import dijkstra
    from sklearn.neighbors import kneighbors_graph

    pair_rows = pd.DataFrame(pair_rows).copy()
    if "selected_k_graph" in pair_rows and pair_rows["selected_k_graph"].notna().any():
        k_graph = int(pair_rows["selected_k_graph"].dropna().iloc[0])
    else:
        k_graph = int(default_k_graph)
    reference_pc20 = np.asarray(reference_pc20, dtype=np.float32)
    graph = kneighbors_graph(reference_pc20, n_neighbors=k_graph, mode="distance", include_self=False)
    graph = graph.maximum(graph.T).tocsr()
    source_nodes = pair_rows["source_global_node"].to_numpy(dtype=int)
    unique_sources, inverse_sources = np.unique(source_nodes, return_inverse=True)
    dist_matrix, predecessors = dijkstra(graph, directed=False, indices=unique_sources, return_predecessors=True)
    if dist_matrix.ndim == 1:
        dist_matrix = dist_matrix[None, :]
        predecessors = predecessors[None, :]
    paths = {}
    for local_row, (_, row) in zip(inverse_sources, pair_rows.iterrows()):
        pair_id = int(row["pair_id"])
        source_node = int(row["source_global_node"])
        target_node = int(row["target_global_node"])
        dist_value = float(dist_matrix[int(local_row), target_node])
        path_idx, used_fallback = extract_path_indices(predecessors[int(local_row)], source_node, target_node, dist_value, len(reference_pc20))
        if used_fallback or len(path_idx) < 2:
            start = np.asarray(X0_pc[int(row["source_idx"])], dtype=np.float32)
            end = np.asarray(X1_pc[int(row["target_idx"])], dtype=np.float32)
            paths[pair_id] = np.vstack([start, end]).astype(np.float32)
        else:
            paths[pair_id] = reference_pc20[path_idx].astype(np.float32)
    return paths
