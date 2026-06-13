from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .artifacts import save_figure_formats
from .ch04_tutorial_data import CH04_PALETTE
from .graph_paths import (
    build_endpoint_knn_graph_grid,
    extract_path_indices,
    knn_density_scorer,
    point_on_polyline,
    polyline_length,
    resample_polyline,
)
from .ot import compute_cost_matrix, sample_from_plan, sinkhorn_plan


DEFAULT_EXP8_T_VALUES = (0.25, 0.50, 0.75)


def density_scorer(reference_points, k: int = 15):
    return knn_density_scorer(reference_points, k=k)


def candidate_path_diagnostics(
    source_points,
    target_points,
    pi,
    graph_points,
    graph,
    score_fn,
    n_candidates: int,
    seed: int = 90,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    from scipy.sparse.csgraph import dijkstra
    from sklearn.neighbors import NearestNeighbors

    idx0, idx1 = sample_from_plan(pi, int(n_candidates), seed=seed)
    nearest_graph = NearestNeighbors(n_neighbors=1).fit(graph_points)
    source_nodes = nearest_graph.kneighbors(np.asarray(source_points)[idx0], return_distance=False)[:, 0].astype(int)
    target_nodes = nearest_graph.kneighbors(np.asarray(target_points)[idx1], return_distance=False)[:, 0].astype(int)
    unique_sources, inverse_sources = np.unique(source_nodes, return_inverse=True)
    dist_matrix, predecessors = dijkstra(graph, directed=False, indices=unique_sources, return_predecessors=True)
    if dist_matrix.ndim == 1:
        dist_matrix = dist_matrix[None, :]
        predecessors = predecessors[None, :]

    rows = []
    path_cache: dict[int, np.ndarray] = {}
    for pair_id, (i, j, s_node, e_node, src_row) in enumerate(zip(idx0, idx1, source_nodes, target_nodes, inverse_sources)):
        start = np.asarray(source_points[int(i)], dtype=np.float32)
        end = np.asarray(target_points[int(j)], dtype=np.float32)
        dist_value = float(dist_matrix[int(src_row), int(e_node)])
        path_idx, used_fallback = extract_path_indices(
            predecessors[int(src_row)], int(s_node), int(e_node), dist_value, len(graph_points)
        )
        graph_path_nodes = np.asarray(graph_points[path_idx], dtype=np.float32)
        if used_fallback or len(graph_path_nodes) < 2:
            graph_path_nodes = np.vstack([start, end]).astype(np.float32)
        graph_length = polyline_length(graph_path_nodes)
        straight_length = float(np.linalg.norm(end - start))
        straight_mid = 0.5 * (start + end)
        graph_mid = point_on_polyline(graph_path_nodes, 0.5)
        spct, srad = score_fn(straight_mid)
        gpct, grad = score_fn(graph_mid)
        path_cache[int(pair_id)] = graph_path_nodes
        rows.append(
            {
                "pair_id": int(pair_id),
                "source_idx": int(i),
                "target_idx": int(j),
                "source_graph_node": int(s_node),
                "target_graph_node": int(e_node),
                "euclidean_distance": straight_length,
                "graph_path_num_nodes": int(len(path_idx)),
                "graph_path_length": graph_length,
                "straight_path_length": straight_length,
                "path_length_ratio": graph_length / max(straight_length, 1e-12),
                "used_fallback": bool(used_fallback),
                "straight_mid_knn_radius": float(srad[0]),
                "graph_mid_knn_radius": float(grad[0]),
                "straight_minus_graph_mid_knn_radius": float(srad[0] - grad[0]),
                "straight_mid_density_percentile": float(spct[0]),
                "graph_mid_density_percentile": float(gpct[0]),
                "straight_minus_graph_density_percentile": float(spct[0] - gpct[0]),
            }
        )
    return pd.DataFrame(rows), path_cache


def select_diagnostic_pairs(candidate_diag, max_pairs: int = 18, min_pairs: int = 6) -> pd.DataFrame:
    diag = pd.DataFrame(candidate_diag).copy()
    diag["selection_score"] = (
        diag["straight_minus_graph_density_percentile"].rank(pct=True)
        + diag["straight_minus_graph_mid_knn_radius"].rank(pct=True)
        + diag["euclidean_distance"].rank(pct=True)
        + diag["path_length_ratio"].rank(pct=True)
    )
    attempts = [
        (0.50, True, "nonfallback_nodes_ge4_upper_half_positive_midpoint_difference"),
        (0.35, True, "relaxed_distance_quantile_positive_midpoint_difference"),
        (0.00, True, "positive_midpoint_difference_any_distance"),
        (0.35, False, "relaxed_best_nonfallback_nodes_ge4"),
        (0.00, False, "best_nonfallback_nodes_ge4"),
    ]
    selected = pd.DataFrame()
    selected_reason = "no_pairs_selected"
    for q, require_positive, reason in attempts:
        distance_threshold = float(diag["euclidean_distance"].quantile(q))
        mask = (~diag["used_fallback"]) & (diag["graph_path_num_nodes"] >= 4) & (diag["euclidean_distance"] >= distance_threshold)
        if require_positive:
            mask &= (diag["straight_minus_graph_mid_knn_radius"] > 0) | (diag["straight_minus_graph_density_percentile"] > 0)
        pool = diag.loc[mask].sort_values("selection_score", ascending=False)
        if len(pool) >= int(min_pairs):
            selected = pool.head(int(max_pairs)).copy()
            selected_reason = reason
            break
        if len(pool) > len(selected):
            selected = pool.copy()
            selected_reason = reason + "_too_few"
    if selected.empty:
        selected = diag.loc[~diag["used_fallback"]].sort_values("selection_score", ascending=False).head(int(max_pairs)).copy()
        selected_reason = "last_resort_nonfallback_selection"
    selected["selected_reason"] = selected_reason
    selected["selected_rank"] = np.arange(len(selected), dtype=int)
    return selected


def stats_for_selected_pairs(
    selected_diag,
    source_points,
    target_points,
    path_cache: Mapping[int, np.ndarray],
    score_fn,
    t_values: Iterable[float] = DEFAULT_EXP8_T_VALUES,
) -> pd.DataFrame:
    stat_rows = []
    for _, row in pd.DataFrame(selected_diag).iterrows():
        pair_id = int(row["pair_id"])
        source_idx = int(row["source_idx"])
        target_idx = int(row["target_idx"])
        start = np.asarray(source_points[source_idx], dtype=np.float32)
        end = np.asarray(target_points[target_idx], dtype=np.float32)
        graph_path_nodes = np.asarray(path_cache[pair_id], dtype=np.float32)
        for tval in t_values:
            tval = float(tval)
            straight_point = (1.0 - tval) * start + tval * end
            graph_point = point_on_polyline(graph_path_nodes, tval)
            spct, srad = score_fn(straight_point)
            gpct, grad = score_fn(graph_point)
            common = {
                "pair_id": pair_id,
                "source_idx": source_idx,
                "target_idx": target_idx,
                "graph_path_num_nodes": int(row["graph_path_num_nodes"]),
                "used_fallback": bool(row["used_fallback"]),
                "selected_reason": str(row["selected_reason"]),
                "t": tval,
            }
            stat_rows.append({**common, "path_type": "straight_chord", "density_radius_percentile": float(spct[0]), "knn_radius": float(srad[0])})
            stat_rows.append({**common, "path_type": "knn_graph_shortest_path", "density_radius_percentile": float(gpct[0]), "knn_radius": float(grad[0])})
    return pd.DataFrame(stat_rows)


def paired_path_differences(stats) -> pd.DataFrame:
    wide = pd.DataFrame(stats).pivot_table(index=["pair_id", "t"], columns="path_type", values=["density_radius_percentile", "knn_radius"])
    out = pd.DataFrame(index=wide.index)
    out["straight_minus_graph_density_percentile"] = wide[("density_radius_percentile", "straight_chord")] - wide[("density_radius_percentile", "knn_graph_shortest_path")]
    out["straight_minus_graph_knn_radius"] = wide[("knn_radius", "straight_chord")] - wide[("knn_radius", "knn_graph_shortest_path")]
    return out.reset_index()


def _save_single_figure(fig, fig_dir: str | Path, filename: str) -> Path:
    import matplotlib.pyplot as plt

    paths = save_figure_formats(fig, fig_dir, Path(filename).stem, formats=(Path(filename).suffix.lstrip(".") or "png",), dpi=220, close=True)
    plt.close(fig)
    return paths[0]


def _grouped_path_boxplots(axes, stats, t_values, colors, *, eb_labels: bool = False):
    import matplotlib.pyplot as plt

    path_order = ["straight_chord", "knn_graph_shortest_path"]

    def grouped_boxplot(ax, metric, ylabel, title):
        positions, data, facecolors, tick_positions = [], [], [], []
        for ti, tval in enumerate(t_values):
            center = ti * 3.0
            tick_positions.append(center)
            vals_by_type = {}
            for offset, path_type in [(-0.42, path_order[0]), (0.42, path_order[1])]:
                sub = stats[(stats["t"] == float(tval)) & (stats["path_type"] == path_type)].sort_values("pair_id")
                vals = sub[metric].to_numpy()
                vals_by_type[path_type] = sub[["pair_id", metric]].set_index("pair_id")[metric]
                positions.append(center + offset)
                data.append(vals)
                facecolors.append(colors[path_type])
            common_ids = vals_by_type[path_order[0]].index.intersection(vals_by_type[path_order[1]].index)
            for pid in common_ids:
                ax.plot(
                    [center - 0.42, center + 0.42],
                    [vals_by_type[path_order[0]].loc[pid], vals_by_type[path_order[1]].loc[pid]],
                    color="0.70", alpha=0.22 if eb_labels else 0.35, linewidth=0.55 if eb_labels else 0.7, zorder=1,
                )
        bp = ax.boxplot(data, positions=positions, widths=0.62, patch_artist=True, showfliers=False, zorder=2)
        for patch, color in zip(bp["boxes"], facecolors):
            patch.set_facecolor(color)
            patch.set_alpha(0.46 if eb_labels else 0.48)
        for median in bp["medians"]:
            median.set_color("0.12")
            median.set_linewidth(1.1)
        rng = np.random.default_rng(193 if eb_labels else 123)
        for pos, vals, color in zip(positions, data, facecolors):
            if len(vals) == 0:
                continue
            sample = vals if len(vals) <= 180 else rng.choice(vals, size=180, replace=False)
            jitter = rng.normal(scale=0.045, size=len(sample))
            ax.scatter(np.full(len(sample), pos) + jitter, sample, s=8 if eb_labels else 10, color=color, alpha=0.32 if eb_labels else 0.45, linewidths=0, zorder=3)
        ax.set_xticks(tick_positions, [str(t) for t in t_values])
        ax.set_xlabel("intermediate time t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.20)

    grouped_boxplot(axes[0], "density_radius_percentile", "kNN-radius percentile" if eb_labels else "density-radius percentile", "Panel A: higher = farther from EB support" if eb_labels else "Panel A: density percentile")
    grouped_boxplot(axes[1], "knn_radius", "20D PC kNN radius" if eb_labels else "kNN radius", "Panel B: local support distance" if eb_labels else "Panel B: kNN radius")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=colors[path_order[0]], label="straight chord", markersize=8, alpha=0.75),
        plt.Line2D([0], [0], marker="s", linestyle="", color=colors[path_order[1]], label="20D kNN graph path" if eb_labels else "kNN graph path", markersize=8, alpha=0.75),
    ]
    axes[1].legend(handles=handles, frameon=False, loc="best")


def plot_path_statistic_supplement(
    stats,
    *,
    fig_dir: str | Path,
    filename: str,
    title: str,
    t_values: Iterable[float] = DEFAULT_EXP8_T_VALUES,
    palette: Mapping[str, str] = CH04_PALETTE,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = {"straight_chord": palette["diagnostic"], "knn_graph_shortest_path": palette["program"]}
    _grouped_path_boxplots(axes, pd.DataFrame(stats), tuple(float(t) for t in t_values), colors, eb_labels=False)
    fig.suptitle(title, y=1.02)
    return _save_single_figure(fig, fig_dir, filename)


def plot_paths_2d(
    observed_points,
    source_points,
    target_points,
    selected_diag,
    path_cache: Mapping[int, np.ndarray],
    *,
    fig_dir: str | Path,
    filename: str,
    title: str,
    path_points: int = 41,
    t_values: Iterable[float] = DEFAULT_EXP8_T_VALUES,
    max_pairs: int = 16,
    palette: Mapping[str, str] = CH04_PALETTE,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(np.asarray(observed_points)[:, 0], np.asarray(observed_points)[:, 1], s=6, color="0.72", alpha=0.24, linewidths=0, label="observed toy states")
    shown = pd.DataFrame(selected_diag).sort_values("selected_rank").head(int(max_pairs))
    for n, (_, row) in enumerate(shown.iterrows()):
        pair_id = int(row["pair_id"])
        start = np.asarray(source_points[int(row["source_idx"])])
        end = np.asarray(target_points[int(row["target_idx"])])
        graph_path = resample_polyline(path_cache[pair_id], int(path_points))
        straight_path = np.linspace(start, end, int(path_points))
        show_label = n == 0
        ax.plot(straight_path[:, 0], straight_path[:, 1], color=palette["diagnostic"], alpha=0.35, linewidth=1.0, linestyle="-", label="straight chord" if show_label else None)
        ax.plot(graph_path[:, 0], graph_path[:, 1], color=palette["program"], alpha=0.70, linewidth=1.15, linestyle="--", label="kNN graph path" if show_label else None)
        for tval in t_values:
            tval = float(tval)
            spt = (1.0 - tval) * start + tval * end
            gpt = point_on_polyline(path_cache[pair_id], tval)
            ax.scatter(spt[0], spt[1], s=16, marker="o", facecolors="none", edgecolors=palette["diagnostic"], linewidths=0.8, alpha=0.65)
            ax.scatter(gpt[0], gpt[1], s=18, marker="x", color=palette["program"], linewidths=0.9, alpha=0.80)
    ax.set_title(title)
    ax.set_xlabel("toy state 1")
    ax.set_ylabel("toy state 2")
    ax.legend(frameon=False, loc="best")
    return _save_single_figure(fig, fig_dir, filename)


def build_endpoint_graph_grid(reference_points, source_nodes, target_nodes, k_grid):
    return build_endpoint_knn_graph_grid(reference_points, source_nodes, target_nodes, k_grid=k_grid)


def load_or_compute_ot_plan(coupling_path: str | Path, X0, X1, epsilon: float = 0.05) -> tuple[np.ndarray, str]:
    coupling_path = Path(coupling_path)
    if coupling_path.exists():
        z = np.load(coupling_path, allow_pickle=True)
        if "pi_ot" in z.files and z["pi_ot"].shape == (len(X0), len(X1)):
            return np.asarray(z["pi_ot"], dtype=float), "loaded_exp1_pc20_ot_coupling"
    C_tmp, _ = compute_cost_matrix(X0, X1, normalize=True)
    pi_tmp, _ = sinkhorn_plan(C_tmp, epsilon=float(epsilon), return_info=True)
    return np.asarray(pi_tmp, dtype=float), "computed_temporary_pc20_ot_coupling"


def save_figure_pair(fig, fig_dir: str | Path, png_name: str, pdf_name: str) -> tuple[Path, Path]:
    paths = save_figure_formats(fig, fig_dir, Path(png_name).stem, formats=("png",), dpi=220, close=False)
    pdf_paths = save_figure_formats(fig, fig_dir, Path(pdf_name).stem, formats=("pdf",), close=True)
    return paths[0], pdf_paths[0]


def plot_eb_path_statistics(
    stats,
    *,
    fig_dir: str | Path,
    filename_png: str,
    filename_pdf: str,
    t_values: Iterable[float] = DEFAULT_EXP8_T_VALUES,
    palette: Mapping[str, str] = CH04_PALETTE,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    connected = pd.DataFrame(stats)
    connected = connected[~connected["used_fallback"].astype(bool)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharex=True)
    colors = {"straight_chord": palette["diagnostic"], "knn_graph_shortest_path": palette["program"]}
    _grouped_path_boxplots(axes, connected, tuple(float(t) for t in t_values), colors, eb_labels=True)
    fig.suptitle("EB 20D off-manifold diagnostic; PHATE is not used for metrics", y=1.02)
    return save_figure_pair(fig, fig_dir, filename_png, filename_pdf)
