from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .artifacts import sample_rows
from .ch04_tutorial_data import CH04_PALETTE, METHOD_COLORS, METHOD_LABELS, REP_PAIR_QUANTILES, REP_TRAJ_QUANTILES, display_ch04_figure, save_ch04_figure

def method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method).replace("_", " "))


def method_color(method: str) -> str:
    return METHOD_COLORS.get(str(method), "0.35")

def phate_limits(*arrays, pad_fraction: float = 0.055):
    pts = np.vstack([np.asarray(a, dtype=float)[:, :2] for a in arrays])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = span * float(pad_fraction)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def apply_phate_axes(ax, limits, xlabel: bool = True, ylabel: bool = True):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PHATE 1 (display only)" if xlabel else "")
    ax.set_ylabel("PHATE 2 (display only)" if ylabel else "")


def draw_eb_background(
    ax,
    X0_phate,
    X1_phate,
    *,
    limits=None,
    palette: Mapping[str, str] = CH04_PALETTE,
    source_alpha: float = 0.10,
    target_alpha: float = 0.10,
):
    ax.scatter(X0_phate[:, 0], X0_phate[:, 1], s=8, color=palette["source"], alpha=source_alpha, linewidths=0)
    ax.scatter(X1_phate[:, 0], X1_phate[:, 1], s=8, color=palette["target"], alpha=target_alpha, linewidths=0)
    if limits is not None:
        apply_phate_axes(ax, limits)


def select_representatives_by_quantile(values, quantiles=REP_PAIR_QUANTILES):
    values = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        raise ValueError("No finite values available for representative selection")
    used = set()
    rows = []
    for q in quantiles:
        target = float(np.quantile(values[finite], q))
        order = finite[np.argsort(np.abs(values[finite] - target), kind="mergesort")]
        pick = next((int(i) for i in order if int(i) not in used), int(order[0]))
        used.add(pick)
        rows.append({"row_index": pick, "quantile": float(q), "value": float(values[pick]), "target": target})
    return pd.DataFrame(rows)


def selected_pair_frame(pair_frame, quantiles=REP_PAIR_QUANTILES):
    reps = select_representatives_by_quantile(pair_frame["pc20_chord_length"].to_numpy(), quantiles=quantiles)
    selected = pair_frame.iloc[reps["row_index"].to_numpy()].copy().reset_index(drop=True)
    selected["selected_quantile"] = reps["quantile"].to_numpy()
    selected["selection_target_pc20_chord_length"] = reps["target"].to_numpy()
    return selected


def _display_table(frame):
    from IPython.display import display

    display(frame)


def plot_pair_panels(
    X0_phate,
    X1_phate,
    panels,
    *,
    fig_dir: str | Path,
    filename: str,
    title: str,
    value_label: str = "PC-20 chord length",
    palette: Mapping[str, str] = CH04_PALETTE,
    seed: int = 42,
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2), squeeze=False)
    axes_flat = axes[0]
    all_values = [np.asarray(panel["values"], dtype=float) for panel in panels if panel.get("values") is not None]
    norm = None
    if all_values:
        finite = np.concatenate([v[np.isfinite(v)] for v in all_values if np.any(np.isfinite(v))])
        norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max())) if finite.size else None
    for ax, panel in zip(axes_flat, panels):
        idx0, idx1 = np.asarray(panel["idx0"], dtype=int), np.asarray(panel["idx1"], dtype=int)
        ax.scatter(X0_phate[:, 0], X0_phate[:, 1], s=8, color=palette["source"], alpha=0.20, linewidths=0)
        ax.scatter(X1_phate[:, 0], X1_phate[:, 1], s=8, color=palette["target"], alpha=0.20, linewidths=0)
        keep = sample_rows(len(idx0), min(panel.get("max_lines", 100), len(idx0)), seed=panel.get("seed", seed))
        segments = np.stack([X0_phate[idx0[keep]], X1_phate[idx1[keep]]], axis=1)
        values = panel.get("values")
        if values is not None and norm is not None:
            lc = LineCollection(segments, cmap="viridis", norm=norm, linewidths=0.8, alpha=0.55)
            lc.set_array(np.asarray(values, dtype=float)[keep])
            ax.add_collection(lc)
        else:
            ax.add_collection(LineCollection(segments, colors=panel.get("color", "0.45"), linewidths=0.7, alpha=0.25))
        ax.set_title(panel["title"])
        ax.set_xlabel("PHATE 1")
        ax.set_ylabel("PHATE 2")
    if norm is not None:
        fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes_flat), fraction=0.035, pad=0.02, label=value_label)
    fig.suptitle(title, y=1.02)
    return save_ch04_figure(fig, fig_dir, filename)


def plot_phate_pairs(X0_phate, X1_phate, idx0, idx1, *, fig_dir: str | Path, title: str, max_lines: int = 120, seed: int = 42, values=None):
    return plot_pair_panels(
        X0_phate,
        X1_phate,
        [{"title": title, "idx0": idx0, "idx1": idx1, "values": values, "seed": seed, "max_lines": max_lines}],
        fig_dir=fig_dir,
        filename="_temporary_pair_panel.png",
        title=title,
        seed=seed,
    )


def add_local_arrows(ax, projected_traj, observed_phate, color, max_arrows: int = 28, seed: int = 42):
    from sklearn.neighbors import NearestNeighbors

    projected_traj = np.asarray(projected_traj, dtype=float)
    observed_phate = np.asarray(observed_phate, dtype=float)
    if projected_traj.shape[0] < 3 or projected_traj.shape[-1] != 2:
        return
    mid_step = projected_traj.shape[0] // 2
    anchors = projected_traj[mid_step]
    deltas = projected_traj[min(mid_step + 1, projected_traj.shape[0] - 1)] - projected_traj[max(mid_step - 1, 0)]
    nn_obs = NearestNeighbors(n_neighbors=min(15, len(observed_phate))).fit(observed_phate)
    obs_r = nn_obs.kneighbors(observed_phate, return_distance=True)[0][:, -1]
    dist = nn_obs.kneighbors(anchors, return_distance=True)[0][:, 0]
    valid = np.flatnonzero(dist <= float(np.quantile(obs_r, 0.75)))
    if valid.size == 0:
        return
    valid = valid[sample_rows(len(valid), min(max_arrows, len(valid)), seed=seed)]
    ax.quiver(
        anchors[valid, 0],
        anchors[valid, 1],
        deltas[valid, 0],
        deltas[valid, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        color=color,
        alpha=0.75,
    )


def plot_projected_trajectories(
    paths,
    X0_phate,
    X1_phate,
    pc_to_phate,
    *,
    fig_dir: str | Path,
    filename: str,
    title: str,
    palette: Mapping[str, str] = CH04_PALETTE,
    max_lines: int = 45,
    local_arrows: bool = True,
    seed: int = 42,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(paths), figsize=(5.0 * len(paths), 4.2), squeeze=False)
    observed_phate = np.vstack([X0_phate, X1_phate])
    for ax, (name, traj) in zip(axes[0], paths.items()):
        ax.scatter(X0_phate[:, 0], X0_phate[:, 1], s=8, color=palette["source"], alpha=0.18, linewidths=0)
        ax.scatter(X1_phate[:, 0], X1_phate[:, 1], s=8, color=palette["target"], alpha=0.16, linewidths=0)
        keep = sample_rows(traj.shape[1], min(max_lines, traj.shape[1]), seed=seed)
        selected = np.asarray(traj[:, keep, :], dtype=np.float32)
        ph = pc_to_phate(selected.reshape(-1, selected.shape[-1])).reshape(selected.shape[0], selected.shape[1], 2)
        color = palette.get(name, "0.35")
        for j in range(ph.shape[1]):
            ax.plot(ph[:, j, 0], ph[:, j, 1], color=color, alpha=0.55, linewidth=1.0)
        if local_arrows:
            add_local_arrows(ax, ph, observed_phate, color=color, seed=seed + 7)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("PHATE 1 (display only)")
        ax.set_ylabel("PHATE 2 (display only)")
    fig.suptitle(title, y=1.02)
    return save_ch04_figure(fig, fig_dir, filename)


def plot_metric_lines(table, *, fig_dir: str | Path, x: str, y: str, hue: str, filename: str, title: str, nfe_grid: Iterable[int] = (4, 8, 16, 32, 64)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for name, group in table.groupby(hue):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", linewidth=1.5, label=str(name).replace("_", " "))
    ax.set_xscale("log", base=2 if set(table[x]).issubset(set(nfe_grid)) else 10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(frameon=False)
    return save_ch04_figure(fig, fig_dir, filename)


def plot_heatmap(matrix, *, fig_dir: str | Path, title: str, filename: str, max_size: int = 120, cmap: str = "viridis", seed: int = 42):
    import matplotlib.pyplot as plt

    M = np.asarray(matrix)
    rows = sample_rows(M.shape[0], min(max_size, M.shape[0]), seed=seed)
    cols = sample_rows(M.shape[1], min(max_size, M.shape[1]), seed=seed + 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M[np.ix_(rows, cols)], aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("target subset")
    ax.set_ylabel("source subset")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_ch04_figure(fig, fig_dir, filename)


def plot_table_image(frame: pd.DataFrame, *, fig_dir: str | Path, filename: str, title: str, max_rows: int = 12):
    import matplotlib.pyplot as plt

    shown = frame.head(max_rows).copy()
    fig, ax = plt.subplots(figsize=(min(14, 1.5 + 1.4 * shown.shape[1]), 0.8 + 0.35 * len(shown)))
    ax.axis("off")
    ax.set_title(title, loc="left")
    tbl = ax.table(cellText=shown.round(4).astype(str).values, colLabels=shown.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.25)
    return save_ch04_figure(fig, fig_dir, filename)


def plot_representative_pairs(pair_frame, X0_phate, X1_phate, *, fig_dir: str | Path, filename: str, title: str, color: str, limits):
    import matplotlib.pyplot as plt

    selected = selected_pair_frame(pair_frame)
    fig, ax = plt.subplots(figsize=(4.35, 4.05))
    draw_eb_background(ax, X0_phate, X1_phate, limits=limits, source_alpha=0.10, target_alpha=0.10)
    for _, row in selected.iterrows():
        p0 = X0_phate[int(row["idx0"])]
        p1 = X1_phate[int(row["idx1"])]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=1.9, alpha=0.88, solid_capstyle="round")
    idx0 = selected["idx0"].astype(int).to_numpy()
    idx1 = selected["idx1"].astype(int).to_numpy()
    ax.scatter(X0_phate[idx0, 0], X0_phate[idx0, 1], s=58, color=CH04_PALETTE["source"], edgecolor="white", linewidth=0.8, zorder=4, label="selected source")
    ax.scatter(X1_phate[idx1, 0], X1_phate[idx1, 1], s=72, marker="^", color=CH04_PALETTE["target"], edgecolor="white", linewidth=0.8, zorder=5, label="selected target")
    ax.plot([], [], color=color, linewidth=1.9, label="endpoint chord")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderpad=0.2)
    path = save_ch04_figure(fig, fig_dir, filename)
    display_ch04_figure(fig_dir, filename, width=620)
    _display_table(selected[["idx0", "idx1", "pc20_chord_length", "selected_quantile"]].round(4))
    return path, selected


def plot_chord_length_ecdf(pair_stats_by_method, metrics_table, *, fig_dir: str | Path, filename: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.75))
    for method in ["random_cfm", "ot_cfm"]:
        vals = np.asarray(pair_stats_by_method[method]["pc20_chord_length"], dtype=float)
        vals = np.sort(vals[np.isfinite(vals)])
        y = np.arange(1, vals.size + 1) / float(vals.size)
        color = method_color(method)
        row = metrics_table.loc[metrics_table["method"] == method].iloc[0]
        mean = float(row["mean_pc20_chord_length"])
        median = float(row["median_pc20_chord_length"])
        ax.plot(vals, y, color=color, linewidth=2.0, label=f"{method_label(method)} mean {mean:.2f}; median {median:.2f}")
        ax.axvline(median, color=color, linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_title("Endpoint chord length distribution")
    ax.set_xlabel("Endpoint chord length in standardized PC-20")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    path = save_ch04_figure(fig, fig_dir, filename)
    display_ch04_figure(fig_dir, filename, width=650)
    _display_table(metrics_table[["method", "mean_pc20_chord_length", "median_pc20_chord_length", "std_pc20_chord_length", "training_space"]].round(4))
    return path


def rollout_representative_indices(reference_traj, quantiles=REP_TRAJ_QUANTILES):
    displacement = np.linalg.norm(np.asarray(reference_traj[-1] - reference_traj[0], dtype=float), axis=1)
    reps = select_representatives_by_quantile(displacement, quantiles=quantiles)
    return reps["row_index"].astype(int).to_numpy(), reps


def project_selected_trajectory(traj, selected_idx, pc_to_phate_fn):
    selected = np.asarray(traj[:, selected_idx, :], dtype=np.float32)
    return pc_to_phate_fn(selected.reshape(-1, selected.shape[-1])).reshape(selected.shape[0], selected.shape[1], 2)


def plot_representative_rollouts(traj, selected_idx, X0_phate, X1_phate, pc_to_phate_fn, *, fig_dir: str | Path, filename: str, title: str, color: str, limits):
    import matplotlib.pyplot as plt

    ph = project_selected_trajectory(traj, selected_idx, pc_to_phate_fn)
    fig, ax = plt.subplots(figsize=(4.35, 4.05))
    draw_eb_background(ax, X0_phate, X1_phate, limits=limits, source_alpha=0.09, target_alpha=0.09)
    for j in range(ph.shape[1]):
        ax.plot(ph[:, j, 0], ph[:, j, 1], color=color, alpha=0.86, linewidth=2.0, solid_capstyle="round")
        start = max(0, ph.shape[0] - 6)
        ax.annotate("", xy=ph[-1, j], xytext=ph[start, j], arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 10})
    ax.scatter(ph[0, :, 0], ph[0, :, 1], s=58, color=CH04_PALETTE["source"], edgecolor="white", linewidth=0.8, zorder=4, label="source")
    ax.scatter(ph[-1, :, 0], ph[-1, :, 1], s=70, marker="X", color=color, edgecolor="white", linewidth=0.8, zorder=5, label="generated endpoint")
    ax.scatter([], [], s=28, color=CH04_PALETTE["target"], alpha=0.55, linewidths=0, label="EB target background")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, borderpad=0.2)
    path = save_ch04_figure(fig, fig_dir, filename)
    display_ch04_figure(fig_dir, filename, width=620)
    return path


def observed_near_velocity_indices(reference_traj, pc_to_phate_fn, observed_phate, n_arrows: int = 6, seed: int = 42):
    from sklearn.neighbors import NearestNeighbors

    mid = reference_traj.shape[0] // 2
    anchors = pc_to_phate_fn(reference_traj[mid].astype(np.float32))
    nn_obs = NearestNeighbors(n_neighbors=min(15, len(observed_phate))).fit(observed_phate)
    obs_r = nn_obs.kneighbors(observed_phate, return_distance=True)[0][:, -1]
    dist = nn_obs.kneighbors(anchors, return_distance=True)[0][:, 0]
    valid = np.flatnonzero(dist <= float(np.quantile(obs_r, 0.85)))
    if valid.size == 0:
        valid = np.arange(reference_traj.shape[1])
    disp = np.linalg.norm(reference_traj[-1] - reference_traj[0], axis=1)
    reps = select_representatives_by_quantile(disp[valid], quantiles=np.linspace(0.25, 0.90, int(n_arrows)))
    return valid[reps["row_index"].astype(int).to_numpy()]


def plot_local_velocity_examples(trajs, selected_idx, X0_phate, X1_phate, pc_to_phate_fn, *, fig_dir: str | Path, filename: str, limits):
    import matplotlib.pyplot as plt

    methods = [("random_cfm", trajs["random_cfm"]), ("ot_cfm", trajs["ot_cfm"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.85), sharex=True, sharey=True)
    for ax, (method, traj) in zip(axes, methods):
        color = method_color(method)
        draw_eb_background(ax, X0_phate, X1_phate, limits=limits, source_alpha=0.08, target_alpha=0.08)
        mid = traj.shape[0] // 2
        step = min(mid + 3, traj.shape[0] - 1)
        anchors = pc_to_phate_fn(traj[mid, selected_idx, :].astype(np.float32))
        next_points = pc_to_phate_fn(traj[step, selected_idx, :].astype(np.float32))
        delta = next_points - anchors
        ax.quiver(anchors[:, 0], anchors[:, 1], delta[:, 0] * 2.0, delta[:, 1] * 2.0, angles="xy", scale_units="xy", scale=1.0, width=0.0065, color=color, alpha=0.9)
        ax.scatter(anchors[:, 0], anchors[:, 1], s=40, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.set_title(f"{method_label(method)} local arrows")
    axes[0].set_ylabel("PHATE 2 (display only)")
    axes[1].set_ylabel("")
    path = save_ch04_figure(fig, fig_dir, filename)
    display_ch04_figure(fig_dir, filename, width=850)
    return path


def plot_reflow_representative_trajectories(trajs, selected_idx, X0_phate, X1_phate, pc_to_phate_fn, *, fig_dir: str | Path, filename: str, limits):
    import matplotlib.pyplot as plt

    methods = [("ot_cfm", "OT-CFM"), ("reflow_1", "Reflow 1"), ("reflow_2", "Reflow 2")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.85), sharex=True, sharey=True)
    for ax, (method, title) in zip(axes, methods):
        color = method_color(method)
        draw_eb_background(ax, X0_phate, X1_phate, limits=limits, source_alpha=0.075, target_alpha=0.075)
        ph = project_selected_trajectory(trajs[method], selected_idx, pc_to_phate_fn)
        for j in range(ph.shape[1]):
            ax.plot(ph[:, j, 0], ph[:, j, 1], color=color, linewidth=1.9, alpha=0.86, solid_capstyle="round")
            start = max(0, ph.shape[0] - 6)
            ax.annotate("", xy=ph[-1, j], xytext=ph[start, j], arrowprops={"arrowstyle": "->", "color": color, "lw": 1.45, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9})
        ax.scatter(ph[0, :, 0], ph[0, :, 1], s=45, color=CH04_PALETTE["source"], edgecolor="white", linewidth=0.7, zorder=4)
        ax.scatter(ph[-1, :, 0], ph[-1, :, 1], s=58, marker="X", color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.set_title(title)
    for ax in axes[1:]:
        ax.set_ylabel("")
    path = save_ch04_figure(fig, fig_dir, filename)
    display_ch04_figure(fig_dir, filename, width=980)
    return path


def plot_metric_bar_grid(table, methods, metric_specs, *, fig_dir: str | Path, filename: str, title: str | None = None, log_metrics=None):
    import matplotlib.pyplot as plt

    log_metrics = set(log_metrics or [])
    plot_table = table.set_index("method").loc[list(methods)].reset_index()
    ncols = min(4, len(metric_specs))
    nrows = int(math.ceil(len(metric_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), squeeze=False)
    for ax, (metric, label, ylabel) in zip(axes.ravel(), metric_specs):
        vals = plot_table[metric].astype(float).to_numpy()
        colors = [method_color(m) for m in plot_table["method"]]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, alpha=0.88, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels([method_label(m) for m in plot_table["method"]], rotation=25, ha="right")
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        if metric in log_metrics:
            ax.set_yscale("log")
        finite_vals = vals[np.isfinite(vals) & ((vals > 0) if metric in log_metrics else np.isfinite(vals))]
        if finite_vals.size:
            ymax = float(np.nanmax(vals))
            if metric in log_metrics:
                ax.set_ylim(max(float(np.nanmin(finite_vals)) * 0.55, 1e-5), ymax * 1.8)
            else:
                ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
        for xi, yi in zip(x, vals):
            if np.isfinite(yi):
                ax.text(xi, yi * (1.10 if metric in log_metrics else 1.025), f"{yi:.3g}", ha="center", va="bottom", fontsize=7)
    for ax in axes.ravel()[len(metric_specs) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    path = save_ch04_figure(fig, fig_dir, filename)
    display_ch04_figure(fig_dir, filename, width=950)
    _display_table(plot_table[["method"] + [m[0] for m in metric_specs]].round(4))
    return path
