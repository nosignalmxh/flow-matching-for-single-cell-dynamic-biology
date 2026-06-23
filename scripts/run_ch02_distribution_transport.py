from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_ch02")


EXPECTED_FIGURE_STEMS = [
    "fig02_02_static_ot_endpoint_transport",
    "fig02_03_same_endpoints_different_paths",
    "fig02_04_dynamic_ot_low_action",
    "fig02_06_training_bottleneck_timing",
]

EXPECTED_TABLES = [
    "table02_01_coupling_diagnostics.csv",
    "table02_02_path_diagnostics.csv",
    "table02_03_dynamic_ot_energy_proxy.csv",
    "table02_04_cnf_training_bottleneck.csv",
]

CONCEPT_BOUNDARIES = [
    "OT endpoint relation is model-implied, not observed lineage.",
    "Path energy is computed in PC-20.",
    "Dynamic OT panel is low-action intuition, not solved Benamou-Brenier.",
    "Solver-in-loop baseline is not full likelihood CNF.",
    "Chapter 2 stops at the CNF training bottleneck; Flow Matching training is deferred to Chapter 3.",
]


def _find_project_root() -> Path:
    script_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    for candidate in [
        cwd,
        cwd / "flow_matching_for_dynamic_biology",
        cwd.parent,
        script_root,
    ]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    return script_root


def _prepare_imports(project_root: Path) -> None:
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _save_fig_both(fig, fig_dir: Path, stem: str, artifact_base: Path) -> list[str]:
    import matplotlib.pyplot as plt

    from src.utils import savefig

    written: list[str] = []
    for suffix in [".png", ".svg"]:
        path = savefig(fig, fig_dir / f"{stem}{suffix}", dpi=300)
        written.append(_display_path(path, artifact_base))
    plt.close(fig)
    return written


def _save_table(table: pd.DataFrame, out_dir: Path, filename: str, artifact_base: Path) -> str:
    from src.utils import save_table

    path = save_table(table, out_dir / filename)
    return _display_path(path, artifact_base)


def _set_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )


def _sorted_time_labels(labels: np.ndarray) -> list[str]:
    values = np.unique(labels.astype(str)).tolist()

    def key(value: str):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(values, key=key)


def _subsample_indices(indices: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= int(max_n):
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=int(max_n), replace=False))


def _median_positive_scale(C: np.ndarray) -> float:
    positive = np.asarray(C, dtype=float)[np.asarray(C) > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _coupling_diagnostic_row(
    method: str,
    epsilon: float | None,
    pi: np.ndarray,
    C_raw: np.ndarray,
    C_norm: np.ndarray,
    info: dict | None,
    cost_scale: float,
) -> dict:
    from src.core.ot import coupling_diagnostics

    raw = coupling_diagnostics(pi, C_raw)
    norm = coupling_diagnostics(pi, C_norm)
    info = info or {}
    return {
        "method": method,
        "epsilon": np.nan if epsilon is None else float(epsilon),
        "expected_cost_raw": raw["expected_cost"],
        "expected_cost_normalized": norm["expected_cost"],
        "entropy": norm["entropy"],
        "effective_support": norm["effective_support"],
        "row_l1_error": norm["row_l1_error"],
        "col_l1_error": norm["col_l1_error"],
        "sinkhorn_converged": bool(info.get("sinkhorn_converged", False)) if epsilon is not None else np.nan,
        "sinkhorn_n_iter": int(info.get("n_iter", 0)) if epsilon is not None else np.nan,
        "sinkhorn_backend": str(info.get("backend", "independent")) if epsilon is not None else "independent",
        "cost_scale": float(cost_scale),
    }


def _draw_endpoint_cloud(ax, X0_plot: np.ndarray, X1_plot: np.ndarray) -> None:
    ax.scatter(X0_plot[:, 0], X0_plot[:, 1], s=9, c="#4267B2", alpha=0.35, linewidths=0, label="source")
    ax.scatter(X1_plot[:, 0], X1_plot[:, 1], s=9, c="#D55E00", alpha=0.35, linewidths=0, label="target")


def _draw_arrows(
    ax,
    X0_plot: np.ndarray,
    X1_plot: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    color: str,
    alpha: float = 0.35,
    width: float = 0.002,
    linewidth: float | None = None,
    mutation_scale: float = 7.0,
) -> None:
    lw = float(linewidth if linewidth is not None else max(width * 400.0, 0.6))
    for src_idx, tgt_idx in zip(i0, i1):
        start = X0_plot[int(src_idx)]
        end = X1_plot[int(tgt_idx)]
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "alpha": float(alpha),
                "lw": lw,
                "mutation_scale": float(mutation_scale),
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=3,
        )


def _path_stats(traj: np.ndarray, tau_grid: np.ndarray, straight_midpoint: np.ndarray) -> dict:
    traj = np.asarray(traj, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    dt = np.diff(tau_grid)
    dx = np.diff(traj, axis=0)
    segment_lengths = np.linalg.norm(dx, axis=-1)
    velocities = dx / dt[:, None, None]
    midpoint = traj[int(np.argmin(np.abs(tau_grid - 0.5)))]
    endpoint_dist = np.linalg.norm(traj[-1] - traj[0], axis=1)
    return {
        "mean_endpoint_distance": float(endpoint_dist.mean()),
        "mean_path_length": float(segment_lengths.sum(axis=0).mean()),
        "energy_proxy": float(np.mean(np.sum(velocities**2, axis=-1))),
        "midpoint_deviation": float(np.linalg.norm(midpoint - straight_midpoint, axis=1).mean()),
    }


def _brownian_bridge_trajectories(
    x0: np.ndarray,
    x1: np.ndarray,
    tau_grid: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tau_grid = np.asarray(tau_grid, dtype=float)
    dt = np.diff(tau_grid)
    increments = rng.normal(scale=np.sqrt(dt)[:, None, None], size=(len(dt), len(x0), x0.shape[1]))
    brownian = np.concatenate([np.zeros((1, len(x0), x0.shape[1])), np.cumsum(increments, axis=0)], axis=0)
    bridge_noise = brownian - tau_grid[:, None, None] * brownian[-1][None, :, :]
    base = (1.0 - tau_grid[:, None, None]) * x0[None, :, :] + tau_grid[:, None, None] * x1[None, :, :]
    return base + float(sigma) * bridge_noise


def _energy_and_length_pc(traj: np.ndarray, tau_grid: np.ndarray) -> tuple[float, float]:
    tau_grid = np.asarray(tau_grid, dtype=float)
    dx = np.diff(traj, axis=0)
    dt = np.diff(tau_grid)
    velocities = dx / dt[:, None, None]
    energy = float(np.mean(np.sum(velocities**2, axis=-1)))
    length = float(np.linalg.norm(dx, axis=-1).sum(axis=0).mean())
    return energy, length


def _require_phate():
    try:
        import phate
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "phate is required for Chapter 2 EB visualization and out-of-sample path transforms."
        ) from exc
    return phate


def _fit_phate_model(X_pc20: np.ndarray, seed: int):
    phate = _require_phate()
    model = phate.PHATE(
        n_components=2,
        knn=5,
        decay=40,
        n_landmark=min(2000, int(X_pc20.shape[0])),
        random_state=int(seed),
        n_jobs=1,
        verbose=0,
    )
    model.fit(np.asarray(X_pc20, dtype=float))
    return model


def _phate_transform(model, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pre-fit PHATE should not be used.*")
        return np.asarray(model.transform(np.asarray(X, dtype=float)), dtype=float)


def _run_exp1_static_ot(eb: dict, fig_dir: Path, out_dir: Path, artifact_base: Path, seed: int):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    from src.core.ot import (
        compute_ot_coupling_from_cost,
        independent_coupling,
        pairwise_squared_distances,
        sample_pair_indices_from_coupling,
    )

    rng = np.random.default_rng(seed)
    source_time = "1"
    target_time = "2"
    time_labels = eb["time"].astype(str)
    idx0_all = np.flatnonzero(time_labels == source_time)
    idx1_all = np.flatnonzero(time_labels == target_time)
    if len(idx0_all) == 0 or len(idx1_all) == 0:
        raise ValueError(f"EB source/target times {source_time}->{target_time} were not found.")

    idx0 = _subsample_indices(idx0_all, 250, rng)
    idx1 = _subsample_indices(idx1_all, 250, rng)
    X0_cost = np.asarray(eb["X_cost"][idx0], dtype=float)
    X1_cost = np.asarray(eb["X_cost"][idx1], dtype=float)
    X0_plot = np.asarray(eb["X_plot"][idx0], dtype=float)
    X1_plot = np.asarray(eb["X_plot"][idx1], dtype=float)

    C_raw = pairwise_squared_distances(X0_cost, X1_cost)
    cost_scale = _median_positive_scale(C_raw)
    C_norm = C_raw / max(cost_scale, 1e-12)

    epsilons = [0.001, 0.01, 0.05, 0.1, 1.0, 5.0]
    pi_ind = independent_coupling(len(X0_cost), len(X1_cost))
    rows = [_coupling_diagnostic_row("independent", None, pi_ind, C_raw, C_norm, None, cost_scale)]

    plans: dict[float, np.ndarray] = {}
    infos: dict[float, dict] = {}
    for epsilon in epsilons:
        pi, info = compute_ot_coupling_from_cost(
            C_norm,
            epsilon=float(epsilon),
            return_info=True,
            num_iter_max=5000,
            stop_thr=1e-9,
        )
        plans[float(epsilon)] = pi
        infos[float(epsilon)] = info
        rows.append(_coupling_diagnostic_row("sinkhorn_ot", epsilon, pi, C_raw, C_norm, info, cost_scale))

    table = pd.DataFrame(rows)
    table_path = _save_table(table, out_dir, "table02_01_coupling_diagnostics.csv", artifact_base)

    main_epsilon = 0.05
    pi_main = plans[main_epsilon]
    i0_ind = rng.integers(0, len(X0_plot), size=32)
    i1_ind = rng.integers(0, len(X1_plot), size=32)
    i0_ot, i1_ot = sample_pair_indices_from_coupling(pi_main, batch_size=42, seed=seed + 10)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0))
    ax = axes[0, 0]
    _draw_endpoint_cloud(ax, X0_plot, X1_plot)
    _draw_arrows(
        ax,
        X0_plot,
        X1_plot,
        i0_ind,
        i1_ind,
        color="#4D4D4D",
        alpha=0.68,
        linewidth=0.85,
        mutation_scale=8.0,
    )
    ax.set_title("A. Independent endpoint coupling")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    ax.legend(frameon=False, loc="best")

    ax = axes[0, 1]
    _draw_endpoint_cloud(ax, X0_plot, X1_plot)
    _draw_arrows(
        ax,
        X0_plot,
        X1_plot,
        i0_ot,
        i1_ot,
        color="#008A70",
        alpha=0.78,
        linewidth=0.95,
        mutation_scale=8.5,
    )
    ax.set_title(f"B. Sinkhorn OT endpoint coupling, epsilon={main_epsilon}")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")

    ax = axes[1, 0]
    row_order = np.lexsort((X0_plot[:, 1], X0_plot[:, 0]))
    col_order = np.lexsort((X1_plot[:, 1], X1_plot[:, 0]))
    sorted_pi = pi_main[row_order][:, col_order]
    positive = sorted_pi[sorted_pi > 0]
    norm = LogNorm(vmin=max(float(positive.min()), 1e-12), vmax=float(sorted_pi.max())) if positive.size else None
    im = ax.imshow(sorted_pi, aspect="auto", cmap="magma", norm=norm)
    ax.set_title("C. Transport plan sorted by PHATE coordinate")
    ax.set_xlabel("target cells")
    ax.set_ylabel("source cells")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="coupling mass")

    ax = axes[1, 1]
    eps_df = table[table["method"] == "sinkhorn_ot"].copy()
    converged = eps_df["sinkhorn_converged"].astype(bool).to_numpy()
    eps_values = eps_df["epsilon"].to_numpy(dtype=float)
    support_values = eps_df["effective_support"].to_numpy(dtype=float)
    cost_values = eps_df["expected_cost_raw"].to_numpy(dtype=float)
    ax.plot(eps_values, support_values, color="#0072B2", alpha=0.35, linewidth=1.0)
    ax.scatter(
        eps_values[converged],
        support_values[converged],
        marker="o",
        s=42,
        color="#0072B2",
        label="effective support",
        zorder=4,
    )
    ax.scatter(
        eps_values[~converged],
        support_values[~converged],
        marker="x",
        s=58,
        color="#777777",
        linewidths=1.5,
        label="not converged",
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Sinkhorn epsilon on normalized PC-20 cost")
    ax.set_ylabel("effective support")
    ax.axvline(main_epsilon, color="#222222", linestyle="--", linewidth=1.0, alpha=0.8)
    main_row = eps_df[np.isclose(eps_df["epsilon"], main_epsilon)].iloc[0]
    ax.scatter(
        [main_epsilon],
        [float(main_row["effective_support"])],
        marker="*",
        s=145,
        color="#111111",
        edgecolors="white",
        linewidths=0.5,
        label="main eps=0.05",
        zorder=6,
    )
    ax2 = ax.twinx()
    ax2.plot(eps_values, cost_values, color="#D55E00", alpha=0.35, linewidth=1.0)
    ax2.scatter(
        eps_values[converged],
        cost_values[converged],
        marker="s",
        s=34,
        color="#D55E00",
        label="raw expected cost",
        zorder=4,
    )
    ax2.scatter(
        eps_values[~converged],
        cost_values[~converged],
        marker="x",
        s=58,
        color="#777777",
        linewidths=1.5,
        zorder=5,
    )
    ax2.set_ylabel("expected raw cost in PC-20")
    ax.set_title("D. Epsilon changes plan diffuseness and cost")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ordered_handles = [
        Line2D([0], [0], marker="o", color="#0072B2", linestyle="None", markersize=5, label="effective support"),
        Line2D([0], [0], marker="s", color="#D55E00", linestyle="None", markersize=5, label="raw expected cost"),
        Line2D([0], [0], marker="x", color="#777777", linestyle="None", markersize=6, label="not converged"),
        Line2D([0], [0], marker="*", color="#111111", linestyle="--", markersize=8, label="main eps=0.05"),
    ]
    ax.legend(handles=ordered_handles, frameon=False, loc="best")

    fig.text(
        0.02,
        0.01,
        (
            "OT computed in PC-20; PHATE only for visualization. Endpoint arrows are model-implied, not observed lineage. "
            "Sinkhorn OT at ε=0.05 reduces expected PC-20 transport cost from "
            f"{rows[0]['expected_cost_raw']:.1f} to {float(main_row['expected_cost_raw']):.1f}; smaller ε did not converge here."
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    figure_paths = _save_fig_both(fig, fig_dir, "fig02_02_static_ot_endpoint_transport", artifact_base)

    return {
        "table": table,
        "table_paths": [table_path],
        "figure_paths": figure_paths,
        "source_time": source_time,
        "target_time": target_time,
        "idx0": idx0,
        "idx1": idx1,
        "X0_cost": X0_cost,
        "X1_cost": X1_cost,
        "X0_plot": X0_plot,
        "X1_plot": X1_plot,
        "C_raw": C_raw,
        "C_norm": C_norm,
        "cost_scale": cost_scale,
        "main_epsilon": main_epsilon,
        "pi_main": pi_main,
        "epsilons": epsilons,
        "sinkhorn_backend": infos[main_epsilon].get("backend", ""),
    }


def _run_exp2_path_geometry(fig_dir: Path, out_dir: Path, artifact_base: Path, seed: int):
    import matplotlib.pyplot as plt

    from src.core.ot import compute_ot_coupling_from_cost, pairwise_squared_distances, sample_pair_indices_from_coupling
    from src.core.paths import curved_path, linear_path
    from src.data.toy import make_y_branching_snapshots

    toy = make_y_branching_snapshots(
        n_cells=400,
        timepoints=(0.0, 1.0),
        rare_fate_fraction=0.5,
        noise=0.075,
        seed=seed,
    )
    X = np.asarray(toy.X, dtype=float)
    times = np.asarray(toy.time, dtype=float)
    X0 = X[np.isclose(times, 0.0)]
    X1 = X[np.isclose(times, 1.0)]
    C_raw = pairwise_squared_distances(X0, X1)
    C_norm = C_raw / max(_median_positive_scale(C_raw), 1e-12)
    pi = compute_ot_coupling_from_cost(C_norm, epsilon=0.05)
    i0, i1 = sample_pair_indices_from_coupling(pi, batch_size=80, seed=seed + 20)
    x0 = X0[i0]
    x1 = X1[i1]

    tau_grid = np.linspace(0.0, 1.0, 51)
    straight = np.stack([linear_path(x0, x1, tau)[0] for tau in tau_grid], axis=0)
    curved = np.stack([curved_path(x0, x1, tau, curvature=0.20, direction="normal") for tau in tau_grid], axis=0)
    stochastic = _brownian_bridge_trajectories(x0, x1, tau_grid, sigma=0.23, seed=seed + 21)
    straight_mid = 0.5 * (x0 + x1)

    rows = []
    for family, traj, notes in [
        ("straight bridge", straight, "Native 2D straight interpolation for the sampled endpoint pairs."),
        ("curved deterministic bridge", curved, "Native 2D deterministic bridge with fixed endpoint coupling."),
        ("Brownian stochastic bridge", stochastic, "Discrete sampled-path proxy; not a strong action conclusion."),
    ]:
        stats = _path_stats(traj, tau_grid, straight_mid)
        midpoint = traj[int(np.argmin(np.abs(tau_grid - 0.5)))]
        if family == "Brownian stochastic bridge":
            spread = float(np.linalg.norm(midpoint - midpoint.mean(axis=0, keepdims=True), axis=1).mean())
        elif family == "straight bridge":
            spread = 0.0
        else:
            spread = float(np.std(np.linalg.norm(midpoint - straight_mid, axis=1)))
        rows.append(
            {
                "path_family": family,
                **stats,
                "midpoint_spread": spread,
                "n_pairs": int(len(x0)),
                "n_steps": int(len(tau_grid)),
                "notes": notes,
            }
        )
    table = pd.DataFrame(rows)
    table_path = _save_table(table, out_dir, "table02_02_path_diagnostics.csv", artifact_base)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), sharex=True, sharey=True)
    background_kw = dict(s=12, alpha=0.18, linewidths=0)
    path_count = 42
    for ax in axes:
        ax.scatter(X0[:, 0], X0[:, 1], c="#4267B2", **background_kw)
        ax.scatter(X1[:, 0], X1[:, 1], c="#D55E00", **background_kw)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("native 2D state 1")
    axes[0].set_ylabel("native 2D state 2")

    _draw_arrows(axes[0], X0, X1, i0[:38], i1[:38], color="#555555", alpha=0.42, linewidth=0.65, mutation_scale=6.5)
    axes[0].set_title("A. Endpoint coupling only")

    for pair_idx in range(path_count):
        axes[1].plot(straight[:, pair_idx, 0], straight[:, pair_idx, 1], color="#0072B2", alpha=0.20, linewidth=0.9)
    axes[1].set_title("B. Straight bridges")

    for pair_idx in range(path_count):
        axes[2].plot(curved[:, pair_idx, 0], curved[:, pair_idx, 1], color="#009E73", alpha=0.23, linewidth=0.9)
    axes[2].set_title("C. Curved bridges")

    for repeat, alpha in enumerate([0.18, 0.13, 0.10]):
        stoch_plot = _brownian_bridge_trajectories(
            x0[:24],
            x1[:24],
            tau_grid,
            sigma=0.23,
            seed=seed + 30 + repeat,
        )
        for pair_idx in range(stoch_plot.shape[1]):
            axes[3].plot(stoch_plot[:, pair_idx, 0], stoch_plot[:, pair_idx, 1], color="#CC79A7", alpha=alpha, linewidth=0.8)
    axes[3].set_title("D. Stochastic bridge samples")
    fig.text(
        0.02,
        0.01,
        "Toy 2D native computation and visualization. These path constructions are not observed lineage; the same endpoint coupling does not determine the path.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    figure_paths = _save_fig_both(fig, fig_dir, "fig02_03_same_endpoints_different_paths", artifact_base)
    return {"table": table, "table_paths": [table_path], "figure_paths": figure_paths}


def _run_exp3_dynamic_ot(
    eb: dict,
    exp1: dict,
    fig_dir: Path,
    out_dir: Path,
    artifact_base: Path,
    seed: int,
):
    import matplotlib.pyplot as plt

    from src.core.ot import compute_ot_coupling_from_cost, pairwise_squared_distances, sample_pair_indices_from_coupling

    X_pc20 = np.asarray(eb["X_cost"], dtype=float)
    time_labels = eb["time"].astype(str)
    times = _sorted_time_labels(time_labels)
    phate_model = _fit_phate_model(X_pc20, seed=seed)
    X_phate_fit = _phate_transform(phate_model, X_pc20)

    rng = np.random.default_rng(seed + 40)
    adjacent_arrow_data = []
    for edge_idx, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
        idx0_all = np.flatnonzero(time_labels == t0)
        idx1_all = np.flatnonzero(time_labels == t1)
        idx0 = _subsample_indices(idx0_all, 100, rng)
        idx1 = _subsample_indices(idx1_all, 100, rng)
        C_raw = pairwise_squared_distances(X_pc20[idx0], X_pc20[idx1])
        C_norm = C_raw / max(_median_positive_scale(C_raw), 1e-12)
        pi = compute_ot_coupling_from_cost(C_norm, epsilon=0.05)
        row_mass = np.clip(pi.sum(axis=1, keepdims=True), 1e-12, None)
        n_edge_arrows = 8
        order = np.lexsort((X_phate_fit[idx0, 1], X_phate_fit[idx0, 0]))
        chosen_rows = order[np.linspace(0, len(order) - 1, n_edge_arrows, dtype=int)]
        starts = X_phate_fit[idx0[chosen_rows]]
        barycentric_targets = (pi[chosen_rows] @ X_phate_fit[idx1]) / row_mass[chosen_rows]
        adjacent_arrow_data.append(
            {
                "starts": starts,
                "deltas": barycentric_targets - starts,
                "source_time": t0,
                "target_time": t1,
                "color": ["#0072B2", "#009E73", "#D55E00", "#CC79A7"][edge_idx % 4],
            }
        )

    pair_i0, pair_i1 = sample_pair_indices_from_coupling(exp1["pi_main"], batch_size=20, seed=seed + 60)
    x0 = exp1["X0_cost"][pair_i0]
    x1 = exp1["X1_cost"][pair_i1]
    tau_grid = np.linspace(0.0, 1.0, 41)
    economical = (1.0 - tau_grid[:, None, None]) * x0[None, :, :] + tau_grid[:, None, None] * x1[None, :, :]

    direction = x1 - x0
    random_vec = rng.normal(size=direction.shape)
    projection = (np.sum(random_vec * direction, axis=1, keepdims=True) / np.clip(np.sum(direction**2, axis=1, keepdims=True), 1e-12, None)) * direction
    perpendicular = random_vec - projection
    perpendicular /= np.clip(np.linalg.norm(perpendicular, axis=1, keepdims=True), 1e-12, None)
    amplitude = 0.75 * np.linalg.norm(direction, axis=1, keepdims=True)
    detour = economical + np.sin(np.pi * tau_grid)[:, None, None] * amplitude[None, :, :] * perpendicular[None, :, :]

    energy_econ, length_econ = _energy_and_length_pc(economical, tau_grid)
    energy_detour, length_detour = _energy_and_length_pc(detour, tau_grid)
    table = pd.DataFrame(
        [
            {
                "path_family": "economical_straight_pc20",
                "energy_proxy": energy_econ,
                "mean_path_length_pc20": length_econ,
                "n_pairs": int(len(x0)),
                "n_steps": int(len(tau_grid)),
                "notes": "Straight PC-20 bridge for sampled OT endpoint pairs.",
            },
            {
                "path_family": "detour_perpendicular_pc20",
                "energy_proxy": energy_detour,
                "mean_path_length_pc20": length_detour,
                "n_pairs": int(len(x0)),
                "n_steps": int(len(tau_grid)),
                "notes": "Fixed random perpendicular perturbation in PC-20; energy proxy intuition only.",
            },
        ]
    )
    table_path = _save_table(table, out_dir, "table02_03_dynamic_ot_energy_proxy.csv", artifact_base)

    econ_phate = _phate_transform(phate_model, economical.reshape(-1, economical.shape[-1])).reshape(len(tau_grid), len(x0), 2)
    detour_phate = _phate_transform(phate_model, detour.reshape(-1, detour.shape[-1])).reshape(len(tau_grid), len(x0), 2)

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0), gridspec_kw={"width_ratios": [1.1, 1.1, 1.1, 0.82]})
    time_as_float = np.array([float(t) for t in time_labels], dtype=float)
    sc = axes[0].scatter(
        X_phate_fit[:, 0],
        X_phate_fit[:, 1],
        c=time_as_float,
        s=8,
        alpha=0.45,
        cmap="viridis",
        linewidths=0,
    )
    axes[0].set_title("A. EB empirical density path")
    axes[0].set_xlabel("fitted PHATE 1")
    axes[0].set_ylabel("fitted PHATE 2")
    fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04, label="time")

    axes[1].scatter(X_phate_fit[:, 0], X_phate_fit[:, 1], c=time_as_float, s=5, alpha=0.10, cmap="viridis", linewidths=0)
    for item in adjacent_arrow_data:
        starts = item["starts"]
        deltas = item["deltas"]
        axes[1].quiver(
            starts[:, 0],
            starts[:, 1],
            deltas[:, 0],
            deltas[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=item["color"],
            alpha=0.82,
            width=0.0032,
            headwidth=3.6,
            headlength=4.8,
            headaxislength=4.0,
            linewidths=0.25,
            zorder=4,
        )
        axes[1].plot([], [], color=item["color"], linewidth=1.4, label=f"{item['source_time']}->{item['target_time']}")
    axes[1].set_title("B. Adjacent-time barycentric OT arrows")
    axes[1].set_xlabel("fitted PHATE 1")
    axes[1].set_ylabel("fitted PHATE 2")
    axes[1].legend(frameon=False, loc="best", ncol=2, handlelength=1.2, columnspacing=0.8)

    plot_pairs = min(12, econ_phate.shape[1])
    for pair_idx in range(plot_pairs):
        axes[2].plot(econ_phate[:, pair_idx, 0], econ_phate[:, pair_idx, 1], color="#0072B2", alpha=0.45, linewidth=1.0)
        axes[2].plot(detour_phate[:, pair_idx, 0], detour_phate[:, pair_idx, 1], color="#D55E00", alpha=0.40, linewidth=1.0)
    axes[2].scatter(econ_phate[0, :plot_pairs, 0], econ_phate[0, :plot_pairs, 1], s=13, c="#4267B2", alpha=0.7, linewidths=0)
    axes[2].scatter(econ_phate[-1, :plot_pairs, 0], econ_phate[-1, :plot_pairs, 1], s=13, c="#D55E00", alpha=0.7, linewidths=0)
    axes[2].set_title("C1. Economical vs detour paths")
    axes[2].set_xlabel("fitted PHATE 1")
    axes[2].set_ylabel("fitted PHATE 2")
    axes[2].plot([], [], color="#0072B2", label="economical")
    axes[2].plot([], [], color="#D55E00", label="detour")
    axes[2].legend(frameon=False, loc="lower left")

    energy_values = [energy_econ, energy_detour]
    bars = axes[3].bar([0, 1], energy_values, color=["#0072B2", "#D55E00"])
    axes[3].set_xticks([0, 1], ["straight", "detour"])
    axes[3].set_ylabel("energy proxy (PC-20)")
    axes[3].set_title("C2. PC-20 action proxy")
    ratio = energy_detour / max(energy_econ, 1e-12)
    axes[3].set_ylim(0.0, max(energy_values) * 1.28)
    axes[3].text(0.5, max(energy_values) * 1.16, f"detour / straight = {ratio:.2f}x", ha="center", va="bottom", fontsize=9)
    for bar, value in zip(bars, energy_values):
        axes[3].text(bar.get_x() + bar.get_width() / 2, value + max(energy_values) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    fig.text(
        0.02,
        0.01,
        "Energy is computed in PC-20. PHATE is only visualization. This is an energy proxy, not a solved dynamic OT problem.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    figure_paths = _save_fig_both(fig, fig_dir, "fig02_04_dynamic_ot_low_action", artifact_base)
    return {
        "table": table,
        "table_paths": [table_path],
        "figure_paths": figure_paths,
        "phate_model": phate_model,
        "energy_ratio": float(energy_detour / max(energy_econ, 1e-12)),
    }


def _sample_gaussian_mixture_torch(n: int, centers: "torch.Tensor", scale: float, generator, device: str):
    import torch

    idx = torch.randint(0, centers.shape[0], (int(n),), generator=generator, device=device)
    return centers[idx] + float(scale) * torch.randn((int(n), centers.shape[1]), generator=generator, device=device)


def _torch_mmd_rbf(X, Y, gamma: float = 0.5):
    import torch

    dxx = torch.cdist(X, X).pow(2)
    dyy = torch.cdist(Y, Y).pow(2)
    dxy = torch.cdist(X, Y).pow(2)
    return torch.exp(-gamma * dxx).mean() + torch.exp(-gamma * dyy).mean() - 2.0 * torch.exp(-gamma * dxy).mean()


def _euler_integrate_torch(model, x0, n_solver_steps: int):
    import torch

    x = x0
    dt = 1.0 / int(n_solver_steps)
    for step in range(int(n_solver_steps)):
        t = torch.full((x.shape[0], 1), step * dt, dtype=x.dtype, device=x.device)
        x = x + dt * model(x, t)
    return x, int(n_solver_steps)


def _run_exp4_training_proxy(fig_dir: Path, out_dir: Path, artifact_base: Path, quick_mode: bool, seed: int):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import torch

    from src.evaluation.metrics import mmd_rbf
    from src.core.models import VelocityMLP
    from src.utils import set_seed

    set_seed(seed)
    torch.set_num_threads(min(max(torch.get_num_threads(), 1), 4))
    device = "cpu"
    batch_size = 256
    n_steps = 100 if quick_mode else 200
    n_solver_steps = 16 if quick_mode else 24
    eval_solver_steps = 32

    angles = torch.linspace(0, 2 * np.pi, 9, dtype=torch.float32)[:-1]
    source_centers = torch.stack([1.2 * torch.cos(angles), 1.2 * torch.sin(angles)], dim=1).to(device)
    rot = torch.tensor([[0.78, -0.62], [0.62, 0.78]], dtype=torch.float32, device=device)
    target_centers = (source_centers @ rot.T) * torch.tensor([0.9, 1.15], device=device) + torch.tensor([0.65, -0.25], device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def sample_source(n: int):
        return _sample_gaussian_mixture_torch(n, source_centers, 0.16, generator, device)

    def sample_target(n: int):
        return _sample_gaussian_mixture_torch(n, target_centers, 0.18, generator, device)

    def train_solver_loop():
        model = VelocityMLP(x_dim=2, hidden_dim=64, hidden_layers=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        step_times = []
        nfe_values = []
        start = time.perf_counter()
        final_loss = np.nan
        for step in range(n_steps):
            x0 = sample_source(batch_size)
            target = sample_target(batch_size)
            opt.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            pred, nfe = _euler_integrate_torch(model, x0, n_solver_steps=n_solver_steps)
            loss = _torch_mmd_rbf(pred, target, gamma=0.6)
            loss.backward()
            opt.step()
            elapsed = time.perf_counter() - step_start
            step_times.append(elapsed)
            nfe_values.append(nfe)
            final_loss = float(loss.detach().cpu())
        total = time.perf_counter() - start
        return model, np.asarray(step_times), np.asarray(nfe_values), total, final_loss

    solver_model, solver_times, solver_nfe, solver_total, solver_final = train_solver_loop()

    with torch.no_grad():
        eval_source = sample_source(512)
        eval_target = sample_target(512)
        solver_pred, _ = _euler_integrate_torch(solver_model, eval_source.clone(), n_solver_steps=eval_solver_steps)
    eval_target_np = eval_target.cpu().numpy()
    solver_mmd = mmd_rbf(solver_pred.cpu().numpy(), eval_target_np)

    rows = [
        {
            "method": "solver-in-the-loop Neural ODE training proxy",
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "mean_time_per_step_ms": float(solver_times.mean() * 1000.0),
            "median_time_per_step_ms": float(np.median(solver_times) * 1000.0),
            "nfe_per_step_mean": float(solver_nfe.mean()),
            "nfe_per_step_median": float(np.median(solver_nfe)),
            "total_wall_time_sec": float(solver_total),
            "final_loss": float(solver_final),
            "final_distribution_mmd": float(solver_mmd),
            "notes": "Pedagogical proxy: backpropagates through a differentiable Euler ODE rollout with MMD loss; no likelihood or divergence tracking.",
        }
    ]
    table = pd.DataFrame(rows)
    table_path = _save_table(table, out_dir, "table02_04_cnf_training_bottleneck.csv", artifact_base)
    provenance_path = _save_table(table, out_dir, "table02_04_training_cost_proxy.csv", artifact_base)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.5))
    ax = axes[0, 0]
    ax.axis("off")
    box_specs = [
        (0.04, 0.62, 0.21, 0.18, "sample x0"),
        (0.31, 0.62, 0.26, 0.18, "ODE rollout\nlearned velocity"),
        (0.63, 0.62, 0.24, 0.18, "endpoint /\ndensity loss"),
        (0.31, 0.22, 0.31, 0.18, "backprop\nthrough solver"),
    ]
    for x, y, w, h, label in box_specs:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.0, edgecolor="#444444", facecolor="#F5F5F5"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.31, 0.71), xytext=(0.25, 0.71), arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    ax.annotate("", xy=(0.63, 0.71), xytext=(0.57, 0.71), arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    ax.annotate("", xy=(0.47, 0.40), xytext=(0.73, 0.62), arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    ax.annotate("", xy=(0.44, 0.62), xytext=(0.44, 0.40), arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.5, connectionstyle="arc3,rad=-0.35"))
    ax.text(0.04, 0.88, "CNF training bottleneck", fontsize=10, color="#0072B2", weight="bold")
    ax.text(0.31, 0.52, "solver inside\ntraining loop", fontsize=9, color="#0072B2", ha="center", weight="bold")
    ax.set_title("A. CNF training control flow")

    ax = axes[0, 1]
    label = ["solver-in-loop\nproxy"]
    mean_step = float(table.iloc[0]["mean_time_per_step_ms"])
    median_step = float(table.iloc[0]["median_time_per_step_ms"])
    total_wall = float(table.iloc[0]["total_wall_time_sec"])
    nfe_mean = float(table.iloc[0]["nfe_per_step_mean"])
    final_mmd = float(table.iloc[0]["final_distribution_mmd"])
    ax.bar(label, [mean_step], color=["#0072B2"])
    ax.set_ylabel("mean time per step (ms)")
    ax.set_title("B. Mean time per training step")
    ax.set_ylim(0.0, mean_step * 1.28)
    ax.text(0, mean_step * 1.05, f"{mean_step:.2f} ms", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    ax.bar(label, [nfe_mean], color=["#0072B2"])
    ax.set_ylabel("velocity evaluations per step")
    ax.set_title("C. Velocity evaluations / NFE per step")
    ax.set_ylim(0.0, nfe_mean * 1.30)
    ax.text(0, nfe_mean * 1.05, f"{nfe_mean:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    bars = ax.bar(label, [total_wall], color=["#0072B2"])
    ax.set_ylabel("total wall-clock time (sec)")
    ax.set_title(f"D. Total wall-clock time for {n_steps} steps")
    ax.set_ylim(0.0, total_wall * 1.25)
    for bar, value in zip(bars, [total_wall]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + total_wall * 0.04, f"{float(value):.2f} sec", ha="center", va="bottom", fontsize=9)
    fig.text(
        0.02,
        0.01,
        "The CNF-style solver-in-the-loop proxy rolls out the learned ODE before evaluating the loss, requiring "
        f"{nfe_mean:.0f} velocity evaluations per step in this quick run. This is a training-control-flow bottleneck, not a full likelihood CNF benchmark.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    figure_paths = _save_fig_both(fig, fig_dir, "fig02_06_training_bottleneck_timing", artifact_base)
    return {
        "table": table,
        "table_paths": [table_path],
        "provenance_table_paths": [provenance_path],
        "figure_paths": figure_paths,
        "cnf_proxy_mean_step_time_ms": mean_step,
        "cnf_proxy_median_step_time_ms": median_step,
        "cnf_proxy_nfe_per_step": nfe_mean,
        "cnf_proxy_total_wall_time_sec": total_wall,
        "cnf_proxy_final_distribution_mmd": final_mmd,
    }


def _run_optional_exp5_cost_sensitivity(
    eb_path: Path,
    out_dir: Path,
    artifact_base: Path,
    seed: int,
) -> dict:
    from src.data.loading import load_eb_timecourse_for_ch03
    from src.evaluation.metrics import coupling_l1_distance
    from src.core.ot import compute_ot_coupling_from_cost, coupling_diagnostics, pairwise_squared_distances

    rng = np.random.default_rng(seed + 90)
    eb50 = load_eb_timecourse_for_ch03(
        path=eb_path,
        cost_embedding="pcs",
        plot_embedding="phate",
        n_cost_dims=50,
        max_cells_per_time=900,
        seed=seed,
    )
    labels = eb50["time"].astype(str)
    idx0 = _subsample_indices(np.flatnonzero(labels == "1"), 250, rng)
    idx1 = _subsample_indices(np.flatnonzero(labels == "2"), 250, rng)
    plans = {}
    rows = []
    for dims in [5, 10, 20, 50]:
        X0 = np.asarray(eb50["X_cost"][idx0, :dims], dtype=float)
        X1 = np.asarray(eb50["X_cost"][idx1, :dims], dtype=float)
        C_raw = pairwise_squared_distances(X0, X1)
        scale = _median_positive_scale(C_raw)
        C_norm = C_raw / max(scale, 1e-12)
        pi, info = compute_ot_coupling_from_cost(C_norm, epsilon=0.05, return_info=True)
        plans[int(dims)] = pi
        diag = coupling_diagnostics(pi, C_raw)
        rows.append(
            {
                "pc_dims": int(dims),
                "epsilon": 0.05,
                "expected_cost_raw": diag["expected_cost"],
                "entropy": diag["entropy"],
                "effective_support": diag["effective_support"],
                "row_l1_error": diag["row_l1_error"],
                "col_l1_error": diag["col_l1_error"],
                "sinkhorn_converged": bool(info.get("sinkhorn_converged", False)),
                "sinkhorn_backend": str(info.get("backend", "")),
                "cost_scale": float(scale),
            }
        )
    base_plan = plans[20]
    for row in rows:
        row["l1_to_pc20_plan"] = float(coupling_l1_distance(plans[int(row["pc_dims"])], base_plan))
    table = pd.DataFrame(rows)
    table_path = _save_table(table, out_dir, "table02_optional_ot_cost_sensitivity.csv", artifact_base)
    return {"table": table, "table_paths": [table_path], "figure_paths": []}


def run_ch02(quick_mode: bool = True, project_root: str | Path | None = None, seed: int = 42) -> dict:
    started = time.perf_counter()
    project_root = Path(project_root) if project_root is not None else _find_project_root()
    project_root = project_root.resolve()
    _prepare_imports(project_root)

    import matplotlib

    matplotlib.use("Agg", force=True)
    _set_plot_style()

    from src.data.loading import load_eb_timecourse_for_ch03
    from src.utils import ensure_dir, set_seed

    set_seed(seed)
    _require_phate()

    paper_root = ensure_dir(project_root)
    fig_dir = ensure_dir(paper_root / "figures" / "ch02")
    out_dir = ensure_dir(paper_root / "outputs" / "ch02")
    eb_path = project_root / "data" / "trajectorynet_eb" / "eb_velocity_v5.npz"
    eb = load_eb_timecourse_for_ch03(
        path=eb_path,
        cost_embedding="pcs",
        plot_embedding="phate",
        n_cost_dims=20,
        max_cells_per_time=900,
        seed=seed,
    )

    figures_written: list[str] = []
    tables_written: list[str] = []

    exp1 = _run_exp1_static_ot(eb, fig_dir, out_dir, paper_root, seed)
    figures_written.extend(exp1["figure_paths"])
    tables_written.extend(exp1["table_paths"])

    exp2 = _run_exp2_path_geometry(fig_dir, out_dir, paper_root, seed)
    figures_written.extend(exp2["figure_paths"])
    tables_written.extend(exp2["table_paths"])

    exp3 = _run_exp3_dynamic_ot(eb, exp1, fig_dir, out_dir, paper_root, seed)
    figures_written.extend(exp3["figure_paths"])
    tables_written.extend(exp3["table_paths"])

    exp4 = _run_exp4_training_proxy(fig_dir, out_dir, paper_root, quick_mode=quick_mode, seed=seed)
    figures_written.extend(exp4["figure_paths"])
    tables_written.extend(exp4["table_paths"])

    optional_exp5 = _run_optional_exp5_cost_sensitivity(eb_path, out_dir, paper_root, seed)
    tables_written.extend(optional_exp5["table_paths"])

    summary = {
        "quick_mode": bool(quick_mode),
        "seed": int(seed),
        "runtime_sec": float(time.perf_counter() - started),
        "eb_path": _display_path(eb_path, project_root),
        "source_time": exp1["source_time"],
        "target_time": exp1["target_time"],
        "n_source_ot": int(exp1["X0_cost"].shape[0]),
        "n_target_ot": int(exp1["X1_cost"].shape[0]),
        "cost_space": "PC-20",
        "visualization_space": "PHATE",
        "phate_transform_used": True,
        "phate_installed": True,
        "static_ot_cost_scale": "median_positive_cost",
        "static_ot_cost_scale_value": float(exp1["cost_scale"]),
        "static_ot_main_epsilon": float(exp1["main_epsilon"]),
        "sinkhorn_backend_main_epsilon": str(exp1["sinkhorn_backend"]),
        "dynamic_ot_energy_ratio_detour_over_economical": float(exp3["energy_ratio"]),
        "cnf_proxy_mean_step_time_ms": float(exp4["cnf_proxy_mean_step_time_ms"]),
        "cnf_proxy_nfe_per_step": float(exp4["cnf_proxy_nfe_per_step"]),
        "cnf_proxy_total_wall_time_sec": float(exp4["cnf_proxy_total_wall_time_sec"]),
        "optional_exp5_cost_sensitivity": True,
        "figures_written": figures_written,
        "tables_written": tables_written,
        "provenance_tables_written": exp4.get("provenance_table_paths", []),
        "concept_boundaries": CONCEPT_BOUNDARIES,
    }
    summary_path = out_dir / "ch02_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["summary_json"] = _display_path(summary_path, paper_root)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 2 figures and tables.")
    parser.add_argument("--full", action="store_true", help="Use full training steps for Exp 4. Default is quick mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_ch02(quick_mode=not args.full, project_root=args.project_root, seed=args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
