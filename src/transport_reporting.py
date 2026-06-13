from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.artifacts import display_saved_figure, safe_relpath, save_csv, save_figure_formats
from src.ot import coupling_diagnostics


def display_path(path: str | Path, base: str | Path) -> str:
    return safe_relpath(path, root=base)


def save_fig_both(
    fig,
    fig_dir: str | Path,
    stem: str,
    artifact_base: str | Path,
    *,
    formats: Iterable[str] = ("png", "svg"),
    dpi: int = 300,
    pad_inches: float = 0.1,
) -> list[str]:
    paths = save_figure_formats(fig, fig_dir, stem, formats=formats, dpi=dpi, close=True, pad_inches=pad_inches)
    return [display_path(path, artifact_base) for path in paths]


def show_saved_png(fig_dir: str | Path, stem: str, width: int = 420) -> None:
    display_saved_figure(Path(fig_dir) / f"{stem}.png", width=width)


def save_ch02_table(table: pd.DataFrame, out_dir: str | Path, filename: str, artifact_base: str | Path) -> str:
    path = save_csv(Path(out_dir) / filename, table)
    return display_path(path, artifact_base)


def table_preview(table: pd.DataFrame, columns: list[str] | None = None, n: int = 8) -> pd.DataFrame:
    preview = table if columns is None else table.loc[:, columns]
    return preview.head(n)


def sorted_time_labels(labels: np.ndarray) -> list[str]:
    values = np.unique(np.asarray(labels).astype(str)).tolist()

    def key(value: str):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(values, key=key)


def subsample_indices(indices: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= int(max_n):
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=int(max_n), replace=False))


def coupling_diagnostic_row(
    method: str,
    epsilon,
    pi: np.ndarray,
    C_raw: np.ndarray,
    C_norm: np.ndarray,
    info: dict | None,
    cost_scale: float,
) -> dict:
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


def draw_endpoint_cloud(ax, X0_plot: np.ndarray, X1_plot: np.ndarray) -> None:
    ax.scatter(X0_plot[:, 0], X0_plot[:, 1], s=9, c="#4267B2", alpha=0.35, linewidths=0, label="source")
    ax.scatter(X1_plot[:, 0], X1_plot[:, 1], s=9, c="#D55E00", alpha=0.35, linewidths=0, label="target")


def draw_arrows(
    ax,
    X0_plot: np.ndarray,
    X1_plot: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    color: str,
    alpha: float = 0.35,
    linewidth: float | None = 0.8,
    mutation_scale: float = 7.0,
    width: float = 0.002,
    shrink_a: float = 0.0,
    shrink_b: float = 0.0,
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
                "shrinkA": float(shrink_a),
                "shrinkB": float(shrink_b),
            },
            zorder=3,
        )


def draw_fixed_endpoint_pairs_panel(
    ax,
    background_phate: np.ndarray,
    source_phate: np.ndarray,
    target_phate: np.ndarray,
    colors: dict[str, str],
    *,
    title_fontsize: int = 12,
    label_fontsize: int = 11,
    tick_fontsize: int = 9,
    legend_fontsize: int = 8,
) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.scatter(
        background_phate[:, 0],
        background_phate[:, 1],
        s=4,
        c=colors["background"],
        alpha=0.09,
        linewidths=0,
        zorder=1,
    )
    pair_indices = np.arange(len(source_phate))
    draw_arrows(
        ax,
        source_phate,
        target_phate,
        pair_indices,
        pair_indices,
        color=colors["connection"],
        alpha=0.30,
        linewidth=0.70,
        mutation_scale=5.5,
        shrink_a=2.0,
        shrink_b=2.0,
    )
    ax.scatter(source_phate[:, 0], source_phate[:, 1], s=34, c=colors["source"], alpha=0.90, linewidths=0, label="source", zorder=4)
    ax.scatter(target_phate[:, 0], target_phate[:, 1], s=34, c=colors["target"], alpha=0.90, linewidths=0, label="target", zorder=4)
    ax.set_title("Fixed endpoint pairs", fontsize=title_fontsize, pad=7)
    ax.set_xlabel("PHATE 1", fontsize=label_fontsize)
    ax.set_ylabel("PHATE 2", fontsize=label_fontsize)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=legend_fontsize,
        handlelength=1.0,
        borderaxespad=0.0,
    )


def path_stats(traj: np.ndarray, tau_grid: np.ndarray, straight_midpoint: np.ndarray) -> dict:
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


def brownian_bridge_trajectories(
    x0: np.ndarray,
    x1: np.ndarray,
    tau_grid: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    local_rng = np.random.default_rng(seed)
    tau_grid = np.asarray(tau_grid, dtype=float)
    dt = np.diff(tau_grid)
    increments = local_rng.normal(scale=np.sqrt(dt)[:, None, None], size=(len(dt), len(x0), x0.shape[1]))
    brownian = np.concatenate([np.zeros((1, len(x0), x0.shape[1])), np.cumsum(increments, axis=0)], axis=0)
    bridge_noise = brownian - tau_grid[:, None, None] * brownian[-1][None, :, :]
    base = (1.0 - tau_grid[:, None, None]) * x0[None, :, :] + tau_grid[:, None, None] * x1[None, :, :]
    return base + float(sigma) * bridge_noise


def energy_and_length_pc(traj: np.ndarray, tau_grid: np.ndarray) -> tuple[float, float]:
    tau_grid = np.asarray(tau_grid, dtype=float)
    dx = np.diff(np.asarray(traj, dtype=float), axis=0)
    dt = np.diff(tau_grid)
    velocities = dx / dt[:, None, None]
    energy = float(np.mean(np.sum(velocities**2, axis=-1)))
    length = float(np.linalg.norm(dx, axis=-1).sum(axis=0).mean())
    return energy, length


def action_per_pair_pc(traj: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=float)
    dx = np.diff(np.asarray(traj, dtype=float), axis=0)
    dt = np.diff(tau_grid)
    velocities = dx / dt[:, None, None]
    return np.mean(np.sum(velocities**2, axis=-1), axis=0)


def sample_gaussian_mixture_torch(n: int, centers, scale: float, generator, device: str):
    import torch

    idx = torch.randint(0, centers.shape[0], (int(n),), generator=generator, device=device)
    return centers[idx] + float(scale) * torch.randn((int(n), centers.shape[1]), generator=generator, device=device)


def torch_mmd_rbf(X, Y, gamma: float = 0.5):
    import torch

    dxx = torch.cdist(X, X).pow(2)
    dyy = torch.cdist(Y, Y).pow(2)
    dxy = torch.cdist(X, Y).pow(2)
    return torch.exp(-gamma * dxx).mean() + torch.exp(-gamma * dyy).mean() - 2.0 * torch.exp(-gamma * dxy).mean()


def euler_integrate_torch(model, x0, n_solver_steps: int):
    import torch

    x = x0
    dt = 1.0 / int(n_solver_steps)
    for step in range(int(n_solver_steps)):
        t = torch.full((x.shape[0], 1), step * dt, dtype=x.dtype, device=x.device)
        x = x + dt * model(x, t)
    return x, int(n_solver_steps)


def build_expected_artifact_paths(
    fig_dir: str | Path,
    panel_figure_stems: Iterable[str],
    out_dir: str | Path,
    expected_table_names: Iterable[str],
    expected_output_names: Iterable[str],
    *,
    figure_suffixes: Iterable[str] = (".png", ".svg"),
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    fig_dir = Path(fig_dir)
    out_dir = Path(out_dir)
    expected_figures = [
        fig_dir / f"{stem}{suffix}"
        for stem in panel_figure_stems
        for suffix in figure_suffixes
    ]
    expected_tables = [out_dir / filename for filename in expected_table_names]
    expected_outputs = [out_dir / filename for filename in expected_output_names]
    expected_artifact_paths = expected_figures + expected_tables + expected_outputs
    return expected_figures, expected_tables, expected_outputs, expected_artifact_paths


def write_artifact_manifest(
    expected_artifact_paths: Iterable[str | Path],
    expected_figures: Iterable[str | Path],
    expected_tables: Iterable[str | Path],
    expected_outputs: Iterable[str | Path],
    manifest_path: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    project_root = Path(project_root)
    expected_artifact_paths = [Path(path) for path in expected_artifact_paths]
    expected_figures = [Path(path) for path in expected_figures]
    expected_tables = [Path(path) for path in expected_tables]
    expected_outputs = [Path(path) for path in expected_outputs]

    missing_artifacts = [str(path.relative_to(project_root)) for path in expected_artifact_paths if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError("Missing Chapter 2 artifacts:\n" + "\n".join(missing_artifacts))

    empty_artifacts = [str(path.relative_to(project_root)) for path in expected_artifact_paths if path.stat().st_size == 0]
    if empty_artifacts:
        raise FileNotFoundError("Empty Chapter 2 artifacts:\n" + "\n".join(empty_artifacts))

    artifact_manifest = pd.DataFrame(
        {
            "artifact": [str(path.relative_to(project_root)) for path in expected_artifact_paths],
            "kind": ["figure"] * len(expected_figures) + ["table"] * len(expected_tables) + ["output"] * len(expected_outputs),
            "bytes": [path.stat().st_size for path in expected_artifact_paths],
        }
    )
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest.to_csv(manifest_path, index=False)
    if manifest_path.stat().st_size == 0:
        raise FileNotFoundError(f"Empty Chapter 2 artifact manifest: {manifest_path.relative_to(project_root)}")
    return artifact_manifest
