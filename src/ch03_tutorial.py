from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .artifacts import (
    display_saved_figure,
    display_saved_figures,
    display_table,
    figure_paths_from_name,
    json_ready,
    safe_relpath,
    save_csv,
    save_figure,
    save_json,
    save_paper_table,
)


@dataclass(frozen=True)
class Ch03Context:
    project_root: Path
    fig_dir: Path
    table_dir: Path
    output_dir: Path


def _append_unique(bucket: list[str], item: str) -> None:
    if item not in bucket:
        bucket.append(item)


def _torch_module_base():
    try:
        import torch

        return torch.nn.Module
    except Exception:
        return object


class EndpointODEFunc(_torch_module_base()):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nfe = 0

    def reset_nfe(self) -> None:
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        tt = x.new_ones((x.shape[0], 1)) * t
        return self.model(x, tt)


@dataclass
class Ch03ArtifactTracker:
    context: Ch03Context
    paper_figure_mode: bool = True
    figures_written: list[str] = field(default_factory=list)
    paper_ready_png_written: list[str] = field(default_factory=list)
    paper_ready_pdf_written: list[str] = field(default_factory=list)
    tables_written: list[str] = field(default_factory=list)
    paper_tables_written: list[str] = field(default_factory=list)

    def rel(self, path: str | Path) -> str:
        return safe_relpath(path, root=self.context.project_root)

    def save_csv(self, frame, filename: str | Path) -> Path:
        path = save_csv(self.context.table_dir / filename, pd.DataFrame(frame))
        _append_unique(self.tables_written, self.rel(path))
        return path

    def save_run_json(self, payload: Any, filename: str | Path) -> Path:
        return save_json(self.context.output_dir / filename, payload)

    def display_saved_figure(self, path: str | Path, width: int | None = None) -> Path:
        return display_saved_figure(path, width=width)

    def display_saved_figures(self, paths: Iterable[str | Path], width: int | None = None) -> list[Path]:
        return display_saved_figures(paths, width=width)

    def display_table(self, frame, columns: list[str] | None = None, n: int = 10) -> pd.DataFrame:
        return display_table(pd.DataFrame(frame), columns=columns, n=n)

    def save_figure(self, fig, filename: str | Path, dpi: int = 300, write_pdf: bool | None = None) -> Path:
        import matplotlib.pyplot as plt

        if write_pdf is None:
            write_pdf = self.paper_figure_mode
        png_path = save_figure(fig, self.context.fig_dir, filename, dpi=dpi, write_pdf=write_pdf)
        _append_unique(self.figures_written, self.rel(png_path))
        if write_pdf:
            _append_unique(self.paper_ready_pdf_written, self.rel(png_path.with_suffix(".pdf")))
        plt.close(fig)
        return png_path

    def save_paper_figure(self, fig, stem: str | Path, dpi: int = 300) -> tuple[Path, Path]:
        import matplotlib.pyplot as plt

        png_path = save_figure(fig, self.context.fig_dir, stem, dpi=dpi, write_pdf=True)
        pdf_path = png_path.with_suffix(".pdf")
        _append_unique(self.paper_ready_png_written, self.rel(png_path))
        _append_unique(self.paper_ready_pdf_written, self.rel(pdf_path))
        _append_unique(self.figures_written, self.rel(png_path))
        plt.close(fig)
        return png_path, pdf_path

    def save_paper_table(self, frame, stem: str | Path, index: bool = False) -> tuple[Path, Path, Path]:
        paths = save_paper_table(self.context.table_dir / stem, pd.DataFrame(frame), index=index)
        for path in paths:
            _append_unique(self.paper_tables_written, self.rel(path))
        return paths


def resolve_project_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    candidates = [start_path, *start_path.parents]
    candidates.extend(
        [
            Path("/home/xmabs/flow_matching_for_dynamic_biology/flow_matching_for_dynamic_biology"),
            Path("/import/home4/xmabs/flow_matching_for_dynamic_biology/flow_matching_for_dynamic_biology"),
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "models.py").exists() and (candidate / "notebooks").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate project root from {start_path}")


def make_ch03_context(project_root: str | Path | None = None) -> Ch03Context:
    root = resolve_project_root(project_root)
    fig_dir = root / "figures" / "ch03"
    table_dir = root / "tables" / "ch03"
    output_dir = root / "outputs" / "ch03"
    for path in [fig_dir, table_dir, output_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return Ch03Context(project_root=root, fig_dir=fig_dir, table_dir=table_dir, output_dir=output_dir)


PAPER_COLORS = {
    "source": "#4C78A8",
    "target": "#B8B8B8",
    "target_red": "#C44E52",
    "generated": "#2F7F73",
    "cfm": "#2F7FBD",
    "cnf": "#D55E00",
    "euler": "#4C78A8",
    "midpoint": "#55A868",
    "dopri5": "#C44E52",
    "low": "#2F7F73",
    "high": "#C44E52",
}


def set_paper_style(base_font_size: float = 8.5) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="white",
        font="DejaVu Sans",
        rc={
            "font.family": "DejaVu Sans",
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 0.8,
            "axes.labelsize": base_font_size,
            "xtick.labelsize": base_font_size - 1.0,
            "ytick.labelsize": base_font_size - 1.0,
            "legend.fontsize": base_font_size - 1.0,
            "figure.titlesize": base_font_size + 1.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    plt.rcParams["axes.grid"] = False


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", color="0.12")


def short_strategy_label(strategy: str) -> str:
    return {
        "uniform": "uniform",
        "logit_normal_sigma_0.5": "logit sigma=.5",
        "logit_normal_sigma_1.0": "logit sigma=1",
        "logit_normal_sigma_2.0": "logit sigma=2",
        "beta_2_2": "beta(2,2)",
        "beta_0.5_0.5": "beta(.5,.5)",
        "cosine": "cosine",
    }.get(str(strategy), str(strategy))


def round_float(values, decimals: int):
    return pd.to_numeric(values, errors="coerce").round(int(decimals))


def format_solver_diagnostics_paper_table(solver_table) -> pd.DataFrame:
    solver_table = pd.DataFrame(solver_table)
    solver_paper = pd.DataFrame()
    solver_paper["Solver"] = solver_table["sampler"].astype(str)
    solver_paper["Steps"] = solver_table["steps"].astype(str)
    solver_paper["NFE"] = solver_table["nfe"].astype(int)
    solver_paper["Time (ms)"] = round_float(1000.0 * solver_table["wall_time_sec"], 1)
    solver_paper["MMD (20D) ↓"] = round_float(solver_table["mmd_20d"], 4)
    solver_paper["Sliced W2 (20D) ↓"] = round_float(solver_table["sliced_w2_20d"], 3)
    solver_paper["Straightness (20D)"] = round_float(solver_table["trajectory_straightness_20d"], 3).astype(object)
    solver_paper.loc[solver_table["sampler"].astype(str).eq("dopri5"), "Straightness (20D)"] = "N/A"
    return solver_paper[
        [
            "Solver",
            "Steps",
            "NFE",
            "Time (ms)",
            "MMD (20D) ↓",
            "Sliced W2 (20D) ↓",
            "Straightness (20D)",
        ]
    ]


def clean_spines(ax, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="0.90", linewidth=0.55)


def format_metric_axis(ax, metric: str) -> None:
    labels = {
        "mmd_20d": "MMD to target (20D PCs)",
        "endpoint_mmd_20d": "Endpoint MMD (20D PCs)",
        "sliced_w2_20d": "Sliced W2 (20D PCs)",
        "val_mse_20d": "Val CFM MSE (20D PCs)",
        "train_mse_20d": "Train CFM MSE (20D PCs)",
        "straightness_ratio_20d": "Straightness ratio in 20D PCs",
    }
    ax.set_ylabel(labels.get(metric, metric.replace("_", " ")))
    clean_spines(ax, grid_axis="y")


def add_note(ax, text: str, loc: str = "lower left") -> None:
    xy = {
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
    }[loc]
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=xy[2],
        va=xy[3],
        fontsize=7.0,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 2.0, "alpha": 0.88},
    )


def as_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def subsample_idx(n: int, max_n: int | None = None, seed: int = 42) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    local_rng = np.random.default_rng(seed)
    return np.sort(local_rng.choice(n, size=int(max_n), replace=False))


def robust_limits(*arrays, q_low: float = 1.0, q_high: float = 99.0, margin: float = 0.08):
    chunks = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        chunks.append(arr.reshape(-1, arr.shape[-1])[:, :2])
    if not chunks:
        return (-1.0, 1.0), (-1.0, 1.0)
    X = np.vstack(chunks)
    X = X[np.isfinite(X).all(axis=1)]
    lo = np.percentile(X, q_low, axis=0)
    hi = np.percentile(X, q_high, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - margin * span
    hi = hi + margin * span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def format_axis(ax, xlim=None, ylim=None, xlabel: str = "state 1", ylabel: str = "state 2", title: str | None = None) -> None:
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_and_close_figure(
    fig,
    fig_dir: str | Path,
    filename: str | Path,
    *,
    dpi: int = 300,
    write_pdf: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    path = save_figure(fig, fig_dir, filename, dpi=dpi, write_pdf=write_pdf)
    plt.close(fig)
    return path


def write_required_artifact_manifest(
    manifest_path: str | Path,
    *,
    expected_figures: Iterable[str | Path] = (),
    expected_tables: Iterable[str | Path] = (),
    expected_outputs: Iterable[str | Path] = (),
) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    outputs_without_manifest = [
        Path(path)
        for path in expected_outputs
        if Path(path).resolve() != manifest_path.resolve()
    ]

    artifact_manifest = check_required_artifacts(
        expected_figures=expected_figures,
        expected_tables=expected_tables,
        expected_outputs=outputs_without_manifest,
    )
    save_csv(manifest_path, artifact_manifest)

    artifact_manifest = check_required_artifacts(
        expected_figures=expected_figures,
        expected_tables=expected_tables,
        expected_outputs=[*outputs_without_manifest, manifest_path],
    )
    save_csv(manifest_path, artifact_manifest)
    return artifact_manifest


def make_eight_gaussians(n=1500, radius=2.0, noise=0.08, seed=42):
    local_rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    centers = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    component = local_rng.integers(0, 8, size=n)
    points = centers[component] + local_rng.normal(scale=noise, size=(n, 2))
    return points.astype(np.float32), component


def make_single_gaussian(n=1500, loc=(0.0, 0.0), scale=(0.42, 0.28), angle=0.35, seed=43):
    local_rng = np.random.default_rng(seed)
    raw = local_rng.normal(size=(n, 2)).astype(np.float32)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    scaling = np.diag(np.asarray(scale, dtype=np.float32))
    points = raw @ scaling @ rotation.T + np.asarray(loc, dtype=np.float32)
    return points.astype(np.float32)


def make_random_pair_batch_fn(X0, X1, seed=42):
    X0 = np.asarray(X0, dtype=np.float32)
    X1 = np.asarray(X1, dtype=np.float32)
    local_rng = np.random.default_rng(seed)

    def pair_batch_fn(batch_size):
        idx0 = local_rng.integers(0, len(X0), size=int(batch_size))
        idx1 = local_rng.integers(0, len(X1), size=int(batch_size))
        return {"x0": X0[idx0].astype(np.float32), "x1": X1[idx1].astype(np.float32)}

    return pair_batch_fn


def train_val_indices(n: int, train_fraction: float = 0.8, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if n < 2:
        raise ValueError(f"Need at least two observations to create a train/validation split; got {n}.")
    local_rng = np.random.default_rng(seed)
    perm = local_rng.permutation(n)
    n_train = int(np.floor(float(train_fraction) * n))
    n_train = min(max(n_train, 1), n - 1)
    train = np.sort(perm[:n_train])
    val = np.sort(perm[n_train:])
    return train, val


def val_cfm_mse(
    model,
    x0_np,
    x1_np,
    t_values,
    *,
    device: str = "cpu",
    max_eval_pairs: int | None = None,
    seed: int = 42,
) -> float:
    import torch

    from .losses import cfm_loss_from_pairs

    x0_np = np.asarray(x0_np, dtype=np.float32)
    x1_np = np.asarray(x1_np, dtype=np.float32)
    if len(x0_np) != len(x1_np):
        raise ValueError(f"x0_np and x1_np must have the same length; got {len(x0_np)} and {len(x1_np)}.")
    if max_eval_pairs is not None and len(x0_np) > int(max_eval_pairs):
        idx = subsample_idx(len(x0_np), int(max_eval_pairs), seed=seed)
        x0_np = x0_np[idx]
        x1_np = x1_np[idx]

    was_training = bool(getattr(model, "training", False))
    model.eval()
    x0_t = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
    x1_t = torch.as_tensor(x1_np, dtype=torch.float32, device=device)
    losses = []
    with torch.no_grad():
        for t_value in np.asarray(t_values, dtype=np.float32):
            t_t = torch.full((x0_t.shape[0], 1), float(t_value), dtype=torch.float32, device=device)
            loss = cfm_loss_from_pairs(model, x0_t, x1_t, s=t_t)
            losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return float(np.mean(losses))


def record_skip(skipped_items: list[dict[str, str]], item: Any, reason: Any) -> None:
    skipped_items.append({"item": str(item), "reason": str(reason)})


def endpoint_mmd_sliced_20d(
    samples_20d,
    target_20d,
    *,
    seed: int = 42,
    n_projections: int = 128,
) -> dict[str, float]:
    from .metrics import mmd_rbf, sliced_wasserstein_distance

    return {
        "endpoint_mmd_20d": float(mmd_rbf(samples_20d, target_20d)),
        "sliced_w2_20d": float(
            sliced_wasserstein_distance(
                samples_20d,
                target_20d,
                n_projections=int(n_projections),
                seed=seed,
            )
        ),
    }


def eval_cfm_endpoint_20d(
    model,
    x0_np,
    target_np,
    *,
    n_steps: int = 30,
    seed: int = 42,
    device: str = "cpu",
    n_projections: int = 128,
) -> tuple[np.ndarray, int, dict[str, float]]:
    import torch

    from .sampling import euler_sample

    model.eval()
    with torch.no_grad():
        x0_t = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
        endpoint_t, _, nfe = euler_sample(model, x0_t, n_steps=int(n_steps), return_traj=False)
    endpoint_np = as_np(endpoint_t)
    metrics = endpoint_mmd_sliced_20d(endpoint_np, target_np, seed=seed, n_projections=n_projections)
    return endpoint_np, int(nfe), metrics


def eval_ode_endpoint_20d(
    model,
    x0_np,
    target_np,
    *,
    rtol: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    n_projections: int = 128,
) -> tuple[np.ndarray, int, dict[str, float]]:
    import torch

    from .sampling import odeint_sample

    model.eval()
    with torch.no_grad():
        x0_t = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
        endpoint_t, _, nfe = odeint_sample(model, x0_t, rtol=float(rtol), atol=float(rtol), method="dopri5")
    endpoint_np = as_np(endpoint_t)
    metrics = endpoint_mmd_sliced_20d(endpoint_np, target_np, seed=seed, n_projections=n_projections)
    return endpoint_np, int(nfe), metrics


def sample_t_numpy(
    strategy: str,
    n: int,
    *,
    seed: int = 42,
    strategy_specs: Mapping[str, Mapping[str, float]] | None = None,
) -> np.ndarray:
    local_rng = np.random.default_rng(seed)
    strategy = str(strategy)
    if strategy == "uniform":
        t = local_rng.random(int(n))
    elif strategy.startswith("logit_normal"):
        if strategy_specs is not None and strategy in strategy_specs:
            sigma = float(strategy_specs[strategy]["sigma"])
        else:
            sigma = float(strategy.rsplit("_", 1)[-1])
        z = local_rng.normal(loc=0.0, scale=sigma, size=int(n))
        t = 1.0 / (1.0 + np.exp(-z))
    elif strategy == "beta_2_2":
        t = local_rng.beta(2.0, 2.0, size=int(n))
    elif strategy == "beta_0.5_0.5":
        t = local_rng.beta(0.5, 0.5, size=int(n))
    elif strategy == "cosine":
        u = local_rng.random(int(n))
        t = 0.5 * (1.0 - np.cos(np.pi * u))
    else:
        raise ValueError(strategy)
    return np.clip(np.asarray(t, dtype=np.float32), 1e-4, 1.0 - 1e-4)


def sample_t_torch(
    strategy: str,
    batch_size: int,
    device,
    *,
    seed: int | None = None,
    strategy_specs: Mapping[str, Mapping[str, float]] | None = None,
):
    import torch

    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    arr = sample_t_numpy(strategy, int(batch_size), seed=seed, strategy_specs=strategy_specs)
    return torch.as_tensor(arr[:, None], dtype=torch.float32, device=device)


def per_trajectory_straightness(traj, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 3:
        raise ValueError("traj must have shape (T, N, D)")
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    path_len = steps.sum(axis=0)
    endpoint = np.linalg.norm(traj[-1] - traj[0], axis=-1)
    ratio = path_len / np.maximum(endpoint, float(eps))
    return path_len, endpoint, ratio


def normalize_skipped_items(items: Iterable[Any]) -> list[dict[str, str]]:
    normalized = []
    for item in items:
        if isinstance(item, dict):
            normalized.append({"item": str(item.get("item", "unknown")), "reason": str(item.get("reason", ""))})
        else:
            normalized.append({"item": str(item), "reason": "recorded by earlier section"})
    return normalized


def plot_endpoint_pairs_phate(
    X0_plot,
    X1_plot,
    *,
    n_pairs: int = 80,
    seed: int = 42,
    source_time: str | int | None = None,
    target_time: str | int | None = None,
):
    import matplotlib.pyplot as plt

    X0_plot = np.asarray(X0_plot, dtype=np.float32)
    X1_plot = np.asarray(X1_plot, dtype=np.float32)
    if len(X0_plot) == 0 or len(X1_plot) == 0:
        raise ValueError("Endpoint-pair plot requires non-empty source and target arrays.")

    local_rng = np.random.default_rng(seed)
    n_pairs = min(int(n_pairs), max(len(X0_plot), len(X1_plot)))
    idx0 = local_rng.integers(0, len(X0_plot), size=n_pairs)
    idx1 = local_rng.integers(0, len(X1_plot), size=n_pairs)
    x0p = X0_plot[idx0]
    x1p = X1_plot[idx1]
    xlim, ylim = robust_limits(X0_plot, X1_plot, x0p, x1p, margin=0.10)

    source_label = "source" if source_time is None else f"source time {source_time}"
    target_label = "target" if target_time is None else f"target time {target_time}"
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sidx = subsample_idx(len(X0_plot), 650, seed=91)
    tidx = subsample_idx(len(X1_plot), 650, seed=92)
    ax.scatter(X0_plot[sidx, 0], X0_plot[sidx, 1], s=7, color="#3267A8", alpha=0.38, linewidths=0, label=source_label)
    ax.scatter(X1_plot[tidx, 0], X1_plot[tidx, 1], s=7, color="#C9463D", alpha=0.34, linewidths=0, label=target_label)
    for a, b in zip(x0p, x1p):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.25", alpha=0.24, linewidth=0.55)
    format_axis(ax, xlim, ylim, xlabel="PHATE 1", ylabel="PHATE 2", title="Sampled EB train endpoint pairs displayed in PHATE")
    ax.legend(frameon=False, loc="best")
    add_note(ax, "20D training chords; PHATE display only", loc="lower left")
    return fig


def build_toy_velocity_probe(model, X0, X1, device="cpu", seed=42):
    import torch

    local_rng = np.random.default_rng(seed)
    n_pairs = min(1800, len(X0), len(X1))
    i0 = local_rng.integers(0, len(X0), size=n_pairs)
    i1 = local_rng.integers(0, len(X1), size=n_pairs)
    x0 = np.asarray(X0, dtype=np.float32)[i0]
    x1 = np.asarray(X1, dtype=np.float32)[i1]
    t_value = 0.5
    x_t = (1.0 - t_value) * x0 + t_value * x1
    u_t = x1 - x0

    center = np.median(x_t, axis=0)
    dist = np.linalg.norm(x_t - center[None, :], axis=1)
    radius = np.quantile(dist, 0.18)
    local = np.flatnonzero(dist <= max(radius, 1e-6))
    if len(local) > 120:
        local = local[subsample_idx(len(local), 120, seed=seed + 1)]

    model.eval()
    with torch.no_grad():
        center_t = torch.as_tensor(center[None, :], dtype=torch.float32, device=device)
        time_t = torch.full((1, 1), t_value, dtype=torch.float32, device=device)
        pred = model(center_t, time_t).detach().cpu().numpy()[0]
    mean_conditional = u_t[local].mean(axis=0)
    return {
        "x_t": x_t,
        "u_t": u_t,
        "local": local,
        "center": center,
        "t_value": t_value,
        "network_velocity": pred,
        "mean_conditional_velocity": mean_conditional,
    }


def plot_toy_conditional_vs_marginal(model, X0, X1, probe):
    import matplotlib.pyplot as plt

    x_t = probe["x_t"]
    u_t = probe["u_t"]
    local = probe["local"]
    center = probe["center"]
    pred = probe["network_velocity"]
    mean_conditional = probe["mean_conditional_velocity"]

    xlim, ylim = robust_limits(X0, X1, x_t[local], margin=0.12)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    source_idx = subsample_idx(len(X0), 550, seed=71)
    target_idx = subsample_idx(len(X1), 550, seed=72)
    ax.scatter(X0[source_idx, 0], X0[source_idx, 1], s=5, color="#4C78A8", alpha=0.18, linewidths=0, label="source")
    ax.scatter(X1[target_idx, 0], X1[target_idx, 1], s=5, color="#D1495B", alpha=0.18, linewidths=0, label="target")
    ax.scatter(x_t[local, 0], x_t[local, 1], s=11, color="0.55", alpha=0.35, linewidths=0)
    ax.quiver(
        x_t[local, 0],
        x_t[local, 1],
        u_t[local, 0],
        u_t[local, 1],
        angles="xy",
        scale_units="xy",
        scale=12,
        width=0.003,
        color="#9CC9C2",
        alpha=0.52,
        label="conditional velocities",
    )
    ax.quiver(
        [center[0]],
        [center[1]],
        [mean_conditional[0]],
        [mean_conditional[1]],
        angles="xy",
        scale_units="xy",
        scale=5,
        width=0.010,
        color="0.18",
        alpha=0.85,
        label="local conditional average",
    )
    ax.quiver(
        [center[0]],
        [center[1]],
        [pred[0]],
        [pred[1]],
        angles="xy",
        scale_units="xy",
        scale=5,
        width=0.013,
        color="#B9352B",
        alpha=0.95,
        label="network prediction",
    )
    ax.scatter([center[0]], [center[1]], s=42, color="#B9352B", edgecolor="white", linewidth=0.7, zorder=5)
    format_axis(ax, xlim, ylim, xlabel="toy state 1", ylabel="toy state 2", title="Conditional velocities collapse to a marginal vector")
    ax.legend(frameon=False, loc="best")
    return fig


def prepare_toy_hierarchy_objects(model, X0, X1, device="cpu", seed=42):
    import torch

    local_rng = np.random.default_rng(seed)
    n_paths = min(90, len(X0), len(X1))
    i0 = local_rng.integers(0, len(X0), size=n_paths)
    i1 = local_rng.integers(0, len(X1), size=n_paths)
    x0 = np.asarray(X0, dtype=np.float32)[i0]
    x1 = np.asarray(X1, dtype=np.float32)[i1]
    xlim, ylim = robust_limits(X0, X1, x0, x1, margin=0.12)

    grid_n = 20
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)

    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(pts, dtype=torch.float32, device=device)
        t_t = torch.full((pts.shape[0], 1), 0.5, dtype=torch.float32, device=device)
        v = model(x_t, t_t).detach().cpu().numpy()
    norm = np.linalg.norm(v, axis=1)
    cap = np.percentile(norm[norm > 0], 88) if np.any(norm > 0) else 1.0
    v = v * np.minimum(1.0, cap / np.clip(norm[:, None], 1e-12, None))
    return {"x0": x0, "x1": x1, "xlim": xlim, "ylim": ylim, "grid_points": pts, "grid_velocity": v}


def plot_toy_cfm_object_hierarchy(X0, X1, hierarchy):
    import matplotlib.pyplot as plt

    x0 = hierarchy["x0"]
    x1 = hierarchy["x1"]
    xlim = hierarchy["xlim"]
    ylim = hierarchy["ylim"]
    pts = hierarchy["grid_points"]
    v = hierarchy["grid_velocity"]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharex=True, sharey=True)
    source_idx = subsample_idx(len(X0), 500, seed=81)
    target_idx = subsample_idx(len(X1), 500, seed=82)
    for ax in axes:
        ax.scatter(X0[source_idx, 0], X0[source_idx, 1], s=5, color="#4C78A8", alpha=0.14, linewidths=0)
        ax.scatter(X1[target_idx, 0], X1[target_idx, 1], s=5, color="#D1495B", alpha=0.14, linewidths=0)
        format_axis(ax, xlim, ylim, xlabel="toy state 1", ylabel="toy state 2")

    a, b = x0[0], x1[0]
    path = np.stack([(1.0 - t) * a + t * b for t in np.linspace(0, 1, 50)], axis=0)
    axes[0].plot(path[:, 0], path[:, 1], color="0.18", linewidth=1.6)
    axes[0].scatter([a[0], b[0]], [a[1], b[1]], s=40, color=["#4C78A8", "#D1495B"], edgecolor="white", linewidth=0.7, zorder=4)
    mid = 0.5 * (a + b)
    vel = b - a
    axes[0].quiver([mid[0]], [mid[1]], [vel[0]], [vel[1]], angles="xy", scale_units="xy", scale=4, width=0.012, color="#2F6B5E")
    axes[0].set_title("A. one endpoint path")

    for j in range(len(x0)):
        axes[1].plot([x0[j, 0], x1[j, 0]], [x0[j, 1], x1[j, 1]], color="0.25", alpha=0.18, linewidth=0.65)
    axes[1].set_title("B. many endpoint paths")
    axes[1].set_ylabel("")

    axes[2].quiver(pts[:, 0], pts[:, 1], v[:, 0], v[:, 1], angles="xy", scale_units="xy", scale=16, width=0.004, color="#B9352B", alpha=0.72)
    axes[2].set_title("C. learned marginal field at t=0.5")
    axes[2].set_ylabel("")
    fig.suptitle("CFM object hierarchy on a 2D toy problem")
    return fig


def check_required_artifacts(
    *,
    expected_figures: Iterable[str | Path] = (),
    expected_tables: Iterable[str | Path] = (),
    expected_outputs: Iterable[str | Path] = (),
) -> pd.DataFrame:
    rows = []
    for kind, paths in [
        ("figure", expected_figures),
        ("table", expected_tables),
        ("output", expected_outputs),
    ]:
        for path in paths:
            artifact = Path(path)
            rows.append(
                {
                    "kind": kind,
                    "path": str(artifact),
                    "exists": artifact.exists(),
                    "bytes": artifact.stat().st_size if artifact.exists() else 0,
                }
            )
    manifest = pd.DataFrame(rows)
    missing = manifest.loc[~manifest["exists"]] if not manifest.empty else manifest
    if not missing.empty:
        raise FileNotFoundError("Missing required Chapter 3 artifacts:\n" + missing.to_string(index=False))
    empty = manifest.loc[manifest["bytes"].eq(0)] if not manifest.empty else manifest
    if not empty.empty:
        raise FileNotFoundError("Empty required Chapter 3 artifacts:\n" + empty.to_string(index=False))
    return manifest


def relative_paths(paths: Iterable[str | Path], root: str | Path) -> list[str]:
    root = Path(root).resolve()
    rels = []
    for path in paths:
        path = Path(path).resolve()
        rels.append(str(path.relative_to(root) if path.is_relative_to(root) else path))
    return rels
