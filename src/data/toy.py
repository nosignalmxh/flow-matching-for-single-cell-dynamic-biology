from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ToySnapshots:
    X: np.ndarray
    time: np.ndarray
    condition: np.ndarray
    fate_label: np.ndarray
    cell_id: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_id": self.cell_id,
                "state_1": self.X[:, 0],
                "state_2": self.X[:, 1],
                "x0": self.X[:, 0],
                "x1": self.X[:, 1],
                "time": self.time,
                "condition": self.condition,
                "fate_label": self.fate_label,
            }
        )


def _validate_branching_args(
    n_cells: int,
    timepoints: Iterable[float],
    rare_fate_fraction: float,
    branch_time: float,
    noise: float,
) -> np.ndarray:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    times = np.asarray(list(timepoints), dtype=float)
    if times.size == 0:
        raise ValueError("timepoints must contain at least one value")
    if np.any(~np.isfinite(times)):
        raise ValueError("timepoints must be finite")
    if not 0.0 <= rare_fate_fraction <= 1.0:
        raise ValueError("rare_fate_fraction must be in [0, 1]")
    if not 0.0 < branch_time < 1.0:
        raise ValueError("branch_time must be between 0 and 1")
    if noise < 0.0:
        raise ValueError("noise must be non-negative")
    return times


def _counts_per_timepoint(n_cells: int, n_timepoints: int) -> np.ndarray:
    counts = np.full(n_timepoints, n_cells // n_timepoints, dtype=int)
    counts[: n_cells % n_timepoints] += 1
    return counts


def _branching_centers(
    t: float,
    is_rare: np.ndarray,
    branch_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t < branch_time * 0.75:
        progress = t / max(branch_time, 1e-6)
        x = -1.15 + 0.85 * progress
        y = np.zeros_like(is_rare, dtype=float)
        fate = np.full(is_rare.shape, "progenitor", dtype=object)
    elif t < branch_time:
        progress = (t - branch_time * 0.75) / max(branch_time * 0.25, 1e-6)
        x = -0.3 + 0.18 * progress
        y = np.zeros_like(is_rare, dtype=float)
        fate = np.full(is_rare.shape, "transition", dtype=object)
    else:
        branch_progress = (t - branch_time) / max(1.0 - branch_time, 1e-6)
        x = -0.12 + 1.32 * branch_progress
        y = np.where(is_rare, -0.85 * branch_progress, 0.85 * branch_progress)
        fate = np.where(is_rare, "rare", "major").astype(object)
    return np.full(is_rare.shape, x, dtype=float), y, fate


def make_branching_snapshot_toy(
    n_cells: int = 4000,
    timepoints: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    rare_fate_fraction: float = 0.12,
    branch_time: float = 0.45,
    noise: float = 0.08,
    condition: str = "control",
    condition_effect: str | None = None,
    seed: int = 42,
) -> ToySnapshots:
    """Generate destructive 2D branching population snapshots.

    Cells are sampled independently at each timepoint. ``cell_id`` is a sampled
    snapshot identifier and does not encode longitudinal identity.
    """
    times = _validate_branching_args(n_cells, timepoints, rare_fate_fraction, branch_time, noise)
    rng = np.random.default_rng(seed)
    counts = _counts_per_timepoint(n_cells, len(times))

    xs, ts, cs, fs, ids = [], [], [], [], []
    start = 0
    for t, n_t in zip(times, counts):
        rare_frac = rare_fate_fraction
        speed = 1.0
        cond = condition
        if condition_effect == "more_rare_fate":
            rare_frac = min(0.45, rare_fate_fraction * 2.0)
            cond = "perturbed"
        elif condition_effect == "slower":
            speed = 0.75
            cond = "perturbed"
        elif condition_effect is not None:
            raise ValueError("condition_effect must be one of None, 'more_rare_fate', or 'slower'")

        tau = np.clip(t * speed, 0.0, 1.0)
        is_rare = rng.random(n_t) < rare_frac
        x, y, fate = _branching_centers(float(tau), is_rare, branch_time)

        points = np.stack([x, y], axis=1)
        points += rng.normal(scale=noise, size=points.shape)
        xs.append(points)
        ts.append(np.full(n_t, t))
        cs.append(np.full(n_t, cond))
        fs.append(fate)
        ids.append(np.asarray([f"cell_{i:06d}" for i in range(start, start + n_t)], dtype=object))
        start += n_t

    return ToySnapshots(
        X=np.concatenate(xs, axis=0).astype("float32"),
        time=np.concatenate(ts, axis=0).astype("float32"),
        condition=np.concatenate(cs, axis=0),
        fate_label=np.concatenate(fs, axis=0),
        cell_id=np.concatenate(ids, axis=0),
    )


def make_hidden_branching_paths(
    n_paths: int = 120,
    time_grid: Iterable[float] | None = None,
    rare_fate_fraction: float = 0.12,
    branch_time: float = 0.45,
    noise: float = 0.02,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate hypothetical same-cell paths for visual explanation only."""
    grid = np.linspace(0.0, 1.0, 40) if time_grid is None else np.asarray(list(time_grid), dtype=float)
    _validate_branching_args(n_paths, grid, rare_fate_fraction, branch_time, noise)
    rng = np.random.default_rng(seed)
    is_rare_path = rng.random(n_paths) < rare_fate_fraction

    rows = []
    for path_idx, is_rare in enumerate(is_rare_path):
        path_noise = rng.normal(scale=noise, size=(len(grid), 2))
        for t_idx, t in enumerate(grid):
            x, y, _ = _branching_centers(float(t), np.asarray([is_rare]), branch_time)
            point = np.array([x[0], y[0]], dtype=float) + path_noise[t_idx]
            rows.append(
                {
                    "path_id": f"path_{path_idx:04d}",
                    "time": float(t),
                    "state_1": float(point[0]),
                    "state_2": float(point[1]),
                    "fate_label": "rare" if is_rare else "major",
                }
            )
    return pd.DataFrame(rows)


def make_y_branching_snapshots(
    n_cells: int = 5000,
    timepoints: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    rare_fate_fraction: float = 0.12,
    condition_effect: str | None = None,
    noise: float = 0.08,
    seed: int = 42,
) -> ToySnapshots:
    """Backward-compatible name for the branching snapshot toy."""
    return make_branching_snapshot_toy(
        n_cells=n_cells,
        timepoints=timepoints,
        rare_fate_fraction=rare_fate_fraction,
        condition_effect=condition_effect,
        noise=noise,
        seed=seed,
    )
