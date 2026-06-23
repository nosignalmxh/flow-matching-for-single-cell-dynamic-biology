from __future__ import annotations

import numpy as np
import pandas as pd


def _as_points(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("points must be a 1D or 2D array")
    return arr


def _as_time(t, n: int):
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        t = np.full(n, float(t))
    t = t.reshape(-1)
    if t.size == 1 and n != 1:
        t = np.full(n, float(t[0]))
    if t.size != n:
        raise ValueError(f"t must have length 1 or {n}, got {t.size}")
    return t.reshape(-1, 1)


def _normal_vectors(x0, x1, direction: str = "normal"):
    d = x1 - x0
    if d.shape[1] < 2:
        return np.zeros_like(d)
    normal_2d = np.stack([-d[:, 1], d[:, 0]], axis=1)
    norm = np.linalg.norm(normal_2d, axis=1, keepdims=True)
    normal_2d = normal_2d / np.clip(norm, 1e-12, None)
    if direction == "negative":
        normal_2d = -normal_2d
    elif direction not in {"normal", "positive"}:
        raise ValueError("direction must be one of 'normal', 'positive', or 'negative'")
    normal = np.zeros_like(d)
    normal[:, :2] = normal_2d
    return normal


def linear_path(x0, x1, t):
    x0 = _as_points(x0)
    x1 = _as_points(x1)
    t = _as_time(t, len(x0))
    xt = (1.0 - t) * x0 + t * x1
    ut = x1 - x0
    return xt, ut


def noisy_path(x0, x1, t, noise_scale: float = 0.05, seed: int = 42):
    xt, ut = linear_path(x0, x1, t)
    rng = np.random.default_rng(seed)
    bump = np.sin(np.pi * _as_time(t, len(xt)))
    return xt + bump * rng.normal(scale=noise_scale, size=xt.shape), ut


def curved_path(x0, x1, t, curvature: float = 0.35, direction: str = "normal"):
    """Deterministic curved bridge with endpoints fixed."""
    x0 = _as_points(x0)
    x1 = _as_points(x1)
    t = _as_time(t, len(x0))
    base = (1.0 - t) * x0 + t * x1
    d = x1 - x0
    distance = np.linalg.norm(d, axis=1, keepdims=True)
    normal = _normal_vectors(x0, x1, direction=direction)
    return base + float(curvature) * np.sin(np.pi * t) * distance * normal


def brownian_bridge_path(x0, x1, t, sigma: float = 0.12, seed: int = 42):
    """Sample stochastic bridge points with variance sigma^2 t(1-t)."""
    x0 = _as_points(x0)
    x1 = _as_points(x1)
    t = _as_time(t, len(x0))
    mean = (1.0 - t) * x0 + t * x1
    std = float(sigma) * np.sqrt(np.clip(t * (1.0 - t), 0.0, None))
    rng = np.random.default_rng(seed)
    return mean + std * rng.normal(size=mean.shape)


def path_velocity_linear(x0, x1, t=None):
    """Return x1 - x0."""
    x0 = _as_points(x0)
    x1 = _as_points(x1)
    return x1 - x0


def path_velocity_curved(x0, x1, t, curvature: float = 0.35):
    """Return analytic velocity for curved_path."""
    x0 = _as_points(x0)
    x1 = _as_points(x1)
    t = _as_time(t, len(x0))
    d = x1 - x0
    distance = np.linalg.norm(d, axis=1, keepdims=True)
    normal = _normal_vectors(x0, x1)
    return d + float(curvature) * np.pi * np.cos(np.pi * t) * distance * normal


def path_length(points):
    points = np.asarray(points)
    return np.linalg.norm(np.diff(points, axis=0), axis=-1).sum(axis=0)


def path_energy(x0, x1):
    return ((np.asarray(x1) - np.asarray(x0)) ** 2).sum(axis=-1).mean()


def path_diagnostics(x0, x1, path_points_by_tau: dict[float, np.ndarray]) -> pd.DataFrame:
    """Return path length, energy proxy, and midpoint spread diagnostics."""
    x0 = _as_points(x0)
    x1 = _as_points(x1)
    if not path_points_by_tau:
        raise ValueError("path_points_by_tau cannot be empty")

    taus = sorted(float(tau) for tau in path_points_by_tau)
    stacked = np.stack([_as_points(path_points_by_tau[tau]) for tau in taus], axis=0)
    if stacked.shape[1:] != x0.shape:
        raise ValueError("all path point arrays must match x0 shape")

    segment_lengths = np.linalg.norm(np.diff(stacked, axis=0), axis=-1)
    per_pair_length = segment_lengths.sum(axis=0)
    dt = np.diff(np.asarray(taus, dtype=float))
    dt = np.clip(dt, 1e-12, None)
    velocities = np.diff(stacked, axis=0) / dt[:, None, None]
    energy_proxy = float(np.mean(np.sum(velocities**2, axis=-1)))

    midpoint_tau = min(taus, key=lambda tau: abs(tau - 0.5))
    midpoint = _as_points(path_points_by_tau[midpoint_tau])
    straight_midpoint = 0.5 * (x0 + x1)
    rows = [
        {
            "mean_endpoint_distance": float(np.mean(np.linalg.norm(x1 - x0, axis=1))),
            "mean_path_length": float(np.mean(per_pair_length)),
            "energy_proxy": energy_proxy,
            "midpoint_spread": float(np.mean(np.linalg.norm(midpoint - straight_midpoint, axis=1))),
        }
    ]
    return pd.DataFrame(rows)
