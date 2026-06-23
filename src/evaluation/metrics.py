from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.ot import pairwise_squared_distances


def mmd_rbf(X, Y, gamma: float | None = None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = 1.0 / max(X.shape[1], 1)
    kxx = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1)).mean()
    kyy = np.exp(-gamma * ((Y[:, None] - Y[None]) ** 2).sum(-1)).mean()
    kxy = np.exp(-gamma * ((X[:, None] - Y[None]) ** 2).sum(-1)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def fate_proportion(labels):
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)
    return {str(v): float(c / counts.sum()) for v, c in zip(vals, counts)}


def mean_pair_distance(x0, x1) -> float:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    return float(np.mean(np.sum((x1 - x0) ** 2, axis=1)))


def endpoint_label_mismatch(source_labels, target_labels) -> float:
    """For toy diagnostics only; not biological ground-truth pairing."""
    source_labels = np.asarray(source_labels).astype(str)
    target_labels = np.asarray(target_labels).astype(str)
    if source_labels.shape != target_labels.shape:
        raise ValueError("source_labels and target_labels must have the same shape")
    return float(np.mean(source_labels != target_labels))


def vector_field_energy(vectors) -> float:
    vectors = np.asarray(vectors, dtype=float)
    return float(np.mean(np.sum(vectors**2, axis=1)))


def sliced_wasserstein_distance(X, Y, n_projections: int = 128, seed: int = 42) -> float:
    """Approximate sliced W2 distance between two point clouds."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same feature dimension")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(n_projections), X.shape[1]))
    directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12, None)

    n = min(X.shape[0], Y.shape[0])
    distances = []
    for direction in directions:
        x_proj = np.sort(X @ direction)
        y_proj = np.sort(Y @ direction)
        if X.shape[0] != n:
            x_grid = np.linspace(0.0, 1.0, X.shape[0])
            x_proj = np.interp(np.linspace(0.0, 1.0, n), x_grid, x_proj)
        if Y.shape[0] != n:
            y_grid = np.linspace(0.0, 1.0, Y.shape[0])
            y_proj = np.interp(np.linspace(0.0, 1.0, n), y_grid, y_proj)
        distances.append(np.mean((x_proj - y_proj) ** 2))
    return float(np.sqrt(np.mean(distances)))


def sliced_w2(X, Y, n_projections: int = 128, seed: int = 42) -> float:
    return sliced_wasserstein_distance(X, Y, n_projections=n_projections, seed=seed)


def _median_gamma(X, Y) -> float:
    Z = np.vstack([np.asarray(X, dtype=float), np.asarray(Y, dtype=float)])
    if len(Z) > 512:
        rng = np.random.default_rng(42)
        Z = Z[rng.choice(len(Z), size=512, replace=False)]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)
    positive = d2[d2 > 0]
    if positive.size == 0:
        return 1.0
    return float(1.0 / np.median(positive))


def endpoint_metrics(X_pred, X_target, gamma: float | None = None) -> dict:
    X_pred = np.asarray(X_pred, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    if gamma is None:
        gamma = _median_gamma(X_pred, X_target)
    centroid = X_target.mean(axis=0)
    return {
        "mmd_rbf": mmd_rbf(X_pred, X_target, gamma=gamma),
        "sliced_w2": sliced_wasserstein_distance(X_pred, X_target),
        "mean_l2_to_target_centroid": float(np.linalg.norm(X_pred - centroid[None, :], axis=1).mean()),
    }


def trajectory_stability(traj) -> dict:
    """Return max_abs, mean_step_norm, max_step_norm, has_nan."""
    traj = np.asarray(traj, dtype=float)
    has_nan = bool(np.isnan(traj).any())
    if traj.shape[0] < 2:
        step_norm = np.array([0.0])
    else:
        step_norm = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    return {
        "max_abs": float(np.nanmax(np.abs(traj))) if traj.size else 0.0,
        "mean_step_norm": float(np.nanmean(step_norm)) if step_norm.size else 0.0,
        "max_step_norm": float(np.nanmax(step_norm)) if step_norm.size else 0.0,
        "has_nan": has_nan,
    }


def _validate_traj(traj: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 3:
        raise ValueError("traj must have shape (T, N, D)")
    if traj.shape[0] < 2:
        raise ValueError("traj must contain at least two time points")
    if traj.shape[1] == 0 or traj.shape[2] == 0:
        raise ValueError("traj must contain at least one trajectory and one feature")
    if not np.all(np.isfinite(traj)):
        raise ValueError("traj must contain only finite values")
    return traj


def trajectory_path_length(traj: np.ndarray) -> float:
    """Mean integrated path length over a trajectory array of shape (T, N, D)."""
    traj = _validate_traj(traj)
    step_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    return float(step_lengths.sum(axis=0).mean())


def path_length(traj) -> float:
    return trajectory_path_length(np.asarray(traj, dtype=float))


def trajectory_path_energy(traj: np.ndarray, times: np.ndarray | None = None) -> float:
    """Mean squared-speed action proxy from sampled trajectories."""
    traj = _validate_traj(traj)
    if times is None:
        times = np.linspace(0.0, 1.0, traj.shape[0])
    times = np.asarray(times, dtype=float).reshape(-1)
    if times.shape[0] != traj.shape[0]:
        raise ValueError("times length must match traj.shape[0]")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("times must be strictly increasing")
    dx = np.diff(traj, axis=0)
    speed2 = np.sum((dx / dt[:, None, None]) ** 2, axis=-1)
    return float((speed2 * dt[:, None]).sum(axis=0).mean())


def path_energy(traj, times=None) -> float:
    return trajectory_path_energy(np.asarray(traj, dtype=float), times=times)


def trajectory_straightness(traj: np.ndarray, eps: float = 1e-8) -> float:
    """Mean integrated length divided by endpoint distance minus one."""
    traj = _validate_traj(traj)
    step_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=-1).sum(axis=0)
    endpoint_dist = np.linalg.norm(traj[-1] - traj[0], axis=-1)
    return float(np.mean(step_lengths / np.maximum(endpoint_dist, float(eps)) - 1.0))


def tortuosity_straightness(traj) -> float:
    return trajectory_straightness(np.asarray(traj, dtype=float))


def straightness(traj) -> float:
    return tortuosity_straightness(traj)


def straightness_action_S(traj, times=None) -> float:
    """Planning straightness action against the endpoint chord."""
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 3 or traj.shape[0] < 2:
        raise ValueError("traj must have shape (T, N, D) with T >= 2")
    if times is None:
        times = np.linspace(0.0, 1.0, traj.shape[0])
    times = np.asarray(times, dtype=float)
    dt = np.diff(times)
    if len(dt) != traj.shape[0] - 1 or np.any(dt <= 0):
        raise ValueError("times must be strictly increasing and match traj")
    vel = np.diff(traj, axis=0) / dt[:, None, None]
    chord = traj[-1] - traj[0]
    sq = np.sum((chord[None, :, :] - vel) ** 2, axis=-1)
    return float(np.sum(sq.mean(axis=1) * dt))


def off_manifold_knn_distance(points: np.ndarray, reference: np.ndarray, k: int = 10) -> float:
    """Mean kNN distance from trajectory points to observed states."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if points.ndim == 3:
        points = points.reshape(-1, points.shape[-1])
    if points.ndim != 2 or reference.ndim != 2:
        raise ValueError("points and reference must be 2D arrays, or points may be a (T, N, D) trajectory")
    if points.shape[1] != reference.shape[1]:
        raise ValueError("points and reference must have the same feature dimension")
    if points.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("points and reference must be nonempty")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(reference)):
        raise ValueError("points and reference must contain only finite values")
    k = max(1, min(int(k), reference.shape[0]))
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(reference)
        distances, _ = nn.kneighbors(points)
    except Exception:
        d = np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=-1)
        distances = np.sort(d, axis=1)[:, :k]
    return float(distances.mean())


def off_manifold_knn(points, reference, k: int = 15, batch_size: int = 1000) -> float:
    points_arr = np.asarray(points, dtype=np.float32)
    reference_arr = np.asarray(reference, dtype=np.float32)
    if points_arr.ndim == 3:
        points_arr = points_arr.reshape(-1, points_arr.shape[-1])
    if points_arr.ndim != 2 or reference_arr.ndim != 2:
        raise ValueError("points and reference must be 2D arrays, or points may be a (T, N, D) trajectory")
    if points_arr.shape[1] != reference_arr.shape[1]:
        raise ValueError("points and reference must have the same feature dimension")
    if points_arr.shape[0] <= batch_size:
        return off_manifold_knn_distance(points_arr, reference_arr, k=k)
    from sklearn.neighbors import NearestNeighbors

    kk = max(1, min(int(k), reference_arr.shape[0]))
    nn = NearestNeighbors(n_neighbors=kk, algorithm="ball_tree", leaf_size=40, n_jobs=1)
    nn.fit(reference_arr)
    total = 0.0
    count = 0
    for start in range(0, points_arr.shape[0], int(batch_size)):
        distances, _ = nn.kneighbors(points_arr[start:start + int(batch_size)])
        total += float(distances.sum())
        count += int(distances.size)
    return total / max(count, 1)


def plan_entropy(pi) -> float:
    pi = np.asarray(pi, dtype=float)
    p = pi[pi > 0]
    return -float(np.sum(p * np.log(p))) if p.size else 0.0


def effective_support(pi) -> float:
    entropy = plan_entropy(pi)
    return float(np.exp(entropy)) if np.asarray(pi)[np.asarray(pi) > 0].size else 0.0


def topk_nn_overlap(X_a, X_b, k: int = 15) -> float:
    from .representations import nearest_neighbor_overlap

    return nearest_neighbor_overlap(X_a, X_b, k=k)


def coupling_topk_overlap(pi_a, pi_b, k: int = 15) -> float:
    pi_a = np.asarray(pi_a, dtype=float)
    pi_b = np.asarray(pi_b, dtype=float)
    if pi_a.shape != pi_b.shape:
        raise ValueError("coupling shapes differ")
    k = max(1, min(int(k), pi_a.shape[1]))
    rows = []
    for a, b in zip(pi_a, pi_b):
        ta = set(np.argpartition(-a, kth=k - 1)[:k].tolist())
        tb = set(np.argpartition(-b, kth=k - 1)[:k].tolist())
        rows.append(len(ta & tb) / float(k))
    return float(np.mean(rows))


def evaluate_endpoint(pred, target, seed: int = 42) -> dict:
    return {
        "endpoint_mmd": mmd_rbf(pred, target),
        "sliced_w2": sliced_w2(pred, target, seed=seed),
    }


def fate_mass_error(pred_labels, target_labels) -> float:
    """L1 difference between generated/predicted and target fate proportions."""
    pred_labels = np.asarray(pred_labels).astype(str)
    target_labels = np.asarray(target_labels).astype(str)
    if pred_labels.ndim != 1 or target_labels.ndim != 1:
        raise ValueError("pred_labels and target_labels must be 1D arrays")
    if pred_labels.shape[0] == 0 or target_labels.shape[0] == 0:
        raise ValueError("pred_labels and target_labels must be nonempty")
    labels = sorted(set(pred_labels.tolist()) | set(target_labels.tolist()))
    pred_counts = {label: 0 for label in labels}
    target_counts = {label: 0 for label in labels}
    for label in pred_labels:
        pred_counts[label] += 1
    for label in target_labels:
        target_counts[label] += 1
    pred_total = float(pred_labels.shape[0])
    target_total = float(target_labels.shape[0])
    return float(
        sum(abs(pred_counts[label] / pred_total - target_counts[label] / target_total) for label in labels)
    )


def coupling_l1_distance(pi_a: np.ndarray, pi_b: np.ndarray) -> float:
    """L1 distance between two normalized couplings with the same shape."""
    pi_a = np.asarray(pi_a, dtype=float)
    pi_b = np.asarray(pi_b, dtype=float)
    if pi_a.shape != pi_b.shape:
        raise ValueError(f"couplings must have the same shape, got {pi_a.shape} and {pi_b.shape}")
    if pi_a.ndim != 2:
        raise ValueError("couplings must be 2D")
    if np.any(~np.isfinite(pi_a)) or np.any(~np.isfinite(pi_b)):
        raise ValueError("couplings must contain only finite values")
    pi_a = np.clip(pi_a, 0.0, None)
    pi_b = np.clip(pi_b, 0.0, None)
    pi_a = pi_a / max(float(pi_a.sum()), 1e-15)
    pi_b = pi_b / max(float(pi_b.sum()), 1e-15)
    return float(np.abs(pi_a - pi_b).sum())


def normalized_cost_matrix(X0: np.ndarray, X1: np.ndarray) -> tuple[np.ndarray, float]:
    """Pairwise squared distances divided by the median positive cost."""
    C = pairwise_squared_distances(X0, X1)
    positive = C[C > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-12)
    return (C / scale).astype(np.float32), scale


def readout_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error in a common readout/program space."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} and {target.shape}")
    return float(np.mean((pred - target) ** 2))


def distribution_readout_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Distribution-level diagnostics for two readout clouds.

    These metrics intentionally do not assume row-wise pairing. They compare
    the predicted and target readout distributions through MMD, sliced W2, and
    summary-vector differences.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError("pred and target must be 2D arrays")
    if pred.shape[1] != target.shape[1]:
        raise ValueError("pred and target must have the same feature dimension")
    if pred.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("pred and target must be nonempty")
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    diff = pred_mean - target_mean
    return {
        "program_readout_mmd": mmd_rbf(pred, target, gamma=_median_gamma(pred, target)),
        "program_readout_sliced_w2": sliced_wasserstein_distance(pred, target),
        "program_readout_mean_abs_error": float(np.mean(np.abs(diff))),
        "program_readout_centroid_l2": float(np.linalg.norm(diff)),
        "program_readout_mean_mse": float(np.mean(diff**2)),
    }


def abundance_weight_summary(weights: np.ndarray, labels: np.ndarray | None = None) -> pd.DataFrame:
    """Summarize illustrative abundance weights by label for Chapter 6 boundary panels."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size == 0:
        raise ValueError("weights must be nonempty")
    if np.any(~np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")
    if labels is None:
        labels = np.full(weights.shape[0], "all", dtype=object)
    labels = np.asarray(labels).astype(str)
    if labels.shape[0] != weights.shape[0]:
        raise ValueError("labels length must match weights")
    rows = []
    total = float(weights.sum())
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        mass = float(weights[mask].sum())
        rows.append(
            {
                "label": label,
                "n_cells": int(mask.sum()),
                "abundance_mass": mass,
                "abundance_fraction": mass / max(total, 1e-15),
            }
        )
    return pd.DataFrame(rows)
