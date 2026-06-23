from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def pairwise_squared_distances(X0, X1):
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    return ((X0[:, None, :] - X1[None, :, :]) ** 2).sum(axis=-1)


def median_positive_scale(C: np.ndarray) -> float:
    C = np.asarray(C, dtype=float)
    positive = C[C > 0]
    return float(np.median(positive)) if positive.size else 1.0


def _as_balanced_coupling(pi: np.ndarray, n_source: int, n_target: int) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (n_source, n_target):
        raise ValueError(f"coupling shape must be {(n_source, n_target)}, got {pi.shape}")
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    pi = np.clip(pi, 0.0, None)
    total = float(pi.sum())
    if total <= 0:
        return independent_coupling(n_source, n_target)
    return pi / total


def _sinkhorn_scaling_from_cost(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-9,
    return_info: bool = False,
):
    C = np.asarray(C, dtype=float)
    scaled = C - float(np.nanmin(C))
    K = np.exp(-scaled / max(float(epsilon), 1e-8))
    K = np.clip(K, 1e-300, None)
    pi = K
    converged = False
    n_iter = 0
    for n_iter in range(1, int(num_iter_max) + 1):
        row = pi.sum(axis=1, keepdims=True)
        pi *= a[:, None] / np.clip(row, 1e-300, None)
        col = pi.sum(axis=0, keepdims=True)
        pi *= b[None, :] / np.clip(col, 1e-300, None)
        if n_iter % 10 == 0 or n_iter == int(num_iter_max):
            if _marginal_l1_error(pi, a, b) <= float(stop_thr):
                converged = True
                break
    if return_info:
        return pi, {"n_iter": int(n_iter), "converged": bool(converged)}
    return pi


def _marginal_l1_error(pi: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(pi.sum(axis=1) - a).sum() + np.abs(pi.sum(axis=0) - b).sum())


def independent_coupling(n_source: int, n_target: int) -> np.ndarray:
    """Uniform product coupling with row/column marginals."""
    if n_source <= 0 or n_target <= 0:
        raise ValueError("n_source and n_target must be positive")
    return np.full((int(n_source), int(n_target)), 1.0 / (int(n_source) * int(n_target)), dtype=float)


def random_pair_indices(n_source: int, n_target: int, batch_size: int, seed: int = 42):
    """Independent random row/column pairing for visual contrast."""
    if n_source <= 0 or n_target <= 0:
        raise ValueError("n_source and n_target must be positive")
    rng = np.random.default_rng(seed)
    i0 = rng.integers(0, int(n_source), size=int(batch_size))
    i1 = rng.integers(0, int(n_target), size=int(batch_size))
    return i0, i1


def sample_independent_pairs(X0, X1, n_pairs: int, seed: int = 42):
    return random_pair_indices(len(X0), len(X1), batch_size=int(n_pairs), seed=seed)


def compute_cost_matrix(x0, x1, normalize: bool = True):
    C = pairwise_squared_distances(np.asarray(x0, dtype=np.float32), np.asarray(x1, dtype=np.float32)).astype(np.float32)
    if not normalize:
        return C, 1.0
    scale = median_positive_scale(C)
    scale = max(scale, 1e-12)
    return (C / scale).astype(np.float32), scale


def compute_ot_coupling(X0, X1, epsilon: float = 0.05):
    """Return a small balanced coupling. Uses POT Sinkhorn when available."""
    C = pairwise_squared_distances(X0, X1)
    return compute_ot_coupling_from_cost(C, epsilon=epsilon, return_info=False)


def sinkhorn_plan(C, epsilon: float = 0.05, return_info: bool = False):
    return compute_ot_coupling_from_cost(np.asarray(C, dtype=np.float32), epsilon=float(epsilon), return_info=return_info)


def compute_ot_coupling_from_cost(
    C: np.ndarray,
    epsilon: float = 0.05,
    return_info: bool = False,
    num_iter_max: int = 5000,
    stop_thr: float = 1e-9,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Return a balanced Sinkhorn coupling from a precomputed cost matrix.

    When ``return_info`` is true, convergence diagnostics are returned with the
    plan so teaching code can report numerical caveats instead of suppressing
    them.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a 2D cost matrix")
    if C.shape[0] == 0 or C.shape[1] == 0:
        raise ValueError("C must have nonempty source and target dimensions")
    if np.any(~np.isfinite(C)):
        raise ValueError("C must contain only finite entries")
    if np.any(C < 0):
        raise ValueError("C must be nonnegative")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite value")

    n_source, n_target = C.shape
    a = np.full(n_source, 1.0 / n_source)
    b = np.full(n_target, 1.0 / n_target)
    warning_messages: list[str] = []
    backend = "pot"
    n_iter = 0
    fallback_used = False
    raw_valid = True

    try:
        import ot

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ot.sinkhorn(
                a,
                b,
                C,
                reg=epsilon,
                numItermax=int(num_iter_max),
                stopThr=float(stop_thr),
                log=True,
                warn=True,
            )
        warning_messages.extend(str(item.message) for item in caught)
        if isinstance(result, tuple) and len(result) == 2:
            pi, log = result
            n_iter = int(log.get("niter", len(log.get("err", [])) if isinstance(log, dict) else 0))
        else:
            pi = result
    except Exception as exc:
        warning_messages.append(f"POT sinkhorn failed: {exc}")
        backend = "scaling_fallback"
        fallback_used = True
        pi, fallback_info = _sinkhorn_scaling_from_cost(
            C,
            a,
            b,
            epsilon,
            num_iter_max=int(num_iter_max),
            stop_thr=float(stop_thr),
            return_info=True,
        )
        n_iter = int(fallback_info["n_iter"])

    pi = np.asarray(pi, dtype=float)
    if pi.shape != C.shape or np.any(~np.isfinite(pi)) or np.any(pi < 0) or float(np.nan_to_num(pi, nan=0.0).sum()) <= 0:
        raw_valid = False
        warning_messages.append("Sinkhorn returned an invalid plan; used iterative scaling fallback.")
        backend = "scaling_fallback"
        fallback_used = True
        pi, fallback_info = _sinkhorn_scaling_from_cost(
            C,
            a,
            b,
            epsilon,
            num_iter_max=int(num_iter_max),
            stop_thr=float(stop_thr),
            return_info=True,
        )
        n_iter = int(fallback_info["n_iter"])

    pi = _as_balanced_coupling(pi, n_source, n_target)
    marginal_error = _marginal_l1_error(pi, a, b)
    if marginal_error > 1e-4:
        warning_messages.append(
            f"Sinkhorn marginal error {marginal_error:.3g} exceeded tolerance; used iterative scaling fallback."
        )
        backend = "scaling_fallback"
        fallback_used = True
        pi, fallback_info = _sinkhorn_scaling_from_cost(
            C,
            a,
            b,
            epsilon,
            num_iter_max=int(num_iter_max),
            stop_thr=float(stop_thr),
            return_info=True,
        )
        n_iter = int(fallback_info["n_iter"])
        pi = _as_balanced_coupling(pi, n_source, n_target)

    row_l1_error = float(np.abs(pi.sum(axis=1) - a).sum())
    col_l1_error = float(np.abs(pi.sum(axis=0) - b).sum())
    warning_text = "; ".join(dict.fromkeys(message for message in warning_messages if message))
    warning_lower = warning_text.lower()
    warning_indicates_nonconvergence = any(
        token in warning_lower for token in ["not converge", "numerical errors", "invalid plan", "failed"]
    )
    sinkhorn_converged = bool(
        raw_valid
        and row_l1_error + col_l1_error <= 1e-4
        and not warning_indicates_nonconvergence
    )
    info = {
        "sinkhorn_converged": sinkhorn_converged,
        "row_l1_error": row_l1_error,
        "col_l1_error": col_l1_error,
        "n_iter": int(n_iter),
        "warning_message": warning_text,
        "backend": backend,
        "fallback_used": bool(fallback_used),
    }
    if return_info:
        return pi, info
    return pi


def sample_pairs_from_coupling(X0, X1, pi, batch_size: int = 256, seed: int = 42):
    i0, i1 = sample_pair_indices_from_coupling(pi, batch_size=batch_size, seed=seed)
    return X0[i0], X1[i1]


def sample_pair_indices_from_coupling(
    pi: np.ndarray,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample row/column indices from a coupling matrix."""
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 2:
        raise ValueError("pi must be a 2D coupling matrix")
    if np.any(~np.isfinite(pi)):
        raise ValueError("pi must contain only finite entries")
    if np.any(pi < 0):
        raise ValueError("pi must have nonnegative mass")
    total = float(pi.sum())
    if total <= 0:
        raise ValueError("pi must have positive total mass")

    rng = np.random.default_rng(seed)
    p = (pi / total).reshape(-1)
    flat = rng.choice(len(p), size=int(batch_size), replace=True, p=p)
    i0, i1 = np.unravel_index(flat, pi.shape)
    return np.asarray(i0, dtype=int), np.asarray(i1, dtype=int)


def sample_from_plan(pi, n_pairs: int, seed: int = 42):
    return sample_pair_indices_from_coupling(np.asarray(pi, dtype=float), batch_size=int(n_pairs), seed=seed)


def coupling_diagnostics(pi: np.ndarray, C: np.ndarray | None = None) -> dict:
    """Return mass, marginal errors, entropy, expected cost, and support size."""
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 2:
        raise ValueError("pi must be a 2D coupling matrix")
    pi = _as_balanced_coupling(pi, pi.shape[0], pi.shape[1])
    n_source, n_target = pi.shape
    row_target = np.full(n_source, 1.0 / n_source)
    col_target = np.full(n_target, 1.0 / n_target)
    positive = pi[pi > 0]
    entropy = -float(np.sum(positive * np.log(positive))) if positive.size else 0.0
    if C is None:
        expected_cost = np.nan
    else:
        C = np.asarray(C, dtype=float)
        if C.shape != pi.shape:
            raise ValueError(f"C shape must match pi shape {pi.shape}, got {C.shape}")
        expected_cost = float(np.sum(pi * C))
    return {
        "total_mass": float(pi.sum()),
        "row_l1_error": float(np.abs(pi.sum(axis=1) - row_target).sum()),
        "col_l1_error": float(np.abs(pi.sum(axis=0) - col_target).sum()),
        "entropy": entropy,
        "expected_cost": expected_cost,
        "effective_support": float(np.exp(entropy)),
    }


def barycentric_projection(
    X_target: np.ndarray,
    pi: np.ndarray,
    source_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return row-normalized barycentric targets for source cells."""
    X_target = np.asarray(X_target, dtype=float)
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 2:
        raise ValueError("pi must be a 2D coupling matrix")
    if X_target.shape[0] != pi.shape[1]:
        raise ValueError("X_target rows must match pi columns")
    if source_weights is None:
        row_mass = pi.sum(axis=1)
    else:
        row_mass = np.asarray(source_weights, dtype=float).reshape(-1)
        if row_mass.shape[0] != pi.shape[0]:
            raise ValueError("source_weights length must match pi rows")
    safe_mass = np.clip(row_mass, 1e-15, None)
    return (pi @ X_target) / safe_mass[:, None]


def fate_probabilities(
    pi: np.ndarray,
    target_labels: np.ndarray,
    source_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate row-normalized transport mass by target labels."""
    pi = np.asarray(pi, dtype=float)
    target_labels = np.asarray(target_labels)
    if pi.ndim != 2:
        raise ValueError("pi must be a 2D coupling matrix")
    if target_labels.shape[0] != pi.shape[1]:
        raise ValueError("target_labels length must match pi columns")
    if source_weights is None:
        row_mass = pi.sum(axis=1)
    else:
        row_mass = np.asarray(source_weights, dtype=float).reshape(-1)
        if row_mass.shape[0] != pi.shape[0]:
            raise ValueError("source_weights length must match pi rows")
    row_conditional = pi / np.clip(row_mass[:, None], 1e-15, None)
    rows = []
    for source_index in range(pi.shape[0]):
        row = {"source_index": source_index}
        for label in sorted(pd.Series(target_labels).astype(str).unique()):
            row[f"prob_{label}"] = float(row_conditional[source_index, target_labels.astype(str) == label].sum())
        rows.append(row)
    return pd.DataFrame(rows)
