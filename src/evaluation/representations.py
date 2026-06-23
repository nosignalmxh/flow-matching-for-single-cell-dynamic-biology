from __future__ import annotations

import numpy as np


CH06_PROGRAM_ORDER = ["progenitor_trunk", "transition", "major_fate", "rare_fate"]


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def gene_program_scores(X, programs: dict[str, list[int]]):
    X = np.asarray(X, dtype=float)
    scores = []
    names = []
    for name, idx in programs.items():
        scores.append(X[:, idx].mean(axis=1))
        names.append(name)
    return np.stack(scores, axis=1), names


def log_normalized_matrix(adata, layer: str = "log_normalized") -> np.ndarray:
    """Return a dense log-normalized expression matrix from an AnnData object."""
    if layer in adata.layers:
        return _to_dense(adata.layers[layer])
    if layer == "X":
        return _to_dense(adata.X)
    raise KeyError(f"{layer!r} was not found in adata.layers")


def program_index_dict(
    adata,
    program_key: str = "program",
    include_background: bool = False,
) -> dict[str, list[int]]:
    """Build a deterministic program-to-gene-index mapping from ``adata.var``."""
    if program_key not in adata.var:
        raise KeyError(f"{program_key!r} was not found in adata.var")
    labels = adata.var[program_key].astype(str).to_numpy()
    ordered = [name for name in CH06_PROGRAM_ORDER if np.any(labels == name)]
    extras = sorted(
        name
        for name in np.unique(labels).tolist()
        if name not in set(ordered) and (include_background or name != "background")
    )
    names = ordered + extras
    if include_background and "background" in set(labels) and "background" not in names:
        names.append("background")
    return {name: np.flatnonzero(labels == name).astype(int).tolist() for name in names}


def gene_program_scores_from_adata(
    adata,
    layer: str = "log_normalized",
    program_key: str = "program",
    include_background: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Compute mean log expression per gene program."""
    X = log_normalized_matrix(adata, layer=layer)
    programs = program_index_dict(adata, program_key=program_key, include_background=include_background)
    scores, names = readout_program_scores_from_matrix(X, programs)
    return scores.astype(np.float32), names


def standardize_train_space(X0: np.ndarray, X1: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit mean/std on source+target and return standardized arrays plus metadata."""
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    if X0.ndim != 2 or X1.ndim != 2:
        raise ValueError("X0 and X1 must be 2D arrays")
    if X0.shape[1] != X1.shape[1]:
        raise ValueError("X0 and X1 must have the same feature dimension")
    combined = np.vstack([X0, X1])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    meta = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return ((X0 - mean) / std).astype(np.float32), ((X1 - mean) / std).astype(np.float32), meta


def fit_pca_state_space(
    X: np.ndarray,
    n_components: int = 30,
    seed: int = 42,
) -> dict:
    """Fit a small PCA state space and return transform metadata."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_components = max(1, min(int(n_components), X.shape[0], X.shape[1]))
    mean = X.mean(axis=0)
    centered = X - mean
    try:
        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components, random_state=seed)
        coords = model.fit_transform(X)
        components = model.components_
        ratio = model.explained_variance_ratio_
    except Exception:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :n_components] * s[:n_components]
        components = vt[:n_components]
        denom = float(np.sum(s**2))
        ratio = (s[:n_components] ** 2) / denom if denom > 0 else np.zeros(n_components)
    return {
        "coords": np.asarray(coords, dtype=np.float32),
        "mean": np.asarray(mean, dtype=np.float32),
        "components": np.asarray(components, dtype=np.float32),
        "explained_variance_ratio": np.asarray(ratio, dtype=float),
        "n_components": int(n_components),
        "n_features": int(X.shape[1]),
    }


def pca_inverse_transform(coords: np.ndarray, pca_state: dict) -> np.ndarray:
    """Map PCA coordinates back to the original feature space."""
    coords = np.asarray(coords, dtype=float)
    components = np.asarray(pca_state["components"], dtype=float)
    mean = np.asarray(pca_state["mean"], dtype=float)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array")
    if coords.shape[1] != components.shape[0]:
        raise ValueError("coords feature dimension must match PCA components")
    return (coords @ components + mean[None, :]).astype(np.float32)


def readout_program_scores_from_matrix(
    X_expr: np.ndarray,
    programs: dict[str, list[int]],
) -> tuple[np.ndarray, list[str]]:
    """Compute program readout from an observed or reconstructed expression matrix."""
    X_expr = np.asarray(X_expr, dtype=float)
    if X_expr.ndim != 2:
        raise ValueError("X_expr must be a 2D array")
    names = list(programs.keys())
    scores = []
    for name in names:
        idx = np.asarray(programs[name], dtype=int)
        if idx.size == 0:
            raise ValueError(f"program {name!r} has no genes")
        if idx.min() < 0 or idx.max() >= X_expr.shape[1]:
            raise ValueError(f"program {name!r} has gene indices outside X_expr")
        scores.append(X_expr[:, idx].mean(axis=1))
    return np.stack(scores, axis=1).astype(np.float32), names


def nearest_neighbor_overlap(X_a: np.ndarray, X_b: np.ndarray, k: int = 15) -> float:
    """Mean top-k nearest-neighbor overlap between two representations of the same cells."""
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    if X_a.ndim != 2 or X_b.ndim != 2:
        raise ValueError("X_a and X_b must be 2D arrays")
    if X_a.shape[0] != X_b.shape[0]:
        raise ValueError("X_a and X_b must describe the same number of cells")
    n = X_a.shape[0]
    if n <= 1:
        return 1.0
    k = max(1, min(int(k), n - 1))

    def _neighbors(X):
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
        np.fill_diagonal(d2, np.inf)
        return np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    nn_a = _neighbors(X_a)
    nn_b = _neighbors(X_b)
    overlaps = []
    for row_a, row_b in zip(nn_a, nn_b):
        overlaps.append(len(set(row_a.tolist()) & set(row_b.tolist())) / float(k))
    return float(np.mean(overlaps))
