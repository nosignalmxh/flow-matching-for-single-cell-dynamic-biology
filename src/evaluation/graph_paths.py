from __future__ import annotations

import numpy as np


def polyline_length(points) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def point_on_polyline(points, t):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError("polyline must contain at least one point")
    if len(points) == 1:
        return points[0]
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        return points[min(int(round(float(t) * (len(points) - 1))), len(points) - 1)]
    arclen = np.r_[0.0, np.cumsum(seg) / total]
    return np.array([np.interp(float(t), arclen, points[:, d]) for d in range(points.shape[1])], dtype=np.float32)


def resample_polyline(points, n_points: int = 41) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, int(n_points))
    return np.stack([point_on_polyline(points, t) for t in grid], axis=0)


def extract_path_indices(pred_row, source_node, target_node, dist_value, n_nodes) -> tuple[list[int], bool]:
    source_node = int(source_node)
    target_node = int(target_node)
    if source_node == target_node:
        return [source_node], False
    if not np.isfinite(dist_value):
        return [source_node, target_node], True
    path = [target_node]
    cur = target_node
    for _ in range(int(n_nodes) + 5):
        cur = int(pred_row[cur])
        if cur < 0:
            return [source_node, target_node], True
        path.append(cur)
        if cur == source_node:
            return path[::-1], False
    return [source_node, target_node], True


def knn_density_scorer(reference_points, k: int = 15):
    from sklearn.neighbors import NearestNeighbors

    reference_points = np.asarray(reference_points, dtype=np.float32)
    kk = max(1, min(int(k), len(reference_points)))
    nn = NearestNeighbors(n_neighbors=kk).fit(reference_points)
    ref_radius = nn.kneighbors(reference_points, return_distance=True)[0][:, -1]
    ref_sorted = np.sort(ref_radius)

    def score(points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points[None]
        radius = nn.kneighbors(points, return_distance=True)[0][:, -1]
        percentile = np.searchsorted(ref_sorted, radius, side="right") / len(ref_sorted) * 100.0
        return percentile.astype(float), radius.astype(float)

    return score, ref_radius


def _knn_distance_graph(points, k: int):
    from sklearn.neighbors import kneighbors_graph

    kk = max(1, min(int(k), len(points) - 1))
    graph = kneighbors_graph(points, n_neighbors=kk, mode="distance", include_self=False)
    return graph.maximum(graph.T).tocsr(), kk


def build_connected_knn_graph(points, k_grid) -> tuple:
    """Choose the first connected sparse kNN graph, preferring low density."""
    from scipy.sparse.csgraph import connected_components

    points = np.asarray(points, dtype=np.float32)
    graph_rows = []
    best = None
    for k in k_grid:
        graph, kk = _knn_distance_graph(points, int(k))
        n_components, component_labels = connected_components(graph, directed=False)
        density = float(graph.nnz / max(graph.shape[0] * graph.shape[1], 1))
        row = {
            "k_graph": kk,
            "n_components": int(n_components),
            "n_edges_undirected": int(graph.nnz // 2),
            "graph_density": density,
            "largest_component_fraction": float(np.bincount(component_labels).max() / len(points)),
        }
        graph_rows.append(row)
        if n_components == 1 and best is None:
            best = (graph, row, graph_rows)
            if density <= 0.05:
                break
    if best is None:
        best_idx = int(np.argmin([r["n_components"] for r in graph_rows]))
        graph, _ = _knn_distance_graph(points, graph_rows[best_idx]["k_graph"])
        best = (graph, graph_rows[best_idx], graph_rows)
    return best


def build_endpoint_knn_graph_grid(
    reference_points,
    source_nodes,
    target_nodes,
    k_grid,
    *,
    min_connected_fraction: float = 0.80,
) -> tuple:
    """Build kNN graphs and select the sparsest graph connecting enough endpoints."""
    from scipy.sparse.csgraph import connected_components

    reference_points = np.asarray(reference_points, dtype=np.float32)
    source_nodes = np.asarray(source_nodes, dtype=int)
    target_nodes = np.asarray(target_nodes, dtype=int)
    graph_rows = []
    graphs = {}
    labels_by_k = {}
    for k in k_grid:
        graph, kk = _knn_distance_graph(reference_points, int(k))
        n_components, component_labels = connected_components(graph, directed=False)
        connected = component_labels[source_nodes] == component_labels[target_nodes]
        density = float(graph.nnz / max(graph.shape[0] * graph.shape[1], 1))
        row = {
            "k_graph": kk,
            "n_components": int(n_components),
            "largest_component_fraction": float(np.bincount(component_labels).max() / len(reference_points)),
            "graph_density": density,
            "n_edges_undirected": int(graph.nnz // 2),
            "endpoint_connected_fraction": float(connected.mean()) if len(connected) else 0.0,
        }
        graph_rows.append(row)
        graphs[kk] = graph
        labels_by_k[kk] = component_labels
    connected_candidates = [r for r in graph_rows if r["endpoint_connected_fraction"] >= float(min_connected_fraction)]
    if connected_candidates:
        selected_info = sorted(connected_candidates, key=lambda r: (r["k_graph"], r["graph_density"]))[0]
    else:
        selected_info = sorted(graph_rows, key=lambda r: (-r["endpoint_connected_fraction"], r["k_graph"]))[0]
    selected_k = selected_info["k_graph"]
    return graphs[selected_k], selected_info, graph_rows, labels_by_k[selected_k]
