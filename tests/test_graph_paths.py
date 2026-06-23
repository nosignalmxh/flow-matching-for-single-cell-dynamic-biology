from __future__ import annotations

import numpy as np
import pytest


def test_polyline_helpers_resample_by_arclength_and_handle_degenerate_paths():
    from src.evaluation.graph_paths import point_on_polyline, polyline_length, resample_polyline

    path = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    assert polyline_length(path) == pytest.approx(4.0)
    np.testing.assert_allclose(point_on_polyline(path, 0.25), np.asarray([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(point_on_polyline(path, 0.75), np.asarray([2.0, 1.0], dtype=np.float32))

    sampled = resample_polyline(path, n_points=5)
    assert sampled.shape == (5, 2)
    np.testing.assert_allclose(sampled[0], path[0])
    np.testing.assert_allclose(sampled[-1], path[-1])

    repeated = np.asarray([[3.0, 1.0], [3.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(point_on_polyline(repeated, 0.9), repeated[-1])


def test_extract_path_indices_reports_fallbacks():
    from src.evaluation.graph_paths import extract_path_indices

    pred_row = np.asarray([-9999, 0, 1, 2], dtype=int)
    assert extract_path_indices(pred_row, 0, 3, dist_value=1.0, n_nodes=4) == ([0, 1, 2, 3], False)
    assert extract_path_indices(pred_row, 0, 3, dist_value=np.inf, n_nodes=4) == ([0, 3], True)
    assert extract_path_indices(np.asarray([-9999, -9999]), 0, 1, dist_value=1.0, n_nodes=2) == ([0, 1], True)


def test_knn_density_scorer_and_graph_builders():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")

    from src.evaluation.graph_paths import build_connected_knn_graph, build_endpoint_knn_graph_grid, knn_density_scorer

    points = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    score_fn, ref_radius = knn_density_scorer(points, k=2)
    pct, radius = score_fn(np.asarray([[0.5], [10.0]], dtype=np.float32))
    assert ref_radius.shape == (5,)
    assert pct.shape == radius.shape == (2,)
    assert pct[1] > pct[0]

    graph, selected_info, graph_rows = build_connected_knn_graph(points, k_grid=[1, 2])
    assert graph.shape == (5, 5)
    assert selected_info["n_components"] == 1
    assert graph_rows[0]["k_graph"] == 1

    source_nodes = np.asarray([0, 1], dtype=int)
    target_nodes = np.asarray([3, 4], dtype=int)
    graph2, info2, rows2, labels = build_endpoint_knn_graph_grid(points, source_nodes, target_nodes, k_grid=[1, 2])
    assert graph2.shape == (5, 5)
    assert info2["endpoint_connected_fraction"] >= 0.8
    assert len(rows2) == 2
    assert labels.shape == (5,)
