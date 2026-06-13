from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_distribution_transport_before_fm.ipynb"


def test_median_positive_scale_ignores_zeros_and_falls_back_to_one():
    from src.ot import median_positive_scale

    C = np.asarray([[0.0, 2.0, 8.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    assert median_positive_scale(C) == 4.0
    assert median_positive_scale(np.zeros((2, 2), dtype=np.float32)) == 1.0


def test_transport_reporting_helpers_cover_reusable_transport_math():
    from src.transport_reporting import (
        action_per_pair_pc,
        brownian_bridge_trajectories,
        coupling_diagnostic_row,
        energy_and_length_pc,
        path_stats,
        sorted_time_labels,
        subsample_indices,
    )

    assert sorted_time_labels(np.asarray(["10", "2", "a", "1"])) == ["1", "2", "10", "a"]

    rng = np.random.default_rng(7)
    sampled = subsample_indices(np.arange(20), max_n=5, rng=rng)
    assert sampled.shape == (5,)
    assert np.all(sampled[:-1] <= sampled[1:])

    pi = np.asarray([[0.25, 0.25], [0.25, 0.25]], dtype=float)
    C_raw = np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=float)
    row = coupling_diagnostic_row("independent", None, pi, C_raw, C_raw, None, cost_scale=2.0)
    assert row["method"] == "independent"
    assert np.isnan(row["epsilon"])
    assert row["expected_cost_raw"] == 1.0
    assert row["sinkhorn_backend"] == "independent"

    x0 = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    x1 = np.asarray([[1.0, 0.0], [2.0, 0.0]])
    tau_grid = np.linspace(0.0, 1.0, 5)
    straight = np.stack([(1 - tau) * x0 + tau * x1 for tau in tau_grid], axis=0)
    stats = path_stats(straight, tau_grid, straight_midpoint=0.5 * (x0 + x1))
    assert stats["mean_endpoint_distance"] == 1.0
    assert stats["midpoint_deviation"] == 0.0

    bridge = brownian_bridge_trajectories(x0, x1, tau_grid, sigma=0.1, seed=3)
    np.testing.assert_allclose(bridge[0], x0)
    np.testing.assert_allclose(bridge[-1], x1)

    energy, length = energy_and_length_pc(straight, tau_grid)
    per_pair_action = action_per_pair_pc(straight, tau_grid)
    assert energy == per_pair_action.mean()
    assert length == 1.0


def test_ch02_notebook_imports_reusable_helpers_from_src():
    payload = json.loads(NOTEBOOK_PATH.read_text())
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "from src.transport_reporting import (" in code_text
    for local_definition in [
        "def sorted_time_labels",
        "def subsample_indices",
        "def coupling_diagnostic_row",
        "def draw_endpoint_cloud",
        "def draw_arrows",
        "def path_stats",
        "def brownian_bridge_trajectories",
        "def energy_and_length_pc",
        "def action_per_pair_pc",
        "def sample_gaussian_mixture_torch",
        "def torch_mmd_rbf",
        "def euler_integrate_torch",
    ]:
        assert local_definition not in code_text
