from __future__ import annotations

import numpy as np
import pytest


def test_ch04_metric_aliases_match_notebook_behaviour():
    from src.evaluation.metrics import (
        effective_support,
        evaluate_endpoint,
        path_energy,
        path_length,
        plan_entropy,
        sliced_w2,
        straightness,
        straightness_action_S,
        tortuosity_straightness,
    )

    X = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    Y = np.asarray([[0.1, 0.0], [1.0, 0.2], [0.0, 0.7]], dtype=float)
    assert sliced_w2(X, Y, seed=11) == pytest.approx(
        evaluate_endpoint(X, Y, seed=11)["sliced_w2"]
    )

    traj = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(path_length(traj), 2.0)
    np.testing.assert_allclose(path_energy(traj, times=np.asarray([0.0, 0.5, 1.0])), 4.0)
    np.testing.assert_allclose(tortuosity_straightness(traj), 0.0)
    np.testing.assert_allclose(straightness(traj), tortuosity_straightness(traj))
    np.testing.assert_allclose(straightness_action_S(traj), 0.0)

    pi = np.asarray([[0.5, 0.0], [0.25, 0.25]], dtype=float)
    entropy = -(0.5 * np.log(0.5) + 0.25 * np.log(0.25) + 0.25 * np.log(0.25))
    assert plan_entropy(pi) == pytest.approx(entropy)
    assert effective_support(pi) == pytest.approx(np.exp(entropy))


def test_ch04_coupling_helpers_match_notebook_behaviour():
    from src.evaluation.metrics import coupling_topk_overlap, topk_nn_overlap
    from src.core.ot import compute_cost_matrix, sample_from_plan, sample_independent_pairs, sinkhorn_plan

    X0 = np.asarray([[0.0], [2.0]], dtype=np.float32)
    X1 = np.asarray([[1.0], [3.0]], dtype=np.float32)
    C_norm, scale = compute_cost_matrix(X0, X1, normalize=True)
    np.testing.assert_allclose(scale, 1.0)
    np.testing.assert_allclose(C_norm, np.asarray([[1.0, 9.0], [1.0, 1.0]], dtype=np.float32))

    C_raw, raw_scale = compute_cost_matrix(X0, X1, normalize=False)
    np.testing.assert_allclose(raw_scale, 1.0)
    np.testing.assert_allclose(C_raw, np.asarray([[1.0, 9.0], [1.0, 1.0]], dtype=np.float32))

    pi = sinkhorn_plan(C_norm, epsilon=0.5)
    assert pi.shape == C_norm.shape
    np.testing.assert_allclose(pi.sum(), 1.0)

    i0, i1 = sample_independent_pairs(X0, X1, n_pairs=5, seed=7)
    np.testing.assert_array_equal(i0, np.asarray([1, 1, 1, 1, 1]))
    np.testing.assert_array_equal(i1, np.asarray([1, 1, 0, 0, 0]))

    pi_sample = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    s0, s1 = sample_from_plan(pi_sample, n_pairs=4, seed=3)
    np.testing.assert_array_equal(s0, np.zeros(4, dtype=int))
    np.testing.assert_array_equal(s1, np.ones(4, dtype=int))

    pi_a = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    pi_b = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    assert coupling_topk_overlap(pi_a, pi_b, k=1) == pytest.approx(1.0)
    assert topk_nn_overlap(np.eye(3), np.eye(3), k=1) == pytest.approx(1.0)
