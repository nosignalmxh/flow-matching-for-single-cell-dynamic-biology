from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_load_eb_data_standardizes_pc20_and_writes_summary(tmp_path: Path):
    from src.manifold_reporting import load_eb_data

    pcs = np.arange(30 * 25, dtype=np.float32).reshape(30, 25)
    phate = np.column_stack([np.linspace(0.0, 1.0, 30), np.linspace(2.0, 3.0, 30)]).astype(np.float32)
    labels = np.asarray(["0"] * 10 + ["1"] * 12 + ["2"] * 8)
    eb_path = tmp_path / "eb.npz"
    np.savez(eb_path, pcs=pcs, phate=phate, sample_labels=labels)

    loaded = load_eb_data(
        eb_path,
        source_time="1",
        target_time="2",
        out_dir=tmp_path,
        max_cells_per_time=5,
        seed=7,
    )

    assert loaded["X0_pc"].shape == (5, 20)
    assert loaded["X1_pc"].shape == (5, 20)
    assert loaded["X0_phate"].shape == (5, 2)
    assert loaded["summary"]["training_space"] == "standardized PC-20 from pcs[:, :20]"
    assert loaded["summary"]["n_source_full"] == 12
    assert loaded["summary"]["n_source_used"] == 5
    assert (tmp_path / "eb_data_summary.json").exists()
    np.testing.assert_allclose(loaded["pcs20_all"].mean(axis=0), np.zeros(20), atol=2e-6)


def test_select_representatives_by_quantile_is_deterministic_and_unique():
    from src.manifold_reporting import select_representatives_by_quantile

    selected = select_representatives_by_quantile(
        np.asarray([0.0, 2.0, 4.0, 8.0, 16.0]),
        quantiles=(0.0, 0.5, 0.5, 1.0),
    )

    assert selected["row_index"].tolist() == [0, 2, 1, 4]
    assert selected["quantile"].tolist() == [0.0, 0.5, 0.5, 1.0]


def test_fit_pc_to_phate_mapper_uses_distance_weighted_neighbors():
    from src.manifold_reporting import fit_pc_to_phate_mapper

    pcs = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    phate = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    mapper = fit_pc_to_phate_mapper(pcs, phate, n_neighbors=2)

    mapped = mapper(np.asarray([[0.25]], dtype=np.float32))

    assert mapped.shape == (1, 2)
    assert 0.0 < mapped[0, 0] < 10.0
    assert mapped[0, 1] == pytest.approx(0.0)


def test_fate_conditioned_plan_is_row_balanced_and_label_aware():
    from src.manifold_reporting import fate_conditioned_plan

    X0 = np.asarray([[0.0], [10.0]], dtype=np.float32)
    X1 = np.asarray([[0.1], [9.9], [20.0]], dtype=np.float32)
    plan = fate_conditioned_plan(X0, X1, ["a", "b"], ["a", "b", "a"])

    assert plan.shape == (2, 3)
    np.testing.assert_allclose(plan.sum(), 1.0)
    np.testing.assert_allclose(plan.sum(axis=1), [0.5, 0.5])
    assert plan[0, 1] == pytest.approx(0.0)
    assert plan[1, 0] == pytest.approx(0.0)
    assert plan[1, 2] == pytest.approx(0.0)


def test_load_toy_snapshots_casts_time_to_float(tmp_path: Path):
    from src.manifold_reporting import load_toy_snapshots

    path = tmp_path / "toy.csv"
    path.write_text("time,state_1,state_2,fate_label\n0.5,1,2,a\n1.0,3,4,b\n", encoding="utf-8")

    frame = load_toy_snapshots(path)

    assert frame["time"].dtype.kind == "f"
    assert frame["time"].tolist() == [0.5, 1.0]


def test_midpoint_direction_dispersion_reports_chord_statistics():
    from src.manifold_reporting import midpoint_direction_dispersion

    X0 = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    X1 = np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pi = np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=float)

    stats = midpoint_direction_dispersion(X0, X1, pi, n_pairs=6, k=2, seed=3)

    assert list(stats.columns) == [
        "idx0",
        "idx1",
        "pc20_chord_length",
        "midpoint_direction_angular_std_deg",
    ]
    assert len(stats) == 6
    np.testing.assert_allclose(stats["pc20_chord_length"], np.ones(6))


def test_build_artifact_manifest_records_run_config_and_missing_files(tmp_path: Path):
    from src.manifold_reporting import build_artifact_manifest

    fig_dir = tmp_path / "figures"
    out_dir = tmp_path / "outputs"
    fig_dir.mkdir()
    out_dir.mkdir()
    (fig_dir / "present.png").write_bytes(b"figure")
    (out_dir / "table.csv").write_text("a\n1\n", encoding="utf-8")

    manifest = build_artifact_manifest(
        project_root=tmp_path,
        fig_dir=fig_dir,
        out_dir=out_dir,
        run_config={"DEVICE": "cpu"},
        expected_figures=["present.png", "missing.png"],
        expected_tables=["table.csv"],
        dependency_files=[out_dir / "table.csv"],
    )

    assert manifest.loc[manifest["artifact"] == "RUN_CONFIG:DEVICE=cpu", "kind"].item() == "run_config"
    assert manifest.loc[manifest["artifact"] == "figures/present.png", "exists"].item() is True
    assert manifest.loc[manifest["artifact"] == "figures/missing.png", "exists"].item() is False
    assert manifest.loc[manifest["artifact"] == "outputs/table.csv", "kind"].tolist() == [
        "table_or_json",
        "dependency",
    ]


def test_train_or_load_model_rejects_checkpoint_metadata_mismatch(tmp_path: Path):
    import torch

    from src.manifold_reporting import train_or_load_model

    X0 = np.zeros((2, 3), dtype=np.float32)
    X1 = np.ones((2, 3), dtype=np.float32)
    pi = np.eye(2, dtype=float) / 2.0
    ckpt = tmp_path / "demo_d3_steps5_batch2_seed11_model.pt"
    torch.save({"state_dict": {}, "input_dim": 99, "steps": 5}, ckpt)

    with pytest.raises(ValueError, match="Checkpoint metadata mismatch"):
        train_or_load_model(
            "demo",
            X0,
            X1,
            pi,
            cache_dir=tmp_path,
            steps=5,
            batch_size=2,
            seed=11,
            device=torch.device("cpu"),
        )


def test_train_or_load_model_rejects_architecture_metadata_mismatch(tmp_path: Path):
    import torch

    from src.manifold_reporting import train_or_load_model

    X0 = np.zeros((2, 3), dtype=np.float32)
    X1 = np.ones((2, 3), dtype=np.float32)
    pi = np.eye(2, dtype=float) / 2.0
    ckpt = tmp_path / "demo_d3_steps5_batch2_seed11_model.pt"
    torch.save(
        {
            "state_dict": {},
            "input_dim": 3,
            "steps": 5,
            "batch_size": 2,
            "seed": 11,
            "hidden": 64,
            "layers": 4,
        },
        ckpt,
    )

    with pytest.raises(ValueError, match="hidden: found 64, expected 128"):
        train_or_load_model(
            "demo",
            X0,
            X1,
            pi,
            cache_dir=tmp_path,
            steps=5,
            batch_size=2,
            seed=11,
            hidden=128,
            layers=4,
            device=torch.device("cpu"),
        )


def test_training_helper_import_can_replace_notebook_local_definition():
    from src.manifold_reporting import train_or_load_model

    assert callable(train_or_load_model)


def test_plot_metric_bar_grid_smoke(tmp_path: Path):
    from src.manifold_reporting import plot_metric_bar_grid

    table = pd.DataFrame(
        {
            "method": ["random_cfm", "ot_cfm"],
            "straightness_action_S": [1.2, 0.8],
            "path_energy_action": [2.5, 1.7],
        }
    )

    path = plot_metric_bar_grid(
        table,
        methods=["random_cfm", "ot_cfm"],
        metric_specs=[
            ("straightness_action_S", "Straightness action S", "lower"),
            ("path_energy_action", "Path energy/action", "PC-20"),
        ],
        fig_dir=tmp_path,
        filename="bars.png",
    )

    assert path.exists()
    assert path.stat().st_size > 0
