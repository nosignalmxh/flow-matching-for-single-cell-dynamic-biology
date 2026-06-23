from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_MONOLITH_PATH = (
    PROJECT_ROOT
    / "archive"
    / "notebooks_retired_20260604_ch03_split"
    / "03_flow_matching_from_scratch.ipynb"
)
RETIRED_INDEX_PATH = PROJECT_ROOT / "archive" / "03_flow_matching_from_scratch.ipynb"
CH03_TOY_NOTEBOOK = "chapter3_1_flow_matching_from_scratch.ipynb"
CH03_EB_MAIN_RETIRED_NOTEBOOK = "03_2_eb20d_main_flow_matching.ipynb"
CH03_EB_MAIN_NOTEBOOK = "chapter3_2_eb_flow_matching.ipynb"
CH03_EB_ABLATION_RETIRED_NOTEBOOK = "03_3_eb20d_baselines_ablations_and_claim_audit.ipynb"
CH03_EB_ABLATION_NOTEBOOK = "chapter3_3_eb_ablations.ipynb"


SPLIT_SPECS = {
    CH03_TOY_NOTEBOOK: {
        "min_code_cells": 18,
        "required_headings": [
            "2D Toy Sanity Check",
            "Conditional Velocity Versus Marginal Velocity",
            "CFM Object Hierarchy",
        ],
        "required_artifacts": [
            "fig_toy_endpoint_distributions.png",
            "fig_toy_loss.png",
            "fig_toy_evolution.png",
            "fig03_02_conditional_vs_marginal_toy.png",
            "fig03_03_cfm_object_hierarchy_toy.png",
        ],
        "forbidden_headings": [
            "Load EB Data",
            "CNF-Endpoint Baseline",
            "Time Sampling Strategy Ablation",
        ],
    },
    CH03_EB_MAIN_NOTEBOOK: {
        "min_code_cells": 19,
        "required_headings": [
            "Data audit",
            "Train/validation split",
            "Endpoint pairing",
            "Model config",
            "Training loop",
            "Loss table",
            "Loss figure",
            "Model cache",
            "Endpoint pair visualization",
            "Sampling and population evolution",
            "Euler step sensitivity",
        ],
        "required_artifacts": [
            "ch03_eb_timepoint_counts.csv",
            "ch03_eb20d_train_val_split.csv",
            "ch03_eb20d_training_log.csv",
            "ch03_euler_step_sensitivity.csv",
            "figB1_eb20d_train_val_loss.png",
            "fig03_04_eb_endpoint_pairs_phate.png",
            "fig03_08_eb_population_evolution_phate.png",
            "fig03_09_euler_step_sensitivity_phate.png",
            "ch03_eb20d_velocity_mlp_seed42.pt",
            "ch03_eb20d_main_config_seed42.json",
        ],
        "forbidden_headings": [
            "2D Toy Sanity Check",
            "CNF-Endpoint Baseline",
            "Time Sampling Strategy Ablation",
            "Network Capacity Ablation",
        ],
    },
    CH03_EB_ABLATION_NOTEBOOK: {
        "min_code_cells": 24,
        "required_headings": [
            "Solver Comparison Lite",
            "ODE-in-the-loop Training Cost",
            "Time Sampling Strategy Ablation",
            "Network Capacity Ablation",
            "Trajectory Straightness in 20D",
        ],
        "required_artifacts": [
            "fig03_10_nfe_vs_endpoint_error.png",
            "figE1_fm_vs_cnf_compute_quality.png",
            "figE1_fm_vs_cnf_cumulative_time.png",
            "figE2_capacity_endpoint_mmd.png",
            "figE2_capacity_val_mse.png",
            "figE3_time_sampling_distributions.png",
            "figE3_time_sampling_endpoint_mmd.png",
            "figE3_time_sampling_final_bar.png",
            "figE3_time_sampling_val_mse.png",
            "figE5_endpoint_distance_vs_straightness.png",
            "figE5_representative_trajectories_phate.png",
            "figE5_straightness_hist.png",
            "table03_01_solver_diagnostics.csv",
            "tableE1_cfm_vs_cnf_endpoint.csv",
            "tableE3_time_sampling_ablation.csv",
            "tableT2_training_hyperparams_capacity.csv",
            "tableE5_trajectory_straightness.csv",
            "paper_table03_01_solver_diagnostics.csv",
            "paper_table03_01_solver_diagnostics.md",
            "paper_table03_01_solver_diagnostics.tex",
            "paper_tableE1_cfm_vs_cnf_endpoint.csv",
            "paper_tableE1_cfm_vs_cnf_endpoint.md",
            "paper_tableE1_cfm_vs_cnf_endpoint.tex",
            "paper_tableE3_time_sampling_ablation.csv",
            "paper_tableE3_time_sampling_ablation.md",
            "paper_tableE3_time_sampling_ablation.tex",
            "paper_tableT2_training_hyperparams_capacity.csv",
            "paper_tableT2_training_hyperparams_capacity.md",
            "paper_tableT2_training_hyperparams_capacity.tex",
            "paper_tableE5_trajectory_straightness_summary.csv",
            "paper_tableE5_trajectory_straightness_summary.md",
            "paper_tableE5_trajectory_straightness_summary.tex",
        ],
        "forbidden_headings": [
            "2D Toy Sanity Check",
            "Train EB 20D VelocityMLP",
            "Configuration table",
            "Metrics table",
            "Claim boundary",
        ],
    },
}

LEGACY_CH03_STATIC_FIGURES: set[str] = set()


def _payload(filename: str) -> dict:
    return json.loads((PROJECT_ROOT / "notebooks" / filename).read_text())


def _payload_from_path(path: Path) -> dict:
    return json.loads(path.read_text())


def _sources(filename: str, cell_type: str | None = None) -> list[str]:
    payload = _payload(filename)
    return [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell_type is None or cell.get("cell_type") == cell_type
    ]


def _artifact_referenced(text: str, artifact: str) -> bool:
    if artifact in text:
        return True
    stem = artifact.rsplit(".", 1)[0]
    return stem != artifact and stem in text


def test_ch03_canonical_replacements_exist_and_retired_active_copies_are_removed():
    replacements = {
        "03_1_toy_flow_matching_from_scratch.ipynb": CH03_TOY_NOTEBOOK,
        CH03_EB_MAIN_RETIRED_NOTEBOOK: CH03_EB_MAIN_NOTEBOOK,
        CH03_EB_ABLATION_RETIRED_NOTEBOOK: CH03_EB_ABLATION_NOTEBOOK,
    }

    for retired_name, replacement_name in replacements.items():
        assert not (PROJECT_ROOT / "notebooks" / retired_name).exists(), retired_name
        assert (PROJECT_ROOT / "notebooks" / replacement_name).exists(), replacement_name


def test_ch03_split_notebooks_exist_and_are_tutorial_sized():
    for filename, spec in SPLIT_SPECS.items():
        payload = _payload(filename)
        code_sources = _sources(filename, "code")
        markdown_text = "\n".join(_sources(filename, "markdown"))
        code_lengths = [len(source.splitlines()) for source in code_sources]

        assert payload["nbformat"] >= 4
        assert len(code_sources) >= spec["min_code_cells"], filename
        assert max(code_lengths) <= 90, (filename, max(code_lengths))

        for heading in spec["required_headings"]:
            assert heading in markdown_text, (filename, heading)
        for heading in spec["forbidden_headings"]:
            assert heading not in markdown_text, (filename, heading)
        for index, source in enumerate(code_sources, 1):
            if filename == CH03_TOY_NOTEBOOK:
                assert source.strip(), (filename, index)
                assert not all(
                    line.strip().startswith("#") or not line.strip()
                    for line in source.splitlines()
                ), (filename, index)
            compile(source, f"{filename}:code-cell-{index}", "exec")


def test_ch03_split_notebooks_cover_artifacts_without_overlap():
    artifact_owners: dict[str, str] = {}

    for filename, spec in SPLIT_SPECS.items():
        code_text = "\n".join(_sources(filename, "code"))
        markdown_text = "\n".join(_sources(filename, "markdown"))
        notebook_text = code_text + "\n" + markdown_text

        assert "from src.tutorial_init import" in code_text, filename
        assert "make_ch03_run_config()" in code_text, filename
        assert "make_save_and_show(" in code_text, filename
        if filename == CH03_TOY_NOTEBOOK:
            assert "raise FileNotFoundError" not in code_text, filename
        else:
            assert "make_save_and_show(" in code_text, filename
        if filename != CH03_TOY_NOTEBOOK:
            assert "display_table(" in code_text, filename

        display_markers = [
            "display(Image(",
            "display_saved_figure(",
            "display_saved_figures(",
            "display_png(",
            "make_save_and_show(",
        ]
        assert any(marker in code_text for marker in display_markers), filename

        for artifact in spec["required_artifacts"]:
            assert _artifact_referenced(notebook_text, artifact), (filename, artifact)
            previous_owner = artifact_owners.setdefault(artifact, filename)
            assert previous_owner == filename, (artifact, previous_owner, filename)

    expected_all = {artifact for spec in SPLIT_SPECS.values() for artifact in spec["required_artifacts"]}
    assert set(artifact_owners) == expected_all


def test_ch03_split_notebooks_cover_legacy_monolith_artifact_contract():
    archived_payload = _payload_from_path(ARCHIVED_MONOLITH_PATH)
    archived_text = "\n".join("".join(cell.get("source", [])) for cell in archived_payload["cells"])
    split_text = "\n".join(
        "\n".join(_sources(filename))
        for filename in SPLIT_SPECS
    )

    expected_legacy_artifacts = {
        artifact
        for spec in SPLIT_SPECS.values()
        for artifact in spec["required_artifacts"]
    }
    expected_legacy_artifacts.update(LEGACY_CH03_STATIC_FIGURES)

    for artifact in expected_legacy_artifacts:
        assert _artifact_referenced(archived_text, artifact) or _artifact_referenced(split_text, artifact), artifact
        assert _artifact_referenced(split_text, artifact), artifact


def test_ch03_required_outputs_remain_referenced_after_manifest_removal():
    for filename, spec in SPLIT_SPECS.items():
        notebook_text = "\n".join(_sources(filename))
        for artifact in spec["required_artifacts"]:
            assert _artifact_referenced(notebook_text, artifact), (filename, artifact)


def test_ch03_shared_tutorial_helpers_are_generic(tmp_path):
    from src.visualization import flow_matching as tutorial

    for name in [
        "json_ready",
        "save_json",
        "save_csv",
        "display_saved_figure",
        "display_saved_figures",
        "check_required_artifacts",
    ]:
        assert hasattr(tutorial, name), name

    json_path = tutorial.save_json(tmp_path / "nested" / "payload.json", {"x": 1})
    csv_path = tutorial.save_csv(tmp_path / "nested" / "table.csv", [{"a": 1}, {"a": 2}])

    assert json_path.exists()
    assert csv_path.exists()

    manifest = tutorial.check_required_artifacts(
        expected_figures=[json_path],
        expected_tables=[csv_path],
    )
    assert set(manifest["kind"]) == {"figure", "table"}
    assert manifest["exists"].all()


def test_ch03_toy_helpers_are_extracted_and_callable():
    torch = pytest.importorskip("torch")
    import matplotlib.pyplot as plt

    from src.visualization import flow_matching as tutorial

    X0, components = tutorial.make_eight_gaussians(n=96, seed=11)
    X1 = tutorial.make_single_gaussian(n=96, seed=12)
    pair_batch_fn = tutorial.make_random_pair_batch_fn(X0, X1, seed=13)
    batch = pair_batch_fn(10)

    assert X0.shape == (96, 2)
    assert X1.shape == (96, 2)
    assert X0.dtype == np.float32
    assert X1.dtype == np.float32
    assert components.shape == (96,)
    assert set(batch) == {"x0", "x1"}
    assert batch["x0"].shape == (10, 2)
    assert batch["x1"].shape == (10, 2)

    class ConstantVelocity(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 0.25)

    model = ConstantVelocity()
    probe = tutorial.build_toy_velocity_probe(model, X0, X1, device="cpu", seed=14)
    assert probe["center"].shape == (2,)
    assert probe["network_velocity"].shape == (2,)
    assert probe["mean_conditional_velocity"].shape == (2,)
    assert len(probe["local"]) > 0

    hierarchy = tutorial.prepare_toy_hierarchy_objects(model, X0, X1, device="cpu", seed=15)
    assert hierarchy["grid_points"].shape == (400, 2)
    assert hierarchy["grid_velocity"].shape == (400, 2)

    fig = tutorial.plot_toy_conditional_vs_marginal(model, X0, X1, probe)
    fig2 = tutorial.plot_toy_cfm_object_hierarchy(X0, X1, hierarchy)
    assert len(fig.axes) == 1
    assert len(fig2.axes) == 3
    legend = fig.axes[0].get_legend()
    assert legend is not None
    anchor_x0 = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).x0
    axis_x1 = fig.axes[0].get_position().x1
    assert anchor_x0 > axis_x1
    plt.close(fig)
    plt.close(fig2)


def test_ch03_toy_notebook_calls_extracted_helpers():
    code_text = "\n".join(_sources(CH03_TOY_NOTEBOOK, "code"))

    for local_definition in [
        "def save_figure(",
        "def save_csv(",
        "def display_saved_figure(",
        "def display_saved_figures(",
        "def display_table(",
        "def make_eight_gaussians(",
        "def make_single_gaussian(",
        "def make_random_pair_batch_fn(",
        "def build_toy_velocity_probe(",
        "def plot_toy_conditional_vs_marginal(",
        "def prepare_toy_hierarchy_objects(",
        "def plot_toy_cfm_object_hierarchy(",
    ]:
        assert local_definition not in code_text, local_definition

    for helper_call in [
        "make_save_and_show(",
        "ch03.make_eight_gaussians(",
        "ch03.make_single_gaussian(",
        "ch03.make_random_pair_batch_fn(",
        "ch03.build_toy_velocity_probe(",
        "ch03.plot_toy_conditional_vs_marginal(",
        "ch03.prepare_toy_hierarchy_objects(",
        "ch03.plot_toy_cfm_object_hierarchy(",
    ]:
        assert helper_call in code_text, helper_call


def test_ch03_toy_notebook_keeps_reader_facing_outputs_clean():
    payload = _payload(CH03_TOY_NOTEBOOK)
    code_sources = _sources(CH03_TOY_NOTEBOOK, "code")
    code_text = "\n".join(code_sources)
    markdown_text = "\n".join(_sources(CH03_TOY_NOTEBOOK, "markdown"))

    assert "def find_project_root(" not in code_text
    assert "bootstrap(" in code_text
    assert '{"setting": "project_root"' not in code_text
    assert "display_table(run_mode" not in code_text
    assert "display_table(artifact_manifest" not in code_text
    assert "expected_png_figures" not in code_text
    assert "artifact_manifest" not in code_text
    assert "Final artifact audit" not in markdown_text
    assert "Artifact validation" not in markdown_text

    setup_outputs = payload["cells"][3].get("outputs", [])
    assert len(setup_outputs) <= 1
    assert all("text/html" not in (output.get("data") or {}) for output in setup_outputs)


def test_ch03_toy_notebook_uses_compact_figure_and_summary_cells():
    payload = _payload(CH03_TOY_NOTEBOOK)
    code_sources = _sources(CH03_TOY_NOTEBOOK, "code")
    code_text = "\n".join(code_sources)
    markdown_text = "\n".join(_sources(CH03_TOY_NOTEBOOK, "markdown"))

    assert len(payload["cells"]) <= 33
    assert code_text.count("ch03.save_and_close_figure(") == 0
    assert code_text.count("save_and_show(") == 6
    assert "save_and_show = make_save_and_show(" in code_text

    for noisy_name in [
        "toy_design = pd.DataFrame",
        "model_summary = pd.DataFrame",
        "sampling_design = pd.DataFrame",
        "trajectory_summary = pd.DataFrame",
        "velocity_probe_summary = pd.DataFrame",
        "hierarchy_preview = pd.DataFrame",
    ]:
        assert noisy_name not in code_text, noisy_name

    assert "step / n_steps" in code_text
    assert "nfe = 0" not in code_text
    assert "nfe * dt" not in code_text
    assert "return x, traj, int(n_steps)" in code_text

    markdown_lower = markdown_text.lower()
    for teaching_phrase in [
        "multi-modal source",
        "conditional velocity is constant",
        "training never integrates this ODE",
        "paper Figure 3.2",
        "Panel A",
    ]:
        assert teaching_phrase.lower() in markdown_lower, teaching_phrase


def test_ch03_toy_notebook_teaches_core_cfm_inline():
    code_text = "\n".join(_sources(CH03_TOY_NOTEBOOK, "code"))

    hidden_core_paths = [
        "from src.core.train import train_cfm_steps",
        "train_cfm_steps(",
        "from src.core.losses import cfm_loss_from_pairs",
        "cfm_loss_from_pairs(",
        "from src.core.sampling import euler_sample",
        "euler_sample(",
    ]
    for hidden_path in hidden_core_paths:
        assert hidden_path not in code_text, hidden_path

    inline_fragments = [
        "def make_cfm_batch(",
        "x_t = (1.0 - t) * x0 + t * x1",
        "target_velocity = x1 - x0",
        "def cfm_loss_from_batch(",
        "pred_velocity = model(batch[\"x_t\"], batch[\"t\"])",
        "torch.mean((pred_velocity - batch[\"target_velocity\"]) ** 2)",
        "for step in range(",
        "optimizer.zero_grad(",
        "loss.backward()",
        "optimizer.step()",
        "def euler_rollout(",
        "dt = 1.0 / n_steps",
        "x = x + dt * velocity",
    ]
    for fragment in inline_fragments:
        assert fragment in code_text, fragment


def test_ch03_eb20d_main_helpers_are_extracted_and_callable():
    torch = pytest.importorskip("torch")
    import matplotlib.pyplot as plt

    from src.visualization import flow_matching as tutorial

    X0 = np.linspace(-1.0, 1.0, 80, dtype=np.float32).reshape(20, 4)
    X1 = X0 + 0.5

    train_idx, val_idx = tutorial.train_val_indices(len(X0), train_fraction=0.7, seed=21)
    assert len(train_idx) == 14
    assert len(val_idx) == 6
    assert set(train_idx).isdisjoint(set(val_idx))
    assert np.array_equal(train_idx, np.sort(train_idx))
    assert np.array_equal(val_idx, np.sort(val_idx))

    pair_batch_fn = tutorial.make_random_pair_batch_fn(X0[train_idx], X1[train_idx], seed=22)
    batch = pair_batch_fn(5)
    assert batch["x0"].shape == (5, 4)
    assert batch["x1"].shape == (5, 4)

    class ZeroVelocity(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    mse = tutorial.val_cfm_mse(
        ZeroVelocity(),
        X0[val_idx],
        X1[val_idx],
        np.asarray([0.25, 0.50, 0.75], dtype=np.float32),
        device="cpu",
        max_eval_pairs=4,
        seed=23,
    )
    assert np.isfinite(mse)
    assert mse > 0

    fig = tutorial.plot_endpoint_pairs_phate(
        X0[:, :2],
        X1[:, :2],
        n_pairs=6,
        seed=24,
        source_time="0",
        target_time="1",
    )
    assert len(fig.axes) == 1
    plt.close(fig)


def test_ch03_eb20d_main_notebook_calls_extracted_helpers():
    code_text = "\n".join(_sources(CH03_EB_MAIN_NOTEBOOK, "code"))

    for local_definition in [
        "def save_figure(",
        "def save_paper_figure(",
        "def save_csv(",
        "def save_paper_table(",
        "def save_run_json(",
        "def display_saved_figure(",
        "def display_saved_figures(",
        "def display_table(",
        "def train_val_indices(",
        "def val_cfm_mse(",
        "def plot_endpoint_pairs_phate(",
        "def raise_for_missing_artifacts(",
    ]:
        assert local_definition not in code_text, local_definition

    for helper_call in [
        "make_save_and_show(",
        "ch03.train_val_indices(",
        "ch03.make_random_pair_batch_fn(",
        "ch03.val_cfm_mse(",
        "ch03.plot_endpoint_pairs_phate(",
    ]:
        assert helper_call in code_text, helper_call


def test_ch03_eb20d_main_notebook_keeps_reader_facing_outputs_clean():
    payload = _payload(CH03_EB_MAIN_NOTEBOOK)
    code_sources = _sources(CH03_EB_MAIN_NOTEBOOK, "code")
    code_text = "\n".join(code_sources)
    markdown_text = "\n".join(_sources(CH03_EB_MAIN_NOTEBOOK, "markdown"))

    assert len(payload["cells"]) <= 34
    assert "artifact_locations" not in code_text
    assert "artifact_manifest" not in code_text
    assert "Final artifact audit" not in markdown_text
    assert "Artifact validation" not in markdown_text

    for noisy_name in [
        "model_config_table = pd.DataFrame",
        "validation_plan = pd.DataFrame",
        "sampling_plan = pd.DataFrame",
        "model_cache_table = pd.DataFrame",
        "data_audit = pd.DataFrame",
        "pairing_table = pd.DataFrame",
    ]:
        assert noisy_name not in code_text, noisy_name

    for noisy_display in [
        "display_table(artifact_locations",
        "display_table(model_config_table",
        "display_table(validation_plan",
        "display_table(sampling_plan",
        "display_table(model_cache_table",
        "display_table(data_audit",
        "display_table(split_table",
        "display_table(pairing_table",
    ]:
        assert noisy_display not in code_text, noisy_display

    assert "ch03.save_csv(TABLE_DIR / \"ch03_eb20d_train_val_split.csv\", split_table)" in code_text
    assert "X_cost shape=" in code_text
    assert "source train/val=" in code_text
    assert "Training pairs draw fresh independent endpoints" in code_text
    assert code_text.count("save_and_show(") == 5
    assert "save_and_show = make_save_and_show(" in code_text


def test_ch03_eb20d_main_notebook_teaches_core_cfm_inline():
    code_text = "\n".join(_sources(CH03_EB_MAIN_NOTEBOOK, "code"))
    markdown_text = "\n".join(_sources(CH03_EB_MAIN_NOTEBOOK, "markdown")).lower()

    hidden_core_paths = [
        "from src.core.losses import cfm_loss_from_pairs",
        "cfm_loss_from_pairs(",
        "from src.core.sampling import euler_sample",
        "euler_sample(",
    ]
    for hidden_path in hidden_core_paths:
        assert hidden_path not in code_text, hidden_path

    inline_fragments = [
        "def make_cfm_batch(",
        "x_t = (1.0 - t) * x0 + t * x1",
        "target_velocity = x1 - x0",
        "def cfm_loss_from_batch(",
        "pred_velocity = model(batch[\"x_t\"], batch[\"t\"])",
        "((pred_velocity - batch[\"target_velocity\"]) ** 2).mean(dim=-1).mean()",
        "for step in range(1, eb_steps + 1):",
        "optimizer.zero_grad(",
        "loss.backward()",
        "optimizer.step()",
        "def euler_rollout(",
        "times = torch.linspace(0.0, 1.0, n_steps + 1",
        "dt = times[step + 1] - times[step]",
        "x = x + dt * velocity",
    ]
    for fragment in inline_fragments:
        assert fragment in code_text, fragment

    for teaching_phrase in [
        "20d pc space",
        "phate is display-only",
        "endpoint pairs",
        "population evolution",
        "Euler step count",
    ]:
        assert teaching_phrase.lower() in markdown_text, teaching_phrase

    assert "paper figure" not in markdown_text
    assert "figure 3." not in markdown_text


def test_ch03_eb20d_ablation_helpers_are_extracted_and_callable(tmp_path):
    torch = pytest.importorskip("torch")
    import matplotlib.pyplot as plt

    from src.visualization import flow_matching as tutorial

    context = tutorial.Ch03Context(
        project_root=tmp_path,
        fig_dir=tmp_path / "figures" / "ch03",
        table_dir=tmp_path / "tables" / "ch03",
        output_dir=tmp_path / "outputs" / "ch03",
    )
    tracker = tutorial.Ch03ArtifactTracker(context, paper_figure_mode=True)

    table_path = tracker.save_csv([{"a": 1}], "table.csv")
    assert table_path.exists()
    assert tracker.tables_written == ["tables/ch03/table.csv"]

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig_path = tracker.save_figure(fig, "figure.png")
    assert fig_path.exists()
    assert "figures/ch03/figure.png" in tracker.figures_written
    assert "figures/ch03/figure.pdf" in tracker.paper_ready_pdf_written

    paper_paths = tracker.save_paper_table([{"metric": "mmd", "value": 0.5}], "paper_table")
    assert all(path.exists() for path in paper_paths)
    assert set(tracker.paper_tables_written) == {
        "tables/ch03/paper_table.csv",
        "tables/ch03/paper_table.tex",
        "tables/ch03/paper_table.md",
    }

    t = tutorial.sample_t_numpy(
        "logit_normal_sigma_2.0",
        64,
        seed=31,
        strategy_specs={"logit_normal_sigma_2.0": {"sigma": 2.0}},
    )
    assert t.shape == (64,)
    assert t.dtype == np.float32
    assert np.all((t > 0.0) & (t < 1.0))
    t_tensor = tutorial.sample_t_torch("beta_2_2", 8, "cpu", seed=32)
    assert tuple(t_tensor.shape) == (8, 1)

    traj = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [1.0, 1.0]],
        ]
    )
    path_len, endpoint, ratio = tutorial.per_trajectory_straightness(traj)
    assert np.allclose(path_len, [2.0, 2.0])
    assert np.allclose(endpoint, [2.0, np.sqrt(2.0)])
    assert np.allclose(ratio, [1.0, np.sqrt(2.0)])

    normalized = tutorial.normalize_skipped_items(["x", {"item": "y", "reason": "z"}])
    assert normalized == [
        {"item": "x", "reason": "recorded by earlier section"},
        {"item": "y", "reason": "z"},
    ]

    class LinearVelocity(torch.nn.Module):
        def forward(self, x, t):
            return torch.ones_like(x) * t

    ode_func = tutorial.EndpointODEFunc(LinearVelocity())
    out = ode_func(torch.tensor(0.5), torch.zeros(3, 2))
    assert torch.allclose(out, torch.full((3, 2), 0.5))
    assert ode_func.nfe == 1
    ode_func.reset_nfe()
    assert ode_func.nfe == 0

    solver_table = pd.DataFrame(
        [
            {"sampler": "euler", "nfe": 10, "mmd_20d": 0.42},
            {"sampler": "euler", "nfe": 20, "mmd_20d": 0.31},
            {"sampler": "dopri5", "nfe": 17, "mmd_20d": 0.29},
        ]
    )
    fig, ax = tutorial.plot_nfe_vs_error(solver_table, y="mmd_20d")
    assert ax.get_xlabel() == "number of function evaluations (NFE)"
    assert ax.get_ylabel() == "mmd 20d"
    assert len(ax.lines) == 2
    plt.close(fig)


def test_format_solver_diagnostics_paper_table_rounds_and_labels():
    from src.visualization import flow_matching as tutorial

    solver_table = pd.DataFrame(
        [
            {
                "sampler": "euler",
                "steps": 20,
                "nfe": 20.0,
                "wall_time_sec": 0.01234,
                "mmd_20d": 0.123456,
                "sliced_w2_20d": 0.98765,
                "trajectory_straightness_20d": 1.23456,
            },
            {
                "sampler": "dopri5",
                "steps": "adaptive",
                "nfe": 41,
                "wall_time_sec": 0.12345,
                "mmd_20d": 0.222222,
                "sliced_w2_20d": 1.11119,
                "trajectory_straightness_20d": 9.99999,
            },
        ]
    )

    formatted = tutorial.format_solver_diagnostics_paper_table(solver_table)

    assert list(formatted.columns) == [
        "Solver",
        "Steps",
        "NFE",
        "Time (ms)",
        "MMD (20D) ↓",
        "Sliced W2 (20D) ↓",
        "Straightness (20D)",
    ]
    assert formatted.to_dict("records") == [
        {
            "Solver": "euler",
            "Steps": "20",
            "NFE": 20,
            "Time (ms)": 12.3,
            "MMD (20D) ↓": 0.1235,
            "Sliced W2 (20D) ↓": 0.988,
            "Straightness (20D)": 1.235,
        },
        {
            "Solver": "dopri5",
            "Steps": "adaptive",
            "NFE": 41,
            "Time (ms)": 123.4,
            "MMD (20D) ↓": 0.2222,
            "Sliced W2 (20D) ↓": 1.111,
            "Straightness (20D)": "N/A",
        },
    ]
    assert "Solver" not in solver_table.columns


def test_ch03_eb20d_ablation_notebook_calls_extracted_helpers():
    code_text = "\n".join(_sources(CH03_EB_ABLATION_NOTEBOOK, "code"))

    for local_definition in [
        "def _round_float(",
        "def _rel(",
        "def save_figure(",
        "def save_paper_figure(",
        "def save_csv(",
        "def save_paper_table(",
        "def save_run_json(",
        "def display_saved_figure(",
        "def display_saved_figures(",
        "def display_table(",
        "def train_val_indices(",
        "def val_cfm_mse(",
        "def record_skip(",
        "def make_seeded_pair_batch_fn(",
        "def endpoint_mmd_sliced_20d(",
        "def eval_cfm_endpoint_20d(",
        "def eval_ode_endpoint_20d(",
        "class EndpointODEFunc(",
        "def sample_t_numpy(",
        "def sample_t_torch(",
        "def per_trajectory_straightness(",
        "def normalize_skipped_items(",
    ]:
        assert local_definition not in code_text, local_definition

    for helper_call in [
        "ch03.Ch03ArtifactTracker(",
            "make_save_and_show(",
            "tracker.save_paper_table(",
        "ch03.train_val_indices(",
        "ch03.make_random_pair_batch_fn(",
        "ch03.val_cfm_mse(",
        "ch03.eval_cfm_endpoint_20d(",
        "ch03.eval_ode_endpoint_20d(",
        "ch03.EndpointODEFunc(",
        "ch03.sample_t_numpy(",
        "ch03.sample_t_torch(",
        "ch03.per_trajectory_straightness(",
        "ch03.format_solver_diagnostics_paper_table(",
    ]:
        assert helper_call in code_text, helper_call

    assert "from src.plots import" not in code_text
    assert "ch03.plot_nfe_vs_error(" in code_text


def test_ch03_eb20d_ablation_notebook_keeps_reader_facing_outputs_clean():
    payload = _payload(CH03_EB_ABLATION_NOTEBOOK)
    code_sources = _sources(CH03_EB_ABLATION_NOTEBOOK, "code")
    code_text = "\n".join(code_sources)
    markdown_text = "\n".join(_sources(CH03_EB_ABLATION_NOTEBOOK, "markdown"))

    assert len(payload["cells"]) <= 65
    assert "artifact_manifest" not in code_text
    assert "Final artifact audit" not in markdown_text
    assert "Artifact validation" not in markdown_text

    for index, source in enumerate(code_sources, 1):
        assert source.strip(), (CH03_EB_ABLATION_NOTEBOOK, index)
        assert not all(
            line.strip().startswith("#") or not line.strip()
            for line in source.splitlines()
        ), (CH03_EB_ABLATION_NOTEBOOK, index)

    for noisy_heading in [
        "### Question",
        "### Configuration table",
        "### Run",
        "### Metrics table",
        "### Figure",
        "### Claim boundary",
    ]:
        assert noisy_heading not in markdown_text, noisy_heading

    for noisy_config_display in [
        "tracker.display_table(solver_config",
        "tracker.display_table(E1_config",
        "tracker.display_table(time_config",
        "tracker.display_table(capacity_config_table",
        "tracker.display_table(straight_config",
    ]:
        assert noisy_config_display not in code_text, noisy_config_display

    for explanatory_print in [
        "solver settings:",
        "ODE-in-loop cost benchmark:",
        "time-sampling strategies:",
        "capacity configs:",
        "straightness diagnostic:",
    ]:
        assert explanatory_print in code_text, explanatory_print

    assert "figE1_cfm_vs_cnf_endpoint_training_cost" not in code_text
    assert "figE1_cfm_vs_cnf_endpoint_samples_phate" not in code_text
    assert "val_endpoint_mmd_20d" not in code_text
    assert "fixed_ot_pairs" in code_text
    assert "relative_slowdown_vs_fm" in code_text

    assert "save_and_show(" in code_text
    assert "save_and_show = make_save_and_show(" in code_text


def test_old_ch03_monolith_is_retired_index_only():
    active_payload = _payload_from_path(RETIRED_INDEX_PATH)
    active_text = "\n".join("".join(cell.get("source", [])) for cell in active_payload["cells"])
    active_code_sources = [
        "".join(cell.get("source", []))
        for cell in active_payload["cells"]
        if cell.get("cell_type") == "code"
    ]

    assert "retired" in active_text.lower()
    for filename in SPLIT_SPECS:
        assert filename in active_text
    assert len(active_code_sources) <= 1
    assert "Train EB 20D VelocityMLP" not in active_text
    assert "CNF-Endpoint Baseline" not in active_text
    assert "save_figure(" not in active_text
    assert "train_velocity_model" not in active_text

    archived_payload = _payload_from_path(ARCHIVED_MONOLITH_PATH)
    archived_text = "\n".join("".join(cell.get("source", [])) for cell in archived_payload["cells"])
    archived_code_sources = [
        "".join(cell.get("source", []))
        for cell in archived_payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert len(archived_code_sources) >= 10
    assert "Train EB 20D VelocityMLP" in archived_text
    assert "CNF-Endpoint Baseline" in archived_text
