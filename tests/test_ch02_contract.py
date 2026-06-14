from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_ch02_distribution_transport.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_distribution_transport_before_fm.ipynb"
TUTORIAL_HELPER_PATH = PROJECT_ROOT / "src" / "transport_reporting.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_ch02_distribution_transport", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ch02_runner_declares_required_artifacts_and_boundaries():
    module = _load_runner()
    assert hasattr(module, "run_ch02")

    expected_figures = {
        "fig02_02_static_ot_endpoint_transport",
        "fig02_03_same_endpoints_different_paths",
        "fig02_04_dynamic_ot_low_action",
        "fig02_06_training_bottleneck_timing",
    }
    expected_tables = {
        "table02_01_coupling_diagnostics.csv",
        "table02_02_path_diagnostics.csv",
        "table02_03_dynamic_ot_energy_proxy.csv",
        "table02_04_cnf_training_bottleneck.csv",
    }
    assert expected_figures.issubset(set(module.EXPECTED_FIGURE_STEMS))
    assert expected_tables.issubset(set(module.EXPECTED_TABLES))

    boundary_text = "\n".join(module.CONCEPT_BOUNDARIES)
    assert "not observed lineage" in boundary_text
    assert "PC-20" in boundary_text
    assert "not solved Benamou-Brenier" in boundary_text
    assert "not full likelihood CNF" in boundary_text
    assert "Flow Matching training is deferred to Chapter 3" in boundary_text


def test_ch02_notebook_is_paper_facing_reproducibility_notebook():
    payload = json.loads(NOTEBOOK_PATH.read_text())
    assert payload["nbformat"] >= 4
    text = "".join("".join(cell.get("source", [])) for cell in payload["cells"])
    code_text = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "code"
    )
    assert "Chapter 2. Distribution Transport Before Flow Matching" in text
    assert "run_ch02(quick_mode=True, seed=42)" not in text
    assert "from run_ch02_distribution_transport import run_ch02" not in text
    assert len(payload["cells"]) >= 35
    for required in [
        "C_raw",
        "C_norm",
        "cost_scale",
        "pi_ind",
        "plans",
        "coupling_table",
        "pi_main",
        "sampled_endpoint_pairs",
        "path_diagnostics",
        "energy_table",
        "cnf_training_table",
    ]:
        assert required in code_text
    assert "table02_01_coupling_diagnostics.csv" in text
    assert "fig02_04c_pc20_action_proxy" in text
    assert "table02_04_cnf_training_bottleneck.csv" in text
    assert "table02_optional_ot_cost_sensitivity.csv" in text


def test_ch02_notebook_has_second_round_tutorial_quality_gates():
    payload = json.loads(NOTEBOOK_PATH.read_text())
    code_sources = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    code_text = "\n".join(code_sources)
    helper_source = TUTORIAL_HELPER_PATH.read_text()
    max_code_lines = max(len(source.splitlines()) for source in code_sources)

    assert len(code_sources) >= 42
    assert max_code_lines <= 60
    assert "raise FileNotFoundError" in helper_source

    display_markers = [
        "display(Image(",
        "display_saved_figure(",
        "display_saved_figures(",
        "display_png(",
        "show_saved_png(",
    ]
    assert any(marker in code_text for marker in display_markers)
    assert "pd.DataFrame(" in code_text or "show_saved_png(" in code_text

    assert "plt.subplots(2, 2" not in code_text
    assert "plt.subplots(1, 4" not in code_text
    assert 'figure_suffixes: Iterable[str] = (".png", ".svg")' in helper_source
    assert "Independent: geometry ignored" in code_text
    assert "Sinkhorn OT: PC-20 cost-guided" in code_text
    assert "Coarse transport blocks" in code_text
    assert "cells grouped by PHATE-1 quantiles" in code_text
    assert "coarse_pi" in code_text
    assert "row_normalized_coarse_pi" in code_text
    assert "row-normalized coupling mass" in code_text
    assert "each row sums to 1 source group" in code_text
    assert "Epsilon trade-off" in code_text
    assert "PHATE display only" in code_text
    assert "chosen eps=0.05" in code_text
    assert "fig02_06a_cnf_control_flow" not in code_text
    assert "box_specs" not in code_text
    assert "A. Independent" not in code_text
    assert "B. Sinkhorn" not in code_text
    assert "C. Soft" not in code_text
    assert "D. Epsilon" not in code_text
    assert "Independent endpoint coupling" not in code_text
    assert "Sinkhorn OT endpoint coupling" not in code_text

    for stem in [
        "fig02_02a_static_ot_independent_endpoint_coupling",
        "fig02_02b_static_ot_sinkhorn_endpoint_coupling",
        "fig02_02c_static_ot_transport_plan_heatmap",
        "fig02_02d_static_ot_epsilon_sensitivity",
        "fig02_03a_endpoint_coupling_only",
        "fig02_03b_straight_bridges",
        "fig02_03c_curved_bridges",
        "fig02_03d_stochastic_bridge_samples",
        "fig02_04a_fixed_endpoint_pairs",
        "fig02_04b_path_construction_fixed_endpoints",
        "fig02_04c_pc20_action_proxy",
        "fig02_06b_mean_step_time",
        "fig02_06c_velocity_evaluations_per_step",
        "fig02_06d_total_wall_clock_time",
    ]:
        assert stem in code_text

    for filename in [
        "table02_01_coupling_diagnostics.csv",
        "table02_optional_ot_cost_sensitivity.csv",
        "table02_02_path_diagnostics.csv",
        "table02_03_dynamic_ot_energy_proxy.csv",
        "table02_04_cnf_training_bottleneck.csv",
        "table02_04_training_cost_proxy.csv",
    ]:
        assert filename in code_text


def test_ch02_expected_artifacts_exist():
    figure_dir = PROJECT_ROOT / "figures" / "ch02"
    output_dir = PROJECT_ROOT / "outputs" / "ch02"
    runner_figure_stems = [
        "fig02_02_static_ot_endpoint_transport",
        "fig02_03_same_endpoints_different_paths",
        "fig02_04_dynamic_ot_low_action",
        "fig02_06_training_bottleneck_timing",
    ]
    notebook_panel_figure_stems = [
        "fig02_02a_static_ot_independent_endpoint_coupling",
        "fig02_02b_static_ot_sinkhorn_endpoint_coupling",
        "fig02_02c_static_ot_transport_plan_heatmap",
        "fig02_02d_static_ot_epsilon_sensitivity",
        "fig02_03a_endpoint_coupling_only",
        "fig02_03b_straight_bridges",
        "fig02_03c_curved_bridges",
        "fig02_03d_stochastic_bridge_samples",
        "fig02_04a_fixed_endpoint_pairs",
        "fig02_04b_path_construction_fixed_endpoints",
        "fig02_04c_pc20_action_proxy",
        "fig02_06b_mean_step_time",
        "fig02_06c_velocity_evaluations_per_step",
        "fig02_06d_total_wall_clock_time",
    ]
    for stem in runner_figure_stems + notebook_panel_figure_stems:
        assert (figure_dir / f"{stem}.png").exists()
        assert (figure_dir / f"{stem}.svg").exists()

    expected_tables = [
        "table02_01_coupling_diagnostics.csv",
        "table02_optional_ot_cost_sensitivity.csv",
        "table02_02_path_diagnostics.csv",
        "table02_03_dynamic_ot_energy_proxy.csv",
        "table02_04_cnf_training_bottleneck.csv",
        "table02_04_training_cost_proxy.csv",
    ]
    for filename in expected_tables:
        assert (output_dir / filename).exists()


def test_ch02_runner_uses_final_figure_annotations():
    source = RUNNER_PATH.read_text()
    assert "not converged" in source
    assert "main eps=0.05" in source
    assert "energy proxy (PC-20)" in source
    assert "CNF training bottleneck" in source
    assert "Flow Matching training is deferred to Chapter 3" in source
    assert "table02_04_cnf_training_bottleneck.csv" in source
    assert "total wall-clock time for" in source.lower()
    assert "Loss vs wall-clock" not in source
    assert "training loss" not in source
    assert "solver-in-the-loop Neural ODE training proxy" in source
    assert "FM is faster" not in source
    assert "local velocity regression" not in source
    assert "training_step_time_ratio_solver_over_local" not in source


def test_ch02_notebook_documents_final_claim_boundaries_and_numbers():
    payload = json.loads(NOTEBOOK_PATH.read_text())
    text = "".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert (
        "Sinkhorn OT at ε=0.05 reduces expected PC-20 transport cost from 299.6 "
        "to 161.4, while concentrating mass relative to the independent coupling. "
        "Smaller ε may reduce cost further but did not converge in this run."
    ) in text
    assert "PHATE is visualization only" in text
    assert "not solved Benamou-Brenier" in text
    assert "CNF training bottleneck" in text
    assert "Solver-in-loop baseline is not full likelihood CNF" in text
    assert "Flow Matching training is deferred to Chapter 3" in text
    assert "The next chapter asks whether the same velocity field can be trained without solving the learned ODE inside the training loop." in text
    assert "Local velocity regression previews Chapter 3" not in text
    assert "local velocity regression" not in text
    assert "FM is faster" not in text
    assert "Flow Matching loss" not in text
    assert "full likelihood CNF training" not in text
    assert "25x" not in text
