from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "04_3_sampling_depth_and_claim_boundaries.ipynb"


def test_wfrfm_sampling_helpers_summarize_mass_convention_outputs(tmp_path: Path):
    from src.ch04_sampling_depth_tutorial import (
        make_wfrfm_agreement_summary,
        make_wfrfm_growth_delta_grid,
        resolve_wfrfm_output_suffix,
        wfrfm_output_name,
    )

    assert resolve_wfrfm_output_suffix(tmp_path, env_suffix=" _full ") == "full"
    assert wfrfm_output_name("table4_6c_wfrfm_growth_by_bin", "csv", "full") == "table4_6c_wfrfm_growth_by_bin_full.csv"

    growth = pd.DataFrame(
        {
            "setting": ["raw_observed_depth", "raw_observed_depth", "equal_depth", "equal_depth"],
            "eval_time": [1, 1, 1, 1],
            "state_bin": ["0", "1", "0", "1"],
            "mean_g": [1.5, -0.25, 0.5, -0.75],
        }
    )
    delta = make_wfrfm_growth_delta_grid(growth)
    np.testing.assert_allclose(delta.matrix, np.asarray([[1.0, 0.5]]))
    assert delta.eval_order == [1]
    assert delta.bin_order == ["0", "1"]

    sensitivity = pd.DataFrame(
        {
            "spearman_growth_rank": [0.8, 1.0],
            "top_expanding_overlap_k3": [1.0, 0.5],
            "top_shrinking_overlap_k3": [0.5, 0.5],
            "sign_agreement": [0.25, 0.75],
        }
    )
    summary = make_wfrfm_agreement_summary(sensitivity)
    assert summary["metric"].tolist() == ["Spearman rank", "Top expanding", "Top shrinking", "Sign agreement"]
    np.testing.assert_allclose(summary["value"], [0.9, 0.75, 0.5, 0.5])


def test_final_figure_package_manifest_validates_figures_and_sources(tmp_path: Path):
    from src.ch04_sampling_depth_tutorial import FINAL_FIGURE_CLAIMS, write_final_figure_package

    final_fig_dir = tmp_path / "figures" / "new3"
    final_fig_dir.mkdir(parents=True)
    source = tmp_path / "outputs" / "table.csv"
    source.parent.mkdir(parents=True)
    source.write_text("a\n1\n", encoding="utf-8")

    for stem in FINAL_FIGURE_CLAIMS:
        for ext in ("png", "pdf", "svg"):
            (final_fig_dir / f"{stem}.{ext}").write_bytes(b"artifact")

    package = write_final_figure_package(
        project_root=tmp_path,
        final_fig_dir=final_fig_dir,
        source_paths={"table.csv": source},
        final_source_paths={"table.csv": "outputs/table.csv"},
    )

    assert package.qa_table["status"].tolist() == ["pass", "pass", "pass", "pass", "pass"]
    assert package.readme_path.exists()
    assert package.manifest_path.exists()
    assert "Raw destructive snapshot counts" in package.readme_path.read_text(encoding="utf-8")


def test_bridge_sampling_diagnostic_preserves_sampling_depth_contract():
    from src.ch04_sampling_depth_tutorial import bridge_sampling_diagnostic

    pcs_all = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [1.0, 0.0],
            [1.1, 0.0],
            [1.0, 0.1],
            [1.1, 0.1],
            [1.2, 0.0],
            [1.2, 0.1],
        ],
        dtype=np.float32,
    )
    labels_all = np.asarray(["1", "1", "1", "1", "2", "2", "2", "2", "2", "2"])
    state_bins = np.asarray(["0", "0", "1", "1", "0", "0", "1", "1", "1", "1"])

    original = bridge_sampling_diagnostic(
        pcs_all=pcs_all,
        labels_all=labels_all,
        state_bins=state_bins,
        all_bins=["0", "1"],
        time_a="1",
        time_b="2",
        sampling_setting="original_depth",
        cap=5,
        seed=17,
        sinkhorn_epsilon=0.2,
    )
    repeated = bridge_sampling_diagnostic(
        pcs_all=pcs_all,
        labels_all=labels_all,
        state_bins=state_bins,
        all_bins=["0", "1"],
        time_a="1",
        time_b="2",
        sampling_setting="original_depth",
        cap=5,
        seed=17,
        sinkhorn_epsilon=0.2,
    )
    equal_depth = bridge_sampling_diagnostic(
        pcs_all=pcs_all,
        labels_all=labels_all,
        state_bins=state_bins,
        all_bins=["0", "1"],
        time_a="1",
        time_b="2",
        sampling_setting="equal_depth",
        cap=5,
        seed=18,
        sinkhorn_epsilon=0.2,
    )

    assert original == repeated
    assert original["time_bridge"] == "1->2"
    assert original["n_source"] == 4
    assert original["n_target"] == 5
    assert equal_depth["n_source"] == 4
    assert equal_depth["n_target"] == 4
    assert original["diagnostic_type"] == "ot_sampled_endpoint_diagnostic_not_trained_cfm"
    assert "not trained CFM" in original["claim_boundary"]
    for key in [
        "endpoint_mmd_pc20",
        "sliced_w2_pc20",
        "state_bin_terminal_proportion_error",
        "expected_cost_normalized",
        "effective_support",
    ]:
        assert np.isfinite(original[key]), key
    assert isinstance(original["sinkhorn_converged"], bool)

    with pytest.raises(ValueError, match="unknown sampling_setting"):
        bridge_sampling_diagnostic(
            pcs_all=pcs_all,
            labels_all=labels_all,
            state_bins=state_bins,
            all_bins=["0", "1"],
            time_a="1",
            time_b="2",
            sampling_setting="bad",
            cap=5,
            seed=17,
            sinkhorn_epsilon=0.2,
        )


def test_sampling_depth_notebook_uses_src_helpers_instead_of_long_local_blocks():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_sources = ["".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "code"]
    code_text = "\n".join(code_sources)

    assert "from src.ch04_sampling_depth_tutorial import" in code_text
    assert "def save_pub_figure(" not in code_text
    assert "def load_eb_data(" not in code_text
    assert "def bridge_sampling_diagnostic(" not in code_text
    assert "def resolve_wfrfm_output_suffix(" not in code_text
    assert "FINAL_FIGURE_CLAIMS = {" not in code_text
    assert max(source.count("\n") + bool(source) for source in code_sources) <= 100
