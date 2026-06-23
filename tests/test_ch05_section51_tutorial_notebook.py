from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "chapter5_1_timecourse_suite.ipynb"
RETIRED_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_1_single_cell_timecourse_main_suite.ipynb"
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_ch05_section51_notebook.py"


def _notebook_text() -> str:
    payload = json.loads(NOTEBOOK_PATH.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def _notebook_code_lengths() -> list[int]:
    payload = json.loads(NOTEBOOK_PATH.read_text())
    lengths = []
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            lengths.append(source.count("\n") + bool(source))
    return lengths


def test_section51_canonical_notebook_exists_and_retired_active_copy_is_removed():
    assert NOTEBOOK_PATH.exists()
    assert not RETIRED_NOTEBOOK_PATH.exists()


def test_section51_tutorial_helpers_are_generic_and_reusable(tmp_path):
    from src.experiments import timecourse_config as tutorial

    for name in [
        "Section51Config",
        "resolve_project_root",
        "make_section51_config",
        "ensure_ch05_dirs",
        "json_ready",
        "save_json",
        "save_csv",
        "write_section51_artifacts",
        "display_png",
        "preview_frame",
        "audit_section51_main_text_results",
        "verify_expected_display_values",
        "section51_expected_display_values",
    ]:
        assert hasattr(tutorial, name), name

    assert tutorial.json_ready({"x": np.float32(1.5), "y": np.asarray([1, 2])}) == {"x": 1.5, "y": [1, 2]}

    json_path = tutorial.save_json(tmp_path / "nested" / "payload.json", {"a": np.int64(2)})
    csv_path = tutorial.save_csv(tmp_path / "nested" / "table.csv", pd.DataFrame({"a": [1]}))
    assert json.loads(json_path.read_text()) == {"a": 2}
    assert pd.read_csv(csv_path).shape == (1, 1)


def test_section51_notebook_is_tutorial_style_and_keeps_core_experiment_visible():
    text = _notebook_text()
    code_lengths = _notebook_code_lengths()

    for required in [
        "## 1. What this experiment tests",
        "## 4. Define pair topology and model variants",
        "## 5. Train pairwise local bridges",
        "## 6. Train shared global bridges",
        "## 7. Roll out predictions",
        "## 8. Compute endpoint metrics",
        "## 9. Compute hand-off diagnostics",
        "## 12. Redraw Figure 5.1 as independent paper panels",
    ]:
        assert required in text

    assert "run_eb_section51_main_suite(" not in text
    for required_core_step in [
        "for run_seed in SECTION51_MAIN_SUITE_SEEDS",
        "_train_local_bridge(",
        "_train_global_bridge_model(",
        "_local_sequence_rollout(",
        "_global_rollout(",
        "endpoint_distribution_metrics(",
        "diag_rows.append",
    ]:
        assert required_core_step in text

    for forbidden in [
        "fig_5_1_main_suite",
        "fig5_1_combined",
        "draw_fig5_1_combined_panel",
        "display_png(figure_path)",
    ]:
        assert forbidden not in text

    for expected_panel in [
        "fig5_1_time_pair_designs.png",
        "fig5_1_hidden_t2_recovery.png",
        "fig5_1_seen_t4_rollout.png",
        "fig5_1_velocity_jump.png",
    ]:
        assert expected_panel in text

    assert "from src.tutorial_init import apply_tutorial_plot_style, bootstrap, make_save_and_show" in text
    assert "save_and_show = make_save_and_show(" in text
    assert "display_fn=save_and_show" in text
    assert "display_fn=display_png" not in text
    assert "from src.visualization.timecourse import" in text
    assert "_fig5_1_metric_table" not in text
    assert "_fig5_1_draw_metric_bars" not in text
    assert "_fig5_1_crop_white_margin" not in text
    assert len(code_lengths) >= 18
    assert max(code_lengths) <= 90


def test_section51_notebook_keeps_legacy_skip_rng_alignment_control():
    text = _notebook_text()

    uniform_variant = '"variant": "shared_skip_uniform_6000"'
    weighted_variant = '"variant": "shared_skip_adj2_skip1_9000"'
    assert uniform_variant in text
    assert text.index(uniform_variant) < text.index(weighted_variant)

    for required in [
        "main_method_variants = [",
        "analysis_specs = [",
        "if spec[\"variant\"] in main_method_variants",
        "for spec in analysis_specs:",
    ]:
        assert required in text


def test_section51_notebook_explains_cfm_training_wrappers():
    text = _notebook_text()

    for required in [
        "CFM velocity-regression training loop",
        "_train_local_bridge",
        "_train_global_bridge_model",
        "cfm_loss_from_pairs",
        "zero_grad",
        "backward",
        "step",
    ]:
        assert required in text


def test_section51_builder_is_retired_and_does_not_write_notebook():
    source = BUILDER_PATH.read_text()
    assert "maintained directly" in source
    assert "notebooks/chapter5_1_timecourse_suite.ipynb" in source
    assert "raise SystemExit" in source
    assert "nbf.write" not in source
    assert "new_notebook" not in source

    result = subprocess.run(
        [sys.executable, str(BUILDER_PATH)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "maintained directly" in result.stderr + result.stdout
