from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_1_single_cell_timecourse_main_suite.ipynb"
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


def _minimal_summary_frame() -> pd.DataFrame:
    rows = []
    for target in ["hidden_t2", "seen_t4"]:
        for variant, family, steps, sampling in [
            ("pairwise_local_bridges_6000", "pairwise", 6000, "local_pairwise"),
            ("shared_adjacent_only_6000", "shared_adjacent", 6000, "uniform_adjacent"),
            ("shared_skip_uniform_6000", "shared_skip", 6000, "uniform_adjacent_skip"),
            ("shared_skip_adj2_skip1_9000", "shared_skip", 9000, "adjacent_total_2_3_skip_total_1_3"),
        ]:
            rows.append(
                {
                    "variant": variant,
                    "variant_family": family,
                    "target": target,
                    "training_steps_total": steps,
                    "sampling": sampling,
                    "mmd_rbf_mean": 0.1,
                    "mmd_rbf_std": 0.01,
                    "sliced_w2_mean": 0.2,
                    "sliced_w2_std": 0.02,
                    "centroid_l2_mean": 0.3,
                    "centroid_l2_std": 0.03,
                    "n_seeds": 3,
                }
            )
    return pd.DataFrame(rows)


def test_section51_tutorial_helpers_are_generic_and_reusable(tmp_path):
    from src import timecourse_reporting as tutorial

    for name in [
        "Section51Config",
        "resolve_project_root",
        "make_section51_config",
        "ensure_ch05_dirs",
        "json_ready",
        "save_json",
        "save_csv",
        "build_main_suite_figure",
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

    figure_path = tutorial.build_main_suite_figure(_minimal_summary_frame(), tmp_path)
    assert figure_path.name == "fig_5_1_main_suite.png"
    assert figure_path.exists()


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
        "## 14. Final audit",
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

    assert "display_png(figure_path)" in text
    assert "from src.timecourse_figures import" in text
    assert "_fig5_1_metric_table" not in text
    assert "_fig5_1_draw_metric_bars" not in text
    assert "_fig5_1_crop_white_margin" not in text
    assert len(code_lengths) >= 18
    assert max(code_lengths) <= 90


def test_section51_builder_is_retired_and_does_not_write_notebook():
    source = BUILDER_PATH.read_text()
    assert "maintained directly" in source
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
