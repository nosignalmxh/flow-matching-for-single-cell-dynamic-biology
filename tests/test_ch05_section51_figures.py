from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics_frame() -> pd.DataFrame:
    rows = []
    for seed in [42, 43, 44]:
        for variant in [
            "pairwise_local_bridges_6000",
            "shared_adjacent_only_6000",
            "shared_skip_adj2_skip1_9000",
        ]:
            for target in ["hidden_t2", "seen_t4"]:
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "target": target,
                        "mmd_rbf": 0.08 + 0.001 * (seed - 42),
                        "sliced_w2": 0.30 + 0.01 * (seed - 42),
                        "centroid_l2": 0.70 + 0.02 * (seed - 42),
                    }
                )
    return pd.DataFrame(rows)


def _diagnostics_frame() -> pd.DataFrame:
    rows = []
    for seed in [42, 43, 44]:
        for variant in [
            "pairwise_local_bridges_6000",
            "shared_adjacent_only_6000",
            "shared_skip_adj2_skip1_9000",
        ]:
            for boundary in ["t1", "t3"]:
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "boundary": boundary,
                        "velocity_jump_mean_l2": 4.0 + 0.1 * (seed - 42),
                    }
                )
    return pd.DataFrame(rows)


def test_fig5_1_panel_tables_preserve_real_sources_and_sem():
    from src.timecourse_figures import build_fig5_1_panel_tables

    tables = build_fig5_1_panel_tables(_metrics_frame(), _diagnostics_frame())

    assert tables.missing_entries == []
    expected_metric_columns = {
        "variant",
        "method",
        "tick_label",
        "target",
        "metric",
        "mean",
        "sd",
        "sem",
        "n_seeds",
        "source_dataframe",
        "source_table",
        "error_bar",
    }
    expected_diag_columns = expected_metric_columns - {"tick_label", "target"} | {"boundary"}
    assert tables.hidden_t2_values.shape[0] == 6
    assert tables.seen_t4_values.shape[0] == 6
    assert tables.velocity_jump_values.shape[0] == 6
    assert set(tables.hidden_t2_values.columns) == expected_metric_columns
    assert set(tables.seen_t4_values.columns) == expected_metric_columns
    assert set(tables.velocity_jump_values.columns) == expected_diag_columns
    assert set(tables.panel_sources["source_table"]) == {
        "adjacent_pairs and skip_pairs defined in this notebook; figure shows three main-text methods",
        "tab_5_1_main_suite.csv / section51_metrics",
        "tab_5_1_main_suite_diag.csv / section51_diag",
    }

    observed = tables.hidden_t2_values.set_index(["variant", "metric"]).loc[
        ("pairwise_local_bridges_6000", "mmd_rbf"),
        "sem",
    ]
    expected = np.std([0.08, 0.081, 0.082], ddof=1) / np.sqrt(3)
    assert observed == expected


def test_fig5_1_panel_drawing_and_registration_are_reusable(tmp_path):
    from src.timecourse_figures import (
        build_fig5_1_panel_tables,
        draw_fig5_1_panels,
        draw_fig5_1_hidden_t2_panel,
        register_fig5_1_artifacts,
    )

    tables = build_fig5_1_panel_tables(_metrics_frame(), _diagnostics_frame())
    displayed = []
    paths = draw_fig5_1_panels(
        tables,
        adjacent_pairs=[("0", "1"), ("1", "3"), ("3", "4")],
        skip_pairs=[("0", "1"), ("1", "3"), ("3", "4"), ("0", "3"), ("1", "4"), ("0", "4")],
        fig_dir=tmp_path / "figures" / "ch05",
        display_fn=lambda path, width=None: displayed.append((Path(path).name, width)),
    )

    assert len(paths) == 10
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert len(displayed) == 5
    assert "fig5_1_combined_png" in paths

    hidden_only = draw_fig5_1_hidden_t2_panel(
        tables.hidden_t2_values,
        fig_dir=tmp_path / "single_panel",
        display_fn=None,
    )
    assert sorted(hidden_only) == ["fig5_1_hidden_t2_recovery_pdf", "fig5_1_hidden_t2_recovery_png"]

    run_summary_path = tmp_path / "outputs" / "ch05" / "run_summary.json"
    path_table = register_fig5_1_artifacts(
        paths,
        panel_sources=tables.panel_sources,
        missing_entries=tables.missing_entries,
        run_summary_path=run_summary_path,
        project_root=tmp_path,
    )

    assert path_table.shape == (10, 3)
    summary = json.loads(run_summary_path.read_text())
    section = summary["section_5_1_main_suite"]
    assert section["figure_5_1_redrawn_missing_entries"] == []
    assert "figures/ch05/fig5_1_combined.png" in summary["expected_artifacts"]
