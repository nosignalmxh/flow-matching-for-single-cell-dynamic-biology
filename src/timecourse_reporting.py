from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import resolve_project_root as _resolve_project_root

from .artifacts import json_ready, save_csv, save_json


@dataclass(frozen=True)
class Section51Config:
    project_root: Path
    fig_dir: Path
    table_dir: Path
    output_dir: Path
    eb_path: Path
    default_seed: int
    suite_seeds: list[int]
    batch_size: int
    nfe: int
    eb_max_cells_per_time: int
    device: Any


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not seeds:
        raise ValueError("Seed list is empty.")
    return seeds




resolve_project_root = partial(_resolve_project_root, markers=("src/single_cell_experiments.py",))


def ensure_ch05_dirs(project_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(project_root).resolve()
    fig_dir = root / "figures" / "ch05"
    table_dir = root / "tables" / "ch05"
    output_dir = root / "outputs" / "ch05"
    for path in [fig_dir, table_dir, output_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir, output_dir


def make_section51_config(project_root: str | Path | None = None) -> Section51Config:
    import torch

    root = resolve_project_root(project_root)
    fig_dir, table_dir, output_dir = ensure_ch05_dirs(root)
    default_seed = int(os.environ.get("CH05_SEED", "42"))
    suite_seeds = parse_seed_list(os.environ.get("CH05_SECTION51_MAIN_SUITE_SEEDS", "42,43,44"))
    return Section51Config(
        project_root=root,
        fig_dir=fig_dir,
        table_dir=table_dir,
        output_dir=output_dir,
        eb_path=root / "data" / "trajectorynet_eb" / "eb_velocity_v5.npz",
        default_seed=default_seed,
        suite_seeds=suite_seeds,
        batch_size=int(os.environ.get("CH05_BATCH_SIZE", "256")),
        nfe=int(os.environ.get("CH05_NFE", "32")),
        eb_max_cells_per_time=int(os.environ.get("CH05_EB_MAX_CELLS_PER_TIME", "900")),
        device=torch.device(os.environ.get("CH05_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")),
    )


def build_main_suite_figure(summary: pd.DataFrame, fig_dir: str | Path, filename: str = "fig_5_1_main_suite.png") -> Path:
    import matplotlib.pyplot as plt

    variant_order = [
        "pairwise_local_bridges_6000",
        "shared_adjacent_only_6000",
        "shared_skip_uniform_6000",
        "shared_skip_adj2_skip1_9000",
    ]
    variant_labels = ["Pairwise", "Shared adj", "Skip uniform", "Skip 2:1"]
    metric_cols = [("mmd_rbf", "MMD RBF"), ("sliced_w2", "Sliced W2"), ("centroid_l2", "Centroid L2")]
    target_rows = [("hidden_t2", "hidden_t2"), ("seen_t4", "seen_t4")]
    colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.6), sharex=True)
    for row_idx, (target, target_label) in enumerate(target_rows):
        target_df = summary[summary["target"].eq(target)].set_index("variant").reindex(variant_order)
        for col_idx, (metric, metric_label) in enumerate(metric_cols):
            ax = axes[row_idx, col_idx]
            ax.bar(variant_labels, target_df[f"{metric}_mean"].to_numpy(), color=colors)
            ax.set_title(f"{target_label} / {metric_label}")
            ax.set_ylabel(metric_label)
            ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Section 5.1 unified EB main suite: global distribution metrics")

    path = Path(fig_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def preview_frame(frame: pd.DataFrame, columns: list[str] | None = None, n: int = 8) -> pd.DataFrame:
    view = frame.loc[:, columns].head(n) if columns is not None else frame.head(n)
    try:
        from IPython.display import display

        display(view)
    except Exception:
        pass
    return view


def display_png(path: str | Path, width: int | None = None) -> Path:
    path = Path(path)
    try:
        from IPython.display import Image, display

        kwargs = {"filename": str(path)}
        if width is not None:
            kwargs["width"] = int(width)
        display(Image(**kwargs))
    except Exception:
        pass
    return path


def write_section51_artifacts(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    diag_summary: pd.DataFrame,
    main_text_results: pd.DataFrame,
    summary_payload: dict,
    config: Section51Config,
) -> dict[str, Path]:
    metrics_path = config.table_dir / "tab_5_1_main_suite.csv"
    summary_path = config.table_dir / "tab_5_1_main_suite_summary.csv"
    diag_path = config.table_dir / "tab_5_1_main_suite_diag.csv"
    diag_summary_path = config.table_dir / "tab_5_1_main_suite_diag_summary.csv"
    main_text_path = config.table_dir / "tab_5_1_main_text_results.csv"
    figure_path = build_main_suite_figure(summary, config.fig_dir)

    for path, frame in [
        (metrics_path, metrics),
        (summary_path, summary),
        (diag_path, diagnostics),
        (diag_summary_path, diag_summary),
        (main_text_path, main_text_results),
    ]:
        save_csv(path, frame)

    run_summary_path = config.output_dir / "run_summary.json"
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text())
    else:
        run_summary = {}
    run_summary["section_5_1_main_suite"] = {
        "metrics_table": str(metrics_path.relative_to(config.project_root)),
        "summary_table": str(summary_path.relative_to(config.project_root)),
        "diag_table": str(diag_path.relative_to(config.project_root)),
        "diag_summary_table": str(diag_summary_path.relative_to(config.project_root)),
        "main_text_results_table": str(main_text_path.relative_to(config.project_root)),
        "figure": str(figure_path.relative_to(config.project_root)),
        **summary_payload,
    }

    run_summary.setdefault("expected_artifacts", [])
    for artifact in [metrics_path, summary_path, diag_path, diag_summary_path, main_text_path, figure_path]:
        rel = str(artifact.relative_to(config.project_root))
        if rel not in run_summary["expected_artifacts"]:
            run_summary["expected_artifacts"].append(rel)
    save_json(run_summary_path, run_summary)

    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "diagnostics": diag_path,
        "diagnostic_summary": diag_summary_path,
        "main_text": main_text_path,
        "figure": figure_path,
        "run_summary": run_summary_path,
    }


def audit_section51_main_text_results(audit: pd.DataFrame, notebook_source: str | None = None) -> None:
    expected_claim_parts = {
        "hidden_t2_main_comparison",
        "seen_t4_long_horizon",
        "hidden_t2_skip_tradeoff",
        "velocity_jump_diagnostic",
    }
    expected_source_tables = {
        "tab_5_1_main_suite_summary.csv",
        "tab_5_1_main_suite_diag_summary.csv",
    }
    excluded_metrics = {"cluster" + "_mass" + "_l1", "rare" + "_cluster" + "_error"}
    expected_velocity_rows = {
        ("pairwise_local_bridges_6000", "t1", "velocity_jump_mean_l2"),
        ("pairwise_local_bridges_6000", "t3", "velocity_jump_mean_l2"),
        ("shared_adjacent_only_6000", "t1", "velocity_jump_mean_l2"),
        ("shared_adjacent_only_6000", "t3", "velocity_jump_mean_l2"),
        ("shared_skip_adj2_skip1_9000", "t1", "velocity_jump_mean_l2"),
        ("shared_skip_adj2_skip1_9000", "t3", "velocity_jump_mean_l2"),
        ("shared_skip_uniform_6000", "t1", "velocity_jump_mean_l2"),
        ("shared_skip_uniform_6000", "t3", "velocity_jump_mean_l2"),
        ("pairwise_local_bridges_6000", "t1_t3_mean", "velocity_jump_mean_l2_mean_over_hand_offs"),
        ("shared_adjacent_only_6000", "t1_t3_mean", "velocity_jump_mean_l2_mean_over_hand_offs"),
        ("shared_skip_adj2_skip1_9000", "t1_t3_mean", "velocity_jump_mean_l2_mean_over_hand_offs"),
        ("shared_skip_uniform_6000", "t1_t3_mean", "velocity_jump_mean_l2_mean_over_hand_offs"),
    }
    observed_velocity_rows = set(
        audit[audit["claim_part"].eq("velocity_jump_diagnostic")]
        [["method_or_variant", "target_or_boundary", "metric"]]
        .itertuples(index=False, name=None)
    )

    assert audit.shape == (28, 12), audit.shape
    assert set(audit["claim_part"]) == expected_claim_parts
    assert set(audit["source_table"]) == expected_source_tables
    assert not set(audit["metric"]).intersection(excluded_metrics)
    assert expected_velocity_rows.issubset(observed_velocity_rows)

    if notebook_source is not None:
        old_table_names = [
            "tab_5_1_" + "multi_timepoint.csv",
            "tab_5_1_" + "skip_pair_ablation_summary.csv",
        ]
        for old_table_name in old_table_names:
            assert old_table_name not in notebook_source


def section51_expected_display_values() -> list[tuple[str, str, str, str, str]]:
    return [
        ("hidden_t2_main_comparison", "pairwise_local_bridges_6000", "hidden_t2", "mmd_rbf", "0.0836"),
        ("hidden_t2_main_comparison", "pairwise_local_bridges_6000", "hidden_t2", "sliced_w2", "0.408"),
        ("hidden_t2_main_comparison", "shared_adjacent_only_6000", "hidden_t2", "mmd_rbf", "0.0565"),
        ("hidden_t2_main_comparison", "shared_adjacent_only_6000", "hidden_t2", "sliced_w2", "0.356"),
        ("hidden_t2_main_comparison", "shared_skip_adj2_skip1_9000", "hidden_t2", "mmd_rbf", "0.0675"),
        ("hidden_t2_main_comparison", "shared_skip_adj2_skip1_9000", "hidden_t2", "sliced_w2", "0.384"),
        ("seen_t4_long_horizon", "shared_adjacent_only_6000", "seen_t4", "sliced_w2", "0.322"),
        ("seen_t4_long_horizon", "shared_adjacent_only_6000", "seen_t4", "centroid_l2", "0.904"),
        ("seen_t4_long_horizon", "shared_skip_adj2_skip1_9000", "seen_t4", "sliced_w2", "0.215"),
        ("seen_t4_long_horizon", "shared_skip_adj2_skip1_9000", "seen_t4", "centroid_l2", "0.664"),
        (
            "seen_t4_long_horizon",
            "relative_improvement_skip_2to1_vs_adjacent_only",
            "seen_t4",
            "sliced_w2_pct_lower",
            "33.1% lower",
        ),
        (
            "seen_t4_long_horizon",
            "relative_improvement_skip_2to1_vs_adjacent_only",
            "seen_t4",
            "centroid_l2_pct_lower",
            "26.6% lower",
        ),
        ("velocity_jump_diagnostic", "pairwise_local_bridges_6000", "t1", "velocity_jump_mean_l2", "8.03"),
        ("velocity_jump_diagnostic", "pairwise_local_bridges_6000", "t3", "velocity_jump_mean_l2", "8.66"),
        ("velocity_jump_diagnostic", "shared_adjacent_only_6000", "t1", "velocity_jump_mean_l2", "8.14"),
        ("velocity_jump_diagnostic", "shared_adjacent_only_6000", "t3", "velocity_jump_mean_l2", "4.81"),
        ("velocity_jump_diagnostic", "shared_skip_adj2_skip1_9000", "t1", "velocity_jump_mean_l2", "9.12"),
        ("velocity_jump_diagnostic", "shared_skip_adj2_skip1_9000", "t3", "velocity_jump_mean_l2", "4.45"),
        ("velocity_jump_diagnostic", "shared_skip_uniform_6000", "t1", "velocity_jump_mean_l2", "5.79"),
        ("velocity_jump_diagnostic", "shared_skip_uniform_6000", "t3", "velocity_jump_mean_l2", "2.26"),
        (
            "velocity_jump_diagnostic",
            "pairwise_local_bridges_6000",
            "t1_t3_mean",
            "velocity_jump_mean_l2_mean_over_hand_offs",
            "8.35",
        ),
        (
            "velocity_jump_diagnostic",
            "shared_adjacent_only_6000",
            "t1_t3_mean",
            "velocity_jump_mean_l2_mean_over_hand_offs",
            "6.47",
        ),
        (
            "velocity_jump_diagnostic",
            "shared_skip_adj2_skip1_9000",
            "t1_t3_mean",
            "velocity_jump_mean_l2_mean_over_hand_offs",
            "6.79",
        ),
        (
            "velocity_jump_diagnostic",
            "shared_skip_uniform_6000",
            "t1_t3_mean",
            "velocity_jump_mean_l2_mean_over_hand_offs",
            "4.03",
        ),
    ]


def verify_expected_display_values(
    audit: pd.DataFrame,
    expected_values: list[tuple[str, str, str, str, str]] | None = None,
) -> int:
    expected = expected_values or section51_expected_display_values()
    audit_indexed = audit.set_index(["claim_part", "method_or_variant", "target_or_boundary", "metric"])
    for claim_part, method, target_or_boundary, metric, display_value in expected:
        observed = str(audit_indexed.loc[(claim_part, method, target_or_boundary, metric), "display_value"])
        assert observed == display_value, (claim_part, method, target_or_boundary, metric, observed, display_value)
    return len(expected)
