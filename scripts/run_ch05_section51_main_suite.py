from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_ch05")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.timecourse import (
    build_ch05_section51_main_text_results,
    load_eb_ch05,
    run_eb_section51_main_suite,
    set_global_seed,
    summarize_eb_section51_main_suite,
)


def json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def save_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    return path


def save_figure(fig, fig_dir: Path, filename: str) -> Path:
    path = fig_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_main_suite_figure(summary: pd.DataFrame, fig_dir: Path) -> Path:
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
    return save_figure(fig, fig_dir, "fig_5_1_main_suite.png")


def _section51_summary_payload(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    summary, diag_summary = summarize_eb_section51_main_suite(metrics, diagnostics)
    main_text, payload = build_ch05_section51_main_text_results(summary, diag_summary)
    return summary, diag_summary, main_text, payload


def write_official_outputs(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    fig_dir = PROJECT_ROOT / "figures" / "ch05"
    table_dir = PROJECT_ROOT / "tables" / "ch05"
    out_dir = PROJECT_ROOT / "outputs" / "ch05"
    for path in [fig_dir, table_dir, out_dir]:
        path.mkdir(parents=True, exist_ok=True)

    summary, diag_summary, main_text, payload = _section51_summary_payload(metrics, diagnostics)

    metrics_path = table_dir / "tab_5_1_main_suite.csv"
    summary_path = table_dir / "tab_5_1_main_suite_summary.csv"
    diag_path = table_dir / "tab_5_1_main_suite_diag.csv"
    diag_summary_path = table_dir / "tab_5_1_main_suite_diag_summary.csv"
    main_text_path = table_dir / "tab_5_1_main_text_results.csv"
    fig_path = build_main_suite_figure(summary, fig_dir)

    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    diagnostics.to_csv(diag_path, index=False)
    diag_summary.to_csv(diag_summary_path, index=False)
    main_text.to_csv(main_text_path, index=False)

    run_summary_path = out_dir / "run_summary.json"
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text())
    else:
        run_summary = {}
    run_summary["section_5_1_main_suite"] = {
        "metrics_table": str(metrics_path.relative_to(PROJECT_ROOT)),
        "summary_table": str(summary_path.relative_to(PROJECT_ROOT)),
        "diag_table": str(diag_path.relative_to(PROJECT_ROOT)),
        "diag_summary_table": str(diag_summary_path.relative_to(PROJECT_ROOT)),
        "main_text_results_table": str(main_text_path.relative_to(PROJECT_ROOT)),
        "figure": str(fig_path.relative_to(PROJECT_ROOT)),
        **payload,
    }
    run_summary.setdefault("expected_artifacts", [])
    for artifact in [metrics_path, summary_path, diag_path, diag_summary_path, main_text_path, fig_path]:
        rel = str(artifact.relative_to(PROJECT_ROOT))
        if rel not in run_summary["expected_artifacts"]:
            run_summary["expected_artifacts"].append(rel)
    save_json(run_summary_path, run_summary)

    for path in [metrics_path, summary_path, diag_path, diag_summary_path, main_text_path, fig_path]:
        print("Saved:", path.relative_to(PROJECT_ROOT))
    print("Updated:", run_summary_path.relative_to(PROJECT_ROOT))
    print(json.dumps(json_ready(run_summary["section_5_1_main_suite"]), indent=2, sort_keys=True))


def aggregate_prefixed_outputs(prefixes: list[str]) -> None:
    table_dir = PROJECT_ROOT / "tables" / "ch05"
    metrics_frames = []
    diag_frames = []
    for prefix in prefixes:
        metrics_path = table_dir / f"tab_5_1_main_suite_{prefix}.csv"
        diag_path = table_dir / f"tab_5_1_main_suite_diag_{prefix}.csv"
        if not metrics_path.exists() or not diag_path.exists():
            raise FileNotFoundError(f"Missing prefixed main-suite outputs for {prefix}: {metrics_path}, {diag_path}")
        metrics_frames.append(pd.read_csv(metrics_path))
        diag_frames.append(pd.read_csv(diag_path))
    write_official_outputs(
        pd.concat(metrics_frames, ignore_index=True),
        pd.concat(diag_frames, ignore_index=True),
    )


def main() -> None:
    import torch

    aggregate_prefixes = [
        part.strip()
        for part in os.environ.get("CH05_SECTION51_AGGREGATE_PREFIXES", "").split(",")
        if part.strip()
    ]
    if aggregate_prefixes:
        aggregate_prefixed_outputs(aggregate_prefixes)
        return

    default_seed = int(os.environ.get("CH05_SEED", "42"))
    seeds = [
        int(part.strip())
        for part in os.environ.get("CH05_SECTION51_MAIN_SUITE_SEEDS", str(default_seed)).split(",")
        if part.strip()
    ]
    batch_size = int(os.environ.get("CH05_BATCH_SIZE", "256"))
    nfe = int(os.environ.get("CH05_NFE", "32"))
    eb_max_cells_per_time = int(os.environ.get("CH05_EB_MAX_CELLS_PER_TIME", "900"))
    device = torch.device(os.environ.get("CH05_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    table_dir = PROJECT_ROOT / "tables" / "ch05"
    out_dir = PROJECT_ROOT / "outputs" / "ch05"
    for path in [table_dir, out_dir]:
        path.mkdir(parents=True, exist_ok=True)

    set_global_seed(default_seed)
    eb = load_eb_ch05(
        PROJECT_ROOT / "data" / "trajectorynet_eb" / "eb_velocity_v5.npz",
        max_cells_per_time=eb_max_cells_per_time,
        seed=default_seed,
        n_pc=20,
    )
    metrics, diagnostics, cache = run_eb_section51_main_suite(
        eb,
        batch_size=batch_size,
        nfe=nfe,
        seed=default_seed,
        seeds=seeds,
        device=device,
    )
    output_prefix = os.environ.get("CH05_SECTION51_OUTPUT_PREFIX", "").strip()
    if output_prefix:
        metrics_path = table_dir / f"tab_5_1_main_suite_{output_prefix}.csv"
        diag_path = table_dir / f"tab_5_1_main_suite_diag_{output_prefix}.csv"
        summary_path = out_dir / f"run_summary_section51_main_suite_{output_prefix}.json"
        metrics.to_csv(metrics_path, index=False)
        diagnostics.to_csv(diag_path, index=False)
        save_json(summary_path, cache["summary_payload"])
        print("Saved:", metrics_path.relative_to(PROJECT_ROOT))
        print("Saved:", diag_path.relative_to(PROJECT_ROOT))
        print("Saved:", summary_path.relative_to(PROJECT_ROOT))
        print(json.dumps(json_ready(cache["summary_payload"]), indent=2, sort_keys=True))
        return

    write_official_outputs(metrics, diagnostics)


if __name__ == "__main__":
    main()
