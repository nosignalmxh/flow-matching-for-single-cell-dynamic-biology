from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.timecourse import (
    _summarize_eb_skip_pair_ablation,
    load_eb_ch05,
    run_eb_skip_pair_ablation,
    set_global_seed,
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


def build_skip_pair_ablation_figure(eb_skip_metrics: pd.DataFrame, fig_dir: Path) -> Path:
    variant_order = [
        "shared_adjacent_only_6000",
        "shared_adjacent_only_9000",
        "shared_adjacent_only_12000",
        "shared_skip_uniform_6000",
        "shared_skip_uniform_12000",
        "shared_skip_adj2_skip1_9000",
        "shared_skip_adj3_skip1_8000",
        "shared_skip_medium_only_9000",
    ]
    variant_labels = [
        "adj 6k",
        "adj 9k",
        "adj 12k",
        "skip uni 6k",
        "skip uni 12k",
        "skip 2:1 9k",
        "skip 3:1 8k",
        "skip med 9k",
    ]
    family_by_variant = (
        eb_skip_metrics[["variant", "variant_family"]]
        .drop_duplicates()
        .set_index("variant")["variant_family"]
        .to_dict()
    )
    family_colors = {"adjacent_only": "#54A24B", "skip": "#F58518"}
    ablation_plot = (
        eb_skip_metrics.groupby(["variant", "variant_family", "target"], observed=False)[
            ["mmd_rbf", "sliced_w2", "centroid_l2"]
        ]
        .mean()
        .reset_index()
    )
    panel_specs = [
        ("seen_t4", "sliced_w2", "seen_t4 Sliced W2"),
        ("seen_t4", "centroid_l2", "seen_t4 Centroid L2"),
        ("hidden_t2", "mmd_rbf", "hidden_t2 MMD RBF"),
        ("hidden_t2", "sliced_w2", "hidden_t2 Sliced W2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8), sharex=True)
    for ax, (target, metric, title) in zip(axes.ravel(), panel_specs):
        panel = (
            ablation_plot[ablation_plot["target"].eq(target)]
            .set_index("variant")
            .reindex(variant_order)
            .reset_index()
        )
        values = panel[metric].to_numpy()
        colors = [family_colors[family_by_variant[v]] for v in variant_order]
        bars = ax.bar(np.arange(len(variant_order)), values, color=colors, alpha=0.88)
        for family in ["adjacent_only", "skip"]:
            family_mask = panel["variant_family"].eq(family)
            if family_mask.any():
                best_local_idx = int(panel.loc[family_mask, metric].astype(float).idxmin())
                bars[best_local_idx].set_edgecolor("black")
                bars[best_local_idx].set_linewidth(1.8)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xticks(np.arange(len(variant_order)))
        ax.set_xticklabels(variant_labels, rotation=30, ha="right")
    fig.suptitle("EB skip-pair ablation: rollout and hidden-time global metrics")
    return save_figure(fig, fig_dir, "fig_5_1_skip_pair_ablation.png")


def write_official_outputs(
    eb_skip_metrics: pd.DataFrame,
    eb_skip_diag: pd.DataFrame,
    summary: dict,
) -> None:
    fig_dir = PROJECT_ROOT / "figures" / "ch05"
    table_dir = PROJECT_ROOT / "tables" / "ch05"
    out_dir = PROJECT_ROOT / "outputs" / "ch05"
    for path in [fig_dir, table_dir, out_dir]:
        path.mkdir(parents=True, exist_ok=True)

    metrics_path = table_dir / "tab_5_1_skip_pair_ablation.csv"
    diag_path = table_dir / "tab_5_1_skip_pair_ablation_diag.csv"
    eb_skip_metrics.to_csv(metrics_path, index=False)
    eb_skip_diag.to_csv(diag_path, index=False)
    fig_path = build_skip_pair_ablation_figure(eb_skip_metrics, fig_dir)

    run_summary_path = out_dir / "run_summary.json"
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text())
    else:
        run_summary = {}
    run_summary["eb_skip_pair_ablation"] = summary
    run_summary.setdefault("expected_artifacts", [])
    for artifact in [metrics_path, diag_path, fig_path]:
        rel = str(artifact.relative_to(PROJECT_ROOT))
        if rel not in run_summary["expected_artifacts"]:
            run_summary["expected_artifacts"].append(rel)
    save_json(run_summary_path, run_summary)

    print("Saved:", metrics_path.relative_to(PROJECT_ROOT))
    print("Saved:", diag_path.relative_to(PROJECT_ROOT))
    print("Saved:", fig_path.relative_to(PROJECT_ROOT))
    print("Updated:", run_summary_path.relative_to(PROJECT_ROOT))


def aggregate_prefixed_outputs(prefixes: list[str]) -> None:
    table_dir = PROJECT_ROOT / "tables" / "ch05"
    metrics_frames = []
    diag_frames = []
    for prefix in prefixes:
        metrics_path = table_dir / f"tab_5_1_skip_pair_ablation_{prefix}.csv"
        diag_path = table_dir / f"tab_5_1_skip_pair_ablation_diag_{prefix}.csv"
        if not metrics_path.exists() or not diag_path.exists():
            raise FileNotFoundError(f"Missing prefixed ablation outputs for {prefix}: {metrics_path}, {diag_path}")
        metrics_frames.append(pd.read_csv(metrics_path))
        diag_frames.append(pd.read_csv(diag_path))
    eb_skip_metrics = pd.concat(metrics_frames, ignore_index=True)
    eb_skip_diag = pd.concat(diag_frames, ignore_index=True)
    seeds = sorted(int(seed) for seed in eb_skip_metrics["seed"].unique())
    summary = _summarize_eb_skip_pair_ablation(eb_skip_metrics, eb_skip_diag, seeds=seeds)
    write_official_outputs(eb_skip_metrics, eb_skip_diag, summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


def main() -> None:
    import torch

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_ch05")
    aggregate_prefixes = [
        part.strip()
        for part in os.environ.get("CH05_EB_SKIP_AGGREGATE_PREFIXES", "").split(",")
        if part.strip()
    ]
    if aggregate_prefixes:
        aggregate_prefixed_outputs(aggregate_prefixes)
        return

    default_seed = int(os.environ.get("CH05_SEED", "42"))
    seeds = [
        int(part.strip())
        for part in os.environ.get("CH05_EB_SKIP_ABLATION_SEEDS", str(default_seed)).split(",")
        if part.strip()
    ]
    batch_size = int(os.environ.get("CH05_BATCH_SIZE", "256"))
    nfe = int(os.environ.get("CH05_NFE", "32"))
    eb_max_cells_per_time = int(os.environ.get("CH05_EB_MAX_CELLS_PER_TIME", "900"))
    device = torch.device(os.environ.get("CH05_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    fig_dir = PROJECT_ROOT / "figures" / "ch05"
    table_dir = PROJECT_ROOT / "tables" / "ch05"
    out_dir = PROJECT_ROOT / "outputs" / "ch05"
    for path in [fig_dir, table_dir, out_dir]:
        path.mkdir(parents=True, exist_ok=True)

    set_global_seed(default_seed)
    eb = load_eb_ch05(
        PROJECT_ROOT / "data" / "trajectorynet_eb" / "eb_velocity_v5.npz",
        max_cells_per_time=eb_max_cells_per_time,
        seed=default_seed,
        n_pc=20,
    )
    eb_skip_metrics, eb_skip_diag, eb_skip_cache = run_eb_skip_pair_ablation(
        eb,
        batch_size=batch_size,
        nfe=nfe,
        seed=default_seed,
        seeds=seeds,
        device=device,
    )
    output_prefix = os.environ.get("CH05_EB_SKIP_OUTPUT_PREFIX", "").strip()
    if output_prefix:
        metrics_path = table_dir / f"tab_5_1_skip_pair_ablation_{output_prefix}.csv"
        diag_path = table_dir / f"tab_5_1_skip_pair_ablation_diag_{output_prefix}.csv"
        summary_path = out_dir / f"run_summary_skip_pair_ablation_{output_prefix}.json"
        eb_skip_metrics.to_csv(metrics_path, index=False)
        eb_skip_diag.to_csv(diag_path, index=False)
        save_json(summary_path, eb_skip_cache["decision_summary"])
        print("Saved:", metrics_path.relative_to(PROJECT_ROOT))
        print("Saved:", diag_path.relative_to(PROJECT_ROOT))
        print("Saved:", summary_path.relative_to(PROJECT_ROOT))
        print(json.dumps(json_ready(eb_skip_cache["decision_summary"]), indent=2, sort_keys=True))
        return

    metrics_path = table_dir / "tab_5_1_skip_pair_ablation.csv"
    diag_path = table_dir / "tab_5_1_skip_pair_ablation_diag.csv"
    eb_skip_metrics.to_csv(metrics_path, index=False)
    eb_skip_diag.to_csv(diag_path, index=False)
    write_official_outputs(eb_skip_metrics, eb_skip_diag, eb_skip_cache["decision_summary"])
    print(json.dumps(json_ready(eb_skip_cache["decision_summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
