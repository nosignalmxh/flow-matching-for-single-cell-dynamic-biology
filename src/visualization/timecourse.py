from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..artifacts import json_ready


FIG5_1_STEMS = {
    "time_pair_designs": "fig5_1_time_pair_designs",
    "hidden_t2_recovery": "fig5_1_hidden_t2_recovery",
    "seen_t4_rollout": "fig5_1_seen_t4_rollout",
    "velocity_jump": "fig5_1_velocity_jump",
}

FIG5_1_METHODS = [
    "pairwise_local_bridges_6000",
    "shared_adjacent_only_6000",
    "shared_skip_adj2_skip1_9000",
]
FIG5_1_METHOD_LABELS = {
    "pairwise_local_bridges_6000": "Pairwise local bridge",
    "shared_adjacent_only_6000": "Shared adjacent-only field",
    "shared_skip_adj2_skip1_9000": "Shared adjacent+skip field",
}
FIG5_1_TICK_LABELS = {
    "pairwise_local_bridges_6000": "Pairwise\nlocal",
    "shared_adjacent_only_6000": "Shared\nadjacent-only",
    "shared_skip_adj2_skip1_9000": "Shared\nadjacent+skip",
}
FIG5_1_COLORS = {
    "pairwise_local_bridges_6000": "#B56B67",
    "shared_adjacent_only_6000": "#6A9BC8",
    "shared_skip_adj2_skip1_9000": "#8C93D8",
}
FIG5_1_PAIRWISE_SEGMENT_COLORS = ["#B56B67", "#C88A62", "#9B7AA5"]
FIG5_1_HIDDEN_COLOR = "#D88A82"
FIG5_1_GRID_COLOR = "#E8E8E8"
FIG5_1_TEXT_GREY = "#666666"
FIG5_1_BAR_ALPHA = 0.84


@dataclass(frozen=True)
class Fig51PanelTables:
    hidden_t2_values: pd.DataFrame
    seen_t4_values: pd.DataFrame
    velocity_jump_values: pd.DataFrame
    panel_sources: pd.DataFrame
    missing_entries: list[str]


DisplayFn = Callable[[str | Path, int | None], object]


def configure_fig5_1_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def build_fig5_1_panel_tables(section51_metrics: pd.DataFrame, section51_diag: pd.DataFrame) -> Fig51PanelTables:
    missing_entries: list[str] = []
    hidden_t2_values = _metric_table(
        section51_metrics,
        "hidden_t2",
        ["mmd_rbf", "sliced_w2"],
        missing_entries,
    )
    seen_t4_values = _metric_table(
        section51_metrics,
        "seen_t4",
        ["sliced_w2", "centroid_l2"],
        missing_entries,
    )
    velocity_jump_values = _diag_table(section51_diag, missing_entries)
    _assert_complete(hidden_t2_values, seen_t4_values, velocity_jump_values, missing_entries)

    panel_sources = pd.DataFrame(
        [
            {
                "panel": "Training time-pair designs",
                "source_dataframe": "panel schematic",
                "source_table": (
                    "adjacent_pairs and skip_pairs defined in this notebook; "
                    "figure shows three main-text methods"
                ),
            },
            {
                "panel": "Hidden t=2 recovery",
                "source_dataframe": "fig5_1_hidden_t2_values",
                "source_table": "tab_5_1_main_suite.csv / section51_metrics",
            },
            {
                "panel": "Seen t=4 rollout",
                "source_dataframe": "fig5_1_seen_t4_values",
                "source_table": "tab_5_1_main_suite.csv / section51_metrics",
            },
            {
                "panel": "Velocity jump at hand-offs",
                "source_dataframe": "fig5_1_velocity_jump_values",
                "source_table": "tab_5_1_main_suite_diag.csv / section51_diag",
            },
        ]
    )
    return Fig51PanelTables(
        hidden_t2_values=hidden_t2_values,
        seen_t4_values=seen_t4_values,
        velocity_jump_values=velocity_jump_values,
        panel_sources=panel_sources,
        missing_entries=missing_entries,
    )


def draw_fig5_1_panels(
    tables: Fig51PanelTables,
    *,
    adjacent_pairs: list[tuple[str, str]],
    skip_pairs: list[tuple[str, str]],
    fig_dir: str | Path,
    display_fn: DisplayFn | None = None,
) -> dict[str, Path]:
    configure_fig5_1_style()
    fig_dir = Path(fig_dir)
    paths: dict[str, Path] = {}

    paths.update(
        draw_fig5_1_time_pair_designs(
            adjacent_pairs=adjacent_pairs,
            skip_pairs=skip_pairs,
            fig_dir=fig_dir,
            display_fn=display_fn,
        )
    )
    paths.update(
        draw_fig5_1_hidden_t2_panel(tables.hidden_t2_values, fig_dir=fig_dir, display_fn=display_fn)
    )
    paths.update(
        draw_fig5_1_seen_t4_panel(tables.seen_t4_values, fig_dir=fig_dir, display_fn=display_fn)
    )
    paths.update(draw_fig5_1_velocity_jump_panel(tables.velocity_jump_values, fig_dir=fig_dir, display_fn=display_fn))
    return paths


def draw_fig5_1_time_pair_designs(
    *,
    adjacent_pairs: list[tuple[str, str]],
    skip_pairs: list[tuple[str, str]],
    fig_dir: str | Path,
    display_fn: DisplayFn | None = None,
) -> dict[str, Path]:
    configure_fig5_1_style()
    return _draw_time_pair_designs(
        adjacent_pairs=adjacent_pairs,
        skip_pairs=skip_pairs,
        fig_dir=Path(fig_dir),
        display_fn=display_fn,
    )


def draw_fig5_1_hidden_t2_panel(
    values: pd.DataFrame,
    *,
    fig_dir: str | Path,
    display_fn: DisplayFn | None = None,
) -> dict[str, Path]:
    configure_fig5_1_style()
    return _draw_metric_bars(
        values,
        ["mmd_rbf", "sliced_w2"],
        "Hidden t=2 recovery",
        FIG5_1_STEMS["hidden_t2_recovery"],
        Path(fig_dir),
        display_fn=display_fn,
        width=820,
    )


def draw_fig5_1_seen_t4_panel(
    values: pd.DataFrame,
    *,
    fig_dir: str | Path,
    display_fn: DisplayFn | None = None,
) -> dict[str, Path]:
    configure_fig5_1_style()
    return _draw_metric_bars(
        values,
        ["sliced_w2", "centroid_l2"],
        "Seen t=4 rollout",
        FIG5_1_STEMS["seen_t4_rollout"],
        Path(fig_dir),
        display_fn=display_fn,
        width=820,
    )


def draw_fig5_1_velocity_jump_panel(
    values: pd.DataFrame,
    *,
    fig_dir: str | Path,
    display_fn: DisplayFn | None = None,
) -> dict[str, Path]:
    configure_fig5_1_style()
    return _draw_velocity_jump(values, Path(fig_dir), display_fn=display_fn)


def register_fig5_1_artifacts(
    fig5_1_paths: dict[str, Path],
    *,
    panel_sources: pd.DataFrame,
    missing_entries: list[str],
    run_summary_path: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    project_root = Path(project_root)
    run_summary_path = Path(run_summary_path)
    run_summary = json.loads(run_summary_path.read_text()) if run_summary_path.exists() else {}
    section_record = run_summary.setdefault("section_5_1_main_suite", {})
    section_record["figure_5_1_redrawn_panels"] = {
        key: str(path.relative_to(project_root)) for key, path in fig5_1_paths.items()
    }
    section_record["figure_5_1_redrawn_panel_sources"] = panel_sources.to_dict(orient="records")
    section_record["figure_5_1_redrawn_error_bars"] = (
        "Quantitative panels use mean +/- SEM across seed-level rows from "
        "tab_5_1_main_suite.csv and tab_5_1_main_suite_diag.csv."
    )
    section_record["figure_5_1_redrawn_missing_entries"] = missing_entries

    run_summary.setdefault("expected_artifacts", [])
    obsolete_artifacts = {
        "figures/ch05/fig_5_1_main_suite.png",
        "figures/ch05/fig_5_1_main_suite_preview.png",
        "figures/ch05/fig_5_1_main_suite_preview.pdf",
        "figures/ch05/fig_5_1_A_time_pair_designs.png",
        "figures/ch05/fig_5_1_A_time_pair_designs.pdf",
        "figures/ch05/fig_5_1_B_hidden_t2_recovery.png",
        "figures/ch05/fig_5_1_B_hidden_t2_recovery.pdf",
        "figures/ch05/fig_5_1_C_seen_t4_rollout.png",
        "figures/ch05/fig_5_1_C_seen_t4_rollout.pdf",
        "figures/ch05/fig_5_1_D_velocity_jump.png",
        "figures/ch05/fig_5_1_D_velocity_jump.pdf",
        "figures/ch05/fig5_1_time_pair_designs.pdf",
        "figures/ch05/fig5_1_hidden_t2_recovery.pdf",
        "figures/ch05/fig5_1_seen_t4_rollout.pdf",
        "figures/ch05/fig5_1_velocity_jump.pdf",
        "figures/ch05/fig5_1_combined.png",
        "figures/ch05/fig5_1_combined.pdf",
    }
    run_summary["expected_artifacts"] = [
        rel for rel in run_summary["expected_artifacts"] if rel not in obsolete_artifacts
    ]
    for path in fig5_1_paths.values():
        rel = str(path.relative_to(project_root))
        if rel not in run_summary["expected_artifacts"]:
            run_summary["expected_artifacts"].append(rel)
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.write_text(json.dumps(json_ready(run_summary), indent=2, sort_keys=True), encoding="utf-8")

    return pd.DataFrame(
        [
            {
                "artifact": key,
                "relative_path": str(path.relative_to(project_root)),
                "absolute_path": str(path),
            }
            for key, path in fig5_1_paths.items()
        ]
    ).sort_values("artifact")


def _save_display(
    fig,
    fig_dir: Path,
    stem: str,
    *,
    width: int,
    display_fn: DisplayFn | None,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    png_path = fig_dir / f"{stem}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if display_fn is not None:
        display_fn(png_path, width)
    return {f"{stem}_png": png_path}


def _metric_table(
    raw: pd.DataFrame,
    target: str,
    metrics: list[str],
    missing_entries: list[str],
) -> pd.DataFrame:
    rows = []
    for variant in FIG5_1_METHODS:
        target_rows = raw[raw["variant"].eq(variant) & raw["target"].eq(target)]
        if target_rows.empty:
            missing_entries.append(f"missing endpoint rows: {variant}|{target}")
            continue
        for metric in metrics:
            values = target_rows[metric].astype(float).to_numpy()
            n = int(np.sum(~np.isnan(values)))
            if n == 0:
                missing_entries.append(f"missing metric values: {variant}|{target}|{metric}")
                continue
            sd = float(np.nanstd(values, ddof=1)) if n > 1 else np.nan
            sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
            rows.append(
                {
                    "variant": variant,
                    "method": FIG5_1_METHOD_LABELS[variant],
                    "tick_label": FIG5_1_TICK_LABELS[variant],
                    "target": target,
                    "metric": metric,
                    "mean": float(np.nanmean(values)),
                    "sd": sd,
                    "sem": sem,
                    "n_seeds": n,
                    "source_dataframe": "section51_metrics",
                    "source_table": "tab_5_1_main_suite.csv",
                    "error_bar": "SEM across seeds" if n > 1 else "none; single seed",
                }
            )
    return pd.DataFrame(rows)


def _diag_table(raw: pd.DataFrame, missing_entries: list[str]) -> pd.DataFrame:
    rows = []
    for boundary in ["t1", "t3"]:
        for variant in FIG5_1_METHODS:
            target_rows = raw[raw["variant"].eq(variant) & raw["boundary"].eq(boundary)]
            if target_rows.empty:
                missing_entries.append(f"missing diagnostic rows: {variant}|{boundary}")
                continue
            values = target_rows["velocity_jump_mean_l2"].astype(float).to_numpy()
            n = int(np.sum(~np.isnan(values)))
            sd = float(np.nanstd(values, ddof=1)) if n > 1 else np.nan
            sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
            rows.append(
                {
                    "variant": variant,
                    "method": FIG5_1_METHOD_LABELS[variant],
                    "boundary": boundary,
                    "metric": "velocity_jump_mean_l2",
                    "mean": float(np.nanmean(values)),
                    "sd": sd,
                    "sem": sem,
                    "n_seeds": n,
                    "source_dataframe": "section51_diag",
                    "source_table": "tab_5_1_main_suite_diag.csv",
                    "error_bar": "SEM across seeds" if n > 1 else "none; single seed",
                }
            )
    return pd.DataFrame(rows)


def _metric_label(metric: str) -> str:
    return {
        "mmd_rbf": "MMD",
        "sliced_w2": "Sliced W2",
        "centroid_l2": "Centroid L2",
        "velocity_jump_mean_l2": "Velocity jump",
    }[metric]


def _value_label(metric: str, value: float) -> str:
    if metric == "mmd_rbf":
        return f"{value:.4f}"
    if metric in {"sliced_w2", "centroid_l2"}:
        return f"{value:.3f}"
    if metric == "velocity_jump_mean_l2":
        return f"{value:.2f}"
    return f"{value:.3g}"


def _style_axis(ax, ylabel: str, show_lower_note: bool = True) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=FIG5_1_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_lower_note:
        ax.text(
            0.98,
            0.94,
            "lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#9A9A9A",
        )


def _draw_metric_bars(
    data: pd.DataFrame,
    metrics: list[str],
    title: str,
    stem: str,
    fig_dir: Path,
    *,
    display_fn: DisplayFn | None,
    width: int,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 3.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        metric_df = data[data["metric"].eq(metric)].set_index("variant").reindex(FIG5_1_METHODS)
        x = np.arange(len(FIG5_1_METHODS))
        means = metric_df["mean"].to_numpy(dtype=float)
        sems = metric_df["sem"].to_numpy(dtype=float)
        yerr = np.where(np.isnan(sems), 0.0, sems)
        bars = ax.bar(
            x,
            means,
            width=0.68,
            color=[FIG5_1_COLORS[v] for v in FIG5_1_METHODS],
            alpha=FIG5_1_BAR_ALPHA,
            yerr=yerr,
            capsize=2.5,
            error_kw={"linewidth": 0.8, "capthick": 0.8, "ecolor": "#555555"},
        )
        ymax = max(float(np.nanmax(means + yerr)), 1e-9)
        ax.set_ylim(0, ymax * 1.24)
        ax.set_title(_metric_label(metric), pad=8)
        _style_axis(ax, _metric_label(metric))
        ax.set_xticks(x, [FIG5_1_TICK_LABELS[v] for v in FIG5_1_METHODS])
        for bar, mean, err in zip(bars, means, yerr):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + err + ymax * 0.035,
                _value_label(metric, mean),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#444444",
            )
    fig.suptitle(title, x=0.01, ha="left", y=1.05, fontsize=13, fontweight="bold")
    fig.text(0.01, -0.03, "Bars show mean; error bars show SEM across 3 seeds.", fontsize=8.5, color="#777777")
    return _save_display(fig, fig_dir, stem, width=width, display_fn=display_fn)


def _draw_arc(ax, start, end, y, color, lw, linestyle="-", rad=0.11, alpha=1.0) -> None:
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            (start, y),
            (end, y),
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            mutation_scale=1,
            zorder=1,
        )
    )


def _draw_time_pair_designs(
    *,
    adjacent_pairs: list[tuple[str, str]],
    skip_pairs: list[tuple[str, str]],
    fig_dir: Path,
    display_fn: DisplayFn | None,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    time_points = [0, 1, 2, 3, 4]
    y_by_variant = dict(zip(FIG5_1_METHODS, [2.0, 1.0, 0.0]))
    skip_only_pairs = [pair for pair in skip_pairs if pair not in adjacent_pairs]

    ax.axvline(2, color=FIG5_1_HIDDEN_COLOR, linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.45, zorder=0)
    ax.text(2, 2.30, "hidden t=2\nheld out", ha="center", va="bottom", fontsize=9.5, color="#B75F59", alpha=0.92)

    for variant in FIG5_1_METHODS:
        y = y_by_variant[variant]
        color = FIG5_1_COLORS[variant]
        for t in time_points:
            edge_color = "#C85F57" if t == 2 else color
            lw = 1.7 if t == 2 else 1.4
            ax.scatter(t, y, s=58, facecolors="white", edgecolors=edge_color, linewidths=lw, zorder=4)
        ax.text(-0.20, y, FIG5_1_METHOD_LABELS[variant], ha="right", va="center", fontsize=10.5, color=color)

    for color, pair in zip(FIG5_1_PAIRWISE_SEGMENT_COLORS, adjacent_pairs):
        _draw_arc(
            ax,
            int(pair[0]),
            int(pair[1]),
            y_by_variant["pairwise_local_bridges_6000"],
            color,
            2.2,
            rad=0.10,
            alpha=0.88,
        )
    ax.text(
        4.35,
        y_by_variant["pairwise_local_bridges_6000"],
        "independent\nlocal models",
        va="center",
        fontsize=9.5,
        color=FIG5_1_TEXT_GREY,
    )

    for pair in adjacent_pairs:
        _draw_arc(
            ax,
            int(pair[0]),
            int(pair[1]),
            y_by_variant["shared_adjacent_only_6000"],
            FIG5_1_COLORS["shared_adjacent_only_6000"],
            2.4,
            rad=0.10,
            alpha=0.86,
        )
    ax.text(
        4.35,
        y_by_variant["shared_adjacent_only_6000"],
        "one shared\n$v_\\theta(x,t)$",
        va="center",
        fontsize=9.5,
        color=FIG5_1_TEXT_GREY,
    )

    for pair in adjacent_pairs:
        _draw_arc(
            ax,
            int(pair[0]),
            int(pair[1]),
            y_by_variant["shared_skip_adj2_skip1_9000"],
            FIG5_1_COLORS["shared_skip_adj2_skip1_9000"],
            2.8,
            rad=0.10,
            alpha=0.84,
        )
    for idx, pair in enumerate(skip_only_pairs):
        _draw_arc(
            ax,
            int(pair[0]),
            int(pair[1]),
            y_by_variant["shared_skip_adj2_skip1_9000"],
            FIG5_1_COLORS["shared_skip_adj2_skip1_9000"],
            1.15,
            linestyle="--",
            rad=-0.16 - 0.04 * idx,
            alpha=0.70,
        )
    ax.text(
        4.35,
        y_by_variant["shared_skip_adj2_skip1_9000"],
        "adjacent:skip\n= 2:1",
        va="center",
        fontsize=9.5,
        color=FIG5_1_TEXT_GREY,
    )

    ax.set_title("Training time-pair designs", loc="left", pad=10, fontsize=13, fontweight="bold")
    ax.set_xticks(time_points, [f"t={t}" for t in time_points])
    ax.set_yticks([])
    ax.set_xlabel("Time point")
    ax.set_xlim(-1.25, 4.95)
    ax.set_ylim(-0.55, 2.65)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return _save_display(
        fig,
        fig_dir,
        FIG5_1_STEMS["time_pair_designs"],
        width=820,
        display_fn=display_fn,
    )


def _draw_velocity_jump(
    values: pd.DataFrame,
    fig_dir: Path,
    *,
    display_fn: DisplayFn | None,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    x = np.arange(2)
    width = 0.22
    offsets = np.linspace(-width, width, len(FIG5_1_METHODS))

    for offset, variant in zip(offsets, FIG5_1_METHODS):
        rows = values[values["variant"].eq(variant)].set_index("boundary").reindex(["t1", "t3"])
        means = rows["mean"].to_numpy(dtype=float)
        sems = rows["sem"].to_numpy(dtype=float)
        yerr = np.where(np.isnan(sems), 0.0, sems)
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            color=FIG5_1_COLORS[variant],
            alpha=FIG5_1_BAR_ALPHA,
            label=FIG5_1_METHOD_LABELS[variant],
            yerr=yerr,
            capsize=2.4,
            error_kw={"linewidth": 0.8, "capthick": 0.8, "ecolor": "#555555"},
        )
        for bar, mean, err in zip(bars, means, yerr):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + err + 0.18,
                _value_label("velocity_jump_mean_l2", mean),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#444444",
            )

    ax.set_title("Velocity jump at hand-offs", loc="left", pad=10, fontsize=13, fontweight="bold")
    _style_axis(ax, "Velocity jump")
    ax.set_xticks(x, ["t=1", "t=3"])
    ymax = float(np.nanmax(values["mean"] + values["sem"].fillna(0)))
    ax.set_ylim(0, ymax * 1.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.text(0.01, -0.03, "Bars show mean; error bars show SEM across 3 seeds.", fontsize=8.5, color="#777777")
    return _save_display(fig, fig_dir, FIG5_1_STEMS["velocity_jump"], width=820, display_fn=display_fn)


def _assert_complete(
    hidden_t2_values: pd.DataFrame,
    seen_t4_values: pd.DataFrame,
    velocity_jump_values: pd.DataFrame,
    missing_entries: list[str],
) -> None:
    expected_numeric = len(FIG5_1_METHODS) * (2 + 2 + 2)
    observed_numeric = len(hidden_t2_values) + len(seen_t4_values) + len(velocity_jump_values)
    if observed_numeric != expected_numeric:
        missing_entries.append(f"expected {expected_numeric} quantitative rows, observed {observed_numeric}")
    if missing_entries:
        raise RuntimeError("Figure 5.1 redraw missing real data: " + "; ".join(missing_entries))
