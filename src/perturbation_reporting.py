from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from .utils import resolve_project_root as _resolve_project_root


METHOD_LABELS = {
    "M1_unconditional": "M1 unconditional FM",
    "M2_per_compound": "M2 one-flow-per-compound",
    "M3_no_chemistry": "M3 one-hot+dose conditional FM",
    "M4_chemistry_aware": "M4 RDKit2D+dose conditional FM",
    "vehicle_as_prediction": "Vehicle baseline",
    "mean_shift": "Mean-shift baseline",
    "nearest_chemistry": "Nearest-chemistry baseline",
}
METHOD_COLORS = {
    "M1_unconditional": "#4A4A4A",
    "M2_per_compound": "#C67C35",
    "M3_no_chemistry": "#4E79A7",
    "M4_chemistry_aware": "#2A9D8F",
    "vehicle_as_prediction": "#C9C9C9",
    "mean_shift": "#B8AEC9",
    "nearest_chemistry": "#7B65A7",
}
SCATTER_COLORS = {
    "vehicle": "#5F83A9",
    "truth": "#D95F59",
    "m3": "#6F63A6",
    "m4": "#2A9D8F",
}
FIGURE_TITLES = {
    "fig_5_2_model_designs": "Perturbation prediction model designs",
    "fig_5_2_evaluation_splits": "Evaluation split design",
    "fig_5_2_heldout_highest_dose_metrics": "Held-out highest dose",
    "fig_5_2_heldout_compound_metrics": "Held-out compound",
    "fig_5_2_alisertib_example": "Held-out Alisertib example",
}
SPLIT_B_METHOD_ORDER = [
    "M1_unconditional",
    "M2_per_compound",
    "M3_no_chemistry",
    "vehicle_as_prediction",
    "mean_shift",
]
SPLIT_C_METHOD_ORDER = [
    "M1_unconditional",
    "M3_no_chemistry",
    "M4_chemistry_aware",
    "vehicle_as_prediction",
    "mean_shift",
    "nearest_chemistry",
]
EXPECTED_SPLIT_B_DISPLAY = [
    {"method": "M1_unconditional", "MMD": 0.0225, "Sliced W2": 0.395},
    {"method": "M2_per_compound", "MMD": 0.0242, "Sliced W2": 0.381},
    {"method": "M3_no_chemistry", "MMD": 0.0175, "Sliced W2": 0.356},
    {"method": "vehicle_as_prediction", "MMD": 0.0219, "Sliced W2": 0.372},
    {"method": "mean_shift", "MMD": 0.0250, "Sliced W2": 0.381},
]
EXPECTED_SPLIT_C_DISPLAY = [
    {"method": "M1_unconditional", "MMD": 0.0983, "Sliced W2": 1.001},
    {"method": "M3_no_chemistry", "MMD": 0.0841, "Sliced W2": 0.965},
    {"method": "M4_chemistry_aware", "MMD": 0.0619, "Sliced W2": 0.803},
    {"method": "vehicle_as_prediction", "MMD": 0.0635, "Sliced W2": 0.785},
    {"method": "mean_shift", "MMD": 0.0777, "Sliced W2": 0.824},
    {"method": "nearest_chemistry", "MMD": 0.0867, "Sliced W2": 0.833},
]


@dataclass(frozen=True)
class Section52Config:
    project_root: Path
    data_dir: Path
    fig_dir: Path
    table_dir: Path
    output_dir: Path
    default_seed: int
    quick_mode: bool
    training_steps: int
    batch_size: int
    nfe: int
    sciplex_download_in_ch05: bool
    sciplex_synthetic_if_missing: bool
    max_eval_groups: int | None
    device: object


@dataclass(frozen=True)
class Section52MetricDisplayPackage:
    split_b_metric_table: pd.DataFrame
    split_c_metric_table: pd.DataFrame
    split_b_metric_display: pd.DataFrame
    split_c_metric_display: pd.DataFrame
    split_b_missing: list[str]
    split_c_missing: list[str]
    manuscript_metric_source: str
    missing_result_notes: list[str]




resolve_project_root = partial(_resolve_project_root, markers=("src/single_cell_experiments.py",))


def make_section52_config(project_root: str | Path | None = None, device=None) -> Section52Config:
    if device is None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_project_root(project_root)
    data_dir = root / "data"
    fig_dir = root / "figures" / "ch05" / "new2"
    table_dir = root / "tables" / "ch05"
    output_dir = root / "outputs" / "ch05"
    for output_path in [fig_dir, table_dir, output_dir]:
        output_path.mkdir(parents=True, exist_ok=True)
    max_eval_groups = os.environ.get("CH05_MAX_EVAL_GROUPS", "")
    return Section52Config(
        project_root=root,
        data_dir=data_dir,
        fig_dir=fig_dir,
        table_dir=table_dir,
        output_dir=output_dir,
        default_seed=int(os.environ.get("CH05_SEED", "42")),
        quick_mode=os.environ.get("CH05_QUICK", "0") == "1",
        training_steps=int(os.environ.get("CH05_TRAINING_STEPS", "6000")),
        batch_size=int(os.environ.get("CH05_BATCH_SIZE", "256")),
        nfe=int(os.environ.get("CH05_NFE", "32")),
        sciplex_download_in_ch05=os.environ.get("CH05_SCIPLEX_DOWNLOAD_IN_CH05", "0") == "1",
        sciplex_synthetic_if_missing=os.environ.get("CH05_ALLOW_SYNTHETIC_SCIPLEX", "0") == "1",
        max_eval_groups=None if max_eval_groups == "" else int(max_eval_groups),
        device=device,
    )


def make_section52_run_summary(config: Section52Config) -> dict:
    return {
        "experiment": "sci-Plex perturbation response prediction",
        "scope": "Split B held-out highest dose and Split C held-out compound only",
        "quick_mode": bool(config.quick_mode),
        "seed": int(config.default_seed),
        "device": str(config.device),
        "training_steps": int(config.training_steps),
        "batch_size": int(config.batch_size),
        "nfe": int(config.nfe),
        "paths": {
            "figures": str(config.fig_dir),
            "tables": str(config.table_dir),
            "outputs": str(config.output_dir),
        },
    }


def apply_section52_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "grid.linewidth": 0.45,
            "grid.color": "#E9E9E9",
        }
    )


def wrapped_method_label(method, method_labels: dict[str, str], width: int = 23) -> str:
    label = method_labels.get(method, str(method))
    return "\n".join(textwrap.wrap(label, width=width))


def short_compound_label(name, width: int = 18, aliases: dict[str, str] | None = None) -> str:
    aliases = {"Aminoglutethimide": "Aminoglutethimide"} if aliases is None else aliases
    text = aliases.get(str(name), str(name)).replace(" (", "\n(")
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [part])
    return "\n".join(lines[:2])


def metric_table_for_split(summary, split_name, method_order, method_labels: dict[str, str]):
    frame = pd.DataFrame(summary).loc[pd.DataFrame(summary)["split_name"].eq(split_name)].copy()
    available = set(frame["method"].astype(str))
    missing = [method_labels[m] for m in method_order if m not in available]
    ordered = [m for m in method_order if m in available]
    frame["_order"] = frame["method"].map({method: i for i, method in enumerate(ordered)})
    frame = frame.loc[frame["method"].isin(ordered)].sort_values("_order").drop(columns="_order")
    frame["method_label"] = frame["method"].map(method_labels)
    return frame.reset_index(drop=True), missing


def metric_value_table(frame):
    frame = pd.DataFrame(frame)
    if "MMD" in frame.columns and "Sliced W2" in frame.columns:
        label_col = "method_label" if "method_label" in frame.columns else "method"
        out = frame[[label_col, "MMD", "Sliced W2"]].copy().rename(columns={label_col: "method_label"})
    else:
        out = frame[["method_label", "program_readout_mmd", "program_readout_sliced_w2"]].copy().rename(
            columns={
                "program_readout_mmd": "MMD",
                "program_readout_sliced_w2": "Sliced W2",
            }
        )
    return out.rename(columns={"method_label": "method"})


def make_metric_display_table(rows, source_label, method_labels: dict[str, str]):
    frame = pd.DataFrame(rows)
    frame["method_label"] = frame["method"].map(method_labels)
    frame["metric_display_source"] = source_label
    return frame


def save_figure_pair(fig, fig_dir: str | Path, stem: str, tight: bool = True) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    png_path = fig_dir / f"{Path(stem).stem}.png"
    pdf_path = fig_dir / f"{Path(stem).stem}.pdf"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=0.8)
    fig.savefig(png_path, dpi=420, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def display_figure_output(paths):
    from .artifacts import display_saved_figure as _display_saved_figure

    return _display_saved_figure(paths["png"] if isinstance(paths, dict) else paths)


def draw_tiny_cloud(ax, center, color, seed: int, seed_offset=0, n=18, sx=0.023, sy=0.018, alpha=0.62):
    local = np.random.default_rng(int(seed) + seed_offset)
    pts = local.normal(size=(n, 2))
    pts[:, 0] = center[0] + pts[:, 0] * sx
    pts[:, 1] = center[1] + pts[:, 1] * sy
    ax.scatter(pts[:, 0], pts[:, 1], s=7.5, color=color, alpha=alpha, linewidths=0, clip_on=False)


def draw_velocity_box(ax, x, y, w, h, text, color, fontsize=8.8):
    import matplotlib.patches as mpatches

    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.007,rounding_size=0.016",
        linewidth=0.85,
        edgecolor=color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#252525")


def draw_method_tile(ax, method, formula, note, seed: int, seed_offset):
    import matplotlib.patches as mpatches

    color = METHOD_COLORS[method]
    long_formula = method in {"M3_no_chemistry", "M4_chemistry_aware"}
    box_x = 0.295 if long_formula else 0.365
    box_w = 0.430 if long_formula else 0.285
    formula_size = 8.3 if method == "M4_chemistry_aware" else 8.6 if long_formula else 9.4
    response_x = 0.875 if long_formula else 0.82

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tile = mpatches.FancyBboxPatch(
        (0.018, 0.035),
        0.964,
        0.895,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=0.75,
        edgecolor="#D6D6D6",
        facecolor="#FFFFFF",
    )
    ax.add_patch(tile)
    method_title_size = 8.8 if long_formula else 9.0
    ax.text(0.058, 0.880, METHOD_LABELS[method], ha="left", va="top", fontsize=method_title_size, weight="bold", color=color)

    draw_tiny_cloud(ax, (0.145, 0.50), "#AEBBC8", seed=seed, seed_offset=seed_offset, n=17)
    draw_velocity_box(ax, box_x, 0.420, box_w, 0.155, formula, color, fontsize=formula_size)
    draw_tiny_cloud(ax, (response_x, 0.50), color, seed=seed, seed_offset=seed_offset + 20, n=18, sx=0.026, sy=0.020, alpha=0.64)
    ax.annotate("", xy=(box_x - 0.020, 0.50), xytext=(0.230, 0.50), arrowprops={"arrowstyle": "->", "lw": 0.82, "color": "#6F6F6F"})
    ax.annotate("", xy=(response_x - 0.070, 0.50), xytext=(box_x + box_w + 0.020, 0.50), arrowprops={"arrowstyle": "->", "lw": 0.82, "color": "#6F6F6F"})
    ax.text(0.145, 0.325, "control", ha="center", va="center", fontsize=6.8, color="#666666")
    ax.text(response_x, 0.325, "response", ha="center", va="center", fontsize=6.8, color="#666666")

    if method == "M2_per_compound":
        for j, yy in enumerate([0.690, 0.635, 0.580]):
            ax.plot([0.43, 0.57], [yy, yy], color=color, lw=0.9, alpha=0.62)
            ax.text(0.590, yy, f"c{j + 1}", va="center", fontsize=6.1, color="#666666")
    elif method == "M3_no_chemistry":
        ax.add_patch(mpatches.Rectangle((0.374, 0.644), 0.055, 0.028, facecolor=color, alpha=0.18, edgecolor=color, linewidth=0.50))
        ax.text(0.448, 0.658, "one-hot", va="center", fontsize=5.25, color="#666666")
        ax.text(0.552, 0.658, "+ d", va="center", fontsize=5.35, color="#666666")
    elif method == "M4_chemistry_aware":
        for k in range(6):
            ax.add_patch(mpatches.Rectangle((0.390 + 0.016 * k, 0.640), 0.011, 0.034, facecolor=color, alpha=0.22 + 0.06 * (k % 3), edgecolor="none"))
        ax.text(0.525, 0.657, "+ d", va="center", fontsize=6.2, color="#666666")
    else:
        for j, resp_color in enumerate(["#D95F59", "#F28E2B", "#59A14F"]):
            draw_tiny_cloud(ax, (0.795 + 0.025 * j, 0.655), resp_color, seed=seed, seed_offset=seed_offset + 40 + j, n=6, sx=0.010, sy=0.009, alpha=0.42)

    ax.text(0.058, 0.145, note, ha="left", va="center", fontsize=7.4, color="#3A3A3A", wrap=True)


def build_model_design_figure(fig_dir: str | Path, seed: int) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.65))
    fig.suptitle(FIGURE_TITLES["fig_5_2_model_designs"], fontsize=10.2, y=0.972)
    method_tiles = [
        ("M1_unconditional", r"$v_\theta(x,\tau)$", "pooled response; condition ignored"),
        ("M2_per_compound", r"$v_\theta^{(c)}(x,\tau)$", "compound-specific field; no cross-compound sharing"),
        ("M3_no_chemistry", r"$v_\theta(x,\tau,e_{onehot}(c),d)$", "seen-compound dose extrapolation"),
        ("M4_chemistry_aware", r"$v_\theta(x,\tau,\mathrm{RDKit2D}(c),d)$", "chemistry-aware transfer signal"),
    ]
    for i, (ax, (method, formula, note)) in enumerate(zip(axes.ravel(), method_tiles)):
        draw_method_tile(ax, method, formula, note, seed=seed, seed_offset=100 + i)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.045, top=0.890, wspace=0.075, hspace=0.155)
    return save_figure_pair(fig, fig_dir, "fig_5_2_model_designs", tight=False)


def select_split_grid_compounds(metadata: pd.DataFrame, max_compounds: int = 6) -> list[str]:
    compounds_all = sorted(metadata.loc[~metadata["is_vehicle"], "compound"].astype(str).unique().tolist())
    priority_compounds = ["Alisertib (MLN8237)", "BMS-754807", "Crizotinib", "Dacinostat", "Givinostat", "Quisinostat"]
    compounds_for_grid = [compound for compound in priority_compounds if compound in compounds_all]
    for compound in compounds_all:
        if len(compounds_for_grid) >= max_compounds:
            break
        if compound not in compounds_for_grid:
            compounds_for_grid.append(compound)
    return compounds_for_grid


def split_status_matrix(split_meta: pd.DataFrame, compounds: list[str], doses: list[float]) -> pd.DataFrame:
    status = pd.DataFrame("missing", index=compounds, columns=doses)
    treated = split_meta.loc[~split_meta["is_vehicle"].astype(bool)].copy()
    for compound in compounds:
        for dose in doses:
            group = treated.loc[treated["compound"].astype(str).eq(compound) & np.isclose(treated["dose"].astype(float), dose)]
            if group.empty:
                continue
            if group["split"].eq("test").any():
                status.loc[compound, dose] = "test"
            elif group["split"].eq("train").any():
                status.loc[compound, dose] = "train"
    return status


def draw_split_grid(ax, status: pd.DataFrame, title: str, boundary_label: str, claim_label: str):
    import matplotlib.patches as mpatches

    n_rows, n_cols = status.shape
    colors = {"train": "#DCE6EE", "test": "#F2A65A", "missing": "#F6F6F6"}
    test_cells = []
    for r, compound in enumerate(status.index):
        y = n_rows - 1 - r
        for c, dose in enumerate(status.columns):
            value = status.loc[compound, dose]
            edge = "#A94E2A" if value == "test" else "#FFFFFF"
            lw = 1.05 if value == "test" else 0.72
            ax.add_patch(mpatches.Rectangle((c, y), 1, 1, facecolor=colors[value], edgecolor=edge, linewidth=lw))
            if value == "test":
                test_cells.append((c, y))
    if test_cells:
        coords = np.asarray(test_cells, dtype=float)
        cols = np.unique(coords[:, 0])
        rows = np.unique(coords[:, 1])
        if len(cols) == 1:
            label_x = float(cols[0] + 0.5)
            label_y = float(rows.max() + 0.5)
        elif len(rows) == 1:
            label_x = float(coords[:, 0].mean() + 0.5)
            label_y = float(rows[0] + 0.5)
        else:
            label_x = float(coords[:, 0].mean() + 0.5)
            label_y = float(coords[:, 1].mean() + 0.5)
        ax.text(label_x, label_y, "test", ha="center", va="center", fontsize=5.8, color="#7A301D", weight="bold")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([f"{dose:g}" for dose in status.columns], fontsize=6.6)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([short_compound_label(c, width=18) for c in reversed(status.index)], fontsize=5.7)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("dose (nM)", fontsize=7.0, labelpad=4)
    ax.set_title(title, fontsize=8.3, pad=19, weight="bold")
    ax.text(0.5, 1.055, boundary_label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color="#333333")
    ax.text(0.5, 1.005, claim_label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color="#666666")
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_evaluation_split_figure(metadata: pd.DataFrame, split_b: pd.DataFrame, split_c: pd.DataFrame, fig_dir: str | Path) -> dict[str, Path]:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    compounds_for_grid = select_split_grid_compounds(metadata)
    doses_for_grid = [10.0, 100.0, 1000.0, 10000.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharey=True)
    draw_split_grid(
        axes[0],
        split_status_matrix(split_b, compounds_for_grid, doses_for_grid),
        "Held-out highest dose",
        "seen compound, unseen highest dose",
        "dose extrapolation",
    )
    draw_split_grid(
        axes[1],
        split_status_matrix(split_c, compounds_for_grid, doses_for_grid),
        "Held-out compound",
        "unseen compound, all doses held out",
        "compound transfer",
    )
    handles = [
        mpatches.Patch(facecolor="#DCE6EE", edgecolor="#FFFFFF", label="train"),
        mpatches.Patch(facecolor="#F2A65A", edgecolor="#A94E2A", label="test"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.895), ncol=2, frameon=False, handlelength=1.0, columnspacing=1.3, fontsize=6.6)
    fig.text(0.985, 0.018, "schematic subset of compounds", ha="right", va="bottom", fontsize=6.0, color="#777777")
    fig.suptitle(FIGURE_TITLES["fig_5_2_evaluation_splits"], fontsize=10.1, y=0.977)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.17, top=0.705, wspace=0.16)
    return save_figure_pair(fig, fig_dir, "fig_5_2_evaluation_splits", tight=False)


def plot_metric_panel(frame: pd.DataFrame, title: str, fig_dir: str | Path, stem: str, note: str | None = None) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    metrics = [("MMD", "MMD", "{:.4f}"), ("Sliced W2", "Sliced W2", "{:.3f}")]
    n = len(frame)
    y = np.arange(n - 1, -1, -1)
    labels = [METHOD_LABELS[method] for method in frame["method"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharey=True)
    baseline_methods = {"vehicle_as_prediction", "mean_shift", "nearest_chemistry"}
    model_count = sum(method not in baseline_methods for method in frame["method"])
    separator_y = n - model_count - 0.5

    for ax, (col, metric_title, fmt) in zip(axes, metrics):
        values = frame[col].to_numpy(dtype=float)
        colors = [METHOD_COLORS[method] for method in frame["method"]]
        edge_colors = ["#8A8A8A" if method in baseline_methods else "#333333" for method in frame["method"]]
        ax.barh(y, values, height=0.58, color=colors, edgecolor=edge_colors, linewidth=0.45)
        ax.axhline(separator_y, color="#BDBDBD", linewidth=0.6, linestyle=(0, (2.2, 2.2)), zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.9)
        ax.set_ylim(-0.60, n - 0.30)
        ax.set_xlabel("metric value", fontsize=7.0)
        ax.set_title(metric_title, fontsize=8.3, pad=5)
        ax.text(0.99, 1.025, "lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color="#8A8A8A")
        ax.grid(axis="x", color="#E8E8E8", linewidth=0.42)
        ax.set_axisbelow(True)
        xmax = max(float(np.nanmax(values)) * 1.25, 1e-6)
        ax.set_xlim(0, xmax)
        for yi, value in zip(y, values):
            ax.text(value + xmax * 0.018, yi, fmt.format(value), ha="left", va="center", fontsize=6.6, color="#333333")
        ax.tick_params(axis="x", labelsize=6.7, length=2, width=0.55)
        ax.tick_params(axis="y", length=0, pad=3)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.6)
            ax.spines[spine].set_color("#777777")
    axes[1].tick_params(axis="y", labelleft=False)
    if note:
        fig.text(0.055, 0.030, note, ha="left", va="bottom", fontsize=6.2, color="#666666")
    fig.suptitle(title, fontsize=10.3, y=0.978)
    fig.subplots_adjust(left=0.315, right=0.985, bottom=0.175, top=0.805, wspace=0.10)
    return save_figure_pair(fig, fig_dir, stem, tight=False)


def required_section52_figure_stems() -> list[str]:
    return [
        "fig_5_2_model_designs",
        "fig_5_2_evaluation_splits",
        "fig_5_2_heldout_highest_dose_metrics",
        "fig_5_2_heldout_compound_metrics",
        "fig_5_2_alisertib_example",
    ]


def section52_required_paths(config: Section52Config, figure_paths: dict[str, dict[str, Path]]) -> tuple[list[Path], list[Path]]:
    required_figure_paths: list[Path] = []
    for stem in required_section52_figure_stems():
        required_figure_paths.extend([Path(figure_paths[stem]["png"]), Path(figure_paths[stem]["pdf"])])
    required_paths = [
        *required_figure_paths,
        config.table_dir / "tab_5_2_sciplex_splits.csv",
        config.output_dir / "rdkit2d_compound_features.npz",
        config.output_dir / "rdkit2d_diagnostics.json",
        config.output_dir / "rdkit2d_audit.csv",
        config.output_dir / "sciplex_metrics_by_group.csv",
        config.output_dir / "sciplex_metrics_summary.csv",
        config.output_dir / "real_data_audit.json",
        config.output_dir / "run_summary_perturbation_sciplex.json",
    ]
    return required_figure_paths, required_paths


def finite_metric_checks(metric_frames: dict[str, pd.DataFrame]) -> dict[str, bool]:
    checks = {}
    for name, frame in metric_frames.items():
        numeric = frame.select_dtypes(include=[np.number])
        checks[name] = bool(np.isfinite(numeric.to_numpy()).all()) if numeric.size else True
    return checks


def section52_figure_data_sources() -> dict[str, str]:
    return {
        "fig_5_2_model_designs": "deterministic schematic from Section 5.2 model definitions; no experimental metrics",
        "fig_5_2_evaluation_splits": "split_b, split_c, metadata, and tables/ch05/tab_5_2_sciplex_splits.csv",
        "fig_5_2_heldout_highest_dose_metrics": "split_b_metric_display synchronized to manuscript main-text table; raw cached metrics remain in outputs/ch05/sciplex_metrics_summary.csv",
        "fig_5_2_heldout_compound_metrics": "split_c_metric_display synchronized to manuscript main-text table; raw cached metrics remain in outputs/ch05/sciplex_metrics_summary.csv",
        "fig_5_2_alisertib_example": "split_c_cache['predictions'][representative_key] and split-aware PCA state states['Split C held-out compound']",
    }


def section52_figure_title_audit() -> pd.DataFrame:
    title_audit = pd.DataFrame(
        [
            {"figure": stem, "title": FIGURE_TITLES[stem], "has_panel_letter_prefix": False}
            for stem in required_section52_figure_stems()
        ]
    )
    forbidden_title_tokens = ("A.", "B.", "C.", "D.", "Panel A", "Panel B", "Panel C", "Panel D")
    if any(str(title).startswith(forbidden_title_tokens) or "Panel " in str(title) for title in FIGURE_TITLES.values()):
        raise ValueError("A figure title contains an A/B/C/D panel-style prefix.")
    return title_audit


def audit_formula_labels() -> None:
    formula_strings = [
        r"$v_\theta(x,\tau)$",
        r"$v_\theta^{(c)}(x,\tau)$",
        r"$v_\theta(x,\tau,e_{onehot}(c),d)$",
        r"$v_\theta(x,\tau,\mathrm{RDKit2D}(c),d)$",
    ]
    code_style_formula_tokens = ["v" + "_theta", "RDKit2D" + "(c), " + "dose", "e" + "(c)"]
    source_for_formula_check = "\n".join(formula_strings)
    if any(token in source_for_formula_check for token in code_style_formula_tokens):
        raise ValueError("Code-style formula text reached formula labels.")


def build_section52_run_summary(
    *,
    run_summary: dict,
    config: Section52Config,
    figure_paths: dict[str, dict[str, Path]],
    metric_frames: dict[str, pd.DataFrame],
    split_b_metric_display: pd.DataFrame,
    split_c_metric_display: pd.DataFrame,
    representative_key,
    manuscript_metric_source: str,
    missing_result_notes: list[str],
) -> tuple[dict, list[Path], dict[str, bool], pd.DataFrame]:
    required_figure_paths, required_paths = section52_required_paths(config, figure_paths)
    checks = finite_metric_checks(metric_frames)
    figure_data_sources = section52_figure_data_sources()
    figure_title_audit = section52_figure_title_audit()
    audit_formula_labels()

    if not missing_result_notes:
        missing_result_notes.append("No requested plotted model, baseline, or metric is missing.")

    section52_summary = {
        "generated_files": [str(path.relative_to(config.project_root)) for path in required_figure_paths],
        "figure_data_sources": figure_data_sources,
        "metric_display_source": manuscript_metric_source,
        "raw_metric_source": "sciplex_summary from outputs/ch05/sciplex_metrics_summary.csv",
        "heldout_highest_dose_metrics": metric_value_table(split_b_metric_display),
        "heldout_compound_metrics": metric_value_table(split_c_metric_display),
        "alisertib_scatter": {
            "compound": representative_key[0],
            "dose": float(representative_key[1]),
            "split_name": "Split C held-out compound",
            "source_variable": "split_c_cache['predictions'][representative_key]",
        },
        "missing_or_not_applicable": missing_result_notes,
        "no_panel_letter_titles": True,
        "no_code_style_formula_text": True,
        "combined_figure_generated": False,
        "interpretation_note": (
            "M4 improves over M3 on held-out-compound distribution-shape metrics; "
            "vehicle and mean-shift baselines remain competitive and are visually separated as baselines."
        ),
    }

    run_summary["splits_evaluated"] = [
        "Split B held-out highest dose",
        "Split C held-out compound",
    ]
    run_summary["key_metrics"] = {
        "heldout_highest_dose": metric_value_table(split_b_metric_display),
        "heldout_compound": metric_value_table(split_c_metric_display),
        "representative_heldout": run_summary.get("sciplex_representative_heldout", {}),
    }
    run_summary["section52_independent_figures"] = {
        stem: {fmt: str(path.relative_to(config.project_root)) for fmt, path in paths.items()}
        for stem, paths in figure_paths.items()
    }
    run_summary["section52_figure_summary"] = section52_summary
    run_summary["finite_metric_checks"] = checks
    run_summary["expected_artifacts"] = [str(path.relative_to(config.project_root)) for path in required_paths]
    if bool(run_summary["sciplex_data"]["summary"].get("is_synthetic", False)):
        raise ValueError("Synthetic sci-Plex data reached final summary; refusing to write perturbation run summary.")
    if "synthetic" in str(run_summary["sciplex_data"]["summary"].get("source", "")).lower():
        raise ValueError("Synthetic-labeled sci-Plex source reached final summary; refusing to write perturbation run summary.")
    return section52_summary, required_paths, checks, figure_title_audit


def audit_section52_artifacts(required_paths: list[Path], finite_checks: dict[str, bool], project_root: str | Path) -> pd.DataFrame:
    project_root = Path(project_root)
    missing_paths = []
    for path in required_paths:
        if not path.exists() or path.stat().st_size <= 0:
            missing_paths.append(str(path))
    if missing_paths:
        raise FileNotFoundError(f"Missing or empty required artifacts: {missing_paths}")
    if not all(finite_checks.values()):
        raise ValueError(f"Non-finite numeric metrics detected: {finite_checks}")
    return pd.DataFrame(
        {
            "path": [str(path.relative_to(project_root)) for path in required_paths],
            "bytes": [path.stat().st_size for path in required_paths],
        }
    )


def make_metric_display_table_from_summary(
    source_path,
    split_name,
    method_order,
    expected_display,
    method_labels: dict[str, str],
    *,
    project_root=None,
):
    source_path = Path(source_path)
    payload = json.loads(source_path.read_text())
    rows = payload.get("key_metrics", {}).get("sciplex_summary", payload.get("sciplex_metrics_summary", []))
    raw = pd.DataFrame(rows)
    required = {"split_name", "method", "program_readout_mmd", "program_readout_sliced_w2"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Metric display source is missing required columns: {source_path}")
    frame = raw.loc[raw["split_name"].eq(split_name) & raw["method"].isin(method_order)].copy()
    frame["_order"] = frame["method"].map({method: i for i, method in enumerate(method_order)})
    frame = frame.sort_values("_order").drop(columns="_order")
    if frame["method"].tolist() != list(method_order):
        raise ValueError(f"Metric display source does not contain all requested methods for {split_name}")
    frame["MMD"] = frame["program_readout_mmd"].astype(float).round(4)
    frame["Sliced W2"] = frame["program_readout_sliced_w2"].astype(float).round(3)
    frame["method_label"] = frame["method"].map(method_labels)
    if project_root is None:
        metric_display_source = str(source_path)
    else:
        metric_display_source = str(source_path.relative_to(Path(project_root)))
    frame["metric_display_source"] = metric_display_source
    expected = pd.DataFrame(expected_display)
    got = frame[["method", "MMD", "Sliced W2"]].reset_index(drop=True)
    want = expected[["method", "MMD", "Sliced W2"]].reset_index(drop=True)
    if not got.equals(want):
        raise ValueError(f"Display metrics do not match the manuscript table for {split_name}:\n{got}\n!=\n{want}")
    return frame


def build_section52_metric_display_tables(
    sciplex_summary,
    manuscript_metric_source_path: str | Path,
    *,
    project_root: str | Path | None = None,
    method_labels: dict[str, str] = METHOD_LABELS,
) -> Section52MetricDisplayPackage:
    manuscript_metric_source_path = Path(manuscript_metric_source_path)
    if not manuscript_metric_source_path.exists():
        raise FileNotFoundError(
            f"Manuscript metric source {manuscript_metric_source_path} is missing; "
            "cannot synchronize Figure 5.2 metric panels to the main-text table."
        )

    split_b_metric_table, split_b_missing = metric_table_for_split(
        sciplex_summary,
        "Split B held-out highest dose",
        SPLIT_B_METHOD_ORDER,
        method_labels,
    )
    split_c_metric_table, split_c_missing = metric_table_for_split(
        sciplex_summary,
        "Split C held-out compound",
        SPLIT_C_METHOD_ORDER,
        method_labels,
    )
    split_b_metric_display = make_metric_display_table_from_summary(
        manuscript_metric_source_path,
        "Split B held-out highest dose",
        SPLIT_B_METHOD_ORDER,
        EXPECTED_SPLIT_B_DISPLAY,
        method_labels,
        project_root=project_root,
    )
    split_c_metric_display = make_metric_display_table_from_summary(
        manuscript_metric_source_path,
        "Split C held-out compound",
        SPLIT_C_METHOD_ORDER,
        EXPECTED_SPLIT_C_DISPLAY,
        method_labels,
        project_root=project_root,
    )
    if project_root is None:
        manuscript_metric_source = str(manuscript_metric_source_path)
    else:
        manuscript_metric_source = str(manuscript_metric_source_path.relative_to(Path(project_root)))

    missing_result_notes = []
    if split_b_missing:
        missing_result_notes.append("Held-out highest dose missing from raw summary: " + ", ".join(split_b_missing))
    if split_c_missing:
        missing_result_notes.append("Held-out compound missing from raw summary: " + ", ".join(split_c_missing))
    missing_result_notes.append(
        "M2 one-flow-per-compound is not plotted for held-out compound because an unseen compound has no trained per-compound flow."
    )
    return Section52MetricDisplayPackage(
        split_b_metric_table=split_b_metric_table,
        split_c_metric_table=split_c_metric_table,
        split_b_metric_display=split_b_metric_display,
        split_c_metric_display=split_c_metric_display,
        split_b_missing=split_b_missing,
        split_c_missing=split_c_missing,
        manuscript_metric_source=manuscript_metric_source,
        missing_result_notes=missing_result_notes,
    )
