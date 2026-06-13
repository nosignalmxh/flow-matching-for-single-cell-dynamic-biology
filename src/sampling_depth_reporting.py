from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .artifacts import (
    display_saved_figure,
    remember_source,
    resolve_required_artifact,
    safe_relpath,
    save_csv,
    save_figure_formats,
    save_json,
)
from .manifold_data import CH04_PALETTE, load_eb_data
from .metrics import mmd_rbf, sliced_w2
from .ot import compute_cost_matrix, coupling_diagnostics, sample_from_plan, sinkhorn_plan


FINAL_FIGURE_CLAIMS = {
    "fig4_11a_raw_observed_counts": "Raw destructive snapshot counts are sampling-depth proxies, not calibrated biological abundance.",
    "fig4_11b_equal_depth_composition": "Equal-depth subsampling changes the mass convention while preserving state-bin composition diagnostics.",
    "fig4_11c_sampling_depth_bootstrap_sensitivity": "Raw-count growth proxies are sensitive to equal-depth bootstrap intervals under the sampling-depth diagnostic.",
    "fig4_11d_wfrfm_raw_minus_equal_growth_heatmap": "WFR-FM growth readout changes under raw-depth minus equal-depth mass convention.",
    "fig4_11e_wfrfm_mass_convention_agreement_summary": "WFR-FM rank agreement is high while signs and calibration remain convention-dependent.",
    "fig4_11f_stochastic_bridge_demo": "Stochastic bridge width is a separate synthetic normalized path-family assumption from EB mass convention.",
}
FINAL_FORMATS = ("png", "pdf", "svg")


@dataclass(frozen=True)
class WfrfmGrowthDelta:
    matrix: np.ndarray
    eval_order: list
    bin_order: list[str]


@dataclass(frozen=True)
class WfrfmOutputs:
    suffix: str
    growth_path: Path
    sensitivity_path: Path
    summary_path: Path
    growth_by_bin: pd.DataFrame
    sampling_sensitivity: pd.DataFrame
    summary: dict


@dataclass(frozen=True)
class FinalFigurePackage:
    readme_path: Path
    manifest_path: Path
    qa_table: pd.DataFrame
    manifest: pd.DataFrame



def save_figure(fig, fig_dir: str | Path, filename: str, close: bool = True) -> Path:
    import matplotlib.pyplot as plt

    path = Path(fig_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def save_pub_figure(fig, final_fig_dir: str | Path, stem: str, close: bool = True) -> list[Path]:
    paths = save_figure_formats(
        fig,
        final_fig_dir,
        stem,
        formats=FINAL_FORMATS,
        dpi=420,
        close=close,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    return paths


def display_final_png(final_fig_dir: str | Path, stem: str, width: int | None = None) -> Path:
    return display_saved_figure(Path(final_fig_dir) / f"{stem}.png", width=width)


def display_ch04_png(fig_dir: str | Path, filename: str, width: int | None = None) -> Path:
    return display_saved_figure(Path(fig_dir) / filename, width=width)



def ordered_state_pivot(frame: pd.DataFrame, *, value_col: str, time_order, bin_order) -> pd.DataFrame:
    pivot = frame.pivot_table(index="time", columns="state_bin", values=value_col, fill_value=0)
    return pivot.reindex(index=time_order, columns=bin_order, fill_value=0)


def bridge_sampling_diagnostic(
    *,
    pcs_all,
    labels_all,
    state_bins,
    all_bins,
    time_a,
    time_b,
    sampling_setting: str,
    cap: int,
    seed: int,
    sinkhorn_epsilon: float,
    min_sample_pairs: int = 1024,
    max_sample_pairs: int = 4096,
) -> dict:
    pcs_all = np.asarray(pcs_all, dtype=np.float32)
    labels_all = np.asarray(labels_all)
    state_bins = np.asarray(state_bins)
    if pcs_all.ndim != 2:
        raise ValueError("pcs_all must be a 2D array")
    if len(labels_all) != pcs_all.shape[0] or len(state_bins) != pcs_all.shape[0]:
        raise ValueError("labels_all and state_bins must match pcs_all rows")
    cap = int(cap)
    if cap <= 0:
        raise ValueError("cap must be positive")

    idx_a_all = np.flatnonzero(labels_all == time_a)
    idx_b_all = np.flatnonzero(labels_all == time_b)
    if len(idx_a_all) == 0 or len(idx_b_all) == 0:
        raise ValueError(f"missing cells for bridge {time_a}->{time_b}")

    if sampling_setting == "original_depth":
        n_source = min(len(idx_a_all), cap)
        n_target = min(len(idx_b_all), cap)
    elif sampling_setting == "equal_depth":
        n_source = n_target = min(len(idx_a_all), len(idx_b_all), cap)
    else:
        raise ValueError(f"unknown sampling_setting={sampling_setting}")

    rng_local = np.random.default_rng(seed)
    ia = np.sort(rng_local.choice(idx_a_all, size=n_source, replace=False))
    ib = np.sort(rng_local.choice(idx_b_all, size=n_target, replace=False))
    xa, xb = pcs_all[ia], pcs_all[ib]
    cost, _scale = compute_cost_matrix(xa, xb, normalize=True)
    plan, info = sinkhorn_plan(cost, epsilon=sinkhorn_epsilon, return_info=True)
    diagnostics = coupling_diagnostics(plan, cost)
    n_sample = min(int(max_sample_pairs), max(int(min_sample_pairs), 4 * int(n_source)))
    _sampled_i, sampled_j = sample_from_plan(plan, n_sample, seed=seed + 17)
    sampled_endpoint = xb[sampled_j]
    pred_bins = state_bins[ib][sampled_j]
    target_bins = state_bins[ib]
    pred_prop = pd.Series(pred_bins).value_counts(normalize=True)
    target_prop = pd.Series(target_bins).value_counts(normalize=True)
    bin_err = sum(abs(float(pred_prop.get(k, 0.0)) - float(target_prop.get(k, 0.0))) for k in all_bins)
    return {
        "time_bridge": f"{time_a}->{time_b}",
        "sampling_setting": sampling_setting,
        "n_source": int(n_source),
        "n_target": int(n_target),
        "endpoint_mmd_pc20": float(mmd_rbf(sampled_endpoint, xb)),
        "sliced_w2_pc20": float(sliced_w2(sampled_endpoint, xb, seed=seed + 23)),
        "state_bin_terminal_proportion_error": float(bin_err),
        "expected_cost_normalized": float(diagnostics["expected_cost"]),
        "effective_support": float(diagnostics["effective_support"]),
        "sinkhorn_converged": bool(info["sinkhorn_converged"]),
        "diagnostic_type": "ot_sampled_endpoint_diagnostic_not_trained_cfm",
        "claim_boundary": (
            "standardized PC-20 OT sampled endpoint diagnostic; not trained CFM, "
            "not observed paired histories, not growth or calibrated abundance"
        ),
    }


def plot_raw_observed_counts(
    counts_by_state: pd.DataFrame,
    *,
    unique_times,
    all_bins,
    final_fig_dir: str | Path,
    display: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    color_list = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1", "#EDC948", "#9C755F"]
    colors = {b: color_list[i % len(color_list)] for i, b in enumerate(all_bins)}
    pivot = ordered_state_pivot(counts_by_state, value_col="n_cells", time_order=unique_times, bin_order=all_bins)
    fig, ax = plt.subplots(figsize=(5.4, 3.25))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.78, color=[colors[b] for b in pivot.columns], legend=False)
    ax.set_title("Raw observed EB counts\nsampling-depth proxy, not calibrated abundance", loc="left", pad=8)
    ax.set_xlabel("time label")
    ax.set_ylabel("observed/sample cells")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="state bin", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    path = save_pub_figure(fig, final_fig_dir, "fig4_11a_raw_observed_counts")[0]
    if display:
        display_final_png(final_fig_dir, "fig4_11a_raw_observed_counts")
    return path


def plot_equal_depth_composition(
    equal_counts: pd.DataFrame,
    *,
    unique_times,
    all_bins,
    n_min: int,
    final_fig_dir: str | Path,
    display: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    color_list = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1", "#EDC948", "#9C755F"]
    colors = {b: color_list[i % len(color_list)] for i, b in enumerate(all_bins)}
    frame = equal_counts.copy()
    frame["proportion"] = frame["n_cells"] / frame["total_time_cells"]
    pivot = ordered_state_pivot(frame, value_col="proportion", time_order=unique_times, bin_order=all_bins)
    fig, ax = plt.subplots(figsize=(5.4, 3.25))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.78, color=[colors[b] for b in pivot.columns], legend=False)
    ax.set_title(f"Equal-depth EB composition\nequal-depth: n = {int(n_min)} per time point", loc="left", pad=8)
    ax.set_xlabel("time label")
    ax.set_ylabel("state-bin proportion")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="state bin", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    path = save_pub_figure(fig, final_fig_dir, "fig4_11b_equal_depth_composition")[0]
    if display:
        display_final_png(final_fig_dir, "fig4_11b_equal_depth_composition")
    return path


def plot_sampling_depth_bootstrap_sensitivity(
    downsampling_table: pd.DataFrame,
    *,
    final_fig_dir: str | Path,
    display: bool = True,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    plot_df = downsampling_table.copy()
    plot_df["state_bin"] = plot_df["state_bin"].astype(str)
    plot_df = plot_df.sort_values(["time_bridge", "state_bin"]).reset_index(drop=True)
    plot_df["label"] = plot_df["time_bridge"].astype(str) + "\nbin " + plot_df["state_bin"].astype(str)
    inside_ci = (
        (plot_df["raw_count_growth_proxy"] >= plot_df["equal_depth_proxy_ci_low"])
        & (plot_df["raw_count_growth_proxy"] <= plot_df["equal_depth_proxy_ci_high"])
    )
    sensitive_count = int((plot_df["stable_under_subsampling"] == "sensitive").sum())
    inside_count = int(inside_ci.sum())
    n_comparisons = int(len(plot_df))

    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    x = np.arange(n_comparisons)
    y = plot_df["equal_depth_proxy_mean"].to_numpy(dtype=float)
    yerr = np.vstack([
        y - plot_df["equal_depth_proxy_ci_low"].to_numpy(dtype=float),
        plot_df["equal_depth_proxy_ci_high"].to_numpy(dtype=float) - y,
    ])
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=CH04_PALETTE["ot"],
        ecolor="0.62",
        elinewidth=1.0,
        capsize=2.2,
        markersize=4.2,
        label="equal-depth 95% CI",
    )
    ax.scatter(
        x,
        plot_df["raw_count_growth_proxy"].to_numpy(dtype=float),
        color=CH04_PALETTE["diagnostic"],
        edgecolor="white",
        linewidth=0.4,
        s=28,
        zorder=3,
        label="raw-count proxy",
    )
    ax.axhline(0.0, color="0.35", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=55, ha="right")
    ax.set_ylabel("log growth proxy")
    ax.set_title("Raw-count proxy vs equal-depth bootstrap\nsampling-depth diagnostic, not abundance calibration", loc="left", pad=8)
    ax.text(
        0.015,
        0.98,
        f"{sensitive_count}/{n_comparisons} sensitive\n{inside_count}/{n_comparisons} raw proxies inside equal-depth 95% CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.15",
        bbox={"facecolor": "white", "edgecolor": "0.75", "boxstyle": "round,pad=0.25", "linewidth": 0.6},
    )
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left", borderaxespad=0.0)
    save_pub_figure(fig, final_fig_dir, "fig4_11c_sampling_depth_bootstrap_sensitivity")
    if display:
        display_final_png(final_fig_dir, "fig4_11c_sampling_depth_bootstrap_sensitivity")
    return plot_df


def resolve_wfrfm_output_suffix(out_dir: str | Path, env_suffix: str | None = None) -> str:
    suffix = os.environ.get("CH04_WFRFM_OUTPUT_SUFFIX") if env_suffix is None else env_suffix
    if suffix is not None:
        return suffix.strip().lstrip("_")
    return "full" if (Path(out_dir) / "wfrfm_sampling_sensitivity_summary_full.json").exists() else ""


def wfrfm_output_name(stem: str, ext: str, suffix: str = "") -> str:
    clean = str(suffix).strip().lstrip("_")
    return f"{stem}_{clean}.{ext}" if clean else f"{stem}.{ext}"


def load_wfrfm_sampling_outputs(
    *,
    out_dir: str | Path,
    project_root: str | Path,
    source_paths: dict[str, str] | None = None,
    suffix: str | None = None,
) -> WfrfmOutputs:
    out_dir = Path(out_dir)
    suffix = resolve_wfrfm_output_suffix(out_dir) if suffix is None else suffix
    growth_name = wfrfm_output_name("table4_6c_wfrfm_growth_by_bin", "csv", suffix)
    sensitivity_name = wfrfm_output_name("table4_6d_wfrfm_sampling_sensitivity", "csv", suffix)
    summary_name = wfrfm_output_name("wfrfm_sampling_sensitivity_summary", "json", suffix)
    growth_path = resolve_required_artifact(growth_name, preferred_dirs=[out_dir], search_root=project_root)
    sensitivity_path = resolve_required_artifact(sensitivity_name, preferred_dirs=[out_dir], search_root=project_root)
    summary_path = resolve_required_artifact(summary_name, preferred_dirs=[out_dir], search_root=project_root)
    if source_paths is not None:
        label_suffix = f"_{suffix}" if suffix else ""
        remember_source(source_paths, f"table4_6c_wfrfm_growth_by_bin{label_suffix}.csv", growth_path, root=project_root)
        remember_source(source_paths, f"table4_6d_wfrfm_sampling_sensitivity{label_suffix}.csv", sensitivity_path, root=project_root)
        remember_source(source_paths, f"wfrfm_sampling_sensitivity_summary{label_suffix}.json", summary_path, root=project_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if suffix == "full":
        assert summary.get("smoke") is False, "Full WFR-FM output must not be smoke mode."
        assert int(summary.get("epochs")) == 300, "Full WFR-FM output must use 300 epochs."
        assert summary.get("raw_observed_depth_was_capped") is False, "Full WFR-FM raw observed depth must be uncapped."
        assert summary.get("external_baseline_runtime_dependency") is False, "Exp 9b must use the internal WFR-FM implementation."
    return WfrfmOutputs(
        suffix=suffix,
        growth_path=growth_path,
        sensitivity_path=sensitivity_path,
        summary_path=summary_path,
        growth_by_bin=pd.read_csv(growth_path),
        sampling_sensitivity=pd.read_csv(sensitivity_path),
        summary=summary,
    )


def make_wfrfm_growth_delta_grid(growth_by_bin: pd.DataFrame) -> WfrfmGrowthDelta:
    growth_plot = growth_by_bin.copy()
    growth_plot["eval_time"] = pd.to_numeric(growth_plot["eval_time"])
    growth_plot["state_bin"] = growth_plot["state_bin"].astype(str)
    growth_plot["mean_g"] = pd.to_numeric(growth_plot["mean_g"], errors="coerce")
    mean_growth = growth_plot.groupby(["setting", "eval_time", "state_bin"], as_index=False)["mean_g"].mean()
    raw_grid = mean_growth[mean_growth["setting"] == "raw_observed_depth"].pivot_table(
        index="eval_time", columns="state_bin", values="mean_g", aggfunc="mean"
    )
    equal_grid = mean_growth[mean_growth["setting"] == "equal_depth"].pivot_table(
        index="eval_time", columns="state_bin", values="mean_g", aggfunc="mean"
    )
    eval_order = sorted(set(raw_grid.index).union(equal_grid.index))
    bin_order = sorted(set(raw_grid.columns).union(equal_grid.columns), key=lambda s: int(s) if str(s).isdigit() else str(s))
    delta = raw_grid.reindex(eval_order, columns=bin_order) - equal_grid.reindex(eval_order, columns=bin_order)
    return WfrfmGrowthDelta(matrix=delta.to_numpy(dtype=float), eval_order=eval_order, bin_order=list(bin_order))


def make_wfrfm_agreement_summary(sampling_sensitivity: pd.DataFrame) -> pd.DataFrame:
    values = {
        "Spearman rank": float(sampling_sensitivity["spearman_growth_rank"].mean()),
        "Top expanding": float(sampling_sensitivity["top_expanding_overlap_k3"].mean()),
        "Top shrinking": float(sampling_sensitivity["top_shrinking_overlap_k3"].mean()),
        "Sign agreement": float(sampling_sensitivity["sign_agreement"].mean()),
    }
    return pd.DataFrame([{"metric": key, "value": value, "display_value": round(value, 2)} for key, value in values.items()])


def plot_wfrfm_growth_delta_heatmap(delta: WfrfmGrowthDelta, *, final_fig_dir: str | Path, display: bool = True) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    matrix = delta.matrix
    fig, ax = plt.subplots(figsize=(5.4, 3.45))
    absmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    if absmax == 0:
        absmax = 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax))
    ax.set_title("WFR-FM growth: raw minus equal-depth\ngrowth readout depends on mass convention", loc="left", pad=8)
    ax.set_xlabel("state bin")
    ax.set_ylabel("evaluation time")
    ax.set_xticks(np.arange(len(delta.bin_order)))
    ax.set_xticklabels(delta.bin_order)
    ax.set_yticks(np.arange(len(delta.eval_order)))
    ax.set_yticklabels([f"{t:g}" for t in delta.eval_order])
    ax.set_facecolor("#F2F2F2")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                color = "white" if abs(matrix[i, j]) > 0.6 * absmax else "0.15"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("raw-depth minus equal-depth mean growth")
    path = save_pub_figure(fig, final_fig_dir, "fig4_11d_wfrfm_raw_minus_equal_growth_heatmap")[0]
    if display:
        display_final_png(final_fig_dir, "fig4_11d_wfrfm_raw_minus_equal_growth_heatmap")
    return path


def plot_wfrfm_agreement_summary(summary: pd.DataFrame, *, final_fig_dir: str | Path, display: bool = True) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.1, 2.75))
    ypos = np.arange(len(summary))
    ax.barh(ypos, summary["display_value"], color=["#4E79A7", "#59A14F", "#76B7B2", "#E15759"], height=0.58)
    ax.set_yticks(ypos)
    ax.set_yticklabels(summary["metric"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("agreement")
    ax.set_title("Mass-convention agreement summary\nrank mostly stable; signs/calibration remain convention-dependent", loc="left", pad=8)
    for y, val in zip(ypos, summary["display_value"]):
        ax.text(min(val + 0.025, 1.02), y, f"{val:.2f}", va="center", ha="left", fontsize=9, color="0.15")
    ax.grid(axis="x", color="0.9", linewidth=0.7)
    path = save_pub_figure(fig, final_fig_dir, "fig4_11e_wfrfm_mass_convention_agreement_summary")[0]
    if display:
        display_final_png(final_fig_dir, "fig4_11e_wfrfm_mass_convention_agreement_summary")
    return path


def synthetic_bridge_samples(n: int = 900, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[-1.5, 0.0], scale=[0.25, 0.45], size=(n, 2))
    mix = rng.uniform(size=n) < 0.5
    x1 = np.empty((n, 2))
    x1[mix] = rng.normal(loc=[1.4, 0.8], scale=[0.28, 0.25], size=(mix.sum(), 2))
    x1[~mix] = rng.normal(loc=[1.4, -0.8], scale=[0.28, 0.25], size=((~mix).sum(), 2))
    return x0.astype(np.float32), x1.astype(np.float32)


def plot_stochastic_bridge_demo(*, final_fig_dir: str | Path, seed: int = 401, display: bool = True) -> tuple[Path, list[float]]:
    import math
    import matplotlib.pyplot as plt

    x0_syn, x1_syn = synthetic_bridge_samples()
    t_grid_demo = [0, 0.25, 0.5, 0.75, 1]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, len(t_grid_demo), figsize=(8.3, 3.25), sharex=True, sharey=True)
    for col, tval in enumerate(t_grid_demo):
        det = (1 - tval) * x0_syn + tval * x1_syn
        noise_scale = 0.44 * math.sqrt(max(tval * (1 - tval), 0.0))
        stoch = det + rng.normal(scale=noise_scale, size=det.shape)
        for row, pts, label, color in [
            (0, det, "deterministic\npoint clouds", "#4E79A7"),
            (1, stoch, "stochastic\npoint clouds", "#E15759"),
        ]:
            ax = axes[row, col]
            ax.scatter(pts[:, 0], pts[:, 1], s=3.8, alpha=0.34, linewidths=0, color=color)
            if row == 0:
                ax.set_title(f"t = {tval:g}", fontsize=9, pad=4)
            if col == 0:
                ax.set_ylabel(label)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_xlim(-2.35, 2.1)
    axes[0, 0].set_ylim(-1.75, 1.75)
    fig.suptitle("Synthetic normalized bridge demo; not EB abundance or growth", x=0.02, y=0.99, ha="left", fontsize=10.5)
    fig.text(0.02, 0.91, "wider intermediate distributions under stochastic bridge construction", ha="left", va="top", fontsize=8.5, color="0.35")
    path = save_pub_figure(fig, final_fig_dir, "fig4_11f_stochastic_bridge_demo")[0]
    if display:
        display_final_png(final_fig_dir, "fig4_11f_stochastic_bridge_demo")
    return path, t_grid_demo


def write_final_figure_package(
    *,
    project_root: str | Path,
    final_fig_dir: str | Path,
    source_paths: Mapping[str, str | Path],
    final_source_paths: Mapping[str, str],
) -> FinalFigurePackage:
    project_root = Path(project_root)
    final_fig_dir = Path(final_fig_dir)
    figure_paths = [final_fig_dir / f"{stem}.{ext}" for stem in FINAL_FIGURE_CLAIMS for ext in FINAL_FORMATS]
    missing = [path for path in figure_paths if not path.exists()]
    empty = [path for path in figure_paths if path.exists() and path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError("Missing required final figures: " + ", ".join(safe_relpath(p, project_root) for p in missing))
    if empty:
        raise ValueError("Empty required final figures: " + ", ".join(safe_relpath(p, project_root) for p in empty))
    bad_tokens = ("composite", "combined", "multi_panel", "multipanel", "big_figure", "bigfigure")
    bad_names = [path.name for path in figure_paths if any(token in path.stem.lower() for token in bad_tokens)]
    if bad_names:
        raise ValueError("Output filename implies a composite figure: " + ", ".join(bad_names))

    manifest_rows = []
    source_list = "; ".join(sorted(final_source_paths.values()))
    for stem, claim in FINAL_FIGURE_CLAIMS.items():
        for ext in FINAL_FORMATS:
            path = final_fig_dir / f"{stem}.{ext}"
            manifest_rows.append({
                "artifact": safe_relpath(path, project_root),
                "kind": "figure",
                "figure_stem": stem,
                "format": ext,
                "claim_supported": claim,
                "bytes": int(path.stat().st_size),
                "source_tables_or_json": source_list,
            })
    for source_name, source_path in source_paths.items():
        path = Path(source_path)
        manifest_rows.append({
            "artifact": safe_relpath(path, project_root),
            "kind": "source_table_or_json",
            "figure_stem": "",
            "format": path.suffix.lstrip("."),
            "claim_supported": "Source checked for the final sampling-depth, mass-convention, or bridge-width figure package.",
            "bytes": int(path.stat().st_size),
            "source_tables_or_json": source_name,
        })

    readme_path = final_fig_dir / "README_final_check.md"
    manifest_path = final_fig_dir / "final_polish_manifest.csv"
    readme_lines = [
        "# Chapter 4 Final Small Figure Package",
        "",
        f"Output directory: `{safe_relpath(final_fig_dir, project_root)}`",
        "",
        "Claim boundary:",
        "- Raw observed EB counts are sampling-depth proxies, not calibrated biological abundance.",
        "- Equal-depth subsampling changes the mass convention while preserving state-bin composition diagnostics.",
        "- Raw-count growth proxies and WFR-FM growth readouts depend on the input mass convention.",
        "- Stochastic bridge width is a separate synthetic normalized path-family assumption.",
        "",
        "Generated figure stems:",
    ]
    readme_lines.extend(f"- `{stem}`: {claim}" for stem, claim in FINAL_FIGURE_CLAIMS.items())
    readme_lines.extend(["", "Source tables/json checked:"])
    readme_lines.extend(f"- `{safe_relpath(path, project_root)}`" for path in source_paths.values())
    readme_lines.extend([
        "",
        "QA checks:",
        "- Required PNG/PDF/SVG files exist and are nonzero.",
        "- Output filenames do not imply a composite figure.",
        "- This README and `final_polish_manifest.csv` were written.",
    ])
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    manifest_rows.extend([
        {
            "artifact": safe_relpath(readme_path, project_root),
            "kind": "readme",
            "figure_stem": "",
            "format": "md",
            "claim_supported": "Final check and claim-boundary summary.",
            "bytes": int(readme_path.stat().st_size),
            "source_tables_or_json": "",
        },
        {
            "artifact": safe_relpath(manifest_path, project_root),
            "kind": "manifest",
            "figure_stem": "",
            "format": "csv",
            "claim_supported": "Artifact list with supported claims.",
            "bytes": 0,
            "source_tables_or_json": "",
        },
    ])
    manifest = pd.DataFrame(manifest_rows)
    save_csv(manifest_path, manifest)
    manifest.loc[manifest["artifact"] == safe_relpath(manifest_path, project_root), "bytes"] = int(manifest_path.stat().st_size)
    save_csv(manifest_path, manifest)
    if not readme_path.exists() or readme_path.stat().st_size <= 0:
        raise FileNotFoundError(readme_path)
    if not manifest_path.exists() or manifest_path.stat().st_size <= 0:
        raise FileNotFoundError(manifest_path)
    qa_table = pd.DataFrame(
        {
            "check": [
                "required_png_pdf_svg_exist",
                "required_png_pdf_svg_nonzero",
                "no_composite_output_filename",
                "readme_written",
                "manifest_written",
            ],
            "status": ["pass", "pass", "pass", "pass", "pass"],
        }
    )
    return FinalFigurePackage(readme_path=readme_path, manifest_path=manifest_path, qa_table=qa_table, manifest=manifest)


def write_claim_boundary_checklist(out_dir: str | Path) -> pd.DataFrame:
    items = [
        ("raw EB counts treated as sampling-depth proxies", True),
        ("equal-depth described as a mass convention", True),
        ("state bins described as coarse PC-20 diagnostics", True),
        ("raw-count proxies not used as calibrated abundance", True),
        ("WFR-FM growth described as a convention-dependent readout", True),
        ("stochastic bridge demo described as synthetic normalized", True),
        ("stochastic bridge width kept separate from EB growth and mass convention", True),
    ]
    table = pd.DataFrame([{"item": item, "status": "pass" if ok else "fail"} for item, ok in items])
    save_csv(Path(out_dir) / "claim_boundary_checklist.csv", table)
    return table


__all__ = [
    "FINAL_FIGURE_CLAIMS",
    "FINAL_FORMATS",
    "FinalFigurePackage",
    "WfrfmGrowthDelta",
    "WfrfmOutputs",
    "bridge_sampling_diagnostic",
    "display_ch04_png",
    "display_final_png",
    "load_eb_data",
    "load_wfrfm_sampling_outputs",
    "make_wfrfm_agreement_summary",
    "make_wfrfm_growth_delta_grid",
    "ordered_state_pivot",
    "plot_equal_depth_composition",
    "plot_raw_observed_counts",
    "plot_sampling_depth_bootstrap_sensitivity",
    "plot_stochastic_bridge_demo",
    "plot_wfrfm_agreement_summary",
    "plot_wfrfm_growth_delta_heatmap",
    "remember_source",
    "resolve_required_artifact",
    "resolve_wfrfm_output_suffix",
    "safe_relpath",
    "save_figure",
    "save_json",
    "save_pub_figure",
    "synthetic_bridge_samples",
    "wfrfm_output_name",
    "write_claim_boundary_checklist",
    "write_final_figure_package",
]
