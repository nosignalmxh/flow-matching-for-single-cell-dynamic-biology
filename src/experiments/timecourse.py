from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.metrics import mmd_rbf, sliced_wasserstein_distance
from ..core.models import VelocityMLP


def set_global_seed(seed: int = 42) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def cluster_mass_metrics(pred_labels, target_labels, all_clusters=None) -> dict:
    pred_labels = np.asarray(pred_labels).astype(str)
    target_labels = np.asarray(target_labels).astype(str)
    if all_clusters is None:
        clusters = sorted(set(pred_labels.tolist()) | set(target_labels.tolist()))
    else:
        clusters = [str(c) for c in list(all_clusters)]
    pred_counts = pd.Series(pred_labels).value_counts()
    target_counts = pd.Series(target_labels).value_counts()
    pred_total = max(int(len(pred_labels)), 1)
    target_total = max(int(len(target_labels)), 1)
    pred_mass = np.asarray([float(pred_counts.get(c, 0) / pred_total) for c in clusters])
    target_mass = np.asarray([float(target_counts.get(c, 0) / target_total) for c in clusters])
    rare_n = max(1, int(np.ceil(0.25 * len(clusters))))
    rare_n = min(rare_n, len(clusters))
    rare_order = sorted(clusters, key=lambda c: (target_counts.get(c, 0), c))
    rare_clusters = rare_order[:rare_n]
    rare_idx = np.asarray([clusters.index(c) for c in rare_clusters], dtype=int)
    return {
        "cluster_mass_l1": float(np.abs(pred_mass - target_mass).sum()),
        "rare_cluster_error": float(np.abs(pred_mass[rare_idx] - target_mass[rare_idx]).sum()),
        "rare_clusters": rare_clusters,
    }


def endpoint_distribution_metrics(X_pred, X_target, kmeans=None, all_clusters=None) -> dict:
    X_pred = np.asarray(X_pred, dtype=np.float32)
    X_target = np.asarray(X_target, dtype=np.float32)
    metrics = {
        "mmd_rbf": float(mmd_rbf(X_pred, X_target)),
        "sliced_w2": float(sliced_wasserstein_distance(X_pred, X_target, n_projections=64)),
        "centroid_l2": float(np.linalg.norm(X_pred.mean(axis=0) - X_target.mean(axis=0))),
    }
    if kmeans is not None:
        pred_labels = kmeans.predict(X_pred)
        target_labels = kmeans.predict(X_target)
        metrics.update(cluster_mass_metrics(pred_labels, target_labels, all_clusters=all_clusters))
    return metrics


def load_eb_ch05(
    path: str | Path = "data/trajectorynet_eb/eb_velocity_v5.npz",
    max_cells_per_time: int | None = 900,
    seed: int = 42,
    n_pc: int = 20,
) -> dict:
    from sklearn.cluster import KMeans

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    pcs_raw = np.asarray(z["pcs"], dtype=np.float32)[:, : int(n_pc)]
    phate = np.asarray(z["phate"], dtype=np.float32)[:, :2]
    labels = np.asarray(z["sample_labels"]).astype(str)
    time_values = {"0": 0.0, "1": 0.25, "2": 0.50, "3": 0.75, "4": 1.0}
    mean = pcs_raw.mean(axis=0)
    std = pcs_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    pcs = ((pcs_raw - mean) / std).astype(np.float32)

    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels).tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        idx = np.flatnonzero(labels == label)
        if max_cells_per_time is not None and len(idx) > int(max_cells_per_time):
            idx = np.sort(rng.choice(idx, size=int(max_cells_per_time), replace=False))
        selected.append(idx)
    idx_all = np.sort(np.concatenate(selected))
    pcs = pcs[idx_all]
    phate = phate[idx_all]
    labels = labels[idx_all]
    kmeans = KMeans(n_clusters=8, random_state=seed, n_init=10).fit(pcs)
    by_time = {label: pcs[labels == label] for label in np.unique(labels)}
    phate_by_time = {label: phate[labels == label] for label in np.unique(labels)}
    counts = pd.Series(labels, name="time").value_counts().sort_index().rename_axis("time").reset_index(name="n_cells")
    return {
        "X": pcs,
        "phate": phate,
        "labels": labels,
        "by_time": by_time,
        "phate_by_time": phate_by_time,
        "time_values": time_values,
        "kmeans": kmeans,
        "counts": counts,
        "path": str(path),
        "pc_mean": mean.astype(np.float32),
        "pc_std": std.astype(np.float32),
    }


def _torch_device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_local_bridge(X0, X1, steps: int, batch_size: int, seed: int, device=None, hidden: int = 128, layers: int = 4):
    import torch

    from ..core.losses import cfm_loss_from_pairs

    device = _torch_device(device)
    rng = np.random.default_rng(seed)
    X0 = _as_float32(X0)
    X1 = _as_float32(X1)
    model = VelocityMLP(x_dim=X0.shape[1], hidden_dim=hidden, hidden_layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    for step in range(1, int(steps) + 1):
        i0 = rng.integers(0, len(X0), size=int(batch_size))
        i1 = rng.integers(0, len(X1), size=int(batch_size))
        x0 = torch.as_tensor(X0[i0], dtype=torch.float32, device=device)
        x1 = torch.as_tensor(X1[i1], dtype=torch.float32, device=device)
        loss = cfm_loss_from_pairs(model, x0, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step == int(steps) or step % max(1, int(steps) // 5) == 0:
            rows.append({"step": step, "loss": float(loss.detach().cpu())})
    return model, pd.DataFrame(rows)


def _train_global_bridge_model(
    by_time: dict,
    pairs: list[tuple[str, str]],
    time_values: dict[str, float],
    steps: int,
    batch_size: int,
    seed: int,
    device=None,
    hidden: int = 128,
    layers: int = 4,
    pair_weights=None,
):
    import torch

    device = _torch_device(device)
    rng = np.random.default_rng(seed)
    pair_prob = None
    if pair_weights is not None:
        pair_prob = np.asarray(pair_weights, dtype=np.float64)
        if pair_prob.shape != (len(pairs),):
            raise ValueError(f"pair_weights must have length {len(pairs)}, got shape {pair_prob.shape}")
        if not np.isfinite(pair_prob).all():
            raise ValueError("pair_weights must be finite")
        if np.any(pair_prob < 0):
            raise ValueError("pair_weights must be nonnegative")
        total = float(pair_prob.sum())
        if total <= 0.0:
            raise ValueError("pair_weights must have positive sum")
        pair_prob = pair_prob / total
    x_dim = next(iter(by_time.values())).shape[1]
    model = VelocityMLP(x_dim=x_dim, hidden_dim=hidden, hidden_layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    for step in range(1, int(steps) + 1):
        if pair_prob is None:
            pair_ids = rng.integers(0, len(pairs), size=int(batch_size))
        else:
            pair_ids = rng.choice(len(pairs), size=int(batch_size), replace=True, p=pair_prob)
        x0_rows = []
        x1_rows = []
        t0_rows = []
        t1_rows = []
        for pair_id in pair_ids:
            a, b = pairs[int(pair_id)]
            A = by_time[str(a)]
            B = by_time[str(b)]
            x0_rows.append(A[rng.integers(0, len(A))])
            x1_rows.append(B[rng.integers(0, len(B))])
            t0_rows.append(float(time_values[str(a)]))
            t1_rows.append(float(time_values[str(b)]))
        x0 = torch.as_tensor(np.asarray(x0_rows), dtype=torch.float32, device=device)
        x1 = torch.as_tensor(np.asarray(x1_rows), dtype=torch.float32, device=device)
        t0 = torch.as_tensor(np.asarray(t0_rows)[:, None], dtype=torch.float32, device=device)
        t1 = torch.as_tensor(np.asarray(t1_rows)[:, None], dtype=torch.float32, device=device)
        alpha = torch.rand((int(batch_size), 1), dtype=torch.float32, device=device)
        t = t0 + alpha * (t1 - t0)
        xs = (1.0 - alpha) * x0 + alpha * x1
        target_v = (x1 - x0) / torch.clamp(t1 - t0, min=1e-6)
        pred = model(xs, t)
        loss = ((pred - target_v) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step == int(steps) or step % max(1, int(steps) // 5) == 0:
            rows.append({"step": step, "loss": float(loss.detach().cpu())})
    return model, pd.DataFrame(rows)


def integrate_velocity(model, x0, t_start: float = 0.0, t_end: float = 1.0, n_steps: int = 32, condition=None, device=None):
    import torch

    device = _torch_device(device)
    model.eval()
    x = torch.as_tensor(x0, dtype=torch.float32, device=device)
    if condition is not None:
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        if cond.ndim == 1:
            cond = cond[None, :].expand(x.shape[0], -1)
    else:
        cond = None
    times = torch.linspace(float(t_start), float(t_end), int(n_steps) + 1, device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(int(n_steps)):
            t = torch.full((x.shape[0], 1), times[i], dtype=torch.float32, device=device)
            dt = times[i + 1] - times[i]
            if cond is None:
                x = x + dt * model(x, t)
            else:
                x = x + dt * model(x, t, cond)
    return x.detach().cpu().numpy().astype(np.float32)


def _local_sequence_rollout(models: dict, X0, sequence, n_steps_per_segment: int, device=None):
    x = _as_float32(X0)
    for pair, frac in sequence:
        model = models[pair]
        steps = max(1, int(round(n_steps_per_segment * float(frac))))
        x = integrate_velocity(model, x, 0.0, float(frac), n_steps=steps, device=device)
    return x


def _global_rollout(model, X0, t_end: float, n_steps: int, device=None):
    return integrate_velocity(model, X0, 0.0, float(t_end), n_steps=n_steps, device=device)


def _boundary_residual_local(model, X0, X1, side: str, n: int = 256, seed: int = 42, device=None) -> float:
    import torch

    from ..core.losses import cfm_loss_from_pairs

    device = _torch_device(device)
    rng = np.random.default_rng(seed)
    i0 = rng.integers(0, len(X0), size=min(int(n), len(X0)))
    i1 = rng.integers(0, len(X1), size=min(int(n), len(X0)))
    m = min(len(i0), len(i1))
    tau = rng.uniform(0.0, 0.05, size=m) if side == "start" else rng.uniform(0.95, 1.0, size=m)
    x0 = torch.as_tensor(X0[i0[:m]], dtype=torch.float32, device=device)
    x1 = torch.as_tensor(X1[i1[:m]], dtype=torch.float32, device=device)
    s = torch.as_tensor(tau[:, None], dtype=torch.float32, device=device)
    loss = cfm_loss_from_pairs(model, x0, x1, s=s)
    return float(loss.detach().cpu())


def run_eb_pairwise_vs_shared(
    eb: dict,
    training_steps: int,
    batch_size: int = 256,
    nfe: int = 32,
    seed: int = 42,
    device=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    by_time = eb["by_time"]
    time_values = eb["time_values"]
    adjacent_pairs = [("0", "1"), ("1", "3"), ("3", "4")]
    skip_pairs = adjacent_pairs + [("0", "3"), ("1", "4"), ("0", "4")]
    pair_steps = max(1, int(training_steps) // len(adjacent_pairs))
    pair_models = {}
    histories = {}
    for i, pair in enumerate(adjacent_pairs):
        model, hist = _train_local_bridge(
            by_time[pair[0]], by_time[pair[1]], pair_steps, batch_size, seed + 10 + i, device=device
        )
        pair_models[pair] = model
        histories[f"pairwise_{pair[0]}_{pair[1]}"] = hist
    shared_adj, hist_adj = _train_global_bridge_model(
        by_time, adjacent_pairs, time_values, training_steps, batch_size, seed + 30, device=device
    )
    shared_skip, hist_skip = _train_global_bridge_model(
        by_time, skip_pairs, time_values, training_steps, batch_size, seed + 40, device=device
    )
    histories["shared_adjacent_only"] = hist_adj
    histories["shared_adjacent_skip"] = hist_skip

    X0 = by_time["0"]
    predictions = {
        "pairwise_local_bridges": {
            "hidden_t2": _local_sequence_rollout(
                pair_models, X0, [((("0", "1")), 1.0), ((("1", "3")), 0.5)], nfe, device=device
            ),
            "seen_t4": _local_sequence_rollout(
                pair_models,
                X0,
                [((("0", "1")), 1.0), ((("1", "3")), 1.0), ((("3", "4")), 1.0)],
                nfe,
                device=device,
            ),
        },
        "shared_adjacent_only": {
            "hidden_t2": _global_rollout(shared_adj, X0, 0.5, nfe * 2, device=device),
            "seen_t4": _global_rollout(shared_adj, X0, 1.0, nfe * 4, device=device),
        },
        "shared_adjacent_skip": {
            "hidden_t2": _global_rollout(shared_skip, X0, 0.5, nfe * 2, device=device),
            "seen_t4": _global_rollout(shared_skip, X0, 1.0, nfe * 4, device=device),
        },
    }

    rows = []
    targets = {"hidden_t2": by_time["2"], "seen_t4": by_time["4"]}
    for method, pred_by_target in predictions.items():
        for target_name, X_pred in pred_by_target.items():
            metrics = endpoint_distribution_metrics(X_pred, targets[target_name], eb["kmeans"], all_clusters=np.arange(8))
            rows.append(
                {
                    "experiment": "EB multi-timepoint",
                    "method": method,
                    "target": target_name,
                    "training_steps_total": int(training_steps) if method != "pairwise_local_bridges" else int(pair_steps * 3),
                    "steps_per_pair": int(pair_steps) if method == "pairwise_local_bridges" else 0,
                    **{k: v for k, v in metrics.items() if k != "rare_clusters"},
                    "rare_clusters": ",".join(metrics["rare_clusters"]),
                }
            )

    diag_rows = []
    boundary_specs = [("t1", ("0", "1"), ("1", "3"), "1", 0.25), ("t3", ("1", "3"), ("3", "4"), "3", 0.75)]
    for label, left_pair, right_pair, boundary_time, t_global in boundary_specs:
        Xb = by_time[boundary_time]
        import torch

        device_obj = _torch_device(device)
        xb = torch.as_tensor(Xb, dtype=torch.float32, device=device_obj)
        with torch.no_grad():
            v_left = pair_models[left_pair](xb, torch.ones((len(Xb), 1), device=device_obj)).detach().cpu().numpy()
            v_right = pair_models[right_pair](xb, torch.zeros((len(Xb), 1), device=device_obj)).detach().cpu().numpy()
        diag_rows.append(
            {
                "method": "pairwise_local_bridges",
                "boundary": label,
                "velocity_jump_mean_l2": float(np.linalg.norm(v_left - v_right, axis=1).mean()),
                "boundary_start_residual": _boundary_residual_local(
                    pair_models[right_pair], by_time[right_pair[0]], by_time[right_pair[1]], "start", seed=seed, device=device
                ),
                "boundary_end_residual": _boundary_residual_local(
                    pair_models[left_pair], by_time[left_pair[0]], by_time[left_pair[1]], "end", seed=seed + 1, device=device
                ),
            }
        )
        for method, model in [("shared_adjacent_only", shared_adj), ("shared_adjacent_skip", shared_skip)]:
            with torch.no_grad():
                xbt = torch.as_tensor(Xb, dtype=torch.float32, device=device_obj)
                t_left = torch.full((len(Xb), 1), max(0.0, t_global - 0.01), device=device_obj)
                t_right = torch.full((len(Xb), 1), min(1.0, t_global + 0.01), device=device_obj)
                vl = model(xbt, t_left).detach().cpu().numpy()
                vr = model(xbt, t_right).detach().cpu().numpy()
            diag_rows.append(
                {
                    "method": method,
                    "boundary": label,
                    "velocity_jump_mean_l2": float(np.linalg.norm(vl - vr, axis=1).mean()),
                    "boundary_start_residual": 0.0,
                    "boundary_end_residual": 0.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(diag_rows), {"predictions": predictions, "histories": histories}


def _record_from_row(row: pd.Series, metric: str) -> dict:
    return {
        "variant": str(row["variant"]),
        "variant_family": str(row["variant_family"]),
        "training_steps_total": int(row["training_steps_total"]),
        metric: float(row[metric]),
        "seen_t4_sliced_w2": float(row["sliced_w2"]),
        "seen_t4_centroid_l2": float(row["centroid_l2"]),
    }


def _summarize_eb_skip_pair_ablation(metrics: pd.DataFrame, diagnostics: pd.DataFrame, seeds: list[int]) -> dict:
    metric_cols = ["mmd_rbf", "sliced_w2", "centroid_l2"]
    grouped = (
        metrics.groupby(["variant", "variant_family", "training_steps_total", "target"], observed=False)[metric_cols]
        .mean()
        .reset_index()
    )
    seen = grouped[grouped["target"].eq("seen_t4")].copy()
    hidden = grouped[grouped["target"].eq("hidden_t2")].copy()
    adjacent_seen = seen[seen["variant_family"].eq("adjacent_only")].copy()
    skip_seen = seen[seen["variant_family"].eq("skip")].copy()

    best_adj_sliced = adjacent_seen.sort_values("sliced_w2", kind="mergesort").iloc[0]
    best_skip_sliced = skip_seen.sort_values("sliced_w2", kind="mergesort").iloc[0]
    best_adj_centroid = adjacent_seen.sort_values("centroid_l2", kind="mergesort").iloc[0]
    best_skip_centroid = skip_seen.sort_values("centroid_l2", kind="mergesort").iloc[0]

    baseline_seen = seen[seen["variant"].eq("shared_adjacent_only_6000")].iloc[0]
    baseline_hidden = hidden[hidden["variant"].eq("shared_adjacent_only_6000")].iloc[0]
    skip_beats_baseline_rows = skip_seen[
        (skip_seen["sliced_w2"] < float(baseline_seen["sliced_w2"]))
        & (skip_seen["centroid_l2"] < float(baseline_seen["centroid_l2"]))
    ].copy()
    hidden_by_variant = hidden.set_index("variant")
    hidden_not_badly_degraded_variants = []
    skip_beats_adjacent_only_6000_seen_metrics_only = bool(len(skip_beats_baseline_rows) > 0)
    if skip_beats_adjacent_only_6000_seen_metrics_only:
        for variant in skip_beats_baseline_rows["variant"]:
            hidden_row = hidden_by_variant.loc[str(variant)]
            if (
                float(hidden_row["mmd_rbf"]) <= 1.2 * float(baseline_hidden["mmd_rbf"])
                and float(hidden_row["sliced_w2"]) <= 1.2 * float(baseline_hidden["sliced_w2"])
            ):
                hidden_not_badly_degraded_variants.append(str(variant))
    skip_beats_adjacent_only_6000 = bool(len(hidden_not_badly_degraded_variants) > 0)

    same_budget_comparisons = []
    skip_beats_same_budget_adjacent_only = False
    for _, skip_row in skip_seen.iterrows():
        same_budget_adj = adjacent_seen[
            adjacent_seen["training_steps_total"].eq(int(skip_row["training_steps_total"]))
        ].copy()
        if same_budget_adj.empty:
            continue
        adj_row = same_budget_adj.sort_values("sliced_w2", kind="mergesort").iloc[0]
        beats = bool(
            float(skip_row["sliced_w2"]) < float(adj_row["sliced_w2"])
            and float(skip_row["centroid_l2"]) < float(adj_row["centroid_l2"])
        )
        skip_beats_same_budget_adjacent_only = bool(skip_beats_same_budget_adjacent_only or beats)
        same_budget_comparisons.append(
            {
                "skip_variant": str(skip_row["variant"]),
                "adjacent_variant": str(adj_row["variant"]),
                "training_steps_total": int(skip_row["training_steps_total"]),
                "skip_seen_sliced_w2": float(skip_row["sliced_w2"]),
                "adjacent_seen_sliced_w2": float(adj_row["sliced_w2"]),
                "skip_seen_centroid_l2": float(skip_row["centroid_l2"]),
                "adjacent_seen_centroid_l2": float(adj_row["centroid_l2"]),
                "skip_beats_adjacent_on_both_seen_metrics": beats,
            }
        )

    diag_mean = (
        diagnostics.groupby(["variant", "variant_family"], observed=False)["velocity_jump_mean_l2"]
        .mean()
        .reset_index()
    )
    baseline_jump = float(diag_mean.loc[diag_mean["variant"].eq("shared_adjacent_only_6000"), "velocity_jump_mean_l2"].iloc[0])
    best_skip_jump = diag_mean[diag_mean["variant_family"].eq("skip")].sort_values("velocity_jump_mean_l2").iloc[0]
    skip_improves_velocity_jump = bool(float(best_skip_jump["velocity_jump_mean_l2"]) < baseline_jump)

    if skip_beats_adjacent_only_6000:
        claim = "skip pairs improve long-horizon rollout under weighted/exposure-matched sampling"
    elif skip_improves_velocity_jump and not skip_beats_adjacent_only_6000_seen_metrics_only:
        claim = "skip pairs improve boundary consistency but not rollout metrics"
    else:
        claim = "skip pairs do not provide stable benefit under current coupling/budget"

    return {
        "seeds": [int(s) for s in seeds],
        "best_adjacent_only_seen_sliced_w2": _record_from_row(best_adj_sliced, "sliced_w2"),
        "best_skip_seen_sliced_w2": _record_from_row(best_skip_sliced, "sliced_w2"),
        "best_adjacent_only_seen_centroid_l2": _record_from_row(best_adj_centroid, "centroid_l2"),
        "best_skip_seen_centroid_l2": _record_from_row(best_skip_centroid, "centroid_l2"),
        "skip_beats_adjacent_only_6000": skip_beats_adjacent_only_6000,
        "skip_beats_adjacent_only_6000_seen_metrics_only": skip_beats_adjacent_only_6000_seen_metrics_only,
        "skip_beats_same_budget_adjacent_only": skip_beats_same_budget_adjacent_only,
        "same_budget_comparisons": same_budget_comparisons,
        "baseline_adjacent_only_6000_hidden_t2": {
            "mmd_rbf": float(baseline_hidden["mmd_rbf"]),
            "sliced_w2": float(baseline_hidden["sliced_w2"]),
        },
        "hidden_not_badly_degraded_threshold": "skip hidden_t2 mmd_rbf and sliced_w2 <= 1.2x shared_adjacent_only_6000",
        "hidden_not_badly_degraded_skip_variants_that_beat_adjacent_only_6000": hidden_not_badly_degraded_variants,
        "best_skip_velocity_jump_mean_l2": {
            "variant": str(best_skip_jump["variant"]),
            "velocity_jump_mean_l2": float(best_skip_jump["velocity_jump_mean_l2"]),
        },
        "baseline_adjacent_only_6000_velocity_jump_mean_l2": baseline_jump,
        "recommended_main_text_claim": claim,
    }


def run_eb_skip_pair_ablation(
    eb: dict,
    batch_size: int = 256,
    nfe: int = 32,
    seed: int = 42,
    seeds: list[int] | tuple[int, ...] | None = None,
    device=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    by_time = eb["by_time"]
    time_values = eb["time_values"]
    adjacent_pairs = [("0", "1"), ("1", "3"), ("3", "4")]
    skip_pairs = adjacent_pairs + [("0", "3"), ("1", "4"), ("0", "4")]
    medium_skip_pairs = adjacent_pairs + [("0", "3"), ("1", "4")]
    if seeds is None:
        seeds = [int(seed)]
    else:
        seeds = [int(s) for s in seeds]

    variants = [
        {
            "variant": "shared_adjacent_only_6000",
            "variant_family": "adjacent_only",
            "pairs": adjacent_pairs,
            "training_steps_total": 6000,
            "sampling": "uniform",
            "pair_weights": None,
        },
        {
            "variant": "shared_adjacent_only_9000",
            "variant_family": "adjacent_only",
            "pairs": adjacent_pairs,
            "training_steps_total": 9000,
            "sampling": "uniform",
            "pair_weights": None,
        },
        {
            "variant": "shared_adjacent_only_12000",
            "variant_family": "adjacent_only",
            "pairs": adjacent_pairs,
            "training_steps_total": 12000,
            "sampling": "uniform",
            "pair_weights": None,
        },
        {
            "variant": "shared_skip_uniform_6000",
            "variant_family": "skip",
            "pairs": skip_pairs,
            "training_steps_total": 6000,
            "sampling": "uniform",
            "pair_weights": None,
        },
        {
            "variant": "shared_skip_uniform_12000",
            "variant_family": "skip",
            "pairs": skip_pairs,
            "training_steps_total": 12000,
            "sampling": "uniform",
            "pair_weights": None,
        },
        {
            "variant": "shared_skip_adj2_skip1_9000",
            "variant_family": "skip",
            "pairs": skip_pairs,
            "training_steps_total": 9000,
            "sampling": "adjacent_total_2_3_skip_total_1_3",
            "pair_weights": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        },
        {
            "variant": "shared_skip_adj3_skip1_8000",
            "variant_family": "skip",
            "pairs": skip_pairs,
            "training_steps_total": 8000,
            "sampling": "adjacent_total_3_4_skip_total_1_4",
            "pair_weights": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0],
        },
        {
            "variant": "shared_skip_medium_only_9000",
            "variant_family": "skip",
            "pairs": medium_skip_pairs,
            "training_steps_total": 9000,
            "sampling": "adjacent_total_2_3_medium_skip_total_1_3",
            "pair_weights": [2.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0, 1.0 / 6.0, 1.0 / 6.0],
        },
    ]

    rows = []
    diag_rows = []
    histories = {}
    X0 = by_time["0"]
    targets = {"hidden_t2": by_time["2"], "seen_t4": by_time["4"]}
    boundary_specs = [("t1", "1", 0.25), ("t3", "3", 0.75)]

    for seed_idx, run_seed in enumerate(seeds):
        for variant_idx, spec in enumerate(variants):
            model_seed = int(run_seed) + 1000 + variant_idx * 97
            model, hist = _train_global_bridge_model(
                by_time,
                spec["pairs"],
                time_values,
                int(spec["training_steps_total"]),
                batch_size,
                model_seed,
                device=device,
                pair_weights=spec["pair_weights"],
            )
            histories[(int(run_seed), spec["variant"])] = hist
            predictions = {
                "hidden_t2": _global_rollout(model, X0, 0.5, nfe * 2, device=device),
                "seen_t4": _global_rollout(model, X0, 1.0, nfe * 4, device=device),
            }
            pair_weights = spec["pair_weights"]
            if pair_weights is None:
                pair_weights_repr = "uniform"
            else:
                weights = np.asarray(pair_weights, dtype=float)
                pair_weights_repr = ",".join(f"{w:.6g}" for w in (weights / weights.sum()))
            for target_name, X_pred in predictions.items():
                metrics = endpoint_distribution_metrics(X_pred, targets[target_name])
                rows.append(
                    {
                        "experiment": "EB skip-pair ablation",
                        "seed": int(run_seed),
                        "variant": spec["variant"],
                        "variant_family": spec["variant_family"],
                        "target": target_name,
                        "training_steps_total": int(spec["training_steps_total"]),
                        "sampling": spec["sampling"],
                        "pairs": ";".join(f"{a}-{b}" for a, b in spec["pairs"]),
                        "pair_weights": pair_weights_repr,
                        "mmd_rbf": float(metrics["mmd_rbf"]),
                        "sliced_w2": float(metrics["sliced_w2"]),
                        "centroid_l2": float(metrics["centroid_l2"]),
                    }
                )

            import torch

            device_obj = _torch_device(device)
            with torch.no_grad():
                for boundary, boundary_time, t_global in boundary_specs:
                    Xb = by_time[boundary_time]
                    xb = torch.as_tensor(Xb, dtype=torch.float32, device=device_obj)
                    t_left = torch.full((len(Xb), 1), max(0.0, t_global - 0.01), device=device_obj)
                    t_right = torch.full((len(Xb), 1), min(1.0, t_global + 0.01), device=device_obj)
                    vl = model(xb, t_left).detach().cpu().numpy()
                    vr = model(xb, t_right).detach().cpu().numpy()
                    diag_rows.append(
                        {
                            "experiment": "EB skip-pair ablation",
                            "seed": int(run_seed),
                            "variant": spec["variant"],
                            "variant_family": spec["variant_family"],
                            "boundary": boundary,
                            "training_steps_total": int(spec["training_steps_total"]),
                            "sampling": spec["sampling"],
                            "velocity_jump_mean_l2": float(np.linalg.norm(vl - vr, axis=1).mean()),
                        }
                    )

    metrics_df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)
    summary = _summarize_eb_skip_pair_ablation(metrics_df, diag_df, seeds)
    return metrics_df, diag_df, {"decision_summary": summary, "histories": histories}


def run_eb_section51_main_suite(
    eb: dict,
    batch_size: int = 256,
    nfe: int = 32,
    seed: int = 42,
    seeds: list[int] | tuple[int, ...] | None = None,
    device=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    by_time = eb["by_time"]
    time_values = eb["time_values"]
    adjacent_pairs = [("0", "1"), ("1", "3"), ("3", "4")]
    skip_pairs = adjacent_pairs + [("0", "3"), ("1", "4"), ("0", "4")]
    if seeds is None:
        seeds = [int(seed)]
    else:
        seeds = [int(s) for s in seeds]

    shared_specs = [
        {
            "variant": "shared_adjacent_only_6000",
            "variant_family": "shared_adjacent",
            "pairs": adjacent_pairs,
            "training_steps_total": 6000,
            "sampling": "uniform_adjacent",
            "pair_weights": None,
            "seed_offset": 30,
        },
        {
            "variant": "shared_skip_uniform_6000",
            "variant_family": "shared_skip",
            "pairs": skip_pairs,
            "training_steps_total": 6000,
            "sampling": "uniform_adjacent_skip",
            "pair_weights": None,
            "seed_offset": 40,
        },
        {
            "variant": "shared_skip_adj2_skip1_9000",
            "variant_family": "shared_skip",
            "pairs": skip_pairs,
            "training_steps_total": 9000,
            "sampling": "adjacent_total_2_3_skip_total_1_3",
            "pair_weights": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            "seed_offset": 50,
        },
    ]

    rows = []
    diag_rows = []
    histories = {}
    targets = {"hidden_t2": by_time["2"], "seen_t4": by_time["4"]}
    boundary_specs = [("t1", "1", 0.25), ("t3", "3", 0.75)]
    X0 = by_time["0"]

    for run_seed in seeds:
        run_seed = int(run_seed)
        set_global_seed(seed)
        pair_steps = 6000 // len(adjacent_pairs)
        pair_models = {}
        for pair_idx, pair in enumerate(adjacent_pairs):
            model, hist = _train_local_bridge(
                by_time[pair[0]],
                by_time[pair[1]],
                pair_steps,
                batch_size,
                run_seed + 10 + pair_idx,
                device=device,
            )
            pair_models[pair] = model
            histories[(run_seed, f"pairwise_{pair[0]}_{pair[1]}")] = hist

        predictions = {
            "pairwise_local_bridges_6000": {
                "variant_family": "pairwise",
                "training_steps_total": int(pair_steps * len(adjacent_pairs)),
                "sampling": "local_pairwise",
                "pairs": adjacent_pairs,
                "pair_weights": "",
                "hidden_t2": _local_sequence_rollout(
                    pair_models,
                    X0,
                    [((("0", "1")), 1.0), ((("1", "3")), 0.5)],
                    nfe,
                    device=device,
                ),
                "seen_t4": _local_sequence_rollout(
                    pair_models,
                    X0,
                    [((("0", "1")), 1.0), ((("1", "3")), 1.0), ((("3", "4")), 1.0)],
                    nfe,
                    device=device,
                ),
            }
        }

        shared_models = {}
        for spec in shared_specs:
            model, hist = _train_global_bridge_model(
                by_time,
                spec["pairs"],
                time_values,
                int(spec["training_steps_total"]),
                batch_size,
                run_seed + int(spec["seed_offset"]),
                device=device,
                pair_weights=spec["pair_weights"],
            )
            histories[(run_seed, spec["variant"])] = hist
            shared_models[spec["variant"]] = model
            pair_weights = spec["pair_weights"]
            if pair_weights is None:
                pair_weights_repr = "uniform"
            else:
                weights = np.asarray(pair_weights, dtype=float)
                pair_weights_repr = ",".join(f"{w:.6g}" for w in (weights / weights.sum()))
            predictions[spec["variant"]] = {
                "variant_family": spec["variant_family"],
                "training_steps_total": int(spec["training_steps_total"]),
                "sampling": spec["sampling"],
                "pairs": spec["pairs"],
                "pair_weights": pair_weights_repr,
                "hidden_t2": _global_rollout(model, X0, 0.5, nfe * 2, device=device),
                "seen_t4": _global_rollout(model, X0, 1.0, nfe * 4, device=device),
            }

        for variant, payload in predictions.items():
            for target_name in ["hidden_t2", "seen_t4"]:
                metrics = endpoint_distribution_metrics(payload[target_name], targets[target_name])
                rows.append(
                    {
                        "experiment": "EB Section 5.1 main suite",
                        "seed": int(run_seed),
                        "variant": variant,
                        "variant_family": payload["variant_family"],
                        "target": target_name,
                        "training_steps_total": int(payload["training_steps_total"]),
                        "sampling": payload["sampling"],
                        "pairs": ";".join(f"{a}-{b}" for a, b in payload["pairs"]),
                        "pair_weights": payload["pair_weights"],
                        "mmd_rbf": float(metrics["mmd_rbf"]),
                        "sliced_w2": float(metrics["sliced_w2"]),
                        "centroid_l2": float(metrics["centroid_l2"]),
                    }
                )

        import torch

        device_obj = _torch_device(device)
        for boundary, boundary_time, t_global in boundary_specs:
            Xb = by_time[boundary_time]
            xb = torch.as_tensor(Xb, dtype=torch.float32, device=device_obj)
            left_pair = ("0", "1") if boundary == "t1" else ("1", "3")
            right_pair = ("1", "3") if boundary == "t1" else ("3", "4")
            with torch.no_grad():
                v_left = pair_models[left_pair](xb, torch.ones((len(Xb), 1), device=device_obj)).detach().cpu().numpy()
                v_right = pair_models[right_pair](xb, torch.zeros((len(Xb), 1), device=device_obj)).detach().cpu().numpy()
            diag_rows.append(
                {
                    "experiment": "EB Section 5.1 main suite",
                    "seed": int(run_seed),
                    "variant": "pairwise_local_bridges_6000",
                    "variant_family": "pairwise",
                    "boundary": boundary,
                    "training_steps_total": int(pair_steps * len(adjacent_pairs)),
                    "sampling": "local_pairwise",
                    "velocity_jump_mean_l2": float(np.linalg.norm(v_left - v_right, axis=1).mean()),
                }
            )
            for spec in shared_specs:
                model = shared_models[spec["variant"]]
                with torch.no_grad():
                    t_left = torch.full((len(Xb), 1), max(0.0, t_global - 0.01), device=device_obj)
                    t_right = torch.full((len(Xb), 1), min(1.0, t_global + 0.01), device=device_obj)
                    vl = model(xb, t_left).detach().cpu().numpy()
                    vr = model(xb, t_right).detach().cpu().numpy()
                diag_rows.append(
                    {
                        "experiment": "EB Section 5.1 main suite",
                        "seed": int(run_seed),
                        "variant": spec["variant"],
                        "variant_family": spec["variant_family"],
                        "boundary": boundary,
                        "training_steps_total": int(spec["training_steps_total"]),
                        "sampling": spec["sampling"],
                        "velocity_jump_mean_l2": float(np.linalg.norm(vl - vr, axis=1).mean()),
                    }
                )

    metrics_df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)
    summary_df, diag_summary_df = summarize_eb_section51_main_suite(metrics_df, diag_df)
    main_text_df, summary_payload = build_ch05_section51_main_text_results(summary_df, diag_summary_df)
    return metrics_df, diag_df, {
        "histories": histories,
        "summary": summary_df,
        "diag_summary": diag_summary_df,
        "main_text_results": main_text_df,
        "summary_payload": summary_payload,
    }


def summarize_eb_section51_main_suite(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["mmd_rbf", "sliced_w2", "centroid_l2"]
    group_cols = ["variant", "variant_family", "target", "training_steps_total", "sampling"]
    summary = (
        metrics.groupby(group_cols, observed=False)
        .agg(
            mmd_rbf_mean=("mmd_rbf", "mean"),
            mmd_rbf_std=("mmd_rbf", "std"),
            sliced_w2_mean=("sliced_w2", "mean"),
            sliced_w2_std=("sliced_w2", "std"),
            centroid_l2_mean=("centroid_l2", "mean"),
            centroid_l2_std=("centroid_l2", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary = summary[
        group_cols
        + [
            "mmd_rbf_mean",
            "mmd_rbf_std",
            "sliced_w2_mean",
            "sliced_w2_std",
            "centroid_l2_mean",
            "centroid_l2_std",
            "n_seeds",
        ]
    ].sort_values(["target", "variant"]).reset_index(drop=True)

    diag_group_cols = ["variant", "variant_family", "boundary", "training_steps_total", "sampling"]
    diag_summary = (
        diagnostics.groupby(diag_group_cols, observed=False)
        .agg(
            velocity_jump_mean_l2_mean=("velocity_jump_mean_l2", "mean"),
            velocity_jump_mean_l2_std=("velocity_jump_mean_l2", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    diag_summary = diag_summary[
        diag_group_cols + ["velocity_jump_mean_l2_mean", "velocity_jump_mean_l2_std", "n_seeds"]
    ].sort_values(["boundary", "variant"]).reset_index(drop=True)
    return summary, diag_summary


def _display_metric(value: float, metric: str) -> str:
    if pd.isna(value):
        return ""
    if metric == "mmd_rbf":
        return f"{float(value):.4f}"
    if metric in {"sliced_w2", "centroid_l2"}:
        return f"{float(value):.3f}"
    if metric in {"velocity_jump_mean_l2", "velocity_jump_mean_l2_mean_over_hand_offs"}:
        return f"{float(value):.2f}"
    return f"{float(value):.4g}"


def _summary_row(summary: pd.DataFrame, variant: str, target: str) -> pd.Series:
    rows = summary[summary["variant"].eq(variant) & summary["target"].eq(target)]
    if rows.empty:
        raise KeyError(f"Missing Section 5.1 summary row: variant={variant}, target={target}")
    return rows.iloc[0]


def _diag_summary_row(diag_summary: pd.DataFrame, variant: str, boundary: str) -> pd.Series:
    rows = diag_summary[diag_summary["variant"].eq(variant) & diag_summary["boundary"].eq(boundary)]
    if rows.empty:
        raise KeyError(f"Missing Section 5.1 diagnostic row: variant={variant}, boundary={boundary}")
    return rows.iloc[0]


def build_ch05_section51_main_text_results(
    summary: pd.DataFrame,
    diag_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    def add_metric_row(claim_part, row, target_or_boundary, metric, note=""):
        value = float(row[f"{metric}_mean"])
        std = row.get(f"{metric}_std", np.nan)
        rows.append(
            {
                "section": "5.1",
                "claim_part": claim_part,
                "source_table": "tab_5_1_main_suite_summary.csv",
                "source_row_id": f"{row['variant']}|{target_or_boundary}|{metric}",
                "method_or_variant": str(row["variant"]),
                "target_or_boundary": str(target_or_boundary),
                "metric": metric,
                "value": value,
                "std": float(std) if not pd.isna(std) else np.nan,
                "n_seeds": int(row["n_seeds"]),
                "display_value": _display_metric(value, metric),
                "note": note,
            }
        )

    def add_diag_row(claim_part, row, boundary, note=""):
        metric = "velocity_jump_mean_l2"
        value = float(row[f"{metric}_mean"])
        std = row.get(f"{metric}_std", np.nan)
        rows.append(
            {
                "section": "5.1",
                "claim_part": claim_part,
                "source_table": "tab_5_1_main_suite_diag_summary.csv",
                "source_row_id": f"{row['variant']}|{boundary}|{metric}",
                "method_or_variant": str(row["variant"]),
                "target_or_boundary": str(boundary),
                "metric": metric,
                "value": value,
                "std": float(std) if not pd.isna(std) else np.nan,
                "n_seeds": int(row["n_seeds"]),
                "display_value": _display_metric(value, metric),
                "note": note,
            }
        )

    def add_diag_mean_row(variant: str, t1_row: pd.Series, t3_row: pd.Series) -> float:
        metric = "velocity_jump_mean_l2_mean_over_hand_offs"
        value = 0.5 * (
            float(t1_row["velocity_jump_mean_l2_mean"]) + float(t3_row["velocity_jump_mean_l2_mean"])
        )
        rows.append(
            {
                "section": "5.1",
                "claim_part": "velocity_jump_diagnostic",
                "source_table": "tab_5_1_main_suite_diag_summary.csv",
                "source_row_id": f"{variant}|t1_t3_mean|velocity_jump_mean_l2",
                "method_or_variant": variant,
                "target_or_boundary": "t1_t3_mean",
                "metric": metric,
                "value": float(value),
                "std": np.nan,
                "n_seeds": int(min(t1_row["n_seeds"], t3_row["n_seeds"])),
                "display_value": _display_metric(value, metric),
                "note": "Mean of t1 and t3 hand-off jump means from tab_5_1_main_suite_diag_summary.csv.",
            }
        )
        return float(value)

    pair_hidden = _summary_row(summary, "pairwise_local_bridges_6000", "hidden_t2")
    adj_hidden = _summary_row(summary, "shared_adjacent_only_6000", "hidden_t2")
    skip_hidden = _summary_row(summary, "shared_skip_adj2_skip1_9000", "hidden_t2")
    adj_seen = _summary_row(summary, "shared_adjacent_only_6000", "seen_t4")
    skip_seen = _summary_row(summary, "shared_skip_adj2_skip1_9000", "seen_t4")

    for row in [pair_hidden, adj_hidden, skip_hidden]:
        for metric in ["mmd_rbf", "sliced_w2"]:
            add_metric_row("hidden_t2_main_comparison", row, "hidden_t2", metric)

    for row in [adj_seen, skip_seen]:
        for metric in ["sliced_w2", "centroid_l2"]:
            add_metric_row("seen_t4_long_horizon", row, "seen_t4", metric)

    for metric in ["sliced_w2", "centroid_l2"]:
        adj_value = float(adj_seen[f"{metric}_mean"])
        skip_value = float(skip_seen[f"{metric}_mean"])
        pct_lower = 100.0 * (adj_value - skip_value) / adj_value
        rows.append(
            {
                "section": "5.1",
                "claim_part": "seen_t4_long_horizon",
                "source_table": "tab_5_1_main_suite_summary.csv",
                "source_row_id": f"shared_skip_adj2_skip1_9000_vs_shared_adjacent_only_6000|seen_t4|{metric}",
                "method_or_variant": "relative_improvement_skip_2to1_vs_adjacent_only",
                "target_or_boundary": "seen_t4",
                "metric": f"{metric}_pct_lower",
                "value": float(pct_lower),
                "std": np.nan,
                "n_seeds": int(min(adj_seen["n_seeds"], skip_seen["n_seeds"])),
                "display_value": f"{pct_lower:.1f}% lower",
                "note": "Computed from mean values in tab_5_1_main_suite_summary.csv.",
            }
        )

    for row in [adj_hidden, skip_hidden]:
        for metric in ["mmd_rbf", "sliced_w2"]:
            add_metric_row("hidden_t2_skip_tradeoff", row, "hidden_t2", metric)

    velocity_variants = [
        "pairwise_local_bridges_6000",
        "shared_adjacent_only_6000",
        "shared_skip_adj2_skip1_9000",
    ]
    velocity_rows = {}
    velocity_means = {}
    for variant in velocity_variants:
        for boundary in ["t1", "t3"]:
            row = _diag_summary_row(diag_summary, variant, boundary)
            velocity_rows[(variant, boundary)] = row
            add_diag_row("velocity_jump_diagnostic", row, boundary)
        velocity_means[variant] = add_diag_mean_row(
            variant,
            velocity_rows[(variant, "t1")],
            velocity_rows[(variant, "t3")],
        )

    main_text = pd.DataFrame(rows)
    does_skip_beat_pairwise_hidden = bool(
        float(skip_hidden["mmd_rbf_mean"]) < float(pair_hidden["mmd_rbf_mean"])
        and float(skip_hidden["sliced_w2_mean"]) < float(pair_hidden["sliced_w2_mean"])
    )
    does_skip_beat_adj_seen = bool(
        float(skip_seen["sliced_w2_mean"]) < float(adj_seen["sliced_w2_mean"])
        and float(skip_seen["centroid_l2_mean"]) < float(adj_seen["centroid_l2_mean"])
    )
    does_skip_degrade_hidden_vs_adj = bool(
        float(skip_hidden["mmd_rbf_mean"]) > float(adj_hidden["mmd_rbf_mean"])
        or float(skip_hidden["sliced_w2_mean"]) > float(adj_hidden["sliced_w2_mean"])
    )
    def rounded_diag(variant: str, boundary: str) -> float:
        return round(float(velocity_rows[(variant, boundary)]["velocity_jump_mean_l2_mean"]), 2)

    def rounded_mean(variant: str) -> float:
        return round(float(velocity_means[variant]), 2)

    velocity_jump_summary = {
        "pairwise_t1": rounded_diag("pairwise_local_bridges_6000", "t1"),
        "pairwise_t3": rounded_diag("pairwise_local_bridges_6000", "t3"),
        "pairwise_mean_t1_t3": rounded_mean("pairwise_local_bridges_6000"),
        "shared_adjacent_only_t1": rounded_diag("shared_adjacent_only_6000", "t1"),
        "shared_adjacent_only_t3": rounded_diag("shared_adjacent_only_6000", "t3"),
        "shared_adjacent_only_mean_t1_t3": rounded_mean("shared_adjacent_only_6000"),
        "shared_skip_2to1_t1": rounded_diag("shared_skip_adj2_skip1_9000", "t1"),
        "shared_skip_2to1_t3": rounded_diag("shared_skip_adj2_skip1_9000", "t3"),
        "shared_skip_2to1_mean_t1_t3": rounded_mean("shared_skip_adj2_skip1_9000"),
        "interpretation": (
            "Pairwise local bridges show large hand-off jumps; shared fields reduce average hand-off "
            "jump, with skip variants strongest at the late hand-off depending on sampling. This is a "
            "hand-off velocity discontinuity diagnostic, not proof that learned dynamics are more "
            "biologically faithful."
        ),
    }

    payload = {
        "hidden_t2_main_comparison": main_text[main_text["claim_part"].eq("hidden_t2_main_comparison")].to_dict(orient="records"),
        "seen_t4_long_horizon": main_text[main_text["claim_part"].eq("seen_t4_long_horizon")].to_dict(orient="records"),
        "hidden_t2_skip_tradeoff": main_text[main_text["claim_part"].eq("hidden_t2_skip_tradeoff")].to_dict(orient="records"),
        "velocity_jump_diagnostic": velocity_jump_summary,
        "velocity_jump_diagnostic_rows": main_text[main_text["claim_part"].eq("velocity_jump_diagnostic")].to_dict(orient="records"),
        "claim_boundary": (
            "shared field improves hidden-time recovery; weighted skip supervision improves seen endpoint "
            "long-horizon rollout but may degrade hidden-time recovery, so skip is a long-horizon endpoint "
            "constraint rather than a uniformly better dynamics model."
        ),
        "does_skip_2to1_beat_pairwise_on_hidden_t2": does_skip_beat_pairwise_hidden,
        "does_skip_2to1_beat_adjacent_only_on_seen_t4": does_skip_beat_adj_seen,
        "does_skip_2to1_degrade_hidden_t2_vs_adjacent_only": does_skip_degrade_hidden_vs_adj,
    }
    return main_text, payload
