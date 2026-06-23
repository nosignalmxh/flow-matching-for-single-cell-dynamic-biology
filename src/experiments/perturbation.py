from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd

from ..evaluation.metrics import distribution_readout_metrics
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


def _torch_device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@dataclass
class SciplexConditionEncoder:
    train_compounds: list[str]
    rdkit_by_compound: dict[str, np.ndarray]
    rdkit_dim: int

    @classmethod
    def from_metadata(cls, metadata: pd.DataFrame, rdkit_by_compound: dict[str, np.ndarray]):
        is_vehicle = metadata["is_vehicle"].astype(bool) if "is_vehicle" in metadata.columns else pd.Series(False, index=metadata.index)
        train = metadata[metadata["split"].eq("train") & ~is_vehicle]
        compounds = sorted(train["compound"].astype(str).unique().tolist())
        if rdkit_by_compound:
            rdkit_dim = int(len(next(iter(rdkit_by_compound.values()))))
        else:
            rdkit_dim = 0
        return cls(train_compounds=compounds, rdkit_by_compound=rdkit_by_compound, rdkit_dim=rdkit_dim)

    @property
    def onehot_dim(self) -> int:
        return len(self.train_compounds)

    def encode(self, metadata: pd.DataFrame, mode: str) -> np.ndarray | None:
        mode = str(mode)
        if mode == "none":
            return None
        log_dose = metadata["log_dose"].to_numpy(dtype=np.float32)[:, None]
        if mode == "dose":
            return log_dose.astype(np.float32)
        if mode == "onehot_dose":
            out = np.zeros((len(metadata), self.onehot_dim), dtype=np.float32)
            index = {compound: i for i, compound in enumerate(self.train_compounds)}
            for row, compound in enumerate(metadata["compound"].astype(str)):
                if compound in index:
                    out[row, index[compound]] = 1.0
            return np.hstack([out, log_dose]).astype(np.float32)
        if mode == "rdkit_dose":
            desc = np.zeros((len(metadata), self.rdkit_dim), dtype=np.float32)
            for row, compound in enumerate(metadata["compound"].astype(str)):
                if compound in self.rdkit_by_compound:
                    desc[row] = self.rdkit_by_compound[compound]
            return np.hstack([desc, log_dose]).astype(np.float32)
        raise ValueError(f"Unknown condition mode: {mode}")


def choose_heldout_compound(metadata: pd.DataFrame, preferred=("Belinostat", "Vorinostat", "Trametinib")) -> tuple[str, str]:
    treated = metadata.loc[~metadata["is_vehicle"]].copy()
    available = set(treated["compound"].astype(str))
    for compound in preferred:
        if compound in available:
            counts = treated.loc[treated["compound"].eq(compound)].groupby("dose").size()
            if len(counts) > 0 and counts.max() >= 5:
                return compound, "preferred_available"
            return compound, "preferred_available_low_count"
    counts = treated.groupby("compound").size().sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No treated compounds available for held-out compound split")
    return str(counts.index[0]), "fallback_most_cells"


def sciplex_split_counts(metadata_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, meta in metadata_by_split.items():
        for split, group in meta.groupby("split", observed=False):
            rows.append(
                {
                    "split_name": split_name,
                    "split": split,
                    "n_cells": int(len(group)),
                    "n_vehicle": int(group["is_vehicle"].sum()),
                    "n_treated": int((~group["is_vehicle"]).sum()),
                    "K_compounds": int(group.loc[~group["is_vehicle"], "compound"].nunique()),
                    "n_doses": int(group.loc[~group["is_vehicle"], "dose"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _normalized_split_id(split_name: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(split_name)).strip("_")


def should_include_nearest_chemistry(split_name: str) -> bool:
    split_id = _normalized_split_id(split_name)
    return (
        split_id.startswith("split_c")
        or "heldout_compound" in split_id
        or "held_out_compound" in split_id
    )


def _train_sciplex_cfm(
    X: np.ndarray,
    metadata: pd.DataFrame,
    condition: np.ndarray | None,
    condition_dim: int,
    steps: int,
    batch_size: int,
    seed: int,
    device=None,
    hidden: int = 96,
    layers: int = 3,
):
    import torch

    from ..core.losses import cfm_loss_from_pairs

    device = _torch_device(device)
    rng = np.random.default_rng(seed)
    source_idx = np.flatnonzero(metadata["train_role"].to_numpy() == "source")
    target_idx = np.flatnonzero(metadata["train_role"].to_numpy() == "target")
    if len(source_idx) == 0 or len(target_idx) == 0:
        raise ValueError("sci-Plex CFM training requires nonempty source and target pools")
    model = VelocityMLP(
        x_dim=X.shape[1], hidden_dim=int(hidden), hidden_layers=int(layers), condition_dim=int(condition_dim)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    for step in range(1, int(steps) + 1):
        i0 = rng.choice(source_idx, size=int(batch_size), replace=True)
        i1 = rng.choice(target_idx, size=int(batch_size), replace=True)
        x0 = torch.as_tensor(X[i0], dtype=torch.float32, device=device)
        x1 = torch.as_tensor(X[i1], dtype=torch.float32, device=device)
        if condition_dim > 0:
            cond = torch.as_tensor(condition[i1], dtype=torch.float32, device=device)
        else:
            cond = None
        loss = cfm_loss_from_pairs(model, x0, x1, condition=cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step == int(steps) or step % max(1, int(steps) // 5) == 0:
            rows.append({"step": int(step), "loss": float(loss.detach().cpu())})
    return model, pd.DataFrame(rows)


def _sample_source(X, idx, n, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(idx, dtype=int), size=int(n), replace=True)
    return X[chosen]


def _mean_shift_prediction(X, meta, source_eval_idx, compound: str, dose: float, n: int, seed: int):
    source = _sample_source(X, source_eval_idx, n, seed)
    train_source = X[np.flatnonzero(meta["train_role"].to_numpy() == "source")]
    train_targets = meta["train_role"].to_numpy() == "target"
    exact = train_targets & meta["compound"].eq(compound).to_numpy() & np.isclose(meta["dose"].to_numpy(dtype=float), float(dose))
    if exact.sum() == 0:
        same_compound = train_targets & meta["compound"].eq(compound).to_numpy()
        if same_compound.sum() > 0:
            train_doses = meta.loc[same_compound, "dose"].to_numpy(dtype=float)
            nearest = train_doses[np.argmin(np.abs(train_doses - float(dose)))]
            exact = same_compound & np.isclose(meta["dose"].to_numpy(dtype=float), nearest)
        else:
            same_dose = train_targets & np.isclose(meta["dose"].to_numpy(dtype=float), float(dose))
            exact = same_dose if same_dose.sum() > 0 else train_targets
    shift = X[np.flatnonzero(exact)].mean(axis=0) - train_source.mean(axis=0)
    return (source + shift).astype(np.float32)


def _nearest_chem_prediction(X, meta, source_eval_idx, compound, dose, n, rdkit_by_compound, train_compounds, seed):
    if compound not in rdkit_by_compound or not train_compounds:
        return _mean_shift_prediction(X, meta, source_eval_idx, compound, dose, n, seed)
    target = rdkit_by_compound[compound]
    seen = np.vstack([rdkit_by_compound[c] for c in train_compounds if c in rdkit_by_compound])
    seen_names = [c for c in train_compounds if c in rdkit_by_compound]
    if len(seen_names) == 0:
        return _mean_shift_prediction(X, meta, source_eval_idx, compound, dose, n, seed)
    nearest = seen_names[int(np.argmin(np.linalg.norm(seen - target[None, :], axis=1)))]
    return _mean_shift_prediction(X, meta, source_eval_idx, nearest, dose, n, seed)


def _predict_group(model, X, source_eval_idx, n, condition_vec, seed, nfe, device=None):
    source = _sample_source(X, source_eval_idx, n, seed)
    if condition_vec is None:
        cond = None
    else:
        cond = np.repeat(condition_vec[None, :].astype(np.float32), int(n), axis=0)
    return integrate_velocity(model, source, 0.0, 1.0, n_steps=nfe, condition=cond, device=device)


def evaluate_sciplex_split(
    X: np.ndarray,
    split_metadata: pd.DataFrame,
    rdkit_by_compound: dict[str, np.ndarray],
    split_name: str,
    training_steps: int,
    batch_size: int = 256,
    nfe: int = 32,
    seed: int = 42,
    device=None,
    max_eval_groups: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    meta = split_metadata.copy().reset_index(drop=True)
    X = _as_float32(X)
    if "log_dose" not in meta.columns:
        meta["log_dose"] = np.log1p(np.clip(meta["dose"].to_numpy(dtype=float), 0.0, None))
    is_train = meta["split"].eq("train")
    is_test = meta["split"].eq("test")
    is_vehicle = meta["is_vehicle"].astype(bool)
    meta["train_role"] = "unused"
    meta.loc[is_train & is_vehicle, "train_role"] = "source"
    meta.loc[is_train & (~is_vehicle), "train_role"] = "target"
    source_eval_idx = np.flatnonzero((is_test & is_vehicle).to_numpy())
    if len(source_eval_idx) == 0:
        source_eval_idx = np.flatnonzero(is_vehicle.to_numpy())
    eval_groups = (
        meta.loc[is_test & (~is_vehicle), ["compound", "dose"]]
        .drop_duplicates()
        .sort_values(["compound", "dose"])
        .reset_index(drop=True)
    )
    if max_eval_groups is not None and len(eval_groups) > int(max_eval_groups):
        eval_groups = eval_groups.head(int(max_eval_groups))

    encoder = SciplexConditionEncoder.from_metadata(meta, rdkit_by_compound=rdkit_by_compound)
    cond_m3 = encoder.encode(meta, "onehot_dose")
    cond_m4 = encoder.encode(meta, "rdkit_dose")
    cond_m1 = None
    models = {}
    histories = {}
    model_specs = [
        ("M1_unconditional", cond_m1, 0, 96, 3),
        ("M3_no_chemistry", cond_m3, cond_m3.shape[1], 96, 3),
        ("M4_chemistry_aware", cond_m4, cond_m4.shape[1], 96, 3),
    ]
    for i, (method, cond, cond_dim, hidden, layers) in enumerate(model_specs):
        model, hist = _train_sciplex_cfm(
            X, meta, cond, cond_dim, training_steps, batch_size, seed + i * 17, device=device, hidden=hidden, layers=layers
        )
        models[method] = (model, cond, cond_dim)
        histories[method] = hist

    m2_models = {}
    include_nearest_chemistry = should_include_nearest_chemistry(split_name)
    if not include_nearest_chemistry:
        eval_compounds = sorted(eval_groups["compound"].astype(str).unique().tolist())
        for j, compound in enumerate(eval_compounds):
            sub_meta = meta.copy()
            sub_meta.loc[(sub_meta["train_role"] == "target") & (~sub_meta["compound"].eq(compound)), "train_role"] = "unused"
            if (sub_meta["train_role"] == "target").sum() == 0:
                continue
            cond_dose = encoder.encode(sub_meta, "dose")
            model, hist = _train_sciplex_cfm(
                X,
                sub_meta,
                cond_dose,
                1,
                training_steps,
                batch_size,
                seed + 100 + j,
                device=device,
                hidden=64,
                layers=2,
            )
            m2_models[compound] = (model, cond_dose, hist)

    rows = []
    prediction_cache = {}
    train_compounds = encoder.train_compounds
    for group_i, group in eval_groups.iterrows():
        compound = str(group["compound"])
        dose = float(group["dose"])
        target_idx = np.flatnonzero(is_test.to_numpy() & (~is_vehicle.to_numpy()) & meta["compound"].eq(compound).to_numpy() & np.isclose(meta["dose"].to_numpy(dtype=float), dose))
        if len(target_idx) == 0:
            continue
        X_target = X[target_idx]
        n = len(target_idx)
        base_seed = seed + 1000 + int(group_i)
        pred_by_method = {
            "vehicle_as_prediction": _sample_source(X, source_eval_idx, n, base_seed),
            "mean_shift": _mean_shift_prediction(X, meta, source_eval_idx, compound, dose, n, base_seed + 1),
        }
        if include_nearest_chemistry:
            pred_by_method["nearest_chemistry"] = _nearest_chem_prediction(
                X, meta, source_eval_idx, compound, dose, n, rdkit_by_compound, train_compounds, base_seed + 2
            )
        for method, (model, cond, _) in models.items():
            if cond is None:
                cond_vec = None
            else:
                cond_vec = cond[target_idx[0]]
            pred_by_method[method] = _predict_group(model, X, source_eval_idx, n, cond_vec, base_seed + 3, nfe, device=device)
        if compound in m2_models:
            model, cond_dose, _ = m2_models[compound]
            pred_by_method["M2_per_compound"] = _predict_group(
                model, X, source_eval_idx, n, cond_dose[target_idx[0]], base_seed + 4, nfe, device=device
            )
        for method, pred in pred_by_method.items():
            metrics = distribution_readout_metrics(pred, X_target)
            rows.append(
                {
                    "split_name": split_name,
                    "compound": compound,
                    "dose": dose,
                    "method": method,
                    "n_target_cells": int(n),
                    "training_steps": int(training_steps),
                    **metrics,
                }
            )
        prediction_cache[(compound, dose)] = {"target": X_target, **pred_by_method}
    return pd.DataFrame(rows), {
        "models": models,
        "m2_models": m2_models,
        "histories": histories,
        "encoder": encoder,
        "predictions": prediction_cache,
    }


def aggregate_metric_table(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in rows.columns
        if col.startswith("program_readout_") or col in {"n_target_cells", "training_steps"}
    ]
    group_cols = ["split_name", "method"]
    out = rows.groupby(group_cols, observed=False)[metric_cols].mean(numeric_only=True).reset_index()
    return out.sort_values(["split_name", "method"]).reset_index(drop=True)
