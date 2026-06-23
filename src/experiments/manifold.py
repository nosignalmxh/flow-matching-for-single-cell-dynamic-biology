from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..artifacts import load_npz, load_pt, save_csv, save_npz, save_pt
from ..core.models import VelocityMLP as BaseVelocityMLP
from ..core.models import count_parameters
from ..core.ot import independent_coupling
from ..data.samplers import CouplingPairSampler
from .flow_runtime import rollout_euler, train_cfm


def fate_conditioned_plan(X0, X1, source_labels, target_labels, source_labels_for_plan=None) -> np.ndarray:
    """Build a row-balanced empirical plan constrained by matched fate labels."""
    source_labels = np.asarray(source_labels).astype(str)
    target_labels = np.asarray(target_labels).astype(str)
    labels_for_plan = source_labels if source_labels_for_plan is None else np.asarray(source_labels_for_plan).astype(str)
    pi = np.zeros((len(X0), len(X1)), dtype=float)
    X0 = np.asarray(X0, dtype=np.float32)
    X1 = np.asarray(X1, dtype=np.float32)
    for i, lab in enumerate(labels_for_plan):
        cols = np.flatnonzero(target_labels == lab)
        if cols.size == 0:
            cols = np.arange(len(X1))
        d2 = np.sum((X1[cols] - X0[i]) ** 2, axis=1)
        scale = max(float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0, 1e-6)
        w = np.exp(-d2 / scale)
        w = w / np.clip(w.sum(), 1e-12, None)
        pi[i, cols] = w / len(X0)
    return pi / np.clip(pi.sum(), 1e-12, None)

class Ch04VelocityMLP(BaseVelocityMLP):
    """Chapter 4 VelocityMLP wrapper with the notebook's default architecture."""

    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 4):
        super().__init__(x_dim=int(input_dim), hidden_dim=int(hidden), hidden_layers=int(layers))

class PlanPairSampler:
    """Chapter 4 wrapper around a fixed empirical coupling sampler."""

    def __init__(self, X0, X1, pi=None, seed: int = 42, labels0=None, labels1=None):
        self.X0 = np.asarray(X0, dtype=np.float32)
        self.X1 = np.asarray(X1, dtype=np.float32)
        if pi is None:
            pi = independent_coupling(len(self.X0), len(self.X1))
        self.pi = np.asarray(pi, dtype=float)
        self.src_sampler = CouplingPairSampler(self.X0, self.X1, self.pi, seed=seed, labels0=labels0, labels1=labels1)

    def sample(self, batch_size: int = 256):
        return self.src_sampler.sample(int(batch_size))

def _validate_checkpoint_metadata(payload: dict, ckpt: Path, expected: dict[str, int]) -> None:
    mismatched = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if key in payload and int(payload[key]) != int(value)
    }
    if mismatched:
        details = ", ".join(f"{key}: found {found}, expected {expected_value}" for key, (found, expected_value) in mismatched.items())
        raise ValueError(f"Checkpoint metadata mismatch for {ckpt}: {details}")

def train_or_load_model(
    name: str,
    X0,
    X1,
    pi,
    *,
    cache_dir: str | Path,
    steps: int,
    batch_size: int,
    seed: int = 42,
    device=None,
    lr: float = 1e-3,
    hidden: int = 128,
    layers: int = 4,
    log_every: int = 250,
):
    cache_dir = Path(cache_dir)
    cache_tag = f"d{X0.shape[1]}_steps{int(steps)}_batch{int(batch_size)}_seed{int(seed)}"
    ckpt = cache_dir / f"{name}_{cache_tag}_model.pt"
    hist_path = cache_dir / f"{name}_{cache_tag}_history.csv"
    model = Ch04VelocityMLP(input_dim=X0.shape[1], hidden=hidden, layers=layers)
    if device is not None:
        model = model.to(device)
    if ckpt.exists():
        payload = load_pt(ckpt, map_location=device)
        _validate_checkpoint_metadata(
            payload,
            ckpt,
            {
                "input_dim": int(X0.shape[1]),
                "steps": int(steps),
                "batch_size": int(batch_size),
                "seed": int(seed),
                "hidden": int(hidden),
                "layers": int(layers),
            },
        )
        model.load_state_dict(payload["state_dict"])
        history = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
        print(f"Loaded {name} from {ckpt}")
        return model, history
    sampler = PlanPairSampler(X0, X1, pi=pi, seed=seed)
    history = train_cfm(
        model,
        sampler,
        steps=int(steps),
        batch_size=int(batch_size),
        lr=float(lr),
        device=device,
        seed=int(seed),
        log_every=int(log_every),
    )
    save_pt(
        ckpt,
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X0.shape[1]),
            "seed": int(seed),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "hidden": int(hidden),
            "layers": int(layers),
        },
    )
    save_csv(hist_path, history)
    print(f"Trained {name}; parameters={count_parameters(model)}; final loss={history.loss.iloc[-1]:.4f}")
    return model, history

def train_reflow_round(
    name: str,
    base_model,
    X0,
    *,
    cache_dir: str | Path,
    train_or_load: Callable,
    steps: int,
    nfe: int,
    seed: int = 42,
    device=None,
):
    endpoint_path = Path(cache_dir) / f"{name}_induced_endpoint.npz"
    if endpoint_path.exists():
        Z = load_npz(endpoint_path)["endpoint"]
    else:
        Z = rollout_euler(base_model, X0, nfe=int(nfe), device=device)
        save_npz(endpoint_path, endpoint=Z)
    pi_diag = np.eye(len(X0), len(Z), dtype=float)
    pi_diag /= pi_diag.sum()
    model, hist = train_or_load(name, X0, Z, pi_diag, steps=steps, seed=seed)
    return model, Z, hist

__all__ = [
    "fate_conditioned_plan",
    "Ch04VelocityMLP",
    "PlanPairSampler",
    "train_or_load_model",
    "train_reflow_round",
]
