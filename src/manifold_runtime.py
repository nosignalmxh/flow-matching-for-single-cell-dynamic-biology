from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .artifacts import load_npz, load_pt, save_csv, save_npz, save_pt
from .flow_runtime import rollout_euler, train_cfm
from .models import VelocityMLP as BaseVelocityMLP
from .models import count_parameters
from .ot import independent_coupling, sample_from_plan
from .samplers import CouplingPairSampler


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


def midpoint_direction_dispersion(X0, X1, pi, n_pairs: int = 4096, k: int = 25, seed: int = 42):
    from sklearn.neighbors import NearestNeighbors

    idx0, idx1 = sample_from_plan(pi, n_pairs, seed=seed)
    x0, x1 = X0[idx0], X1[idx1]
    chords = x1 - x0
    chord_norm = np.linalg.norm(chords, axis=1)
    directions = chords / np.clip(chord_norm[:, None], 1e-12, None)
    midpoints = 0.5 * (x0 + x1)
    n_neighbors = max(2, min(int(k), len(midpoints)))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(midpoints)
    neigh = nn.kneighbors(midpoints, return_distance=False)
    angular_std = []
    for ids in neigh:
        local_dirs = directions[ids]
        mean_dir = local_dirs.mean(axis=0)
        mean_norm = np.linalg.norm(mean_dir)
        if mean_norm < 1e-12:
            angles = np.full(len(ids), 90.0)
        else:
            mean_dir = mean_dir / mean_norm
            cosang = np.clip(local_dirs @ mean_dir, -1.0, 1.0)
            angles = np.degrees(np.arccos(cosang))
        angular_std.append(float(np.std(angles)))
    return pd.DataFrame(
        {
            "idx0": idx0,
            "idx1": idx1,
            "pc20_chord_length": chord_norm,
            "midpoint_direction_angular_std_deg": angular_std,
        }
    )
