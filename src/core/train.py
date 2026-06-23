from __future__ import annotations

import time

import pandas as pd


def _tensor_batch(batch: dict, batch_size: int, device: str):
    import torch

    x0 = torch.as_tensor(batch["x0"], dtype=torch.float32, device=device)
    x1 = torch.as_tensor(batch["x1"], dtype=torch.float32, device=device)
    condition = batch.get("condition", None)
    if condition is not None:
        condition = torch.as_tensor(condition, dtype=torch.float32, device=device)
    if x0.shape[0] != batch_size or x1.shape[0] != batch_size:
        raise ValueError("pair_batch_fn returned a batch with the wrong leading dimension")
    return x0, x1, condition


def train_cfm_steps(
    model,
    pair_batch_fn,
    optimizer,
    n_steps: int,
    batch_size: int,
    device: str = "cpu",
    log_every: int = 10,
) -> pd.DataFrame:
    """Train local velocity regression and return step/loss/wall_time/nfe_train."""
    from .losses import cfm_loss_from_pairs

    rows = []
    model.to(device)
    model.train()
    start = time.perf_counter()
    last = start
    for step in range(1, int(n_steps) + 1):
        batch = pair_batch_fn(int(batch_size))
        x0, x1, condition = _tensor_batch(batch, int(batch_size), device)
        loss = cfm_loss_from_pairs(model, x0, x1, condition=condition)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        now = time.perf_counter()
        if step == 1 or step % int(log_every) == 0 or step == int(n_steps):
            rows.append(
                {
                    "method": "cfm_local_regression",
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "wall_time_sec": float(now - start),
                    "sec_per_step": float((now - last) / max(1, step - (rows[-1]["step"] if rows else 0))),
                    "nfe_train_per_step": 1,
                    "batch_size": int(batch_size),
                }
            )
            last = now
    return pd.DataFrame(rows)


def train_rollout_endpoint_baseline(
    model,
    pair_batch_fn,
    optimizer,
    n_steps: int,
    batch_size: int,
    rollout_steps: int = 8,
    device: str = "cpu",
    log_every: int = 10,
) -> pd.DataFrame:
    """Train by unrolling Euler inside the loss and matching endpoint x1.

    This is a simulation-dependent Neural-ODE/CNF-style endpoint baseline,
    not a full likelihood CNF with divergence tracking.
    """
    from .sampling import _call_model

    rows = []
    model.to(device)
    model.train()
    start = time.perf_counter()
    last = start
    rollout_steps = int(rollout_steps)
    for step in range(1, int(n_steps) + 1):
        import torch

        batch = pair_batch_fn(int(batch_size))
        x0, x1, condition = _tensor_batch(batch, int(batch_size), device)
        x = x0
        times = torch.linspace(0.0, 1.0, rollout_steps + 1, device=x0.device, dtype=x0.dtype)
        for i in range(rollout_steps):
            s = torch.full((x.shape[0], 1), times[i], device=x.device, dtype=x.dtype)
            dt = times[i + 1] - times[i]
            x = x + dt * _call_model(model, x, s, condition)
        loss = ((x - x1) ** 2).sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        now = time.perf_counter()
        if step == 1 or step % int(log_every) == 0 or step == int(n_steps):
            rows.append(
                {
                    "method": "simulation_dependent_rollout_endpoint",
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "wall_time_sec": float(now - start),
                    "sec_per_step": float((now - last) / max(1, step - (rows[-1]["step"] if rows else 0))),
                    "nfe_train_per_step": int(rollout_steps),
                    "batch_size": int(batch_size),
                }
            )
            last = now
    return pd.DataFrame(rows)


def train_cfm(model, sampler, optimizer, n_steps: int = 500, batch_size: int = 256, device: str = "cpu"):
    """Backward-compatible list-returning Chapter 1-3 helper."""

    def pair_batch_fn(size):
        return sampler.sample(size)

    history = train_cfm_steps(
        model,
        pair_batch_fn,
        optimizer,
        n_steps=n_steps,
        batch_size=batch_size,
        device=device,
        log_every=1,
    )
    return history["loss"].tolist()
