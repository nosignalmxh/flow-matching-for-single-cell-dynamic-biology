from __future__ import annotations

import numpy as np


def make_time_batch(batch_size: int, device):
    import torch

    return torch.rand(int(batch_size), 1, device=device)


def train_cfm(
    model,
    pair_sampler,
    steps: int,
    batch_size: int,
    lr: float,
    device,
    seed: int,
    log_every: int = 250,
):
    import torch

    from ..core.train import train_cfm_steps
    from ..utils import set_seed

    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    return train_cfm_steps(
        model,
        pair_sampler.sample,
        optimizer,
        n_steps=int(steps),
        batch_size=int(batch_size),
        device=device,
        log_every=int(log_every),
    )


def rollout_euler(model, x0, nfe: int, device):
    import torch

    from ..core.sampling import euler_sample

    model.eval()
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    with torch.no_grad():
        endpoint, _, _ = euler_sample(model, x0_t, n_steps=int(nfe), return_traj=False)
    return endpoint.detach().cpu().numpy().astype(np.float32)


def trajectory_rollout(model, x0, nfe: int, device, return_path: bool = True):
    import torch

    from ..core.sampling import euler_sample

    model.eval()
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    with torch.no_grad():
        endpoint, traj_t, _ = euler_sample(model, x0_t, n_steps=int(nfe), return_traj=return_path)
    endpoint_np = endpoint.detach().cpu().numpy().astype(np.float32)
    if return_path:
        traj_np = traj_t.detach().cpu().numpy().astype(np.float32)
        return endpoint_np, traj_np, np.linspace(0.0, 1.0, int(nfe) + 1)
    return endpoint_np


def coarse_step_error(model, x0, nfe_coarse: int = 4, nfe_fine: int = 64, device=None) -> float:
    if device is None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_coarse = rollout_euler(model, x0, nfe=int(nfe_coarse), device=device)
    z_fine = rollout_euler(model, x0, nfe=int(nfe_fine), device=device)
    return float(np.linalg.norm(z_coarse - z_fine, axis=1).mean())
