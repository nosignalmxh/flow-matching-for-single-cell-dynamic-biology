from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def euler_integrate(model, x0, n_steps: int = 50, condition=None):
    import torch

    xs = [x0]
    x = x0
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device)
    for i in range(n_steps):
        t = times[i].expand(x.shape[0], 1)
        dt = times[i + 1] - times[i]
        v = model(x, t, condition) if condition is not None else model(x, t)
        x = x + dt * v
        xs.append(x)
    return xs


def _call_model(model, x, s, condition=None):
    if condition is None:
        return model(x, s)
    return model(x, s, condition)


if torch is None:  # pragma: no cover
    def _no_grad(fn):
        return fn
else:
    _no_grad = torch.no_grad()


@_no_grad
def euler_sample(model, x0, n_steps: int = 50, condition=None, return_traj: bool = True):
    """Integrate the learned velocity field with fixed-step Euler."""
    import torch

    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    x = x0
    traj = [x0] if return_traj else None
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device, dtype=x0.dtype)
    for i in range(n_steps):
        s = torch.full((x.shape[0], 1), times[i], device=x.device, dtype=x.dtype)
        dt = times[i + 1] - times[i]
        x = x + dt * _call_model(model, x, s, condition)
        if return_traj:
            traj.append(x)
    if return_traj:
        return x, torch.stack(traj, dim=0), n_steps
    return x, None, n_steps


@_no_grad
def midpoint_sample(model, x0, n_steps: int = 50, condition=None, return_traj: bool = True):
    """Integrate the learned velocity field with fixed-step midpoint."""
    import torch

    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    x = x0
    traj = [x0] if return_traj else None
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device, dtype=x0.dtype)
    for i in range(n_steps):
        s = torch.full((x.shape[0], 1), times[i], device=x.device, dtype=x.dtype)
        dt = times[i + 1] - times[i]
        k1 = _call_model(model, x, s, condition)
        s_mid = s + 0.5 * dt
        x_mid = x + 0.5 * dt * k1
        k2 = _call_model(model, x_mid, s_mid, condition)
        x = x + dt * k2
        if return_traj:
            traj.append(x)
    if return_traj:
        return x, torch.stack(traj, dim=0), 2 * n_steps
    return x, None, 2 * n_steps


@_no_grad
def odeint_sample(
    model,
    x0,
    condition=None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "dopri5",
):
    """Use torchdiffeq to sample and count actual RHS calls."""
    import torch

    try:
        from torchdiffeq import odeint
    except ImportError as exc:
        raise ImportError("odeint_sample requires torchdiffeq.") from exc

    nfe = 0

    def rhs(t, x):
        nonlocal nfe
        nfe += 1
        tt = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype) * t
        return _call_model(model, x, tt, condition)

    t_grid = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
    traj = odeint(rhs, x0, t_grid, method=method, rtol=rtol, atol=atol)
    return traj[-1], traj, int(nfe)
