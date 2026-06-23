from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def cfm_batch(x0, x1, s=None, t=None):
    if torch is None:
        raise ImportError("cfm_batch requires PyTorch.")
    if s is not None and t is not None:
        raise ValueError("Use either s or t, not both")
    if t is not None:
        s = t
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {tuple(x0.shape)} and {tuple(x1.shape)}")
    if s is None:
        s = torch.rand(x0.shape[0], 1, device=x0.device, dtype=x0.dtype)
    if s.ndim == 1:
        s = s[:, None]
    s = s.to(device=x0.device, dtype=x0.dtype)
    if s.shape != (x0.shape[0], 1):
        raise ValueError(f"s must have shape ({x0.shape[0]}, 1) or ({x0.shape[0]},), got {tuple(s.shape)}")
    x_s = (1.0 - s) * x0 + s * x1
    u_s = x1 - x0
    return x_s, s, u_s


def cfm_loss(model, x0, x1, condition=None):
    return cfm_loss_from_pairs(model, x0, x1, condition=condition)


def cfm_loss_from_pairs(model, x0, x1, s=None, condition=None, return_batch: bool = False):
    """Compute MSE velocity regression loss for endpoint-pair CFM."""
    x_s, s, u_s = cfm_batch(x0, x1, s=s)
    pred = model(x_s, s, condition) if condition is not None else model(x_s, s)
    if pred.shape != u_s.shape or u_s.shape != x0.shape:
        raise ValueError(
            "CFM prediction, target velocity, and x0 must have matching shapes; "
            f"got pred={tuple(pred.shape)}, u_s={tuple(u_s.shape)}, x0={tuple(x0.shape)}"
        )
    loss = ((pred - u_s) ** 2).mean(dim=-1).mean()
    if return_batch:
        return loss, {"x_s": x_s, "s": s, "u_s": u_s, "pred": pred}
    return loss
