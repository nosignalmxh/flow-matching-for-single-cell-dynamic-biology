from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class VelocityMLP(nn.Module if nn is not None else object):
    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 64,
        condition_dim: int = 0,
        hidden_layers: int = 2,
        time_embed_dim: int | None = None,
    ):
        if nn is None:
            raise ImportError("VelocityMLP requires PyTorch.")
        super().__init__()
        if time_embed_dim is not None and time_embed_dim != 1:
            raise ValueError("VelocityMLP keeps scalar time in this tutorial; use time_embed_dim=None or 1")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_layers = int(hidden_layers)
        self.time_embed_dim = time_embed_dim

        input_dim = self.x_dim + 1 + self.condition_dim
        layers = []
        last_dim = input_dim
        for _ in range(max(self.hidden_layers, 1)):
            layers.extend([nn.Linear(last_dim, self.hidden_dim), nn.SiLU()])
            last_dim = self.hidden_dim
        layers.append(nn.Linear(last_dim, self.x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, condition=None):
        if t.ndim == 1:
            t = t[:, None]
        if t.shape[0] == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0], 1)
        pieces = [x, t]
        if condition is not None:
            pieces.append(condition)
        return self.net(torch.cat(pieces, dim=-1))


class ConditionalVelocityMLP(VelocityMLP):
    def __init__(self, x_dim: int = 2, n_conditions: int = 2, condition_embed_dim: int = 8, hidden_dim: int = 64):
        if nn is None:
            raise ImportError("ConditionalVelocityMLP requires PyTorch.")
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(n_conditions, condition_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1 + condition_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, t, condition):
        if t.ndim == 1:
            t = t[:, None]
        emb = self.embedding(condition.long().view(-1))
        return self.net(torch.cat([x, t, emb], dim=-1))


def count_parameters(model) -> int:
    """Return the number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
