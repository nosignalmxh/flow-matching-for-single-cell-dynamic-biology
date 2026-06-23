from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(table, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if isinstance(table, pd.DataFrame):
        table.to_csv(path, index=False)
    else:
        pd.DataFrame(table).to_csv(path, index=False)
    return path


def resolve_project_root(start: str | Path | None = None, *, markers=("src", "notebooks")) -> Path:
    start_path = Path(start or os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    raw_candidates = [
        start_path,
        *start_path.parents,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        Path("/home/xmabs/flow_matching_for_dynamic_biology/flow_matching_for_dynamic_biology"),
        Path("/import/home4/xmabs/flow_matching_for_dynamic_biology/flow_matching_for_dynamic_biology"),
    ]
    candidates = []
    seen = set()
    for candidate in raw_candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    marker_paths = tuple(Path(marker) for marker in markers)
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in marker_paths):
            return candidate.resolve()
    marker_text = ", ".join(str(marker) for marker in marker_paths)
    raise FileNotFoundError(f"Could not locate project root from {start_path}; required markers: {marker_text}")
