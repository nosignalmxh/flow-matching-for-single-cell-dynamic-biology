from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_npz(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path):
    return np.load(Path(path), allow_pickle=True)


def save_csv(path: str | Path, frame: pd.DataFrame | Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def figure_paths_from_name(fig_dir: str | Path, filename_or_stem: str | Path) -> tuple[Path, Path, str]:
    name = str(filename_or_stem)
    path = Path(name)
    stem = path.stem if path.suffix else name
    fig_dir = Path(fig_dir)
    return fig_dir / f"{stem}.png", fig_dir / f"{stem}.pdf", stem


def save_figure(
    fig,
    fig_dir: str | Path,
    filename_or_stem: str | Path,
    *,
    dpi: int = 300,
    write_pdf: bool = True,
    bbox_inches: str = "tight",
    pad_inches: float = 0.025,
) -> Path:
    png_path, pdf_path, _ = figure_paths_from_name(fig_dir, filename_or_stem)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    if write_pdf:
        fig.savefig(pdf_path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    return png_path


def save_paper_table(path_stem: str | Path, frame: pd.DataFrame, *, index: bool = False) -> tuple[Path, Path, Path]:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(frame)
    csv_path = path_stem.with_suffix(".csv")
    tex_path = path_stem.with_suffix(".tex")
    md_path = path_stem.with_suffix(".md")
    table.to_csv(csv_path, index=index)
    tex_path.write_text(table.to_latex(index=index, escape=False, float_format=lambda x: f"{x:.4g}"), encoding="utf-8")
    md_path.write_text(table.to_markdown(index=index), encoding="utf-8")
    return csv_path, tex_path, md_path


def display_saved_figure(path: str | Path, *, width: int | None = None) -> Path:
    from IPython.display import Image, display

    path = Path(path)
    display(Image(filename=str(path), width=width))
    return path


def display_saved_figures(paths: Iterable[str | Path], *, width: int | None = None) -> list[Path]:
    displayed = []
    for path in paths:
        displayed.append(display_saved_figure(path, width=width))
    return displayed


def display_table(frame: pd.DataFrame, columns: list[str] | None = None, n: int = 10) -> pd.DataFrame:
    from IPython.display import display

    preview = pd.DataFrame(frame)
    if columns is not None:
        preview = preview.loc[:, columns]
    preview = preview.head(n)
    display(preview)
    return preview


def save_pt(path: str | Path, payload: Any) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_pt(path: str | Path, map_location=None) -> Any:
    import torch

    return torch.load(Path(path), map_location=map_location)


def artifact_exists(*paths: str | Path) -> bool:
    return all(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths)


def stable_hash(*items) -> str:
    h = hashlib.sha1()
    for item in items:
        h.update(str(item).encode())
    return h.hexdigest()[:10]


def sample_rows(n: int, max_n: int | None, seed: int) -> np.ndarray:
    n = int(n)
    idx = np.arange(n)
    if max_n is None or n <= int(max_n):
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=int(max_n), replace=False))


def as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def to_tensor(x, device):
    import torch

    return torch.as_tensor(x, dtype=torch.float32, device=device)


def ensure_finite(name: str, x) -> None:
    if not np.all(np.isfinite(np.asarray(x))):
        raise ValueError(f"{name} contains non-finite values")


def safe_relpath(path: str | Path, root: str | Path | None = None) -> str:
    path = Path(path)
    root_path = Path.cwd() if root is None else Path(root)
    try:
        return str(path.resolve().relative_to(root_path.resolve()))
    except ValueError:
        return str(path)


def resolve_required_artifact(filename: str | Path, preferred_dirs) -> Path:
    filename = Path(filename)
    if filename.is_absolute():
        if filename.exists() and filename.stat().st_size > 0:
            return filename
        raise FileNotFoundError(filename)
    candidates = [Path(directory) / filename for directory in preferred_dirs]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"Missing required artifact {filename}; checked {[str(p) for p in candidates]}")


def remember_source(sources: dict[str, str], name: str, path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path)
    sources[str(name)] = safe_relpath(path, root=root)
    return path
