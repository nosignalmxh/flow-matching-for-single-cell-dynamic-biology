from __future__ import annotations

import os
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import display_saved_figure, save_figure


@dataclass(frozen=True)
class Ch03RunConfig:
    seed: int
    quick_mode: bool
    smoke_mode: bool
    paper_figure_mode: bool
    device: Any


@dataclass(frozen=True)
class Ch04RunConfig:
    seeds: list[int]
    default_seed: int
    source_time: str
    target_time: str
    training_steps: int
    batch_size: int
    default_nfe: int
    nfe_grid: list[int]
    sinkhorn_epsilon: float
    epsilon_grid: list[float]
    smoke_mode: bool
    device: Any


@dataclass(frozen=True)
class BootstrapResult:
    project_root: Path
    fig_dir: Path
    out_dir: Path
    seed: int
    quick_mode: bool | None
    device: Any


def resolve_project_root() -> Path:
    """Return the nearest parent directory containing both src/ and notebooks/."""
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root from {start}; expected a parent containing src/ and notebooks/."
    )


def apply_tutorial_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _default_device(env_name: str, fallback: str = "cuda") -> Any:
    import torch

    value = os.environ.get(env_name)
    if value:
        return torch.device(value)
    default = fallback if torch.cuda.is_available() else "cpu"
    return torch.device(default)


def bootstrap(
    *,
    chapter: str,
    seed: int = 42,
    quick_mode: bool | None = None,
    smoke_mode: bool | None = None,
    set_deterministic: bool = True,
) -> BootstrapResult:
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig_{chapter}")
    if chapter == "ch02":
        os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{chapter}")

    project_root = resolve_project_root()
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from .utils import ensure_dir, set_seed

    del smoke_mode, set_deterministic

    set_seed(seed)
    fig_dir = ensure_dir(project_root / "figures" / chapter)
    out_dir = ensure_dir(project_root / "outputs" / chapter)
    device = _default_device(f"{chapter.upper()}_DEVICE")
    root_display = os.path.relpath(project_root, Path.cwd().resolve())

    print(f"project_root={root_display}")
    print(f"seed={seed}")
    print(f"quick_mode={quick_mode}")
    print(f"fig_dir={fig_dir.relative_to(project_root)}")
    print(f"out_dir={out_dir.relative_to(project_root)}")
    print(f"device={device}")

    return BootstrapResult(
        project_root=project_root,
        fig_dir=fig_dir,
        out_dir=out_dir,
        seed=seed,
        quick_mode=quick_mode,
        device=device,
    )


def make_ch03_run_config(
    seed_env: str = "CH03_SEED",
    quick_env: str = "CH03_QUICK",
    smoke_env: str = "CH03_SMOKE_MODE",
) -> Ch03RunConfig:
    return Ch03RunConfig(
        seed=int(os.environ.get(seed_env, "42")),
        quick_mode=os.environ.get(quick_env, "1") == "1",
        smoke_mode=os.environ.get(smoke_env, "0") == "1",
        paper_figure_mode=os.environ.get("CH03_PAPER_FIGURE_MODE", "1") == "1",
        device=_default_device("CH03_DEVICE"),
    )


def make_ch04_run_config() -> Ch04RunConfig:
    training_steps = int(os.environ.get("CH04_TRAINING_STEPS", "1500"))
    batch_size = int(os.environ.get("CH04_BATCH_SIZE", "256"))
    default_nfe = int(os.environ.get("CH04_DEFAULT_NFE", "64"))
    nfe_grid = [2, 4, 8, 16, 32, 64]
    smoke_mode = os.environ.get("CH04_SMOKE_MODE", "0") == "1"

    if smoke_mode:
        training_steps = min(training_steps, 20)
        batch_size = min(batch_size, 64)
        default_nfe = min(default_nfe, 8)
        nfe_grid = [2, 4, 8]

    return Ch04RunConfig(
        seeds=[42, 43, 44],
        default_seed=42,
        source_time="1",
        target_time="2",
        training_steps=training_steps,
        batch_size=batch_size,
        default_nfe=default_nfe,
        nfe_grid=nfe_grid,
        sinkhorn_epsilon=float(os.environ.get("CH04_SINKHORN_EPSILON", "0.05")),
        epsilon_grid=[0.01, 0.02, 0.05, 0.1, 0.5],
        smoke_mode=smoke_mode,
        device=_default_device("CH04_DEVICE"),
    )


def _display_path_from_saved(saved: Any, fig_dir: Path | None = None) -> Path:
    if isinstance(saved, (list, tuple)):
        paths = [Path(path) for path in saved]
        for path in paths:
            if path.suffix.lower() == ".png":
                if path.exists() or fig_dir is None or path.is_absolute():
                    return path
                return fig_dir / path.name
        if paths:
            path = paths[0]
            if path.exists() or fig_dir is None or path.is_absolute():
                return path
            return fig_dir / path.name
    path = Path(saved)
    if path.exists() or fig_dir is None or path.is_absolute():
        return path
    return fig_dir / path.name


def make_save_and_show(
    fig_dir: str | Path,
    *,
    width: int = 900,
    write_pdf: bool = False,
    save_fn: Callable[..., Path] | None = None,
) -> Callable[..., Any]:
    fig_dir = Path(fig_dir)

    def _call_save(save: Callable[..., Any], fig, filename: str | Path, write_pdf_value: bool) -> Any:
        parameters = list(inspect.signature(save).parameters.values())
        positional_params = [
            param
            for param in parameters
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        kwargs = (
            {"write_pdf": write_pdf_value}
            if any(param.name == "write_pdf" or param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters)
            else {}
        )
        if len(positional_params) >= 3 and positional_params[1].name in {"fig_dir", "figure_dir"}:
            return save(fig, fig_dir, filename, **kwargs)
        return save(fig, filename, **kwargs)

    def save_and_show(fig, filename: str | Path | int | None = None, *, width: int = width, write_pdf: bool | None = None) -> Path:
        if isinstance(fig, (str, Path)) and not hasattr(fig, "savefig"):
            path = Path(fig)
            display_width = int(filename) if isinstance(filename, int) else width
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            return display_saved_figure(path, width=display_width)
        if filename is None:
            raise TypeError("save_and_show() missing required filename for figure input")
        save = save_fn or save_figure
        saved = _call_save(save, fig, filename, write_pdf if write_pdf is not None else bool(write_pdf_default))
        png_path = _display_path_from_saved(saved, fig_dir)
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
        if not png_path.exists() or png_path.stat().st_size == 0:
            raise FileNotFoundError(png_path)
        display_saved_figure(png_path, width=width)
        return saved

    write_pdf_default = write_pdf
    return save_and_show
