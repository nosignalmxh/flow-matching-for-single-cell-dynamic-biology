from __future__ import annotations

import os
import sys
from pathlib import Path


def test_tutorial_init_exports_root_resolution_and_configs(monkeypatch, tmp_path):
    from src.tutorial_init import (
        apply_tutorial_plot_style,
        make_ch03_run_config,
        make_ch04_run_config,
        resolve_project_root,
    )

    project = tmp_path / "project"
    nested = project / "notebooks" / "scratch"
    (project / "src").mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert resolve_project_root() == project.resolve()

    monkeypatch.setenv("CH03_SEED", "7")
    monkeypatch.setenv("CH03_QUICK", "0")
    monkeypatch.setenv("CH03_SMOKE_MODE", "1")
    monkeypatch.setenv("CH03_PAPER_FIGURE_MODE", "0")
    monkeypatch.setenv("CH03_DEVICE", "cpu")
    ch03 = make_ch03_run_config()
    assert ch03.seed == 7
    assert ch03.quick_mode is False
    assert ch03.smoke_mode is True
    assert ch03.paper_figure_mode is False
    assert str(ch03.device) == "cpu"

    monkeypatch.setenv("CH04_TRAINING_STEPS", "123")
    monkeypatch.setenv("CH04_BATCH_SIZE", "99")
    monkeypatch.setenv("CH04_DEFAULT_NFE", "16")
    monkeypatch.setenv("CH04_SINKHORN_EPSILON", "0.2")
    monkeypatch.setenv("CH04_SMOKE_MODE", "0")
    monkeypatch.setenv("CH04_DEVICE", "cpu")
    ch04 = make_ch04_run_config()
    assert ch04.seeds == [42, 43, 44]
    assert ch04.default_seed == 42
    assert ch04.source_time == "1"
    assert ch04.target_time == "2"
    assert ch04.training_steps == 123
    assert ch04.batch_size == 99
    assert ch04.default_nfe == 16
    assert ch04.nfe_grid == [2, 4, 8, 16, 32, 64]
    assert ch04.sinkhorn_epsilon == 0.2
    assert ch04.epsilon_grid == [0.01, 0.02, 0.05, 0.1, 0.5]
    assert ch04.smoke_mode is False
    assert str(ch04.device) == "cpu"

    apply_tutorial_plot_style()
    import matplotlib.pyplot as plt

    assert plt.rcParams["figure.dpi"] == 130
    assert plt.rcParams["savefig.dpi"] == 320
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False


def test_bootstrap_sets_paths_seed_env_and_summary(monkeypatch, tmp_path, capsys):
    from src.tutorial_init import bootstrap

    project = tmp_path / "project"
    nested = project / "notebooks" / "scratch"
    (project / "src").mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("CH02_DEVICE", "cpu")
    while str(project.resolve()) in sys.path:
        sys.path.remove(str(project.resolve()))

    boot = bootstrap(chapter="ch02", seed=7, quick_mode=True)

    assert boot.project_root == project.resolve()
    assert boot.fig_dir == project / "figures" / "ch02"
    assert boot.out_dir == project / "outputs" / "ch02"
    assert boot.seed == 7
    assert boot.quick_mode is True
    assert str(boot.device) == "cpu"
    assert sys.path[0] == str(project.resolve())
    assert os.environ["MPLCONFIGDIR"] == "/tmp/mplconfig_ch02"
    assert os.environ["NUMBA_CACHE_DIR"] == "/tmp/numba_cache_ch02"
    assert boot.fig_dir.is_dir()
    assert boot.out_dir.is_dir()

    assert capsys.readouterr().out.strip().splitlines() == [
        f"project_root={os.path.relpath(project.resolve(), nested.resolve())}",
        "seed=7",
        "quick_mode=True",
        "fig_dir=figures/ch02",
        "out_dir=outputs/ch02",
        "device=cpu",
    ]


def test_make_save_and_show_saves_png_and_optional_pdf(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from src.tutorial_init import make_save_and_show

    displayed: list[tuple[str, int | None]] = []

    def fake_display_saved_figure(path, *, width=None):
        displayed.append((Path(path).name, width))
        return Path(path)

    monkeypatch.setattr("src.tutorial_init.display_saved_figure", fake_display_saved_figure)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    save_and_show = make_save_and_show(tmp_path, write_pdf=True)
    path = save_and_show(fig, "example_plot.png", width=640)

    assert path == tmp_path / "example_plot.png"
    assert (tmp_path / "example_plot.png").exists()
    assert (tmp_path / "example_plot.pdf").exists()
    assert displayed == [("example_plot.png", 640)]

    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [0, 1])
    path2 = save_and_show(fig2, "no_pdf.png", write_pdf=False)
    assert path2 == tmp_path / "no_pdf.png"
    assert not (tmp_path / "no_pdf.pdf").exists()

    path3 = save_and_show(path2, 500)
    assert path3 == path2
    assert displayed[-1] == ("no_pdf.png", 500)


def test_make_save_and_show_accepts_custom_save_function(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from src.tutorial_init import make_save_and_show

    saved: list[tuple[str, bool | None]] = []

    def fake_display_saved_figure(path, *, width=None):
        return Path(path)

    def custom_save(fig, fig_dir, filename, *, write_pdf=None):
        path = Path(fig_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        saved.append((Path(filename).name, write_pdf))
        return path

    monkeypatch.setattr("src.tutorial_init.display_saved_figure", fake_display_saved_figure)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_and_show = make_save_and_show(tmp_path, write_pdf=True, save_fn=custom_save)

    assert save_and_show(fig, "custom.png") == tmp_path / "custom.png"
    assert saved == [("custom.png", True)]


def test_make_save_and_show_accepts_bound_save_function_without_fig_dir(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from src.tutorial_init import make_save_and_show

    displayed: list[Path] = []
    saved: list[tuple[str, bool | None]] = []

    def fake_display_saved_figure(path, *, width=None):
        displayed.append(Path(path))
        return Path(path)

    class TrackerLike:
        def __init__(self, fig_dir: Path):
            self.fig_dir = fig_dir

        def save_figure(self, fig, filename, *, write_pdf=None):
            path = self.fig_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            saved.append((Path(filename).name, write_pdf))
            return path

    monkeypatch.setattr("src.tutorial_init.display_saved_figure", fake_display_saved_figure)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    save_and_show = make_save_and_show(
        tmp_path,
        write_pdf=True,
        save_fn=TrackerLike(tmp_path).save_figure,
    )

    assert save_and_show(fig, "bound.png") == tmp_path / "bound.png"
    assert saved == [("bound.png", True)]
    assert displayed == [tmp_path / "bound.png"]


def test_make_save_and_show_accepts_save_function_without_write_pdf(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from src.tutorial_init import make_save_and_show

    displayed: list[Path] = []
    saved: list[str] = []

    def fake_display_saved_figure(path, *, width=None):
        displayed.append(Path(path))
        return Path(path)

    def save_without_write_pdf(fig, filename, close=True):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        saved.append(f"{Path(filename).name}:{close}")
        return path

    monkeypatch.setattr("src.tutorial_init.display_saved_figure", fake_display_saved_figure)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    save_and_show = make_save_and_show(tmp_path, write_pdf=False, save_fn=save_without_write_pdf)

    assert save_and_show(fig, "no_kw.png") == tmp_path / "no_kw.png"
    assert saved == ["no_kw.png:True"]
    assert displayed == [tmp_path / "no_kw.png"]


def test_make_save_and_show_preserves_multi_path_save_return(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from src.tutorial_init import make_save_and_show

    displayed: list[Path] = []

    def fake_display_saved_figure(path, *, width=None):
        displayed.append(Path(path))
        return Path(path)

    def save_many(fig, fig_dir, filename, *, write_pdf=None):
        stem = Path(filename).stem
        png = Path(fig_dir) / f"{stem}.png"
        svg = Path(fig_dir) / f"{stem}.svg"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png)
        svg.write_text("<svg></svg>")
        return [png, svg]

    monkeypatch.setattr("src.tutorial_init.display_saved_figure", fake_display_saved_figure)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_and_show = make_save_and_show(tmp_path, save_fn=save_many)

    paths = save_and_show(fig, "multi")
    assert paths == [tmp_path / "multi.png", tmp_path / "multi.svg"]
    assert displayed == [tmp_path / "multi.png"]


def test_make_save_and_show_displays_relative_multi_path_save_return(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from src.tutorial_init import make_save_and_show

    displayed: list[Path] = []

    def fake_display_saved_figure(path, *, width=None):
        displayed.append(Path(path))
        return Path(path)

    def save_relative(fig, fig_dir, filename, *, write_pdf=None):
        stem = Path(filename).stem
        png = Path(fig_dir) / f"{stem}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png)
        return [f"figures/ch02/{stem}.png", f"figures/ch02/{stem}.svg"]

    monkeypatch.setattr("src.tutorial_init.display_saved_figure", fake_display_saved_figure)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_and_show = make_save_and_show(tmp_path / "figures" / "ch02", save_fn=save_relative)

    paths = save_and_show(fig, "relative")
    assert paths == ["figures/ch02/relative.png", "figures/ch02/relative.svg"]
    assert displayed == [tmp_path / "figures" / "ch02" / "relative.png"]
