from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_artifact_helpers_roundtrip_json_csv_npz_and_scalars(tmp_path):
    from src.artifacts import json_ready, load_json, load_npz, save_csv, save_json, save_npz

    payload = {
        "path": Path("figures/example.png"),
        "scalar": np.float32(1.25),
        "array": np.asarray([1, 2, 3], dtype=np.int64),
        "frame": pd.DataFrame({"x": [1, 2], "y": [3.5, 4.5]}),
    }

    ready = json_ready(payload)
    assert ready == {
        "path": "figures/example.png",
        "scalar": pytest.approx(1.25),
        "array": [1, 2, 3],
        "frame": [{"x": 1, "y": 3.5}, {"x": 2, "y": 4.5}],
    }

    json_path = save_json(tmp_path / "nested" / "payload.json", payload)
    assert load_json(json_path)["array"] == [1, 2, 3]

    csv_path = save_csv(tmp_path / "tables" / "table.csv", payload["frame"])
    assert pd.read_csv(csv_path).to_dict(orient="list") == {"x": [1, 2], "y": [3.5, 4.5]}

    npz_path = save_npz(tmp_path / "cache" / "arrays.npz", values=np.asarray([4, 5], dtype=np.float32))
    with load_npz(npz_path) as loaded:
        np.testing.assert_allclose(loaded["values"], np.asarray([4, 5], dtype=np.float32))


def test_artifact_helpers_validate_arrays_paths_and_hashes(tmp_path):
    from src.artifacts import artifact_exists, ensure_finite, sample_rows, stable_hash

    existing = tmp_path / "nonempty.txt"
    existing.write_text("x")
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    assert artifact_exists(existing)
    assert not artifact_exists(empty)
    assert not artifact_exists(tmp_path / "missing.txt")

    np.testing.assert_array_equal(sample_rows(5, None, seed=7), np.arange(5))
    np.testing.assert_array_equal(sample_rows(10, 4, seed=7), np.asarray([5, 6, 8, 9]))
    assert stable_hash("a", 1, Path("b")) == "8b0646ff98"

    ensure_finite("ok", np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="bad contains non-finite values"):
        ensure_finite("bad", np.asarray([np.nan]))


def test_artifact_helpers_save_figures_and_paper_tables(tmp_path):
    import matplotlib.pyplot as plt

    from src.artifacts import figure_paths_from_name, save_figure, save_figure_formats, save_paper_table

    fig_dir = tmp_path / "figures"
    png_path, pdf_path, stem = figure_paths_from_name(fig_dir, "example_plot.png")
    assert stem == "example_plot"
    assert png_path == fig_dir / "example_plot.png"
    assert pdf_path == fig_dir / "example_plot.pdf"

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    saved_png = save_figure(fig, fig_dir, "example_plot.png", dpi=72, write_pdf=True)
    plt.close(fig)
    assert saved_png == png_path
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [0, 1])
    multi_paths = save_figure_formats(fig2, fig_dir, "multi_format", formats=("png", "svg"), dpi=72, close=True)
    assert [path.name for path in multi_paths] == ["multi_format.png", "multi_format.svg"]
    assert all(path.exists() and path.stat().st_size > 0 for path in multi_paths)
    assert not plt.fignum_exists(fig2.number)

    csv_path, tex_path, md_path = save_paper_table(
        tmp_path / "tables" / "summary",
        pd.DataFrame({"metric": ["mmd"], "value": [0.125]}),
    )
    assert csv_path.read_text().splitlines()[0] == "metric,value"
    assert "mmd" in tex_path.read_text()
    assert "mmd" in md_path.read_text()


def test_artifact_helpers_resolve_and_remember_sources(tmp_path):
    from src.artifacts import remember_source, resolve_required_artifact, safe_relpath

    project_root = tmp_path / "project"
    nested = project_root / "outputs" / "ch04"
    nested.mkdir(parents=True)
    artifact = nested / "diagnostic.csv"
    artifact.write_text("x,y\n1,2\n")

    assert safe_relpath(artifact, root=project_root) == "outputs/ch04/diagnostic.csv"
    assert resolve_required_artifact("diagnostic.csv", preferred_dirs=[], search_root=project_root) == artifact

    sources = {}
    remembered = remember_source(sources, "diagnostic", artifact, root=project_root)
    assert remembered == artifact
    assert sources == {"diagnostic": "outputs/ch04/diagnostic.csv"}


def test_utils_resolve_project_root_accepts_reorganized_src(monkeypatch, tmp_path):
    from src.utils import resolve_project_root

    project = tmp_path / "project"
    nested = project / "notebooks" / "scratch"
    (project / "src" / "core").mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert not (project / "src" / "models.py").exists()
    assert resolve_project_root() == project.resolve()


def test_flow_runtime_euler_helpers_preserve_zero_velocity():
    torch = pytest.importorskip("torch")

    from src.experiments.flow_runtime import coarse_step_error, make_time_batch, rollout_euler, trajectory_rollout

    class ZeroVelocity(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = ZeroVelocity()
    x0 = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    batch_t = make_time_batch(3, device=torch.device("cpu"))
    assert batch_t.shape == (3, 1)
    assert batch_t.device.type == "cpu"

    endpoint = rollout_euler(model, x0, nfe=4, device=torch.device("cpu"))
    np.testing.assert_allclose(endpoint, x0)

    endpoint2, traj, times = trajectory_rollout(model, x0, nfe=4, device=torch.device("cpu"))
    np.testing.assert_allclose(endpoint2, x0)
    assert traj.shape == (5, 2, 2)
    np.testing.assert_allclose(traj[0], x0)
    np.testing.assert_allclose(traj[-1], x0)
    np.testing.assert_allclose(times, np.linspace(0.0, 1.0, 5))

    assert coarse_step_error(model, x0, nfe_coarse=2, nfe_fine=4, device=torch.device("cpu")) == 0.0
