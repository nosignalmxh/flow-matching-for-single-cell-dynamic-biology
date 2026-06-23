from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


METHOD_LABELS = {
    "baseline": "Baseline method",
    "model_a": "Model A",
    "model_b": "Model B",
}


def test_metric_table_for_split_orders_methods_and_reports_missing_labels():
    from src.visualization.perturbation import metric_table_for_split, metric_value_table

    summary = pd.DataFrame(
        [
            {"split_name": "heldout", "method": "model_b", "program_readout_mmd": 0.2, "program_readout_sliced_w2": 1.2},
            {"split_name": "heldout", "method": "model_a", "program_readout_mmd": 0.1, "program_readout_sliced_w2": 1.1},
            {"split_name": "other", "method": "baseline", "program_readout_mmd": 0.9, "program_readout_sliced_w2": 1.9},
        ]
    )

    frame, missing = metric_table_for_split(summary, "heldout", ["baseline", "model_a", "model_b"], METHOD_LABELS)

    assert frame["method"].tolist() == ["model_a", "model_b"]
    assert frame["method_label"].tolist() == ["Model A", "Model B"]
    assert missing == ["Baseline method"]
    assert metric_value_table(frame).to_dict(orient="list") == {
        "method": ["Model A", "Model B"],
        "MMD": [0.1, 0.2],
        "Sliced W2": [1.1, 1.2],
    }


def test_make_metric_display_table_from_summary_validates_expected_values(tmp_path):
    from src.visualization.perturbation import make_metric_display_table_from_summary

    source = tmp_path / "summary.json"
    rows = [
        {"split_name": "heldout", "method": "model_a", "program_readout_mmd": 0.12344, "program_readout_sliced_w2": 1.2344},
        {"split_name": "heldout", "method": "model_b", "program_readout_mmd": 0.56789, "program_readout_sliced_w2": 2.3456},
    ]
    source.write_text(json.dumps({"key_metrics": {"sciplex_summary": rows}}), encoding="utf-8")

    expected = pd.DataFrame(
        [
            {"method": "model_a", "MMD": 0.1234, "Sliced W2": 1.234},
            {"method": "model_b", "MMD": 0.5679, "Sliced W2": 2.346},
        ]
    )
    frame = make_metric_display_table_from_summary(
        source,
        "heldout",
        ["model_a", "model_b"],
        expected,
        METHOD_LABELS,
        project_root=tmp_path,
    )

    assert frame["method_label"].tolist() == ["Model A", "Model B"]
    assert frame["metric_display_source"].tolist() == ["summary.json", "summary.json"]

    with pytest.raises(ValueError, match="Display metrics do not match"):
        make_metric_display_table_from_summary(
            source,
            "heldout",
            ["model_a", "model_b"],
            expected.assign(MMD=[0.0, 0.0]),
            METHOD_LABELS,
            project_root=tmp_path,
        )


def test_build_section52_metric_display_tables_centralizes_manuscript_values(tmp_path):
    from src.visualization.perturbation import build_section52_metric_display_tables, metric_value_table

    raw_summary = [
        {"split_name": "Split B held-out highest dose", "method": "M1_unconditional", "program_readout_mmd": 0.50, "program_readout_sliced_w2": 1.50},
        {"split_name": "Split B held-out highest dose", "method": "M3_no_chemistry", "program_readout_mmd": 0.25, "program_readout_sliced_w2": 1.25},
        {"split_name": "Split C held-out compound", "method": "M1_unconditional", "program_readout_mmd": 0.75, "program_readout_sliced_w2": 1.75},
    ]
    manuscript_rows = [
        {"split_name": "Split B held-out highest dose", "method": "M1_unconditional", "program_readout_mmd": 0.02254, "program_readout_sliced_w2": 0.3948},
        {"split_name": "Split B held-out highest dose", "method": "M2_per_compound", "program_readout_mmd": 0.02424, "program_readout_sliced_w2": 0.3811},
        {"split_name": "Split B held-out highest dose", "method": "M3_no_chemistry", "program_readout_mmd": 0.01749, "program_readout_sliced_w2": 0.3559},
        {"split_name": "Split B held-out highest dose", "method": "vehicle_as_prediction", "program_readout_mmd": 0.02194, "program_readout_sliced_w2": 0.3721},
        {"split_name": "Split B held-out highest dose", "method": "mean_shift", "program_readout_mmd": 0.02504, "program_readout_sliced_w2": 0.3814},
        {"split_name": "Split C held-out compound", "method": "M1_unconditional", "program_readout_mmd": 0.09829, "program_readout_sliced_w2": 1.0009},
        {"split_name": "Split C held-out compound", "method": "M3_no_chemistry", "program_readout_mmd": 0.08414, "program_readout_sliced_w2": 0.9648},
        {"split_name": "Split C held-out compound", "method": "M4_chemistry_aware", "program_readout_mmd": 0.06189, "program_readout_sliced_w2": 0.8032},
        {"split_name": "Split C held-out compound", "method": "vehicle_as_prediction", "program_readout_mmd": 0.06349, "program_readout_sliced_w2": 0.7851},
        {"split_name": "Split C held-out compound", "method": "mean_shift", "program_readout_mmd": 0.07771, "program_readout_sliced_w2": 0.8244},
        {"split_name": "Split C held-out compound", "method": "nearest_chemistry", "program_readout_mmd": 0.08668, "program_readout_sliced_w2": 0.8331},
    ]
    source = tmp_path / "run_summary.json"
    source.write_text(json.dumps({"key_metrics": {"sciplex_summary": manuscript_rows}}), encoding="utf-8")

    package = build_section52_metric_display_tables(raw_summary, source, project_root=tmp_path)

    assert package.split_b_metric_table["method"].tolist() == ["M1_unconditional", "M3_no_chemistry"]
    assert package.split_b_missing == ["M2 one-flow-per-compound", "Vehicle baseline", "Mean-shift baseline"]
    assert package.split_c_missing == [
        "M3 one-hot+dose conditional FM",
        "M4 RDKit2D+dose conditional FM",
        "Vehicle baseline",
        "Mean-shift baseline",
        "Nearest-chemistry baseline",
    ]
    assert package.manuscript_metric_source == "run_summary.json"
    assert "Held-out highest dose missing from raw summary" in package.missing_result_notes[0]
    assert package.missing_result_notes[-1].startswith("M2 one-flow-per-compound is not plotted")
    assert metric_value_table(package.split_b_metric_display)["MMD"].tolist() == [0.0225, 0.0242, 0.0175, 0.0219, 0.025]


def test_wrapped_labels_are_stable():
    from src.visualization.perturbation import short_compound_label, wrapped_method_label

    assert wrapped_method_label("model_a", METHOD_LABELS, width=20) == "Model A"
    assert "\n" in wrapped_method_label("very_long_method_name_without_label", METHOD_LABELS, width=10)
    assert short_compound_label("Compound Name (high dose)", width=16) == "Compound Name\n(high dose)"


def test_section52_helpers_expose_display_constants_and_config(tmp_path, monkeypatch):
    from src.visualization import perturbation as tutorial

    monkeypatch.setenv("CH05_SEED", "7")
    monkeypatch.setenv("CH05_QUICK", "1")
    monkeypatch.setenv("CH05_TRAINING_STEPS", "123")
    monkeypatch.setenv("CH05_BATCH_SIZE", "64")
    monkeypatch.setenv("CH05_NFE", "8")
    monkeypatch.setenv("CH05_MAX_EVAL_GROUPS", "3")
    (tmp_path / "src" / "experiments").mkdir(parents=True)
    (tmp_path / "src" / "experiments" / "perturbation.py").write_text("# marker\n", encoding="utf-8")

    config = tutorial.make_section52_config(project_root=tmp_path, device="cpu")

    assert config.project_root == tmp_path.resolve()
    assert config.fig_dir == tmp_path / "figures" / "ch05" / "new2"
    assert config.table_dir == tmp_path / "tables" / "ch05"
    assert config.output_dir == tmp_path / "outputs" / "ch05"
    assert config.default_seed == 7
    assert config.quick_mode is True
    assert config.training_steps == 123
    assert config.batch_size == 64
    assert config.nfe == 8
    assert config.max_eval_groups == 3
    assert tutorial.METHOD_LABELS["M4_chemistry_aware"] == "M4 RDKit2D+dose conditional FM"
    assert "fig_5_2_model_designs" in tutorial.FIGURE_TITLES


def test_section52_save_figure_pair_writes_png_and_pdf(tmp_path):
    from src.visualization.perturbation import save_figure_pair

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    paths = save_figure_pair(fig, tmp_path, "demo_figure")

    assert paths["png"] == tmp_path / "demo_figure.png"
    assert paths["pdf"] == tmp_path / "demo_figure.pdf"
    assert paths["png"].exists()
    assert paths["pdf"].exists()


def test_section52_m2_design_labels_do_not_overlap_velocity_box():
    from matplotlib.patches import FancyBboxPatch

    from src.visualization.perturbation import draw_method_tile

    fig, ax = plt.subplots(figsize=(4.1, 2.325))
    draw_method_tile(
        ax,
        "M2_per_compound",
        r"$v_\theta^{(c)}(x,\tau)$",
        "compound-specific field; no cross-compound sharing",
        seed=42,
        seed_offset=101,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    velocity_box = [
        patch
        for patch in ax.patches
        if isinstance(patch, FancyBboxPatch) and abs(patch.get_x() - 0.365) < 1e-6 and abs(patch.get_y() - 0.420) < 1e-6
    ][0]
    velocity_box_bbox = velocity_box.get_window_extent(renderer)
    compound_labels = [text for text in ax.texts if text.get_text() in {"c1", "c2", "c3"}]

    assert len(compound_labels) == 3
    assert all(not label.get_window_extent(renderer).overlaps(velocity_box_bbox) for label in compound_labels)
    plt.close(fig)


def test_section52_split_status_matrix_marks_train_test_and_missing():
    from src.visualization.perturbation import split_status_matrix

    split_meta = pd.DataFrame(
        [
            {"compound": "A", "dose": 10.0, "is_vehicle": False, "split": "train"},
            {"compound": "A", "dose": 100.0, "is_vehicle": False, "split": "test"},
            {"compound": "B", "dose": 10.0, "is_vehicle": False, "split": "train"},
            {"compound": "vehicle", "dose": 0.0, "is_vehicle": True, "split": "train"},
        ]
    )

    status = split_status_matrix(split_meta, compounds=["A", "B"], doses=[10.0, 100.0])

    assert status.loc["A", 10.0] == "train"
    assert status.loc["A", 100.0] == "test"
    assert status.loc["B", 10.0] == "train"
    assert status.loc["B", 100.0] == "missing"


def test_section52_manifest_builder_checks_finite_metrics_and_artifact_paths(tmp_path):
    from src.visualization.perturbation import Section52Config, build_section52_run_summary

    for directory in ["figures/ch05/new2", "tables/ch05", "outputs/ch05"]:
        (tmp_path / directory).mkdir(parents=True)
    config = Section52Config(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        fig_dir=tmp_path / "figures" / "ch05" / "new2",
        table_dir=tmp_path / "tables" / "ch05",
        output_dir=tmp_path / "outputs" / "ch05",
        default_seed=42,
        quick_mode=False,
        training_steps=6000,
        batch_size=256,
        nfe=32,
        sciplex_download_in_ch05=False,
        sciplex_synthetic_if_missing=False,
        max_eval_groups=None,
        device="cpu",
    )
    figure_paths = {}
    for stem in [
        "fig_5_2_model_designs",
        "fig_5_2_evaluation_splits",
        "fig_5_2_heldout_highest_dose_metrics",
        "fig_5_2_heldout_compound_metrics",
        "fig_5_2_alisertib_example",
    ]:
        png = config.fig_dir / f"{stem}.png"
        pdf = config.fig_dir / f"{stem}.pdf"
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")
        figure_paths[stem] = {"png": png, "pdf": pdf}
    for rel in [
        "tables/ch05/tab_5_2_sciplex_splits.csv",
        "outputs/ch05/rdkit2d_compound_features.npz",
        "outputs/ch05/rdkit2d_diagnostics.json",
        "outputs/ch05/rdkit2d_audit.csv",
        "outputs/ch05/sciplex_metrics_by_group.csv",
        "outputs/ch05/sciplex_metrics_summary.csv",
        "outputs/ch05/real_data_audit.json",
        "outputs/ch05/run_summary_perturbation_sciplex.json",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    metric = pd.DataFrame({"method": ["M1_unconditional"], "MMD": [0.1], "Sliced W2": [0.2]})
    run_summary = {"sciplex_data": {"summary": {"is_synthetic": False, "source": "real"}}}

    summary, required_paths, finite_checks, title_audit = build_section52_run_summary(
        run_summary=run_summary,
        config=config,
        figure_paths=figure_paths,
        metric_frames={"metric": metric},
        split_b_metric_display=metric,
        split_c_metric_display=metric,
        representative_key=("Alisertib", np.float64(100.0)),
        manuscript_metric_source="outputs/ch05/run_summary.json",
        missing_result_notes=[],
    )

    assert summary["no_panel_letter_titles"] is True
    assert finite_checks == {"metric": True}
    assert all(Path(path).is_absolute() for path in required_paths)
    assert title_audit["has_panel_letter_prefix"].eq(False).all()
    assert run_summary["splits_evaluated"] == [
        "Split B held-out highest dose",
        "Split C held-out compound",
    ]
