from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    "chapter4_1_coupling_geometry.ipynb": {
        "min_code_cells": 24,
        "must_display": [
            "fig4_1_independent_coupling_paths.png",
            "fig4_2_random_vs_ot_pairs.png",
            "fig4_2b_epsilon_ablation_pairs.png",
            "fig4_3_reflow_representative_trajectories.png",
        ],
    },
    "chapter4_2_state_space_assumptions.ipynb": {
        "min_code_cells": 21,
        "must_display": [
            "fig4_2_toy_pca30_representative_pairs.png",
            "fig4_2_toy_program4_representative_pairs.png",
            "fig4_2_toy_representation_coupling_summary.png",
            "fig4_2_eb_pc20_coupling_representative_pairs.png",
            "fig4_2_eb_phate_diagnostic_coupling_representative_pairs.png",
            "fig4_2_eb_pc_vs_phate_distance_summary.png",
            "fig4_2_state_space_model_readout_summary.png",
            "fig4_3_toy_single_pair_chord_vs_graph_path.png",
            "fig4_3_eb_chord_vs_graph_matched_examples.png",
            "fig4_3_eb_density_radius_delta.png",
            "fig4_3_eb_knn_radius_delta.png",
            "fig4_3_eb_off_manifold_positive_fraction.png",
        ],
    },
    "chapter4_3_sampling_depth.ipynb": {
        "min_code_cells": 18,
        "must_display": [
            "plot_raw_observed_counts(",
            "plot_sampling_depth_bootstrap_sensitivity",
            "plot_stochastic_bridge_demo",
        ],
    },
    "chapter5_2_perturbation_sciplex.ipynb": {
        "min_code_cells": 11,
        "must_display": [
            "fig_5_2_heldout_highest_dose_metrics",
            "fig_5_2_heldout_compound_metrics",
        ],
    },
    "chapter5_1_timecourse_suite.ipynb": {
        "min_code_cells": 18,
        "must_display": [
            "fig5_1_time_pair_designs.png",
            "fig5_1_hidden_t2_recovery.png",
            "fig5_1_seen_t4_rollout.png",
            "fig5_1_velocity_jump.png",
        ],
    },
}


def _payload(filename: str) -> dict:
    return json.loads((PROJECT_ROOT / "notebooks" / filename).read_text())


def _sources(filename: str, cell_type: str | None = None) -> list[str]:
    payload = _payload(filename)
    return [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell_type is None or cell.get("cell_type") == cell_type
    ]


def test_tutorial_notebooks_have_stepwise_code_cells_and_compile():
    for filename, spec in NOTEBOOKS.items():
        code_sources = _sources(filename, "code")
        code_lengths = [source.count("\n") + bool(source) for source in code_sources]

        assert len(code_sources) >= spec["min_code_cells"], filename
        assert max(code_lengths) <= 100, (filename, max(code_lengths))

        for index, source in enumerate(code_sources, 1):
            compile(source, f"{filename}:code-cell-{index}", "exec")


def test_tutorial_notebooks_display_their_saved_figures_inline():
    for filename, spec in NOTEBOOKS.items():
        notebook_text = "\n".join(_sources(filename))
        code_text = "\n".join(_sources(filename, "code"))
        display_support_markers = [
            "from IPython.display import Image, display",
            "display_png(",
            "display_saved_figure(",
            "display_saved_figures(",
            "display_figure(",
            "display_figure_output(",
            "make_save_and_show(",
        ]
        assert any(marker in code_text for marker in display_support_markers), filename

        for figure_name in spec["must_display"]:
            assert figure_name in notebook_text, (filename, figure_name)

        display_markers = [
            "display(Image(",
            "display_saved_figure(",
            "display_saved_figures(",
            "display_figure(",
            "display_figure_output(",
            "display_png(",
            "make_save_and_show(",
        ]
        assert any(marker in code_text for marker in display_markers), filename


def test_tutorial_notebooks_keep_saved_output_references():
    for filename, spec in NOTEBOOKS.items():
        notebook_text = "\n".join(_sources(filename))
        code_text = "\n".join(_sources(filename, "code"))

        for figure_name in spec["must_display"]:
            assert figure_name in notebook_text, (filename, figure_name)

        output_guard_markers = [
            "raise FileNotFoundError",
            "resolve_required_artifact",
            "section52_required_paths",
            "display_saved_figure",
            "save_figure",
            "save_small_figure",
            "write_section51_artifacts",
            "register_fig5_1_artifacts",
        ]
        assert any(marker in code_text for marker in output_guard_markers), filename
