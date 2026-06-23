from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CH04_1_RETIRED_NOTEBOOK = "04_1_coupling_geometry.ipynb"
CH04_1_NOTEBOOK = "chapter4_1_coupling_geometry.ipynb"
CH04_2_RETIRED_NOTEBOOK = "04_2_state_space_representation_assumptions.ipynb"
CH04_2_NOTEBOOK = "chapter4_2_state_space_assumptions.ipynb"
CH04_3_RETIRED_NOTEBOOK = "04_3_sampling_depth_and_claim_boundaries.ipynb"
CH04_3_NOTEBOOK = "chapter4_3_sampling_depth.ipynb"

NOTEBOOKS = {
    CH04_1_NOTEBOOK: {
        "include": [
            "## Exp 1. Independent vs OT Coupling on EB",
            "## Exp 2. Sinkhorn Epsilon + Minibatch Ablation",
            "## Exp 3. Rectified Flow",
            "## Exp 4. Coupling Diagnostic Table",
        ],
        "exclude": [
            "## Exp 5. Toy Branching: Coupling -> Branch Leakage",
            "## Exp 6. Representation Space Changes Coupling",
            "## Exp 7. EB Representation Sensitivity: PC vs PHATE Coupling",
            "## Exp 8. Euclidean Chord vs Manifold-Aware Path",
            "## Exp 9. EB Equal-Depth Subsampling",
            "## Exp 9b. WFR-FM Sampling-Depth Sensitivity",
            "## Exp 10. Stochastic Bridge Demo",
            "## Exp 11. Prior Boundary Audit",
        ],
        "artifacts": [
            "fig4_1_independent_coupling_paths.png",
            "fig4_2_random_vs_ot_pairs.png",
            "fig4_2b_epsilon_ablation_pairs.png",
            "fig4_3_reflow_representative_trajectories.png",
            "fig4_5_random_vs_ot_projected_trajectories.png",
            "table4_1_path_geometry_diagnostics.csv",
            "table4_1_reflow_ablation.csv",
            "table4_A_sinkhorn_epsilon_ablation.csv",
        ],
    },
    CH04_2_NOTEBOOK: {
        "include": [
            "## Exp 5. Toy Branching: Coupling -> Branch Leakage",
            "## Exp 6. Representation Space Changes Coupling",
            "## Exp 7. EB Representation Sensitivity: PC vs PHATE Coupling",
            "## Exp 8. Euclidean Chord vs Manifold-Aware Path",
            "### Exp 8b. EB 20D PC Real-Data Manifold Diagnostic",
        ],
        "exclude": [
            "## Exp 1. Independent vs OT Coupling on EB",
            "## Exp 2. Sinkhorn Epsilon + Minibatch Ablation",
            "## Exp 3. Rectified Flow",
            "## Exp 4. Coupling Diagnostic Table",
            "## Exp 9. EB Equal-Depth Subsampling",
            "## Exp 9b. WFR-FM Sampling-Depth Sensitivity",
            "## Exp 10. Stochastic Bridge Demo",
            "## Exp 11. Prior Boundary Audit",
        ],
        "artifacts": [
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
            "table4_2_toy_branch_diagnostics.csv",
            "table4_3_representation_coupling_diagnostics.csv",
            "table4_4_state_space_model_metrics.csv",
            "table4_5_eb_representation_coupling_diagnostics.csv",
        ],
    },
    CH04_3_NOTEBOOK: {
        "include": [
            "## Exp 9. EB Equal-Depth Subsampling",
            "## Exp 9b. WFR-FM Sampling-Depth Sensitivity",
            "## Exp 10. Stochastic Bridge Demo",
            "## Exp 11. Prior Boundary Audit",
        ],
        "exclude": [
            "## Exp 1. Independent vs OT Coupling on EB",
            "## Exp 2. Sinkhorn Epsilon + Minibatch Ablation",
            "## Exp 3. Rectified Flow",
            "## Exp 4. Coupling Diagnostic Table",
            "## Exp 5. Toy Branching: Coupling -> Branch Leakage",
            "## Exp 6. Representation Space Changes Coupling",
            "## Exp 7. EB Representation Sensitivity: PC vs PHATE Coupling",
            "## Exp 8. Euclidean Chord vs Manifold-Aware Path",
        ],
        "artifacts": [
            "plot_raw_observed_counts(",
            "plot_sampling_depth_bootstrap_sensitivity",
            "plot_stochastic_bridge_demo",
            "plot_wfrfm_growth_delta_heatmap(",
            "plot_wfrfm_agreement_summary(",
            "table4_6_eb_downsampling_diagnostics.csv",
            "wfrfm_growth_by_bin",
            "wfrfm_sampling_sensitivity",
            "wfrfm_summary",
            "tableA_4_3_prior_boundary_audit.csv",
        ],
    },
}


def _text(path: Path) -> str:
    payload = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def _load_eb_cell_text(path: Path) -> str:
    payload = json.loads(path.read_text())
    for cell in payload["cells"]:
        source = "".join(cell.get("source", []))
        if "EB = load_eb_data()" in source:
            return source
    raise AssertionError(f"missing EB load cell in {path}")


def _code_text(filename: str) -> str:
    payload = json.loads((PROJECT_ROOT / "notebooks" / filename).read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )


def _markdown_text(filename: str) -> str:
    payload = json.loads((PROJECT_ROOT / "notebooks" / filename).read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "markdown"
    )


def test_ch04_1_canonical_notebook_exists_and_retired_active_copy_is_removed():
    assert (PROJECT_ROOT / "notebooks" / CH04_1_NOTEBOOK).exists()
    assert not (PROJECT_ROOT / "notebooks" / CH04_1_RETIRED_NOTEBOOK).exists()


def test_ch04_2_canonical_notebook_exists_and_retired_active_copy_is_removed():
    assert (PROJECT_ROOT / "notebooks" / CH04_2_NOTEBOOK).exists()
    assert not (PROJECT_ROOT / "notebooks" / CH04_2_RETIRED_NOTEBOOK).exists()


def test_ch04_3_canonical_notebook_exists_and_retired_active_copy_is_removed():
    assert (PROJECT_ROOT / "notebooks" / CH04_3_NOTEBOOK).exists()
    assert not (PROJECT_ROOT / "notebooks" / CH04_3_RETIRED_NOTEBOOK).exists()


def test_ch04_split_notebooks_cover_old_experiments_without_crossing_topics():
    for filename, spec in NOTEBOOKS.items():
        path = PROJECT_ROOT / "notebooks" / filename
        assert path.exists(), filename
        text = _text(path)
        for common in [
            "## 0. Setup",
            "## 1. Shared Utilities",
            "## 2. Load EB Data",
        ]:
            assert common in text
        for heading in spec["include"]:
            assert heading in text, (filename, heading)
        for heading in spec["exclude"]:
            assert heading not in text, (filename, heading)
        for artifact in spec["artifacts"]:
            assert artifact in text, (filename, artifact)


def test_ch04_split_notebook_experiment_headings_are_unique_across_splits():
    heading_to_file: dict[str, str] = {}
    for filename in NOTEBOOKS:
        text = _text(PROJECT_ROOT / "notebooks" / filename)
        for line in text.splitlines():
            if line.startswith("## Exp ") or line.startswith("### Exp "):
                assert line not in heading_to_file, (line, filename, heading_to_file[line])
                heading_to_file[line] = filename

    expected = {
        "## Exp 1. Independent vs OT Coupling on EB",
        "## Exp 2. Sinkhorn Epsilon + Minibatch Ablation",
        "## Exp 3. Rectified Flow",
        "## Exp 4. Coupling Diagnostic Table",
        "## Exp 5. Toy Branching: Coupling -> Branch Leakage",
        "## Exp 6. Representation Space Changes Coupling",
        "### Exp 6 Display: Representative Toy Endpoint Links",
        "### Exp 6 Display: Coupling and Model-Readout Summaries",
        "### Exp 6 Display: Shared Model-Readout Metrics",
        "## Exp 7. EB Representation Sensitivity: PC vs PHATE Coupling",
        "### Exp 7 Display: EB Coupling Links and Metric Summary",
        "## Exp 8. Euclidean Chord vs Manifold-Aware Path",
        "### Exp 8 Display: Toy Matched Chord and Graph Path",
        "### Exp 8b. EB 20D PC Real-Data Manifold Diagnostic",
        "### Exp 8b Display: Matched EB Path Examples",
        "### Exp 8b Display: EB Support-Distance Deltas",
        "## Exp 9. EB Equal-Depth Subsampling",
        "## Exp 9b. WFR-FM Sampling-Depth Sensitivity",
        "## Exp 10. Stochastic Bridge Demo",
        "## Exp 11. Prior Boundary Audit",
    }
    assert set(heading_to_file) == expected


def test_ch04_split_notebooks_define_local_training_helper_when_used():
    for filename in NOTEBOOKS:
        text = _text(PROJECT_ROOT / "notebooks" / filename)
        if "train_or_load_model(" in text:
            assert (
                "def train_or_load_model(" in text
                or "ch04t.train_or_load_model" in text
                or "ch04_exp.train_or_load_model" in text
            ), filename


def test_ch04_1_rewrite_keeps_reader_facing_flow_clean():
    payload = json.loads((PROJECT_ROOT / "notebooks" / CH04_1_NOTEBOOK).read_text())
    code_sources = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    code_text = "\n".join(code_sources)
    markdown_text = _markdown_text(CH04_1_NOTEBOOK)

    assert len(payload["cells"]) <= 72
    assert "Path(\"/home/xmabs" not in code_text
    assert "artifact_manifest" not in code_text
    assert "run_config =" not in code_text
    assert "Project root:" not in code_text

    for index, source in enumerate(code_sources, 1):
        assert source.strip(), (CH04_1_NOTEBOOK, index)
        assert not all(
            line.strip().startswith("#") or not line.strip()
            for line in source.splitlines()
        ), (CH04_1_NOTEBOOK, index)

    for noisy_name in [
        "display_table(",
        "run_mode = pd.DataFrame",
        "artifact_locations = pd.DataFrame",
        '{"setting": "project_root"',
        "display(PROJECT_ROOT",
    ]:
        assert noisy_name not in code_text, noisy_name

    for teaching_phrase in [
        "couplings are training assumptions",
        "OT-CFM changes which endpoints are paired",
        "Sinkhorn epsilon",
        "Rectified flow",
    ]:
        assert teaching_phrase.lower() in markdown_text.lower(), teaching_phrase


def test_ch04_1_rewrite_uses_compact_figure_cells():
    notebook_text = _text(PROJECT_ROOT / "notebooks" / CH04_1_NOTEBOOK)
    code_text = _code_text(CH04_1_NOTEBOOK)

    assert "from src.tutorial_init import apply_tutorial_plot_style, bootstrap, make_ch04_run_config, make_save_and_show" in code_text
    assert "config = make_ch04_run_config()" in code_text
    assert "save_and_show = make_save_and_show(" in code_text
    assert code_text.count("save_and_show(") >= 2

    for required_artifact in [
        "fig4_1_independent_coupling_paths.png",
        "fig4_2_random_vs_ot_pairs.png",
        "fig4_2b_epsilon_ablation_pairs.png",
        "fig4_3_reflow_representative_trajectories.png",
        "fig4_5_random_vs_ot_projected_trajectories.png",
        "table4_1_path_geometry_diagnostics.csv",
        "table4_1_reflow_ablation.csv",
        "table4_A_sinkhorn_epsilon_ablation.csv",
    ]:
        assert required_artifact in notebook_text, required_artifact


def test_ch04_1_rewrite_outputs_do_not_show_absolute_paths():
    payload = json.loads((PROJECT_ROOT / "notebooks" / CH04_1_NOTEBOOK).read_text())
    for cell_index, cell in enumerate(payload["cells"]):
        for output in cell.get("outputs", []):
            chunks = []
            chunks.extend(output.get("text", []))
            data = output.get("data") or {}
            for key in ["text/plain", "text/html"]:
                value = data.get(key, "")
                chunks.extend(value if isinstance(value, list) else [str(value)])
            output_text = "".join(chunks)
            assert "/home/xmabs" not in output_text, cell_index
            assert "/import/" not in output_text, cell_index


def test_ch04_split_notebooks_define_eb_aliases_before_using_them():
    def source_defines_alias(source: str, alias: str) -> bool:
        for line in source.splitlines():
            if "=" not in line or "==" in line:
                continue
            left_side = line.split("=", 1)[0]
            names = [part.strip() for part in left_side.split(",")]
            if alias in names:
                return True
        return False

    for filename in NOTEBOOKS:
        path = PROJECT_ROOT / "notebooks" / filename
        payload = json.loads(path.read_text())
        sources = ["".join(cell.get("source", [])) for cell in payload["cells"]]
        for alias in ["X0_eb", "X1_eb", "X0p_eb", "X1p_eb"]:
            first_use = next((i for i, source in enumerate(sources) if alias in source), None)
            if first_use is None:
                continue
            first_definition = next(
                (i for i, source in enumerate(sources) if source_defines_alias(source, alias)),
                None,
            )
            assert first_definition is not None, (filename, alias)
            assert first_definition <= first_use, (filename, alias, first_definition, first_use)


def test_ch04_2_exp8b_uses_cached_full_eb_artifacts_by_default():
    text = _text(PROJECT_ROOT / "notebooks" / CH04_2_NOTEBOOK)
    assert "CH04_RECOMPUTE_EXP8B" in text
    assert "exp8_eb_off_manifold_stats.csv" in text
    assert "exp8_eb_pair_selection_diagnostics.csv" in text


def test_ch04_2_rewrite_removes_ppt_small_figure_section_from_reader_flow():
    text = _text(PROJECT_ROOT / "notebooks" / CH04_2_NOTEBOOK)

    assert "## PPT-ready small figures for Chapter 4.2" not in text
    for ppt_filename in [
        "fig4_2_toy_pca30_representative_pairs.png",
        "fig4_2_toy_program4_representative_pairs.png",
        "fig4_3_eb_off_manifold_positive_fraction.png",
    ]:
        assert ppt_filename in text


def test_ch04_2_rewrite_collapses_exp8b_cache_guard_noise():
    code_text = _code_text(CH04_2_NOTEBOOK)

    assert "EXP8B_CACHE_READY" not in code_text
    assert "exp8b_missing" not in code_text
    assert "if not EXP8B_CACHE_READY" not in code_text
    assert code_text.count("CH04_RECOMPUTE_EXP8B") <= 2


def test_ch04_2_rewrite_has_no_empty_or_comment_only_code_cells():
    payload = json.loads((PROJECT_ROOT / "notebooks" / CH04_2_NOTEBOOK).read_text())
    code_sources = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]

    assert len(payload["cells"]) <= 60
    for index, source in enumerate(code_sources, 1):
        assert source.strip(), (CH04_2_NOTEBOOK, index)
        assert not all(
            line.strip().startswith("#") or not line.strip()
            for line in source.splitlines()
        ), (CH04_2_NOTEBOOK, index)


def test_ch04_2_uses_shared_save_and_show_helper_for_figure_cells():
    code_text = _code_text(CH04_2_NOTEBOOK)

    assert "save_small_figure =" in code_text
    assert "plot_representative_endpoint_pairs(" in code_text
    assert "plot_delta_distribution(" in code_text
    for generated_small_figure in [
        "fig4_2_toy_representation_coupling_summary.png",
        "fig4_2_state_space_model_readout_summary.png",
        "fig4_3_eb_off_manifold_positive_fraction.png",
    ]:
        assert generated_small_figure in code_text
    for old_composite in [
        "fig4_5b_toy_branching_pairs.png",
        "fig4_8_toy_representation_couplings.png",
        "fig4_8b_eb_pc_vs_phate_coupling.png",
        "fig4_10_chord_vs_manifold_path.png",
        "fig4_10_supp_off_manifold_statistics.png",
        "fig4_10_eb_chord_vs_graph_path_phate.png",
        "fig4_10_eb_off_manifold_statistics.png",
    ]:
        assert old_composite not in code_text


def test_ch04_2_small_figure_polish_avoids_overlap_prone_annotations():
    code_text = _code_text(CH04_2_NOTEBOOK)
    state_space_source = (PROJECT_ROOT / "src" / "visualization" / "state_space.py").read_text()

    assert 'f"mean {mean_val' not in state_space_source
    assert '"mean:"' in state_space_source
    assert "subplots_adjust(bottom=" in state_space_source
    assert 'loc="lower right"' not in code_text
    assert 'ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34)' in code_text
    assert '"#6BAED6"' not in code_text
    assert "fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.65))" in code_text
    assert 'metric_specs = [\n    ("native_endpoint_mmd", "Native MMD", 4),' not in code_text
    assert "coupling_bar_specs = [" in code_text
    assert "model_bar_specs = [" in code_text
    assert "cross_table = cross_ax.table(" not in code_text
    assert "native_table = native_ax.table(" not in code_text
    assert "metric_ax.hlines(" not in code_text


def test_ch04_2_shared_utility_setup_is_consolidated():
    payload = json.loads((PROJECT_ROOT / "notebooks" / CH04_2_NOTEBOOK).read_text())
    cells = payload["cells"]
    utility_heading_index = next(
        i
        for i, cell in enumerate(cells)
        if cell.get("cell_type") == "markdown"
        and "## 1. Shared Utilities" in "".join(cell.get("source", []))
    )
    next_heading_index = next(
        i
        for i, cell in enumerate(cells[utility_heading_index + 1 :], utility_heading_index + 1)
        if cell.get("cell_type") == "markdown"
        and "".join(cell.get("source", [])).startswith("## ")
    )
    utility_code_cells = [
        cell for cell in cells[utility_heading_index + 1 : next_heading_index]
        if cell.get("cell_type") == "code"
    ]

    assert len(utility_code_cells) <= 2
