from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    "04_1_coupling_geometry.ipynb": {
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
            "fig4_4_reflow_trajectories.png",
            "fig4_5_random_vs_ot_projected_trajectories.png",
            "table4_1_path_geometry_diagnostics.csv",
            "table4_1_reflow_ablation.csv",
            "table4_A_sinkhorn_epsilon_ablation.csv",
        ],
    },
    "04_2_state_space_representation_assumptions.ipynb": {
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
            "fig4_5_random_vs_ot_projected_trajectories.png",
            "fig4_8_toy_representation_couplings.png",
            "fig4_10_chord_vs_manifold_path.png",
            "fig4_10_eb_chord_vs_graph_path_phate.png",
            "table4_2_toy_branch_diagnostics.csv",
            "table4_3_representation_coupling_diagnostics.csv",
            "table4_4_state_space_model_metrics.csv",
            "table4_5_eb_representation_coupling_diagnostics.csv",
        ],
    },
    "04_3_sampling_depth_and_claim_boundaries.ipynb": {
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
            "fig4_11a_eb_observed_counts.png",
            "fig4_11b_sampling_depth_sensitivity.png",
            "fig4_11c_stochastic_bridge_demo.png",
            "fig4_11d_wfrfm_growth_sensitivity",
            "figA_4_1_prior_strength_sanity_check.png",
            "table4_6_eb_downsampling_diagnostics.csv",
            "table4_6c_wfrfm_growth_by_bin",
            "table4_6d_wfrfm_sampling_sensitivity",
            "wfrfm_sampling_sensitivity_summary",
            "tableA_4_3_prior_boundary_audit.csv",
        ],
    },
}


def _text(path: Path) -> str:
    payload = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def _manifest_cell_text(path: Path) -> str:
    payload = json.loads(path.read_text())
    for cell in payload["cells"]:
        source = "".join(cell.get("source", []))
        if "artifact_manifest_04_" in source:
            return source
    raise AssertionError(f"missing split-specific artifact manifest cell in {path}")


def _load_eb_cell_text(path: Path) -> str:
    payload = json.loads(path.read_text())
    for cell in payload["cells"]:
        source = "".join(cell.get("source", []))
        if "EB = load_eb_data()" in source:
            return source
    raise AssertionError(f"missing EB load cell in {path}")


def test_ch04_split_notebooks_cover_old_experiments_without_crossing_topics():
    for filename, spec in NOTEBOOKS.items():
        path = PROJECT_ROOT / "notebooks" / filename
        assert path.exists(), filename
        text = _text(path)
        manifest_text = _manifest_cell_text(path)

        for common in [
            "## 0. Setup",
            "## 1. Shared Utilities",
            "## 2. Load EB Data",
            "## Artifact Manifest",
        ]:
            assert common in text
        for heading in spec["include"]:
            assert heading in text, (filename, heading)
        for heading in spec["exclude"]:
            assert heading not in text, (filename, heading)
        for artifact in spec["artifacts"]:
            assert artifact in manifest_text, (filename, artifact)


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
        "## Exp 7. EB Representation Sensitivity: PC vs PHATE Coupling",
        "## Exp 8. Euclidean Chord vs Manifold-Aware Path",
        "### Exp 8b. EB 20D PC Real-Data Manifold Diagnostic",
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
            assert "def train_or_load_model(" in text or "ch04t.train_or_load_model" in text, filename


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
    text = _text(PROJECT_ROOT / "notebooks" / "04_2_state_space_representation_assumptions.ipynb")
    assert "CH04_RECOMPUTE_EXP8B" in text
    assert "exp8_eb_off_manifold_stats.csv" in text
    assert "fig4_10_eb_chord_vs_graph_path_phate.png" in text
