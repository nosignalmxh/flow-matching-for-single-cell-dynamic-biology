from __future__ import annotations

import ast
import json
from pathlib import Path


def test_ch04_2_imports_extracted_helper_modules():
    from src.ch04_manifold_diagnostics import candidate_path_diagnostics, stats_for_selected_pairs
    from src.ch04_state_space_figures import plot_delta_distribution, representative_sources_from_plan

    assert callable(candidate_path_diagnostics)
    assert callable(stats_for_selected_pairs)
    assert callable(plot_delta_distribution)
    assert callable(representative_sources_from_plan)


def test_ch04_2_notebook_no_longer_defines_extracted_helpers_locally():
    nb = json.loads(Path("notebooks/04_2_state_space_representation_assumptions.ipynb").read_text())
    code_sources = [
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert max(source.count("\n") + bool(source) for source in code_sources) <= 100

    extracted_names = {
        "exp8_candidate_diagnostics",
        "exp8_select_pairs",
        "exp8_stats_for_selected",
        "exp8_paired_differences",
        "exp8_plot_supplement",
        "exp8_plot_paths_2d",
        "exp8_eb_plot_statistics",
        "representative_sources_from_plan",
        "highest_mass_targets",
        "plot_representative_endpoint_pairs",
        "load_eb_off_manifold_differences",
        "plot_delta_distribution",
        "choose_eb_path_example_ids",
        "reconstruct_eb_graph_paths",
    }
    local_defs = set()
    for source in code_sources:
        tree = ast.parse(source)
        local_defs.update(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )

    assert extracted_names.isdisjoint(local_defs)
    joined = "\n".join(code_sources)
    assert "from src.ch04_manifold_diagnostics import" in joined
    assert "from src.ch04_state_space_figures import" in joined
