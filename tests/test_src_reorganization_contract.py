from __future__ import annotations

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_functional_subpackages_expose_moved_modules():
    module_names = [
        "src.core.models",
        "src.core.losses",
        "src.core.sampling",
        "src.core.train",
        "src.core.paths",
        "src.core.ot",
        "src.data.loading",
        "src.data.samplers",
        "src.data.toy",
        "src.evaluation.metrics",
        "src.evaluation.representations",
        "src.evaluation.graph_paths",
        "src.visualization.flow_matching",
        "src.visualization.transport",
        "src.visualization.manifold",
        "src.visualization.manifold_diagnostics",
        "src.visualization.state_space",
        "src.visualization.sampling_depth",
        "src.visualization.timecourse",
        "src.visualization.perturbation",
        "src.experiments.flow_runtime",
        "src.experiments.manifold",
        "src.experiments.timecourse",
        "src.experiments.timecourse_config",
        "src.experiments.perturbation",
    ]

    for module_name in module_names:
        importlib.import_module(module_name)


def test_moved_public_helpers_import_from_functional_subpackages():
    from src.core.models import VelocityMLP
    from src.core.ot import sinkhorn_plan
    from src.data.loading import fit_pc_to_phate_mapper
    from src.evaluation.metrics import mmd_rbf
    from src.experiments.manifold import train_or_load_model
    from src.visualization.transport import coupling_diagnostic_row

    assert callable(fit_pc_to_phate_mapper)
    assert callable(train_or_load_model)
    assert callable(mmd_rbf)
    assert callable(VelocityMLP)
    assert callable(sinkhorn_plan)
    assert callable(coupling_diagnostic_row)


def test_legacy_top_level_shims_are_removed():
    removed_shims = [
        "flow_matching_reporting.py",
        "flow_runtime.py",
        "graph_paths.py",
        "losses.py",
        "manifold_diagnostics.py",
        "manifold_reporting.py",
        "metrics.py",
        "models.py",
        "ot.py",
        "paths.py",
        "perturbation_experiments.py",
        "perturbation_reporting.py",
        "representations.py",
        "samplers.py",
        "sampling.py",
        "sampling_depth_reporting.py",
        "state_space_figures.py",
        "timecourse_experiments.py",
        "timecourse_figures.py",
        "timecourse_reporting.py",
        "toy.py",
        "train.py",
        "transport_reporting.py",
    ]

    for filename in removed_shims:
        assert not (PROJECT_ROOT / "src" / filename).exists(), filename
