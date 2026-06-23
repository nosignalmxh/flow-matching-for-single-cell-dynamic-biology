from __future__ import annotations

import importlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _notebook_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_ch05_experiment_helpers_are_split_by_notebook_domain():
    timecourse = importlib.import_module("src.experiments.timecourse")
    perturbation = importlib.import_module("src.experiments.perturbation")

    assert not (PROJECT_ROOT / "src" / "single_cell_experiments.py").exists()
    assert hasattr(timecourse, "load_eb_ch05")
    assert hasattr(timecourse, "run_eb_section51_main_suite")
    assert not hasattr(timecourse, "evaluate_sciplex_split")
    assert hasattr(perturbation, "evaluate_sciplex_split")
    assert hasattr(perturbation, "SciplexConditionEncoder")
    assert not hasattr(perturbation, "load_eb_ch05")


def test_ch05_notebooks_and_project_markers_use_split_experiment_modules():
    timecourse_text = _notebook_text(PROJECT_ROOT / "notebooks" / "chapter5_1_timecourse_suite.ipynb")
    perturbation_text = _notebook_text(PROJECT_ROOT / "notebooks" / "chapter5_2_perturbation_sciplex.ipynb")

    assert "from src.experiments.timecourse import (" in timecourse_text
    assert "from src.experiments.perturbation import (" in perturbation_text
    assert "src.single_cell_experiments" not in timecourse_text
    assert "src.single_cell_experiments" not in perturbation_text

    assert 'markers=("src/experiments/timecourse.py",)' in (
        PROJECT_ROOT / "src" / "experiments" / "timecourse_config.py"
    ).read_text(encoding="utf-8")
    assert 'markers=("src/experiments/perturbation.py",)' in (
        PROJECT_ROOT / "src" / "visualization" / "perturbation.py"
    ).read_text(encoding="utf-8")


def test_ch05_command_line_scripts_use_split_timecourse_module():
    section51 = importlib.import_module("scripts.run_ch05_section51_main_suite")
    skip_ablation = importlib.import_module("scripts.run_ch05_eb_skip_pair_ablation")

    assert hasattr(section51, "run_eb_section51_main_suite")
    assert hasattr(skip_ablation, "run_eb_skip_pair_ablation")

    for script_name in ["run_ch05_section51_main_suite.py", "run_ch05_eb_skip_pair_ablation.py"]:
        source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "from src.experiments.timecourse import (" in source
        assert "src.ch05_experiments" not in source
