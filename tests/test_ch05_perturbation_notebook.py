from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_2_perturbation_response_sciplex.ipynb"


def _notebook_text() -> str:
    payload = json.loads(NOTEBOOK_PATH.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def _setup_cell() -> str:
    payload = json.loads(NOTEBOOK_PATH.read_text())
    return "".join(payload["cells"][2].get("source", []))


def _notebook_code_lengths() -> list[int]:
    payload = json.loads(NOTEBOOK_PATH.read_text())
    lengths = []
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            lengths.append(source.count("\n") + bool(source))
    return lengths


def test_ch05_2_notebook_is_split_b_c_perturbation_only():
    assert NOTEBOOK_PATH.exists()
    text = _notebook_text()

    assert "Section 5.2 Perturbation response prediction with sci-Plex" in text
    assert "Split B held-out highest dose" in text
    assert "Split C held-out compound" in text
    assert "pd.concat([split_b_metrics, split_c_metrics], ignore_index=True)" in text

    assert "Split A random sanity" not in text
    assert "split_a_metrics" not in text
    assert "load_eb_ch05" not in text
    assert "run_eb_" not in text
    assert "EB_PATH" not in text
    assert "tab_5_1_" not in text


def test_ch05_2_notebook_declares_only_perturbation_artifacts():
    text = _notebook_text()

    for required in [
        "tab_5_2_sciplex_splits.csv",
        "sciplex_metrics_by_group.csv",
        "sciplex_metrics_summary.csv",
        "fig_5_2_heldout_highest_dose_metrics",
        "fig_5_2_heldout_compound_metrics",
        "fig_5_2_alisertib_example",
        "run_summary_perturbation_sciplex.json",
    ]:
        assert required in text

    for removed in [
        "fig_5_1_eb_pairwise_vs_shared.png",
        "fig_5_1_skip_pair_ablation.png",
        "fig_5_1_main_suite.png",
        "tab_5_1_main_suite.csv",
    ]:
        assert removed not in text


def test_ch05_2_defaults_to_full_section52_reproduction_config():
    setup = _setup_cell()
    helper_source = (PROJECT_ROOT / "src" / "perturbation_reporting.py").read_text()

    assert "CONFIG = ch05s.make_section52_config(PROJECT_ROOT)" in setup
    assert "DEFAULT_SEED = CONFIG.default_seed" in setup
    assert "TRAINING_STEPS = CONFIG.training_steps" in setup
    assert "SCIPLEX_DOWNLOAD_IN_CH05 = CONFIG.sciplex_download_in_ch05" in setup
    assert "SCIPLEX_SYNTHETIC_IF_MISSING = CONFIG.sciplex_synthetic_if_missing" in setup
    assert "MAX_EVAL_GROUPS = CONFIG.max_eval_groups" in setup

    assert 'default_seed=int(os.environ.get("CH05_SEED", "42"))' in helper_source
    assert 'quick_mode=os.environ.get("CH05_QUICK", "0") == "1"' in helper_source
    assert 'training_steps=int(os.environ.get("CH05_TRAINING_STEPS", "6000"))' in helper_source
    assert 'batch_size=int(os.environ.get("CH05_BATCH_SIZE", "256"))' in helper_source
    assert 'nfe=int(os.environ.get("CH05_NFE", "32"))' in helper_source
    assert 'sciplex_download_in_ch05=os.environ.get("CH05_SCIPLEX_DOWNLOAD_IN_CH05", "0") == "1"' in helper_source
    assert 'sciplex_synthetic_if_missing=os.environ.get("CH05_ALLOW_SYNTHETIC_SCIPLEX", "0") == "1"' in helper_source
    assert 'max_eval_groups=None if max_eval_groups == "" else int(max_eval_groups)' in helper_source

    assert '"1500" if QUICK_MODE else "6000"' not in setup
    assert '"128" if QUICK_MODE else "256"' not in setup
    assert '"16" if QUICK_MODE else "32"' not in setup


def test_ch05_2_setup_controls_random_seeds_and_cuda_determinism():
    setup = _setup_cell()

    for required in [
        "import random",
        "random.seed(DEFAULT_SEED)",
        "np.random.seed(DEFAULT_SEED)",
        "torch.manual_seed(DEFAULT_SEED)",
        "torch.cuda.manual_seed_all(DEFAULT_SEED)",
        "torch.backends.cudnn.benchmark = False",
        "torch.backends.cudnn.deterministic = True",
        "torch.use_deterministic_algorithms(True, warn_only=True)",
    ]:
        assert required in setup


def test_ch05_2_notebook_uses_src_helpers_for_display_support():
    text = _notebook_text()
    code_lengths = _notebook_code_lengths()

    for forbidden_inline_helper in [
        "def save_figure_pair(",
        "def draw_tiny_cloud(",
        "def draw_velocity_box(",
        "def draw_method_tile(",
        "def split_status_matrix(",
        "def draw_split_grid(",
        "def plot_metric_panel(",
    ]:
        assert forbidden_inline_helper not in text

    for required_src_call in [
        "ch05s.make_section52_config(",
        "ch05s.build_model_design_figure(",
        "ch05s.build_evaluation_split_figure(",
        "ch05s.plot_metric_panel(",
        "ch05s.build_section52_run_summary(",
        "ch05s.audit_section52_artifacts(",
    ]:
        assert required_src_call in text

    assert len(code_lengths) >= 18
    assert max(code_lengths) <= 90
