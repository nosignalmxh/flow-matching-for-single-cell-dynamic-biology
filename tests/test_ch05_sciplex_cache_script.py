from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_ch05_sciplex_cache.py"
CH04_SPLIT_NOTEBOOK_PATHS = [
    PROJECT_ROOT / "notebooks" / "chapter4_1_coupling_geometry.ipynb",
    PROJECT_ROOT / "notebooks" / "chapter4_2_state_space_assumptions.ipynb",
    PROJECT_ROOT / "notebooks" / "chapter4_3_sampling_depth.ipynb",
]


def test_ch05_sciplex_cache_has_standalone_script_with_safe_defaults():
    assert SCRIPT_PATH.exists()
    source = SCRIPT_PATH.read_text()

    assert "load_or_prepare_sciplex3_a549" in source
    assert "load_lincs_smiles_corpus" in source
    assert "CH05_SCIPLEX_DOWNLOAD" in source
    assert "CH05_ALLOW_SYNTHETIC_SCIPLEX" in source
    assert "default=False" in source
    assert "synthetic_if_missing=args.allow_synthetic" in source
    assert "download=args.download" in source


def test_chapter4_notebook_no_longer_prepares_chapter5_sciplex_cache():
    texts = []
    for path in CH04_SPLIT_NOTEBOOK_PATHS:
        assert path.exists(), path
        payload = json.loads(path.read_text())
        texts.append("\n".join("".join(cell.get("source", [])) for cell in payload["cells"]))
    text = "\n".join(texts)

    assert "sci-Plex 3 A549 data cache for Chapter 5" not in text
    assert "load_or_prepare_sciplex3_a549" not in text
    assert "load_lincs_smiles_corpus" not in text
