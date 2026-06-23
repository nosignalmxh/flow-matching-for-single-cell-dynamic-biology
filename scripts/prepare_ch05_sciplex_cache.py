#!/usr/bin/env python
"""Prepare sci-Plex A549 and LINCS SMILES caches for Chapter 5.2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loading import load_lincs_smiles_corpus, load_or_prepare_sciplex3_a549


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Download real sci-Plex/LINCS assets if local cache files are missing. Can also be enabled with CH05_SCIPLEX_DOWNLOAD=1.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=False,
        help="Allow synthetic fallback only for smoke testing. Can also be enabled with CH05_ALLOW_SYNTHETIC_SCIPLEX=1.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild prepared cache files even if they already exist.")
    parser.add_argument("--hvg-top-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "sciplex3_a549")
    parser.add_argument("--lincs-smiles-dir", type=Path, default=PROJECT_ROOT / "data" / "chemcpa_lincs_smiles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.download = bool(args.download or env_flag("CH05_SCIPLEX_DOWNLOAD", default=False))
    args.allow_synthetic = bool(
        args.allow_synthetic or env_flag("CH05_ALLOW_SYNTHETIC_SCIPLEX", default=False)
    )

    sciplex_cache = load_or_prepare_sciplex3_a549(
        data_dir=args.data_dir,
        lincs_smiles_dir=args.lincs_smiles_dir,
        download=args.download,
        hvg_top_n=args.hvg_top_n,
        overwrite=args.overwrite,
        synthetic_if_missing=args.allow_synthetic,
        seed=args.seed,
    )
    lincs_smiles_cache = load_lincs_smiles_corpus(
        cache_dir=args.lincs_smiles_dir,
        download=args.download,
    )

    report = {
        "download": args.download,
        "allow_synthetic": args.allow_synthetic,
        "sciplex_paths": sciplex_cache.paths,
        "sciplex_summary": sciplex_cache.summary,
        "lincs_smiles_path": str(lincs_smiles_cache.path),
        "n_lincs_smiles": int(len(lincs_smiles_cache.smiles)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
