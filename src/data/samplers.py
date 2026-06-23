from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _sorted_unique(values):
    values = np.asarray(values)
    unique = np.unique(values)
    try:
        return np.asarray(sorted(unique.tolist()), dtype=unique.dtype)
    except TypeError:
        return unique


@dataclass
class SnapshotDataset:
    X: np.ndarray
    time: np.ndarray
    condition: np.ndarray | None = None
    labels: np.ndarray | None = None
    cell_id: np.ndarray | None = None
    split: np.ndarray | None = None

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float32)
        self.time = np.asarray(self.time)
        n = self.X.shape[0]
        if self.time.shape[0] != n:
            raise ValueError("time length must match X.shape[0]")
        for name in ["condition", "labels", "cell_id", "split"]:
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.shape[0] != n:
                raise ValueError(f"{name} length must match X.shape[0]")
            setattr(self, name, arr)
        if self.cell_id is None:
            self.cell_id = np.asarray([f"cell_{i:06d}" for i in range(n)], dtype=object)

    @property
    def timepoints(self):
        return _sorted_unique(self.time)

    @property
    def conditions(self):
        if self.condition is None:
            return [None]
        return _sorted_unique(self.condition).tolist()

    def _mask(self, timepoint=None, condition=None, split=None):
        mask = np.ones(self.X.shape[0], dtype=bool)
        if timepoint is not None:
            mask &= self.time == timepoint
        if condition is not None:
            if self.condition is None:
                mask &= False
            else:
                mask &= self.condition == condition
        if split is not None:
            if self.split is None:
                mask &= False
            else:
                mask &= self.split == split
        return mask

    def cells_at(self, timepoint, condition=None, split=None, return_indices: bool = False):
        mask = self._mask(timepoint=timepoint, condition=condition, split=split)
        idx = np.flatnonzero(mask)
        if return_indices:
            return self.X[idx], idx
        return self.X[idx]

    def snapshot_counts(self) -> pd.DataFrame:
        condition = self.condition if self.condition is not None else np.full(self.X.shape[0], None, dtype=object)
        frame = pd.DataFrame({"time": self.time, "condition": condition})
        counts = frame.groupby(["time", "condition"], dropna=False, observed=False).size().reset_index(name="n_cells")
        return counts.sort_values(["condition", "time"]).reset_index(drop=True)

    def label_counts(self) -> pd.DataFrame:
        condition = self.condition if self.condition is not None else np.full(self.X.shape[0], None, dtype=object)
        labels = self.labels if self.labels is not None else np.full(self.X.shape[0], "unlabeled", dtype=object)
        frame = pd.DataFrame({"time": self.time, "condition": condition, "label": labels})
        counts = (
            frame.groupby(["time", "condition", "label"], dropna=False, observed=False)
            .size()
            .reset_index(name="n_cells")
        )
        return counts.sort_values(["condition", "time", "label"]).reset_index(drop=True)


def make_cell_iid_split(dataset, test_size=0.2, groupby=("time", "condition"), seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = dataset.X.shape[0]
    split = np.full(n, "train", dtype=object)
    group_values = {}
    for key in groupby:
        if key == "time":
            group_values[key] = dataset.time
        elif key == "condition":
            group_values[key] = dataset.condition if dataset.condition is not None else np.full(n, None, dtype=object)
        elif key == "labels":
            group_values[key] = dataset.labels if dataset.labels is not None else np.full(n, None, dtype=object)
        else:
            raise ValueError(f"Unsupported groupby key: {key}")

    frame = pd.DataFrame(group_values)
    for _, idx in frame.groupby(list(group_values.keys()), dropna=False, observed=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        n_test = int(round(len(idx) * test_size))
        if len(idx) > 1 and n_test == 0:
            n_test = 1
        if n_test > 0:
            test_idx = rng.choice(idx, size=min(n_test, len(idx)), replace=False)
            split[test_idx] = "test"
    return split


def make_leave_one_timepoint_split(dataset, heldout_time) -> np.ndarray:
    split = np.full(dataset.X.shape[0], "train", dtype=object)
    split[dataset.time == heldout_time] = "heldout_time"
    return split


def make_heldout_condition_split(dataset, heldout_condition) -> np.ndarray:
    split = np.full(dataset.X.shape[0], "train", dtype=object)
    if dataset.condition is None:
        raise ValueError("heldout condition split requires dataset.condition")
    split[dataset.condition == heldout_condition] = "heldout_condition"
    return split


class PairSampler:
    def __init__(
        self,
        dataset: SnapshotDataset,
        source_time,
        target_time,
        condition=None,
        source_condition=None,
        target_condition=None,
        split=None,
        replace: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.source_time = source_time
        self.target_time = target_time
        self.source_condition = condition if source_condition is None else source_condition
        self.target_condition = condition if target_condition is None else target_condition
        self.condition = condition
        self.split = split
        self.replace = replace
        self.rng = np.random.default_rng(seed)

    def _pool_indices(self, timepoint, condition, role: str):
        _, idx = self.dataset.cells_at(
            timepoint,
            condition=condition,
            split=self.split,
            return_indices=True,
        )
        if len(idx) == 0:
            raise ValueError(
                "No cells available for "
                f"{role} pool with time={timepoint!r}, condition={condition!r}, split={self.split!r}."
            )
        return idx

    def sample(self, batch_size: int = 256):
        idx0_pool = self._pool_indices(self.source_time, self.source_condition, "source")
        idx1_pool = self._pool_indices(self.target_time, self.target_condition, "target")
        if not self.replace and (batch_size > len(idx0_pool) or batch_size > len(idx1_pool)):
            raise ValueError(
                "Cannot sample without replacement because batch_size exceeds at least one snapshot pool."
            )

        idx0 = self.rng.choice(idx0_pool, size=batch_size, replace=self.replace)
        idx1 = self.rng.choice(idx1_pool, size=batch_size, replace=self.replace)
        x0 = self.dataset.X[idx0]
        x1 = self.dataset.X[idx1]
        labels = None
        if self.dataset.labels is not None:
            labels = {"source": self.dataset.labels[idx0], "target": self.dataset.labels[idx1]}

        if self.dataset.condition is None:
            condition = np.full(batch_size, None, dtype=object)
        elif self.source_condition == self.target_condition:
            condition = self.dataset.condition[idx0]
        else:
            condition = {"source": self.dataset.condition[idx0], "target": self.dataset.condition[idx1]}

        batch = {
            "x0": x0,
            "x1": x1,
            "t0": np.full(batch_size, self.source_time, dtype=np.asarray(self.dataset.time).dtype),
            "t1": np.full(batch_size, self.target_time, dtype=np.asarray(self.dataset.time).dtype),
            "condition": condition,
            "labels": labels,
            "idx0": idx0,
            "idx1": idx1,
            "cell_id0": self.dataset.cell_id[idx0] if self.dataset.cell_id is not None else None,
            "cell_id1": self.dataset.cell_id[idx1] if self.dataset.cell_id is not None else None,
        }
        return batch


class CouplingPairSampler:
    """Sample endpoint pairs from a fixed empirical coupling matrix."""

    def __init__(self, X0, X1, pi, seed: int = 42, labels0=None, labels1=None):
        self.X0 = np.asarray(X0, dtype=np.float32)
        self.X1 = np.asarray(X1, dtype=np.float32)
        self.pi = np.asarray(pi, dtype=float)
        if self.X0.ndim != 2 or self.X1.ndim != 2:
            raise ValueError("X0 and X1 must be 2D arrays")
        if self.X0.shape[1] != self.X1.shape[1]:
            raise ValueError("X0 and X1 must have the same feature dimension")
        if self.pi.shape != (self.X0.shape[0], self.X1.shape[0]):
            raise ValueError(f"pi shape must be {(self.X0.shape[0], self.X1.shape[0])}, got {self.pi.shape}")
        if np.any(~np.isfinite(self.pi)):
            raise ValueError("pi must contain only finite entries")
        if np.any(self.pi < 0):
            raise ValueError("pi must have nonnegative mass")
        if float(self.pi.sum()) <= 0:
            raise ValueError("pi must have positive total mass")

        self.rng = np.random.default_rng(seed)
        self.labels0 = None if labels0 is None else np.asarray(labels0)
        self.labels1 = None if labels1 is None else np.asarray(labels1)
        if self.labels0 is not None and self.labels0.shape[0] != self.X0.shape[0]:
            raise ValueError("labels0 length must match X0.shape[0]")
        if self.labels1 is not None and self.labels1.shape[0] != self.X1.shape[0]:
            raise ValueError("labels1 length must match X1.shape[0]")

    def sample(self, batch_size: int = 256) -> dict:
        from ..core.ot import sample_pair_indices_from_coupling

        idx0, idx1 = sample_pair_indices_from_coupling(
            self.pi,
            batch_size=int(batch_size),
            seed=int(self.rng.integers(0, 2**32 - 1)),
        )
        labels = None
        if self.labels0 is not None or self.labels1 is not None:
            labels = {
                "source": self.labels0[idx0] if self.labels0 is not None else None,
                "target": self.labels1[idx1] if self.labels1 is not None else None,
            }
        return {
            "x0": self.X0[idx0],
            "x1": self.X1[idx1],
            "idx0": idx0,
            "idx1": idx1,
            "labels": labels,
        }


RandomPairSampler = PairSampler


class MultiTimePairSampler:
    def __init__(
        self,
        dataset: SnapshotDataset,
        pair_policy: str = "sequential",
        pairs: list[tuple[object, object]] | None = None,
        condition=None,
        split=None,
        replace: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.condition = condition
        self.split = split
        self.replace = replace
        self.rng = np.random.default_rng(seed)
        self.pairs = self._make_pairs(pair_policy=pair_policy, pairs=pairs)
        if len(self.pairs) == 0:
            raise ValueError("MultiTimePairSampler requires at least one time pair")

    def _make_pairs(self, pair_policy, pairs):
        if pairs is not None:
            return list(pairs)
        times = list(self.dataset.timepoints)
        if pair_policy == "sequential":
            return list(zip(times[:-1], times[1:]))
        if pair_policy == "triu":
            return [(times[i], times[j]) for i in range(len(times)) for j in range(i + 1, len(times))]
        if pair_policy == "explicit":
            raise ValueError("pair_policy='explicit' requires pairs=[(t0, t1), ...]")
        raise ValueError("pair_policy must be one of 'sequential', 'triu', or 'explicit'")

    def sample(self, batch_size: int = 256):
        t0, t1 = self.pairs[self.rng.integers(0, len(self.pairs))]
        sampler = PairSampler(
            self.dataset,
            t0,
            t1,
            condition=self.condition,
            split=self.split,
            replace=self.replace,
            seed=int(self.rng.integers(1_000_000_000)),
        )
        batch = sampler.sample(batch_size)
        batch["pair"] = (t0, t1)
        return batch
