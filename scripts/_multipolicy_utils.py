"""Pure-Python utilities for the multi-policy collector.

Kept holosoma-free so the unit test for it can run on the host without
the IsaacSim docker image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------


def select_ckpts(
    ckpt_dir: Path, num_train: int, num_test: int, split: str
) -> list[Path]:
    """Pick ``num_train + num_test`` ckpts uniformly across the run, then
    deterministically partition into a train and test subset of the
    requested sizes. Same arguments => same partition (no RNG).
    """
    all_ckpts = sorted(ckpt_dir.glob("model_*.pt"))
    M = len(all_ckpts)
    total = num_train + num_test
    if total == 0:
        raise ValueError("num_train + num_test must be > 0")
    if M < total:
        raise ValueError(
            f"ckpt_dir has {M} ckpts but {total} requested "
            f"(num_train={num_train}, num_test={num_test})"
        )
    picked = np.linspace(0, M - 1, total, dtype=int)
    if num_test == 0:
        train_idx = picked
        test_idx = np.empty(0, dtype=int)
    elif num_train == 0:
        train_idx = np.empty(0, dtype=int)
        test_idx = picked
    else:
        stride = total // num_test
        test_mask = np.zeros(total, dtype=bool)
        test_mask[::stride] = True
        excess = int(test_mask.sum()) - num_test
        if excess > 0:
            true_positions = np.where(test_mask)[0]
            test_mask[true_positions[-excess:]] = False
        test_idx = picked[test_mask]
        train_idx = picked[~test_mask]
        assert len(train_idx) == num_train, (
            f"train pick count {len(train_idx)} != requested {num_train}"
        )
        assert len(test_idx) == num_test, (
            f"test pick count {len(test_idx)} != requested {num_test}"
        )
        assert len(np.intersect1d(train_idx, test_idx)) == 0, (
            "train and test ckpt picks overlap -- internal bug"
        )

    chosen = train_idx if split == "train" else test_idx
    return [all_ckpts[i] for i in chosen]


# ---------------------------------------------------------------------------
# Per-field memmap saver
# ---------------------------------------------------------------------------


class MemmapSaver:
    """Per-field ``(num_steps, num_envs, *feat) float32`` memmap writer.

    Allocates one ``.npy`` per field on first :meth:`append_chunk` call
    (uses the probe step to discover ``feat`` shapes), then accepts
    chunked writes. Each chunk is a list of dicts (one per env step);
    fields are stacked along a new leading dim and copied into the
    appropriate memmap slice. Calls :meth:`numpy.memmap.flush` after
    every chunk so a mid-run crash leaves the prefix on disk.
    """

    def __init__(self, out_dir: Path, num_steps: int, num_envs: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self._memmaps: dict[str, np.memmap] = {}
        self._field_shapes: dict[str, tuple[int, ...]] = {}
        self._cursor = 0

    def _allocate(self, probe_step: dict[str, np.ndarray]) -> None:
        for k, arr in probe_step.items():
            if arr.shape[0] != self.num_envs:
                raise ValueError(
                    f"field {k!r} has leading dim {arr.shape[0]} != "
                    f"num_envs {self.num_envs}"
                )
            feat = tuple(arr.shape[1:])
            self._field_shapes[k] = feat
            full_shape = (self.num_steps, self.num_envs, *feat)
            path = self.out_dir / f"{k}.npy"
            self._memmaps[k] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=full_shape
            )

    def append_chunk(self, chunk: list[dict[str, np.ndarray]]) -> None:
        if not chunk:
            return
        if not self._memmaps:
            self._allocate(chunk[0])
        T = len(chunk)
        start = self._cursor
        end = start + T
        if end > self.num_steps:
            raise RuntimeError(
                f"saver overflow: chunk would extend to step {end} "
                f"but num_steps={self.num_steps}"
            )
        for k, mm in self._memmaps.items():
            stacked = np.stack(
                [step[k] for step in chunk], axis=0
            ).astype(np.float32, copy=False)
            mm[start:end] = stacked
            mm.flush()
        self._cursor = end

    def close(self) -> None:
        for mm in self._memmaps.values():
            mm.flush()
        self._memmaps.clear()

    @property
    def shape_report(self) -> dict[str, list[int]]:
        return {
            k: [self.num_steps, self.num_envs, *shape]
            for k, shape in self._field_shapes.items()
        }
