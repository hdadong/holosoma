"""Host-runnable unit test for :mod:`_multipolicy_utils`.

Exercises ``select_ckpts`` (partitioning) and ``MemmapSaver`` (per-field
on-disk layout) without dragging in IsaacSim. Run::

    /home/weidong/miniconda3/envs/fasttd3/bin/python \
        /home/weidong/holosoma/scripts/_test_multipolicy_utils.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from _multipolicy_utils import MemmapSaver, select_ckpts  # noqa: E402


def _make_fake_ckpt_dir(num_ckpts: int) -> Path:
    """Create ``num_ckpts`` empty ``model_NNNNNNN.pt`` files for selection
    tests."""
    tmp = Path(tempfile.mkdtemp(prefix="fakeckpts_"))
    for i in range(num_ckpts):
        (tmp / f"model_{(i + 1) * 100:07d}.pt").touch()
    return tmp


def test_select_ckpts_disjoint_and_uniform():
    M = 2529
    d = _make_fake_ckpt_dir(M)
    train = select_ckpts(d, num_train=2000, num_test=200, split="train")
    test = select_ckpts(d, num_train=2000, num_test=200, split="test")
    assert len(train) == 2000, f"train: {len(train)}"
    assert len(test) == 200, f"test: {len(test)}"

    train_names = {p.name for p in train}
    test_names = {p.name for p in test}
    assert train_names.isdisjoint(test_names), \
        f"overlap: {train_names & test_names}"

    def _step(name: str) -> int:
        return int(name.split("_")[1].split(".")[0])

    train_steps = sorted(_step(p.name) for p in train)
    test_steps = sorted(_step(p.name) for p in test)
    # Both should span roughly the full range. With stride=11, picked
    # position 0 falls in the test split, so train_min lands at the
    # second picked position. Allow that asymmetry.
    full_lo, full_hi = 100, 100 * M
    spread = 100 * (M // 200) + 200   # ~one stride-of-picks of slack
    assert train_steps[0] - full_lo <= spread, \
        f"train min too high: {train_steps[0]}"
    assert full_hi - train_steps[-1] <= spread, \
        f"train max too low: {train_steps[-1]}"
    assert test_steps[0] - full_lo <= spread, \
        f"test min too high: {test_steps[0]}"
    assert full_hi - test_steps[-1] <= spread, \
        f"test max too low: {test_steps[-1]}"
    print(
        f"  [OK] select_ckpts(M={M}, 2000+200) disjoint, "
        f"train_steps=[{min(train_steps)},{max(train_steps)}], "
        f"test_steps=[{min(test_steps)},{max(test_steps)}]"
    )


def test_select_ckpts_smoke():
    d = _make_fake_ckpt_dir(2529)
    # smoke-test config from the project plan
    train10 = select_ckpts(d, num_train=10, num_test=0, split="train")
    assert len(train10) == 10
    # the test split should also work with num_train=0
    test_only = select_ckpts(d, num_train=0, num_test=5, split="test")
    assert len(test_only) == 5
    print(f"  [OK] smoke configs: train10={len(train10)}  test_only={len(test_only)}")


def test_select_ckpts_short_dir_raises():
    d = _make_fake_ckpt_dir(100)
    try:
        select_ckpts(d, num_train=200, num_test=10, split="train")
    except ValueError as e:
        print(f"  [OK] short ckpt_dir raises as expected: {e}")
        return
    raise AssertionError("expected ValueError for short ckpt_dir")


def test_memmap_saver_writes_correct_shapes_and_values():
    num_steps, num_envs = 20, 4
    chunk_size = 7  # intentionally not a divisor of num_steps
    out = Path(tempfile.mkdtemp(prefix="memmap_"))
    saver = MemmapSaver(out, num_steps=num_steps, num_envs=num_envs)

    # Build deterministic per-step data so we can verify what landed on disk.
    def step_data(step: int) -> dict[str, np.ndarray]:
        return {
            # (num_envs, 3)
            "robot_root_pos_w": np.full(
                (num_envs, 3), step, dtype=np.float32
            ) + np.arange(num_envs)[:, None] * 0.01,
            # (num_envs, 14, 4)  -- exercise the >1 feat-dim path
            "body_quat": np.full(
                (num_envs, 14, 4), step * 0.1, dtype=np.float32
            ),
            # (num_envs, 1) -- the path scalar fields take after the [:, None] hack
            "ref_step": np.full((num_envs, 1), step, dtype=np.float32),
        }

    chunk: list[dict[str, np.ndarray]] = []
    for t in range(num_steps):
        chunk.append(step_data(t))
        if len(chunk) == chunk_size or t == num_steps - 1:
            saver.append_chunk(chunk)
            chunk = []
    saver.close()

    # Read everything back and check.
    rp = np.load(out / "robot_root_pos_w.npy")
    bq = np.load(out / "body_quat.npy")
    rs = np.load(out / "ref_step.npy")
    assert rp.shape == (num_steps, num_envs, 3), f"rp shape: {rp.shape}"
    assert bq.shape == (num_steps, num_envs, 14, 4), f"bq shape: {bq.shape}"
    assert rs.shape == (num_steps, num_envs, 1), f"rs shape: {rs.shape}"

    for t in range(num_steps):
        expected = step_data(t)
        assert np.allclose(rp[t], expected["robot_root_pos_w"]), \
            f"rp mismatch at step {t}"
        assert np.allclose(bq[t], expected["body_quat"]), \
            f"bq mismatch at step {t}"
        assert np.allclose(rs[t], expected["ref_step"]), \
            f"rs mismatch at step {t}"

    report = saver.shape_report
    assert report["robot_root_pos_w"] == [num_steps, num_envs, 3]
    assert report["body_quat"] == [num_steps, num_envs, 14, 4]
    print(
        f"  [OK] MemmapSaver wrote {num_steps} steps x {num_envs} envs across "
        f"3 fields; verified all values + shapes"
    )


def test_memmap_saver_overflow_raises():
    out = Path(tempfile.mkdtemp(prefix="memmap_of_"))
    saver = MemmapSaver(out, num_steps=3, num_envs=2)
    saver.append_chunk([{"x": np.zeros((2, 1), dtype=np.float32)} for _ in range(3)])
    try:
        saver.append_chunk([{"x": np.zeros((2, 1), dtype=np.float32)}])
    except RuntimeError as e:
        print(f"  [OK] overflow raises: {e}")
        return
    finally:
        saver.close()
    raise AssertionError("expected RuntimeError on overflow")


if __name__ == "__main__":
    print("Running _multipolicy_utils self-checks ...")
    test_select_ckpts_disjoint_and_uniform()
    test_select_ckpts_smoke()
    test_select_ckpts_short_dir_raises()
    test_memmap_saver_writes_correct_shapes_and_values()
    test_memmap_saver_overflow_raises()
    print("PASS: all _multipolicy_utils tests")
