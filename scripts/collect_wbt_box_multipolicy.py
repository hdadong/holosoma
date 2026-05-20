"""Multi-POLICY data collector for the holosoma WBT box-carry policy.

Forked from :mod:`collect_wbt_box_multienv` (commit f0cc1b6). Where the
original runs ONE policy across N envs, this script runs N DIFFERENT
policies in parallel: one ckpt per env, all driven in a single batched
forward by :class:`BatchedActorEnsemble`. Output is no longer a single
``rollout.npz`` -- each captured field is written to its own
``(num_steps, num_envs, *feat).npy`` memmap, flushed every
``--chunk-steps`` env steps so a mid-collection crash leaves recoverable
data on disk.

CLI example (small smoke run -- expects ckpts under ``--ckpt-dir``)::

    python scripts/collect_wbt_box_multipolicy.py \\
        exp:g1-29dof-wbt-fast-sac-w-object-1kg-pre100-app100 \\
        --ckpt-dir=/workspace/holosoma/logs/WholeBodyTracking/<RUN_DIR> \\
        --num-train=10 --num-test=0 --split=train \\
        --num-steps=20 --chunk-steps=10 \\
        --out-dir=/path/to/out \\
        --collector.save-actor-obs=False \\
        --collector.save-critic-obs=False

Forces ``--training.num-envs = len(picked_ckpts)`` (a hard equality, not
a CLI override): each env is bound 1-to-1 to one checkpoint, so envs
and policies are the same count.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger
from pydantic.dataclasses import dataclass as pyd_dataclass

# Line-buffered stdout so progress reaches the docker log promptly.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    pass
print = functools.partial(print, flush=True)  # noqa: A001

# Holosoma deps (same as the single-policy collector).
from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)

# Reuse the per-step transition capture + quat helpers from the original.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from collect_wbt_box_multienv import (  # noqa: E402
    _capture_transition,
    _np,
    _parse_bool,
)
from _batched_actor_ensemble import (  # noqa: E402
    BatchedActorEnsemble,
    build_ensemble_from_ckpts,
)
from _multipolicy_utils import (  # noqa: E402
    MemmapSaver as _MemmapSaver,
    select_ckpts,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pyd_dataclass(frozen=True)
class MultiPolicyConfig:
    """Top-level CLI for the multi-policy collector.

    A separate ``--collector.save-actor-obs`` / ``--collector.save-critic-obs``
    pair lives below for byte-compat with the single-policy launcher.
    """

    ckpt_dir: str
    """Directory containing ``model_*.pt`` checkpoints (one per saved policy)."""

    num_train: int = 2000
    """Number of training-set policies to pick uniformly from the run."""

    num_test: int = 200
    """Number of test-set policies (non-overlapping with train)."""

    split: str = "train"
    """``train`` or ``test`` -- which subset to collect on this invocation."""

    seed: int = 0
    """Driving seed. Use the SAME seed for the train and test runs so the
    IsaacSim physics / DR / reset RNG sequences are identical (only the
    policies and env count differ)."""

    num_steps: int = 500
    """Env steps per policy. 2000 envs x 500 steps = 1M transitions."""

    chunk_steps: int = 50
    """Flush a memmap slice every K env steps. Caps host RAM and keeps
    partial data on disk if IsaacSim crashes mid-run."""

    out_dir: str = "rollout_multipolicy"
    """Directory to write per-field ``.npy`` memmaps into. A ``train/`` or
    ``test/`` subdir is appended based on ``--split``."""

    save_actor_obs: bool = False
    """Persist the (per-policy-dependent) actor_obs tensor. Defaults to
    False because each policy has its own ``obs_indices`` permutation and
    storing this would be confusing; raw fields are kept regardless."""

    save_critic_obs: bool = False
    """Persist the critic_obs tensor. Defaults to False (linear projection
    of raw fields anyway)."""


# ---------------------------------------------------------------------------
# Multi-policy collect loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_multipolicy_collect(
    algo,
    ensemble: BatchedActorEnsemble,
    *,
    num_steps: int,
    chunk_steps: int,
    stochastic: bool,
    save_actor_obs: bool,
    save_critic_obs: bool,
    out_dir: Path,
) -> _MemmapSaver:
    fastsac_env = algo.env
    base_env = algo.unwrapped_env
    sim = base_env.simulator
    motion_command = base_env.command_manager.get_state("motion_command")

    print("[collect-multi] resetting envs (training-style: random phase + noise) ...")
    actor_obs = fastsac_env.reset()
    num_envs = actor_obs.shape[0]
    print(
        f"[collect-multi] num_envs={num_envs} actor_obs_dim={actor_obs.shape[1]}  "
        f"(ensemble N={ensemble.N})"
    )
    if num_envs != ensemble.N:
        raise RuntimeError(
            f"env count {num_envs} != ensemble policy count {ensemble.N} -- "
            "the launcher should set --training.num-envs to the picked-ckpt "
            "count automatically."
        )

    dof_names = list(sim.dof_names)
    track_body_names = [
        sim._body_list[i] for i in motion_command.tracked_body_indexes.tolist()
    ]
    anchor_body_name = sim._body_list[int(motion_command.ref_body_index)]
    print(f"[collect-multi] dof_names={dof_names}")
    print(f"[collect-multi] track_body_names={track_body_names}")
    print(f"[collect-multi] anchor_body_name={anchor_body_name}")

    def _compute_critic_obs():
        d = base_env.obs_buf_dict
        return torch.cat([d[k] for k in fastsac_env._critic_obs_keys], dim=1)

    critic_obs = _compute_critic_obs()
    print(f"[collect-multi] critic_obs_dim={critic_obs.shape[1]}")

    saver = _MemmapSaver(out_dir, num_steps=num_steps, num_envs=num_envs)
    rollout_gen = torch.Generator(device=ensemble.obs_mean.device).manual_seed(0)

    chunk_buf: list[dict[str, np.ndarray]] = []

    for step in range(num_steps):
        # BatchedActorEnsemble.explore() takes RAW per-policy obs and applies
        # each policy's own normalizer internally, so we skip
        # algo.obs_normalizer entirely.
        action_raw = ensemble.explore(
            actor_obs,
            deterministic=not stochastic,
            generator=rollout_gen,
        )

        snap = _capture_transition(base_env, motion_command, actor_obs, critic_obs)
        snap["action"] = action_raw

        next_actor_obs, rew, reset_buf, extras = fastsac_env.step(action_raw)
        time_outs = extras["time_outs"].to(torch.bool)
        reset_bool = reset_buf.to(torch.bool)

        motion_end_threshold = motion_command.motion.time_step_total - 3
        motion_end_triggered = reset_bool & (
            snap["ref_step"] >= motion_end_threshold
        )
        truncated = time_outs | motion_end_triggered
        terminated = reset_bool & (~truncated)
        done = reset_bool

        snap["reward"] = rew
        snap["done"] = done.to(torch.float32)
        snap["terminated"] = terminated.to(torch.float32)
        snap["truncated"] = truncated.to(torch.float32)
        snap["motion_ended"] = motion_end_triggered.to(torch.float32)
        snap["episode_length_timeout"] = time_outs.to(torch.float32)

        # Filter optional fields and ensure (num_envs, *feat) layout.
        step_dict: dict[str, np.ndarray] = {}
        for k, v in snap.items():
            if (k == "actor_obs" and not save_actor_obs) or (
                k == "critic_obs" and not save_critic_obs
            ):
                continue
            arr = _np(v)
            if arr.ndim == 1:
                # (num_envs,) scalars -> (num_envs, 1) so memmap shape is uniform.
                arr = arr[:, None]
            step_dict[k] = arr
        chunk_buf.append(step_dict)

        actor_obs = next_actor_obs
        critic_obs = _compute_critic_obs()

        if (step + 1) % chunk_steps == 0 or (step + 1) == num_steps:
            saver.append_chunk(chunk_buf)
            chunk_buf.clear()
            print(
                f"[collect-multi] flushed up to step {step + 1:4d}/{num_steps}  "
                f"reward[mean]={float(rew.mean()):.4f}  "
                f"done={int(done.sum().item())}/{num_envs}  "
                f"term={int(terminated.sum().item())}/{num_envs}  "
                f"trunc={int(truncated.sum().item())}/{num_envs}"
            )
        elif step < 5:
            print(
                f"[collect-multi] step {step + 1:4d}/{num_steps}  "
                f"reward[mean]={float(rew.mean()):.4f}"
            )

    saver.close()

    metadata = {
        "num_envs": num_envs,
        "num_steps": num_steps,
        "chunk_steps": chunk_steps,
        "ctrl_dt": float(base_env.dt),
        "fps": float(1.0 / base_env.dt),
        "dof_names": dof_names,
        "track_body_names": track_body_names,
        "anchor_body_name": anchor_body_name,
        "motion_file": str(motion_command.motion_cfg.motion_file),
        "motion_time_step_total": int(motion_command.motion.time_step_total),
        "stochastic": bool(stochastic),
        "field_shapes": saver.shape_report,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[collect-multi] metadata: {out_dir / 'metadata.json'}")
    return saver


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_multipolicy_cfg_and_remaining() -> tuple[MultiPolicyConfig, list[str]]:
    """Pull MultiPolicyConfig flags off argv, leaving the rest for the
    holosoma tyro layer to consume."""
    cfg, remaining = tyro.cli(
        MultiPolicyConfig, return_unknown_args=True, add_help=False
    )
    return cfg, list(remaining)


def _parse_training_overrides(remaining: list[str]):
    """Mirror the override parser from the original collector."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--training.headless", dest="headless", type=_parse_bool, default=None)
    p.add_argument("--training.export-onnx", dest="export_onnx",
                   type=_parse_bool, default=None)
    p.add_argument("--training.seed", dest="seed", type=int, default=None)
    p.add_argument("--logger.video.enabled", dest="logger_video_enabled",
                   type=_parse_bool, default=None)
    args, leftover = p.parse_known_args(remaining)
    return args, leftover


def main() -> None:
    init_eval_logging()

    mp_cfg, remaining = _parse_multipolicy_cfg_and_remaining()
    override_args, leftover = _parse_training_overrides(remaining)
    if leftover:
        logger.warning(f"Ignoring unparsed CLI tokens: {leftover}")

    ckpt_dir = Path(mp_cfg.ckpt_dir).expanduser().resolve()
    ckpt_paths = select_ckpts(
        ckpt_dir, mp_cfg.num_train, mp_cfg.num_test, mp_cfg.split
    )
    if not ckpt_paths:
        raise SystemExit(
            f"select_ckpts returned 0 ckpts for split={mp_cfg.split!r}"
        )
    num_envs = len(ckpt_paths)
    print(f"[collect-multi] picked {num_envs} ckpts for split={mp_cfg.split}")
    print(f"[collect-multi]   first: {ckpt_paths[0].name}")
    print(f"[collect-multi]   last : {ckpt_paths[-1].name}")

    # Load the FIRST ckpt to drive algo.setup() / env build. We replace the
    # actor with the ensemble afterwards, so the weights themselves are
    # irrelevant beyond satisfying load_checkpoint's path-exists check.
    template_ckpt = str(ckpt_paths[0])
    checkpoint_cfg = CheckpointConfig(checkpoint=template_ckpt)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()

    # Force --training.num-envs to match the ckpt count.
    training_updates: dict = {
        "num_envs": num_envs,
        "max_eval_steps": int(mp_cfg.num_steps),
        "headless": True,
        "export_onnx": False,
    }
    if override_args.headless is not None:
        training_updates["headless"] = override_args.headless
    if override_args.export_onnx is not None:
        training_updates["export_onnx"] = override_args.export_onnx
    if override_args.seed is not None:
        training_updates["seed"] = override_args.seed
    else:
        training_updates["seed"] = int(mp_cfg.seed)

    tyro_config = dataclasses.replace(
        eval_cfg,
        training=dataclasses.replace(eval_cfg.training, **training_updates),
    )
    if override_args.logger_video_enabled is not None:
        tyro_config = dataclasses.replace(
            tyro_config,
            logger=dataclasses.replace(
                tyro_config.logger,
                video=dataclasses.replace(
                    tyro_config.logger.video,
                    enabled=override_args.logger_video_enabled,
                ),
            ),
        )

    out_dir = Path(mp_cfg.out_dir).expanduser().resolve() / mp_cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[collect-multi] num_envs={tyro_config.training.num_envs}  "
        f"num_steps={tyro_config.training.max_eval_steps}  "
        f"seed={tyro_config.training.seed}  "
        f"out_dir={out_dir}"
    )

    # ----- Setup IsaacSim, build env+algo, load template ckpt -----
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(
        tyro_config.logger, tyro_config.training, get_timestamp(),
        task_name="collect-multi",
    )
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving collector logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device, env=env, config=tyro_config.algo.config,
        log_dir=str(eval_log_dir), multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(checkpoint_path)

    # ----- Build the BatchedActorEnsemble (drop-in for algo.actor) -----
    ref_actor = algo.actor
    obs_dim = sum(
        ref_actor.obs_indices[k]["size"] for k in ref_actor.obs_keys
    )
    n_act = int(ref_actor.n_act)
    log_std_min = float(ref_actor.log_std_min)
    log_std_max = float(ref_actor.log_std_max)
    use_layer_norm = bool(ref_actor.use_layer_norm)
    use_tanh = bool(ref_actor.use_tanh)
    obs_eps = float(algo.obs_normalizer.eps) if algo.obs_normalization else 1e-2

    print(
        f"[collect-multi] building ensemble: obs_dim={obs_dim} n_act={n_act} "
        f"log_std=[{log_std_min},{log_std_max}] "
        f"use_layer_norm={use_layer_norm} use_tanh={use_tanh}"
    )
    ensemble = build_ensemble_from_ckpts(
        ckpt_paths,
        obs_dim=obs_dim,
        n_act=n_act,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        use_layer_norm=use_layer_norm,
        use_tanh=use_tanh,
        obs_eps=obs_eps,
        device=device,
        dtype=torch.float32,
    )

    # Free the original single-policy actor / normalizer memory before the
    # rollout starts (we no longer use them).
    del ref_actor
    algo.actor = ensemble
    torch.cuda.empty_cache()

    # Persist the ckpt manifest alongside the data, including the integer
    # step number (parsed from the filename) so downstream training can
    # filter / sort policies by training progress.
    def _parse_step(p: Path) -> int:
        name = p.stem  # model_0001234
        return int(name.split("_")[-1])

    ckpt_steps = np.asarray([_parse_step(p) for p in ckpt_paths], dtype=np.int64)
    np.save(out_dir / "ckpt_step.npy", ckpt_steps)
    with open(out_dir / "ckpt_paths.json", "w") as f:
        json.dump([str(p) for p in ckpt_paths], f, indent=2)
    print(
        f"[collect-multi] ckpt manifest: ckpt_step.npy + ckpt_paths.json "
        f"({len(ckpt_paths)} policies)"
    )

    # ----- Run collection -----
    try:
        run_multipolicy_collect(
            algo,
            ensemble,
            num_steps=int(mp_cfg.num_steps),
            chunk_steps=int(mp_cfg.chunk_steps),
            stochastic=True,  # forced True: multi-policy diversity wants stochastic.
            save_actor_obs=bool(mp_cfg.save_actor_obs),
            save_critic_obs=bool(mp_cfg.save_critic_obs),
            out_dir=out_dir,
        )
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
