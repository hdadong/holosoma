"""Self-check for :mod:`_batched_actor_ensemble`.

Loads 5 real WBT box-carry checkpoints, builds a ``BatchedActorEnsemble``
from them, and compares its output element-wise to the original
``Actor`` (run separately per checkpoint). Must hit abs diff < 1e-5 on
the deterministic action; stochastic mode is tested by sharing a noise
tensor between the two implementations.

Usage (host, fasttd3 conda env)::

    /home/weidong/miniconda3/envs/fasttd3/bin/python \
        /home/weidong/holosoma/scripts/_test_batched_actor_ensemble.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from _batched_actor_ensemble import build_ensemble_from_ckpts  # noqa: E402


# ---------------------------------------------------------------------------
# Import the reference Actor class directly from the holosoma source file.
# This avoids depending on the rest of the `holosoma` package (which pulls
# in IsaacLab / tensordict / etc.) and lets the self-check run on the
# host with just torch installed.
# ---------------------------------------------------------------------------
def _import_reference_actor():
    actor_src = Path(
        "/home/weidong/holosoma/src/holosoma/holosoma/agents/fast_sac/fast_sac.py"
    )
    spec = importlib.util.spec_from_file_location("_holosoma_fast_sac", actor_src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.Actor


def _make_reference_actor(ActorCls, *, obs_dim, n_act, hidden_dim, device):
    """Instantiate a fresh Actor matching the WBT box-carry hyperparameters.

    ``obs_indices`` is set so ``process_obs`` is the identity (single
    ``actor_obs`` key spanning the full obs vector).
    """
    return ActorCls(
        obs_indices={"actor_obs": {"start": 0, "end": obs_dim, "size": obs_dim}},
        obs_keys=["actor_obs"],
        n_act=n_act,
        num_envs=1,
        hidden_dim=hidden_dim,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        device=device,
    )


def _normalize_obs(obs, mean_1xd, std_1xd, eps):
    """Mirror EmpiricalNormalization.forward(center=True, update=False)."""
    return (obs - mean_1xd) / (std_1xd + eps)


def main():
    ckpt_dir = Path(
        "/home/weidong/holosoma_wbt_boxcarry_logs/WholeBodyTracking/"
        "20260518_143923-g1_29dof_wbt_fast_sac_manager-locomotion"
    )
    all_ckpts = sorted(ckpt_dir.glob("model_*.pt"))
    if len(all_ckpts) < 5:
        raise RuntimeError(f"Need >= 5 ckpts under {ckpt_dir}, found {len(all_ckpts)}")
    # Spread the 5 ckpts across the training run to exercise diverse
    # normalizer stats / action distributions.
    import numpy as np
    pick = np.linspace(0, len(all_ckpts) - 1, 5, dtype=int)
    ckpt_paths = [all_ckpts[i] for i in pick]
    print("Selected ckpts:")
    for i, p in enumerate(ckpt_paths):
        print(f"  [{i}] {p.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OBS_DIM, N_ACT, HIDDEN = 154, 29, 512
    LSMIN, LSMAX, OBS_EPS = -5.0, 0.0, 1e-2

    # Build the ensemble (N=5).
    ensemble = build_ensemble_from_ckpts(
        ckpt_paths,
        obs_dim=OBS_DIM,
        n_act=N_ACT,
        log_std_min=LSMIN,
        log_std_max=LSMAX,
        use_layer_norm=True,
        use_tanh=True,
        obs_eps=OBS_EPS,
        device=device,
        dtype=torch.float32,
    )
    print(f"Ensemble built: N={ensemble.N}, obs_dim={ensemble.obs_dim}, "
          f"n_act={ensemble.n_act}")

    # Random per-policy raw obs (one obs vector per env=policy).
    rng = torch.Generator(device=device).manual_seed(123)
    raw_obs = torch.randn(ensemble.N, OBS_DIM, device=device, generator=rng)

    # ---- Ensemble outputs ----
    mean_ens, log_std_ens = ensemble._trunk(raw_obs)
    action_det_ens = ensemble.explore(raw_obs, deterministic=True)
    shared_noise = torch.randn(
        (ensemble.N, N_ACT), device=device, generator=rng
    )
    action_sto_ens = ensemble.explore(
        raw_obs, deterministic=False, noise=shared_noise
    )

    # ---- Reference: instantiate Actor per ckpt, load state_dict, run ----
    ActorCls = _import_reference_actor()

    mean_ref = torch.zeros_like(mean_ens)
    log_std_ref = torch.zeros_like(log_std_ens)
    action_det_ref = torch.zeros_like(action_det_ens)
    action_sto_ref = torch.zeros_like(action_sto_ens)

    for i, path in enumerate(ckpt_paths):
        ck = torch.load(str(path), map_location=device, weights_only=False)
        actor = _make_reference_actor(
            ActorCls,
            obs_dim=OBS_DIM,
            n_act=N_ACT,
            hidden_dim=HIDDEN,
            device=device,
        )
        actor.load_state_dict(ck["actor_state_dict"])
        actor.eval()

        nsd = ck["obs_normalizer_state"]
        norm_mean = nsd["_mean"].to(device)  # (1, obs_dim)
        norm_std = nsd["_std"].to(device)    # (1, obs_dim)

        # Single-row obs for this policy.
        obs_i = raw_obs[i : i + 1]  # (1, obs_dim)
        normed_i = _normalize_obs(obs_i, norm_mean, norm_std, OBS_EPS)

        with torch.no_grad():
            _, mean_i, log_std_i = actor(normed_i)
            tanh_mean = torch.tanh(mean_i)
            action_det_i = tanh_mean * actor.action_scale + actor.action_bias
            # Stochastic: re-use the SAME noise that the ensemble used.
            std_i = log_std_i.exp()
            raw_sample = mean_i + std_i * shared_noise[i : i + 1]
            action_sto_i = (
                torch.tanh(raw_sample) * actor.action_scale + actor.action_bias
            )

        mean_ref[i] = mean_i[0]
        log_std_ref[i] = log_std_i[0]
        action_det_ref[i] = action_det_i[0]
        action_sto_ref[i] = action_sto_i[0]

    # ---- Compare ----
    def report(name, a, b, tol):
        diff = (a - b).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        verdict = "OK" if max_d < tol else "FAIL"
        print(f"  [{verdict}] {name:25s} max={max_d:.3e}  mean={mean_d:.3e}  "
              f"(tol {tol:.0e})")
        return max_d < tol

    print("\nDiff vs reference Actor:")
    TOL = 1e-5
    ok1 = report("mean", mean_ens, mean_ref, TOL)
    ok2 = report("log_std", log_std_ens, log_std_ref, TOL)
    ok3 = report("action (deterministic)", action_det_ens, action_det_ref, TOL)
    ok4 = report("action (stochastic)", action_sto_ens, action_sto_ref, TOL)

    if not (ok1 and ok2 and ok3 and ok4):
        print("\nFAIL: one or more outputs exceed tolerance.")
        sys.exit(1)
    print("\nPASS: all outputs match within 1e-5.")


if __name__ == "__main__":
    main()
