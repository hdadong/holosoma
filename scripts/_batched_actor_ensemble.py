"""N holosoma FastSAC actors stacked along a leading dim for batched
forward.

Drop-in replacement for the ``(Actor, EmpiricalNormalization)`` pair
used in ``collect_wbt_box_multienv.py`` when each env runs a DIFFERENT
policy checkpoint (rather than the same policy across all envs).

The forward signature mirrors ``Actor.explore`` but pairs each row of the
obs batch with its own policy:

    raw_obs.shape == (N, obs_dim)        # one obs per env, one env per policy
    action = ensemble.explore(raw_obs, deterministic=False)
                                          # action.shape == (N, n_act)
                                          # action[i] = policy_i(raw_obs[i])

All per-policy parameters (Linear weights/biases, LayerNorm gain/bias,
log-std clamp range, tanh action scale/bias, observation normalizer
mean/std) are stacked into a single leading dim ``N``. Forward is
``torch.einsum('ni,nio->no', ...)`` — one batched matmul per layer, no
per-policy Python loop.

The architecture assumed here matches every FastSAC Actor saved by
holosoma WBT training: 3 (Linear + LayerNorm + SiLU) blocks with
hidden sizes ``H -> H/2 -> H/4``, then a single Linear head for the
mean and another for the log-std. ``obs_keys=['actor_obs']`` is
assumed, i.e. ``process_obs`` is the identity. ``use_tanh=True`` is
assumed (matches the WBT box-carry training run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State-dict key layout for a holosoma FastSAC Actor
# ---------------------------------------------------------------------------
# Linear layers inside ``Actor.net`` are at module indices 0, 3, 6.
# LayerNorm modules are at indices 1, 4, 7 (between each Linear and the
# following SiLU). The mean head is a Sequential wrapping one Linear,
# hence the ``.0`` suffix; ``fc_logstd`` is a bare Linear.
_LINEAR_LAYER_PREFIXES = ("net.0", "net.3", "net.6")
_LAYERNORM_PREFIXES = ("net.1", "net.4", "net.7")
_FC_MU_PREFIX = "fc_mu.0"
_FC_LOGSTD_PREFIX = "fc_logstd"


class BatchedActorEnsemble(nn.Module):
    """Stacks N FastSAC Actors and their EmpiricalNormalization buffers."""

    def __init__(
        self,
        *,
        net_W: list[torch.Tensor],
        net_b: list[torch.Tensor],
        net_ln_g: list[torch.Tensor | None],
        net_ln_b: list[torch.Tensor | None],
        fc_mu_W: torch.Tensor,
        fc_mu_b: torch.Tensor,
        fc_logstd_W: torch.Tensor,
        fc_logstd_b: torch.Tensor,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        obs_eps: float,
        log_std_min: float,
        log_std_max: float,
        use_layer_norm: bool,
        use_tanh: bool,
    ):
        super().__init__()
        for i, (W, b) in enumerate(zip(net_W, net_b)):
            self.register_buffer(f"net{i}_W", W)
            self.register_buffer(f"net{i}_b", b)
        if use_layer_norm:
            for i, (g, b) in enumerate(zip(net_ln_g, net_ln_b)):
                self.register_buffer(f"net{i}_ln_g", g)
                self.register_buffer(f"net{i}_ln_b", b)
        self.register_buffer("fc_mu_W", fc_mu_W)
        self.register_buffer("fc_mu_b", fc_mu_b)
        self.register_buffer("fc_logstd_W", fc_logstd_W)
        self.register_buffer("fc_logstd_b", fc_logstd_b)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        self.obs_eps = obs_eps
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_layer_norm = use_layer_norm
        self.use_tanh = use_tanh
        self.num_layers = len(net_W)
        self.N = fc_mu_b.shape[0]
        self.obs_dim = obs_mean.shape[-1]
        self.n_act = fc_mu_b.shape[-1]

    @staticmethod
    def _bmm(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # x:(N, in)  W:(N, in, out)  b:(N, out)  ->  (N, out)
        return torch.einsum("ni,nio->no", x, W) + b

    @staticmethod
    def _bln(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # Replicates torch.nn.LayerNorm(elementwise_affine=True) along
        # the last (feature) dim, broadcast independently per policy.
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return g * (x - mean) * torch.rsqrt(var + eps) + b

    @torch.no_grad()
    def _trunk(self, raw_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) — the deterministic part of ``forward``."""
        assert raw_obs.shape == (self.N, self.obs_dim), (
            f"BatchedActorEnsemble expected (N={self.N}, obs_dim={self.obs_dim}); "
            f"got {tuple(raw_obs.shape)}"
        )
        x = (raw_obs - self.obs_mean) / (self.obs_std + self.obs_eps)
        for i in range(self.num_layers):
            x = self._bmm(x, getattr(self, f"net{i}_W"), getattr(self, f"net{i}_b"))
            if self.use_layer_norm:
                x = self._bln(x, getattr(self, f"net{i}_ln_g"), getattr(self, f"net{i}_ln_b"))
            x = F.silu(x)
        mean = self._bmm(x, self.fc_mu_W, self.fc_mu_b)
        raw_lstd = self._bmm(x, self.fc_logstd_W, self.fc_logstd_b)
        # Same tanh-clamped log_std as Actor.forward (SpinUp/Yarats trick).
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(raw_lstd) + 1.0
        )
        return mean, log_std

    @torch.no_grad()
    def explore(
        self,
        raw_obs: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample (or argmax) an action per policy.

        Args:
            raw_obs: ``(N, obs_dim)`` — pre-normalisation, one obs per env.
            deterministic: if True, return ``tanh(mean) * scale + bias``;
                otherwise sample a tanh-Gaussian action.
            generator: optional ``torch.Generator`` for the Gaussian noise.
            noise: optional pre-sampled noise ``(N, n_act)`` — when given,
                overrides ``generator``. Used by the self-check so the
                ensemble and the reference Actor can share a noise tensor.
        Returns:
            action: ``(N, n_act)``.
        """
        mean, log_std = self._trunk(raw_obs)
        if deterministic:
            raw_action = mean
        else:
            std = log_std.exp()
            if noise is None:
                noise = torch.randn(
                    mean.shape,
                    generator=generator,
                    device=mean.device,
                    dtype=mean.dtype,
                )
            raw_action = mean + std * noise
        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action

    # Surface a few attributes that match the original Actor's interface,
    # for callers that introspect (e.g. logging code that asks for n_act).
    @property
    def num_policies(self) -> int:
        return self.N


# ---------------------------------------------------------------------------
# Loader: stack N holosoma checkpoints into one BatchedActorEnsemble
# ---------------------------------------------------------------------------
def build_ensemble_from_ckpts(
    ckpt_paths: Sequence[str | Path],
    *,
    obs_dim: int,
    n_act: int,
    log_std_min: float,
    log_std_max: float,
    use_layer_norm: bool = True,
    use_tanh: bool = True,
    obs_eps: float = 1e-2,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> BatchedActorEnsemble:
    """Load ``N`` FastSAC checkpoints from disk and stack into an ensemble.

    Each checkpoint is expected to be the dict written by holosoma WBT
    training (top-level keys ``actor_state_dict`` and
    ``obs_normalizer_state``). The first checkpoint is used to discover
    per-Linear ``(in, out)`` shapes; every other checkpoint is asserted
    to match.
    """
    device = torch.device(device)
    if len(ckpt_paths) == 0:
        raise ValueError("ckpt_paths must contain at least one checkpoint.")

    def _load_one(path):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return ck["actor_state_dict"], ck["obs_normalizer_state"]

    asd0, _ = _load_one(ckpt_paths[0])

    # Pull layer shapes from the first ckpt.
    linear_shapes = []  # list of (in, out)
    for prefix in _LINEAR_LAYER_PREFIXES:
        W = asd0[f"{prefix}.weight"]  # shape (out, in) — PyTorch convention
        out_dim, in_dim = W.shape
        linear_shapes.append((in_dim, out_dim))
    # Sanity check: net.0 input dim must equal obs_dim.
    if linear_shapes[0][0] != obs_dim:
        raise ValueError(
            f"obs_dim mismatch: ckpt expects {linear_shapes[0][0]}, "
            f"got {obs_dim}."
        )

    # Accumulator lists keyed by state_dict key (so order matches ckpt list).
    bufs: dict[str, list[torch.Tensor]] = {}

    def push(key: str, tensor: torch.Tensor):
        bufs.setdefault(key, []).append(tensor.to(dtype))

    for path in ckpt_paths:
        asd, nsd = _load_one(path)
        for prefix in _LINEAR_LAYER_PREFIXES:
            push(f"{prefix}.weight", asd[f"{prefix}.weight"])
            push(f"{prefix}.bias", asd[f"{prefix}.bias"])
        if use_layer_norm:
            for prefix in _LAYERNORM_PREFIXES:
                push(f"{prefix}.weight", asd[f"{prefix}.weight"])
                push(f"{prefix}.bias", asd[f"{prefix}.bias"])
        push(f"{_FC_MU_PREFIX}.weight", asd[f"{_FC_MU_PREFIX}.weight"])
        push(f"{_FC_MU_PREFIX}.bias", asd[f"{_FC_MU_PREFIX}.bias"])
        push(f"{_FC_LOGSTD_PREFIX}.weight", asd[f"{_FC_LOGSTD_PREFIX}.weight"])
        push(f"{_FC_LOGSTD_PREFIX}.bias", asd[f"{_FC_LOGSTD_PREFIX}.bias"])
        push("action_scale", asd["action_scale"])
        push("action_bias", asd["action_bias"])
        push("_mean", nsd["_mean"].squeeze(0))  # (1, obs_dim) -> (obs_dim,)
        push("_std", nsd["_std"].squeeze(0))

    def stk(key: str, *, transpose_weight: bool = False) -> torch.Tensor:
        t = torch.stack(bufs[key], dim=0).to(device)
        if transpose_weight:
            # PyTorch Linear weight is (out, in); einsum 'ni,nio->no' wants
            # (in, out) along the trailing two dims.
            t = t.transpose(-1, -2).contiguous()
        return t

    net_W = [stk(f"{p}.weight", transpose_weight=True) for p in _LINEAR_LAYER_PREFIXES]
    net_b = [stk(f"{p}.bias") for p in _LINEAR_LAYER_PREFIXES]
    if use_layer_norm:
        net_ln_g = [stk(f"{p}.weight") for p in _LAYERNORM_PREFIXES]
        net_ln_b = [stk(f"{p}.bias") for p in _LAYERNORM_PREFIXES]
    else:
        net_ln_g = [None] * len(_LINEAR_LAYER_PREFIXES)
        net_ln_b = [None] * len(_LINEAR_LAYER_PREFIXES)

    return BatchedActorEnsemble(
        net_W=net_W,
        net_b=net_b,
        net_ln_g=net_ln_g,
        net_ln_b=net_ln_b,
        fc_mu_W=stk(f"{_FC_MU_PREFIX}.weight", transpose_weight=True),
        fc_mu_b=stk(f"{_FC_MU_PREFIX}.bias"),
        fc_logstd_W=stk(f"{_FC_LOGSTD_PREFIX}.weight", transpose_weight=True),
        fc_logstd_b=stk(f"{_FC_LOGSTD_PREFIX}.bias"),
        action_scale=stk("action_scale"),
        action_bias=stk("action_bias"),
        obs_mean=stk("_mean"),
        obs_std=stk("_std"),
        obs_eps=obs_eps,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        use_layer_norm=use_layer_norm,
        use_tanh=use_tanh,
    )
