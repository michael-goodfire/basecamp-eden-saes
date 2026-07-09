"""DDP-aware full-batch SAE metric reduction for the shared-forward sweep.

The :class:`~basecamp_eden_saes.sae.trainer.SweepTrainer` logs per-step training
metrics from rank 0 only, computed on that rank's *local* SAE batch (per-GPU
8,192 tokens in v3). Under 8-GPU data parallelism that shard is 1/8 of the
65,536-token global batch the optimizer actually steps on, so the logged
variance-explained / loss / L0 curves are a noisy single-shard view of training
(this is the noisy-curve artifact flagged after v2).

This module recomputes those metrics on the **global batch** by all-reducing
sufficient statistics across ranks:

* Variance-explained is a ratio of variances, so per-rank VE cannot simply be
  averaged. We all-reduce the element-wise sum / sum-of-squares of the
  normalized input ``x`` and the reconstruction residual ``x - x_hat`` -- exactly
  the accumulation :func:`~basecamp_eden_saes.sae.trainer.eval_sae_chunked` uses
  for the held-out VE -- then form the global variances on every rank.
* The loss scalars and L0 are per-token means over equal-size shards, so their
  global value is the cross-rank token-weighted mean, obtained from the same
  all-reduced token / element counts.

Dead-feature fraction and the BatchTopK threshold are already DDP-synced inside
goodfire-core (the model's ``last_activated_at`` MAX all-reduce and the sparsity
threshold MIN all-reduce), so those are left to the base callback unchanged.

:class:`DDPSAEMetricsCallback` wraps the reduction as a drop-in
:class:`goodfire_core.saes.callbacks.SAEMetricsCallback` subclass and is the
intended upstreaming target. The sweep trainer calls
:func:`all_reduce_global_sae_metrics` directly so the collective runs on *every*
rank -- calling it on rank 0 alone would deadlock the all-reduce.
"""

from __future__ import annotations

import torch

from goodfire_core.saes.callbacks import SAEMetricsCallback
from goodfire_core.saes.interfaces import SAE, SAELoss

try:  # distributed is only used under torchrun
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore

# Index layout of the all-reduced sufficient-statistic vector.
(
    _SX,       # sum(x)
    _SXX,      # sum(x*x)
    _SR,       # sum(resid)
    _SRR,      # sum(resid*resid)
    _NELEM,    # number of x elements (tokens * d_model)
    _LOSS_W,   # total loss * n_tok (token-weighted)
    _RECON_W,  # reconstruction loss * n_tok
    _SPARSE_W, # sparsity loss * n_tok
    _L0_W,     # total nonzero feature entries over the batch
    _NTOK,     # number of tokens
    _NFIELDS,
) = range(11)


@torch.no_grad()
def all_reduce_global_sae_metrics(
    sae: SAE,
    batch_acts: torch.Tensor,
    loss_obj: SAELoss,
    world_size: int,
    device: str | torch.device,
) -> dict[str, float]:
    """Global-batch VE / loss / L0 for one SAE, all-reduced across DDP ranks.

    MUST be called on every rank in the same order: it issues a single
    ``all_reduce`` collective. Returns the canonical goodfire-core metric keys so
    the caller can overwrite the rank-local values produced by
    :class:`~goodfire_core.saes.callbacks.SAEMetricsCallback`.

    Single-GPU (``world_size <= 1`` or no process group) is a no-op reduction:
    the returned metrics are simply the exact full-batch values for this rank.
    """
    feats = loss_obj.sae_features
    if feats is None:
        return {}
    feats = feats.detach()
    # Recompute x / x_hat in fp32. The training forward ran under bf16 autocast;
    # the held-out eval likewise accumulates VE in fp32 for accuracy. decode() of
    # the stashed sparse features reproduces the same x_hat the loss used (the
    # standard MSE objective reconstructs via decode(features)).
    x, _ = sae._scale_input(batch_acts.float())
    x_hat = sae.decode(feats.float())
    resid = x - x_hat

    xf = x.float()
    rf = resid.float()
    n_tok = feats.shape[0]
    nonzero = (feats > 0).sum().double()  # total nonzero entries over the batch

    stats = torch.zeros(_NFIELDS, dtype=torch.float64, device=device)
    stats[_SX] = xf.sum()
    stats[_SXX] = (xf * xf).sum()
    stats[_SR] = rf.sum()
    stats[_SRR] = (rf * rf).sum()
    stats[_NELEM] = xf.numel()
    stats[_LOSS_W] = float(loss_obj.loss.detach()) * n_tok
    stats[_RECON_W] = float(loss_obj.reconstruction_loss.detach()) * n_tok
    stats[_SPARSE_W] = float(loss_obj.sparsity_loss.detach()) * n_tok
    stats[_L0_W] = nonzero
    stats[_NTOK] = n_tok

    if dist is not None and dist.is_initialized() and world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    s = stats.tolist()
    n_elem = max(s[_NELEM], 1.0)
    n_tok_g = max(s[_NTOK], 1.0)
    var_x = s[_SXX] / n_elem - (s[_SX] / n_elem) ** 2
    var_r = s[_SRR] / n_elem - (s[_SR] / n_elem) ** 2
    ve = 1.0 - var_r / max(var_x, 1e-12)
    return {
        "losses/global_variance_explained": float(ve),
        "losses/loss": s[_LOSS_W] / n_tok_g,
        "losses/reconstruction_loss": s[_RECON_W] / n_tok_g,
        "losses/sparsity_loss": s[_SPARSE_W] / n_tok_g,
        "features/l0_sparsity": s[_L0_W] / n_tok_g,
    }


class DDPSAEMetricsCallback(SAEMetricsCallback):
    """SAEMetricsCallback whose VE / loss / L0 reflect the global DDP batch.

    Upstreaming target. Identical to the base callback except that, at a logging
    step, the reconstruction-quality and sparsity scalars are recomputed on the
    all-reduced global batch instead of the local shard. The collective inside
    :func:`all_reduce_global_sae_metrics` requires this to be invoked on *every*
    rank in lockstep; the sweep trainer therefore calls the free function
    directly (and uses this class only on rank 0 for the remaining namespace --
    dead features, LR, data stats). Provided so the reduction is available as a
    self-contained callback for other goodfire-core training loops.

    Pass ``world_size`` and ``device`` through ``compute_step_metrics`` kwargs.
    """

    def compute_step_metrics(self, step: int, **kwargs):  # type: ignore[override]
        metrics = super().compute_step_metrics(step, **kwargs)
        if not metrics:
            return metrics
        sae = self._model_ref
        loss_obj = kwargs.get("loss_obj")
        batch_acts = kwargs.get("batch_acts")
        world_size = int(kwargs.get("world_size", 1))
        device = kwargs.get(
            "device", batch_acts.device if batch_acts is not None else "cpu"
        )
        if sae is not None and loss_obj is not None and batch_acts is not None:
            metrics.update(
                all_reduce_global_sae_metrics(
                    sae, batch_acts, loss_obj, world_size, device
                )
            )
        return metrics
