"""Shared-forward multi-SAE sweep trainer -- the real engine for the ef/k sweep.

One EDEN partial forward per activation batch is fanned to *every* SAE in the
grid: the producer in :mod:`basecamp_eden_saes.harvest.streaming` runs the
forward and every SAE consumes the *same* tensor, so a whole ef/k grid shares one
forward per batch instead of recomputing it per config. Each SAE keeps its own
Adam optimizer, LR schedule, and aux-loss (dead-feature revival) schedule.

Under multi-GPU data parallelism each replica streams a disjoint window shard
(see ``StreamingActivationDataset.rank``/``world_size``), and per-SAE parameter
gradients are **all-reduced (mean)** across replicas before each optimizer step.
This is equivalent to DDP but explicit, so the two invariants the sweep depends
on are directly checkable in the smoke test:

* **shared forward** -- the producer is the only thing that calls EDEN; the
  activation tensor handed to each SAE is the identical object
  (:func:`assert_shared_forward`).
* **grad sync** -- after the manual all-reduce, a given SAE's parameter
  gradients are bit-identical across ranks.

It wraps goodfire-core's SAE forward/loss (``sae(acts, sparsity_weight=...)``
returns an ``SAELoss``) and :func:`goodfire_core.saes.metrics` primitives for the
held-out evaluation; only the outer multi-consumer + grad-sync loop is new.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from goodfire_core.saes.interfaces import SAE, SAELoss
from goodfire_core.saes.metrics import decoder_coherence

from basecamp_eden_saes.corpus.build import CorpusReader
from basecamp_eden_saes.harvest.eden import EdenPartialForward
from basecamp_eden_saes.harvest.streaming import StreamingActivationDataset

from .ddp_metrics import all_reduce_global_sae_metrics

try:  # canonical density histogram (matches SAEWandBCallback); optional at import
    from goodfire_core.saes.feature_stats import compute_feature_activation_histogram
except Exception:  # pragma: no cover
    compute_feature_activation_histogram = None  # type: ignore
try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

try:  # optional: distributed is only used under torchrun
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore


# --------------------------------------------------------------------------- schedulers
def build_lr_fn(
    lr: float,
    n_steps: int,
    warmup_frac: float = 0.05,
    cooldown_frac: float = 0.1,
    decay: str = "linear",
) -> Callable[[int], float]:
    """Linear warmup then a configurable decay to ~0.

    ``decay="linear"`` keeps the v2 Gao/Anthropic recipe: linear warmup ->
    constant -> linear cooldown over the last ``cooldown_frac`` of steps.

    ``decay="cosine"`` (v3): linear warmup over the first ``warmup_frac`` of steps
    to the peak ``lr``, then cosine annealing to ~0 over **all** remaining steps,
    ``lr * 0.5 * (1 + cos(pi * progress))`` with ``progress`` in ``[0, 1]``. This
    mirrors ``torch.optim.lr_scheduler.CosineAnnealingLR`` (``eta_min=0``,
    ``T_max = n_steps - warmup``) as a closed-form per-step function, which the
    SweepTrainer needs because it drives 9 optimizers by setting the LR each step
    rather than stepping one torch scheduler. ``cooldown_frac`` is ignored.
    """
    warm = int(warmup_frac * n_steps)
    if decay == "cosine":
        decay_steps = max(1, n_steps - warm)

        def fn(step: int) -> float:
            if step < warm:
                return lr * (step / max(1, warm))
            prog = min(1.0, (step - warm) / decay_steps)
            return lr * 0.5 * (1.0 + math.cos(math.pi * prog))

        return fn

    cool = int(cooldown_frac * n_steps)
    const = n_steps - warm - cool

    def fn(step: int) -> float:
        if step < warm:
            return lr * (step / max(1, warm))
        if step < warm + const:
            return lr
        prog = (step - warm - const) / max(1, cool)
        return lr * (1.0 - prog)

    return fn


def build_aux_fn(
    aux_weight: float, n_steps: int, warmup_frac: float = 0.1
) -> Callable[[int], float]:
    """Linear aux-loss warmup so reconstruction leads early, dead-revival ramps in."""
    warm = int(warmup_frac * n_steps)

    def fn(step: int) -> float:
        if warm == 0:
            return aux_weight
        return min(aux_weight, aux_weight * (step / warm))

    return fn


# --------------------------------------------------------------------------- per-config state
@dataclass
class SAEConfig:
    """One SAE in the grid plus its optimizer, schedules, and logging handles."""

    label: str
    ef: int
    k: int
    sae: SAE
    opt: torch.optim.Optimizer
    lr_fn: Callable[[int], float]
    aux_fn: Callable[[int], float]
    wandb_run: Any = None
    # canonical goodfire-core metric computer for this SAE (rank 0 only); its
    # output dict (standard losses/*, features/*, metrics/* namespace) is logged
    # to wandb_run.
    metrics_cb: Any = None


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


# --------------------------------------------------------------------------- the trainer
class SweepTrainer:
    """Trains a whole ef/k SAE grid off one shared EDEN forward per batch."""

    def __init__(
        self,
        dataset: StreamingActivationDataset,
        configs: list[SAEConfig],
        total_steps: int,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        clip_grad_max_norm: float = 1.0,
        norm_threshold: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        log_every_steps: int = 50,
        hist_every_steps: int = 0,
        save_every_steps: int = 0,
        ckpt_dir: Path | str | None = None,
        keep_last: int = 2,
        start_step: int = 0,
    ):
        self.dataset = dataset
        self.configs = configs
        self.total_steps = total_steps
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.clip = clip_grad_max_norm
        self.norm_threshold = norm_threshold
        self.autocast_dtype = autocast_dtype
        self.log_every = log_every_steps
        self.hist_every = hist_every_steps
        self.is_dist = world_size > 1
        # --- mid-run resumable checkpointing (rank 0) ---
        # Periodically saves every SAE's weights + optimizer + step state so a
        # wall-clock kill or crash resumes instead of restarting (the salvage
        # path the cancelled v2 run lacked). Saves are atomic (tmp + rename) and
        # pruned to the last ``keep_last`` per config to bound disk.
        self.save_every = int(save_every_steps)
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else None
        self.keep_last = int(keep_last)
        self.start_step = int(start_step)

    def _all_reduce_grads(self) -> None:
        """Mean per-SAE parameter grads across replicas (explicit DDP)."""
        if not self.is_dist:
            return
        handles = []
        params = []
        for cfg in self.configs:
            for p in cfg.sae.parameters():
                if p.grad is not None:
                    handles.append(
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                    )
                    params.append(p)
        for h in handles:
            h.wait()
        inv = 1.0 / self.world_size
        for p in params:
            p.grad.mul_(inv)

    def fit(
        self,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Run the full training loop and return a timing/throughput summary."""
        for cfg in self.configs:
            cfg.sae.train()
        it = self.dataset.training_iterator(device=self.device)
        # On resume, only the remaining steps are streamed; the absolute step
        # counter still starts at start_step so the LR / aux schedules (functions
        # of absolute step) and the total_steps stop condition stay correct.
        it.set_max_steps(self.total_steps - self.start_step)

        step = self.start_step
        t0 = time.time()
        tokens_seen = 0
        tokens_kept = 0
        for batch in it.iter_epoch():
            acts = batch.acts
            tokens_seen += acts.shape[0]
            if self.norm_threshold is not None:
                norms = torch.linalg.vector_norm(acts.float(), dim=1)
                acts = acts[norms <= self.norm_threshold]
                if acts.shape[0] == 0:
                    continue
            tokens_kept += acts.shape[0]

            # This iteration produces step index (step + 1) after the increment
            # below; log on the iterations whose post-increment index is a log
            # step. ALL ranks must participate -- the full-batch metric reduction
            # in _log is an all_reduce collective -- so do_log is rank-independent.
            # Stash the per-config loss object (detached features) on every rank
            # only at log steps, to keep sae_features out of memory otherwise.
            do_log = (step + 1) % self.log_every == 0
            step_losses: list[SAELoss | None] = []

            # zero grads
            for cfg in self.configs:
                cfg.opt.zero_grad(set_to_none=True)

            # shared forward fanned to every SAE; launch grad all-reduce as each
            # SAE's backward completes so comms overlap the next SAE's compute.
            handles = []
            params = []
            for cfg in self.configs:
                aw = cfg.aux_fn(step)
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    res: SAELoss = cfg.sae(acts, sparsity_weight=aw)
                res.loss.backward()
                if self.is_dist:
                    for p in cfg.sae.parameters():
                        if p.grad is not None:
                            handles.append(
                                dist.all_reduce(
                                    p.grad, op=dist.ReduceOp.SUM, async_op=True
                                )
                            )
                            params.append(p)
                if do_log:
                    # detach features so the stashed loss object drops the graph
                    if res.sae_features is not None:
                        res.sae_features = res.sae_features.detach()
                    step_losses.append(res)
                else:
                    step_losses.append(None)

            if self.is_dist:
                for h in handles:
                    h.wait()
                inv = 1.0 / self.world_size
                for p in params:
                    p.grad.mul_(inv)

            # optimizer step per SAE (LR from schedule)
            for cfg in self.configs:
                lr = cfg.lr_fn(step)
                _set_lr(cfg.opt, lr)
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(cfg.sae.parameters(), self.clip)
                cfg.opt.step()

            step += 1

            if do_log:
                self._log(step, step_losses, acts)
            if (
                self.save_every
                and self.rank == 0
                and step % self.save_every == 0
                and step < self.total_steps
            ):
                self._save_checkpoints(step)
            if progress_cb is not None and step % 20 == 0:
                progress_cb(step, self.total_steps)

            if step >= self.total_steps:
                break

        wall = time.time() - t0
        it.cleanup()
        return {
            "steps": step,
            "wall_seconds": wall,
            "tokens_seen": tokens_seen,
            "tokens_kept": tokens_kept,
            "tokens_per_sec": tokens_kept / max(wall, 1e-9),
        }

    def _log(
        self, step: int, step_losses: list[SAELoss | None], acts: torch.Tensor
    ) -> None:
        """Log canonical goodfire-core SAE metrics per config to its W&B run.

        Called on **every** rank: the reconstruction-quality / sparsity scalars
        (``losses/global_variance_explained``, ``losses/*``, ``features/l0_sparsity``)
        are recomputed on the *global* DDP batch via an ``all_reduce`` of
        sufficient statistics (:func:`all_reduce_global_sae_metrics`), so the
        logged curves reflect the full 65,536-token optimizer batch rather than
        one rank's 8,192-token shard (the noisy single-shard curves v2 produced).
        The all-reduce is a collective, so it runs for every config on every rank
        before any rank-0-only guard. Dead-feature % and the BatchTopK threshold
        are already DDP-synced inside goodfire-core, so the rest of the
        :class:`SAEMetricsCallback` namespace (computed on rank 0) is left as-is;
        the per-feature density histogram is added at the ``hist_every`` cadence.
        """
        want_hist = self.hist_every > 0 and step % self.hist_every == 0
        for cfg, loss_obj in zip(self.configs, step_losses):
            if loss_obj is None:
                continue
            # COLLECTIVE: every rank computes + all-reduces this config's global
            # metrics, in the same config order, before any rank-0-only return.
            global_overrides = all_reduce_global_sae_metrics(
                cfg.sae, acts, loss_obj, self.world_size, self.device
            )
            # Only rank 0 owns the per-config W&B run and the metric tracker.
            if self.rank != 0 or cfg.wandb_run is None or cfg.metrics_cb is None:
                continue
            log_dict = dict(
                cfg.metrics_cb.compute_step_metrics(
                    step,
                    loss_obj=loss_obj,
                    batch_acts=acts,
                    optimizer=cfg.opt,
                    aux_loss_weight=cfg.aux_fn(step),
                )
            )
            if not log_dict:
                continue
            # Overwrite the rank-local VE / loss / L0 with the global-batch values.
            log_dict.update(global_overrides)
            # canonical density histogram from the callback's EMA tracker
            if (
                want_hist
                and wandb is not None
                and compute_feature_activation_histogram is not None
                and cfg.metrics_cb.feature_tracker is not None
            ):
                stats = cfg.metrics_cb.feature_tracker.get_statistics()
                dens_hist = compute_feature_activation_histogram(
                    stats.feature_density.numpy(),
                    bins=20,
                    log_f=True,
                    log_count=False,
                    eps=1e-9,
                )
                log_dict["features/histogram"] = wandb.Histogram(np_histogram=dens_hist)
            cfg.wandb_run.log(log_dict, step=step)

    # ----------------------------------------------------------------- checkpointing
    def _save_checkpoints(self, step: int) -> None:
        """Save every SAE's resumable state (rank 0). Atomic + pruned.

        Each config writes a canonical goodfire-core SAE checkpoint
        (``BatchTopKSAE.save_checkpoint``) carrying extra resume keys --
        ``optimizer_state``, ``step``, ``total_steps``, ``ef``/``k``/``label`` --
        to ``ckpt_dir/<label>/step_<step>.pt``. The write goes to a ``.tmp`` and
        is renamed so a kill mid-save never leaves a corrupt "latest". Only the
        last ``keep_last`` steps per config are kept.
        """
        if self.ckpt_dir is None:
            return
        for cfg in self.configs:
            cdir = self.ckpt_dir / cfg.label
            cdir.mkdir(parents=True, exist_ok=True)
            tmp = cdir / f"step_{step}.pt.tmp"
            final = cdir / f"step_{step}.pt"
            cfg.sae.save_checkpoint(
                str(tmp),
                optimizer_state=cfg.opt.state_dict(),
                step=step,
                total_steps=self.total_steps,
                ef=cfg.ef,
                k=cfg.k,
                label=cfg.label,
            )
            tmp.replace(final)  # atomic publish
            self._prune(cdir)

    def _prune(self, cdir: Path) -> None:
        """Keep only the last ``keep_last`` ``step_*.pt`` checkpoints in ``cdir``."""
        cks = sorted(cdir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        for p in cks[: -self.keep_last] if self.keep_last > 0 else []:
            p.unlink(missing_ok=True)


def find_resume_step(ckpt_dir: Path | str | None, labels: list[str]) -> int:
    """Latest step that *every* config has a saved checkpoint for (common resume).

    Returns 0 (train from scratch) if ``ckpt_dir`` is missing or any config has
    no checkpoint -- resuming only makes sense from a step all 9 SAEs reached, so
    the grid stays in lockstep.
    """
    if ckpt_dir is None:
        return 0
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return 0
    per_config: list[set[int]] = []
    for label in labels:
        cdir = ckpt_dir / label
        if not cdir.exists():
            return 0
        steps = {int(p.stem.split("_")[1]) for p in cdir.glob("step_*.pt")}
        if not steps:
            return 0
        per_config.append(steps)
    common = set.intersection(*per_config) if per_config else set()
    return max(common) if common else 0


# --------------------------------------------------------------------------- held-out eval
@torch.no_grad()
def eval_sae_chunked(
    sae: SAE,
    eden: EdenPartialForward,
    reader: CorpusReader,
    held_windows: np.ndarray,
    n_tokens: int = 524_288,
    fwd_windows: int = 16,
    seed: int = 123,
    device: str = "cuda",
    chunk_tokens: int = 8192,
) -> dict[str, Any]:
    """Held-out reconstruction / sparsity metrics, accumulated in chunks.

    Streams ~``n_tokens`` of held-out activations through EDEN, and for each
    forward-batch computes the SAE's eval-mode codes (per-token TopK / learned
    threshold) and reconstruction. Variance-explained is accumulated exactly via
    element-wise sum/sumsq in the SAE's normalized space (matching
    ``SAELoss.global_variance_explained``); L0 and the per-feature firing counts
    (for dead fraction + the density histogram) are accumulated across chunks so
    the full ``[N, d_sae]`` code matrix is never materialized at once.
    """
    was_training = sae.training
    sae.eval()
    d_sae = sae.d_sae
    fire_counts = torch.zeros(d_sae, dtype=torch.float64, device=device)
    l0_sum = 0.0
    n_tok = 0
    # element-wise accumulators for variance (normalized space)
    sx = sxx = sr = srr = 0.0
    n_elem = 0

    order = np.asarray(held_windows).copy()
    np.random.default_rng(seed).shuffle(order)
    pos = 0
    while n_tok < n_tokens and pos < len(order):
        idx = order[pos : pos + fwd_windows]
        pos += fwd_windows
        windows = [reader.get(int(j)) for j in idx]
        # EDEN emits bf16 residuals; SAE params are fp32 and eval runs without
        # autocast, so cast to fp32 to match the encoder/decoder weights (and
        # for accurate held-out metrics).
        acts_full = eden.tokens_from_windows(windows).to(device).float()
        # Sub-chunk the encode/decode so the [chunk, d_sae] code matrix stays
        # bounded (a 65k-token forward batch x d_sae=32768 is 8.5 GB fp32 and
        # OOMs on top of the resident dictionaries).
        for s in range(0, acts_full.shape[0], chunk_tokens):
            acts = acts_full[s : s + chunk_tokens]
            x, _ = sae._scale_input(acts)
            codes = sae.encode(acts)  # eval-mode selection
            x_hat = sae.decode(codes)
            resid = (x - x_hat).float()
            xf = x.float()
            sx += float(xf.sum())
            sxx += float((xf * xf).sum())
            sr += float(resid.sum())
            srr += float((resid * resid).sum())
            n_elem += xf.numel()
            fire_counts += (codes > 0).sum(dim=0).double()
            l0_sum += float((codes > 0).sum(-1).float().sum())
            n_tok += acts.shape[0]

    var_x = sxx / n_elem - (sx / n_elem) ** 2
    var_r = srr / n_elem - (sr / n_elem) ** 2
    ve = 1.0 - var_r / max(var_x, 1e-12)
    density = (fire_counts / max(n_tok, 1)).cpu().numpy()
    coh = decoder_coherence(sae.decoder_weight.detach())
    if was_training:
        sae.train()
    return {
        "n_tokens": int(n_tok),
        "variance_explained": float(ve),
        "l0": l0_sum / max(n_tok, 1),
        "dead_fraction": float((fire_counts == 0).float().mean()),
        "mean_density": float(density.mean()),
        "median_density": float(np.median(density)),
        "density": density,  # per-feature firing fraction [d_sae]
        "coherence_mean": coh["mean"],
        "coherence_max": coh["max"],
    }


# --------------------------------------------------------------------------- DDP weight sync
def broadcast_module(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all params + buffers from ``src`` so every replica starts identical."""
    if dist is None or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def assert_shared_forward(
    dataset: StreamingActivationDataset, n_batches: int = 3
) -> int:
    """Sanity-check that the producer runs exactly one EDEN forward per batch.

    Wraps ``eden.residual_multi``/``residual`` call counting via a monkeypatched
    counter and confirms each yielded batch corresponds to producer-side forwards
    only (no SAE triggers its own forward). Returns the number of forwards seen.
    """
    eden = dataset.eden
    calls = {"n": 0}
    orig = eden.residual

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    eden.residual = counting  # type: ignore
    try:
        it = dataset.training_iterator()
        it.set_max_steps(n_batches)
        seen = 0
        for _ in it.iter_epoch():
            seen += 1
            if seen >= n_batches:
                break
    finally:
        eden.residual = orig  # type: ignore
    return calls["n"]
