"""One activation producer feeding many SAE consumers.

The partial forward is the expensive part (EDEN-7B); a BatchTopK SAE step is
cheap next to it. ``MultiSAETrainer`` pulls each activation batch once from the
streaming producer and steps *every* SAE on it, so a whole ef/k grid shares one
forward per batch instead of recomputing it per config. Each SAE keeps its own
optimizer (and, under DDP, its own gradient-sync group). This is the engine the
downstream sweep runs on; here it is used to measure shared-activation scaling.

It reuses goodfire-core's SAE forward/loss (``sae(acts, sparsity_weight=...)``
returns an ``SAELoss``); only the outer multi-consumer loop is new.
"""

from __future__ import annotations

import time

import torch

from basecamp_eden_saes.harvest.streaming import StreamingActivationDataset


class MultiSAETrainer:
    """Steps every SAE in a list on one shared streamed activation batch."""

    def __init__(
        self,
        dataset: StreamingActivationDataset,
        saes: list[torch.nn.Module],
        lr: float = 4e-4,
        device: str = "cuda",
        aux_loss_weight: float = 1.0 / 32.0,
    ):
        self.dataset = dataset
        self.saes = saes
        self.device = device
        self.aux_loss_weight = aux_loss_weight
        self.opts = [torch.optim.Adam(s.parameters(), lr=lr) for s in saes]

    def fit(
        self,
        n_steps: int,
        warmup_steps: int = 10,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Run ``n_steps`` shared-forward steps; time the forward and SAE work.

        Returns per-batch timing decomposed into the shared forward cost and the
        aggregate SAE-step cost, plus throughput. ``warmup_steps`` are excluded.
        """
        it = self.dataset.training_iterator(device=self.device)
        it.set_max_steps(n_steps)
        for s in self.saes:
            s.train()

        cuda = torch.cuda.is_available()
        step = 0
        timed_steps = 0
        tokens = 0
        sae_seconds = 0.0
        wall0 = None

        prev = time.time()
        for batch in it.iter_epoch():
            # the producer ran the forward to yield this batch; time since the
            # previous SAE work ended approximates the shared forward + fetch.
            acts = batch.acts
            if cuda:
                torch.cuda.synchronize()
            t_sae0 = time.time()
            for sae, opt in zip(self.saes, self.opts):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    res = sae(acts, sparsity_weight=self.aux_loss_weight)
                res.loss.backward()
                opt.step()
            if cuda:
                torch.cuda.synchronize()
            t_sae1 = time.time()

            if step == warmup_steps:
                wall0 = prev  # start the wall clock at first timed batch boundary
                tokens = 0
                sae_seconds = 0.0
                timed_steps = 0
            if step >= warmup_steps:
                tokens += acts.shape[0]
                sae_seconds += t_sae1 - t_sae0
                timed_steps += 1
            prev = t_sae1
            step += 1
            if step >= n_steps:
                break

        wall = time.time() - (wall0 if wall0 is not None else prev)
        n = len(self.saes)
        return {
            "n_saes": n,
            "timed_steps": timed_steps,
            "tokens": int(tokens),
            "wall_seconds": wall,
            "sae_seconds": sae_seconds,
            "forward_seconds": max(wall - sae_seconds, 0.0),
            "tokens_per_sec": tokens / max(wall, 1e-9),
            "per_batch_seconds": wall / max(timed_steps, 1),
            "per_batch_sae_seconds": sae_seconds / max(timed_steps, 1),
            "per_batch_forward_seconds": max(wall - sae_seconds, 0.0)
            / max(timed_steps, 1),
        }
