"""On-the-fly activation backend + shared SAE construction helpers.

``StreamingActivationDataset`` / ``StreamingIterator`` implement the duck-typed
surface ``goodfire_core.saes.trainer.train_sae`` requires (``steps_per_epoch``,
``set_max_steps``, ``iter_epoch`` yielding objects with ``.acts`` / ``.sequence_ids``,
``advance_epoch``, ``cleanup``, ``d_model``), so the streaming backend is a true
drop-in for the on-disk ``ActivationDataset.training_iterator`` -- the only thing
that differs between the two benchmark arms is where activations come from.

Each training batch runs EDEN's partial forward over a pool of corpus windows,
flattens valid token positions, and emits fixed-size token batches. Normalizer
statistics (per-coordinate mean, mean L2 norm, std of the L2 norm) are estimated
by folding running averages over the first few M streamed tokens, then frozen
into the SAE at construction -- no separate stats pass, no second forward.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np
import torch

from goodfire_core.saes.batch_topk import BatchTopKSAE

from basecamp_eden_saes.corpus.build import CorpusReader

from .eden import EdenPartialForward


class StreamingBatch:
    """A single fixed-size token batch handed to the SAE trainer."""

    __slots__ = ("acts", "sequence_ids")

    def __init__(self, acts: torch.Tensor, sequence_ids: torch.Tensor | None = None):
        self.acts = acts
        self.sequence_ids = sequence_ids


class StreamingIterator:
    """Producer that runs EDEN forwards and emits fixed-size token batches."""

    def __init__(
        self,
        dataset: "StreamingActivationDataset",
        device: str,
        n_epochs: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.ds = dataset
        self.device = device
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.seed = seed
        self.d_model = dataset.d_model
        self._epoch = 0
        self._max_steps: int | None = None

    @property
    def steps_per_epoch(self) -> int:
        if self._max_steps is not None:
            return self._max_steps
        idx = self.ds._rank_windows()
        total_tokens = int(self.ds.reader.index[idx, 1].sum())
        return max(1, total_tokens // self.ds.sae_batch_tokens)

    def set_max_steps(self, max_steps: int) -> None:
        """Cap the number of batches produced (used by the trainer)."""
        self._max_steps = int(max_steps)

    def advance_epoch(self) -> None:
        self._epoch += 1

    def cleanup(self) -> None:
        pass

    def iter_epoch(self) -> Iterator[StreamingBatch]:
        """Yield ``StreamingBatch`` objects of ``sae_batch_tokens`` tokens each."""
        reader = self.ds.reader
        eden = self.ds.eden
        bt = self.ds.sae_batch_tokens
        # Restrict to this dataset's window subset (e.g. train split) and shard
        # by rank so the 8 DDP replicas see disjoint windows (data-parallel).
        order = self.ds._rank_windows().copy()
        rng = np.random.default_rng(self.seed + self._epoch)
        if self.shuffle:
            rng.shuffle(order)

        max_steps = self._max_steps
        steps = 0
        pool: list[torch.Tensor] = []
        pool_n = 0
        pos = 0
        n = len(order)
        while True:
            # refill the pool with one forward-batch of windows
            while pool_n < bt:
                if pos >= n:
                    if max_steps is None:
                        break
                    # loop the corpus to satisfy the requested step count
                    rng.shuffle(order)
                    pos = 0
                idx = order[pos : pos + self.ds.fwd_windows]
                pos += self.ds.fwd_windows
                windows = [reader.get(int(j)) for j in idx]
                flat = eden.tokens_from_windows(windows).to(self.ds.act_dtype)
                pool.append(flat)
                pool_n += flat.shape[0]
            if pool_n < bt:
                break  # corpus exhausted, no max_steps
            cat = torch.cat(pool, dim=0)
            batch = cat[:bt]
            rem = cat[bt:]
            pool = [rem] if rem.shape[0] else []
            pool_n = rem.shape[0]
            yield StreamingBatch(acts=batch, sequence_ids=None)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return


class StreamingActivationDataset:
    """On-the-fly activation backend over a built corpus."""

    def __init__(
        self,
        reader: CorpusReader,
        eden: EdenPartialForward,
        sae_batch_tokens: int = 16384,
        fwd_windows: int = 16,
        act_dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        window_subset: np.ndarray | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.reader = reader
        self.eden = eden
        self.sae_batch_tokens = sae_batch_tokens
        self.fwd_windows = fwd_windows
        self.act_dtype = act_dtype
        self.seed = seed
        self.d_model = eden.d_model
        # window_subset restricts which corpus windows this dataset draws from
        # (e.g. a train/held-out split). rank/world_size shard that subset so
        # each DDP replica streams a disjoint slice of windows.
        if window_subset is None:
            window_subset = np.arange(len(reader))
        self.window_subset = np.asarray(window_subset)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def _rank_windows(self) -> np.ndarray:
        """This rank's disjoint slice of the window subset (deterministic stride)."""
        if self.world_size <= 1:
            return self.window_subset
        return self.window_subset[self.rank :: self.world_size]

    def training_iterator(
        self,
        device: str | None = None,
        n_epochs: int = 1,
        shuffle: bool = True,
        **_: Any,
    ) -> StreamingIterator:
        """Return a :class:`StreamingIterator` over this dataset's windows."""
        return StreamingIterator(
            self,
            device=device or self.eden.device,
            n_epochs=n_epochs,
            shuffle=shuffle,
            seed=self.seed,
        )


# --- normalizer statistics (folded into the stream) --------------------------


def estimate_normalizer_stats(
    eden: EdenPartialForward,
    reader: CorpusReader,
    n_tokens: int = 2_000_000,
    fwd_windows: int = 16,
    seed: int = 42,
    window_subset: np.ndarray | None = None,
) -> dict[str, Any]:
    """Stream ~``n_tokens`` and return ``{per_coord_mean, mean_norm, std_norm}``.

    Per-coordinate mean is accumulated with a numerically stable running update;
    the mean and std of the per-token L2 norm are accumulated as running moments.
    ``window_subset`` restricts the stream to those window indices (e.g. the
    train split) so held-out windows never enter the normalizer fit.
    """
    d = eden.d_model
    mean = torch.zeros(d, dtype=torch.float64, device=eden.device)
    count = 0
    norm_sum = 0.0
    norm_sq_sum = 0.0
    n_norm = 0
    # Build the window order from the (optional) subset, shuffled deterministically.
    if window_subset is None:
        order = np.arange(len(reader))
    else:
        order = np.asarray(window_subset).copy()
    np.random.default_rng(seed).shuffle(order)

    def _batches() -> Iterator[list[np.ndarray]]:
        for s in range(0, len(order), fwd_windows):
            yield [reader.get(int(j)) for j in order[s : s + fwd_windows]]

    for windows in _batches():
        flat = eden.tokens_from_windows(windows).to(torch.float32)
        norms = torch.linalg.vector_norm(flat, dim=1)
        norm_sum += float(norms.sum().item())
        norm_sq_sum += float((norms * norms).sum().item())
        n_norm += norms.numel()
        bn = flat.shape[0]
        new_count = count + bn
        mean = mean + (flat.double().sum(0) - bn * mean) / new_count
        count = new_count
        if count >= n_tokens:
            break
    mean_norm = norm_sum / max(n_norm, 1)
    var_norm = max(norm_sq_sum / max(n_norm, 1) - mean_norm * mean_norm, 0.0)
    return {
        "per_coord_mean": mean.float().cpu(),
        "mean_norm": float(mean_norm),
        "std_norm": float(math.sqrt(var_norm)),
        "num_tokens": int(count),
    }


def build_sae(
    d_model: int,
    ef: int,
    k: int,
    stats: dict[str, Any],
    device: str = "cuda",
    normalizers: tuple[str, ...] = ("mean_center", "scalar"),
    aux_k: int = 512,
) -> BatchTopKSAE:
    """Construct a BatchTopK SAE with the requested normalizer pipeline baked in."""
    cfg = [{"type": n} for n in normalizers]
    sae = BatchTopKSAE(
        d_model=d_model,
        d_sae=ef * d_model,
        k=k,
        aux_k=aux_k,
        normalizers=cfg,
        stats=stats,
    )
    if not sae.normalizers.is_fitted:
        raise RuntimeError(
            f"SAE normalizers not fitted: {sae.normalizers.unfit_layer_names()}"
        )
    return sae.to(device)
