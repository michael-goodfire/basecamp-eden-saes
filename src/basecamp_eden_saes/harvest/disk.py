"""Pre-harvest backend: write EDEN layer-L activations to disk via
goodfire-core's ``ActivationWriter``, then read them back through an
``ActivationDataset`` whose ``training_iterator`` is the same duck-type the
streaming backend implements -- so both arms feed ``train_sae`` identically.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from goodfire_core.storage.filesystem import FilesystemStorage
from goodfire_core.storage.reader import ActivationDataset
from goodfire_core.storage.writer import ActivationWriter

from basecamp_eden_saes import config
from basecamp_eden_saes.corpus.build import CorpusReader

from .eden import EdenPartialForward

logger = logging.getLogger(__name__)


def harvest_to_disk(
    eden: EdenPartialForward,
    reader: CorpusReader,
    out_root: str | Path,
    dataset_name: str,
    n_tokens: int,
    fwd_windows: int = 16,
    dtype: str = "bfloat16",
    shuffle: bool = True,
) -> dict:
    """Partial-forward windows and stream layer-L token activations to disk.

    Returns timing + size summary (one-time harvest cost for Arm A).
    """
    storage = FilesystemStorage(Path(out_root))
    # shuffle=False so the writer flushes chunks to disk as it goes (bounded
    # RAM). A global shuffle buffer would hold the entire harvest in memory --
    # impossible at the 2B-token / 16 TB scale this benchmark extrapolates to.
    writer = ActivationWriter(
        storage,
        dataset_name,
        d_model=eden.d_model,
        mode="token",
        dtype=dtype,
        shuffle=False,
        overwrite=True,
        compute_checksum=False,
    )
    written = 0
    t0 = time.time()
    fwd_seconds = 0.0
    for windows in reader.iter_batches(fwd_windows, shuffle=False):
        tf = time.time()
        flat = eden.tokens_from_windows(windows)  # [N, d] on device
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        fwd_seconds += time.time() - tf
        writer.add(flat.detach().to("cpu"))
        written += flat.shape[0]
        if written >= n_tokens:
            break
    writer.finalize()
    wall = time.time() - t0

    root = Path(out_root) / dataset_name
    nbytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return {
        "dataset_name": dataset_name,
        "tokens": int(written),
        "wall_seconds": wall,
        "forward_seconds": fwd_seconds,
        "forward_tokens_per_sec": written / max(fwd_seconds, 1e-9),
        "harvest_tokens_per_sec": written / max(wall, 1e-9),
        "bytes_on_disk": int(nbytes),
        "gb_per_million_tokens": (nbytes / 1e9) / (written / 1e6) if written else 0.0,
    }


def make_disk_iterator(
    out_root: str | Path,
    dataset_name: str,
    batch_size: int,
    device: str = "cuda",
    n_epochs: int = 1,
    shuffle: bool = True,
    strategy: str = "mmap_stream",
):
    """Build the on-disk training iterator (Arm A's train-phase data source).

    ``strategy="mmap_stream"`` reads activations from disk on demand with
    bounded RAM -- the only viable strategy at the 16 TB / 2B-token scale this
    benchmark extrapolates to (gpu_preload / pinned_cache would need the whole
    dataset resident). Forcing it here keeps the measured throughput honest for
    that scenario rather than auto-selecting an unrealistic in-memory read.
    """
    storage = FilesystemStorage(Path(out_root))
    ds = ActivationDataset(storage, dataset_name, batch_size=batch_size)
    return ds.training_iterator(
        device=device, n_epochs=n_epochs, shuffle=shuffle, strategy=strategy
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``bes-harvest`` (pre-harvest activations to disk)."""
    p = argparse.ArgumentParser(
        description="Harvest EDEN layer-L activations to a goodfire-core store."
    )
    p.add_argument("--model", required=True, help="EDEN model path")
    p.add_argument(
        "--corpus",
        default=None,
        help="corpus dir (default: config.data_paths().corpus)",
    )
    p.add_argument("--out-root", required=True, help="activation store root dir")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--n-tokens", type=float, default=2e9)
    p.add_argument("--fwd-windows", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-seq", type=int, default=4096)
    args = p.parse_args(argv)

    corpus = Path(args.corpus) if args.corpus else config.data_paths().corpus
    eden = EdenPartialForward(
        args.model, layer=args.layer, device=args.device, max_seq=args.max_seq
    )
    reader = CorpusReader(corpus)
    summary = harvest_to_disk(
        eden,
        reader,
        out_root=args.out_root,
        dataset_name=args.dataset_name,
        n_tokens=int(args.n_tokens),
        fwd_windows=args.fwd_windows,
        dtype=args.dtype,
    )
    logger.info("harvest summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
