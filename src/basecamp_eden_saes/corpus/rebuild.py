"""Rebuild a corpus's tokens.bin from its intact provenance (offsets + db).

Used to restore a token store whose ``offsets.npy`` / ``sequences.db`` /
``meta.json`` survived but whose ``tokens.bin`` was lost. Each window's tokens
are regenerated as ``uppercase(contig)[contig_offset : contig_offset+length]``
(byte-level ids) and written at the absolute position recorded in
``offsets.npy``, so the rebuilt store aligns exactly with the existing index.

Only ``tokens.bin`` is written; ``offsets.npy`` / ``sequences.db`` /
``meta.json`` are read-only inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from . import build

logger = logging.getLogger(__name__)

# Valid uppercase nucleotide alphabet for verification. The OG2 corpus keeps N
# and IUPAC ambiguity codes (N is in-distribution for the OG2 model, so its
# builder did not split at N), so verification accepts the full IUPAC set, not
# just A/C/G/T. The stored bases are uppercase (tantan masking is used only for
# the drop decision, not to lowercase survivors).
_VALID_NUC = np.zeros(256, dtype=bool)
for _ch in "ACGTUNRYSWKMBDHV":
    _VALID_NUC[ord(_ch)] = True


def _bare_gtdb(cid: str) -> str:
    # gtdb stream yields "<genome>.fasta.gz:<header>"; db stores the bare header
    return cid.rsplit(":", 1)[-1]


def _identity(cid: str) -> str:
    return cid


def _fill(
    mm: np.ndarray,
    offsets: np.ndarray,
    index: dict[str, list],
    stream: Iterator[tuple[str, str, str]],
    bare: Callable[[str], str],
) -> int:
    remaining = set(index)
    written = 0
    for _source, cid_full, seq in stream:
        cid = bare(cid_full)
        ws = index.get(cid)
        if not ws:
            continue
        arr = np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)
        for wid, coff, length in ws:
            seg = arr[coff : coff + length]
            if len(seg) != length:
                raise RuntimeError(
                    f"contig {cid}: window {wid} wants [{coff}:{coff + length}] "
                    f"but contig is only {len(arr)} long"
                )
            start = int(offsets[wid])
            mm[start : start + length] = seg
            written += 1
        remaining.discard(cid)
        if not remaining:
            break
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} contigs from the db were not found in the stream, "
            f"e.g. {list(remaining)[:3]}"
        )
    return written


def rebuild_tokens(
    full_dir: str | Path,
    gtdb_files: list[str],
    metag_files: list[str],
) -> dict:
    """Regenerate ``tokens.bin`` in ``full_dir`` and return a verification dict."""
    full = Path(full_dir)
    offsets = np.load(full / "offsets.npy")
    total = int(offsets[-1])
    meta = json.loads((full / "meta.json").read_text())
    if total != meta["total_tokens"]:
        raise RuntimeError(
            f"offsets end {total} != meta total_tokens {meta['total_tokens']}"
        )

    db = sqlite3.connect(full / "sequences.db")
    gtdb_idx: dict[str, list] = {}
    metag_idx: dict[str, list] = {}
    for wid, src, _ftag, cid, coff, length in db.execute(
        "SELECT window_id, source, file_tag, contig_id, contig_offset, length FROM windows"
    ):
        (gtdb_idx if src == "gtdb" else metag_idx).setdefault(cid, []).append(
            (int(wid), int(coff), int(length))
        )
    n_windows = sum(len(v) for v in gtdb_idx.values()) + sum(
        len(v) for v in metag_idx.values()
    )

    # w+ creates/overwrites tokens.bin at exactly the recorded total size.
    mm = np.memmap(full / "tokens.bin", dtype=np.uint8, mode="w+", shape=(total,))
    written = 0
    written += _fill(
        mm, offsets, gtdb_idx, build.stream_source("gtdb", gtdb_files), _bare_gtdb
    )
    logger.info("[rebuild] gtdb windows written: %d", written)
    written += _fill(
        mm, offsets, metag_idx, build.stream_source("metagenome", metag_files), _identity
    )
    logger.info("[rebuild] total windows written: %d", written)
    mm.flush()

    if written != n_windows:
        raise RuntimeError(f"wrote {written} windows, expected {n_windows}")

    # --- verification ---
    # 1. size matches total_tokens
    size = (full / "tokens.bin").stat().st_size
    assert size == total, (size, total)
    # 2. every byte is a valid uppercase nucleotide code (catches gaps / wrong
    #    offsets / garbage); the OG2 corpus legitimately contains N + IUPAC codes
    bad = 0
    chunk = 1 << 28
    for s in range(0, total, chunk):
        bad += int((~_VALID_NUC[np.asarray(mm[s : s + chunk])]).sum())
    # 3. spot-check sampled windows decode to their recorded contigs
    rng = np.random.default_rng(0)
    sample_wids = rng.choice(n_windows, size=min(200, n_windows), replace=False)
    spot = []
    for wid in sample_wids:
        cid, coff, length = db.execute(
            "SELECT contig_id, contig_offset, length FROM windows WHERE window_id=?",
            (int(wid),),
        ).fetchone()
        start = int(offsets[int(wid)])
        toks = np.asarray(mm[start : start + length])
        spot.append(bool(_VALID_NUC[toks].all()) and len(toks) == length)

    result = {
        "total_tokens": total,
        "n_windows": n_windows,
        "windows_written": written,
        "bytes_on_disk": size,
        "invalid_nucleotide_bytes": bad,
        "spot_checked": len(spot),
        "spot_all_valid": all(spot),
    }
    logger.info("[rebuild] verification: %s", json.dumps(result, indent=2))
    if bad != 0 or not all(spot):
        raise RuntimeError(f"verification failed: {result}")
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for rebuilding ``tokens.bin`` from provenance."""
    p = argparse.ArgumentParser(description="Rebuild tokens.bin from provenance.")
    p.add_argument("--full-dir", required=True)
    p.add_argument(
        "--gtdb-files", nargs="*", default=list(build.DEFAULT_GTDB_FILES)
    )
    p.add_argument(
        "--metag-files", nargs="*", default=list(build.DEFAULT_METAG_FILES)
    )
    args = p.parse_args(argv)
    rebuild_tokens(args.full_dir, args.gtdb_files, args.metag_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
