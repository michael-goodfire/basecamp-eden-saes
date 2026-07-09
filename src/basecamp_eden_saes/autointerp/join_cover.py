"""Per-genome span-coverage of SAE features over annotation spans.

For one genome and one SAE dictionary, load the stored per-strand codes and, for
every annotation span, record which features cover it (fire in >= ``cover_frac``
of its positions), the same under ``n_null`` circular-shift null replicates, and
the span's length x GC stratum. A :func:`reduce.reduce_partials` step sums these
partials across genomes into the enrichment table.

Partial output (compressed npz) keys, per annotation ``a``:
    ``cov::a``  int32   [F]   covered-span count per feature
    ``nul::a``  float32 [F]   mean covered-span count per feature under the null
    ``his::a``  int64   [S]   span count per length x GC stratum
plus a ``<out>.nspans.json`` mapping ``{a: n_spans}``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M

Span = tuple[str, str, str, int, int]  # (ann_key, contig, strand, start, end)


def read_spans(paths: list[str | Path]) -> list[Span]:
    """Read span TSV files (``ann_key\\tcontig\\tstrand\\tstart\\tend``)."""
    spans: list[Span] = []
    for sp in paths:
        with open(sp) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                a, c, st, s, e = line.split("\t")
                spans.append((a, c, st, int(s), int(e)))
    return spans


def cover_genome(
    accession: str,
    codes_dir: str | Path,
    fasta_path: str | Path,
    bg_npz: str | Path,
    spans: list[Span],
    cover_frac: float = 0.5,
    n_null: int = 8,
    n_features: int = M.F_DEFAULT,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Compute cover/null/hist partials for one genome.

    Returns ``(arrays, n_spans)`` where ``arrays`` maps the ``cov::``/``nul::``/
    ``his::`` keys to numpy arrays and ``n_spans`` maps annotation -> span count.
    """
    bg = np.load(bg_npz)
    len_bins = bg["len_bins"]
    gc_bins = bg["gc_bins"]
    n_gc = int(bg["n_gc"]) if "n_gc" in bg.files else (len(gc_bins) + 1)
    n_strata = (len(len_bins) - 1) * n_gc

    seqs = M.read_contig_sequences(fasta_path)
    codes_dir = Path(codes_dir)

    cover: dict[str, np.ndarray] = {}
    nullc: dict[str, np.ndarray] = {}
    hist: dict[str, np.ndarray] = {}
    nsp: dict[str, int] = {}

    def ensure(a: str) -> None:
        if a not in cover:
            cover[a] = np.zeros(n_features, np.int32)
            nullc[a] = np.zeros(n_features, np.float32)
            hist[a] = np.zeros(n_strata, np.int64)
            nsp[a] = 0

    groups: dict[tuple[str, str], list[Span]] = defaultdict(list)
    for sp in spans:
        groups[(sp[1], sp[2])].append(sp)

    rng = np.random.default_rng(seed)
    for (contig, strand), grp in groups.items():
        fn = codes_dir / accession / f"{contig}.{'plus' if strand == '+' else 'minus'}.npz"
        if not fn.exists():
            continue
        indptr, indices, length_total = M.load_codes(fn)
        seq = seqs.get(contig)
        deltas = (
            [int(rng.integers(length_total // 10, length_total - length_total // 10))
             for _ in range(n_null)]
            if length_total > 40
            else []
        )
        inv_nn = 1.0 / max(len(deltas), 1)
        for a, _c, st, s, e in grp:
            if s < 0 or e > length_total or s >= e:
                continue
            ensure(a)
            length = e - s
            cs, ce = M.span_to_code_coords(s, e, st, length_total)
            fs = M.span_cover_features(indptr, indices, cs, ce, length_total, n_features, cover_frac)
            if fs.size:
                cover[a][fs] += 1
            gc = M.gc_fraction(seq, s, e) if seq is not None else 0.0
            hist[a][M.stratum_index(length, gc, len_bins, gc_bins, n_gc)] += 1
            nsp[a] += 1
            for d in deltas:
                ns = (cs + d) % length_total
                ne = ns + length
                fsn = M.span_cover_features(
                    indptr, indices, ns, ne if ne <= length_total else ne - length_total,
                    length_total, n_features, cover_frac,
                )
                if fsn.size:
                    nullc[a][fsn] += inv_nn

    arrays: dict[str, np.ndarray] = {}
    for a in cover:
        arrays[f"cov::{a}"] = cover[a]
        arrays[f"nul::{a}"] = nullc[a]
        arrays[f"his::{a}"] = hist[a]
    return arrays, nsp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-genome span-coverage partial over the code store.")
    ap.add_argument("--acc", required=True, help="genome accession (e.g. GCF_000005845.2)")
    ap.add_argument("--codes-dir", required=True, help="code-store dir for the dictionary")
    ap.add_argument("--fasta", required=True, help="genome FASTA (for per-span GC)")
    ap.add_argument("--bg", required=True, help="matched-negative bg.npz")
    ap.add_argument("--spans", nargs="+", required=True, help="span TSV file(s)")
    ap.add_argument("--out", required=True, help="output partial .npz path")
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--F", type=int, default=M.F_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    spans = read_spans(args.spans)
    if not spans:
        np.savez_compressed(args.out, _empty=np.array([0]))
        json.dump({}, open(args.out + ".nspans.json", "w"))
        print(f"{args.acc}: no spans")
        return 0
    arrays, nsp = cover_genome(
        args.acc, args.codes_dir, args.fasta, args.bg, spans,
        cover_frac=args.cover, n_null=args.n_null, n_features=args.F, seed=args.seed,
    )
    np.savez_compressed(args.out, **arrays)
    json.dump(nsp, open(args.out + ".nspans.json", "w"))
    print(f"{args.acc}: {len(spans)} spans, {len(nsp)} annotations -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
