"""Build the matched-negative background for span enrichment.

Length-only strata (no GC axis) with the OVERLAP cover rule: for random genomic
spans whose lengths match the real annotation length distribution, record the
fraction that each feature fires in (>=1 position). Consumed by ``reduce`` as
``bg_rate[l, f] = bg_cover[l, f] / bg_count[l]`` = P(feature f fires in a random
length-matched span of length-bin l).

Rationale for dropping the GC axis (researcher decision): a feature firing on
specific GC-rich *contexts* is not the same as firing on GC broadly, so
GC-matching the negatives wrongly subtracts real selectivity. The stored
``gc_bins`` is a single trivial bin so ``stratum_index`` collapses to the length
bin (``n_gc = 1``).

Per-genome ``build`` writes a partial (``bg_count``/``bg_cover`` over length bins);
``merge`` sums the partials into the final ``bg.npz``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .join_cover import read_spans

LEN_BINS = np.array([0, 30, 60, 120, 240, 480, 960, 1_000_000_000], dtype=np.float64)
GC_BINS = np.array([0.0, 1.01], dtype=np.float64)  # single trivial GC bin (n_gc = 1)


def _len_bin(length: float, len_bins: np.ndarray) -> int:
    return int(np.digitize(length, len_bins[1:-1]))


def build_genome(
    accession: str,
    codes_dir: str | Path,
    spans: list,
    n_neg: int = 20000,
    n_features: int = M.F_DEFAULT,
    len_bins: np.ndarray = LEN_BINS,
    seed: int = 0,
    thresh: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_neg`` length-matched random negatives; return (bg_count, bg_cover).

    If ``thresh`` (per-feature peak-firing threshold) is given, negatives use the
    same peak-firing cover rule as the real spans; otherwise the overlap rule.
    """
    codes_dir = Path(codes_dir)
    n_len = len(len_bins) - 1
    bg_count = np.zeros(n_len, np.int64)
    bg_cover = np.zeros((n_len, n_features), np.int64)

    # load all contig-strand code stores for this genome
    stores = []
    for fn in glob.glob(str(Path(codes_dir) / accession / "*.npz")):
        if thresh is not None:
            stores.append(M.load_codes_v(fn))
        else:
            ip, ix, L = M.load_codes(fn)
            stores.append((ip, ix, None, L))
    if not stores:
        return bg_count, bg_cover
    lengths_avail = np.array([s[-1] for s in stores], dtype=np.float64)
    store_p = lengths_avail / lengths_avail.sum()

    span_lengths = np.array([e - s for (_a, _c, _st, s, e) in spans if e > s], dtype=np.int64)
    if span_lengths.size == 0:
        return bg_count, bg_cover

    rng = np.random.default_rng(seed)
    draw_len = span_lengths[rng.integers(0, span_lengths.size, size=n_neg)]
    which = rng.choice(len(stores), size=n_neg, p=store_p)
    for i in range(n_neg):
        indptr, indices, values, L = stores[which[i]]
        length = int(draw_len[i])
        if length >= L or length <= 0:
            continue
        start = int(rng.integers(0, L - length))
        if thresh is not None:
            fs = M.span_peak_cover_features(indptr, indices, values, start, start + length, L, thresh)
        else:
            fs = M.span_cover_features(indptr, indices, start, start + length, L, n_features,
                                       cover_frac=0.0, overlap=True)
        lb = _len_bin(length, len_bins)
        bg_count[lb] += 1
        if fs.size:
            bg_cover[lb, fs] += 1
    return bg_count, bg_cover


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Matched-negative background (length-only, overlap).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="per-genome partial")
    b.add_argument("--acc", required=True)
    b.add_argument("--codes-dir", required=True)
    b.add_argument("--spans", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-neg", type=int, default=20000)
    b.add_argument("--F", type=int, default=M.F_DEFAULT)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--peak-file", default="")
    b.add_argument("--peak-frac", type=float, default=0.5)

    m = sub.add_parser("merge", help="sum partials into bg.npz")
    m.add_argument("--partials", required=True, help="glob for *_bg.npz partials")
    m.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "build":
        spans = read_spans(args.spans)
        thresh = None
        if args.peak_file:
            with np.load(args.peak_file) as _z:
                thresh = (args.peak_frac * _z["peak"]).astype(np.float32)
            thresh[thresh <= 0] = np.inf
        bc, bv = build_genome(args.acc, args.codes_dir, spans, n_neg=args.n_neg,
                               n_features=args.F, seed=args.seed, thresh=thresh)
        np.savez_compressed(args.out, bg_count=bc, bg_cover=bv)
        print(f"{args.acc}: {bc.sum()} negatives -> {args.out}")
        return 0

    # merge
    files = sorted(glob.glob(args.partials))
    bc = bv = None
    for f in files:
        z = np.load(f)
        bc = z["bg_count"].astype(np.int64) if bc is None else bc + z["bg_count"]
        bv = z["bg_cover"].astype(np.int64) if bv is None else bv + z["bg_cover"]
    n_len = len(LEN_BINS) - 1
    np.savez_compressed(args.out, bg_count=bc, bg_cover=bv, len_bins=LEN_BINS,
                        gc_bins=GC_BINS, n_len=np.array(n_len), n_gc=np.array(1))
    print(f"merged {len(files)} partials -> {args.out}; per-bin negatives: {bc.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
