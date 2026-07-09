"""Covered-universe matched-negative background for TED enrichment metrics.

Two stages, exposed as the ``build`` and ``reduce`` subcommands:

``build`` (one genome, one dict): samples background spans ONLY from the CDS of
proteins that carry a TED structure (the "covered universe"), so TED enrichment
measures the domain, not "is a mappable protein". For each sampled span it
computes per-feature cover from the stored codes (same >=50%-of-positions rule,
reverse-strand aware) and its length x GC stratum (reusing the panel's bin
edges), writing a partial ``bg_count`` ``(S,)`` and ``bg_cover`` ``(S, F)``.

``reduce``: sums the per-genome partials into a single ``bg.npz`` in the
matched-negative-background format consumed by ``autointerp.reduce``.

The span/GC/stratum primitives are imported from
:mod:`basecamp_eden_saes.autointerp.metrics`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from basecamp_eden_saes.autointerp.metrics import (
    F_DEFAULT,
    gc_fraction,
    load_codes,
    read_contig_sequences,
    span_cover_features,
    stratum_index,
)


def build_main(argv: list[str] | None = None) -> int:
    """Sample the covered-universe background for one genome + dict."""
    ap = argparse.ArgumentParser(description=build_main.__doc__)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--codes-dir", required=True)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--cds-map", required=True)
    ap.add_argument(
        "--ted-tsv",
        required=True,
        help="this genome's TED domains.tsv; its proteins define the covered universe",
    )
    ap.add_argument("--bg", required=True, help="reference bg.npz for the len/gc bin edges")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--F", type=int, default=F_DEFAULT)
    args = ap.parse_args(argv)

    bg = np.load(args.bg)
    len_bins, gc_bins = bg["len_bins"], bg["gc_bins"]
    n_gc = int(bg["n_gc"]) if "n_gc" in bg else (len(gc_bins) + 1)
    S = (len(len_bins) - 1) * n_gc
    cds = json.load(open(args.cds_map))
    F = args.F

    # covered universe = proteins with a TED structure (from this genome's TED layer)
    ted_prots: set[str] = set()
    if os.path.exists(args.ted_tsv):
        with open(args.ted_tsv) as fh:
            fh.readline()
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 8:
                    ted_prots.add(f[7])  # protein_id column
    prots = []
    for contig, pd in cds.items():
        for pid, info in pd.items():
            if pid in ted_prots:
                prots.append((contig, info))

    bg_count = np.zeros(S, np.int64)
    bg_cover = np.zeros((S, F), np.int64)
    if not prots:
        np.savez_compressed(args.out, bg_count=bg_count, bg_cover=bg_cover)
        print(f"{args.acc}: no mapped proteins", flush=True)
        return 0

    seqs = read_contig_sequences(os.path.join(args.fasta_dir, args.acc, "genomic.fna"))
    rng = np.random.default_rng(args.seed)
    # length pool spanning the len bins so all strata get populated
    lo, hi = 30, int(len_bins[-2]) if len_bins[-1] > 1e6 else int(len_bins[-1])
    hi = max(hi, 400)

    # cache codes per (contig, strand)
    codes: dict[tuple[str, str], tuple | None] = {}

    def get_codes(contig: str, strand: str):
        key = (contig, strand)
        if key not in codes:
            fn = os.path.join(
                args.codes_dir,
                args.acc,
                f"{contig}.{'plus' if strand == '+' else 'minus'}.npz",
            )
            codes[key] = load_codes(fn) if os.path.exists(fn) else None
        return codes[key]

    made = 0
    tries = 0
    while made < args.n and tries < args.n * 5:
        tries += 1
        contig, info = prots[rng.integers(len(prots))]
        strand = info["strand"]
        segs = info["segs"]
        cds_len = sum(b - a for a, b in segs)
        if cds_len < 60:
            continue
        L_nt = int(rng.integers(lo, min(hi, cds_len) + 1))
        # random offset within the CDS (transcription order), map to a contig window
        off = int(rng.integers(0, cds_len - L_nt + 1))
        ordered = segs if strand == "+" else list(reversed(segs))

        def tx(o: int) -> int:
            rem = o
            for a, b in ordered:
                ln = b - a
                if rem <= ln:
                    return (a + rem) if strand == "+" else (b - rem)
                rem -= ln
            a, b = ordered[-1]
            return b if strand == "+" else a

        c1, c2 = tx(off), tx(off + L_nt)
        s, e = min(c1, c2), max(c1, c2)
        cc = get_codes(contig, strand)
        if cc is None:
            continue
        indptr, indices, Lc = cc
        if strand == "-":
            cs, ce = Lc - e, Lc - s
        else:
            cs, ce = s, e
        if cs < 0 or ce > Lc or cs >= ce:
            continue
        fs = span_cover_features(indptr, indices, cs, ce, Lc, F, args.cover)
        gc = gc_fraction(seqs.get(contig), s, e) if seqs.get(contig) is not None else 0.0
        st = stratum_index(e - s, gc, len_bins, gc_bins, n_gc)
        if st >= S:
            continue
        bg_count[st] += 1
        if fs.size:
            bg_cover[st, fs] += 1
        made += 1

    np.savez_compressed(args.out, bg_count=bg_count, bg_cover=bg_cover)
    print(f"{args.acc}: {made} bg spans over {len(prots)} mapped proteins", flush=True)
    return 0


def reduce_main(argv: list[str] | None = None) -> int:
    """Sum per-genome TED background partials into a single bg.npz."""
    ap = argparse.ArgumentParser(description=reduce_main.__doc__)
    ap.add_argument("--partials", required=True, help="glob of build partials")
    ap.add_argument("--ref-bg", required=True, help="reference bg.npz for bin edges")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ref = np.load(args.ref_bg)
    bc = None
    bcov = None
    n = 0
    for f in sorted(glob.glob(args.partials)):
        z = np.load(f)
        if bc is None:
            bc = z["bg_count"].astype(np.int64)
            bcov = z["bg_cover"].astype(np.int64)
        else:
            bc += z["bg_count"]
            bcov += z["bg_cover"]
        n += 1
    np.savez_compressed(
        args.out,
        bg_count=bc,
        bg_cover=bcov,
        len_bins=ref["len_bins"],
        gc_bins=ref["gc_bins"],
        n_len=ref["n_len"],
        n_gc=ref["n_gc"],
    )
    total = int(bc.sum()) if bc is not None else 0
    strata = bc.shape[0] if bc is not None else 0
    print(f"reduced {n} bg partials; total bg spans {total}, strata {strata}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the ``build`` or ``reduce`` subcommand."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", add_help=False)
    sub.add_parser("reduce", add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "build":
        return build_main(rest)
    if args.cmd == "reduce":
        return reduce_main(rest)
    ap.error(f"unknown subcommand {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
