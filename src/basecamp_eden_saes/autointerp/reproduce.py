"""Reproduction check: re-derive the pinned viewer metrics from bundle + codes.

Recomputes, directly from the annotation bundle span layers and the full-sparse
code store (no re-inference, no re-training):

- ``recall``       = covered_spans / n_spans           (both target features)
- ``span_fold``    = covered_spans / E_bg              (matched-negative fold)
- ``lift``         = PPV / prior = (obs/fire_count) / (A/total)   (position-level)

and compares them to the values recorded in the deployed feature atlas (source
experiments #53 / #50 / #40). ``recall`` and ``span_fold`` need only the target
annotation's spans (a few contigs per genome); ``lift`` additionally needs each
lift feature's panel-wide firing total, so those genomes are scanned in full.

Run as a SLURM array (one ``partial`` task per genome accession) then ``reduce``:

    bes-reproduce partial --acc GCF_000005845.2 --out <dir>
    bes-reproduce reduce --partials '<dir>/*.npz' --out reproduction.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .. import config
from . import metrics as M

# Pinned reference values, read from the source-thread artifacts (#53/#50/#40).
DICT_NAME = "bcr_k64"
COVER_FRAC = 0.5
TOLERANCE = 0.02  # +/- 2%

RECALL_SPANFOLD_TARGETS = [
    {"feature": 2044, "ann": "cath|3.40.1090.10", "layer": "cath",
     "recorded": {"recall": 0.2434, "span_fold": 702.956}},
    {"feature": 27629, "ann": "ted|1.10.760.10", "layer": "ted",
     "recorded": {"recall": 0.752, "span_fold": 436.269}},
]
# lift = (obs/fire_count)/(A/total). obs, fire_count and total are measured from
# the code store; A (annotation positions) from the bundle spans.
LIFT_TARGETS = [
    {"feature": 2044, "ann": "cath|3.40.1090.10", "layer": "cath",
     "recorded": {"lift": 244.3}},
]


def _ann_spans_for_genome(paths: config.DataPaths, layer: str, acc: str, ann: str):
    """Spans for one annotation in one genome: list of (contig, strand, s, e)."""
    fp = paths.bundle_spans(layer, acc)
    out = []
    if not fp.exists():
        return out
    with open(fp) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            a, c, st, s, e = line.split("\t")
            if a == ann:
                out.append((c, st, int(s), int(e)))
    return out


def compute_partial(acc: str, paths: config.DataPaths | None = None) -> dict:
    """Per-genome contributions for every target, from the code store + bundle."""
    paths = paths or config.data_paths()
    code_dir = paths.code_dir(DICT_NAME)
    bg = np.load(paths.bg_npz(DICT_NAME))
    len_bins, gc_bins = bg["len_bins"], bg["gc_bins"]
    n_gc = int(bg["n_gc"])
    n_strata = (len(len_bins) - 1) * n_gc
    n_features = M.F_DEFAULT

    try:
        seqs = M.read_contig_sequences(paths.genome_fasta(acc))
    except FileNotFoundError:
        seqs = {}

    out: dict[str, np.ndarray] = {}

    # --- recall + span_fold targets: cover + hist over the target ann's spans ---
    for t in RECALL_SPANFOLD_TARGETS:
        ann, feat = t["ann"], t["feature"]
        cover = 0
        hist = np.zeros(n_strata, np.int64)
        n_spans = 0
        for contig, strand, s, e in _ann_spans_for_genome(paths, t["layer"], acc, ann):
            fn = code_dir / acc / f"{contig}.{'plus' if strand == '+' else 'minus'}.npz"
            if not fn.exists():
                continue
            indptr, indices, L = M.load_codes(fn)
            if s < 0 or e > L or s >= e:
                continue
            cs, ce = M.span_to_code_coords(s, e, strand, L)
            fs = M.span_cover_features(indptr, indices, cs, ce, L, n_features, COVER_FRAC)
            if feat in fs:
                cover += 1
            gc = M.gc_fraction(seqs[contig], s, e) if contig in seqs else 0.0
            hist[M.stratum_index(e - s, gc, len_bins, gc_bins, n_gc)] += 1
            n_spans += 1
        key = f"{ann}#{feat}"
        out[f"cover::{key}"] = np.array([cover], np.int64)
        out[f"nspans::{key}"] = np.array([n_spans], np.int64)
        out[f"hist::{key}"] = hist

    # --- lift targets: obs (in-ann firings), fire_count (panel-wide), total ---
    lift_feats = sorted({t["feature"] for t in LIFT_TARGETS})
    # obs per (ann,feature)
    for t in LIFT_TARGETS:
        ann, feat = t["ann"], t["feature"]
        obs = 0
        a_positions = 0
        for contig, strand, s, e in _ann_spans_for_genome(paths, t["layer"], acc, ann):
            fn = code_dir / acc / f"{contig}.{'plus' if strand == '+' else 'minus'}.npz"
            if not fn.exists():
                continue
            indptr, indices, L = M.load_codes(fn)
            if s < 0 or e > L or s >= e:
                continue
            cs, ce = M.span_to_code_coords(s, e, strand, L)
            obs += M.span_firing_positions(indptr, indices, cs, ce, feat)
            a_positions += e - s
        out[f"obs::{ann}#{feat}"] = np.array([obs], np.int64)
        out[f"apos::{ann}#{feat}"] = np.array([a_positions], np.int64)

    # fire_count (per lift feature) + total positions: scan every contig-strand.
    genome_dir = code_dir / acc
    fire = {f: 0 for f in lift_feats}
    total_pos = 0
    if genome_dir.is_dir():
        for fn in sorted(genome_dir.glob("*.npz")):
            z = np.load(fn)
            idx = z["indices"]
            total_pos += int(z["indptr"].shape[0] - 1)
            for f in lift_feats:
                fire[f] += int(np.count_nonzero(idx == f))
    out["total_pos"] = np.array([total_pos], np.int64)
    for f in lift_feats:
        out[f"fire::{f}"] = np.array([fire[f]], np.int64)
    return out


def reduce_and_compare(partial_files: list[str], paths: config.DataPaths | None = None) -> dict:
    """Sum per-genome partials, compute metrics, compare to recorded values."""
    paths = paths or config.data_paths()
    bg_rate = None
    bg = np.load(paths.bg_npz(DICT_NAME))
    bg_count = bg["bg_count"].astype(np.float64)
    bg_cover = bg["bg_cover"].astype(np.float64)
    bg_rate = bg_cover / np.maximum(bg_count[:, None], 1.0)
    S = bg_rate.shape[0]

    acc = {}
    for f in partial_files:
        z = np.load(f)
        for k in z.files:
            arr = z[k]
            if k in acc:
                acc[k] = acc[k] + arr
            else:
                acc[k] = arr.copy()

    rows = []

    def check(name, feature, ann, recorded, reproduced):
        within = abs(reproduced - recorded) <= TOLERANCE * abs(recorded)
        rows.append({
            "name": name, "feature": feature, "ann": ann,
            "recorded": round(float(recorded), 4), "reproduced": round(float(reproduced), 4),
            "tolerance": f"+/-{int(TOLERANCE*100)}%", "within": bool(within),
        })

    for t in RECALL_SPANFOLD_TARGETS:
        key = f"{t['ann']}#{t['feature']}"
        cover = int(acc[f"cover::{key}"][0])
        n_spans = int(acc[f"nspans::{key}"][0])
        hist = acc[f"hist::{key}"].astype(np.float64)[:S]
        recall = cover / n_spans if n_spans else 0.0
        exp_bg = float((hist[:, None] * bg_rate[:, t["feature"]:t["feature"]+1]).sum())
        span_fold = cover / exp_bg if exp_bg > 0 else float("inf")
        check("recall", t["feature"], t["ann"], t["recorded"]["recall"], recall)
        check("span_fold", t["feature"], t["ann"], t["recorded"]["span_fold"], span_fold)

    total = int(acc["total_pos"][0])
    for t in LIFT_TARGETS:
        key = f"{t['ann']}#{t['feature']}"
        obs = int(acc[f"obs::{key}"][0])
        a_pos = int(acc[f"apos::{key}"][0])
        fire = int(acc[f"fire::{t['feature']}"][0])
        ppv = obs / fire if fire else 0.0
        prior = a_pos / total if total else 0.0
        lift = ppv / prior if prior > 0 else float("inf")
        check("lift", t["feature"], t["ann"], t["recorded"]["lift"], lift)

    all_within = all(r["within"] for r in rows)
    return {
        "dict": DICT_NAME, "cover_frac": COVER_FRAC, "tolerance": TOLERANCE,
        "n_genomes": len(partial_files), "all_within": all_within, "metrics": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Atlas-metrics reproduction check.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("partial", help="compute per-genome partial")
    p.add_argument("--acc", required=True)
    p.add_argument("--out", required=True, help="output dir for the partial npz")
    r = sub.add_parser("reduce", help="reduce partials and compare to recorded")
    r.add_argument("--partials", required=True, help="glob for partial npz files")
    r.add_argument("--out", required=True, help="reproduction.json output path")
    args = ap.parse_args(argv)

    if args.cmd == "partial":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        arrays = compute_partial(args.acc)
        outfp = Path(args.out) / f"{args.acc}.npz"
        np.savez_compressed(outfp, **arrays)
        print(f"{args.acc}: partial -> {outfp}")
        return 0

    files = sorted(glob.glob(args.partials))
    result = reduce_and_compare(files)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["all_within"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
