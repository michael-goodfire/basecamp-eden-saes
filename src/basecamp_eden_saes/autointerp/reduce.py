"""Reduce per-genome span-coverage partials into the enrichment table.

Sums the ``cov::``/``nul::``/``his::`` partials across genomes, applies the
matched-negative background, and emits one significant (annotation, feature) row
per pair that clears the support / fold / null / FDR gates:

    recall         = covered / n_spans
    precision_fold = covered / E_bg,   E_bg = sum_s hist[s] * bg_rate[s, feature]
    null_fold      = covered / null_covered      (circular-shift null)
    p              = Poisson upper tail of ``covered`` given mean ``E_bg``  -> q (BH)

``bg_rate[s, f] = bg_cover[s, f] / bg_count[s]`` is read from the per-dictionary
``bg.npz`` (35 length x GC strata). The reduce is deterministic; only the null-
derived columns depend on the circular-shift seed used at ``join_cover`` time.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M

MIN_SPANS = 20
FOLD_MIN = 2.0
NULL_FOLD_MIN = 1.5
Q_MAX = 0.05


def load_background(bg_npz: str | Path) -> np.ndarray:
    """Return ``bg_rate`` of shape ``(S, F)`` from a matched-negative ``bg.npz``."""
    bg = np.load(bg_npz)
    bg_count = bg["bg_count"].astype(np.float64)
    bg_cover = bg["bg_cover"].astype(np.float64)
    return bg_cover / np.maximum(bg_count[:, None], 1.0)


def panel_nspans(partials: list[str | Path]) -> dict[str, int]:
    """Cheaply sum per-ann span counts across partials (reads only the tiny
    ``.nspans.json`` sidecars, no arrays). Used to prefilter annotations to the
    support gate before the expensive array sum."""
    nsp: dict[str, int] = defaultdict(int)
    for f in partials:
        ns_path = str(f) + ".nspans.json"
        if Path(ns_path).exists():
            for a, v in json.load(open(ns_path)).items():
                nsp[a] += int(v)
    return dict(nsp)


def sum_partials(
    partials: list[str | Path], prefix: str = "", keep_anns: set | None = None
) -> tuple[dict, dict, dict, dict]:
    """Sum partial npz files. Returns ``(cover, nullc, hist, n_spans)`` dicts.

    ``keep_anns``: if given, only these annotation ids are accumulated (skips the
    array-add + decompression-materialize for all others) -- a large speedup when
    most annotations are rare (e.g. EC/KO singletons) and will gate out anyway.
    """
    cover: dict[str, np.ndarray] = {}
    nullc: dict[str, np.ndarray] = {}
    hist: dict[str, np.ndarray] = {}
    nsp: dict[str, int] = defaultdict(int)
    for f in partials:
        with np.load(f) as z:  # context-manage: close each NpzFile handle (avoids buffer accumulation over 152 partials)
          ns_path = str(f) + ".nspans.json"
          ns = json.load(open(ns_path)) if Path(ns_path).exists() else {}
          for k in z.files:
            if "::" not in k:
                continue
            kind, ann = k.split("::", 1)
            if prefix and not ann.startswith(prefix):
                continue
            if keep_anns is not None and ann not in keep_anns:
                continue
            if kind == "cov":
                cover[ann] = cover.get(ann, 0) + z[k]
            elif kind == "nul":
                nullc[ann] = nullc.get(ann, 0) + z[k]
            elif kind == "his":
                hist[ann] = hist.get(ann, 0) + z[k]
        for a, v in ns.items():
            if not prefix or a.startswith(prefix):
                nsp[a] += int(v)
    return cover, nullc, hist, dict(nsp)


def enrichment_rows(
    cover: dict,
    nullc: dict,
    hist: dict,
    nsp: dict,
    bg_rate: np.ndarray,
    labels: dict[str, str] | None = None,
    min_spans: int = MIN_SPANS,
    gated: bool = True,
) -> list[dict]:
    """Compute enrichment rows from summed partials.

    With ``gated=True`` only pairs clearing the support/fold/null/FDR gates are
    returned; with ``gated=False`` every fired pair is returned (used by the
    reproduction check, which targets a specific pair).
    """
    labels = labels or {}
    S = bg_rate.shape[0]
    rows: list[dict] = []
    for ann in sorted(cover):
        n = nsp.get(ann, int(hist[ann].sum()))
        if n < min_spans:
            continue
        cov = cover[ann].astype(np.float64)
        nul = nullc[ann].astype(np.float64)
        h = hist[ann].astype(np.float64)[:S]
        exp_bg = (h[:, None] * bg_rate).sum(axis=0)  # (F,) expected covered spans
        feats = np.where(cov > 0)[0]
        if feats.size == 0:
            continue
        # VECTORIZED over features (was a per-feature scipy.sf loop -> ~1000x fewer calls).
        from scipy.stats import poisson
        c_arr = cov[feats]
        e_arr = np.maximum(exp_bg[feats], 1e-9)
        nul_arr = nul[feats]
        k_arr = np.round(c_arr).astype(np.int64)
        pvals = poisson.sf(k_arr - 1, e_arr)
        pvals = np.where(k_arr <= 0, 1.0, pvals)
        q_arr = M.bh_qvalues(pvals)
        pf_arr = np.where(e_arr > 0, c_arr / e_arr, np.inf)
        nf_arr = np.where(nul_arr > 0, c_arr / nul_arr, c_arr / 0.5)
        recall_arr = c_arr / n
        if gated:
            mask = (q_arr <= Q_MAX) & (pf_arr >= FOLD_MIN) & (nf_arr >= NULL_FOLD_MIN)
        else:
            mask = np.ones(feats.size, dtype=bool)
        for i in np.where(mask)[0]:
            f = int(feats[i]); c = c_arr[i]; recall = recall_arr[i]
            pf = pf_arr[i]; nf = nf_arr[i]; p = float(pvals[i]); qv = float(q_arr[i])
            rlo, rhi = M.wilson_interval(c, n)
            rows.append({
                "ann": ann, "label": labels.get(ann, ""), "feature": f,
                "n_spans": int(n), "n_covered": int(c),
                "recall": round(recall, 4), "recall_lo": round(rlo, 4),
                "recall_hi": round(rhi, 4), "precision_fold": round(float(pf), 3),
                "recall_lift": round(float(nf), 3), "p": float(p), "q": float(qv),
            })
    rows.sort(key=lambda r: (r["ann"], -r["recall"]))
    return rows


def reduce_partials(
    partials: list[str | Path],
    bg_npz: str | Path,
    prefix: str = "",
    labels: dict[str, str] | None = None,
    min_spans: int = MIN_SPANS,
) -> list[dict]:
    """Convenience: sum partials and compute gated enrichment rows."""
    cover, nullc, hist, nsp = sum_partials(partials, prefix=prefix)
    bg_rate = load_background(bg_npz)
    return enrichment_rows(cover, nullc, hist, nsp, bg_rate, labels=labels, min_spans=min_spans)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reduce span-coverage partials into enrichment rows.")
    ap.add_argument("--partials", required=True, help="glob for *_partial.npz")
    ap.add_argument("--bg", required=True)
    ap.add_argument("--labels", default="", help="optional json {ann: label}")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--prefix", default="", help="only reduce anns with this key prefix")
    ap.add_argument("--min-spans", type=int, default=MIN_SPANS)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.labels)) if args.labels and Path(args.labels).exists() else {}
    files = sorted(glob.glob(args.partials))
    print(f"{len(files)} partial files")
    rows = reduce_partials(files, args.bg, prefix=args.prefix, labels=labels, min_spans=args.min_spans)
    json.dump(rows, open(out / "span_enrichment.json", "w"))
    print(f"wrote {len(rows)} significant rows -> {out / 'span_enrichment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
