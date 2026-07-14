"""Assemble the bcr28 viewer association layer from the recomputed overlap metric.

Consumes the per-genome cover partials (``cov::``/``nul::``/``his::``, OVERLAP
rule) + a length-only overlap ``bg.npz`` and produces the viewer association
sidecars. The metric per (feature f, annotation a):

  recall        = fraction of a's spans f fires in (overlap)
  precision_fold= recall / matched-negative length-only bg rate  (span_fold)
  recall_lift   = recall / recall-on-circularly-shifted-spans    (was 'null_fold':
                  how much more f fires on the real spans than at random)
  q             = BH-FDR of the Poisson upper tail of covered vs expected-bg

Rather than keep every significant pair (~90M under permissive overlap -> 99.9%
of features grounded, uninterpretable), we keep the **top-K associations per
feature ranked by recall_lift**, storing each entry's full metrics so a
significance threshold can be chosen downstream without recomputing. A light
validity pre-filter (precision_fold >= FOLD_MIN, recall_lift >= LIFT_MIN,
q <= Q_MAX) removes depleted/insignificant pairs; it never removes a pair that
would make a feature's top-K, and thresholds can only be *raised* later.

The summed arrays are cached (``summed_arrays.npz``) so reduce reruns (K / gate
tweaks) skip the expensive 152-partial sum.

Outputs: ``span_rates.json`` {feat:{ann:[precision_fold,cover_rate,bg_rate]}},
``detected_by_feature.json`` {feat:[entry...]}, ``index_rows.json`` (top per
feature), ``assoc_summary.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..autointerp import metrics as M
from ..autointerp import reduce as R

FOLD_MIN = 2.0     # precision_fold light pre-filter
LIFT_MIN = 1.5     # recall_lift light pre-filter
TOPK = 50


def _ann_class(ann: str) -> str:
    return ann.split("|", 1)[0]


def load_cached_or_sum(parts, out, min_spans):
    cache = out / "summed_arrays.npz"
    cache_nsp = out / "summed_nspans.json"
    if cache.exists() and cache_nsp.exists():
        print(f"loading cached sum from {cache}", flush=True)
        with np.load(cache) as z:
            cover = {k[5:]: z[k] for k in z.files if k.startswith("cov::")}
            nullc = {k[5:]: z[k] for k in z.files if k.startswith("nul::")}
            hist = {k[5:]: z[k] for k in z.files if k.startswith("his::")}
        nsp = json.load(open(cache_nsp))
        return cover, nullc, hist, nsp
    print("prefiltering annotations by panel n_spans...", flush=True)
    panel_ns = R.panel_nspans(parts)
    keep = {a for a, n in panel_ns.items() if n >= min_spans}
    print(f"keep {len(keep)}/{len(panel_ns)} annotations >= {min_spans} spans; summing {len(parts)} partials...", flush=True)
    cover, nullc, hist, nsp = R.sum_partials(parts, keep_anns=keep)
    arrs = {}
    for a in cover:
        arrs[f"cov::{a}"] = cover[a]; arrs[f"nul::{a}"] = nullc[a]; arrs[f"his::{a}"] = hist[a]
    np.savez(str(cache), **arrs)
    json.dump(nsp, open(cache_nsp, "w"))
    print(f"cached sum -> {cache}", flush=True)
    return cover, nullc, hist, nsp


def build(cover_dir, bg_npz, labels, prefixes, out_dir, min_spans=20, topk=TOPK):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    parts = [p for p in glob.glob(str(Path(cover_dir) / "*.npz")) if not p.endswith(".json")]
    cover_all, null_all, hist_all, nsp_all = load_cached_or_sum(parts, out, min_spans)
    bg_rate = R.load_background(bg_npz)
    from scipy.stats import poisson
    S = bg_rate.shape[0]

    want = set(prefixes)
    anns = [a for a in cover_all if _ann_class(a) in want and nsp_all.get(a, 0) >= min_spans]
    print(f"reducing {len(anns)} annotations (top-{topk} per feature by recall_lift)...", flush=True)

    # Collect gated survivors as flat arrays (no per-pair dicts) -> per-feature top-K at the end.
    F_feat, F_annidx, F_rec, F_pf, F_rl, F_q, F_ncov = ([] for _ in range(7))
    ann_list = []
    per_layer = defaultdict(int)
    for ai, ann in enumerate(anns):
        ann_list.append(ann)
        n = nsp_all.get(ann, int(hist_all[ann].sum()))
        cov = cover_all[ann].astype(np.float64)
        nul = null_all[ann].astype(np.float64)
        h = hist_all[ann].astype(np.float64)[:S]
        exp_bg = (h[:, None] * bg_rate).sum(axis=0)
        feats = np.where(cov > 0)[0]
        if feats.size == 0:
            continue
        c = cov[feats]
        e = np.maximum(exp_bg[feats], 1e-9)
        nu = nul[feats]
        pf = np.where(e > 0, c / e, np.inf)
        rl = np.where(nu > 0, c / nu, c / 0.5)
        k = np.round(c).astype(np.int64)
        pv = poisson.sf(k - 1, e)
        pv = np.where(k <= 0, 1.0, pv)
        q = M.bh_qvalues(pv)
        m = (pf >= FOLD_MIN) & (rl >= LIFT_MIN) & (q <= R.Q_MAX)
        if not m.any():
            continue
        idx = np.where(m)[0]
        per_layer[_ann_class(ann)] += int(idx.size)
        F_feat.append(feats[idx].astype(np.int32))
        F_annidx.append(np.full(idx.size, ai, np.int32))
        F_rec.append((c[idx] / n).astype(np.float32))
        F_pf.append(pf[idx].astype(np.float32))
        F_rl.append(rl[idx].astype(np.float32))
        F_q.append(q[idx].astype(np.float64))
        F_ncov.append(c[idx].astype(np.int32))
    if not F_feat:
        print("no survivors"); return {}

    feat = np.concatenate(F_feat); annidx = np.concatenate(F_annidx)
    rec = np.concatenate(F_rec); pf = np.concatenate(F_pf); rl = np.concatenate(F_rl)
    q = np.concatenate(F_q); ncov = np.concatenate(F_ncov)
    total_pairs = feat.size
    print(f"{total_pairs} light-significant pairs; taking top-{topk} per feature by precision_fold...", flush=True)

    # per-feature top-K by precision_fold: sort by (feature, -precision_fold), keep first K per feature
    order = np.lexsort((-rl, feat))  # per-feature top-K by recall_lift (primary metric)
    feat_s = feat[order]
    # rank within each feature group
    grp_start = np.concatenate(([0], np.where(np.diff(feat_s) != 0)[0] + 1))
    keep_mask = np.zeros(feat.size, bool)
    for gi in range(grp_start.size):
        s = grp_start[gi]
        e_ = grp_start[gi + 1] if gi + 1 < grp_start.size else feat.size
        keep_mask[order[s:min(e_, s + topk)]] = True

    span_rates: dict[str, dict] = defaultdict(dict)
    detected: dict[str, list] = defaultdict(list)
    kept_idx = np.where(keep_mask)[0]
    for j in kept_idx:
        f = int(feat[j]); fs = str(f); ann = ann_list[int(annidx[j])]
        cov_rate = float(rec[j]); pfj = float(pf[j])
        bg = round(cov_rate / pfj, 6) if pfj > 0 else 0.0
        rlo, rhi = M.wilson_interval(int(ncov[j]), nsp_all.get(ann, 0))
        rlv = round(float(rl[j]), 3)
        entry = {
            "id": ann, "class": _ann_class(ann), "pretty": labels.get(ann, "") or ann,
            # primary metric first, then secondary; null_fold kept as an alias of
            # recall_lift for bcr_k64 viewer-JS schema compatibility.
            "recall_lift": rlv, "precision_fold": round(pfj, 3), "recall": round(cov_rate, 4),
            "span_fold": round(pfj, 3), "null_fold": rlv, "cover_rate": round(cov_rate, 4),
            "recall_lo": round(rlo, 4), "recall_hi": round(rhi, 4),
            "n_spans": int(nsp_all.get(ann, 0)), "n_covered": int(ncov[j]),
            "fdr": float(q[j]), "matched_bg_rate": bg,
        }
        detected[fs].append(entry)
        span_rates[fs][ann] = [round(pfj, 3), round(cov_rate, 4), bg]
    for fs in detected:
        detected[fs].sort(key=lambda e: -(e["recall_lift"] or 0))

    index_rows = []
    for fs, ents in detected.items():
        top = ents[0]
        index_rows.append({"feature_id": int(fs), "top_annotation": top["pretty"],
                           "top_class": top["class"], "top_recall_lift": top["recall_lift"],
                           "top_recall": top["recall"], "top_precision_fold": top["precision_fold"]})

    json.dump(span_rates, open(out / "span_rates.json", "w"))
    json.dump(detected, open(out / "detected_by_feature.json", "w"))
    json.dump(index_rows, open(out / "index_rows.json", "w"))
    summary = {
        "topk_per_feature": topk, "features_with_associations": len(detected),
        "kept_pairs": int(kept_idx.size), "light_significant_pairs_total": int(total_pairs),
        "per_layer_light_pairs": dict(per_layer),
        "light_prefilter": {"precision_fold>=": FOLD_MIN, "recall_lift>=": LIFT_MIN, "q<=": R.Q_MAX},
        "ranked_by": "recall_lift",
        "note": "top-K per feature by recall_lift (primary); precision_fold + recall stored per entry; raise thresholds downstream.",
    }
    json.dump(summary, open(out / "assoc_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cover-dir", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-spans", type=int, default=20)
    ap.add_argument("--topk", type=int, default=TOPK)
    a = ap.parse_args(argv)

    labels: dict[str, str] = {}
    for lf in a.labels:
        if os.path.exists(lf):
            j = json.load(open(lf))
            for k, v in j.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        labels[f"{k}|{kk}"] = vv
                else:
                    labels[k] = v
    build(a.cover_dir, a.bg, labels, a.prefixes, a.out, min_spans=a.min_spans, topk=a.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
