"""Assemble the bcr28 viewer association layer from the recomputed overlap metric.

Consumes the per-genome partials produced by the recompute pass:
  * cover partials (``cov::``/``nul::``/``his::``, OVERLAP rule) + a length-only
    overlap ``bg.npz``          -> span-level: precision_fold / cover_rate / null_fold / FDR
  * position partials (``obs::``) + per-annotation positions ``A``  -> position-level:
    pos_rate / lift (= PPV / prior)

Outputs the viewer association sidecars (not the feature JSONs, which are merged at
deploy time so the #3-finalized exemplars/structures are preserved):
  * ``span_rates.json`` = {feature: {ann: [precision_fold, cover_rate, bg_rate]}}
  * ``rates.json``      = {feature: {ann: [pos_rate, bg_rate]}}
  * ``detected_by_feature.json`` = {feature: [ full detected entry dicts ]}
  * ``index_rows.json`` = per-feature [feature_id, top_annotation, top_class, top_fold, ...]
  * ``assoc_summary.json`` = counts (n significant pairs, features grounded, per-layer)

Gates (retained): precision_fold >= 2, null_fold >= 1.5, BH-q <= 0.05, n_spans >= 20.
null_fold is the primary specificity guard under the overlap + GC-free background.
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

FOLD_CAP = 999.0
CLASS_OF = {"pfam": "pfam", "cath": "cath", "ted": "ted", "rfam": "rfam",
            "mge": "mge", "reg": "reg", "ec": "ec", "ko": "ko", "cog": "cog"}


def _ann_class(ann: str) -> str:
    return ann.split("|", 1)[0]


def load_positions(pos_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Sum obs::ann across genomes -> (obs[ann]=int64[F], A[ann]=positions)."""
    obs: dict[str, np.ndarray] = {}
    A: dict[str, int] = defaultdict(int)
    for f in sorted(glob.glob(str(Path(pos_dir) / "*.npz"))):
        z = np.load(f)
        for k in z.files:
            if k.startswith("obs::"):
                ann = k[5:]
                obs[ann] = z[k].astype(np.int64) if ann not in obs else obs[ann] + z[k]
        pj = f + ".pos.json"
        if os.path.exists(pj):
            for a, v in json.load(open(pj)).items():
                A[a] += int(v)
    return obs, dict(A)


def build(cover_dir, bg_npz, pos_dir, labels, fire_count, total_pos, prefixes,
          out_dir, min_spans=20):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    parts = [p for p in glob.glob(str(Path(cover_dir) / "*.npz")) if not p.endswith(".json")]
    obs, A = load_positions(pos_dir) if pos_dir else ({}, {})

    span_rates: dict[str, dict] = defaultdict(dict)
    rates: dict[str, dict] = defaultdict(dict)
    detected: dict[str, list] = defaultdict(list)
    per_layer = defaultdict(int)

    # Read the 152 partials ONCE (they are ~300 MB each); reduce every prefix from
    # the in-memory sums rather than re-globbing per prefix.
    print(f"summing {len(parts)} partials once...", flush=True)
    cover_all, null_all, hist_all, nsp_all = R.sum_partials(parts)
    bg_rate = R.load_background(bg_npz)
    for pfx in prefixes:
        keep = [a for a in cover_all if a.split("|", 1)[0] == pfx]
        sub_cover = {a: cover_all[a] for a in keep}
        sub_null = {a: null_all[a] for a in keep}
        sub_hist = {a: hist_all[a] for a in keep}
        sub_nsp = {a: nsp_all.get(a, 0) for a in keep}
        rows = R.enrichment_rows(sub_cover, sub_null, sub_hist, sub_nsp, bg_rate,
                                 labels=labels, min_spans=min_spans)
        for r in rows:
            ann, f = r["ann"], r["feature"]
            fs = str(f)
            cov_rate, pf = r["recall"], r["precision_fold"]
            bg = round(cov_rate / pf, 6) if pf > 0 else 0.0
            entry = {
                "id": ann, "class": _ann_class(ann), "pretty": r.get("label", "") or ann,
                "span_fold": pf, "cover_rate": cov_rate, "recall": cov_rate,
                "recall_lo": r["recall_lo"], "recall_hi": r["recall_hi"],
                "n_spans": r["n_spans"], "n_covered": r["n_covered"],
                "null_fold": r["null_fold"], "fdr": r["q"], "matched_bg_rate": bg,
            }
            # position-level lift
            if ann in obs and A.get(ann) and fire_count[f] > 0 and total_pos > 0:
                ob = float(obs[ann][f]); Aa = A[ann]
                pos_rate = (ob + 0.5) / Aa
                bg_rate = fire_count[f] / total_pos
                ppv = min(max(ob / fire_count[f], 0.0), 1.0)
                prior = Aa / total_pos
                entry["pos_rate"] = round(pos_rate, 9)
                entry["bg_rate"] = round(bg_rate, 9)
                entry["fold"] = round(min(pos_rate / bg_rate, FOLD_CAP), 3) if bg_rate > 0 else 0.0
                entry["ppv"] = round(ppv, 6); entry["prior"] = round(prior, 9)
                entry["lift"] = round(ppv / prior, 3) if prior > 0 else 0.0
                rates[fs][ann] = [entry["pos_rate"], entry["bg_rate"]]
            span_rates[fs][ann] = [pf, cov_rate, bg]
            detected[fs].append(entry)
            per_layer[pfx] += 1

    # sort each feature's detected by null_fold then recall (primary guard first)
    for fs in detected:
        detected[fs].sort(key=lambda e: (-(e["null_fold"] or 0), -(e["recall"] or 0)))

    # index rows: top annotation per feature
    index_rows = []
    for fs, ents in detected.items():
        top = ents[0]
        index_rows.append({"feature_id": int(fs), "top_annotation": top["pretty"],
                           "top_class": top["class"], "top_null_fold": top["null_fold"],
                           "top_recall": top["recall"], "top_fold": top.get("fold", 0.0)})

    json.dump(span_rates, open(out / "span_rates.json", "w"))
    json.dump(rates, open(out / "rates.json", "w"))
    json.dump(detected, open(out / "detected_by_feature.json", "w"))
    json.dump(index_rows, open(out / "index_rows.json", "w"))
    n_pairs = sum(len(v) for v in detected.values())
    summary = {"n_significant_pairs": n_pairs, "n_features_grounded": len(detected),
               "per_layer_pairs": dict(per_layer), "min_spans": min_spans,
               "gates": {"precision_fold>=": R.FOLD_MIN, "null_fold>=": R.NULL_FOLD_MIN, "q<=": R.Q_MAX}}
    json.dump(summary, open(out / "assoc_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cover-dir", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--pos-dir", default="")
    ap.add_argument("--labels", nargs="*", default=[], help="label json files (merged)")
    ap.add_argument("--index-rows", required=True, help="existing viewer index_rows.json (fire_count/density)")
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-spans", type=int, default=20)
    a = ap.parse_args(argv)

    labels: dict[str, str] = {}
    for lf in a.labels:
        if os.path.exists(lf):
            j = json.load(open(lf))
            # accept flat {ann:name} or nested {class:{id:name}}
            for k, v in j.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        labels[f"{k}|{kk}"] = vv
                else:
                    labels[k] = v

    rows = json.load(open(a.index_rows))
    fire_count = np.zeros(max(r["feature_id"] for r in rows) + 1, np.float64)
    tot_est = []
    for r in rows:
        fire_count[r["feature_id"]] = r.get("fire_count", 0) or 0
        if r.get("density"):
            tot_est.append(r["fire_count"] / r["density"])
    total_pos = float(np.median(tot_est)) if tot_est else 0.0
    print(f"total_positions ~ {total_pos:.4e}")

    build(a.cover_dir, a.bg, a.pos_dir or None, labels, fire_count, total_pos,
          a.prefixes, a.out, min_spans=a.min_spans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
