"""Embed BOTH enrichment metrics + their rate components into every detected /
top_annotations entry of the canonical feature JSONs (in place).

Adds per entry, alongside the existing position-level `fold` (Haldane
circular-shift, capped at 999):
  span_fold        span-level precision_fold = cover_rate / matched_bg_rate (the gate)
  cover_rate       fraction of the fold's spans the feature covers (= recall)
  matched_bg_rate  expected covered fraction under the matched-negative background
  pos_rate         in-fold per-position firing rate (position fold numerator)
  bg_rate          background per-position firing rate (position fold denominator)

pos_rate/bg_rate come from rates.json; span_fold/cover_rate/matched_bg_rate from
span_rates.json (ted/cath from #40, pfam/genomic from #38). Exposing pos_rate/
bg_rate reveals the true ratio behind a capped `fold: 999.0`. Written in place
with open(path,"w") so the served hardlink (same inode) reflects the change.
"""
from __future__ import annotations

import argparse
import json
import os

# Canonical on-cluster defaults (overridable via CLI).
DEFAULT_CANON = "/mnt/data/artifacts/silico/eden_feature_viewer_canonical/viewer"
DEFAULT_D42_STATS = (
    "/mnt/data/artifacts/silico/experiments/_flat/"
    "exp_01kwzzbdj5es78ep4xs1f6aqqt/atlas/struct_stats.json"
)
DEFAULT_GENOMIC_A = (
    "/mnt/data/artifacts/silico/experiments/_flat/"
    "exp_01kx07y2wnf1q8j11ajkpnatw1/atlas/genomic_A.json"
)

DICTS = ("og2", "bcr_k64", "bcr_k16")


def augment_entry(e, ann_id, rates_f, span_f, A_of, total, fire_count) -> bool:
    """Attach the two enrichment metrics (+ rate components + lift) to one entry.

    Returns True if any field was added.  The position-level ``lift = PPV / prior``
    formula is load-bearing for a downstream reproduction check and is preserved
    exactly.
    """
    pos = rates_f.get(ann_id)
    if pos:
        e["pos_rate"], e["bg_rate"] = pos[0], pos[1]
        # position-level effect size -> LIFT = PPV / prior.
        #   pos_rate = (obs + 0.5) / A  ->  obs = pos_rate*A - 0.5 (firings in annotation)
        #   PPV   = obs / fire_count        (P(in annotation | feature fires), bounded [0,1])
        #   prior = A / total_positions     (P(in annotation))
        #   lift  = PPV / prior             (unbounded, replaces the capped shift-fold in display)
        A = A_of.get(ann_id)
        if A and fire_count > 0 and total > 0:
            obs = pos[0] * A - 0.5
            ppv = min(max(obs / fire_count, 0.0), 1.0)
            prior = A / total
            if prior > 0:
                e["ppv"] = round(ppv, 6)
                e["prior"] = round(prior, 9)
                e["lift"] = round(ppv / prior, 3)
    sp = span_f.get(ann_id)
    if sp:
        e["span_fold"], e["cover_rate"], e["matched_bg_rate"] = sp[0], sp[1], sp[2]
    return e is not None and (pos is not None or sp is not None)


def embed_dict(dct: str, *, canon: str, A_of: dict, total_pos: dict) -> dict:
    """Embed metrics for one dictionary in place; return counts."""
    with open(f"{canon}/{dct}/rates.json") as fh:
        rates = json.load(fh)
    with open(f"{canon}/{dct}/span_rates.json") as fh:
        span = json.load(fh)
    total = total_pos.get(dct) or total_pos["bcr"]
    feats = set(rates) | set(span)
    n_files = n_det = n_top = n_lift = 0
    for fs in feats:
        fp = f"{canon}/{dct}/feature/latent_{int(fs):05d}.json"
        if not os.path.exists(fp):
            continue
        with open(fp) as fh:
            d = json.load(fh)
        fc = float(d.get("fire_count", 0) or 0)
        rf = rates.get(fs, {})
        sf = span.get(fs, {})
        changed = False
        for e in d.get("detected", []) or []:
            if augment_entry(e, e.get("id", ""), rf, sf, A_of, total, fc):
                n_det += 1
                changed = True
                if "lift" in e:
                    n_lift += 1
        # top_annotations (legacy): join pfam by unversioned id, else leave
        for e in d.get("top_annotations", []) or []:
            cls = e.get("class")
            pretty = e.get("pretty", "")
            cand = None
            if cls == "pfam":
                cand = "pfam|" + str(pretty).split(".")[0]
            elif cls in ("cath", "ted"):
                cand = f"{cls}|{pretty}"
            if cand and augment_entry(e, cand, rf, sf, A_of, total, fc):
                n_top += 1
                changed = True
        if changed:
            with open(fp, "w") as out:  # in-place truncate: preserves the served hardlink inode
                json.dump(d, out)
            n_files += 1
    print(f"{dct}: rewrote {n_files} feature files; augmented {n_det} detected + "
          f"{n_top} top_annotations entries; lift computed on {n_lift}")
    return {"n_files": n_files, "n_det": n_det, "n_top": n_top, "n_lift": n_lift}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--canon", default=DEFAULT_CANON,
                    help="canonical viewer dir (feature JSONs rewritten in place)")
    ap.add_argument("--d42-stats", default=DEFAULT_D42_STATS,
                    help="#42 struct_stats.json (positions + total_positions)")
    ap.add_argument("--genomic-a", default=DEFAULT_GENOMIC_A,
                    help="genomic_A.json (per-annotation positions for genomic layers)")
    ap.add_argument("--dicts", nargs="+", default=list(DICTS), choices=list(DICTS))
    a = ap.parse_args(argv)

    with open(a.d42_stats) as fh:
        stats = json.load(fh)
    with open(a.genomic_a) as fh:
        genomic_a = json.load(fh)
    A_of = {**stats["positions"], **genomic_a}
    total_pos = stats["total_positions"]

    for dct in a.dicts:
        embed_dict(dct, canon=a.canon, A_of=A_of, total_pos=total_pos)
    print("DONE embed_enrichment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
