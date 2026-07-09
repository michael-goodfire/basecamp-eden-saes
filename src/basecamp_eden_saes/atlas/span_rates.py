"""Build the per-dictionary span-enrichment sidecar for the viewer.

The viewer already carries the POSITION-level enrichment (the atlas Haldane
circular-shift fold = pos_rate / bg_rate, in each detected entry's `fold`, with
the two rates in rates.json). This adds the SPAN-level enrichment (the #38/#40
precision_fold = fraction-of-spans-covered / matched-negative-span rate, the gate
number), so the viewer can show both, clearly labeled.

span_rates.json = { feature: { ann_id: [precision_fold, cover_rate, bg_rate] } }
  precision_fold = span gate number (cover_rate / bg_rate)
  cover_rate     = recall = n_covered / n_spans  (fraction of the fold's spans the feature covers)
  bg_rate        = cover_rate / precision_fold   (expected covered fraction under the matched-negative background)

Sources: ted/cath from #40's corrected reduced span-enrichment; pfam + genomic
regulatory layers from #38's span-enrichment. Keyed by the exact `ann` id the
viewer's detected entries use (ted|..., cath|..., pfam|..., and the genomic types).
"""
from __future__ import annotations

import argparse
import json
import os

from .. import config

# Canonical on-cluster defaults (overridable via CLI).
DEFAULT_D40 = "/mnt/data/artifacts/silico/experiments/_flat/exp_01kwyn7x4vfp78d14pzp2hs14g"
DEFAULT_CANON = "/mnt/data/artifacts/silico/eden_feature_viewer_canonical/viewer"

DICTS = ("og2", "bcr_k64", "bcr_k16")


def add(span: dict, path: str, only_prefix=None, skip_prefix=None) -> int:
    """Fold one span_enrichment.json into ``span`` (keyed by feature -> ann id)."""
    n = 0
    with open(path) as fh:
        rows = json.load(fh)
    for r in rows:
        ann = r["ann"]
        pfx = ann.split("|", 1)[0]
        if only_prefix and pfx not in only_prefix:
            continue
        if skip_prefix and pfx in skip_prefix:
            continue
        pf = float(r.get("precision_fold") or 0.0)
        rec = float(r.get("recall") or 0.0)
        bg = (rec / pf) if pf > 0 else 0.0
        span.setdefault(str(r["feature"]), {})[ann] = [round(pf, 3), round(rec, 4), round(bg, 6)]
        n += 1
    return n


def build_dict(dct: str, *, d40: str, d38: str, canon: str) -> str:
    """Write ``<canon>/<dict>/span_rates.json`` for one dictionary; return the path."""
    span: dict = {}
    n_ted = add(span, f"{d40}/metrics/{dct}/reduced/ted/span_enrichment.json")
    n_cath = add(span, f"{d40}/metrics/{dct}/reduced/cath/span_enrichment.json")
    # #38: pfam + genomic regulatory layers (no ted/cath in this file)
    n_38 = add(span, f"{d38}/{dct}/span_enrichment.json")
    out = f"{canon}/{dct}/span_rates.json"
    with open(out, "w") as fh:
        json.dump(span, fh)
    sz = os.path.getsize(out)
    print(f"{dct}: ted={n_ted} cath={n_cath} pfam/genomic={n_38} "
          f"feats={len(span)} -> {out} ({sz/1e6:.1f} MB)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--canon", default=DEFAULT_CANON,
                    help="canonical viewer dir; span_rates.json is written under <canon>/<dict>/")
    ap.add_argument("--d40", default=DEFAULT_D40,
                    help="#40 experiment root (reduced ted/cath span_enrichment)")
    ap.add_argument("--d38", default=None,
                    help="#38 metrics root (pfam/genomic span_enrichment); "
                         "defaults to config.data_paths().bg_root")
    ap.add_argument("--dicts", nargs="+", default=list(DICTS), choices=list(DICTS))
    a = ap.parse_args(argv)
    d38 = a.d38 if a.d38 is not None else str(config.data_paths().bg_root)
    for dct in a.dicts:
        build_dict(dct, d40=a.d40, d38=d38, canon=a.canon)
    print("DONE build_span_rates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
