"""Rebuild the atlas structural (TED) fold layer at the canonical superfamily unit.

For one dictionary, rewrites the viewer's per-feature `detected` TED entries by
collapsing partial-resolution codes (3-level topology + homogeneous comma
multi-domain) into their dominant 4-level superfamily (canon.canonical_target),
then re-derives index_rows top annotations and prunes rates.json. Distinct 4-level
superfamilies and CATH (already superfamily-level, no commas) are untouched.

Metrics for kept superfamilies are the existing values (identical to #40); when a
feature's dominant superfamily is #40-significant but absent from the atlas's
detected list, the entry is constructed from #40's authoritative enrichment table.
No re-inference and no read of the 740 GB activation store -- this is a re-key +
recompute over already-computed per-code metrics.

Outputs a staging tree (only changed feature files + full index_rows.json +
rates.json + report.json) under ``<staging-root>/<dict>/``.

Console entry point: ``bes-build-atlas`` (``atlas.rebuild:main``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from .canon import bare, canonical_target, load_enrichment

# Canonical on-cluster defaults (overridable via CLI).
DEFAULT_CANON = "/mnt/data/artifacts/silico/eden_feature_viewer_canonical/viewer"
DEFAULT_D40 = "/mnt/data/artifacts/silico/experiments/_flat/exp_01kwyn7x4vfp78d14pzp2hs14g"
DEFAULT_D42_ATLAS = (
    "/mnt/data/artifacts/silico/experiments/_flat/exp_01kwzzbdj5es78ep4xs1f6aqqt/atlas"
)
FOLD_CAP = 999.0
MIN_SUPPORT = 20

DICTS = ("og2", "bcr_k16", "bcr_k64")


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_near_dup(stats: dict) -> dict[str, set[str]]:
    """Symmetric near-duplicate adjacency from the atlas struct_stats near_dup list."""
    nd: dict[str, set[str]] = {}
    for x, y, _j in stats.get("near_dup", []):
        nd.setdefault(x, set()).add(y)
        nd.setdefault(y, set()).add(x)
    return nd


def dedup(det: list[dict], near_dup: dict[str, set[str]]) -> list[dict]:
    """Replicate build_atlas jaccard dedup: sort by -fold, greedily keep, absorb
    near-duplicates into the higher-fold kept entry."""
    det = sorted(det, key=lambda e: -(e.get("fold") or 0.0))
    kept: list[dict] = []
    kept_ids: set[str] = set()
    for e in det:
        dup = near_dup.get(e["id"], set())
        if kept_ids & dup:
            for k in kept:
                if k["id"] in dup:
                    k.setdefault("dedup_absorbed", []).append(e["id"])
                    break
            continue
        kept.append(e)
        kept_ids.add(e["id"])
    return kept


def construct_entry(code: str, rec: dict, sf_feat: dict) -> dict:
    """Build a detected TED entry for a superfamily present in #40 enrichment but
    absent from the atlas's detected list."""
    key = "ted|" + code
    if key in sf_feat:
        fold = min(float(sf_feat[key]), FOLD_CAP)
        method = "haldane-circular"
    else:
        fold = min(float(rec.get("precision_fold", 0.0) or 0.0), FOLD_CAP)
        method = "cap"
    ncov, nsp = int(rec["n_covered"]), int(rec["n_spans"])
    lo, hi = wilson_ci(ncov, nsp)
    return {
        "id": key, "class": "ted", "pretty": rec.get("label") or code,
        "fold": round(fold, 3), "recall": round(float(rec["recall"]), 4),
        "n_spans": nsp, "overlap": ncov, "fdr": rec["q"],
        "fold_method": method, "recall_lo": round(lo, 3), "recall_hi": round(hi, 3),
        "low_support": nsp < MIN_SUPPORT, "constructed": True,
    }


def canonicalize_feature(
    d: dict, feat: int, enr: dict, sf_feat: dict, near_dup: dict[str, set[str]]
) -> tuple[list[dict], bool, list[str]]:
    """Return (new_detected, changed, dropped_ids) for one feature."""
    det = d.get("detected", []) or []
    ted = [e for e in det if e.get("class") == "ted"]
    other = [e for e in det if e.get("class") != "ted"]
    if not ted:
        return det, False, []
    groups: dict[str, list[dict]] = {}
    for e in ted:
        tc, _ = canonical_target(bare(e["id"]), feat, enr)
        groups.setdefault(tc, []).append(e)
    new_ted: list[dict] = []
    dropped: list[str] = []
    constructed = False
    for tc, entries in groups.items():
        target = next((e for e in entries if bare(e["id"]) == tc), None)
        if target is not None:
            new_ted.append(target)
            dropped += [e["id"] for e in entries if e is not target]
        else:
            rec = enr.get((feat, tc))
            if rec is None:  # defensive: keep max-overlap fragment
                keep = max(entries, key=lambda e: e.get("overlap", 0))
                new_ted.append(keep)
                dropped += [e["id"] for e in entries if e is not keep]
            else:
                new_ted.append(construct_entry(tc, rec, sf_feat))
                dropped += [e["id"] for e in entries]
                constructed = True
    changed = bool(dropped) or constructed
    if not changed:
        return det, False, []
    new_det = dedup(other + new_ted, near_dup)
    return new_det, True, dropped


def build_dict(
    dct: str, *, canon: str, d40: str, d42_atlas: str, staging_root: str
) -> dict:
    """Canonicalize the TED layer for one dictionary; write the staging tree.

    Returns the report dict (also written to ``<staging-root>/<dict>/report.json``).
    Emits the CLEAN5 reproduction gate for ``bcr_k64`` on stdout.
    """
    t0 = time.time()
    stage = f"{staging_root}/{dct}"
    os.makedirs(f"{stage}/feature", exist_ok=True)

    enr = load_enrichment(f"{d40}/metrics/{dct}/reduced/ted/span_enrichment.json")
    with open(f"{d42_atlas}/struct_folds_{dct}.json") as fh:
        struct_folds = json.load(fh)
    with open(f"{d42_atlas}/struct_stats.json") as fh:
        stats = json.load(fh)
    near_dup = build_near_dup(stats)

    src = f"{canon}/{dct}"
    with open(f"{src}/index_rows.json") as fh:
        index_rows = json.load(fh)
    rows_by_id = {r["feature_id"]: r for r in index_rows}
    with open(f"{src}/rates.json") as fh:
        rates = json.load(fh)

    feats = sorted({f for (f, _c) in enr.keys()})
    n_changed = n_constructed = n_dropped = n_top_changed = 0
    clean5_check: dict[int, dict] = {}
    CLEAN5 = {10255: 0.91, 27629: 0.75, 2044: 0.66, 9968: 0.73, 28943: 0.58}
    for feat in feats:
        fp = f"{src}/feature/latent_{feat:05d}.json"
        if not os.path.exists(fp):
            continue
        with open(fp) as fh:
            d = json.load(fh)
        sf_feat = struct_folds.get(str(feat), {})
        new_det, changed, dropped = canonicalize_feature(d, feat, enr, sf_feat, near_dup)
        if not changed:
            continue
        n_changed += 1
        n_dropped += len(dropped)
        n_constructed += sum(1 for e in new_det if e.get("constructed"))
        d["detected"] = new_det
        with open(f"{stage}/feature/latent_{feat:05d}.json", "w") as out:
            json.dump(d, out)
        # index row
        row = rows_by_id.get(feat)
        if row is not None:
            old_top = row.get("top_annotation")
            top = new_det[0] if new_det else None
            row["top_annotation"] = top["pretty"] if top else "(unlabeled)"
            row["top_class"] = top["class"] if top else "none"
            row["top_fold"] = round(top["fold"], 2) if top else 0.0
            row["top_recall"] = round(top.get("recall", 0.0), 3) if top else 0.0
            if old_top != row["top_annotation"]:
                n_top_changed += 1
        # prune dropped ted keys from rates
        if str(feat) in rates and dropped:
            fr = rates[str(feat)]
            for k in dropped:
                fr.pop(k, None)
        # CLEAN5 verification: corrected superfamily recall present in detected
        if feat in CLEAN5:
            ted_recs = {bare(e["id"]): e for e in new_det if e.get("class") == "ted"}
            clean5_check[feat] = ted_recs

    # write full index_rows + rates
    with open(f"{stage}/index_rows.json", "w") as out:
        json.dump(index_rows, out)
    with open(f"{stage}/rates.json", "w") as out:
        json.dump(rates, out)
    report = {
        "dict": dct, "n_features_changed": n_changed, "n_partial_rows_dropped": n_dropped,
        "n_constructed_superfamily_entries": n_constructed,
        "n_top_annotation_changed": n_top_changed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(f"{stage}/report.json", "w") as out:
        json.dump(report, out, indent=1)
    print("REPORT", json.dumps(report))
    # CLEAN5 gate (bcr_k64 only)
    if dct == "bcr_k64":
        print("CLEAN5 gate:")
        topo = {10255: "1.10.3120.10", 27629: "1.10.760.10", 2044: "3.40.1090.10",
                9968: "2.170.220.10", 28943: "1.10.443.10"}
        allok = True
        for f, tgt in CLEAN5.items():
            recs = clean5_check.get(f, {})
            e = recs.get(topo[f])
            r = e["recall"] if e else None
            ok = r is not None and abs(round(r, 2) - tgt) <= 0.011
            allok &= ok
            print(f"  {f}: {topo[f]} recall={r} (tgt {tgt}) {'OK' if ok else 'MISMATCH'}")
        print("CLEAN5:", "PASS" if allok else "FAIL")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dict", required=True, choices=list(DICTS))
    ap.add_argument("--staging-root", required=True,
                    help="output root; the staging tree is written under <root>/<dict>/")
    ap.add_argument("--canon", default=DEFAULT_CANON,
                    help="canonical viewer dir (source of index_rows/rates/feature JSONs)")
    ap.add_argument("--d40", default=DEFAULT_D40,
                    help="#40 experiment root (reduced TED span_enrichment)")
    ap.add_argument("--d42-atlas", default=DEFAULT_D42_ATLAS,
                    help="#42 atlas dir (struct_folds_<dict>.json + struct_stats.json)")
    a = ap.parse_args(argv)
    build_dict(a.dict, canon=a.canon, d40=a.d40, d42_atlas=a.d42_atlas,
               staging_root=a.staging_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
