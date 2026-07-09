"""Stratified structure-vs-sequence recall, per dictionary (both-strand labels).

For each fold detector (feature ``f``, TED topology fold ``F``), the detector's member
domains are split into the *home* sequence cluster (the cluster of ``f``'s
top-activation exemplar) vs the sequence-*divergent* clusters, and coverage recall is
compared against a sequence-only baseline::

    recall_home = P(f covers member | home cluster)
    recall_div  = P(f covers member | divergent clusters)   <- structure-beyond-sequence
    background  = P(f covers a random panel domain outside fold F)  <- sequence-only baseline

A *structure* detector fires on sequence-divergent members above the sequence-only
baseline: ``recall_div``'s Wilson lower bound ``> background``. A *strong* generalizer
additionally has ``recall_div >= 0.5`` and ``recall_div >= 0.5 * recall_home``.

Inputs are the corrected rejoin outputs (``members/``, ``bg/`` from :mod:`.rejoin`) and
the strand-independent sequence-cluster assignment (clusters.parquet, #41). Writes
``detectors_<dict>.parquet`` and ``summary_<dict>.json`` into the output dir.

CLI (``bes-generalization``): ``bes-generalization <dict> --rejoin-dir ... --clusters ... --out ...``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from basecamp_eden_saes.autointerp import metrics as M

MIN_MEMBERS = 20
MIN_HOME = 3
MIN_DIV = 5
Z = 1.96


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    """Return ``(point, lower, upper)`` Wilson-score estimate of the rate ``k/n``."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    lo, hi = M.wilson_interval(k, n, z)
    return k / n, lo, hi


def analyze_dict(
    dictname: str,
    *,
    rejoin_dir: str | Path,
    clusters_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Compute per-detector home/divergent/background recall for one dictionary.

    ``rejoin_dir`` holds the ``members/`` and ``bg/`` outputs of :mod:`.rejoin`.
    Returns ``(detectors_df, summary)``.
    """
    rejoin_dir = Path(rejoin_dir)
    mem = pd.concat(
        [pd.read_parquet(p) for p in sorted((rejoin_dir / "members").glob("*.parquet"))],
        ignore_index=True,
    )
    bgf = pd.concat(
        [pd.read_parquet(p) for p in sorted((rejoin_dir / "bg").glob("*.parquet"))],
        ignore_index=True,
    )
    n_covered_all = bgf.groupby("feature")["n_covered_all"].sum()
    n_total = sum(int(p.read_text()) for p in (rejoin_dir / "bg").glob("*.n"))

    clu = pd.read_parquet(clusters_path)
    cl_of = dict(zip(clu.domain_id, clu.cluster))
    mem["cluster"] = mem.domain_id.map(cl_of)
    mem = mem.dropna(subset=["cluster"])

    rows = []
    for (f, fold), g in mem.groupby(["feature", "fold"]):
        n = len(g)
        if n < MIN_MEMBERS:
            continue
        nclu = g.cluster.nunique()
        ex_row = g.loc[g.maxact.idxmax()]
        home = ex_row["cluster"]
        home_exemplar = ex_row["domain_id"]
        hg = g[g.cluster == home]
        dg = g[g.cluster != home]
        if len(hg) < MIN_HOME or len(dg) < MIN_DIV or nclu < 2:
            continue
        rh, rh_lo, rh_hi = wilson(int(hg.covered.sum()), len(hg))
        rd, rd_lo, rd_hi = wilson(int(dg.covered.sum()), len(dg))
        cov_own = int(g.covered.sum())
        bg = (int(n_covered_all.get(f, 0)) - cov_own) / max(1, (n_total - n))
        overall_recall = int(g.covered.sum()) / n
        rows.append({
            "feature": int(f), "fold": fold, "n_members": n, "n_clusters": int(nclu),
            "home_cluster": home, "home_exemplar": home_exemplar,
            "n_home": len(hg), "n_div": len(dg),
            "recall_home": rh, "recall_div": rd, "recall_div_lo": rd_lo, "recall_div_hi": rd_hi,
            "recall_overall": overall_recall, "background": bg,
            "div_above_bg": bool(rd_lo > bg),
            "strong_generalizer": bool(rd >= 0.5 and rd >= 0.5 * rh and rd_lo > bg),
            "div_home_ratio": rd / rh if rh > 0 else np.nan,
        })
    df = pd.DataFrame(rows)

    if len(df):
        summ = {
            "dict": dictname, "N_panel_domains": n_total,
            "n_detectors_tested": len(df),
            "n_folds_tested": int(df.fold.nunique()),
            "n_features_tested": int(df.feature.nunique()),
            "frac_structure_detectors": float(df.div_above_bg.mean()),
            "n_structure_detectors": int(df.div_above_bg.sum()),
            "frac_strong_generalizers": float(df.strong_generalizer.mean()),
            "median_recall_home": float(df.recall_home.median()),
            "median_recall_div": float(df.recall_div.median()),
            "median_background": float(df.background.median()),
            "median_div_home_ratio": float(df.div_home_ratio.median()),
        }
    else:
        summ = {"dict": dictname, "n_detectors_tested": 0}
    return df, summ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-detector structure-beyond-sequence recall for one dictionary."
    )
    ap.add_argument("dict", help="dictionary name (e.g. og2, bcr_k64, bcr_k16)")
    ap.add_argument("--rejoin-dir", required=True, help="dir with members/ and bg/ (from bes-rejoin)")
    ap.add_argument("--clusters", required=True, help="clusters.parquet (#41)")
    ap.add_argument("--out", required=True, help="output dir for detectors_<dict>.parquet + summary")
    args = ap.parse_args(argv)

    df, summ = analyze_dict(args.dict, rejoin_dir=args.rejoin_dir, clusters_path=args.clusters)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / f"detectors_{args.dict}.parquet")
    with open(outdir / f"summary_{args.dict}.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
