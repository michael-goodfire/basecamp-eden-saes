"""Cross-model overlap, effect-size refinement, and the #35 spatial-clustering link.

Reads the per-detector generalization parquets (:mod:`.generalization` output) and
produces:

- ``lift = recall_div - background`` and a refined ``structure_strict`` flag
  (significant vs baseline AND ``lift >= lift_floor`` effect-size floor).
- cross-model overlap: TED folds structure-detected in *both* OG2 and BCR (k64),
  by fold code (Jaccard over folds tested in both).
- the #35 link: do structure detectors have higher spatial-clustering z (#35's
  per-feature ``z_W20``) than sequence-bound detectors? Tested per feature with a
  one-sided Mann-Whitney U, plus a Spearman correlation between clustering z and the
  divergent/home recall ratio. Only dictionaries with a supplied clustering npz are
  linked (#35 stored ef8_k64 features -> og2 and bcr_k64).

Writes ``aggregate.json`` and ``detectors_<dict>_ext.parquet`` into the analysis dir.

CLI: ``... --analysis-dir DIR [--dicts og2 bcr_k64 ...] [--clust og2=z.npz bcr_k64=z.npz]``
The #35 clustering npz files are explicit args (each ``DICT=PATH``, key ``z_W20``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from basecamp_eden_saes import config

LIFT_FLOOR = 0.05


def aggregate(
    analysis_dir: str | Path,
    *,
    dicts: list[str] | None = None,
    clust_npz: dict[str, str | Path] | None = None,
    lift_floor: float = LIFT_FLOOR,
) -> dict:
    """Compute cross-model + effect-size + #35-link summary and write ext parquets.

    ``analysis_dir`` holds ``detectors_<dict>.parquet``. ``clust_npz`` maps a
    dictionary name to its #35 clustering npz (key ``z_W20``); omit a dict to skip its
    link. Returns the aggregate summary dict (also written to ``aggregate.json``).
    """
    analysis_dir = Path(analysis_dir)
    dicts = list(dicts) if dicts is not None else list(config.DICTS)
    clust_npz = clust_npz or {}

    out: dict = {"lift_floor": lift_floor}
    det: dict[str, pd.DataFrame] = {}
    for d in dicts:
        df = pd.read_parquet(analysis_dir / f"detectors_{d}.parquet")
        df["lift"] = df.recall_div - df.background
        df["structure_strict"] = df.div_above_bg & (df.lift >= lift_floor)
        det[d] = df
        out[d] = {
            "n_tested": len(df),
            "frac_sig_above_bg": float(df.div_above_bg.mean()) if len(df) else 0.0,
            "frac_structure_strict": float(df.structure_strict.mean()) if len(df) else 0.0,
            "n_structure_strict": int(df.structure_strict.sum()),
            "median_lift": float(df.lift.median()) if len(df) else float("nan"),
            "median_recall_home": float(df.recall_home.median()) if len(df) else float("nan"),
            "median_recall_div": float(df.recall_div.median()) if len(df) else float("nan"),
            "median_div_home_ratio": float(df.div_home_ratio.median()) if len(df) else float("nan"),
        }
        # #35 spatial-clustering link, per FEATURE (clustering z is per-feature;
        # structure_strict is per feature x fold, so aggregate to the feature first).
        npz_path = clust_npz.get(d)
        if npz_path:
            z = np.load(npz_path)["z_W20"]
            df["clust_z"] = df.feature.map(lambda f: float(z[f]))
            perf = df.groupby("feature").agg(
                clust_z=("clust_z", "first"),
                mean_div_home=("div_home_ratio", "mean"),
                frac_structure=("structure_strict", "mean"),
                n_folds=("fold", "nunique"),
            ).reset_index()
            perf["is_structure_feature"] = perf.frac_structure >= 0.5
            sd = perf[perf.is_structure_feature]
            nsd = perf[~perf.is_structure_feature]
            valid = perf.dropna(subset=["mean_div_home", "clust_z"])
            if len(sd) > 5 and len(nsd) > 5:
                _, p = mannwhitneyu(sd.clust_z, nsd.clust_z, alternative="greater")
                rho, prho = spearmanr(valid.clust_z, valid.mean_div_home)
                out[d]["clust35_n_features"] = int(len(perf))
                out[d]["clust35_median_z_structure_feats"] = float(sd.clust_z.median())
                out[d]["clust35_median_z_seqbound_feats"] = float(nsd.clust_z.median())
                out[d]["clust35_mannwhitney_p"] = float(p)
                out[d]["clust35_spearman_ratio_vs_z"] = float(rho)
                out[d]["clust35_spearman_p"] = float(prho)
                out[d]["clust35_global_median_z"] = float(np.nanmedian(z))
                out[d]["clust35_detector_median_z"] = float(perf.clust_z.median())
        df.to_parquet(analysis_dir / f"detectors_{d}_ext.parquet")

    # cross-model overlap OG2 vs BCR (k64), by fold code
    if "og2" in det and "bcr_k64" in det:
        def sfolds(name: str) -> set:
            return set(det[name][det[name].structure_strict].fold)

        og2f, bcrf = sfolds("og2"), sfolds("bcr_k64")
        tested_both = set(det["og2"].fold) & set(det["bcr_k64"].fold)  # universe
        inter = og2f & bcrf
        union = (og2f | bcrf) & tested_both
        out["cross_model_og2_bcr_k64"] = {
            "folds_tested_both": len(tested_both),
            "structure_folds_og2": len(og2f & tested_both),
            "structure_folds_bcr_k64": len(bcrf & tested_both),
            "structure_folds_both": len(inter),
            "jaccard": len(inter) / len(union) if union else 0.0,
        }

    with open(analysis_dir / "aggregate.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def _parse_clust(items: list[str]) -> dict[str, str]:
    """Parse ``DICT=PATH`` entries into a ``{dict: path}`` map."""
    mapping: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--clust entry must be DICT=PATH, got {it!r}")
        k, v = it.split("=", 1)
        mapping[k] = v
    return mapping


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cross-model overlap, effect-size lift, and #35 clustering link."
    )
    ap.add_argument("--analysis-dir", required=True, help="dir with detectors_<dict>.parquet")
    ap.add_argument("--dicts", nargs="+", default=list(config.DICTS), help="dictionary names")
    ap.add_argument("--clust", nargs="*", default=[], help="#35 clustering npz as DICT=PATH (key z_W20)")
    ap.add_argument("--lift-floor", type=float, default=LIFT_FLOOR)
    args = ap.parse_args(argv)

    out = aggregate(
        args.analysis_dir,
        dicts=args.dicts,
        clust_npz=_parse_clust(args.clust),
        lift_floor=args.lift_floor,
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
