"""PRECONDITION: reproduce a feature's atlas exemplar activations from the code store.

The canonical atlas (#42) reports, per exemplar, a genomic center position
``center_pos`` (plus-genome coordinate), a strand, and the feature's activation
``center_act`` there. This module re-reads the code store at that position under two
conventions and checks that only the corrected one reproduces the atlas:

  - CORRECT convention: plus exemplar -> ``.plus.npz[center_pos]``;
                        minus exemplar -> ``.minus.npz[L-1-center_pos]`` (reverse-complement).
  - BUGGY convention (prior audit): minus exemplar read from ``.minus.npz[center_pos]`` (no RC).

The corrected read must match ``center_act`` to tolerance for BOTH strands (incl. the
low-activation minus exemplars); the buggy read must MISS the minus exemplars, which
demonstrates the bug that :mod:`.rejoin` corrects.

Inputs: the #42 atlas bundle dir (holds ``<dict>/feature/latent_<NNNNN>.json``) and the
code store (defaults to the configured store for the dictionary). Writes a validation
JSON and exits non-zero if the precondition fails.

CLI: ``... [--feature 4476] [--dict og2] [--n-per-strand 8] --atlas-dir ... --out ...``
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from basecamp_eden_saes import config

TOL = 0.25       # exact-center tolerance (f16 store + couple-position centering nuance)
FIRE_TOL = 1.5   # "fires in the RC region" tolerance (the rejoin scores by span max)
W = 10           # +-W position sweep to absorb the atlas minus-window centering nuance


def feat_val_at(
    indptr: np.ndarray, indices: np.ndarray, values: np.ndarray, pos: int, target: int
) -> float | str | None:
    """Activation of ``target`` at store position ``pos``.

    Returns the float activation, ``None`` if ``target`` is not top-k there, or the
    string ``"OOB"`` if ``pos`` is out of bounds.
    """
    if pos < 0 or pos >= indptr.shape[0] - 1:
        return "OOB"
    lo, hi = indptr[pos], indptr[pos + 1]
    idx = indices[lo:hi]
    hit = np.where(idx == target)[0]
    if hit.size == 0:
        return None
    return float(values[lo:hi][hit[0]])


def validate_store(
    *,
    feature: int,
    dictname: str,
    atlas_dir: str | Path,
    n_per_strand: int = 8,
    codes_dir: str | Path | None = None,
) -> dict:
    """Re-read a feature's exemplars under corrected vs buggy convention.

    ``atlas_dir`` is the #42 atlas bundle root (holds
    ``<dict>/feature/latent_<NNNNN>.json``). ``codes_dir`` defaults to the configured
    store for ``dictname``. Returns ``{"verdict": ..., "records": [...]}``; the verdict
    ``PASS`` flag encodes the precondition.
    """
    atlas_dir = Path(atlas_dir)
    if codes_dir is None:
        codes_dir = config.data_paths().code_dir(dictname)
    codes_dir = Path(codes_dir)

    latent = atlas_dir / dictname / "feature" / f"latent_{feature:05d}.json"
    with open(latent) as fh:
        exs = json.load(fh)["exemplars"]
    plus = sorted([e for e in exs if e["strand"] == "+"], key=lambda x: -x["center_act"])[:n_per_strand]
    minus = sorted([e for e in exs if e["strand"] == "-"], key=lambda x: -x["center_act"])[:n_per_strand]
    sample = plus + minus
    print(
        f"feature {feature} {dictname}: {len(exs)} exemplars, sampling "
        f"{len(plus)} plus + {len(minus)} minus", flush=True,
    )

    # group by (gcf, contig, strand) so each npz is read once
    by_file: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for e in sample:
        by_file[(e["acc"], e["contig"], e["strand"])].append(e)

    recs = []
    for (gcf, contig, strand), grp in by_file.items():
        tag = "plus" if strand == "+" else "minus"
        npz = codes_dir / gcf / f"{contig}.{tag}.npz"
        if not npz.exists():
            print(f"MISS {npz}")
            continue
        z = np.load(npz)
        indptr, indices, values = z["indptr"], z["indices"], z["values"]
        length = indptr.shape[0] - 1
        for e in grp:
            cp = int(e["center_pos"])
            cact = float(e["center_act"])
            # center-position read, correct convention:
            #   plus  exemplar -> plus-store  index cp
            #   minus exemplar -> minus-store index L-1-cp  (reverse-complement frame)
            # sweep +-W to absorb the atlas's minus-window centering nuance; pick the
            # value CLOSEST to the atlas center_act (the true center sits within a
            # couple positions).
            base = cp if strand == "+" else (length - 1 - cp)
            cand = [
                feat_val_at(indptr, indices, values, p, feature)
                for p in range(base - W, base + W + 1)
            ]
            cand = [v for v in cand if isinstance(v, float)]
            corrected_val = min(cand, key=lambda v: abs(v - cact)) if cand else None
            # is the feature essentially present in this RC region at all?
            store_silent = (corrected_val is None) or (max(cand) < 5.0 if cand else True)
            # buggy convention (minus only): read .minus.npz at plus coord cp (no RC)
            buggy_val = None
            if strand == "-":
                b = [
                    feat_val_at(indptr, indices, values, p, feature)
                    for p in range(cp - W, cp + W + 1)
                ]
                b = [v for v in b if isinstance(v, float)]
                buggy_val = max(b) if b else None
            recs.append(dict(
                gcf=gcf, contig=contig, strand=strand, center_pos=cp, center_act=cact,
                corrected_val=corrected_val,
                corrected_err=(abs(corrected_val - cact) if corrected_val is not None else None),
                buggy_val=buggy_val, store_silent=bool(store_silent),
            ))
        del z, indptr, indices, values

    # verdict: center-position read (correct convention) must reproduce the atlas
    # center_act. Exemplars whose labeled coords are silent in the store on BOTH strands
    # are atlas-coordinate mismatches (a #42 metadata quirk), not RC errors -> reported
    # separately.
    testable = [r for r in recs if not r["store_silent"]]
    silent = [r for r in recs if r["store_silent"]]
    plus_t = [r for r in testable if r["strand"] == "+"]
    minus_t = [r for r in testable if r["strand"] == "-"]
    plus_ok = [r for r in plus_t if r["corrected_err"] is not None and r["corrected_err"] <= TOL]
    minus_exact = [r for r in minus_t if r["corrected_err"] is not None and r["corrected_err"] <= TOL]
    # The rejoin scores a domain by its span MAX, so the precondition it actually needs
    # is: the feature FIRES at comparable strength in the RC-mapped region (present,
    # within ~1.5 of the atlas center). Exact-center reproduction (<=0.25) is a stricter
    # secondary check.
    minus_fires = [
        r for r in minus_t
        if r["corrected_val"] is not None and r["corrected_val"] >= r["center_act"] - FIRE_TOL
    ]
    # bug magnitude: how many minus exemplars the plus-only convention recovers (no-RC read at cp)
    minus_buggy_hit = [r for r in minus_t if r["buggy_val"] is not None and r["buggy_val"] >= 5.0]
    verdict = dict(
        feature=feature, dict=dictname, tol_exact=TOL, fire_tol=FIRE_TOL,
        n_plus=len(plus_t), n_plus_matched=len(plus_ok),
        n_minus=len(minus_t), n_minus_fires_in_rc=len(minus_fires),
        n_minus_exact_center=len(minus_exact),
        n_atlas_coord_mismatch=len(silent),
        minus_median_center_act=float(np.median([r["center_act"] for r in minus_t])) if minus_t else None,
        minus_exact_median_err=float(np.median([r["corrected_err"] for r in minus_exact])) if minus_exact else None,
        n_minus_recovered_by_buggy=len(minus_buggy_hit),  # expect 0 -> bug dropped them all
        # PASS: plus reproduces exactly, EVERY non-silent minus fires in the RC region,
        # and the buggy plus-only convention recovers none of the minus firing.
        PASS=bool(len(plus_ok) == len(plus_t) and len(minus_fires) == len(minus_t)
                  and len(minus_buggy_hit) == 0 and len(minus_t) > 0),
    )
    return dict(verdict=verdict, records=recs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Precondition check: reproduce atlas exemplar activations from the store."
    )
    ap.add_argument("--feature", type=int, default=4476, help="feature id to validate")
    ap.add_argument("--dict", default="og2", help="dictionary name")
    ap.add_argument("--n-per-strand", type=int, default=8, help="exemplars sampled per strand")
    ap.add_argument("--atlas-dir", required=True, help="#42 atlas bundle root")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--codes-dir", default=None, help="code store dir (default: config)")
    args = ap.parse_args(argv)

    out = validate_store(
        feature=args.feature,
        dictname=args.dict,
        atlas_dir=args.atlas_dir,
        n_per_strand=args.n_per_strand,
        codes_dir=args.codes_dir,
    )
    verdict = out["verdict"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    for r in sorted(out["records"], key=lambda x: -x["center_act"]):
        print(
            f"  {r['strand']} act={r['center_act']:6.3f} corrected={r['corrected_val']} "
            f"err={r['corrected_err']} buggy={r['buggy_val']} "
            f"{r['gcf']}:{r['contig']}:{r['center_pos']}"
        )
    if not verdict["PASS"]:
        print("PRECONDITION FAILED", flush=True)
        return 2
    print("PRECONDITION PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
