"""Both-strand re-join of the stored SAE codes into per-detector coverage.

For one genome and one dictionary, this maps each fold-domain segment onto the code
store and records, per (domain, detector-feature) pair, the fraction of the domain's
nucleotides where the feature fires, its max activation over the domain, and whether
it *covers* the domain. It also records, per detector feature, how many panel domains
(of any fold) it covers -- the sequence-only background rate.

CORRECTED BOTH-STRAND TRANSFORM (the scientific point of this analysis):
    the minus (``.minus.npz``) code store is reverse-complement-ordered, so a domain
    segment at plus-genome coordinates ``[s, e)`` on the ``-`` strand maps to store
    positions ``[L-e, L-s)`` (``L`` = contig length). The prior audit sliced the minus
    store at ``[s, e)`` directly, reading the wrong region and recording minus-strand
    genes as (near) silent. Preserve this transform exactly.

Coverage rule (COVER=0.5, F=32768): feature ``f`` covers domain ``d`` iff ``f`` is
active at ``>= 0.5`` of ``d``'s nucleotide positions (summed across ``d``'s segments,
0-based half-open slices).

Outputs (written under ``<out_dir>/{members,bg}``):
    members/<gcf>.parquet : (domain_id, fold, feature, cover_frac, maxact, covered, aa_len),
        restricted to detector features of each domain's own TED fold (stratified-recall data).
    bg/<gcf>.parquet      : (feature, n_covered_all) -- covered-domain count per detector.
    bg/<gcf>.n            : text file with the panel-domain count for this genome.

CLI (``bes-rejoin``): ``bes-rejoin <dict> <gcf> --detectors ... --domains ... --segments ... --out ...``
The code store defaults to :func:`basecamp_eden_saes.config.data_paths().code_dir` for
the dictionary; override with ``--codes-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from basecamp_eden_saes import config

from .codes import strand_tag

COVER = 0.5
F = 32768

_MEMBER_COLS = ["domain_id", "fold", "feature", "cover_frac", "maxact", "covered", "aa_len"]
_BG_COLS = ["feature", "n_covered_all"]


def rejoin_genome(
    dictname: str,
    gcf: str,
    *,
    detectors_path: str | Path,
    domains_path: str | Path,
    segments_path: str | Path,
    out_dir: str | Path,
    codes_dir: str | Path | None = None,
) -> dict[str, int]:
    """Re-join one genome's codes into member/background coverage parquets.

    ``detectors_path``/``domains_path``/``segments_path`` are the strand-independent
    fold artifacts (detectors_<dict>.parquet, domains.parquet, segments.parquet).
    ``codes_dir`` defaults to the configured code store for ``dictname``. Writes into
    ``<out_dir>/members`` and ``<out_dir>/bg`` and returns a small stats dict.
    """
    out_dir = Path(out_dir)
    if codes_dir is None:
        codes_dir = config.data_paths().code_dir(dictname)
    codes_dir = Path(codes_dir)

    det = pd.read_parquet(detectors_path)
    det_by_fold = {f: np.array(sorted(set(g))) for f, g in det.groupby("fold")["feature"]}
    U = np.array(sorted(set(det.feature)))
    feat2u = np.full(F, -1, np.int32)
    feat2u[U] = np.arange(len(U))
    nU = len(U)

    dom = pd.read_parquet(domains_path)
    dom = dom[dom.gcf == gcf].reset_index(drop=True)
    if dom.empty:
        _write_empty(out_dir, gcf)
        return {"domains": 0, "member_rows": 0, "member_covered": 0}
    row_of = {d: i for i, d in enumerate(dom.domain_id)}
    n_dom = len(dom)

    ucount = np.zeros((n_dom, nU), np.int32)
    umax = np.zeros((n_dom, nU), np.float32)
    ntlen = np.zeros(n_dom, np.int64)

    seg = pd.read_parquet(segments_path)
    seg = seg[seg.domain_id.isin(row_of)]
    root = codes_dir / gcf
    for (contig, strand), grp in seg.groupby(["contig", "strand"], sort=False):
        tag = strand_tag(strand)
        npz = root / f"{contig}.{tag}.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        indptr, indices, values = z["indptr"], z["indices"], z["values"]
        npos = indptr.shape[0] - 1          # = L, contig length
        is_minus = tag == "minus"
        for did, s, e in zip(grp.domain_id, grp.nt_start, grp.nt_end):
            r = row_of[did]
            s, e = int(s), int(e)
            if is_minus:                    # THE FIX: reverse-complement frame [L-e, L-s)
                s, e = npos - e, npos - s
            s0 = max(0, s)
            e0 = min(npos, e)
            if e0 <= s0:
                continue
            lo, hi = indptr[s0], indptr[e0]
            sl = indices[lo:hi]
            uidx = feat2u[sl]
            v = uidx >= 0
            if v.any():
                uu = uidx[v]
                ucount[r] += np.bincount(uu, minlength=nU).astype(np.int32)
                vv = values[lo:hi][v].astype(np.float32)
                order = np.argsort(uu, kind="stable")
                uu_s, vv_s = uu[order], vv[order]
                uniq, first = np.unique(uu_s, return_index=True)
                segmax = np.maximum.reduceat(vv_s, first)
                umax[r, uniq] = np.maximum(umax[r, uniq], segmax)
            ntlen[r] += (e0 - s0)
        del z, indptr, indices, values

    ok = ntlen > 0
    covered = np.zeros((n_dom, nU), bool)
    covered[ok] = ucount[ok] >= (COVER * ntlen[ok])[:, None]

    # background: how many domains (of any fold) each detector feature covers
    bg = pd.DataFrame({"feature": U, "n_covered_all": covered.sum(axis=0).astype(int)})
    bg.to_parquet(_kind_path(out_dir, "bg", gcf))
    (out_dir / "bg" / f"{gcf}.n").write_text(str(int(ok.sum())))

    # member rows (detector features of each domain's own fold only)
    folds = dom.fold.values
    rows = []
    for r in range(n_dom):
        if not ok[r]:
            continue
        fold = folds[r]
        feats = det_by_fold.get(fold)
        if feats is None:
            continue
        uidx = feat2u[feats]
        length = ntlen[r]
        cf = ucount[r, uidx] / length
        mx = umax[r, uidx]
        cov = covered[r, uidx]
        did = dom.domain_id.values[r]
        al = int(dom.aa_len.values[r])
        for f_, c_, m_, cv_ in zip(feats, cf, mx, cov):
            rows.append((did, fold, int(f_), float(c_), float(m_), bool(cv_), al))
    mem = pd.DataFrame(rows, columns=_MEMBER_COLS)
    mem.to_parquet(_kind_path(out_dir, "members", gcf))
    return {
        "domains": int(ok.sum()),
        "member_rows": len(mem),
        "member_covered": int(mem.covered.sum()) if len(mem) else 0,
    }


def _kind_path(out_dir: Path, kind: str, gcf: str) -> Path:
    d = out_dir / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{gcf}.parquet"


def _write_empty(out_dir: Path, gcf: str) -> None:
    pd.DataFrame(columns=_MEMBER_COLS).to_parquet(_kind_path(out_dir, "members", gcf))
    pd.DataFrame(columns=_BG_COLS).to_parquet(_kind_path(out_dir, "bg", gcf))
    (out_dir / "bg" / f"{gcf}.n").write_text("0")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Both-strand rejoin of stored SAE codes -> per-detector coverage."
    )
    ap.add_argument("dict", help="dictionary name (e.g. og2, bcr_k64, bcr_k16)")
    ap.add_argument("gcf", help="genome accession (e.g. GCF_000005845.2)")
    ap.add_argument("--detectors", required=True, help="detectors_<dict>.parquet (#41)")
    ap.add_argument("--domains", required=True, help="domains.parquet (#41)")
    ap.add_argument("--segments", required=True, help="segments.parquet (#41)")
    ap.add_argument("--out", required=True, help="output dir (holds members/ and bg/)")
    ap.add_argument("--codes-dir", default=None, help="code store dir (default: config)")
    args = ap.parse_args(argv)

    stats = rejoin_genome(
        args.dict,
        args.gcf,
        detectors_path=args.detectors,
        domains_path=args.domains,
        segments_path=args.segments,
        out_dir=args.out,
        codes_dir=args.codes_dir,
    )
    print(
        f"{args.dict}/{args.gcf}: domains={stats['domains']} "
        f"member_rows={stats['member_rows']} member_covered={stats['member_covered']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
