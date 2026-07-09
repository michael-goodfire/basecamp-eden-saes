"""Recall-vs-sequence-identity decay curve, per dictionary.

For every tested detector, each member domain's amino-acid sequence identity to the
detector's home exemplar is measured (MMseqs2 ``easy-search`` over the tested-fold
members), then coverage recall is aggregated in identity bins. A *structure* detector
keeps high recall even in the low-identity bins; a *sequence* detector's recall decays
toward the background as identity drops.

Inputs: the generalization ``detectors_<dict>.parquet`` (for tested detectors and home
exemplars), the corrected rejoin ``members/`` parquets, and ``domains.parquet`` (for
amino-acid sequences, column ``seq``). The MMseqs2 binary is an explicit argument.

Writes ``member_identity_<dict>.parquet`` (per member: identity to home + covered) and
``decay_<dict>.parquet`` (id_bin, n, recall) into the output dir.

CLI: ``... <dict> --detectors-dir ... --rejoin-dir ... --domains ... --mmseqs ... --out ...``
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

BINS = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
BINLAB = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]


def identity_curve(
    dictname: str,
    *,
    detectors_dir: str | Path,
    rejoin_dir: str | Path,
    domains_path: str | Path,
    mmseqs: str | Path,
    out_dir: str | Path,
    workdir: str | Path | None = None,
    threads: int | None = None,
) -> pd.DataFrame | None:
    """Build the recall-vs-identity decay curve for one dictionary.

    Returns the decay-curve DataFrame (or ``None`` if no detectors were tested).
    ``workdir`` defaults to a fresh temp dir; ``threads`` defaults to the process's
    CPU affinity count.
    """
    detectors_dir = Path(detectors_dir)
    rejoin_dir = Path(rejoin_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    det = pd.read_parquet(detectors_dir / f"detectors_{dictname}.parquet")
    if det.empty:
        print("no tested detectors")
        return None
    mem = pd.concat(
        [pd.read_parquet(p) for p in sorted((rejoin_dir / "members").glob("*.parquet"))],
        ignore_index=True,
    )
    dom = pd.read_parquet(domains_path)[["domain_id", "seq"]]
    seqof = dict(zip(dom.domain_id, dom.seq))

    tested_folds = set(det.fold)
    memf = mem[mem.fold.isin(tested_folds)]
    member_ids = sorted(set(memf.domain_id))
    exemplars = sorted(set(det.home_exemplar))

    tmp = Path(tempfile.mkdtemp(prefix="idc_", dir=str(workdir) if workdir else None))
    qfaa = tmp / "exemplars.faa"   # query = home exemplars only (few hundred)
    tfaa = tmp / "members.faa"     # target = all tested-fold members
    with open(qfaa, "w") as fh:
        for d in exemplars:
            fh.write(f">{d}\n{seqof[d]}\n")
    with open(tfaa, "w") as fh:
        for d in member_ids:
            fh.write(f">{d}\n{seqof[d]}\n")
    resm = tmp / "res.m8"
    nthreads = str(threads if threads is not None else len(os.sched_getaffinity(0)))
    cmd = [
        str(mmseqs), "easy-search", str(qfaa), str(tfaa), str(resm), str(tmp / "t"),
        "--min-seq-id", "0.0", "-s", "6.0", "--max-seqs", "100000",
        "-e", "10", "--threads", nthreads, "--format-output", "query,target,fident",
    ]
    print(f"mmseqs search: {len(exemplars)} exemplars vs {len(member_ids)} members", flush=True)
    subprocess.run(cmd, check=True)
    hits = pd.read_csv(resm, sep="\t", header=None, names=["q", "t", "fident"])
    hits["pident"] = hits.fident * 100.0
    pid: dict[tuple[str, str], float] = {}
    for q, t, p in zip(hits.q, hits.t, hits.pident):  # key (exemplar, member)
        k = (q, t)
        if p > pid.get(k, -1):
            pid[k] = p

    # per member, identity to its detector's home exemplar
    recs = []
    memf_idx = {(f, fold): g for (f, fold), g in memf.groupby(["feature", "fold"])}
    for r in det.itertuples():
        g = memf_idx.get((r.feature, r.fold))
        if g is None:
            continue
        ex = r.home_exemplar
        for did, cov in zip(g.domain_id, g.covered):
            idnt = 100.0 if did == ex else pid.get((ex, did), 0.0)
            recs.append((r.feature, r.fold, did, idnt, bool(cov)))
    md = pd.DataFrame(recs, columns=["feature", "fold", "domain_id", "id_to_home", "covered"])
    md.to_parquet(out_dir / f"member_identity_{dictname}.parquet")

    md["bin"] = pd.cut(md.id_to_home, BINS, labels=BINLAB, right=False)
    curve = md.groupby("bin", observed=True).agg(
        n=("covered", "size"), recall=("covered", "mean")).reset_index()
    curve.to_parquet(out_dir / f"decay_{dictname}.parquet")
    return curve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recall-vs-sequence-identity decay curve (MMseqs2) for one dictionary."
    )
    ap.add_argument("dict", help="dictionary name (e.g. og2, bcr_k64, bcr_k16)")
    ap.add_argument("--detectors-dir", required=True, help="dir with detectors_<dict>.parquet")
    ap.add_argument("--rejoin-dir", required=True, help="dir with members/ (from bes-rejoin)")
    ap.add_argument("--domains", required=True, help="domains.parquet (with amino-acid seq column)")
    ap.add_argument("--mmseqs", required=True, help="path to the mmseqs binary")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--workdir", default=None, help="parent dir for the mmseqs temp dir")
    ap.add_argument("--threads", type=int, default=None, help="mmseqs threads (default: CPU affinity)")
    args = ap.parse_args(argv)

    curve = identity_curve(
        args.dict,
        detectors_dir=args.detectors_dir,
        rejoin_dir=args.rejoin_dir,
        domains_path=args.domains,
        mmseqs=args.mmseqs,
        out_dir=args.out,
        workdir=args.workdir,
        threads=args.threads,
    )
    if curve is not None:
        print(curve.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
