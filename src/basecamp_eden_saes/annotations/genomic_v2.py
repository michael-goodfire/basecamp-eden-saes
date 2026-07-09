"""Pure-code genome-scale annotation layers for DNA-model SAE interpretation (#38).

No downloads and no external binaries. For ONE accession this produces, per
contig:
  (a) per-nt tracks (numpy arrays, length == contig length) written to
      ``<out>/<acc>/<contig>.npz``;
  (b) span records (biological instances) appended (whole accession) to
      ``<out>/<acc>/spans.jsonl``, each line::

        {"contig","strand","nt_start","nt_end","layer","ann_type","ann_id","label"}

      with 0-based half-open coordinates.

Layers: regulatory (rbs / promoter_-10 / promoter_-35 / terminator),
        operon (operon / operon_internal),
        replication (oriC / ter / dnaA_box + gcskew_cumulative / leading_strand),
        motif (GATC / GANTC / CCWGG methylation motifs).

All coordinates are 0-based half-open, matching :mod:`panel`. Genome inputs
(FASTA/GFF3) are resolved from :func:`basecamp_eden_saes.config.data_paths` via
:func:`panel.genome_dir`; the output directory is an explicit ``--out`` argument.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from basecamp_eden_saes import config

from . import panel

# byte codes for A/C/G/T (genome stored uppercase)
_A, _C, _G, _T = 65, 67, 71, 84


def _pat(s: str) -> list[tuple[int, ...] | None]:
    """Turn an IUPAC-ish motif string into a list of ``(bytes) | None``.

    ``N`` -> ``None`` (wildcard, always matches, never a mismatch); ``W`` -> {A,T}.
    """
    table: dict[str, tuple[int, ...] | None] = {
        "A": (_A,), "C": (_C,), "G": (_G,), "T": (_T,),
        "W": (_A, _T), "S": (_C, _G),
        "R": (_A, _G), "Y": (_C, _T),
        "N": None,
    }
    return [table[c] for c in s]


def kmer_match_starts(arr: np.ndarray, pattern: list, max_mm: int = 0) -> np.ndarray:
    """Vectorized k-mer scan with ``<= max_mm`` mismatches.

    ``arr``: uint8 array of the sequence. ``pattern``: list from :func:`_pat`.
    Returns 0-based start indices where the k-mer matches (half-open k-mer
    ``[start, start+k)``). Wildcard (``None``) positions never count as a mismatch.
    """
    k = len(pattern)
    n = arr.shape[0]
    if n < k:
        return np.empty(0, dtype=np.int64)
    L = n - k + 1
    match = np.zeros(L, dtype=np.int32)
    for i, allowed in enumerate(pattern):
        if allowed is None:
            match += 1
            continue
        sub = arr[i:i + L]
        m = sub == allowed[0]
        for b in allowed[1:]:
            m |= sub == b
        match += m
    return np.nonzero(match >= (k - max_mm))[0]


def _reading_window(seq: str, c0: int, c1: int, strand: str) -> tuple[str, int, int]:
    """Return ``(reading_strand_seq, eff_c0, eff_c1)`` for contig span ``[c0, c1)``.

    Clipped to ``[0, len)``. For ``-`` the seq is reverse-complemented (reading
    order). Offsets into ``reading_strand_seq`` map back to contig via
    :func:`_map_span`.
    """
    n = len(seq)
    a = max(0, c0)
    b = min(n, c1)
    if a >= b:
        return "", a, b
    sub = seq[a:b]
    return (sub if strand == "+" else panel.revcomp(sub)), a, b


def _map_span(c0: int, c1: int, strand: str, o0: int, o1: int) -> tuple[int, int]:
    """Map a sub-range ``[o0, o1)`` within reading window ``[c0, c1)`` to contig coords."""
    if strand == "+":
        return c0 + o0, c0 + o1
    return c1 - o1, c1 - o0


def _feat_id(f: panel.Feature) -> str:
    a = f.attrs
    return (a.get("locus_tag") or a.get("protein_id") or a.get("Name")
            or a.get("gene") or a.get("ID") or "?")


def _find_terminator(w: str) -> tuple[int, int] | None:
    """Detect a GC-rich rho-independent terminator in reading-strand window ``w``.

    A reverse-complement stem of length >=5 (perfect pairing, GC content >=50%), a
    loop of 3-8 nt, immediately followed by a T-rich run (>=4 of next 8 are T).
    Returns ``(o0, o1)`` covering hairpin+tail within ``w``, or ``None``. First hit
    wins, preferring the longest stem then shortest loop.
    """
    n = len(w)
    for L in range(11, 4, -1):  # prefer longer, more stable stems
        for loop in range(3, 9):
            span = 2 * L + loop
            if span >= n:
                continue
            for start in range(0, n - span):
                stem1 = w[start:start + L]
                stem2 = w[start + L + loop:start + span]
                if panel.revcomp(stem2) != stem1:
                    continue
                gc = sum(c in "GC" for c in stem1)
                if gc * 2 < L:  # GC content < 50%
                    continue
                tail = w[start + span:start + span + 8]
                if sum(c == "T" for c in tail) >= 4:
                    return start, start + span + len(tail)
    return None


def build_accession(acc: str, out_dir: str | Path,
                    paths: config.DataPaths | None = None) -> dict:
    """Build all genomic annotation layers for one accession.

    Idempotent (overwrites ``<out_dir>/<acc>/`` outputs). Genome inputs are read
    from :func:`panel.genome_dir`. Returns a dict of span counts.
    """
    gd = panel.genome_dir(acc, paths)
    seqs = panel.read_fasta(gd / "genomic.fna")
    feats = panel.parse_gff(gd / "genomic.gff")

    outdir = Path(out_dir) / acc
    outdir.mkdir(parents=True, exist_ok=True)
    sf = open(outdir / "spans.jsonl", "w")

    by_contig: dict[str, list] = defaultdict(list)
    for f in feats:
        by_contig[f.contig].append(f)

    stats: Counter = Counter()

    # precompiled genome-scale motif patterns
    P_DNAA = _pat("TTATCCACA")
    P_DNAA_RC = _pat(panel.revcomp("TTATCCACA"))  # minus-strand boxes
    METH = {"GATC": _pat("GATC"), "GANTC": _pat("GANTC"), "CCWGG": _pat("CCWGG")}
    METH_TRACK = {"GATC": "meth_gatc", "GANTC": "meth_gantc", "CCWGG": "meth_ccwgg"}

    def emit(contig, strand, s, e, layer, ann_type, ann_id, label, n):
        s = max(0, int(s))
        e = min(n, int(e))
        if s >= e:
            return None
        sf.write(json.dumps({
            "contig": contig, "strand": strand,
            "nt_start": s, "nt_end": e,
            "layer": layer, "ann_type": ann_type,
            "ann_id": ann_id, "label": label,
        }) + "\n")
        stats[f"{layer}:{ann_type}"] += 1
        return s, e

    for contig, seq in seqs.items():
        n = len(seq)
        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        cf = by_contig.get(contig, [])

        tr = {
            "rbs": np.zeros(n, dtype=np.uint8),
            "promoter10": np.zeros(n, dtype=np.uint8),
            "promoter35": np.zeros(n, dtype=np.uint8),
            "terminator": np.zeros(n, dtype=np.uint8),
            "operon": np.zeros(n, dtype=np.uint8),
            "dnaa_box": np.zeros(n, dtype=np.uint8),
            "meth_gatc": np.zeros(n, dtype=np.uint8),
            "meth_gantc": np.zeros(n, dtype=np.uint8),
            "meth_ccwgg": np.zeros(n, dtype=np.uint8),
        }

        # ---------------- layer: regulatory (per CDS) ----------------
        cds = [f for f in cf if f.ftype == "CDS"]
        for f in cds:
            strand = f.strand
            cid = _feat_id(f)
            # translation start / stop on the contig
            if strand == "+":
                tss, stop = f.start, f.end
            else:
                tss, stop = f.end, f.start

            # -- RBS: 20nt upstream of translation start (reading strand) --
            if strand == "+":
                c0, c1 = tss - 20, tss
            else:
                c0, c1 = tss, tss + 20
            w, a, b = _reading_window(seq, c0, c1, strand)
            if w:
                wa = np.frombuffer(w.encode("ascii"), dtype=np.uint8)
                for o in kmer_match_starts(wa, _pat("AGGAGG"), 1):
                    s, e = _map_span(a, b, strand, int(o), int(o) + 6)
                    r = emit(contig, strand, s, e, "regulatory", "rbs",
                             cid, w[o:o + 6], n)
                    if r:
                        tr["rbs"][r[0]:r[1]] = 1
                    break  # one RBS per CDS (closest scan hit)

            # -- Promoter boxes: 40nt upstream of CDS start (reading strand) --
            if strand == "+":
                c0, c1 = tss - 40, tss
            else:
                c0, c1 = tss, tss + 40
            w, a, b = _reading_window(seq, c0, c1, strand)
            if w:
                wa = np.frombuffer(w.encode("ascii"), dtype=np.uint8)
                for o in kmer_match_starts(wa, _pat("TATAAT"), 1):
                    s, e = _map_span(a, b, strand, int(o), int(o) + 6)
                    r = emit(contig, strand, s, e, "regulatory",
                             "promoter_-10", cid, w[o:o + 6], n)
                    if r:
                        tr["promoter10"][r[0]:r[1]] = 1
                for o in kmer_match_starts(wa, _pat("TTGACA"), 1):
                    s, e = _map_span(a, b, strand, int(o), int(o) + 6)
                    r = emit(contig, strand, s, e, "regulatory",
                             "promoter_-35", cid, w[o:o + 6], n)
                    if r:
                        tr["promoter35"][r[0]:r[1]] = 1

            # -- Terminator: 60nt downstream of CDS stop (reading strand) --
            if strand == "+":
                c0, c1 = stop, stop + 60
            else:
                c0, c1 = stop - 60, stop
            w, a, b = _reading_window(seq, c0, c1, strand)
            if len(w) >= 13:
                hit = _find_terminator(w)
                if hit:
                    s, e = _map_span(a, b, strand, hit[0], hit[1])
                    r = emit(contig, strand, s, e, "regulatory", "terminator",
                             cid, w[hit[0]:hit[1]], n)
                    if r:
                        tr["terminator"][r[0]:r[1]] = 1

        # ---------------- layer: operon ----------------
        genes = [f for f in cf if f.ftype == "gene"
                 and f.attrs.get("gene_biotype") == "protein_coding"]
        op_idx = 0
        for strand in ("+", "-"):
            gs = sorted([g for g in genes if g.strand == strand],
                        key=lambda g: g.start)
            groups: list[list] = []
            cur: list = []
            for g in gs:
                if cur and (g.start - cur[-1].end) < 50:
                    cur.append(g)
                else:
                    if cur:
                        groups.append(cur)
                    cur = [g]
            if cur:
                groups.append(cur)
            for grp in groups:
                s = min(x.start for x in grp)
                e = max(x.end for x in grp)
                ng = len(grp)
                emit(contig, strand, s, e, "operon", "operon", op_idx, ng, n)
                if ng >= 2:
                    tr["operon"][s:e] = 1
                    # transcription-first gene: lowest coord on +, highest on -
                    first = grp[0] if strand == "+" else grp[-1]
                    for x in grp:
                        if x is first:
                            continue
                        emit(contig, strand, x.start, x.end, "operon",
                             "operon_internal", _feat_id(x), op_idx, n)
                op_idx += 1

        # ---------------- layer: replication ----------------
        step = np.zeros(n, dtype=np.int8)
        step[arr == _G] = 1
        step[arr == _C] = -1
        cum = np.cumsum(step.astype(np.int64))
        ori = int(np.argmin(cum))  # oriC ~ global min of cumulative skew
        ter = int(np.argmax(cum))  # ter  ~ global max

        cmin, cmax = int(cum.min()), int(cum.max())
        rng = (cmax - cmin) or 1
        tr["gcskew_cumulative"] = (
            2.0 * (cum - cmin) / rng - 1.0).astype(np.float16)

        # leading strand: sign of the smoothed skew derivative (= smoothed step)
        win = min(5001, n if n % 2 else n - 1) or 1
        if win >= 3:
            kern = np.ones(win, dtype=np.float64) / win
            smooth = np.convolve(step.astype(np.float64), kern, mode="same")
        else:
            smooth = step.astype(np.float64)
        tr["leading_strand"] = np.where(smooth >= 0, 1, -1).astype(np.int8)

        emit(contig, "+", ori - 5000, ori + 5000, "replication", "oriC",
             "oriC", ori, n)
        emit(contig, "+", ter - 5000, ter + 5000, "replication", "ter",
             "ter", ter, n)

        # dnaA boxes (both strands)
        for o in kmer_match_starts(arr, P_DNAA, 1):
            o = int(o)
            r = emit(contig, "+", o, o + 9, "replication", "dnaA_box",
                     o, seq[o:o + 9], n)
            if r:
                tr["dnaa_box"][r[0]:r[1]] = 1
        for o in kmer_match_starts(arr, P_DNAA_RC, 1):
            o = int(o)
            r = emit(contig, "-", o, o + 9, "replication", "dnaA_box",
                     o, seq[o:o + 9], n)
            if r:
                tr["dnaa_box"][r[0]:r[1]] = 1

        # ---------------- layer: motif (methylation, palindromic) ----------
        for name, pat in METH.items():
            k = len(pat)
            tname = METH_TRACK[name]
            for o in kmer_match_starts(arr, pat, 0):
                o = int(o)
                emit(contig, "+", o, o + k, "motif", name, o, name, n)
                tr[tname][o:o + k] = 1

        # ---------------- write tracks ----------------
        np.savez_compressed(outdir / f"{contig}.npz", **tr)

    sf.close()
    return dict(stats)


def main(argv: list[str] | None = None) -> int:
    """CLI: build v2 genomic layers for one accession, or a shard of the panel."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("accession", nargs="?", default=None,
                    help="single accession to build; omit to build a shard")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--panel-root", type=Path, default=None,
                    help="panel root for the harvest-order accession list "
                         "(default: config.accessions() bundle list)")
    args = ap.parse_args(argv)

    if args.accession is not None:
        accs = [args.accession]
    else:
        if args.panel_root is not None:
            accs = panel.accessions(args.panel_root)
        else:
            accs = config.accessions()
        accs = accs[args.shard::args.nshards]

    print(f"[genomic_v2 shard {args.shard}/{args.nshards}] {len(accs)} accessions")
    for acc in accs:
        st = build_accession(acc, args.out)
        summary = " ".join(f"{k}={v}" for k, v in sorted(st.items()))
        print(f"  {acc}: {summary}")
    print(f"[genomic_v2 shard {args.shard}] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
