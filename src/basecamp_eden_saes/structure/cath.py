"""Build the CATH-Gene3D structural-domain layer for one proteome.

The Gene3D FunFam HMM library is effectively un-downloadable from the CATH mirror
(the origin ceilings at ~4.19 GB). We instead assign CATH superfamilies by
searching each proteome against the CATH v4.3.0 S95 representative domain
sequences (``cath-domain-seqs-S95.fa``, 62,915 seqs) with ``pyhmmer.phmmer``,
mapping each hit's CATH domain id to its superfamily via ``cath-domain-list.txt``.
This is sequence-similarity based (less sensitive to remote homologs than the
FunFam profile HMMs) but needs no download.

Pipeline per protein: phmmer hits -> greedy non-overlapping resolution by
bitscore -> superfamily -> aa->nt lift via the CDS map -> ``domains.tsv`` plus a
fold-metric ``spans.tsv`` with ``ann_key = cath|C.A.T.H``.
"""

from __future__ import annotations

import argparse
import json
import re

import pyhmmer

DOMRE = re.compile(r"\|([0-9a-zA-Z]+)/")  # cath|4_3_0|12asA00/4-330 -> 12asA00


def _name(x: str | bytes) -> str:
    """Normalise a pyhmmer name to ``str`` (bytes on some versions)."""
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def load_domain_to_sfam(path: str) -> dict[str, str]:
    """Parse ``cath-domain-list.txt``: col0 domain_id, cols1..4 = ``C.A.T.H``."""
    m: dict[str, str] = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        v = line.split()
        if len(v) >= 5:
            m[v[0]] = ".".join(v[1:5])
    return m


def load_sfam_names(path: str) -> dict[str, str]:
    """Parse ``CathNames.txt`` (``node_id  example_dom  :name``) into names."""
    names: dict[str, str] = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        toks = line.split(None, 2)
        if len(toks) >= 3 and ":" in toks[2]:
            names[toks[0]] = toks[2].split(":", 1)[1].strip()
    return names


def resolve_overlaps(doms: list[dict], max_overlap: float = 0.4) -> list[dict]:
    """Greedily keep the highest-scoring non-overlapping domain hits.

    Hits are sorted by descending bitscore; a candidate is dropped if it overlaps
    any already-kept hit by more than ``max_overlap`` of the shorter hit's length.
    """
    doms = sorted(doms, key=lambda d: -d["score"])
    kept: list[dict] = []
    for d in doms:
        if not any(
            (min(d["at"], k["at"]) - max(d["af"], k["af"]))
            > max_overlap * min(d["at"] - d["af"], k["at"] - k["af"])
            for k in kept
        ):
            kept.append(d)
    return kept


def lift(cds_info: dict, af: int, at: int) -> tuple[int, int]:
    """Lift an aa range ``[af, at]`` to a nucleotide span via the CDS segments."""
    strand = cds_info["strand"]
    segs = cds_info["segs"]
    nt_lo, nt_hi = (af - 1) * 3, at * 3
    ordered = [(a, b) for a, b in segs] if strand == "+" else [
        (a, b) for a, b in reversed(segs)
    ]

    def tx(off: int) -> int:
        rem = off
        for a, b in ordered:
            ln = b - a
            if rem <= ln:
                return (a + rem) if strand == "+" else (b - rem)
            rem -= ln
        a, b = ordered[-1]
        return b if strand == "+" else a

    c1, c2 = tx(nt_lo), tx(nt_hi)
    return (min(c1, c2), max(c1, c2))


def main(argv: list[str] | None = None) -> int:
    """Build the CATH layer for one proteome from CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--faa", required=True, help="proteome protein FASTA")
    ap.add_argument("--cds-map", required=True, help="cds_map.json")
    ap.add_argument("--cath-seqs", required=True, help="cath-domain-seqs-S95.fa")
    ap.add_argument("--domain-list", required=True, help="cath-domain-list.txt")
    ap.add_argument("--cath-names", default="", help="CathNames.txt (optional)")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-spans", required=True)
    ap.add_argument("--evalue", type=float, default=1e-3)
    ap.add_argument("--cpus", type=int, default=8)
    args = ap.parse_args(argv)

    dom2sfam = load_domain_to_sfam(args.domain_list)
    sfam_names = load_sfam_names(args.cath_names) if args.cath_names else {}
    cds = json.load(open(args.cds_map))
    pid2contig = {
        pid: (c, info) for c, prots in cds.items() for pid, info in prots.items()
    }

    alpha = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(args.faa, digital=True, alphabet=alpha) as sf:
        proteome = sf.read_block()
    with pyhmmer.easel.SequenceFile(args.cath_seqs, digital=True, alphabet=alpha) as sf:
        cath_seqs = sf.read_block()

    per_prot: dict[str, list[dict]] = {}
    for hits in pyhmmer.hmmer.phmmer(
        cath_seqs, proteome, cpus=args.cpus, E=args.evalue, domE=args.evalue
    ):
        qname = _name(hits.query.name)
        mo = DOMRE.search(qname)
        if not mo:
            continue
        sfam = dom2sfam.get(mo.group(1))
        if sfam is None:
            continue
        for hit in hits:
            pid = _name(hit.name)
            for dom in hit.domains:
                if dom.i_evalue > args.evalue:
                    continue
                per_prot.setdefault(pid, []).append(
                    {
                        "sfam": sfam,
                        "af": dom.alignment.target_from,
                        "at": dom.alignment.target_to,
                        "score": dom.score,
                        "ievalue": dom.i_evalue,
                    }
                )

    n = 0
    with open(args.out_tsv, "w") as tw, open(args.out_spans, "w") as sw:
        tw.write(
            "contig\tnt_start\tnt_end\tstrand\tcath_sfam\tcath_name\tprotein_id\t"
            "aa_from\taa_to\tbitscore\tievalue\n"
        )
        for pid, doms in per_prot.items():
            if pid not in pid2contig:
                continue
            contig, info = pid2contig[pid]
            for d in resolve_overlaps(doms):
                ns, ne = lift(info, d["af"], d["at"])
                nm = sfam_names.get(d["sfam"], "")
                tw.write(
                    f"{contig}\t{ns}\t{ne}\t{info['strand']}\t{d['sfam']}\t{nm}\t{pid}\t"
                    f"{d['af']}\t{d['at']}\t{d['score']:.1f}\t{d['ievalue']:.2e}\n"
                )
                sw.write(f"cath|{d['sfam']}\t{contig}\t{info['strand']}\t{ns}\t{ne}\n")
                n += 1
    print(f"{args.acc}: {n} CATH domains -> {args.out_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
