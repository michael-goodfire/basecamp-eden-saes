"""Build the TED structural-domain layer for one genome.

For each protein: RefSeq id -> UniProt accession(s) (from the mapping) -> TED
domains (from the filtered TED table) -> lift aa->nt via the CDS map. A protein's
``WP_`` id can map to several UniProt entries (identical sequence across
organisms); we take the first that carries TED domains (same sequence -> same
AlphaFold structure -> same TED domains). Spans are keyed by CATH superfamily
(``ted|C.A.T.H``) for direct comparison with the CATH-Gene3D layer; TED domains
with no CATH assignment are written to ``domains.tsv`` but not the fold-metric
spans.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def lift(cds_info: dict, af: int, at: int) -> tuple[int, int]:
    """Lift an aa range ``[af, at]`` to a nucleotide span via the CDS segments."""
    strand = cds_info["strand"]
    segs = cds_info["segs"]
    nt_lo, nt_hi = (af - 1) * 3, at * 3
    ordered = segs if strand == "+" else list(reversed(segs))

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


def parse_bounds(s: str) -> list[tuple[int, int]]:
    """Parse TED boundary strings (discontinuous, ``11-41_290-389``) to segments."""
    segs: list[tuple[int, int]] = []
    for part in s.split("_"):
        if "-" in part:
            a, b = part.split("-")
            try:
                segs.append((int(a), int(b)))
            except ValueError:
                pass
    return segs


def main(argv: list[str] | None = None) -> int:
    """Build the TED layer for one genome from CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--cds-map", required=True, help="cds_map.json")
    ap.add_argument("--mapping", required=True, help="refseq2uniprot.json")
    ap.add_argument("--ted", required=True, help="filtered TED table (ted_filtered.tsv)")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-spans", required=True)
    args = ap.parse_args(argv)

    cds = json.load(open(args.cds_map))
    pid2contig = {
        pid: (c, info) for c, prots in cds.items() for pid, info in prots.items()
    }
    mapping = json.load(open(args.mapping))  # {refseq: [[uni, tax], ...]}

    # TED rows for the uniprots we might need (load all; filtered table is compact)
    ted: dict[str, list[dict]] = defaultdict(list)
    with open(args.ted) as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            ted[f[0]].append(
                {"ted_id": f[1], "consensus": f[2], "bounds": f[3], "cath": f[4]}
            )

    n_dom = n_span = n_prot = 0
    with open(args.out_tsv, "w") as tw, open(args.out_spans, "w") as sw:
        tw.write(
            "contig\tnt_start\tnt_end\tstrand\tcath_sfam\tted_id\tconsensus\t"
            "protein_id\tuniprot\taa_from\taa_to\n"
        )
        for pid, (contig, info) in pid2contig.items():
            unis = mapping.get(pid) or mapping.get(pid.split(".")[0]) or []
            chosen = None
            for acc, _tax in unis:
                if acc in ted:
                    chosen = acc
                    break
            if not chosen:
                continue
            n_prot += 1
            for dom in ted[chosen]:
                for (af, at) in parse_bounds(dom["bounds"]):
                    ns, ne = lift(info, af, at)
                    tw.write(
                        f"{contig}\t{ns}\t{ne}\t{info['strand']}\t{dom['cath']}\t"
                        f"{dom['ted_id']}\t{dom['consensus']}\t{pid}\t{chosen}\t{af}\t{at}\n"
                    )
                    n_dom += 1
                    if dom["cath"] and dom["cath"] != "-":
                        sw.write(
                            f"ted|{dom['cath']}\t{contig}\t{info['strand']}\t{ns}\t{ne}\n"
                        )
                        n_span += 1
    print(
        f"{args.acc}: {n_prot} proteins with TED, {n_dom} TED domain segments, "
        f"{n_span} CATH-assigned spans",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
