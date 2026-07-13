"""eggNOG functional span layer (``ec|`` / ``ko|`` / ``cog|``) for panel v2.

Runs eggNOG-mapper per proteome (separately, in a conda env) and, here, lifts the
per-protein EC number / KEGG-KO / COG functional category to the protein's CDS
nucleotide spans, using the protein_id -> CDS coordinate map read from the RefSeq
GFF3. Emits the standard span TSV (``ann_key\\tcontig\\tstrand\\tstart\\tend``,
0-based half-open) on the CDS coding strand.

Keys:
* ``ec|<number>``   full EC number (e.g. ``ec|2.7.7.7``); partial ``-`` levels kept
* ``ko|<Kxxxxx>``   KEGG ortholog id (``ko:`` prefix stripped)
* ``cog|<letter>``  COG functional category letter (broad, high coverage)

This replaces the Pfam2GO GO layer in the featured set (GO merely echoes Pfam);
eggNOG-GO remains available from the same annotations file if GO is wanted later.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path

COG_LETTER_LABEL = {
    "J": "Translation, ribosomal structure and biogenesis",
    "A": "RNA processing and modification", "K": "Transcription",
    "L": "Replication, recombination and repair",
    "B": "Chromatin structure and dynamics",
    "D": "Cell cycle control, cell division, chromosome partitioning",
    "Y": "Nuclear structure", "V": "Defense mechanisms",
    "T": "Signal transduction mechanisms", "M": "Cell wall/membrane/envelope biogenesis",
    "N": "Cell motility", "Z": "Cytoskeleton", "W": "Extracellular structures",
    "U": "Intracellular trafficking, secretion, vesicular transport",
    "O": "Post-translational modification, protein turnover, chaperones",
    "C": "Energy production and conversion", "G": "Carbohydrate transport and metabolism",
    "E": "Amino acid transport and metabolism", "F": "Nucleotide transport and metabolism",
    "H": "Coenzyme transport and metabolism", "I": "Lipid transport and metabolism",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis, transport and catabolism",
    "R": "General function prediction only", "S": "Function unknown",
}

_PROTID_RE = re.compile(r"protein_id=([^;]+)")


def protein_cds_map(gff_path: str | Path) -> dict[str, list[tuple[str, str, int, int]]]:
    """protein_id -> list of (contig, strand, start, end) 0-based half-open CDS spans.

    Multiple GFF CDS rows for one protein at one locus (join/frameshift) are merged
    to (min start, max end); distinct genomic copies stay separate.
    """
    rows: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for line in open(gff_path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "CDS":
            continue
        m = _PROTID_RE.search(f[8])
        if not m:
            continue
        pid = m.group(1)
        contig, strand = f[0], f[6]
        s, e = int(f[3]) - 1, int(f[4])
        rows[(pid, contig, strand)].append((s, e))
    # merge per (pid, contig, strand) locus; a protein at N genomic copies would
    # share (pid, contig) only if truly co-located, so key includes contig+strand.
    out: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for (pid, contig, strand), segs in rows.items():
        s = min(a for a, _ in segs)
        e = max(b for _, b in segs)
        out[pid].append((contig, strand, s, e))
    return out


def _read_emapper(path: str | Path):
    """Yield dict rows from a .emapper.annotations file (skips ## comment lines)."""
    with open(path) as fh:
        header = None
        for line in fh:
            if line.startswith("##"):
                continue
            if header is None and line.startswith("#"):
                header = line[1:].rstrip("\n").split("\t")
                continue
            if header is None or not line.strip():
                continue
            vals = line.rstrip("\n").split("\t")
            if len(vals) < len(header):
                vals += [""] * (len(header) - len(vals))
            yield dict(zip(header, vals))


def build_accession(emapper_path: str | Path, gff_path: str | Path, out_tsv: str | Path,
                    labels: dict[str, str] | None = None) -> Counter:
    """Lift one proteome's eggNOG annotations to CDS spans."""
    cds = protein_cds_map(gff_path)
    labels = labels if labels is not None else {}
    counts: Counter = Counter()
    with open(out_tsv, "w") as w:
        for row in _read_emapper(emapper_path):
            q = row.get("query") or row.get("#query")
            if not q or q not in cds:
                continue
            keys: list[tuple[str, str]] = []  # (ann_key, label)
            ec = (row.get("EC") or "-").strip()
            if ec and ec != "-":
                for v in ec.split(","):
                    v = v.strip()
                    if v and v != "-":
                        keys.append((f"ec|{v}", f"EC {v}"))
            ko = (row.get("KEGG_ko") or "-").strip()
            if ko and ko != "-":
                for v in ko.split(","):
                    v = v.strip().replace("ko:", "")
                    if v and v != "-":
                        keys.append((f"ko|{v}", f"KEGG ortholog {v}"))
            cog = (row.get("COG_category") or "-").strip()
            if cog and cog not in ("-", ""):
                for letter in cog:
                    if letter.isalpha():
                        keys.append((f"cog|{letter}", COG_LETTER_LABEL.get(letter, letter)))
            if not keys:
                continue
            for (contig, strand, s, e) in cds[q]:
                for ann, lab in keys:
                    w.write(f"{ann}\t{contig}\t{strand}\t{s}\t{e}\n")
                    counts[ann] += 1
                    labels.setdefault(ann, lab)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lift eggNOG EC/KO/COG annotations to CDS span TSVs.")
    ap.add_argument("--emapper-dir", required=True, help="dir with <acc>.emapper.annotations")
    ap.add_argument("--panel", required=True, help="ncbi_dataset/data dir with <acc>/genomic.gff")
    ap.add_argument("--out", required=True, help="output layer dir")
    ap.add_argument("--labels-out", default="")
    ap.add_argument("--acc", default="", help="single accession; else all found")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    labels: dict[str, str] = {}
    grand: Counter = Counter()
    if args.acc:
        accs = [args.acc]
    else:
        accs = sorted(Path(p).name.split(".emapper")[0]
                      for p in glob.glob(str(Path(args.emapper_dir) / "*.emapper.annotations")))
    n = 0
    for acc in accs:
        emap = Path(args.emapper_dir) / f"{acc}.emapper.annotations"
        gff = Path(args.panel) / acc / "genomic.gff"
        if not emap.exists() or not gff.exists():
            print(f"  skip {acc}: emap={emap.exists()} gff={gff.exists()}")
            continue
        c = build_accession(emap, gff, out / f"{acc}.tsv", labels=labels)
        grand.update(c); n += 1
    print(f"eggnog layer: {n} genomes; {len(grand)} distinct keys, {sum(grand.values())} spans")
    top = sorted(grand.items(), key=lambda x: -x[1])[:15]
    for k, v in top:
        print(f"  {k}: {v}")
    if args.labels_out:
        import json
        json.dump(labels, open(args.labels_out, "w"), indent=0)
        print(f"wrote {len(labels)} labels -> {args.labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
