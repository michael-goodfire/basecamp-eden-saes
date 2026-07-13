"""Mobile-genetic-element span layer (``mge|<type>``) for annotation panel v2.

Parses the outputs of three CPU tools run per genome into the standard span-TSV
format (``ann_key\\tcontig\\tstrand\\tstart\\tend``, 0-based half-open):

* **CRISPR arrays** -- MinCED GFF (``repeat_region`` features)  -> ``mge|crispr``
* **IS / transposons** -- ISEScan ``.tsv`` (per-IS rows, family column) -> ``mge|IS_<family>``
* **prophages** -- PhiSpy ``prophage_coordinates.tsv``          -> ``mge|prophage``

Tool coordinates are 1-based inclusive; converted to 0-based half-open on parse.

Strand policy: IS elements carry a biological orientation (ISEScan ``strand``
column), emitted on that strand. CRISPR arrays and prophages are large,
strand-agnostic multi-gene / repeat elements with no single coding strand, so
each is emitted on **both** strands (the SAE may represent them on either reading
direction); this doubles their ``n_spans`` consistently across the panel.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import Counter
from pathlib import Path

LABELS_STATIC = {
    "mge|crispr": "CRISPR array (MinCED)",
    "mge|prophage": "prophage / integrated phage (PhiSpy)",
}


def _emit(w, ann: str, contig: str, strand: str, start: int, end: int) -> bool:
    if start < 0 or end <= start:
        return False
    w.write(f"{ann}\t{contig}\t{strand}\t{start}\t{end}\n")
    return True


def parse_minced_gff(path: str | Path, w, both_strands: bool = True) -> Counter:
    """Parse a MinCED GFF; emit the whole-array ``repeat_region`` spans."""
    c: Counter = Counter()
    if not Path(path).exists():
        return c
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 8:
            continue
        contig, _src, ftype, start, end = f[0], f[1], f[2], f[3], f[4]
        if ftype.lower() not in ("repeat_region", "crispr"):
            continue
        s, e = int(start) - 1, int(end)  # 1-based incl -> 0-based half-open
        strands = ["+", "-"] if both_strands else ["+"]
        for st in strands:
            if _emit(w, "mge|crispr", contig, st, s, e):
                c["mge|crispr"] += 1
    return c


_FAMILY_RE = re.compile(r"[^A-Za-z0-9_.-]")


def parse_isescan_tsv(path: str | Path, w) -> Counter:
    """Parse an ISEScan ``.tsv`` (header row); emit per-IS spans keyed by family."""
    c: Counter = Counter()
    if not Path(path).exists():
        return c
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            contig = row.get("seqID") or row.get("seqid")
            fam = (row.get("family") or "unknown").strip()
            fam = _FAMILY_RE.sub("_", fam) or "unknown"
            try:
                b = int(row["isBegin"]); e = int(row["isEnd"])
            except (KeyError, ValueError):
                continue
            strand = row.get("strand", "+").strip() or "+"
            if strand not in ("+", "-"):
                strand = "+"
            s0, e0 = min(b, e) - 1, max(b, e)  # 1-based incl -> 0-based half-open
            ann = f"mge|IS_{fam}"
            if _emit(w, ann, contig, strand, s0, e0):
                c[ann] += 1
    return c


def parse_phispy(path: str | Path, w, both_strands: bool = True) -> Counter:
    """Parse a PhiSpy ``prophage_coordinates.tsv`` (no header): id, contig, start, end, ..."""
    c: Counter = Counter()
    if not Path(path).exists():
        return c
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        f = line.split("\t")
        if len(f) < 4:
            continue
        contig = f[1]
        try:
            start, end = int(f[2]), int(f[3])
        except ValueError:
            continue
        s, e = min(start, end) - 1, max(start, end)
        strands = ["+", "-"] if both_strands else ["+"]
        for st in strands:
            if _emit(w, "mge|prophage", contig, st, s, e):
                c["mge|prophage"] += 1
    return c


def build_accession(tool_dir: str | Path, acc: str, out_tsv: str | Path) -> Counter:
    """Convert one genome's MinCED / ISEScan / PhiSpy outputs into a span TSV."""
    tool_dir = Path(tool_dir)
    total: Counter = Counter()
    with open(out_tsv, "w") as w:
        total.update(parse_minced_gff(tool_dir / f"{acc}.minced.gff", w))
        # ISEScan writes <outdir>/<name>.tsv; accept either flat or prediction/ layout
        is_tsv = tool_dir / f"{acc}.isescan.tsv"
        if not is_tsv.exists():
            cand = list(glob.glob(str(tool_dir / "isescan" / "**" / "*.tsv"), recursive=True))
            is_tsv = Path(cand[0]) if cand else is_tsv
        total.update(parse_isescan_tsv(is_tsv, w))
        total.update(parse_phispy(tool_dir / f"{acc}.phispy.prophage_coordinates.tsv", w))
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert per-genome MGE tool outputs into mge| span TSVs.")
    ap.add_argument("--tool-dir", required=True, help="dir with <acc>.minced.gff / <acc>.isescan.tsv / <acc>.phispy.*")
    ap.add_argument("--out", required=True, help="output layer dir")
    ap.add_argument("--acc", default="", help="single accession; if omitted, process all found")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    grand: Counter = Counter()
    if args.acc:
        accs = [args.acc]
    else:
        accs = sorted({Path(p).name.split(".minced")[0].split(".isescan")[0].split(".phispy")[0]
                       for p in glob.glob(str(Path(args.tool_dir) / "GCF_*"))})
    for acc in accs:
        c = build_accession(args.tool_dir, acc, out / f"{acc}.tsv")
        grand.update(c)
    print(f"mge layer: {len(accs)} genomes")
    for k in sorted(grand):
        print(f"  {k}: {grand[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
