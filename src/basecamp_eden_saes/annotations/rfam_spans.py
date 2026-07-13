"""Rfam RNA-family span layer (``rfam|<RF_accession>``) for panel v2.

The expensive step (Infernal ``cmscan`` of the panel genomes against ``Rfam.cm``
with clan competition) is owned by the upstream Rfam experiment (#3). This module
does the cheap conversion into the standard span TSV
(``ann_key\\tcontig\\tstrand\\tstart\\tend``, 0-based half-open), in two modes:

* ``--from-bundle`` -- verify / re-key #3's already-deposited ``layers/rfam`` span
  TSVs (preferred; no rescan).
* ``--from-tblout`` -- derive spans directly from #3's per-genome cmscan ``--fmt 2``
  ``.tblout`` files (fallback if the bundle deposit is absent/nonstandard). Keeps
  only inclusion-threshold hits (``inc == '!'``) that survive clan competition
  (``olp`` in ``{'*', '^'}``; ``'='`` hits are clan-removed).

Keyed by stable Rfam accession ``rfam|RF#####``; labels (family id + description +
RNA type) come from the Rfam ``family.txt``.
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path


def load_family_labels(family_txt: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (label, rna_type) dicts keyed by ``rfam|RF#####`` from Rfam family.txt."""
    labels: dict[str, str] = {}
    rtype: dict[str, str] = {}
    if not Path(family_txt).exists():
        return labels, rtype
    for line in open(family_txt, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4 or not f[0].startswith("RF"):
            continue
        rf, fam_id, _n, desc = f[0], f[1], f[2], f[3]
        key = f"rfam|{rf}"
        labels[key] = f"{fam_id}: {desc}"
        # the 'Gene; rRNA;' style type column sits around field 19 (0-based 18)
        for cell in f[15:22]:
            if cell.startswith("Gene;") or cell.startswith("Cis-reg;") or cell.startswith("Intron;"):
                rtype[key] = cell.strip().rstrip(";")
                break
    return labels, rtype


def parse_tblout(path: str | Path) -> list[tuple[str, str, str, int, int]]:
    """Parse one cmscan --fmt 2 .tblout into standard spans, applying inc/clan gates."""
    spans: list[tuple[str, str, str, int, int]] = []
    for line in open(path, errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 20:
            continue
        # --fmt 2 columns: idx(0) tname(1) tacc(2) qname(3) qacc(4) clan(5) mdl(6)
        # mdlfrom(7) mdlto(8) seqfrom(9) seqto(10) strand(11) trunc(12) pass(13)
        # gc(14) bias(15) score(16) Evalue(17) inc(18) olp(19) ...
        rf_acc = f[2]            # target accession RF#####
        contig = f[3]            # query name
        seq_from, seq_to = int(f[9]), int(f[10])
        strand = f[11]
        inc = f[18]              # '!' significant, '?' marginal
        olp = f[19]              # '*'/'^' kept, '=' clan-removed
        if not rf_acc.startswith("RF"):
            continue
        if inc != "!" or olp not in ("*", "^"):
            continue
        s0, e0 = min(seq_from, seq_to) - 1, max(seq_from, seq_to)
        if strand not in ("+", "-"):
            strand = "+"
        spans.append((f"rfam|{rf_acc}", contig, strand, s0, e0))
    return spans


def _write(spans, out_tsv) -> Counter:
    c: Counter = Counter()
    with open(out_tsv, "w") as w:
        for a, contig, st, s, e in spans:
            if s < 0 or e <= s:
                continue
            w.write(f"{a}\t{contig}\t{st}\t{s}\t{e}\n")
            c[a] += 1
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build/verify the rfam| span layer.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-tblout", help="dir of <acc>.tblout (cmscan --fmt 2)")
    src.add_argument("--from-bundle", help="existing layers/rfam dir of <acc>.tsv to verify/re-emit")
    ap.add_argument("--out", required=True, help="output layer dir")
    ap.add_argument("--family-txt", default="", help="Rfam family.txt for labels")
    ap.add_argument("--labels-out", default="")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    grand: Counter = Counter()
    n = 0
    if args.from_tblout:
        for tb in sorted(glob.glob(str(Path(args.from_tblout) / "*.tblout"))):
            acc = Path(tb).name.replace(".tblout", "")
            grand.update(_write(parse_tblout(tb), out / f"{acc}.tsv"))
            n += 1
    else:
        from basecamp_eden_saes.autointerp.join_cover import read_spans
        for tsv in sorted(glob.glob(str(Path(args.from_bundle) / "*.tsv"))):
            acc = Path(tsv).name.replace(".tsv", "")
            spans = read_spans([tsv])  # validates the standard 5-column format
            grand.update(_write(spans, out / f"{acc}.tsv"))
            n += 1

    print(f"rfam layer: {n} genomes; {len(grand)} distinct families, {sum(grand.values())} spans")
    top = sorted(grand.items(), key=lambda x: -x[1])[:12]
    for k, v in top:
        print(f"  {k}: {v}")
    if args.labels_out and args.family_txt:
        import json
        labels, rtype = load_family_labels(args.family_txt)
        labels = {k: labels[k] for k in grand if k in labels}
        json.dump(labels, open(args.labels_out, "w"), indent=0)
        json.dump(rtype, open(str(args.labels_out).replace(".json", ".rnatype.json"), "w"), indent=0)
        print(f"wrote {len(labels)} labels -> {args.labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
