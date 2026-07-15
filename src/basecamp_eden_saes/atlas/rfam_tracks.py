"""Add an Rfam RNA-family track to each viewer exemplar.

The exemplars were built before the Rfam layer existed, so they carry no Rfam
spans. For each exemplar window this reads the genome's Rfam span TSV (from the v2
staging), keeps the families overlapping the window, and writes ``ex['rfam']`` as
window-relative lanes the genome browser renders (same reading-strand-relative,
clipped coordinate convention as the gene tracks). Lane fields: ``start``, ``end``,
``strand`` (reading-strand), ``label`` (family name), ``rf`` (RF accession).
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

from .exemplar_loci import _rel


class GenomeRfam:
    """Per-genome Rfam spans: contig -> list of (rf_acc, strand, start0, end0)."""

    def __init__(self, tsv_path: str | Path):
        self.by_contig: dict[str, list] = defaultdict(list)
        if Path(tsv_path).exists():
            for line in open(tsv_path):
                line = line.rstrip("\n")
                if not line:
                    continue
                ann, c, st, s, e = line.split("\t")
                rf = ann.split("|", 1)[1] if "|" in ann else ann
                self.by_contig[c].append((rf, st, int(s), int(e)))

    def overlapping(self, contig: str, ws: int, we: int):
        return [r for r in self.by_contig.get(contig, []) if r[3] > ws and r[2] < we]


def add_rfam(ex: dict, rf_idx: GenomeRfam, labels: dict[str, str]) -> int:
    contig = ex.get("contig")
    ws, we = int(ex["start"]), int(ex["end"])
    win_strand = ex.get("strand", "+")
    lanes = []
    for rf, gstrand, s, e in rf_idx.overlapping(contig, ws, we):
        r = _rel(s, e, gstrand, ws, we, win_strand)
        if r is None:
            continue
        rs, re_, dst = r
        lanes.append({"start": rs, "end": re_, "strand": dst,
                      "label": labels.get(f"rfam|{rf}", rf), "rf": rf})
    lanes.sort(key=lambda x: (x["start"], x["end"]))
    ex["rfam"] = lanes
    return len(lanes)
