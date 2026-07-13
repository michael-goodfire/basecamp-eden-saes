"""Resolve each viewer exemplar's protein/gene tracks to the SPECIFIC CDS copy at
its genomic locus, from the RefSeq panel GFF.

Fixes a defect in the upstream per-exemplar gene-track assembler: it aggregated
each protein's coordinates across ALL genomic copies (min-start / max-end) and
then assigned proteins to windows by overlap against those aggregated spans. For a
multi-copy mobile element (transposase / IstB / helicase / RT) the aggregated span
covers megabases, so it (a) is mis-assigned as the exemplar ``prot`` and (b) shows
up in dozens of windows' ``genes[]``, each clamped to the full window. This breaks
the gene-track render (piled full-width bars) and the 3D activation painting (the
nt->residue offset is computed from the corrupt ``gs/ge``).

The fix resolves, per exemplar:
  * ``prot``  = the CDS whose interval contains the window's activation center
                (``center_pos``); its true single-copy ``gs/ge`` (0-based half-open),
                strand, protein_id, product.
  * ``genes`` = every CDS actually overlapping ``[window_start, window_end)``, with
                coordinates and strand relative to the window's READING strand
                (``+`` window: ``gene0 - ws``; ``-`` window: mirrored within
                ``[0, width]`` and strand flipped), clipped to ``[0, width]``.

Conventions verified against known-correct exemplars (feature 2 #1 on a ``-`` window,
#10 on a ``+`` window): ``prot.gs = gff_start - 1``, ``prot.ge = gff_end`` (0-based
half-open); gene track coords/strand are reading-strand-relative.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path

_PID = re.compile(r"protein_id=([^;]+)")
_PROD = re.compile(r"product=([^;]+)")


class GenomeCDS:
    """Per-genome CDS index: contig -> sorted list of (start0, end0, strand, pid, product)."""

    def __init__(self, gff_path: str | Path):
        self.by_contig: dict[str, list[tuple[int, int, str, str, str]]] = {}
        for line in open(gff_path):
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            m, p = _PID.search(f[8]), _PROD.search(f[8])
            rec = (int(f[3]) - 1, int(f[4]), f[6],
                   m.group(1) if m else "", (p.group(1) if p else ""))
            self.by_contig.setdefault(f[0], []).append(rec)
        for c in self.by_contig:
            self.by_contig[c].sort()
        self._starts = {c: [r[0] for r in v] for c, v in self.by_contig.items()}

    def overlapping(self, contig: str, ws: int, we: int) -> list[tuple[int, int, str, str, str]]:
        recs = self.by_contig.get(contig, [])
        return [r for r in recs if r[1] > ws and r[0] < we]

    def containing(self, contig: str, pos: int) -> tuple[int, int, str, str, str] | None:
        """Innermost/shortest CDS whose [start0,end0) contains pos (0-based)."""
        hits = [r for r in self.by_contig.get(contig, []) if r[0] <= pos < r[1]]
        if not hits:
            return None
        return min(hits, key=lambda r: r[1] - r[0])  # shortest = most specific


def _rel(gs0: int, ge0: int, gstrand: str, ws: int, we: int, win_strand: str):
    """Reading-strand-relative, clipped [0,width] coords + displayed strand."""
    width = we - ws
    fs, fe = max(0, gs0 - ws), min(width, ge0 - ws)  # forward-clipped
    if fe <= fs:
        return None
    if win_strand == "-":
        rs, re_ = width - fe, width - fs
        dst = "+" if gstrand == "-" else "-"
    else:
        rs, re_ = fs, fe
        dst = gstrand
    return rs, re_, dst


def covered_residues(gs0: int, ge0: int, gstrand: str, ws: int, we: int) -> tuple[int, int]:
    """[res_lo, res_hi) of the protein covered by the window (0-based residue idx)."""
    lo_nt, hi_nt = max(ws, gs0), min(we, ge0)
    if hi_nt <= lo_nt:
        return (0, 0)
    if gstrand == "-":
        res_lo = (ge0 - hi_nt) // 3
        res_hi = -(-(ge0 - lo_nt) // 3)  # ceil
    else:
        res_lo = (lo_nt - gs0) // 3
        res_hi = -(-(hi_nt - gs0) // 3)
    return (res_lo, res_hi)


def fix_exemplar(ex: dict, cds: GenomeCDS) -> dict:
    """Rewrite one exemplar's ``prot`` + ``genes`` in place. Returns a small report."""
    contig = ex["contig"]
    ws, we = int(ex["start"]), int(ex["end"])
    width = we - ws
    win_strand = ex.get("strand", "+")
    center_pos = int(ex.get("center_pos", ws + width // 2))

    # genes[] = all CDS overlapping the window, reading-strand-relative
    new_genes = []
    for gs0, ge0, gst, pid, prod in cds.overlapping(contig, ws, we):
        r = _rel(gs0, ge0, gst, ws, we, win_strand)
        if r is None:
            continue
        rs, re_, dst = r
        new_genes.append({"start": rs, "end": re_, "strand": dst,
                          "label": prod or pid, "product": prod, "protein_id": pid})
    new_genes.sort(key=lambda g: (g["start"], g["end"]))

    # prot = CDS containing the activation center
    spec = cds.containing(contig, center_pos)
    report = {"n_genes_before": len(ex.get("genes") or []), "n_genes_after": len(new_genes)}
    if spec is not None:
        gs0, ge0, gst, pid, prod = spec
        old = ex.get("prot") or {}
        ex["prot"] = {
            "id": pid, "gs": gs0, "ge": ge0, "st": gst, "product": prod,
            "go": old.get("go", []), "domains": old.get("domains", []),
        }
        rlo, rhi = covered_residues(gs0, ge0, gst, ws, we)
        ex["prot"]["res_lo"], ex["prot"]["res_hi"] = rlo, rhi  # covered residue span (fix #3)
        ex["prot"]["prot_len_aa"] = (ge0 - gs0) // 3
        report.update({"prot_id": pid, "prot_gs": gs0, "prot_ge": ge0,
                       "prot_len_bp": ge0 - gs0, "res_cov": [rlo, rhi], "product": prod})
    else:
        report["prot"] = "no CDS at center (intergenic) -> grey"
    ex["genes"] = new_genes
    return report
