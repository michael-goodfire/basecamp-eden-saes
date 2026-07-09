"""Shared panel access + coordinate frame for the annotation build.

Coordinate conventions (must match :mod:`tracks` / :mod:`genomic_v2` exactly):
  - internal coords are 0-based half-open ``[start, end)`` (Python slice);
  - GFF3 input is 1-based inclusive -> converted to ``[start-1, end)`` on load;
  - per-nt tracks are stored per genome strand (``*_plus`` / ``*_minus``); a query
    is resolved relative to a window's *reading* strand (sense vs antisense);
  - ``cds_map.json`` lifts protein residues to nt::

        {contig: {protein_id: {"strand", "phase", "segs": [[s, e], ...]}}}

    where ``segs`` are 0-based half-open nt spans of the CDS on the contig, in
    translation order.

This module is the single contract every annotation builder + the metric engine
imports; coordinate math is not re-implemented elsewhere.

Paths
-----
Genome FASTA/GFF3 are resolved from :func:`basecamp_eden_saes.config.data_paths`
(``genome_fasta(acc)`` / ``panel_genomes``). The per-nt track outputs
(``annot/<acc>/<contig>.npz`` and ``annot/<acc>/cds_map.json``) and the harvest
metadata (``codes/meta.json``) are experiment artifacts not tracked in
:mod:`config`, so functions that read them take an explicit ``panel_root``
argument. The canonical on-cluster ``panel_root`` was
``/mnt/data/artifacts/silico/experiments/_flat/exp_01kwd6629qfvsvdjda4kfyft2z``
(containing ``panel/genomes``, ``annot`` and ``codes``).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from basecamp_eden_saes import config

COMP = bytes.maketrans(b"ACGTacgtN", b"TGCAtgcaN")


# ---------- panel accession / harvest metadata ----------

def accessions(panel_root: str | Path) -> list[str]:
    """Panel accession list in harvest order, from ``<panel_root>/codes/meta.json``.

    (See :func:`basecamp_eden_saes.config.accessions` for the annotation-bundle
    accession list, which may differ in ordering.)
    """
    with open(Path(panel_root) / "codes" / "meta.json") as f:
        return json.load(f)["accs"]


def contig_names(panel_root: str | Path) -> dict:
    """``{acc_index -> {contig_index -> contig_id}}`` from the harvest metadata."""
    with open(Path(panel_root) / "codes" / "meta.json") as f:
        return json.load(f)["contig_names"]


# ---------- genome / GFF loading ----------

def read_fasta(path: str | Path) -> dict[str, str]:
    """Return ``{contig_id: uppercase sequence}``. Handles ``.gz``."""
    path = str(path)
    op = gzip.open if path.endswith(".gz") else open
    out: dict[str, list[str]] = {}
    cur = None
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                out[cur] = []
            elif cur is not None:
                out[cur].append(line.strip())
    return {k: "".join(v).upper() for k, v in out.items()}


def revcomp(s: str) -> str:
    """Reverse-complement an ACGT(N) string."""
    return s.translate(COMP)[::-1]


@dataclass
class Feature:
    """A single GFF3 feature in 0-based half-open coordinates."""

    contig: str
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive
    strand: str  # '+' or '-'
    ftype: str  # gene, CDS, tRNA, rRNA, ncRNA, ...
    attrs: dict


def parse_gff(path: str | Path) -> list[Feature]:
    """Parse GFF3 (1-based inclusive) -> :class:`Feature` in 0-based half-open."""
    path = str(path)
    op = gzip.open if path.endswith(".gz") else open
    feats: list[Feature] = []
    with op(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            contig, _src, ftype, s, e, _score, strand, _phase, attr = p[:9]
            attrs: dict[str, str] = {}
            for kv in attr.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    attrs[k] = v
            feats.append(Feature(contig, int(s) - 1, int(e), strand, ftype, attrs))
    return feats


def genome_dir(acc: str, paths: config.DataPaths | None = None) -> Path:
    """Directory holding ``genomic.fna`` / ``genomic.gff`` for ``acc``.

    Resolved from :func:`basecamp_eden_saes.config.data_paths` (panel genomes).
    """
    paths = paths or config.data_paths()
    return paths.genome_fasta(acc).parent


# ---------- #31 per-nt track loaders (cached) ----------

@lru_cache(maxsize=8)
def load_annot(panel_root: str | Path, acc: str, contig: str) -> dict:
    """Per-nt tracks for one contig -> ``{track: np.ndarray}``.

    Reads ``<panel_root>/annot/<acc>/<contig>.npz``.
    """
    z = np.load(Path(panel_root) / "annot" / acc / f"{contig}.npz")
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=4)
def load_cds_map(panel_root: str | Path, acc: str) -> dict:
    """Load ``<panel_root>/annot/<acc>/cds_map.json`` (protein->CDS-segment map)."""
    with open(Path(panel_root) / "annot" / acc / "cds_map.json") as f:
        return json.load(f)


def lift_aa_to_nt(cds_map: dict, contig: str, protein_id: str,
                  aa_from: int, aa_to: int) -> list[tuple[int, int]]:
    """Lift a protein residue range to nucleotide spans on the contig.

    ``aa_from`` / ``aa_to`` are 1-based inclusive protein coordinates. Returns a
    list of 0-based half-open ``(nt_start, nt_end)`` spans (multiple if the
    residue range crosses a CDS-segment boundary). Strandedness is encoded by the
    segment order in ``cds_map``.
    """
    rec = cds_map.get(contig, {}).get(protein_id)
    if rec is None:
        return []
    strand = rec["strand"]
    segs = rec["segs"]
    # residue r (0-based) occupies codon nt [3r, 3r+3) along the translated strand
    nt_lo = (aa_from - 1) * 3
    nt_hi = aa_to * 3  # half-open end of last residue's codon
    spans: list[tuple[int, int]] = []
    if strand == "+":
        # translation order = ascending contig coords across segs in given order
        cds_pos = 0
        for s, e in segs:
            seg_len = e - s
            a = max(nt_lo, cds_pos)
            b = min(nt_hi, cds_pos + seg_len)
            if a < b:
                spans.append((s + (a - cds_pos), s + (b - cds_pos)))
            cds_pos += seg_len
    else:
        cds_pos = 0
        for s, e in reversed(segs):
            seg_len = e - s
            a = max(nt_lo, cds_pos)
            b = min(nt_hi, cds_pos + seg_len)
            if a < b:
                # cds offset o maps to contig coord (e-1 - o), descending
                hi = e - (a - cds_pos)  # exclusive upper (contig)
                lo = e - (b - cds_pos)  # inclusive lower (contig)
                spans.append((lo, hi))
            cds_pos += seg_len
    return spans
