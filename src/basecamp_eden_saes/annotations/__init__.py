"""Genome-grounded annotation panel for DNA/protein SAE autointerp.

This subpackage builds the per-nucleotide annotation layers that an EDEN SAE
feature's per-nt activations are joined against on genomic coordinates. It has
five pieces:

* :mod:`refseq` -- resolve a curated taxon list to RefSeq accessions and download
  the genome panel (FASTA + GFF3 + protein + CDS) via the NCBI ``datasets`` CLI.
* :mod:`tracks` -- the base per-nt track engine (#31): structural / positional /
  compositional tracks (CDS, codon phase, RNA classes, pseudogene, regulatory,
  intergenic, near-start/stop, GC, GC-skew, low-complexity) plus the protein->CDS
  segment map used to lift residue coordinates to nucleotides.
* :mod:`domains` -- Pfam (pyhmmer ``hmmsearch`` at the gathering cutoff) + GO
  (pfam2go) domain layer, lifting amino-acid spans to nucleotide intervals.
* :mod:`genomic_v2` -- the v2 genome-scale layers (#38): regulatory (RBS /
  promoter -10 / -35 / terminator), operon, replication (oriC / ter / dnaA box /
  cumulative GC-skew / leading strand) and methylation motifs.
* :mod:`panel` -- shared panel loaders + coordinate frame used by the v2 build
  and the metric engine.

Coordinate discipline (uniform across the subpackage): internal coordinates are
0-based half-open ``[start, end)``; GFF3 input (1-based inclusive) is converted on
load; per-nt tracks are stored per genome strand and queried relative to a
window's reading strand (sense vs antisense).
"""
from __future__ import annotations
