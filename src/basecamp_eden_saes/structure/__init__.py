"""Structure-annotation layers for the EDEN SAE feature atlas.

This package builds the structural-domain annotation layers used to test whether
SAE features track protein fold, alongside the sequence (Pfam) layer:

- :mod:`.cath`  -- CATH-Gene3D superfamily layer via phmmer against CATH v4.3.0
  S95 representative domain sequences.
- :mod:`.ted`   -- TED (The Encyclopedia of Domains) fold layer via
  RefSeq -> UniProt -> TED domain -> nucleotide spans.
- :mod:`.ted_filter`, :mod:`.background` -- TED summary stream-filter and the
  covered-universe matched-negative background for TED enrichment metrics.
- :mod:`.mapping` -- RefSeq -> UniProtKB ID mapping with UniParc MD5 fallbacks.
- :mod:`.labels`, :mod:`.pfam_spans` -- human-readable labels and the Pfam
  validation span TSVs.
- :mod:`.coverage`, :mod:`.afdb` -- AlphaFold-DB structure coverage of the atlas
  exemplar proteins (enumerate exemplars, download mmCIF, build the struct index).
- :mod:`.esmfold` -- the ESMFold gap-fold pipeline. This stage was DEFERRED (not
  run at scale); the code ships for reproducibility.

All span TSVs share the uniform format::

    ann_key<TAB>contig<TAB>strand<TAB>nt_start<TAB>nt_end

with ``ann_key`` one of ``cath|C.A.T.H``, ``ted|C.A.T.H`` or ``pfam|PFxxxxx``.
"""

from __future__ import annotations
