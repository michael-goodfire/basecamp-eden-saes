"""Annotation-grounded autointerp: span-level SAE feature x annotation metrics.

The unit of analysis is an annotation *span* (a Pfam/CATH/TED domain instance, a
gene, an ncRNA, a regulatory element). A feature "covers" a span if it fires on at
least ``cover_frac`` of the span's nucleotide positions. From the full-sparse code
store this module computes, per (annotation, feature):

- ``recall``          = covered_spans / n_spans
- ``precision_fold``  = covered_spans / E_bg, the matched length x GC-negative
                        expectation (the span "gate" number, a.k.a. ``span_fold``)
- ``null_fold``       = covered_spans / circular-shift-null covered_spans
- ``p`` / ``q``       = Poisson upper-tail significance, BH-corrected

``metrics`` holds the pure numeric primitives; ``join_cover`` computes per-genome
partials over the code store; ``reduce`` merges partials into the enrichment table;
``reproduce`` re-derives the pinned viewer metrics as the package's reproduction
check.
"""
