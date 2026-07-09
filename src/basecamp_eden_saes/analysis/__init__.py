"""Structure-beyond-sequence generalization via the corrected both-strand rejoin.

This package re-audits whether SAE fold-detector features generalize *structurally*
rather than merely *sequence*-wise. The scientific correction at its core: the
minus-strand code store is reverse-complement-ordered, so a plus-genome span
``[s, e)`` on the minus strand maps to code-store positions ``[L-e, L-s)`` (``L`` =
contig length). The prior audit sliced the minus store at plus coordinates directly,
silencing minus-strand genes; that bug is fixed here (see :mod:`.rejoin`).

Pipeline:

- :mod:`.codes`          -- CSR reader over one contig-strand code store.
- :mod:`.rejoin`         -- both-strand rejoin -> per-detector coverage members/bg.
- :mod:`.generalization` -- per-detector home vs divergent recall vs background.
- :mod:`.aggregate`      -- cross-model overlap, effect-size lift, #35 clustering link.
- :mod:`.identity`       -- recall-vs-sequence-identity decay curve (MMseqs2).
- :mod:`.validate_store` -- precondition: exemplar activations under corrected vs
                            buggy reverse-complement convention.
"""
