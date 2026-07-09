"""basecamp-eden-saes: sparse-autoencoder engine + feature viewer for EDEN-7B.

End-to-end pipeline for training and interpreting BatchTopK sparse autoencoders on
the EDEN-7B DNA foundation model:

- ``corpus``      OpenGenome2 5B-token corpus build (GTDB + metagenome sampling,
                  byte-level DNA tokenization, low-complexity filtering).
- ``harvest``     activation harvesting (on-the-fly streaming, on-disk, and the
                  full-sparse per-nucleotide code store).
- ``sae``         BatchTopK SAE training with a shared-forward expansion x k grid
                  sweep and best-SAE selection.
- ``annotations`` RefSeq annotation-panel build (per-nt tracks, Pfam/GO domains,
                  and the v2 genomic layers) and panel loaders.
- ``autointerp``  annotation-grounded span metrics (recall, matched-negative
                  precision fold, circular-shift null) over the code store.
- ``structure``   CATH-S95 + TED fold layers and AlphaFold-DB structure coverage.
- ``analysis``    structure-beyond-sequence generalization (both-strand rejoin,
                  home vs sequence-divergent recall, recall-vs-identity decay).
- ``atlas``       canonical per-feature atlas builder (fold-granularity
                  canonicalization, dual span/position enrichment).

The standalone feature-viewer front-end lives under ``viewer/`` at the repo root.

See the module docstrings and the top-level README for the pipeline order and the
data inputs (model weights, the annotation bundle, the code store, the structure
cache) that are referenced by path rather than committed.
"""

__version__ = "0.1.0"
