"""OpenGenome2 corpus construction and reading for EDEN-7B SAE training.

:mod:`~basecamp_eden_saes.corpus.build` streams GTDB / metagenome shards from
HuggingFace into a flat token store (``tokens.bin`` + ``index.npy`` +
``sequences.db`` + ``meta.json``) and exposes :class:`CorpusReader` for reading it
back; :mod:`~basecamp_eden_saes.corpus.rebuild` reconstructs ``tokens.bin`` from
the surviving provenance.
"""

from __future__ import annotations
