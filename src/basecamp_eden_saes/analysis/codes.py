"""Query which SAE features fire in a nucleotide span of the stored codes.

Code store layout: ``<codes_dir>/<accession>/<contig>.{plus,minus}.npz``, each a CSR
over the nucleotide positions of one contig-strand::

    indptr  (L+1,) int64    row = nt position; L = contig length
    indices (nnz,) uint16   feature ids firing at that position (top-k)
    values  (nnz,) f16      activation magnitudes

Features firing anywhere in ``[start, end)`` = ``unique(indices[indptr[start]:indptr[end]])``.
Membership is orientation-independent, so recall (does a feature fire in a span) does
not depend on read direction; only the correct ``[start, end)`` slice and strand file
matter. The minus store is reverse-complement-ordered, so callers must map a plus-genome
minus-strand span ``[s, e)`` to ``[L-e, L-s)`` before slicing (see
:func:`basecamp_eden_saes.autointerp.metrics.span_to_code_coords`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def strand_tag(strand: str | int) -> str:
    """Map a strand label to the code-store file tag (``"plus"`` or ``"minus"``)."""
    return "plus" if strand in ("+", "plus", "1", 1) else "minus"


class ContigCodes:
    """Lazily-loaded CSR (``indptr``/``indices``/``values``) for one contig-strand npz."""

    def __init__(self, npz_path: str | Path) -> None:
        z = np.load(npz_path)
        self.indptr: np.ndarray = z["indptr"]
        self.indices: np.ndarray = z["indices"]
        self.values: np.ndarray = z["values"]
        self.n_pos: int = int(self.indptr.shape[0] - 1)

    def span_features(self, start0: int, end0: int) -> np.ndarray:
        """Unique feature ids firing in 0-based half-open ``[start0, end0)``."""
        s = max(0, start0)
        e = min(self.n_pos, end0)
        if e <= s:
            return np.empty(0, dtype=np.uint16)
        lo = self.indptr[s]
        hi = self.indptr[e]
        if hi <= lo:
            return np.empty(0, dtype=np.uint16)
        return np.unique(self.indices[lo:hi])

    def span_feature_maxact(
        self, start0: int, end0: int, values: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature max activation over ``[start0, end0)``.

        Returns ``(feature_ids, maxact)``. ``values`` defaults to this contig's own
        ``values`` array; pass an alternative only to score against a different store.
        """
        if values is None:
            values = self.values
        s = max(0, start0)
        e = min(self.n_pos, end0)
        if e <= s:
            return np.empty(0, np.uint16), np.empty(0, np.float32)
        lo, hi = self.indptr[s], self.indptr[e]
        if hi <= lo:
            return np.empty(0, np.uint16), np.empty(0, np.float32)
        idx = self.indices[lo:hi]
        val = values[lo:hi].astype(np.float32)
        order = np.argsort(idx, kind="stable")
        idx = idx[order]
        val = val[order]
        uniq, first = np.unique(idx, return_index=True)
        maxv = np.maximum.reduceat(val, first)
        return uniq, maxv
