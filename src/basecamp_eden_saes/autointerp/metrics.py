"""Pure numeric primitives for span-level annotation enrichment.

These operate directly on the full-sparse code store (per contig-strand CSR over
nucleotide positions) and the per-dictionary matched-negative background. They are
strand- and wraparound-aware and carry no I/O, no globals, and no experiment glue,
so they are safe to unit-test and reuse.

Code store CSR layout (one ``<contig>.{plus,minus}.npz`` per contig-strand):
    indptr   int64  [L+1]   row = nucleotide position; nnz per row = firing count
    indices  uint16 [nnz]   SAE feature ids (dict size F <= 65536 fits uint16)
    values   float16 [nnz]  activation magnitudes
Row ``r`` maps to genome coordinate ``r`` on the plus store and ``L-1-r`` on the
(reverse-complement-ordered) minus store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

F_DEFAULT = 32768
"""Default SAE dictionary size (features) for the shipped EDEN SAEs."""


def load_codes(path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one contig-strand CSR code file.

    Returns ``(indptr, indices, L)`` where ``L`` is the contig length in
    nucleotides (number of CSR rows).
    """
    z = np.load(path)
    indptr = z["indptr"]
    indices = z["indices"]
    return indptr, indices, int(indptr.shape[0] - 1)


def load_codes_v(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Like :func:`load_codes` but also returns the per-firing activation ``values``
    (float16 magnitudes), needed for the peak-firing cover rule."""
    z = np.load(path)
    indptr = z["indptr"]
    indices = z["indices"]
    values = z["values"]
    return indptr, indices, values, int(indptr.shape[0] - 1)


def span_peak_cover_features(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
    length_total: int,
    thresh: np.ndarray,
) -> np.ndarray:
    """Feature ids that fire STRONGLY inside a span (peak-firing cover rule).

    A feature covers the span if it has at least one in-span activation
    ``value >= thresh[feature]`` where ``thresh = fraction * feature_global_peak``.
    This surfaces a feature's identity (where its strongest firings land) rather
    than any weak off-target firing. ``start > end`` wraps the circular contig.
    """
    if start <= end:
        seg_i = indices[indptr[start]:indptr[end]]
        seg_v = values[indptr[start]:indptr[end]]
    else:
        seg_i = np.concatenate(
            [indices[indptr[start]:indptr[length_total]], indices[indptr[0]:indptr[end]]])
        seg_v = np.concatenate(
            [values[indptr[start]:indptr[length_total]], values[indptr[0]:indptr[end]]])
    if seg_i.size == 0:
        return np.empty(0, dtype=np.int64)
    mask = seg_v.astype(np.float32) >= thresh[seg_i]
    if not mask.any():
        return np.empty(0, dtype=np.int64)
    return np.unique(seg_i[mask])


def span_cover_features(
    indptr: np.ndarray,
    indices: np.ndarray,
    start: int,
    end: int,
    length_total: int,
    n_features: int,
    cover_frac: float,
    overlap: bool = False,
) -> np.ndarray:
    """Feature ids that cover a span.

    Two cover rules:
      * ``overlap=False`` (legacy): feature fires in at least ``cover_frac`` of the
        span's positions (the >=50%-cover rule, kept for 7B reproducibility).
      * ``overlap=True``: feature fires on >=1 position inside the span (the
        approved motif-aware rule; ``cover_frac`` no longer gates). Recall then =
        fraction of an annotation's spans the feature fires in.

    ``start``/``end`` are half-open code-position coordinates. When ``start > end``
    the span wraps the circular contig (used by the circular-shift null); the two
    arcs ``[start, length_total)`` and ``[0, end)`` are concatenated.
    """
    if start <= end:
        seg = indices[indptr[start]:indptr[end]]
        length = end - start
    else:  # wrapped: [start, L) U [0, end)
        seg = np.concatenate(
            [indices[indptr[start]:indptr[length_total]], indices[indptr[0]:indptr[end]]]
        )
        length = (length_total - start) + end
    if seg.size == 0 or length == 0:
        return np.empty(0, dtype=np.int64)
    counts = np.bincount(seg, minlength=n_features)
    thresh = 1.0 if overlap else cover_frac * length
    return np.where(counts >= thresh)[0]


def span_firing_positions(
    indptr: np.ndarray, indices: np.ndarray, start: int, end: int, feature: int
) -> int:
    """Number of positions in ``[start, end)`` where ``feature`` fires.

    This is the position-level (not span-level) count used for the ``lift``
    (PPV/prior) metric. Assumes ``start <= end`` (annotation spans do not wrap).
    """
    if end <= start:
        return 0
    seg = indices[indptr[start]:indptr[end]]
    if seg.size == 0:
        return 0
    return int(np.count_nonzero(seg == feature))


def gc_fraction(seq_bytes: np.ndarray, start: int, end: int) -> float:
    """GC fraction of ``seq_bytes[start:end]`` (uppercased contig as uint8)."""
    sub = seq_bytes[start:end]
    if sub.size == 0:
        return 0.0
    g = np.count_nonzero((sub == ord("G")) | (sub == ord("C")))
    return g / sub.size


def stratum_index(
    length: float,
    gc: float,
    len_edges_full: np.ndarray,
    gc_edges_full: np.ndarray,
    n_gc: int,
) -> int:
    """Map ``(length, gc)`` to a length x GC stratum id in ``[0, n_len*n_gc)``.

    ``len_edges_full`` / ``gc_edges_full`` are the full ``n+1`` bin-edge arrays
    stored in ``bg.npz``; digitizing against the interior edges keeps ids in range.
    Stratum id is ``len_bin * n_gc + gc_bin``.
    """
    lb = int(np.digitize(length, len_edges_full[1:-1]))
    gb = int(np.digitize(gc, gc_edges_full[1:-1]))
    return lb * n_gc + gb


def poisson_upper_tail(k: int, mu: float) -> float:
    """``P(X >= k)`` for ``X ~ Poisson(mu)`` (k = observed covered spans)."""
    from scipy.stats import poisson

    if k <= 0:
        return 1.0
    return float(poisson.sf(k - 1, mu))


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for a 1-D array of p-values."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=np.float64)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion ``k/n``."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def read_contig_sequences(fasta_path: str | Path) -> dict[str, np.ndarray]:
    """Parse a genome FASTA into ``{contig_id: uppercased uint8 sequence}``."""
    seqs: dict[str, np.ndarray] = {}
    cid: str | None = None
    buf: list[str] = []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    seqs[cid] = np.frombuffer("".join(buf).upper().encode(), dtype=np.uint8)
                cid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cid is not None:
        seqs[cid] = np.frombuffer("".join(buf).upper().encode(), dtype=np.uint8)
    return seqs


def span_to_code_coords(start: int, end: int, strand: str, length_total: int) -> tuple[int, int]:
    """Map a genome span ``[start, end)`` on ``strand`` to code-store coordinates.

    The minus store is reverse-complement-ordered, so a minus-strand span maps to
    ``[L-end, L-start)``. Plus-strand spans are unchanged.
    """
    if strand == "+":
        return start, end
    return length_total - end, length_total - start
