"""Unit tests for the autointerp span-metric primitives."""

from __future__ import annotations

import numpy as np

from basecamp_eden_saes.autointerp import metrics as M


def _csr_from_rows(rows: list[list[int]], n_features: int):
    """Build (indptr, indices) for a per-position feature-firing list."""
    indptr = [0]
    indices: list[int] = []
    for r in rows:
        indices.extend(r)
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.uint16)


def test_span_cover_features_threshold():
    # 4 positions; feature 5 fires at 3/4 positions, feature 9 at 1/4.
    indptr, indices = _csr_from_rows([[5], [5, 9], [5], []], n_features=16)
    # cover_frac 0.5 -> only feature 5 (3/4 >= 0.5) qualifies over [0,4)
    got = M.span_cover_features(indptr, indices, 0, 4, 4, 16, 0.5)
    assert set(got.tolist()) == {5}
    # cover_frac 0.2 -> both 5 (0.75) and 9 (0.25)
    got2 = M.span_cover_features(indptr, indices, 0, 4, 4, 16, 0.2)
    assert set(got2.tolist()) == {5, 9}


def test_span_cover_features_wraparound():
    # 5 positions; feature 3 fires at positions 4 and 0 (wraps).
    indptr, indices = _csr_from_rows([[3], [], [], [], [3]], n_features=8)
    got = M.span_cover_features(indptr, indices, 4, 1, 5, 8, 0.5)  # [4,5)U[0,1): 2 pos
    assert set(got.tolist()) == {3}


def test_span_firing_positions():
    indptr, indices = _csr_from_rows([[2], [2, 7], [7], [2]], n_features=8)
    assert M.span_firing_positions(indptr, indices, 0, 4, 2) == 3
    assert M.span_firing_positions(indptr, indices, 0, 4, 7) == 2
    assert M.span_firing_positions(indptr, indices, 1, 1, 2) == 0  # empty span


def test_gc_fraction():
    seq = np.frombuffer(b"ACGTGGCC", dtype=np.uint8)
    assert abs(M.gc_fraction(seq, 0, 8) - 6 / 8) < 1e-9
    assert M.gc_fraction(seq, 0, 0) == 0.0


def test_stratum_index_layout():
    len_edges = np.array([0, 30, 60, 1e9])   # 3 length bins
    gc_edges = np.array([0.0, 0.5, 1.01])    # 2 gc bins
    n_gc = 2
    # length 10 -> len bin 0; gc 0.7 -> gc bin 1 -> 0*2+1 = 1
    assert M.stratum_index(10, 0.7, len_edges, gc_edges, n_gc) == 1
    # length 100 -> len bin 2; gc 0.1 -> gc bin 0 -> 2*2+0 = 4
    assert M.stratum_index(100, 0.1, len_edges, gc_edges, n_gc) == 4


def test_span_to_code_coords():
    assert M.span_to_code_coords(10, 20, "+", 100) == (10, 20)
    # minus strand: reverse-complement-ordered store
    assert M.span_to_code_coords(10, 20, "-", 100) == (80, 90)


def test_bh_qvalues_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.5, 0.9])
    q = M.bh_qvalues(p)
    assert q.shape == p.shape
    assert np.all(q >= 0) and np.all(q <= 1)
    # q is monotone non-decreasing in the p-order
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_wilson_interval_contains_point():
    lo, hi = M.wilson_interval(37, 152)
    assert 0 <= lo < 37 / 152 < hi <= 1


def test_poisson_upper_tail_bounds():
    assert M.poisson_upper_tail(0, 5.0) == 1.0
    # observing many more than expected -> small tail probability
    assert M.poisson_upper_tail(37, 0.05) < 1e-6
