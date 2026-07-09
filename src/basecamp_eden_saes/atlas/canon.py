"""Canonical TED/CATH fold unit for the EDEN feature viewer atlas.

The bug (diagnosed in #52): TED labels each domain at whatever CATH-style level
its classifier is confident about, so #40's raw TED namespace mixes the 4-level
homologous superfamily (`1.10.760.10`), the 3-level topology (`1.10.760`, a tiny
partial-resolution subset), and multi-domain comma strings (`1.10.760,1.10.760`)
as SEPARATE keys. The atlas joined a feature to a fold on the exact raw string,
so it scored features against a partial-resolution fragment of their true fold,
understating recall and inflating a fragile precision ratio.

Fix (matches #52's `resolve_superfamily`, rank.py:48-61): group raw codes by the
3-level topology, then within each topology report the DOMINANT homologous
superfamily (the raw code with the most covered spans, `n_covered`). That single
dominant superfamily's own recall / precision / enrichment are the honest,
granularity-stable metrics; the partial-resolution rows are a separate tiny
bucket and are dropped from the collapsed fold. CATH keys are already emitted at
the 4-level superfamily (`build_cath_layer.py` uses the CATH `sfam`), so they are
already canonical.

This module is pure logic with no filesystem paths; the canonicalization and
metric-selection rules here are relied on by a downstream reproduction check and
must not change.
"""
from __future__ import annotations

import json


def fold_of(code: str) -> str:
    """Single raw CATH/TED code -> 3-level topology unit (matches #50 paths.fold_of)."""
    return ".".join(code.split(".")[:3])


def group_topology(code: str) -> str:
    """Topology-grouping key for a raw TED code, comma-aware.

    #40 TED emits multi-domain regions as comma-joined codes (e.g.
    `1.10.760,1.10.760`). #50's `fold_of` splits on `.` only, so a comma code
    yields the malformed `1.10.760,1` and never groups with its parent topology,
    leaving the fragment as a spurious separate fold row. Here:
      - a single or homogeneous multi-domain code (all constituents share a
        topology) groups under that topology -> it is absorbed into the topology's
        dominant superfamily;
      - a heterogeneous multi-domain code (constituents span different topologies,
        ~1% of TED codes, e.g. `1.25.10,1.25.40`) has no single fold, so it keeps
        its own raw code as the group key (stays an honest multi-fold row).
    """
    tops = {fold_of(p) for p in code.split(",")}
    return tops.pop() if len(tops) == 1 else code


def bare(ann_id: str) -> str:
    """`ted|1.10.760.10` / `cath|1.10.760.10` -> `1.10.760.10`."""
    return ann_id.split("|", 1)[-1]


def load_enrichment(path: str) -> dict[tuple[int, str], dict]:
    """#40 reduced span_enrichment.json -> {(feature:int, bare_code:str): record}.

    record has: n_spans, n_covered, recall, recall_lo, recall_hi,
    precision_fold, null_fold, p, q, label.
    """
    out: dict[tuple[int, str], dict] = {}
    with open(path) as fh:
        for e in json.load(fh):
            out[(int(e["feature"]), bare(e["ann"]))] = e
    return out


def is_superfamily(code: str) -> bool:
    """A distinct 4-level homologous-superfamily code (single, no comma)."""
    return "," not in code and len(code.split(".")) >= 4


def dominant_superfamily(
    feat: int, topology: str, enr: dict
) -> tuple[str | None, dict | None]:
    """Dominant #40 superfamily under a 3-level topology, by covered spans.

    Verbatim logic of #52 rank.best_enrichment / build_evidence.resolve_superfamily
    (candidates = every #40 code under the topology; pick max n_covered). Used to
    reproduce the #52 gallery gate numbers. Returns (bare_code, record) or
    (None, None).
    """
    segs = topology.split(".")
    cands = [(code, e) for (f, code), e in enr.items()
             if f == feat and code.split(".")[:len(segs)] == segs]
    if not cands:
        return None, None
    code, e = max(cands, key=lambda ce: ce[1]["n_covered"])
    return code, e


def dominant_superfamily_4level(
    feat: int, topology: str, enr: dict
) -> tuple[str | None, dict | None]:
    """Dominant 4-level superfamily under a topology (partials excluded).

    Used atlas-wide to decide which superfamily a partial-resolution code is
    absorbed into. Restricting candidates to true 4-level superfamilies keeps
    distinct superfamilies that merely share a topology (e.g. 3.40.50.80 vs
    3.40.50.720) separate, instead of collapsing the whole topology to one fold.
    """
    segs = topology.split(".")
    cands = [(code, e) for (f, code), e in enr.items()
             if f == feat and is_superfamily(code) and code.split(".")[:len(segs)] == segs]
    if not cands:
        return None, None
    code, e = max(cands, key=lambda ce: ce[1]["n_covered"])
    return code, e


def canonical_target(code: str, feat: int, enr: dict) -> tuple[str, bool]:
    """Canonical fold code a raw TED code maps to (atlas-wide rule).

    - Distinct 4-level superfamily (`1.10.760.10`): kept as itself.
    - Heterogeneous multi-domain comma (`1.25.10,1.25.40`): kept as itself (an
      honest multi-fold region; no single parent).
    - Partial-resolution code (3-level `1.10.760`, or homogeneous comma
      `1.10.760,1.10.760`): absorbed into the dominant 4-level superfamily under
      its topology; if the feature has no 4-level superfamily under that topology
      in #40, the code is kept as itself (topology is the best available unit).

    Returns (target_code, absorbed: bool). absorbed=True means this raw code is a
    partial that should be dropped in favour of target_code.
    """
    tops = {fold_of(p) for p in code.split(",")}
    if len(tops) > 1:
        return code, False            # heterogeneous multi-fold: keep
    if is_superfamily(code):
        return code, False            # distinct superfamily: keep
    topo = tops.pop()
    dom, _ = dominant_superfamily_4level(feat, topo, enr)
    if dom is None or dom == code:
        return code, False            # no parent superfamily: keep partial
    return dom, True                  # partial -> absorb into superfamily
