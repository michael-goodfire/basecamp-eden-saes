"""Per-genome POSITION-level annotation coverage over the code store.

Companion to :mod:`join_cover` (which is span-level). For one genome + one SAE
dictionary and a set of annotation spans, accumulate, per annotation key ``a``:

    ``obs::a``  int64 [F]  total per-feature firings at positions inside ``a`` spans
    ``pos::a``  int64 []   total annotation positions (sum of span lengths on the
                           span's reading strand; matches the position-level ``A``)

Summed across genomes these give ``A[a] = sum pos`` and ``obs[f,a]``, from which
the atlas position-level metrics are formed (see ``build_pos_rates``):

    pos_rate = (obs + 0.5) / A            (Haldane)
    bg_rate  = fire_count[f] / total_positions   (feature global density)
    fold     = pos_rate / bg_rate         (position-level, capped at 999 in display)
    ppv      = obs / fire_count ; prior = A / total_positions ; lift = ppv / prior

This is a firing-count accumulation (``np.bincount`` over the CSR feature ids in a
span), not a cover test, so it carries no null model and no bg.npz.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M
from .join_cover import read_spans


def cover_positions_genome(
    accession: str,
    codes_dir: str | Path,
    spans: list[tuple[str, str, str, int, int]],
    n_features: int = M.F_DEFAULT,
) -> tuple[dict, dict]:
    """Return ``(obs, pos)``: obs[a] = int64[F] firing counts, pos[a] = int positions."""
    codes_dir = Path(codes_dir)
    obs: dict[str, np.ndarray] = {}
    pos: dict[str, int] = defaultdict(int)

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for sp in spans:
        groups[(sp[1], sp[2])].append(sp)

    for (contig, strand), grp in groups.items():
        fn = codes_dir / accession / f"{contig}.{'plus' if strand == '+' else 'minus'}.npz"
        if not fn.exists():
            continue
        indptr, indices, L = M.load_codes(fn)
        for a, _c, st, s, e in grp:
            if s < 0 or e > L or s >= e:
                continue
            cs, ce = M.span_to_code_coords(s, e, st, L)
            if cs <= ce:
                seg = indices[indptr[cs]:indptr[ce]]
            else:  # wrapped (shouldn't happen for real annotations)
                seg = np.concatenate([indices[indptr[cs]:indptr[L]], indices[indptr[0]:indptr[ce]]])
            if a not in obs:
                obs[a] = np.zeros(n_features, np.int64)
            if seg.size:
                obs[a] += np.bincount(seg, minlength=n_features).astype(np.int64)
            pos[a] += (e - s)
    return obs, dict(pos)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-genome position-level annotation firing partial.")
    ap.add_argument("--acc", required=True)
    ap.add_argument("--codes-dir", required=True)
    ap.add_argument("--spans", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output partial .npz path")
    ap.add_argument("--F", type=int, default=M.F_DEFAULT)
    args = ap.parse_args(argv)

    spans = read_spans(args.spans)
    obs, pos = cover_positions_genome(args.acc, args.codes_dir, spans, n_features=args.F)
    arrays: dict[str, np.ndarray] = {}
    for a in obs:
        arrays[f"obs::{a}"] = obs[a]
    np.savez_compressed(args.out, **arrays)
    json.dump(pos, open(args.out + ".pos.json", "w"))
    print(f"{args.acc}: {len(spans)} spans, {len(obs)} annotations -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
