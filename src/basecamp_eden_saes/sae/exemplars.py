"""Max-activating sequence windows for a handful of features of a selected SAE.

Concrete specimens to sit beside the metrics (no labels yet -- autointerp is a
later step). Picks ~15 features spread across the firing-density range (plus the
densest "soak-up" feature), streams held-out windows through EDEN + the SAE, and
records, per feature, the top windows by peak activation with the local
nucleotide context.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
from pathlib import Path

import numpy as np
import torch

from goodfire_core.saes.loader import load_sae

from basecamp_eden_saes import config
from basecamp_eden_saes.corpus.build import CorpusReader
from basecamp_eden_saes.harvest.eden import EdenPartialForward

logger = logging.getLogger(__name__)


def pick_features(density: np.ndarray, n: int) -> list[int]:
    """Densest feature + a log-spread of alive features across the density range."""
    alive = np.where(density > 0)[0]
    if len(alive) == 0:
        return []
    picks = [int(density.argmax())]  # the dense soak-up feature
    d_alive = density[alive]
    order = alive[np.argsort(d_alive)]  # ascending density
    # evenly spaced ranks across the alive set (log in rank space)
    ranks = np.unique(np.geomspace(1, len(order), num=n * 2).astype(int)) - 1
    for r in ranks:
        f = int(order[r])
        if f not in picks:
            picks.append(f)
        if len(picks) >= n:
            break
    return picks[:n]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: dump max-activating exemplars for selected SAE features."""
    paths = config.data_paths()
    p = argparse.ArgumentParser(description="Max-activating exemplars for an SAE.")
    p.add_argument("--model", required=True, choices=list(paths.models))
    p.add_argument("--sae", required=True)
    p.add_argument("--density", required=True)
    p.add_argument(
        "--corpus", default=None, help="corpus dir (default: config.data_paths().corpus)"
    )
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--out", required=True)
    p.add_argument("--n-features", type=int, default=15)
    p.add_argument("--n-windows", type=int, default=4000)
    p.add_argument("--topn", type=int, default=6)
    p.add_argument("--ctx", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    corpus = Path(args.corpus) if args.corpus else paths.corpus
    device = args.device
    eden = EdenPartialForward(
        paths.models[args.model], layer=args.layer, device=device, max_seq=4096
    )
    sae = load_sae(args.sae).to(device).eval()
    reader = CorpusReader(corpus)
    density = np.load(args.density)
    feats = pick_features(density, args.n_features)
    feat_t = torch.tensor(feats, device=device)
    logger.info("[%s] features: %s", args.model, feats)

    # held-out tail windows (same split convention as the sweep)
    n = len(reader)
    held = np.arange(n)[n - min(4096, n // 10) :]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(held)
    held = held[: args.n_windows]

    heaps: dict[int, list] = {f: [] for f in feats}  # min-heap of (act, uid, payload)
    uid = 0
    with torch.no_grad():
        for wi in held:
            w = reader.get(int(wi))
            ids, attn, keep = eden.build_batch([w])
            acts = eden.residual(ids, attn)[keep].float()  # [S, d]; fp32 for the SAE
            codes = sae.encode(acts)[:, feat_t].float().cpu().numpy()  # [S, F]
            for j, f in enumerate(feats):
                col = codes[:, j]
                pos = int(col.argmax())
                act = float(col[pos])
                if act <= 0:
                    continue
                h = heaps[f]
                if len(h) < args.topn or act > h[0][0]:
                    lo = max(0, pos - args.ctx)
                    hi = min(len(w), pos + args.ctx + 1)
                    ctx = "".join(chr(int(t)) for t in w[lo:hi])
                    payload = {
                        "window": int(wi),
                        "pos": pos,
                        "act": act,
                        "context": ctx,
                        "ctx_start": lo,
                        "hit": pos - lo,
                    }
                    heapq.heappush(h, (act, uid, payload))
                    uid += 1
                    if len(h) > args.topn:
                        heapq.heappop(h)

    out = {"model": args.model, "layer": args.layer, "sae": args.sae, "features": {}}
    for f in feats:
        items = sorted(heaps[f], key=lambda x: -x[0])
        out["features"][str(f)] = {
            "density": float(density[f]),
            "exemplars": [it[2] for it in items],
        }
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info("[%s] wrote %s (%d features)", args.model, args.out, len(feats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
