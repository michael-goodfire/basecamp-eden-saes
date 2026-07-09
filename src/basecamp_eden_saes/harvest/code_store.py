"""Full-sparse SAE code harvest over the 152-genome annotation panel.

Runs EDEN over every panel position (both strands, non-overlapping 4096-nt
windows that tile each contig exactly), captures the layer-``hook_layer`` decoder
output (== HF ``hidden_states[hook_layer+1]``; confirmed to reproduce the panel's
density at ``hook_layer=28``), and encodes it with one or more BatchTopK SAEs
using their native learned-threshold gating (``sae.features``). Stores the full
sparse codes (every firing, every position) per contig-strand as CSR-style npz.

Store layout (one dir per SAE dictionary, under ``<out>/<sae_name>``)::

    <acc>/<contig>.<strand>.npz  indptr int64[n_pos+1], indices uint16[nnz], values float16[nnz]
    layout.json  [{acc,contig,strand,length,nnz}]
    meta.json    model/sae/hook + n_positions
    fire_count.npy / density.npy / mean_act.npy

Row ``r`` in a ``(contig, strand)`` segment maps to genome coord: plus ``r``;
minus ``length - 1 - r``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from basecamp_eden_saes import config
from basecamp_eden_saes.annotations import panel

logger = logging.getLogger(__name__)

WIN = 4096


def seq_to_ids(s: str) -> np.ndarray:
    """Byte-level token ids (int64) for an uppercase DNA string."""
    return np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8).astype(np.int64)


class LayerCapture:
    """Forward hook that stashes one decoder layer's output tensor."""

    def __init__(self, model, layer_idx: int):
        self.acts: torch.Tensor | None = None
        self.h = model.model.layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out[0] if isinstance(out, tuple) else out

    def close(self) -> None:
        self.h.remove()


@torch.no_grad()
def encode_segment(
    model,
    cap: LayerCapture,
    saes: dict,
    ids: np.ndarray,
    hook_layer: int,
    batch_windows: int,
    dev: str,
) -> tuple[dict, int]:
    """Encode one strand of one contig with each SAE.

    Returns ``(out, npos)`` where ``out`` maps ``sae_name`` to
    ``(indptr, indices uint16, values float16, fire, act_sum)``.
    """
    starts = list(range(0, len(ids), WIN))
    per = {
        name: {
            "indptr": np.zeros(len(ids) + 1, dtype=np.int64),
            "idx": [],
            "val": [],
            "fire": np.zeros(sae.d_sae, dtype=np.int64),
            "act": np.zeros(sae.d_sae, dtype=np.float64),
        }
        for name, sae in saes.items()
    }
    rows = {name: 0 for name in saes}
    for b in range(0, len(starts), batch_windows):
        cs = starts[b : b + batch_windows]
        lens = [min(WIN, len(ids) - s) for s in cs]
        maxlen = max(lens)
        arr = np.zeros((len(cs), maxlen), dtype=np.int64)
        for i, s in enumerate(cs):
            arr[i, : lens[i]] = ids[s : s + lens[i]]
        inp = torch.from_numpy(arr).to(dev)
        model(input_ids=inp, use_cache=False)
        acts = cap.acts  # [B, L, d]
        for name, sae in saes.items():
            st = per[name]
            row = rows[name]
            F = sae.d_sae
            for i, L in enumerate(lens):
                feats = sae.features(acts[i, :L].float())  # [L, F]
                nz = feats > 0
                counts = nz.sum(dim=1).to("cpu").numpy().astype(np.int64)
                ri = torch.nonzero(nz, as_tuple=False)
                fids = ri[:, 1].to(torch.int32).cpu().numpy()
                vals = feats[nz].to(torch.float16).cpu().numpy()
                st["idx"].append(fids.astype(np.uint16))
                st["val"].append(vals)
                st["indptr"][row + 1 : row + 1 + L] = counts
                st["fire"] += np.bincount(fids, minlength=F)
                np.add.at(st["act"], fids, vals.astype(np.float64))
                row += L
            rows[name] = row
    out = {}
    for name in saes:
        st = per[name]
        np.cumsum(st["indptr"], out=st["indptr"])
        idx = np.concatenate(st["idx"]) if st["idx"] else np.zeros(0, np.uint16)
        val = np.concatenate(st["val"]) if st["val"] else np.zeros(0, np.float16)
        out[name] = (st["indptr"], idx, val, st["fire"], st["act"])
    return out, len(ids)


def load_model(model_path: str, dev: str):
    """Load an EDEN causal LM in bf16 with SDPA attention on ``dev``."""
    from transformers import AutoModelForCausalLM

    return (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``bes-harvest-codes`` (full-sparse code store)."""
    ap = argparse.ArgumentParser(description="Full-sparse SAE code harvest.")
    ap.add_argument("--model", required=True, help="EDEN model path")
    ap.add_argument(
        "--saes", required=True, help="name=path,name=path (share the forward)"
    )
    ap.add_argument("--out", required=True, help="base dir; each SAE writes <out>/<name>")
    ap.add_argument("--accs", default="", help="comma list or @file; empty = all panel")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument(
        "--hook-layer", type=int, default=28, help="decoder layer idx (hs[hook+1])"
    )
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--strands", default="both", choices=["both", "plus"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if not args.accs:
        accs = config.accessions()[args.shard :: args.nshards]
    elif args.accs.startswith("@"):
        accs = [x.strip() for x in open(args.accs[1:]) if x.strip()]
    else:
        accs = args.accs.split(",")

    dev = args.device
    from goodfire_core.saes.loader import load_sae

    saes = {}
    for spec in args.saes.split(","):
        name, path = spec.split("=", 1)
        saes[name] = load_sae(path).to(dev).eval()
    model = load_model(args.model, dev)
    cap = LayerCapture(model, args.hook_layer)

    state: dict = {}
    for name, sae in saes.items():
        od = Path(args.out) / name
        od.mkdir(parents=True, exist_ok=True)
        # per-shard layout file: shards run concurrently and a shared layout.json
        # would race on write. The final layout is rebuilt from the .npz files on
        # disk. This per-shard file is for resume only.
        lp = od / f"layout.shard{args.shard}.json"
        layout = json.loads(lp.read_text()) if lp.exists() else []
        state[name] = {
            "od": od,
            "layout": layout,
            "lp": lp,
            "done": {(e["acc"], e["contig"], e["strand"]) for e in layout},
            "fire": np.zeros(sae.d_sae, np.int64),
            "act": np.zeros(sae.d_sae, np.float64),
            "npos": 0,
            "path": None,
        }
    for spec in args.saes.split(","):
        name, path = spec.split("=", 1)
        state[name]["path"] = path

    nstr = 2 if args.strands == "both" else 1
    done_pos = 0

    strands = ["+", "-"] if args.strands == "both" else ["+"]
    t0 = time.time()
    for acc in accs:
        seqs = panel.read_fasta(f"{panel.genome_dir(acc)}/genomic.fna")
        for name in saes:
            (state[name]["od"] / acc).mkdir(parents=True, exist_ok=True)
        for contig, seq in seqs.items():
            for strand in strands:
                todo = [n for n in saes if (acc, contig, strand) not in state[n]["done"]]
                if not todo:
                    continue
                s = seq if strand == "+" else panel.revcomp(seq)
                ids = seq_to_ids(s)
                res, L = encode_segment(
                    model,
                    cap,
                    {n: saes[n] for n in todo},
                    ids,
                    args.hook_layer,
                    args.batch,
                    dev,
                )
                slabel = "plus" if strand == "+" else "minus"
                for name in todo:
                    indptr, idx, val, fire, act = res[name]
                    st = state[name]
                    np.savez(
                        st["od"] / acc / f"{contig}.{slabel}.npz",
                        indptr=indptr,
                        indices=idx,
                        values=val,
                    )
                    st["layout"].append(
                        {
                            "acc": acc,
                            "contig": contig,
                            "strand": strand,
                            "length": L,
                            "nnz": int(len(idx)),
                        }
                    )
                    Path(st["lp"]).write_text(json.dumps(st["layout"]))
                    st["fire"] += fire
                    st["act"] += act
                    st["npos"] += L
            done_pos += nstr * len(seq)
            logger.info("%s %s done  %d positions  %.1fs", acc, contig, done_pos, time.time() - t0)
    for name, sae in saes.items():
        st = state[name]
        (st["od"] / "meta.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "sae": st["path"],
                    "hook_layer": args.hook_layer,
                    "win": WIN,
                    "F": int(sae.d_sae),
                    "n_positions": int(st["npos"]),
                    "strands": args.strands,
                    "name": name,
                }
            )
        )
        np.save(st["od"] / "fire_count.npy", st["fire"])
        np.save(st["od"] / "density.npy", st["fire"] / max(st["npos"], 1))
        np.save(
            st["od"] / "mean_act.npy",
            np.where(st["fire"] > 0, st["act"] / np.maximum(st["fire"], 1), 0.0),
        )
    cap.close()
    logger.info("HARVEST DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
