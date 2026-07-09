"""DEFERRED gap-fold pipeline: fetch gap-protein sequences + ESMFold gap folding.

This stage covers the atlas exemplar proteins that have no AlphaFold-DB model
(no UniProt mapping, or UniProt-mapped but AFDB-missing). It was **DEFERRED** and
was not run at scale; the code ships for reproducibility. AFDB coverage
(:mod:`.afdb`) already accounts for the large majority of coding exemplars.

Two stages, exposed as the ``fetch`` and ``fold`` subcommands:

``fetch`` (:func:`fetch_sequences`): batches the gap RefSeq accessions through
NCBI EFetch (FASTA) into ``sequences.json``.

``fold`` (:func:`fold_gap`): loads ``Synthyra/ESMFold2-Fast`` once and folds each
sequence in single-sequence mode into a shared ``af_cache`` mmCIF (pLDDT in the
B-factor column), sharded across a GPU array via ``--shard/--nshards``.

To run (deferred; requires a GPU + ``transformers``)::

    python -m basecamp_eden_saes.structure.esmfold fetch \\
        --gap <afdb_gap.json> --out <seq_dir>
    python -m basecamp_eden_saes.structure.esmfold fold \\
        --sequences <seq_dir>/sequences.json --out <fold_dir> \\
        --shard $I --nshards $N
"""

from __future__ import annotations

import argparse
import json
import os
import time

import gemmi
import requests

from basecamp_eden_saes import config

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# --- stage 1: sequence fetch -------------------------------------------------


def parse_fasta(text: str) -> dict[str, str]:
    """Map every accession token in each FASTA header to its sequence."""
    out: dict[str, str] = {}
    buf: list[str] = []
    hdr_accs: list[str] = []

    def flush() -> None:
        if hdr_accs and buf:
            seq = "".join(buf)
            for a in hdr_accs:
                out[a] = seq

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            buf = []
            tok = line[1:].split()[0]  # ">WP_000000.1 desc" -> WP_000000.1
            hdr_accs = [tok]
            if "." in tok:  # also index the version-stripped accession
                hdr_accs.append(tok.split(".")[0])
        else:
            buf.append(line.strip())
    flush()
    return out


def efetch_batch(
    ids: list[str], session: requests.Session, api_key: str | None = None
) -> dict[str, str]:
    """POST a batch of protein accessions to EFetch, return ``{acc: seq}``."""
    params = {"db": "protein", "rettype": "fasta", "retmode": "text", "id": ",".join(ids)}
    if api_key:
        params["api_key"] = api_key
    for attempt in range(4):
        try:
            r = session.post(f"{EUTILS}/efetch.fcgi", data=params, timeout=120)
            if r.ok and r.text.startswith(">"):
                return parse_fasta(r.text)
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return {}


def fetch_sequences(argv: list[str] | None = None) -> int:
    """Fetch full protein sequences for the gap proteins via NCBI EFetch."""
    ap = argparse.ArgumentParser(description=fetch_sequences.__doc__)
    ap.add_argument("--gap", required=True, help="JSON {prot_id: {...}} of proteins to fetch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--api-key",
        default=os.environ.get("NCBI_API_KEY"),
        help="NCBI API key (defaults to $NCBI_API_KEY)",
    )
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    gap = json.load(open(args.gap))
    ids = list(gap.keys())
    if args.limit:
        ids = ids[: args.limit]
    rate = 0.12 if args.api_key else 0.36  # ~8/s with key, ~3/s without
    print(
        f"fetching {len(ids)} sequences, batch={args.batch}, "
        f"api_key={'yes' if args.api_key else 'no'}",
        flush=True,
    )

    session = requests.Session()
    seqs: dict[str, str] = {}
    for i in range(0, len(ids), args.batch):
        batch = ids[i : i + args.batch]
        got = efetch_batch(batch, session, args.api_key)
        for pid in batch:  # map back to our prot_ids (which carry a version)
            s = got.get(pid) or got.get(pid.split(".")[0])
            if s:
                seqs[pid] = s
        time.sleep(rate)

    failures = [pid for pid in ids if pid not in seqs]
    with open(os.path.join(args.out, "sequences.json"), "w") as f:
        json.dump(seqs, f)
    with open(os.path.join(args.out, "seq_failures.json"), "w") as f:
        json.dump(failures, f, indent=1)
    lens = sorted(len(s) for s in seqs.values())
    stats = {
        "requested": len(ids),
        "resolved": len(seqs),
        "failed": len(failures),
        "len_min": lens[0] if lens else None,
        "len_median": lens[len(lens) // 2] if lens else None,
        "len_max": lens[-1] if lens else None,
        "n_over_1024": sum(1 for x in lens if x > 1024),
        "n_over_2048": sum(1 for x in lens if x > 2048),
    }
    with open(os.path.join(args.out, "seq_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("===== SEQ STATS =====", flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    return 0


# --- stage 2: ESMFold gap folding -------------------------------------------


def sanitize_key(pid: str) -> str:
    """Filesystem/URL-safe CIF key; prefixed so it never collides with AFDB keys."""
    return "ef2_" + pid.replace("/", "_")


def mean_plddt_from_cif(path: str) -> float | None:
    """Mean per-residue pLDDT (0-100) from the CA B-factor column of a CIF."""
    try:
        st = gemmi.read_structure(path)
        vals = []
        for model in st:
            for chain in model:
                for res in chain:
                    ca = res.find_atom("CA", "*")
                    if ca is not None:
                        vals.append(ca.b_iso)
            break
        return round(sum(vals) / len(vals), 2) if vals else None
    except Exception:
        return None


def ensure_data_prefix(path: str) -> bool:
    """Guarantee a CIF starts with the ``data_`` magic (rewrite via gemmi if not)."""
    with open(path) as f:
        head = f.read(64)
    if head.lstrip().startswith("data_"):
        return True
    try:
        st = gemmi.read_structure(path)
        doc = st.make_mmcif_document()
        doc.write_file(path)
        with open(path) as f:
            return f.read(64).lstrip().startswith("data_")
    except Exception:
        return False


def fold_gap(argv: list[str] | None = None) -> int:
    """Fold gap-protein sequences with ESMFold into shared af_cache CIFs."""
    ap = argparse.ArgumentParser(description=fold_gap.__doc__)
    ap.add_argument("--sequences", required=True, help="sequences.json from fetch")
    ap.add_argument(
        "--cache",
        default=None,
        help="shared af_cache dir for CIFs (default: config.data_paths().af_cache)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Synthyra/ESMFold2-Fast")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument(
        "--maxlen",
        type=int,
        default=2000,
        help="truncate sequences longer than this (H100 context ceiling ~2000 aa)",
    )
    ap.add_argument("--limit", type=int, default=0, help="only first N (smoke)")
    ap.add_argument("--num-loops", type=int, default=3)
    ap.add_argument("--num-sampling-steps", type=int, default=50)
    args = ap.parse_args(argv)
    cache = args.cache or str(config.data_paths().af_cache)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    import torch
    from transformers import AutoModel

    seqs = json.load(open(args.sequences))
    items = sorted(seqs.items())  # deterministic order
    if args.limit:
        items = items[: args.limit]
    items = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] folding {len(items)} proteins", flush=True)

    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).eval().cuda()

    out: dict[str, dict] = {}
    fails: dict[str, str] = {}
    total = len(items)
    for i, (pid, seq) in enumerate(items):
        truncated = False
        s = seq.rstrip("*")
        if len(s) > args.maxlen:
            s = s[: args.maxlen]
            truncated = True
        key = sanitize_key(pid)
        dest = os.path.join(cache, f"{key}.cif")
        try:
            with torch.no_grad():
                res = model.fold_protein(
                    s,
                    num_loops=args.num_loops,
                    num_sampling_steps=args.num_sampling_steps,
                    num_diffusion_samples=1,
                    seed=0,
                )
            model.save_as_cif(res, dest)
            if not ensure_data_prefix(dest):
                raise RuntimeError("CIF missing data_ prefix after rewrite")
            plddt = mean_plddt_from_cif(dest)  # 0-100 scale, matches AFDB records
            ptm = float(getattr(res, "ptm", float("nan")))
            out[pid] = {
                "source": "esmfold2",
                "key": key,
                "plddt": plddt,
                "ptm": round(ptm, 3) if ptm == ptm else None,
                "n_res": len(s),
                "truncated": truncated,
            }
        except Exception as e:
            fails[pid] = str(e)[:300]
            # recover CUDA context after an OOM so later (shorter) folds still run
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    with open(os.path.join(args.out, f"struct_esmfold_shard_{args.shard}.json"), "w") as f:
        json.dump(out, f)
    with open(os.path.join(args.out, f"esmfold_fail_shard_{args.shard}.json"), "w") as f:
        json.dump(fails, f, indent=1)
    print(f"[shard {args.shard}] done: ok={len(out)} fail={len(fails)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the ``fetch`` or ``fold`` subcommand (DEFERRED pipeline)."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", add_help=False)
    sub.add_parser("fold", add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "fetch":
        return fetch_sequences(rest)
    if args.cmd == "fold":
        return fold_gap(rest)
    ap.error(f"unknown subcommand {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
