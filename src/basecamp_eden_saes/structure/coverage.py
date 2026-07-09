"""AlphaFold-DB structure coverage of the SAE feature-atlas exemplar proteins.

Two stages, exposed as the ``enumerate`` and ``index`` subcommands:

``enumerate`` (:func:`enumerate_proteins`): scans every per-feature atlas JSON
across all three dictionaries, records each unique coding exemplar protein with
metadata, counts coding/non-coding exemplars per dict, and joins against
``refseq2uniprot.json`` to split coding proteins into UniProt-mapped (AFDB path)
vs unmapped (ESMFold gap).

``index`` (:func:`build_struct_index`): merges the AFDB records (and any ESMFold
gap-fold shards) into a single ``struct_index.json`` and computes the
per-dictionary structure-coverage stat.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import orjson

DICT_NAMES = ["og2", "bcr_k64", "bcr_k16"]


def scan_file(path: str) -> tuple[Counter, int, int, dict]:
    """Scan one feature JSON.

    Returns ``(counts, noncoding, total, meta)`` where ``counts`` maps coding
    ``prot_id`` -> coding-exemplar count, ``noncoding``/``total`` are exemplar
    tallies, and ``meta`` maps ``prot_id`` -> ``(product, st, acc, contig, gs, ge)``.
    """
    counts: Counter = Counter()
    meta: dict = {}
    noncoding = 0
    total = 0
    with open(path, "rb") as f:
        d = orjson.loads(f.read())
    for e in d.get("exemplars", []):
        total += 1
        p = e.get("prot")
        pid = p.get("id") if isinstance(p, dict) else None
        if not pid:
            noncoding += 1
            continue
        counts[pid] += 1
        if pid not in meta:
            meta[pid] = (
                p.get("product") or "",
                p.get("st") or "",
                e.get("acc") or "",
                e.get("contig") or "",
                p.get("gs"),
                p.get("ge"),
            )
    return counts, noncoding, total, meta


def enumerate_proteins(argv: list[str] | None = None) -> int:
    """Enumerate atlas exemplar proteins and split AFDB-mapped vs gap."""
    ap = argparse.ArgumentParser(description=enumerate_proteins.__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--viewer-root",
        required=True,
        help="dir containing og2/bcr_k64/bcr_k16 subdirs each with a feature/ dir",
    )
    ap.add_argument("--refseq2uni", required=True, help="refseq2uniprot.json")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="scan only N files per dict (smoke)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    dict_dirs = {d: os.path.join(args.viewer_root, d, "feature") for d in DICT_NAMES}

    r2u_raw = json.load(open(args.refseq2uni))
    # value = [[uniprot, taxid], ...]; take first accession as canonical
    r2u: dict[str, str] = {}
    for k, v in r2u_raw.items():
        if v and isinstance(v, list) and v[0]:
            r2u[k] = v[0][0]

    per_dict_counts: dict[str, dict[str, int]] = {}
    all_meta: dict[str, tuple] = {}
    noncoding_by_dict: dict[str, int] = {}
    total_by_dict: dict[str, int] = {}

    for dct, fdir in dict_dirs.items():
        files = sorted(glob.glob(os.path.join(fdir, "latent_*.json")))
        if args.limit:
            files = files[: args.limit]
        counts: Counter = Counter()
        noncoding = 0
        total = 0
        with Pool(args.procs) as pool:
            for c, nc, tot, meta in pool.imap_unordered(scan_file, files, chunksize=64):
                counts.update(c)
                noncoding += nc
                total += tot
                for pid, m in meta.items():
                    if pid not in all_meta:
                        all_meta[pid] = m
        per_dict_counts[dct] = dict(counts)
        noncoding_by_dict[dct] = noncoding
        total_by_dict[dct] = total

    # build proteins.json with uniprot join
    proteins: dict[str, dict] = {}
    for pid, (product, st, acc, contig, gs, ge) in all_meta.items():
        proteins[pid] = {
            "id": pid,
            "product": product,
            "st": st,
            "acc": acc,
            "contig": contig,
            "gs": gs,
            "ge": ge,
            "uni": r2u.get(pid),
        }

    n_unique = len(proteins)
    n_mapped = sum(1 for p in proteins.values() if p["uni"])
    n_gap = n_unique - n_mapped
    gap_prefix: Counter = Counter()
    for p in proteins.values():
        if not p["uni"]:
            pid = p["id"]
            gap_prefix[pid.split("_")[0] if "_" in pid else "other"] += 1

    dict_cov: dict[str, dict] = {}
    for dct, counts in per_dict_counts.items():
        coding_ex = sum(counts.values())
        mapped_ex = sum(
            n for pid, n in counts.items() if proteins.get(pid, {}).get("uni")
        )
        dict_cov[dct] = {
            "coding_ex": coding_ex,
            "noncoding_ex": noncoding_by_dict[dct],
            "total_ex": total_by_dict[dct],
            "unique_coding_prots": len(counts),
            "mapped_coding_ex": mapped_ex,
            "gap_coding_ex": coding_ex - mapped_ex,
            "afdb_reachable_pct": round(100.0 * mapped_ex / coding_ex, 2) if coding_ex else 0.0,
        }

    summary = {
        "unique_exemplar_proteins": n_unique,
        "afdb_mapped_proteins": n_mapped,
        "gap_proteins": n_gap,
        "afdb_mapped_pct": round(100.0 * n_mapped / n_unique, 2) if n_unique else 0.0,
        "gap_prefixes": dict(gap_prefix.most_common()),
        "per_dict": dict_cov,
    }

    with open(os.path.join(args.out, "proteins.json"), "wb") as f:
        f.write(orjson.dumps(proteins))
    with open(os.path.join(args.out, "per_dict_counts.json"), "wb") as f:
        f.write(orjson.dumps(per_dict_counts))
    with open(os.path.join(args.out, "enum_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("===== ENUM SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def build_struct_index(argv: list[str] | None = None) -> int:
    """Merge AFDB + ESMFold records into a struct index + per-dict coverage."""
    ap = argparse.ArgumentParser(description=build_struct_index.__doc__)
    ap.add_argument(
        "--art",
        required=True,
        help="artifact root with enum/, afdb/ and (optional) esmfold/ subdirs",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    proteins = json.load(open(f"{args.art}/enum/proteins.json"))
    per_dict = json.load(open(f"{args.art}/enum/per_dict_counts.json"))
    enum_summary = json.load(open(f"{args.art}/enum/enum_summary.json"))

    struct_index: dict[str, dict] = {}
    afdb_path = f"{args.art}/afdb/struct_afdb.json"
    if os.path.exists(afdb_path):
        for pid, rec in json.load(open(afdb_path)).items():
            struct_index[pid] = rec
    n_afdb = len(struct_index)

    n_ef = 0
    for shard in sorted(glob.glob(f"{args.art}/esmfold/struct_esmfold_shard_*.json")):
        for pid, rec in json.load(open(shard)).items():
            struct_index[pid] = rec
            n_ef += 1

    coverage: dict[str, dict] = {}
    for dct, counts in per_dict.items():
        coding_ex = sum(counts.values())
        afdb_ex = esm_ex = covered_ex = 0
        covered_prots = 0
        for pid, n in counts.items():
            rec = struct_index.get(pid)
            if not rec:
                continue
            covered_ex += n
            covered_prots += 1
            if rec["source"] == "afdb":
                afdb_ex += n
            else:
                esm_ex += n
        nc = enum_summary["per_dict"][dct]["noncoding_ex"]
        coverage[dct] = {
            "coding_ex": coding_ex,
            "covered_ex": covered_ex,
            "coverage_pct": round(100.0 * covered_ex / coding_ex, 2) if coding_ex else 0.0,
            "afdb_ex": afdb_ex,
            "esmfold_ex": esm_ex,
            "uncovered_ex": coding_ex - covered_ex,
            "noncoding_ex": nc,
            "total_ex": coding_ex + nc,
            "unique_coding_prots": len(counts),
            "covered_prots": covered_prots,
        }

    with open(os.path.join(args.out, "struct_index.json"), "wb") as f:
        f.write(orjson.dumps(struct_index))
    with open(os.path.join(args.out, "struct_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2)

    print(
        f"struct_index: {len(struct_index)} proteins ({n_afdb} afdb + {n_ef} esmfold)",
        flush=True,
    )
    print("===== COVERAGE =====", flush=True)
    print(json.dumps(coverage, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the ``enumerate`` or ``index`` subcommand."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enumerate", add_help=False)
    sub.add_parser("index", add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "enumerate":
        return enumerate_proteins(rest)
    if args.cmd == "index":
        return build_struct_index(rest)
    ap.error(f"unknown subcommand {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
