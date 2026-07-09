"""AFDB precompute for the UniProt-mapped exemplar proteins.

Reads ``proteins.json`` (from :mod:`.coverage`) and, for every protein with a
UniProt accession, downloads the AlphaFold-DB mmCIF into ``<cache>/<uni>.cif``
(validated by the ``data_`` magic prefix) and records mean pLDDT (mean CA
B-factor, read via ``gemmi``). Proteins that are UniProt-mapped but have no AFDB
model (404) go to ``afdb_gap.json`` so the ESMFold gap step can fold them.

The AFDB ``/files/`` URL 404s for accessions that are actually modelled, so the
real cif URL is resolved through the prediction API first.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import gemmi
import orjson
import requests

from basecamp_eden_saes import config

API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uni}"


def mean_plddt(cif_path: str) -> float | None:
    """Mean per-residue pLDDT (AFDB stores it in the CA B-factor column)."""
    st = gemmi.read_structure(cif_path)
    vals = []
    for model in st:
        for chain in model:
            for res in chain:
                ca = res.find_atom("CA", "*")
                if ca is not None:
                    vals.append(ca.b_iso)
        break  # first model only
    return round(sum(vals) / len(vals), 2) if vals else None


def fetch_one(uni: str, cache_dir: str, session: requests.Session) -> tuple[str, float | None]:
    """Fetch one AFDB model: ``('ok', plddt) | ('missing', None) | ('error', msg)``."""
    dest = os.path.join(cache_dir, f"{uni}.cif")
    if os.path.exists(dest):
        try:
            if open(dest).read(8).startswith("data_"):
                return ("ok", mean_plddt(dest))
        except Exception:
            pass
    api = API_URL.format(uni=uni)
    last = "?"
    for _ in range(3):
        try:
            r = session.get(api, timeout=30)
            if r.status_code == 404:
                return ("missing", None)
            if r.ok:
                j = r.json()
                if not j or not j[0].get("cifUrl"):
                    return ("missing", None)
                cif_url = j[0]["cifUrl"]
                cr = session.get(cif_url, timeout=60)
                if cr.ok and cr.text.startswith("data_"):
                    tmp = dest + ".tmp"
                    with open(tmp, "w") as f:
                        f.write(cr.text)
                    os.replace(tmp, dest)
                    return ("ok", mean_plddt(dest))
                last = f"cif http {cr.status_code}"
            else:
                last = f"api http {r.status_code}"
        except Exception as e:
            last = str(e)
    return ("error", last)


def main(argv: list[str] | None = None) -> int:
    """Download AFDB models for all UniProt-mapped exemplar proteins."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proteins", required=True, help="proteins.json from coverage enumerate")
    ap.add_argument(
        "--cache",
        default=None,
        help="af_cache dir for CIFs (default: config.data_paths().af_cache)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only first N mapped proteins (smoke)")
    args = ap.parse_args(argv)
    cache = args.cache or str(config.data_paths().af_cache)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    proteins = json.load(open(args.proteins))
    # unique by uniprot accession (many refseq can share a uni); keep prot_ids per uni
    mapped = [(pid, p["uni"]) for pid, p in proteins.items() if p.get("uni")]
    if args.limit:
        mapped = mapped[: args.limit]
    uni_to_pids: dict[str, list[str]] = {}
    for pid, uni in mapped:
        uni_to_pids.setdefault(uni, []).append(pid)
    unis = list(uni_to_pids)
    print(
        f"{len(mapped)} mapped proteins -> {len(unis)} unique UniProt accessions",
        flush=True,
    )

    results: dict[str, tuple[str, float | None]] = {}
    session = requests.Session()
    done = 0
    total = len(unis)
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(fetch_one, uni, cache, session): uni for uni in unis}
        for fut in as_completed(futs):
            uni = futs[fut]
            try:
                results[uni] = fut.result()
            except Exception as e:
                results[uni] = ("error", str(e))
            done += 1
            if done % 500 == 0:
                ok = sum(1 for v in results.values() if v[0] == "ok")
                print(f"  {done}/{total} done  ok={ok}", flush=True)

    struct_afdb: dict[str, dict] = {}
    gap: dict[str, dict] = {}
    n_ok = n_missing = n_error = 0
    for uni, (status, plddt) in results.items():
        for pid in uni_to_pids[uni]:
            if status == "ok":
                struct_afdb[pid] = {"source": "afdb", "key": uni, "uni": uni, "plddt": plddt}
            else:
                gap[pid] = proteins[pid]
        if status == "ok":
            n_ok += 1
        elif status == "missing":
            n_missing += 1
        else:
            n_error += 1

    summary = {
        "mapped_proteins": len(mapped),
        "unique_uniprot": len(unis),
        "afdb_ok_uni": n_ok,
        "afdb_missing_uni": n_missing,
        "afdb_error_uni": n_error,
        "afdb_ok_prots": len(struct_afdb),
        "gap_prots_added_from_afdb_miss": len(gap),
    }
    with open(os.path.join(args.out, "struct_afdb.json"), "wb") as f:
        f.write(orjson.dumps(struct_afdb))
    with open(os.path.join(args.out, "afdb_gap.json"), "wb") as f:
        f.write(orjson.dumps(gap))
    with open(os.path.join(args.out, "afdb_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("===== AFDB SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
