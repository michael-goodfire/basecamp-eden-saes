"""Resolve a curated taxon list to RefSeq accessions and download the panel.

For each row of the taxa TSV (``phylum <tab> query <tab> [pinned_accession]``):
  * pinned accession -> use as-is;
  * else resolve via NCBI Datasets ``summary genome taxon``, preferring the
    designated RefSeq *reference* genome, falling back to the best complete
    RefSeq (``GCF_``) assembly.

Then bulk-download genome FASTA + GFF3 + protein + CDS for all accessions and lay
them out one folder per accession under ``<out>/genomes/<accession>/``. Writes
``<out>/resolved.tsv`` (accession/organism/phylum/assembly level) and
``<out>/accessions.txt``.

Network only (no heavy compute). The NCBI ``datasets`` binary is taken from
``--datasets`` (default: the ``DATASETS_BIN`` env var, else ``datasets`` on PATH).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing text output."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def parse_taxa(path: Path) -> list[tuple[str, str, str | None]]:
    """Parse the taxa TSV -> list of ``(phylum, query, pinned_accession|None)``."""
    rows: list[tuple[str, str, str | None]] = []
    for line in path.read_text().splitlines():
        line = line.rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        phylum = parts[0].strip()
        query = parts[1].strip()
        pinned = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        rows.append((phylum, query, pinned))
    return rows


def summary_taxon(datasets: str, query: str, reference_only: bool) -> list[dict]:
    """Query NCBI Datasets ``summary genome taxon`` -> list of assembly reports."""
    cmd = [datasets, "summary", "genome", "taxon", query,
           "--assembly-source", "RefSeq", "--as-json-lines"]
    if reference_only:
        cmd.append("--reference")
    else:
        cmd += ["--assembly-level", "complete,chromosome"]
    try:
        cp = run(cmd, timeout=90)
    except subprocess.TimeoutExpired:
        return []
    if cp.returncode != 0:
        return []
    out: list[dict] = []
    for ln in cp.stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _accession(rep: dict) -> str | None:
    return rep.get("accession")


def _level(rep: dict) -> str:
    return (rep.get("assembly_info", {}) or {}).get("assembly_level", "") or ""


def _organism(rep: dict) -> str:
    return (rep.get("organism", {}) or {}).get("organism_name", "") or ""


def _gene_count(rep: dict) -> int:
    ann = rep.get("annotation_info", {}) or {}
    stats = (ann.get("stats", {}) or {}).get("gene_counts", {}) or {}
    return int(stats.get("total", 0) or 0)


def resolve_one(datasets: str, query: str) -> dict | None:
    """Resolve a taxon query to the best RefSeq assembly report, or ``None``."""
    # 1) designated RefSeq reference genome
    reps = summary_taxon(datasets, query, reference_only=True)
    reps = [r for r in reps if _accession(r) and _accession(r).startswith("GCF_")]
    if not reps:
        # 2) best complete RefSeq assembly (prefer Complete Genome, then most genes)
        cand = summary_taxon(datasets, query, reference_only=False)
        cand = [r for r in cand if _accession(r) and _accession(r).startswith("GCF_")]
        if not cand:
            return None

        def score(r: dict) -> tuple[int, int]:
            lvl = _level(r).lower()
            lvl_rank = 2 if "complete" in lvl else (1 if "chromosome" in lvl else 0)
            return (lvl_rank, _gene_count(r))

        reps = [max(cand, key=score)]
    r = reps[0]
    return {
        "accession": _accession(r),
        "organism": _organism(r),
        "assembly_level": _level(r),
        "gene_count": _gene_count(r),
    }


def download_batch(datasets: str, accessions: list[str], out_zip: Path) -> bool:
    """Download a batch of accessions to ``out_zip`` (3 retries). ``True`` on success."""
    inputfile = out_zip.with_suffix(".acc.txt")
    inputfile.write_text("\n".join(accessions) + "\n")
    cmd = [datasets, "download", "genome", "accession", "--inputfile", str(inputfile),
           "--include", "genome,gff3,protein,cds", "--filename", str(out_zip)]
    cp: subprocess.CompletedProcess | None = None
    for attempt in range(3):
        try:
            cp = run(cmd, timeout=900)
        except subprocess.TimeoutExpired:
            cp = None
        if cp is not None and cp.returncode == 0 and out_zip.exists():
            return True
        time.sleep(5 * (attempt + 1))
    if cp is not None:
        sys.stderr.write(f"[download] FAILED batch -> {cp.stderr[:500]}\n")
    return False


def unzip_panel(zip_path: Path, genomes_dir: Path) -> dict[str, dict]:
    """Unzip an NCBI Datasets bundle, laying each accession's files into place."""
    tmp = zip_path.with_suffix(".unz")
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    data_dir = tmp / "ncbi_dataset" / "data"
    found: dict[str, dict] = {}
    if not data_dir.exists():
        return found
    for acc_dir in sorted(data_dir.iterdir()):
        if not acc_dir.is_dir():
            continue
        acc = acc_dir.name
        dest = genomes_dir / acc
        dest.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        for f in acc_dir.iterdir():
            n = f.name
            # NB: cds_from_genomic.fna and rna_from_genomic.fna also end with
            # "_genomic.fna" -- the genome is the *_genomic.fna WITHOUT from_genomic.
            if "cds_from_genomic" in n or n.endswith("_cds.fna"):
                shutil.copy(f, dest / "cds.fna")
                files["cds"] = str(dest / "cds.fna")
            elif "rna_from_genomic" in n:
                pass
            elif n.endswith("_genomic.fna") or (n.endswith(".fna") and "from_genomic" not in n):
                shutil.copy(f, dest / "genomic.fna")
                files["fna"] = str(dest / "genomic.fna")
            elif n.endswith(".gff") or n == "genomic.gff":
                shutil.copy(f, dest / "genomic.gff")
                files["gff"] = str(dest / "genomic.gff")
            elif n == "protein.faa" or n.endswith("_protein.faa"):
                shutil.copy(f, dest / "protein.faa")
                files["faa"] = str(dest / "protein.faa")
        found[acc] = files
    shutil.rmtree(tmp, ignore_errors=True)
    return found


def _download_all(datasets: str, accs: list[str], out: Path, genomes_dir: Path,
                  batch: int) -> dict[str, dict]:
    """Download + unpack all accessions in batches; returns the file inventory."""
    all_files: dict[str, dict] = {}
    for b in range(0, len(accs), batch):
        chunk = accs[b:b + batch]
        zp = out / f"batch_{b // batch:03d}.zip"
        print(f"[download] batch {b // batch} ({len(chunk)} accs)...")
        if download_batch(datasets, chunk, zp):
            found = unzip_panel(zp, genomes_dir)
            all_files.update(found)
            zp.unlink(missing_ok=True)
            zp.with_suffix(".acc.txt").unlink(missing_ok=True)
            print(f"  unpacked {len(found)} (total {len(all_files)})")
        else:
            print(f"  batch {b // batch} FAILED")
    return all_files


def main(argv: list[str] | None = None) -> int:
    """CLI: resolve taxa to RefSeq accessions and download the genome panel."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxa", type=Path, help="taxa TSV (required unless --download-only)")
    ap.add_argument("--out", type=Path, required=True, help="panel output directory")
    ap.add_argument("--batch", type=int, default=25, help="accessions per download batch")
    ap.add_argument("--resolve-only", action="store_true", help="resolve, do not download")
    ap.add_argument("--download-only", action="store_true",
                    help="skip resolve; reuse <out>/accessions.txt")
    ap.add_argument("--datasets", default=os.environ.get("DATASETS_BIN", "datasets"),
                    help="NCBI Datasets CLI binary (default: $DATASETS_BIN or 'datasets')")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    genomes_dir = out / "genomes"
    genomes_dir.mkdir(exist_ok=True)

    if args.download_only:
        accs = [a.strip() for a in (out / "accessions.txt").read_text().splitlines()
                if a.strip()]
        print(f"[download-only] {len(accs)} accessions from accessions.txt")
        all_files = _download_all(args.datasets, accs, out, genomes_dir, args.batch)
        (out / "file_inventory.json").write_text(json.dumps(all_files, indent=2))
        n_ok = sum(1 for f in all_files.values() if {"fna", "gff", "faa"} <= set(f))
        print(f"[done] {len(all_files)} genomes, {n_ok} with fna+gff+faa")
        return 0

    if args.taxa is None:
        ap.error("--taxa is required unless --download-only")

    taxa = parse_taxa(Path(args.taxa))
    print(f"[resolve] {len(taxa)} taxa")

    resolved: list[dict] = []
    seen_acc: set[str] = set()
    for i, (phylum, query, pinned) in enumerate(taxa):
        if pinned:
            info: dict | None = {"accession": pinned, "organism": query,
                                 "assembly_level": "pinned", "gene_count": 0}
        else:
            info = resolve_one(args.datasets, query)
        if not info or not info["accession"]:
            print(f"  [{i + 1}/{len(taxa)}] UNRESOLVED: {query}")
            resolved.append({"phylum": phylum, "query": query, "accession": "",
                             "organism": "", "assembly_level": "UNRESOLVED",
                             "gene_count": 0})
            continue
        acc = info["accession"]
        if acc in seen_acc:
            print(f"  [{i + 1}/{len(taxa)}] dup {acc} ({query}) -> skip")
            continue
        seen_acc.add(acc)
        resolved.append({"phylum": phylum, "query": query, **info})
        print(f"  [{i + 1}/{len(taxa)}] {query} -> {acc} "
              f"({info['assembly_level']}, genes={info['gene_count']})")

    man = out / "resolved.tsv"
    with man.open("w") as fh:
        fh.write("phylum\tquery\taccession\torganism\tassembly_level\tgene_count\n")
        for r in resolved:
            fh.write(f"{r['phylum']}\t{r['query']}\t{r['accession']}\t"
                     f"{r.get('organism', '')}\t{r['assembly_level']}\t"
                     f"{r.get('gene_count', 0)}\n")
    accs = [r["accession"] for r in resolved if r["accession"]]
    (out / "accessions.txt").write_text("\n".join(accs) + "\n")
    print(f"[resolve] {len(accs)} accessions -> {man}")
    if args.resolve_only:
        return 0

    all_files = _download_all(args.datasets, accs, out, genomes_dir, args.batch)
    (out / "file_inventory.json").write_text(json.dumps(all_files, indent=2))
    n_complete = sum(1 for f in all_files.values()
                     if {"fna", "gff", "faa", "cds"} <= set(f))
    print(f"[done] {len(all_files)} genomes downloaded, "
          f"{n_complete} with all 4 file types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
