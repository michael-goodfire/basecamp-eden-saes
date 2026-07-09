"""RefSeq protein accession -> UniProtKB mapping, with UniParc MD5 fallbacks.

Three stages, exposed as subcommands of :func:`main`:

``refseq`` (:func:`map_refseq`): RefSeq -> UniProtKB via the UniProt ID-mapping
API. Concurrent, patient-polling, resumable. ``WP_`` (non-redundant RefSeq) maps
one-to-many; all UniProt accessions + taxid are kept.

``uniparc`` (:func:`map_uniparc`): sequence fallback for proteins the RefSeq map
missed (locus tags + accession misses). Computes each protein's sequence MD5 and
queries UniParc (``checksum:`` accepts MD5) -> UPI -> its UniProtKB accessions.

``uniparc-batch`` (:func:`map_uniparc_batch`): re-maps proteins whose current
accessions miss TED. TED/AFDB index proteins predominantly under TrEMBL
accessions, while RefSeq mapping returns the single Swiss-Prot canonical; here we
fetch each protein's FULL UniProtKB accession list via UniParc-by-sequence,
batching many checksums per request and associating results back by MD5.

The mapping value schema is ``{refseq_id: [[uniprot_acc, taxid], ...]}``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

IDMAP_API = "https://rest.uniprot.org/idmapping"
UNIPARC_API = "https://rest.uniprot.org/uniparc/search"
REFSEQ = re.compile(r"^(WP|NP|YP|XP)_")

# Standard genetic code + reverse-complement table, used to translate CDS when a
# protein sequence is absent from protein.faa (locus-tag-only entries).
CODON = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
COMP = str.maketrans("ACGTN", "TGCAN")


# --- shared sequence helpers -------------------------------------------------


def load_seqs(fasta_dir: str) -> dict[str, str]:
    """Map ``protein_id -> sequence`` across all genomes' ``protein.faa`` files."""
    seqs: dict[str, str] = {}
    for f in glob.glob(f"{fasta_dir}/*/protein.faa"):
        pid, buf = None, []
        for line in open(f):
            if line.startswith(">"):
                if pid:
                    seqs[pid] = "".join(buf)
                pid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if pid:
            seqs[pid] = "".join(buf)
    return seqs


def translate_cds(genome_seq: str, info: dict) -> str:
    """Translate a CDS (contig sequence + cds_map info) to a protein sequence."""
    strand, segs = info["strand"], info["segs"]
    nt = "".join(genome_seq[a:b] for a, b in segs)
    if strand == "-":
        nt = nt.translate(COMP)[::-1]
    aa = []
    for i in range(0, len(nt) - 2, 3):
        c = CODON.get(nt[i : i + 3], "X")
        if c == "*":
            break
        aa.append(c)
    return "".join(aa)


def load_genome_contigs(fasta_dir: str, acc: str) -> dict[str, str]:
    """Load ``{contig_id: sequence}`` from ``<fasta_dir>/<acc>/genomic.fna``."""
    seqs: dict[str, str] = {}
    p = f"{fasta_dir}/{acc}/genomic.fna"
    if not os.path.exists(p):
        return seqs
    cid, buf = None, []
    for line in open(p):
        if line.startswith(">"):
            if cid:
                seqs[cid] = "".join(buf)
            cid = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if cid:
        seqs[cid] = "".join(buf)
    return seqs


# --- stage 1: RefSeq ID-mapping ---------------------------------------------


def map_one_batch(batch: list[str], deadline_s: int = 2700) -> dict[str, list]:
    """Run -> poll -> fetch one ID-mapping batch; ``{refseq: [[acc, tax], ...]}``."""
    sess = requests.Session()
    sess.headers["User-Agent"] = "silico-worker"
    job = None
    for _ in range(8):
        try:
            r = sess.post(
                f"{IDMAP_API}/run",
                data={"from": "RefSeq_Protein", "to": "UniProtKB", "ids": ",".join(batch)},
                timeout=90,
            )
            r.raise_for_status()
            job = r.json()["jobId"]
            break
        except Exception:
            time.sleep(8)
    if not job:
        return {}
    t0 = time.time()
    ok = False
    while time.time() - t0 < deadline_s:
        try:
            s = sess.get(f"{IDMAP_API}/status/{job}", timeout=30).json()
        except Exception:
            time.sleep(8)
            continue
        st = s.get("jobStatus")
        if st == "FINISHED" or "results" in s:
            ok = True
            break
        if st in ("ERROR", "FAILED"):
            return {}
        time.sleep(8)
    if not ok:
        return {}
    out: dict[str, list] = {}
    for _ in range(8):
        try:
            rr = sess.get(
                f"{IDMAP_API}/uniprotkb/results/stream/{job}"
                "?format=tsv&fields=accession,organism_id",
                timeout=600,
            )
            rr.raise_for_status()
            for line in rr.text.splitlines()[1:]:
                p = line.split("\t")
                if len(p) >= 2:
                    out.setdefault(p[0], []).append([p[1], p[2] if len(p) > 2 else ""])
            return out
        except Exception:
            time.sleep(10)
    return {}


def map_refseq(argv: list[str] | None = None) -> int:
    """Map RefSeq protein accessions to UniProtKB (resumable)."""
    ap = argparse.ArgumentParser(description=map_refseq.__doc__)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--batch", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    all_ids = [x.strip() for x in open(args.ids) if x.strip()]
    refseq = [i for i in all_ids if REFSEQ.match(i)]
    other = [i for i in all_ids if not REFSEQ.match(i)]

    mapping: dict[str, list] = {}
    if os.path.exists(args.out):
        try:
            mapping = json.load(open(args.out))
        except Exception:
            mapping = {}
    done = set(mapping)
    todo = [i for i in refseq if i not in done]
    print(
        f"total {len(all_ids)}, refseq {len(refseq)}, locus_tags {len(other)}, "
        f"to map now {len(todo)}",
        flush=True,
    )

    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(map_one_batch, b): bi for bi, b in enumerate(batches)}
        for k, fut in enumerate(as_completed(futs)):
            res = fut.result()
            mapping.update(res)
            if (k + 1) % 5 == 0:
                json.dump(mapping, open(args.out, "w"))  # periodic checkpoint

    json.dump(mapping, open(args.out, "w"))
    uni = {a for v in mapping.values() for a, _ in v}
    cov = {
        "total_proteins": len(all_ids),
        "refseq_ids": len(refseq),
        "locus_tag_ids": len(other),
        "refseq_mapped": len(mapping),
        "refseq_coverage_pct": round(100 * len(mapping) / max(len(refseq), 1), 2),
        "overall_coverage_pct": round(100 * len(mapping) / max(len(all_ids), 1), 2),
        "distinct_uniprot": len(uni),
        "unmapped": len(all_ids) - len(mapping),
    }
    json.dump(cov, open(args.coverage, "w"), indent=1)
    print("COVERAGE:", json.dumps(cov), flush=True)
    return 0


# --- stage 2: UniParc MD5 fallback ------------------------------------------


def query_md5(md5: str, cap: int) -> tuple[str, list[str]]:
    """Query UniParc for one sequence checksum; return ``(md5, [accessions])``."""
    sess = requests.Session()
    sess.headers["User-Agent"] = "silico-worker"
    for _ in range(5):
        try:
            r = sess.get(
                UNIPARC_API,
                params={
                    "query": f"checksum:{md5}",
                    "fields": "upi,accession",
                    "format": "tsv",
                },
                timeout=45,
            )
            r.raise_for_status()
            lines = r.text.splitlines()
            if len(lines) < 2:
                return md5, []
            accs = []
            for ln in lines[1:]:
                p = ln.split("\t")
                if len(p) >= 2 and p[1]:
                    for a in p[1].split(";"):
                        a = a.strip().split(".")[0]  # drop version
                        if a:
                            accs.append(a)
            # de-dup, prefer shorter (reviewed Swiss-Prot) accessions first
            seen, ordered = set(), []
            for a in sorted(accs, key=len):
                if a not in seen:
                    seen.add(a)
                    ordered.append(a)
            return md5, ordered[:cap]
        except Exception:
            time.sleep(5)
    return md5, []


def map_uniparc(argv: list[str] | None = None) -> int:
    """Extend the mapping via per-sequence UniParc MD5 lookups."""
    ap = argparse.ArgumentParser(description=map_uniparc.__doc__)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument(
        "--cds-root",
        default="",
        help="annotation dir; enables CDS translation for proteins missing from protein.faa",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--cap", type=int, default=80)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)

    mapping = json.load(open(args.mapping))
    all_ids = [x.strip() for x in open(args.ids) if x.strip()]
    unmapped = [i for i in all_ids if i not in mapping]
    print(
        f"{len(all_ids)} total, {len(mapping)} already mapped, "
        f"{len(unmapped)} to try via sequence",
        flush=True,
    )

    seqs = load_seqs(args.fasta_dir)
    pid2loc: dict[str, tuple] = {}  # pid -> (acc, contig, info)
    genome_cache: dict[str, dict[str, str]] = {}
    if args.cds_root:
        for f in glob.glob(f"{args.cds_root}/*/cds_map.json"):
            acc = f.split("/")[-2]
            cds = json.load(open(f))
            for contig, prots in cds.items():
                for pid, info in prots.items():
                    pid2loc[pid] = (acc, contig, info)

    def get_seq(pid: str) -> str | None:
        s = seqs.get(pid) or seqs.get(pid.split(".")[0])
        if s:
            return s
        if args.cds_root and pid in pid2loc:
            acc, contig, info = pid2loc[pid]
            if acc not in genome_cache:
                genome_cache[acc] = load_genome_contigs(args.fasta_dir, acc)
            gseq = genome_cache[acc].get(contig)
            if gseq:
                aa = translate_cds(gseq, info)
                if aa:
                    return aa
        return None

    md5_to_ids: dict[str, list[str]] = {}
    n_translated = 0
    for pid in unmapped:
        s = get_seq(pid)
        if not s:
            continue
        if not (seqs.get(pid) or seqs.get(pid.split(".")[0])):
            n_translated += 1
        h = hashlib.md5(s.encode()).hexdigest()
        md5_to_ids.setdefault(h, []).append(pid)
    print(
        f"unique sequences (md5) to query: {len(md5_to_ids)} "
        f"({n_translated} via CDS translation)",
        flush=True,
    )

    md5_to_acc: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(query_md5, h, args.cap) for h in md5_to_ids]
        for fut in as_completed(futs):
            h, accs = fut.result()
            if accs:
                md5_to_acc[h] = accs

    added = 0
    for h, ids in md5_to_ids.items():
        accs = md5_to_acc.get(h)
        if not accs:
            continue
        for pid in ids:
            mapping[pid] = [[a, ""] for a in accs]
            added += 1
    json.dump(mapping, open(args.out, "w"))
    cov = {
        "total_proteins": len(all_ids),
        "mapped_after_fallback": len(mapping),
        "coverage_pct": round(100 * len(mapping) / max(len(all_ids), 1), 2),
        "added_by_sequence": added,
        "unique_md5_queried": len(md5_to_ids),
        "unique_md5_hits": len(md5_to_acc),
    }
    json.dump(cov, open(args.coverage, "w"), indent=1)
    print("FALLBACK COVERAGE:", json.dumps(cov), flush=True)
    return 0


# --- stage 3: batched UniParc re-mapping to maximise TED coverage ------------


def query_batch(md5s: list[str]) -> dict[str, list[str]]:
    """OR-query a batch of checksums; return ``{md5: [accessions]}`` keyed by seq MD5."""
    q = "(" + ")OR(".join(f"checksum:{h}" for h in md5s) + ")"
    sess = requests.Session()
    sess.headers["User-Agent"] = "silico-worker"
    for _ in range(5):
        try:
            r = sess.get(
                UNIPARC_API,
                params={
                    "query": q,
                    "fields": "accession,sequence",
                    "format": "json",
                    "size": 500,
                },
                timeout=60,
            )
            r.raise_for_status()
            out: dict[str, list[str]] = {}
            for res in r.json().get("results", []):
                seq = res.get("sequence", {}).get("value", "")
                if not seq:
                    continue
                h = hashlib.md5(seq.encode()).hexdigest()
                accs = [
                    (a["value"] if isinstance(a, dict) else a).split(".")[0]
                    for a in res.get("uniProtKBAccessions", [])
                ]
                if accs:
                    out.setdefault(h, [])
                    out[h].extend(accs)
            return out
        except Exception:
            time.sleep(5)
    return {}


def map_uniparc_batch(argv: list[str] | None = None) -> int:
    """Re-map TED-missing proteins to their full UniProtKB accession lists."""
    ap = argparse.ArgumentParser(description=map_uniparc_batch.__doc__)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--cds-root", required=True)
    ap.add_argument("--ted-accs", required=True, help="ted_filtered.tsv (col0 = accession)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cap", type=int, default=60)
    args = ap.parse_args(argv)

    mapping = json.load(open(args.mapping))
    ids = [x.strip() for x in open(args.ids) if x.strip()]
    ted_accs: set[str] = set()
    with open(args.ted_accs) as f:
        next(f)
        for line in f:
            ted_accs.add(line.split("\t", 1)[0])
    print(f"TED-covered accs: {len(ted_accs)}", flush=True)

    def hits_ted(pid: str) -> bool:
        v = mapping.get(pid) or mapping.get(pid.split(".")[0])
        return bool(v) and any(a in ted_accs for a, _ in v)

    missing = [p for p in ids if not hits_ted(p)]
    print(f"{len(ids)} proteins, {len(missing)} miss TED with current mapping", flush=True)

    seqs = load_seqs(args.fasta_dir)
    pid2loc: dict[str, tuple] = {}
    gcache: dict[str, dict[str, str]] = {}
    for f in glob.glob(f"{args.cds_root}/*/cds_map.json"):
        acc = f.split("/")[-2]
        for contig, prots in json.load(open(f)).items():
            for pid, info in prots.items():
                pid2loc[pid] = (acc, contig, info)

    def seqof(pid: str) -> str | None:
        s = seqs.get(pid) or seqs.get(pid.split(".")[0])
        if s:
            return s
        if pid in pid2loc:
            acc, contig, info = pid2loc[pid]
            if acc not in gcache:
                gcache[acc] = load_genome_contigs(args.fasta_dir, acc)
            g = gcache[acc].get(contig)
            if g:
                return translate_cds(g, info)
        return None

    md5_to_ids: dict[str, list[str]] = {}
    for pid in missing:
        s = seqof(pid)
        if not s:
            continue
        md5_to_ids.setdefault(hashlib.md5(s.encode()).hexdigest(), []).append(pid)
    md5s = list(md5_to_ids)
    print(f"unique sequences to query: {len(md5s)}", flush=True)

    batches = [md5s[i : i + args.batch] for i in range(0, len(md5s), args.batch)]
    md5_to_acc: dict[str, set[str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(query_batch, b) for b in batches]
        for fut in as_completed(futs):
            for h, accs in fut.result().items():
                if h in md5_to_ids:
                    md5_to_acc.setdefault(h, set()).update(accs)

    added = 0
    for h, pids in md5_to_ids.items():
        accs = md5_to_acc.get(h)
        if not accs:
            continue
        # TrEMBL/A0A accessions sort first (TED/AFDB index these); cap keeps TED accs
        val = [[a, ""] for a in sorted(set(accs))[: args.cap]]
        for pid in pids:
            mapping[pid] = val
            added += 1
    json.dump(mapping, open(args.out, "w"))
    ted_hit = sum(1 for p in ids if hits_ted(p))
    cov = {
        "total": len(ids),
        "mapped": len(mapping),
        "ted_hittable": ted_hit,
        "ted_hittable_pct": round(100 * ted_hit / len(ids), 2),
        "reassigned": added,
    }
    json.dump(cov, open(args.coverage, "w"), indent=1)
    print("TED COVERAGE:", json.dumps(cov), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the ``refseq``, ``uniparc`` or ``uniparc-batch`` subcommand."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refseq", add_help=False)
    sub.add_parser("uniparc", add_help=False)
    sub.add_parser("uniparc-batch", add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "refseq":
        return map_refseq(rest)
    if args.cmd == "uniparc":
        return map_uniparc(rest)
    if args.cmd == "uniparc-batch":
        return map_uniparc_batch(rest)
    ap.error(f"unknown subcommand {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
