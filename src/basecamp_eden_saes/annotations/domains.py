"""Pfam domain + GO annotation layer (pyhmmer; one accession per shard step).

For each panel proteome (``protein.faa``):
  1. ``hmmsearch`` Pfam-A against the proteome at the Pfam gathering-threshold
     cutoff (the standard Pfam significance bar) using pyhmmer;
  2. lift each domain's aa span ``[aa_from, aa_to]`` to nucleotide intervals via
     the CDS frame (segments 5'->3', drop ``phase`` leading bases, residue ``a``
     -> coding-nt ``[phase+3(a-1), phase+3a)``), strand-aware;
  3. GO terms are derived later from ``pfam_acc`` via the pfam2go mapping (kept
     out of the per-interval TSV to stay compact).

Writes ``<out>/<acc>/domains.tsv`` (one row per lifted domain interval)::

    contig  nt_start  nt_end  strand  pfam_acc  pfam_name  protein_id
        aa_from  aa_to  bitscore  ievalue

This replaces InterProScan's Java pipeline for the Pfam+GO tracks; SignalP/TM/
Panther members are out of this first pass. Genome inputs default to
:func:`basecamp_eden_saes.config.data_paths` (panel genomes); the per-nt track
directory (with ``cds_map.json``) and the Pfam-A HMM database are explicit args.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from basecamp_eden_saes import config


def load_cds_map(annot_acc_dir: Path) -> dict:
    """Load ``<annot_acc_dir>/cds_map.json`` (protein->CDS-segment map)."""
    f = annot_acc_dir / "cds_map.json"
    return json.loads(f.read_text()) if f.exists() else {}


def assembled_positions(segs: list, strand: str) -> list[int]:
    """Ordered genomic positions of CDS coding nts, 5'->3' (segs already 5'->3')."""
    pos: list[int] = []
    for (s, e) in segs:
        if strand == "-":
            pos.extend(range(e - 1, s - 1, -1))
        else:
            pos.extend(range(s, e))
    return pos


def lift_domain(cds_entry: dict, aa_from: int, aa_to: int) -> list[tuple[int, int]]:
    """aa span (1-based inclusive) -> list of ``(nt_start, nt_end)`` 0-based half-open."""
    strand = cds_entry["strand"]
    phase = cds_entry.get("phase", 0) or 0
    pos = assembled_positions(cds_entry["segs"], strand)
    lo = phase + 3 * (aa_from - 1)
    hi = phase + 3 * aa_to  # exclusive
    if lo >= len(pos):
        return []
    hi = min(hi, len(pos))
    gpos = pos[lo:hi]
    if not gpos:
        return []
    gset = sorted(gpos)
    # collapse to contiguous intervals
    out: list[tuple[int, int]] = []
    start = prev = gset[0]
    for x in gset[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev + 1))
            start = prev = x
    out.append((start, prev + 1))
    return out


def find_cds(cds_map: dict, protein_id: str) -> tuple[str | None, dict | None]:
    """Locate ``(contig, cds_entry)`` for a protein id in ``cds_map``."""
    for cname, pmap in cds_map.items():
        if protein_id in pmap:
            return cname, pmap[protein_id]
    return None, None


def run_hmmsearch(faa: Path, pfam_hmm: Path, cpus: int) -> list[tuple]:
    """Run pyhmmer ``hmmsearch`` (gathering cutoff) of ``pfam_hmm`` vs ``faa``.

    Returns tuples ``(protein_id, pfam_acc, pfam_name, aa_from, aa_to, bitscore,
    ievalue)`` for each included domain (aa coords 1-based inclusive).
    """
    import pyhmmer

    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(str(faa), digital=True, alphabet=alphabet) as sf:
        seqs = sf.read_block()
    hits_out: list[tuple] = []
    with pyhmmer.plan7.HMMFile(str(pfam_hmm)) as hf:
        for top in pyhmmer.hmmer.hmmsearch(hf, seqs, cpus=cpus, bit_cutoffs="gathering"):
            q = top.query
            # pyhmmer 0.12 returns str for name/accession (not bytes) -> no decode.
            pacc = q.accession if q.accession else q.name
            pname = q.name
            for hit in top:
                if not hit.included:
                    continue
                pid = hit.name
                for dom in hit.domains:
                    if not dom.included:
                        continue
                    aln = dom.alignment
                    aa_from, aa_to = aln.target_from, aln.target_to  # 1-based incl
                    hits_out.append((pid, pacc, pname, aa_from, aa_to,
                                     float(dom.score), float(dom.i_evalue)))
    return hits_out


def build_accession_domains(acc: str, genomes_dir: Path, annot: Path, out: Path,
                            pfam_hmm: Path, cpus: int) -> tuple[int, int]:
    """Search + lift Pfam domains for one accession, writing ``domains.tsv``.

    Returns ``(n_domain_hits, n_nt_intervals)``; ``(-1, -1)`` if skipped (no
    ``protein.faa`` or no ``cds_map``), ``(-2, -2)`` if hmmsearch failed.
    """
    faa = genomes_dir / acc / "protein.faa"
    if not faa.exists():
        return -1, -1
    cds_map = load_cds_map(annot / acc)
    if not cds_map:
        return -1, -1
    odir = out / acc
    odir.mkdir(parents=True, exist_ok=True)
    try:
        hits = run_hmmsearch(faa, pfam_hmm, cpus)
    except Exception:
        return -2, -2
    n_lifted = 0
    with (odir / "domains.tsv").open("w") as fh:
        fh.write("contig\tnt_start\tnt_end\tstrand\tpfam_acc\tpfam_name\t"
                 "protein_id\taa_from\taa_to\tbitscore\tievalue\n")
        for (pid, pacc, pname, af, at, bits, iev) in hits:
            cname, entry = find_cds(cds_map, pid)
            if entry is None:
                continue
            for (ns, ne) in lift_domain(entry, af, at):
                fh.write(f"{cname}\t{ns}\t{ne}\t{entry['strand']}\t{pacc}\t{pname}\t"
                         f"{pid}\t{af}\t{at}\t{bits:.1f}\t{iev:.2e}\n")
                n_lifted += 1
    return len(hits), n_lifted


def main(argv: list[str] | None = None) -> int:
    """CLI: build the Pfam/GO domain layer for a shard of the genome panel."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--genomes", type=Path, default=None,
        help="directory of <acc>/protein.faa "
             "(default: config.data_paths().panel_genomes)")
    ap.add_argument("--annot", type=Path, required=True,
                    help="per-nt track directory (with <acc>/cds_map.json)")
    ap.add_argument("--out", type=Path, required=True, help="output domain directory")
    ap.add_argument("--pfam", type=Path, required=True, help="Pfam-A HMM database")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=8)
    args = ap.parse_args(argv)

    genomes_dir = args.genomes or config.data_paths().panel_genomes
    args.out.mkdir(parents=True, exist_ok=True)
    accs = sorted(d.name for d in Path(genomes_dir).iterdir() if d.is_dir())
    accs = accs[args.shard::args.nshards]
    print(f"[domains shard {args.shard}/{args.nshards}] {len(accs)} proteomes")

    for acc in accs:
        n_hits, n_lifted = build_accession_domains(
            acc, Path(genomes_dir), args.annot, args.out, args.pfam, args.cpus)
        if n_hits == -1:
            print(f"  skip {acc} (no protein.faa or no cds_map)")
        elif n_hits == -2:
            print(f"  {acc} hmmsearch FAILED")
        else:
            print(f"  {acc}: {n_hits} domain hits -> {n_lifted} nt intervals")
    print(f"[domains shard {args.shard}] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
