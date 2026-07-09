"""Per-nucleotide annotation tracks for genome-grounded SAE autointerp (#31).

Builds, for each RefSeq genome contig, dense per-nt annotation arrays that an SAE
feature's per-nt activations can be joined against on genomic coordinates.

Coordinate discipline (the silent-failure zone):
  * GFF3 is 1-based inclusive; converted to 0-based half-open ``[start, end)`` on
    load, the convention used everywhere else here and by Python slicing.
  * Strand matters. EDEN reads a window 5'->3'. A ``+``-strand window at model
    position ``i`` reads genome position ``start+i`` on the ``+`` strand; a
    ``-``-strand (reverse-complement) window at model position ``i`` reads genome
    position ``start+L-1-i`` on the ``-`` strand (complemented base).
  * Annotations are built **per genome strand** and queried **relative to the
    window's reading strand**: a CDS the model reads in-frame is a *sense*
    annotation; the gene on the opposite strand is *antisense* (a confound), not
    the same thing.

CDS codon position uses the GFF3 ``phase`` column so frame is exact:
  * ``+`` strand: local offset ``o = x - cds_start``; codon_pos = ((o - phase) % 3) + 1
  * ``-`` strand: offset from 5' end ``o = (cds_end-1) - x``; codon_pos = ((o - phase) % 3) + 1

``main`` builds per-nt tracks per accession. Genome inputs are resolved from
:func:`basecamp_eden_saes.config.data_paths` (panel genomes) unless overridden.
"""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from basecamp_eden_saes import config

# --- FASTA -------------------------------------------------------------------

_COMP = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")
_UPPER = np.arange(256, dtype=np.uint8)
for _c in range(ord("a"), ord("z") + 1):
    _UPPER[_c] = _c - 32


def _open(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_fasta(path: str | Path) -> dict[str, np.ndarray]:
    """Return ``{contig_id: uint8 array of uppercased bytes}``. id = first token."""
    seqs: dict[str, list[str]] = {}
    cur = None
    with _open(path) as fh:
        for line in fh:
            if not line:
                continue
            if line[0] == ">":
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.strip())
    out: dict[str, np.ndarray] = {}
    for cid, chunks in seqs.items():
        raw = np.frombuffer("".join(chunks).encode("ascii", "replace"), dtype=np.uint8)
        out[cid] = _UPPER[raw]
    return out


def revcomp_bytes(arr: np.ndarray) -> np.ndarray:
    """Reverse-complement a uint8 ACGT byte array."""
    return np.frombuffer(bytes(arr.tobytes()).translate(_COMP)[::-1], dtype=np.uint8)


# --- GFF3 --------------------------------------------------------------------

@dataclass
class Feature:
    """A single GFF3 feature in 0-based half-open coordinates."""

    contig: str
    ftype: str
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive
    strand: str  # '+', '-', '.'
    phase: int  # 0/1/2, -1 if none
    attrs: dict


def _parse_attrs(s: str) -> dict:
    d: dict[str, str] = {}
    for kv in s.strip().split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def read_gff(path: str | Path) -> list[Feature]:
    """Parse GFF3 (1-based inclusive) -> :class:`Feature` in 0-based half-open."""
    feats: list[Feature] = []
    with _open(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            contig, _src, ftype, start, end, _score, strand, phase, attrs = p[:9]
            try:
                s = int(start) - 1  # 1-based inclusive -> 0-based inclusive
                e = int(end)  # 1-based inclusive end -> 0-based exclusive
            except ValueError:
                continue
            ph = int(phase) if phase in ("0", "1", "2") else -1
            feats.append(Feature(contig, ftype, s, e, strand, ph, _parse_attrs(attrs)))
    return feats


# --- track construction ------------------------------------------------------

RNA_CODE = {"tRNA": 1, "rRNA": 2, "ncRNA": 3, "tmRNA": 3, "antisense_RNA": 3,
            "RNase_P_RNA": 3, "SRP_RNA": 3}
REG_TYPES = {"riboswitch", "regulatory_region", "binding_site", "promoter",
             "RNase_P_RNA", "sequence_feature"}


@dataclass
class ContigTracks:
    """Per-nt arrays for a single contig (each keyed array has length ``length``)."""

    contig: str
    length: int
    arrays: dict  # name -> np.ndarray (length,)
    # interval annotations (sense, on a strand): list of (start,end,strand,label,klass)
    intervals: list = field(default_factory=list)


def build_contig_tracks(contig: str, seq: np.ndarray, feats: list[Feature],
                        gc_win: int = 50, near: int = 20) -> ContigTracks:
    """Build the strand-explicit per-nt track arrays for one contig.

    ``seq`` is the uint8 byte array from :func:`read_fasta`; ``feats`` is the full
    :func:`read_gff` feature list (filtered to ``contig`` internally).
    """
    L = len(seq)
    a = {
        "cds_plus": np.zeros(L, np.uint8),
        "cds_minus": np.zeros(L, np.uint8),
        "codonpos_plus": np.zeros(L, np.int8),  # 0 none, 1/2/3
        "codonpos_minus": np.zeros(L, np.int8),
        "rna_plus": np.zeros(L, np.int8),  # RNA_CODE
        "rna_minus": np.zeros(L, np.int8),
        "pseudo_plus": np.zeros(L, np.uint8),
        "pseudo_minus": np.zeros(L, np.uint8),
        "reg": np.zeros(L, np.uint8),  # regulatory (strand-agnostic)
        "near_start_plus": np.zeros(L, np.uint8),
        "near_start_minus": np.zeros(L, np.uint8),
        "near_stop_plus": np.zeros(L, np.uint8),
        "near_stop_minus": np.zeros(L, np.uint8),
    }
    cf = [f for f in feats if f.contig == contig]
    for f in cf:
        s, e = max(0, f.start), min(L, f.end)
        if s >= e:
            continue
        if f.ftype == "CDS":
            ph = f.phase if f.phase >= 0 else 0
            idx = np.arange(s, e)
            if f.strand == "-":
                a["cds_minus"][s:e] = 1
                off = (e - 1) - idx
                a["codonpos_minus"][s:e] = ((off - ph) % 3 + 1).astype(np.int8)
                a["near_start_minus"][max(0, e - near):e] = 1  # 5' end at high coord
                a["near_stop_minus"][s:min(L, s + near)] = 1
            else:
                a["cds_plus"][s:e] = 1
                off = idx - s
                a["codonpos_plus"][s:e] = ((off - ph) % 3 + 1).astype(np.int8)
                a["near_start_plus"][s:min(L, s + near)] = 1
                a["near_stop_plus"][max(0, e - near):e] = 1
        elif f.ftype in RNA_CODE:
            tgt = "rna_minus" if f.strand == "-" else "rna_plus"
            a[tgt][s:e] = RNA_CODE[f.ftype]
        elif f.ftype == "pseudogene" or f.attrs.get("pseudo") == "true":
            tgt = "pseudo_minus" if f.strand == "-" else "pseudo_plus"
            a[tgt][s:e] = 1
        elif f.ftype in REG_TYPES:
            a["reg"][s:e] = 1

    # intergenic = not covered by any CDS/RNA/pseudo on either strand
    covered = (a["cds_plus"] | a["cds_minus"] | a["pseudo_plus"] | a["pseudo_minus"]
               | (a["rna_plus"] > 0) | (a["rna_minus"] > 0)).astype(np.uint8)
    a["intergenic"] = (1 - covered).astype(np.uint8)

    # GC content & skew in a centered window (magnitude strand-independent; skew flips on '-')
    is_g = (seq == 71).astype(np.float32)
    is_c = (seq == 67).astype(np.float32)
    is_a = (seq == 65).astype(np.float32)
    is_t = (seq == 84).astype(np.float32)
    k = gc_win
    ker = np.ones(2 * k + 1, np.float32)

    def smooth(x: np.ndarray) -> np.ndarray:
        return np.convolve(x, ker, mode="same")

    g = smooth(is_g)
    c = smooth(is_c)
    at = smooth(is_a) + smooth(is_t)
    gc_tot = g + c
    denom = gc_tot + at
    denom[denom == 0] = 1.0
    a["gc"] = (gc_tot / denom).astype(np.float16)
    gcsd = g + c
    gcsd[gcsd == 0] = 1.0
    a["gc_skew_plus"] = ((g - c) / gcsd).astype(np.float16)  # '-' read = negative of this
    return ContigTracks(contig, L, a)


# --- sense/antisense query view ---------------------------------------------

SENSE_COLS = [
    "sense_CDS", "sense_codon1", "sense_codon2", "sense_codon3",
    "antisense_CDS", "intergenic",
    "sense_tRNA", "sense_rRNA", "sense_ncRNA",
    "sense_pseudogene", "regulatory",
    "near_start", "near_stop", "gc", "gc_skew",
]


def sense_view(tracks: ContigTracks, pos: np.ndarray,
               strand: np.ndarray) -> dict[str, np.ndarray]:
    """Annotation values at genome positions ``pos`` read on ``strand``.

    ``strand`` is a uint8/byte array with 43 (``+``) or 45 (``-``). Returns a dict
    of per-position arrays keyed by :data:`SENSE_COLS` (sense = read-strand
    consistent).
    """
    a = tracks.arrays
    plus = (strand == ord("+"))
    out: dict[str, np.ndarray] = {}

    def pick(plus_arr: np.ndarray, minus_arr: np.ndarray) -> np.ndarray:
        return np.where(plus, plus_arr[pos], minus_arr[pos])

    cds_sense = pick(a["cds_plus"], a["cds_minus"])
    cds_anti = pick(a["cds_minus"], a["cds_plus"])
    codon_sense = pick(a["codonpos_plus"], a["codonpos_minus"])
    rna_sense = pick(a["rna_plus"], a["rna_minus"])
    pseudo_sense = pick(a["pseudo_plus"], a["pseudo_minus"])
    near_start = pick(a["near_start_plus"], a["near_start_minus"])
    near_stop = pick(a["near_stop_plus"], a["near_stop_minus"])
    out["sense_CDS"] = cds_sense.astype(np.uint8)
    out["antisense_CDS"] = cds_anti.astype(np.uint8)
    out["sense_codon1"] = (codon_sense == 1).astype(np.uint8)
    out["sense_codon2"] = (codon_sense == 2).astype(np.uint8)
    out["sense_codon3"] = (codon_sense == 3).astype(np.uint8)
    out["intergenic"] = a["intergenic"][pos].astype(np.uint8)
    out["sense_tRNA"] = (rna_sense == 1).astype(np.uint8)
    out["sense_rRNA"] = (rna_sense == 2).astype(np.uint8)
    out["sense_ncRNA"] = (rna_sense == 3).astype(np.uint8)
    out["sense_pseudogene"] = pseudo_sense.astype(np.uint8)
    out["regulatory"] = a["reg"][pos].astype(np.uint8)
    out["near_start"] = near_start.astype(np.uint8)
    out["near_stop"] = near_stop.astype(np.uint8)
    out["gc"] = a["gc"][pos].astype(np.float32)
    skew = a["gc_skew_plus"][pos].astype(np.float32)
    out["gc_skew"] = np.where(plus, skew, -skew)
    return out


# --- low-complexity + CDS-segment helpers ------------------------------------

def lowcomplex_track(seq: np.ndarray, win: int = 25, thr: float = 1.4) -> np.ndarray:
    """Binary low-complexity: windowed nt Shannon entropy below ``thr`` bits."""
    L = len(seq)
    code = np.full(L, 4, np.int8)
    for i, b in enumerate((65, 67, 71, 84)):
        code[seq == b] = i
    onehot = np.zeros((4, L), np.float32)
    for i in range(4):
        onehot[i] = (code == i)
    ker = np.ones(2 * win + 1, np.float32)
    counts = np.stack([np.convolve(onehot[i], ker, mode="same") for i in range(4)])
    tot = counts.sum(0)
    tot[tot == 0] = 1
    p = counts / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(p > 0, p * np.log2(p), 0.0)).sum(0)
    return (ent < thr).astype(np.uint8)


def sanitize(name: str) -> str:
    """Filesystem-safe contig name (used for per-contig ``.npz`` filenames)."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def cds_segments(feats: list[Feature], contig: str) -> dict:
    """``protein_id -> {strand, phase, segs: [(start, end), ...]}`` for a contig.

    Segments are 0-based half-open, ordered 5'->3'. ``phase`` is the phase of the
    first (5'-most) CDS segment, for aa->nt frame alignment.
    """
    by_pid: dict[str, dict] = {}
    for f in feats:
        if f.contig != contig or f.ftype != "CDS":
            continue
        # protein.faa headers are the protein accession (e.g. NP_414542.1), which
        # the GFF CDS exposes as Name= (or ID=cds-<acc>); protein_id= is absent in
        # RefSeq prok GFF3. Match on Name first, then strip the cds- prefix from ID.
        pid = f.attrs.get("Name") or f.attrs.get("protein_id")
        if not pid:
            idv = f.attrs.get("ID", "")
            pid = idv[4:] if idv.startswith("cds-") else idv
        if not pid:
            continue
        d = by_pid.setdefault(pid, {"strand": f.strand, "phase": f.phase, "segs": []})
        d["segs"].append((f.start, f.end, f.phase))
    out: dict[str, dict] = {}
    for pid, d in by_pid.items():
        segs = sorted(d["segs"], key=lambda s: s[0])
        if d["strand"] == "-":
            segs = segs[::-1]  # 5'->3'
        phase0 = segs[0][2] if segs and segs[0][2] >= 0 else 0
        out[pid] = {"strand": d["strand"], "phase": phase0,
                    "segs": [(s[0], s[1]) for s in segs]}
    return out


# --- CLI ---------------------------------------------------------------------

def build_accession_tracks(acc: str, genomes_dir: Path, out: Path) -> tuple[int, int]:
    """Build + write all per-nt tracks for one accession.

    Reads ``<genomes_dir>/<acc>/{genomic.fna,genomic.gff}`` and writes one
    ``<out>/<acc>/<contig>.npz`` per contig >= 500 nt, plus ``contigs.json``,
    ``cds_map.json`` and ``products.json``. Returns ``(n_contigs, n_cds)``.
    """
    adir = genomes_dir / acc
    fna = adir / "genomic.fna"
    gff = adir / "genomic.gff"
    if not (fna.exists() and gff.exists()):
        return 0, 0
    odir = out / acc
    odir.mkdir(parents=True, exist_ok=True)
    seqs = read_fasta(fna)
    feats = read_gff(gff)
    contigs: dict[str, dict] = {}
    products: list[dict] = []
    cds_map: dict[str, dict] = {}
    for cname, seq in seqs.items():
        L = len(seq)
        if L < 500:
            continue
        tr = build_contig_tracks(cname, seq, feats)
        arrs = dict(tr.arrays)
        arrs["lowcomplex"] = lowcomplex_track(seq)
        np.savez_compressed(odir / f"{sanitize(cname)}.npz", **arrs)
        contigs[cname] = {"length": L, "file": f"{sanitize(cname)}.npz"}
        cds_map[cname] = cds_segments(feats, cname)
    # gene products (for keyword track + interpretation)
    for f in feats:
        if f.ftype == "CDS":
            products.append({"contig": f.contig, "start": f.start, "end": f.end,
                             "strand": f.strand, "product": f.attrs.get("product", ""),
                             "protein_id": f.attrs.get("protein_id", "")})
    (odir / "contigs.json").write_text(json.dumps(contigs))
    (odir / "cds_map.json").write_text(json.dumps(cds_map))
    (odir / "products.json").write_text(json.dumps(products))
    return len(contigs), len(products)


def main(argv: list[str] | None = None) -> int:
    """CLI: build per-nt tracks for a shard of the genome panel."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--genomes", type=Path, default=None,
        help="directory of <acc>/{genomic.fna,genomic.gff} "
             "(default: config.data_paths().panel_genomes)")
    ap.add_argument("--out", type=Path, required=True, help="output track directory")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args(argv)

    genomes_dir = args.genomes or config.data_paths().panel_genomes
    args.out.mkdir(parents=True, exist_ok=True)
    accs = sorted(d.name for d in Path(genomes_dir).iterdir() if d.is_dir())
    accs = accs[args.shard::args.nshards]
    print(f"[tracks shard {args.shard}/{args.nshards}] {len(accs)} genomes")

    for acc in accs:
        n_contigs, n_cds = build_accession_tracks(acc, Path(genomes_dir), args.out)
        if n_contigs == 0 and n_cds == 0:
            print(f"  skip {acc} (missing fna/gff or no contigs)")
        else:
            print(f"  {acc}: {n_contigs} contigs, {n_cds} CDS")
    print(f"[tracks shard {args.shard}] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
