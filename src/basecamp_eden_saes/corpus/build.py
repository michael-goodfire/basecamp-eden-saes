"""OpenGenome2 corpus builder for EDEN-7B SAE training.

Streams OG2 GTDB (``.tar`` of FASTA) and metagenome (``.fasta.gz``) shards from
HuggingFace, splits each contig at non-ACGT runs (so the corpus is pure
uppercase A/C/G/T -- no ``N``, no ambiguity codes, which are out-of-distribution
for EDEN-7B-BCR), runs tantan to score low-complexity / simple-repeat content
and drops windows above a threshold, windows the survivors to a fixed length,
and byte-level tokenizes to ``uint8``.

Byte-level tokenization note: EDEN's tokenizer is a GPT-2-style ByteLevel BPE
where printable ASCII maps to its byte value (``A``=65, ``C``=67, ``G``=71,
``T``=84). For a pure-uppercase-ACGT string this is exactly the raw ASCII bytes,
so ``seq.encode("ascii")`` reproduces the tokenizer's ids 1:1 (asserted in
tests against the real tokenizer). This is far faster than calling the HF
tokenizer per window.

Output layout (a reusable corpus artifact)::

    <out>/
      tokens.bin      flat uint8, concatenation of every window's token ids
      index.npy       int64 [n_windows, 2]  = (byte offset into tokens.bin, length)
      sequences.db    sqlite: per-window provenance
      meta.json       build parameters + summary counts

The flat store keeps exactly 1 byte/nt (no padding waste); a window is
recovered by slicing ``tokens.bin[offset : offset+length]``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sqlite3
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from basecamp_eden_saes import config

logger = logging.getLogger(__name__)

# --- constants ---------------------------------------------------------------

ACGT_BYTES = (65, 67, 71, 84)  # A, C, G, T
# uint8 lookup: True for A/C/G/T (uppercased), False otherwise
_IS_ACGT = np.zeros(256, dtype=bool)
for _b in ACGT_BYTES:
    _IS_ACGT[_b] = True
# uppercase map for bytes a-z -> A-Z
_UPPER = np.arange(256, dtype=np.uint8)
for _c in range(ord("a"), ord("z") + 1):
    _UPPER[_c] = _c - 32

# Public HuggingFace dataset ids (not local paths); safe as module defaults.
HF_REPO_PREFIX = "datasets/arcinstitute/opengenome2"
DEFAULT_GTDB_FILES = ["fasta/gtdb_v220/v214/batch1.tar"]
DEFAULT_METAG_FILES = ["fasta/metagenomes/filtered_metagenomes_pt1.fasta.gz"]


# --- FASTA parsing -----------------------------------------------------------


def iter_fasta(text_stream: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` from a FASTA text stream (line iterator)."""
    header: str | None = None
    chunks: list[str] = []
    for line in text_stream:
        if not line:
            continue
        if line[0] == ">":
            if header is not None:
                yield header, "".join(chunks)
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


# --- sequence processing -----------------------------------------------------


def split_at_non_acgt(seq: str) -> list[tuple[int, np.ndarray]]:
    """Uppercase, then split into maximal runs of A/C/G/T.

    Returns a list of ``(start_offset_in_contig, uint8_array)`` for each run.
    Any non-ACGT byte (N, ambiguity codes, etc.) is a split point and is
    dropped, so runs fall on real assembly gaps and carry no OOD tokens.
    """
    if not seq:
        return []
    raw = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    upper = _UPPER[raw]
    mask = _IS_ACGT[upper]
    if not mask.any():
        return []
    # find run boundaries on the boolean mask
    out: list[tuple[int, np.ndarray]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append((i, upper[i:j].copy()))
        i = j
    return out


def low_complexity_fraction(
    window: np.ndarray, mask_repeats_fn: Callable[[str], str]
) -> float:
    """Fraction of a window flagged low-complexity / simple-repeat by tantan.

    ``mask_repeats_fn`` takes an uppercase DNA string and returns the masked
    string (repeats lowercased). We score the masked fraction only; the window's
    original uppercase bases are kept for survivors.
    """
    s = window.tobytes().decode("ascii")
    masked = mask_repeats_fn(s)
    if len(masked) != len(s):  # defensive
        return 0.0
    # tantan lowercases masked positions; count lowercase a/c/g/t/n
    m = np.frombuffer(masked.encode("ascii"), dtype=np.uint8)
    lc = ((m >= ord("a")) & (m <= ord("z"))).sum()
    return float(lc) / float(len(s))


def windows_of(
    run: np.ndarray, window: int, min_window: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Non-overlapping windows within a single run; yields ``(offset, arr)``.

    Tail window shorter than ``min_window`` is dropped.
    """
    n = len(run)
    for start in range(0, n, window):
        w = run[start : start + window]
        if len(w) < min_window:
            break
        yield start, w


# --- HF streaming sources ----------------------------------------------------


def _hf_open(path: str, retries: int = 8, backoff: float = 5.0):
    """Open a remote OG2 file, retrying transient HF API/network errors.

    The HF file API intermittently returns 504/503/timeout on long streaming
    builds; retry with exponential backoff so a multi-hour corpus build is not
    killed by a single transient gateway error.
    """
    from huggingface_hub import HfFileSystem

    last: Exception | None = None
    for attempt in range(retries):
        try:
            fs = HfFileSystem()
            return fs.open(f"{HF_REPO_PREFIX}/{path}", "rb")
        except Exception as e:  # noqa: BLE001 - retry any transient open error
            last = e
            wait = backoff * (2**attempt)
            logger.warning(
                "[hf_open] attempt %d/%d failed: %r; retrying in %.0fs",
                attempt + 1,
                retries,
                e,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"_hf_open failed after {retries} retries: {last!r}")


def stream_metagenome(path: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(source, contig_id, seq)`` from a gzipped metagenome FASTA."""
    raw = _hf_open(path)
    gz = gzip.GzipFile(fileobj=raw)
    text = io.TextIOWrapper(gz, encoding="ascii", errors="replace")
    for header, seq in iter_fasta(text):
        yield "metagenome", header.split()[0] if header else "?", seq


def _maybe_gunzip(fileobj, name: str):
    # detect gzip by extension or magic bytes
    if name.endswith(".gz"):
        return gzip.GzipFile(fileobj=fileobj)
    head = fileobj.read(2)
    rest = io.BytesIO(head + fileobj.read())
    rest.seek(0)
    if head == b"\x1f\x8b":
        return gzip.GzipFile(fileobj=rest)
    return rest


def stream_gtdb_tar(path: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(source, contig_id, seq)`` from a GTDB ``.tar`` of FASTA files.

    Streamed sequentially (``mode='r|'``) so we never download the whole tar.
    """
    raw = _hf_open(path)
    tar = tarfile.open(fileobj=raw, mode="r|")
    for member in tar:
        if not member.isfile():
            continue
        f = tar.extractfile(member)
        if f is None:
            continue
        data = _maybe_gunzip(f, member.name)
        text = io.TextIOWrapper(data, encoding="ascii", errors="replace")
        genome = Path(member.name).name
        for header, seq in iter_fasta(text):
            cid = f"{genome}:{header.split()[0]}" if header else genome
            yield "gtdb", cid, seq


def stream_source(kind: str, files: list[str]) -> Iterator[tuple[str, str, str]]:
    """Yield ``(source, contig_id, seq)`` from every file for a source ``kind``."""
    for path in files:
        if kind == "gtdb":
            yield from stream_gtdb_tar(path)
        elif kind == "metagenome":
            yield from stream_metagenome(path)
        else:
            raise ValueError(kind)


# --- corpus build ------------------------------------------------------------


@dataclass
class BuildStats:
    """Per-source running counters emitted in ``meta.json``."""

    kind: str
    contigs: int = 0
    runs: int = 0
    windows_kept: int = 0
    windows_dropped_lowcx: int = 0
    windows_dropped_short: int = 0
    tokens: int = 0
    lowcx_frac_sum: float = 0.0


def _get_mask_fn() -> Callable[[str], str]:
    import pytantan

    return pytantan.mask_repeats


@dataclass
class CorpusWriter:
    """Writes the flat token store, per-window index, and provenance sqlite db."""

    out_dir: Path
    tokens_fh: io.BufferedWriter = field(init=False)
    index: list[tuple[int, int]] = field(default_factory=list, init=False)
    offset: int = 0
    db: sqlite3.Connection = field(init=False)

    OWNER_MARKER = ".bes_owner"

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tokens = self.out_dir / "tokens.bin"
        marker = self.out_dir / self.OWNER_MARKER
        # Guardrail: never open a tokens.bin we don't own in write mode. A
        # foreign corpus (no ownership marker) must not be truncated; point the
        # build at a fresh/distinct path instead. Re-running our own build
        # (marker present) is allowed.
        if tokens.exists() and not marker.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing tokens.bin at {self.out_dir}: "
                f"directory is not owned by basecamp-eden-saes (no {self.OWNER_MARKER} "
                f"marker). Use a fresh or distinct output path."
            )
        marker.write_text("basecamp-eden-saes\n")
        self.tokens_fh = open(tokens, "wb")
        self.db = sqlite3.connect(self.out_dir / "sequences.db")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS windows ("
            "window_id INTEGER PRIMARY KEY, source TEXT, contig_id TEXT, "
            "seg_offset INTEGER, length INTEGER, lowcx_frac REAL)"
        )
        self.db.commit()
        self._rows: list[tuple] = []

    def add(
        self,
        source: str,
        contig_id: str,
        seg_offset: int,
        arr: np.ndarray,
        lowcx: float,
    ) -> None:
        """Append one window's tokens + provenance row."""
        wid = len(self.index)
        b = arr.astype(np.uint8).tobytes()
        self.tokens_fh.write(b)
        self.index.append((self.offset, len(b)))
        self._rows.append((wid, source, contig_id, int(seg_offset), len(b), lowcx))
        self.offset += len(b)
        if len(self._rows) >= 10000:
            self._flush_rows()

    def _flush_rows(self) -> None:
        self.db.executemany("INSERT INTO windows VALUES (?,?,?,?,?,?)", self._rows)
        self.db.commit()
        self._rows.clear()

    def close(self) -> None:
        """Flush the last rows, close files, and write ``index.npy``."""
        self._flush_rows()
        self.tokens_fh.close()
        idx = np.asarray(self.index, dtype=np.int64).reshape(-1, 2)
        np.save(self.out_dir / "index.npy", idx)
        self.db.close()


def build_side(
    writer: CorpusWriter,
    kind: str,
    files: list[str],
    target_tokens: int,
    window: int,
    min_window: int,
    lowcx_max: float,
    mask_fn: Callable[[str], str],
    log_every: int = 50_000,
) -> BuildStats:
    """Stream one source into ``writer`` until ``target_tokens`` are accepted."""
    st = BuildStats(kind=kind)
    t0 = time.time()
    for source, contig_id, seq in stream_source(kind, files):
        st.contigs += 1
        for seg_off, run in split_at_non_acgt(seq):
            st.runs += 1
            for w_off, w in windows_of(run, window, min_window):
                lc = low_complexity_fraction(w, mask_fn)
                if lc > lowcx_max:
                    st.windows_dropped_lowcx += 1
                    continue
                writer.add(source, contig_id, seg_off + w_off, w, lc)
                st.windows_kept += 1
                st.tokens += len(w)
                st.lowcx_frac_sum += lc
                if st.windows_kept % log_every == 0:
                    rate = st.tokens / max(time.time() - t0, 1e-9)
                    logger.info(
                        "[%s] kept=%d tokens=%.1fM (%.2fM nt/s) dropped_lc=%d",
                        kind,
                        st.windows_kept,
                        st.tokens / 1e6,
                        rate / 1e6,
                        st.windows_dropped_lowcx,
                    )
                if st.tokens >= target_tokens:
                    return st
    return st


def build_corpus(
    out_dir: Path,
    gtdb_files: list[str],
    metag_files: list[str],
    gtdb_tokens: int,
    metag_tokens: int,
    window: int = 4096,
    min_window: int = 1024,
    lowcx_max: float = 0.25,
    seed: int = 42,
) -> dict:
    """Build the full corpus and return the ``meta.json`` summary dict."""
    mask_fn = _get_mask_fn()
    writer = CorpusWriter(out_dir=Path(out_dir))
    stats: dict = {}
    g = build_side(
        writer, "gtdb", gtdb_files, gtdb_tokens, window, min_window, lowcx_max, mask_fn
    )
    stats["gtdb"] = g.__dict__
    m = build_side(
        writer,
        "metagenome",
        metag_files,
        metag_tokens,
        window,
        min_window,
        lowcx_max,
        mask_fn,
    )
    stats["metagenome"] = m.__dict__
    writer.close()

    # --- preconditions: no N (id 78) anywhere, no over-length windows ---
    idx = np.load(Path(out_dir) / "index.npy")
    lengths = idx[:, 1]
    assert lengths.max() <= window, f"over-length window: {lengths.max()} > {window}"
    toks = np.memmap(Path(out_dir) / "tokens.bin", dtype=np.uint8, mode="r")
    # sample-check for N (78) and any non-ACGT across the whole store
    bad = (~_IS_ACGT[toks]).sum()
    assert bad == 0, f"corpus contains {bad} non-ACGT bytes (incl. N=78)"

    meta = {
        "window": window,
        "min_window": min_window,
        "lowcx_max": lowcx_max,
        "seed": seed,
        "gtdb_files": gtdb_files,
        "metag_files": metag_files,
        "n_windows": int(len(idx)),
        "total_tokens": int(lengths.sum()),
        "stats": stats,
    }
    with open(Path(out_dir) / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("corpus meta: %s", json.dumps(meta, indent=2))
    return meta


# --- corpus reader (used by both backends) -----------------------------------


class CorpusReader:
    """Reads a built corpus: token store + index. Yields batches of windows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.index = np.load(self.path / "index.npy")
        self.tokens = np.memmap(self.path / "tokens.bin", dtype=np.uint8, mode="r")
        self.meta = json.loads((self.path / "meta.json").read_text())

    def __len__(self) -> int:
        return len(self.index)

    def get(self, i: int) -> np.ndarray:
        """Return the uint8 token array for window ``i``."""
        off, length = self.index[i]
        return np.asarray(self.tokens[off : off + length])

    def iter_batches(
        self, batch_windows: int, shuffle: bool = True, seed: int = 42
    ) -> Iterator[list[np.ndarray]]:
        """Yield lists of ``batch_windows`` token arrays over the whole corpus."""
        order = np.arange(len(self.index))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for s in range(0, len(order), batch_windows):
            yield [self.get(int(j)) for j in order[s : s + batch_windows]]


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``bes-build-corpus``."""
    p = argparse.ArgumentParser(description="Build an OG2 corpus for EDEN SAEs.")
    p.add_argument(
        "--out",
        default=None,
        help="output corpus dir (default: config.data_paths().corpus)",
    )
    p.add_argument("--gtdb-tokens", type=float, default=2.5e9)
    p.add_argument("--metag-tokens", type=float, default=2.5e9)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--min-window", type=int, default=1024)
    p.add_argument("--lowcx-max", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gtdb-files", nargs="*", default=DEFAULT_GTDB_FILES)
    p.add_argument("--metag-files", nargs="*", default=DEFAULT_METAG_FILES)
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else config.data_paths().corpus
    build_corpus(
        out_dir=out_dir,
        gtdb_files=args.gtdb_files,
        metag_files=args.metag_files,
        gtdb_tokens=int(args.gtdb_tokens),
        metag_tokens=int(args.metag_tokens),
        window=args.window,
        min_window=args.min_window,
        lowcx_max=args.lowcx_max,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
