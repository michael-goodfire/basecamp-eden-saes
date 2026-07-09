"""Emit a per-genome Pfam ``spans.tsv`` (the fold-metric validation annotation set).

Reads the panel's per-genome ``domains.tsv`` and writes the uniform span format::

    ann_key<TAB>contig<TAB>strand<TAB>nt_start<TAB>nt_end

with ``ann_key = pfam|PF00389`` (Pfam accession, version stripped).
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Convert a Pfam ``domains.tsv`` into the uniform span TSV."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", required=True, help="per-genome domains.tsv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    n = 0
    with open(args.domains) as fh, open(args.out, "w") as w:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in fh:
            v = line.rstrip("\n").split("\t")
            acc = v[idx["pfam_acc"]].split(".")[0]  # PF00389.37 -> PF00389
            w.write(
                f"pfam|{acc}\t{v[idx['contig']]}\t{v[idx['strand']]}\t"
                f"{v[idx['nt_start']]}\t{v[idx['nt_end']]}\n"
            )
            n += 1
    print(f"{args.out}: {n} pfam spans", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
