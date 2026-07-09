"""Stream-filter the ~17.8 GB TED domain summary to our mapped UniProt accessions.

TED columns (tab-separated): 0 ted_id (``AF-<UniProt>-F1-model_v4_TEDxx``),
1 md5, 2 consensus, 3 boundaries (aa, discontinuous joined by ``_``, e.g.
``11-41_290-389``), 4 length, 5 nseg, ... 13 CATH superfamily (``-`` if none),
... 19 organism, 20 lineage. We keep uniprot, ted_id, consensus, boundaries,
cath and taxid, gzip-streamed from disk into ``ted_filtered.tsv``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict

AFRE = re.compile(r"^AF-([A-Z0-9]+)-F1")


def main(argv: list[str] | None = None) -> int:
    """Stream-filter the TED summary gzip down to the mapped accessions."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ted-gz", required=True, help="TED domain summary .gz")
    ap.add_argument("--mapping", required=True, help="refseq2uniprot.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    m = json.load(open(args.mapping))
    keep: set[str] = set()
    for v in m.values():
        for acc, _tax in v:
            keep.add(acc)

    n = kept = 0
    per_uni: dict[str, int] = defaultdict(int)
    with gzip.open(args.ted_gz, "rt") as fh, open(args.out, "w") as w:
        w.write("uniprot\tted_id\tconsensus\tboundaries\tcath\ttaxid\torganism\n")
        for line in fh:
            n += 1
            tab = line.find("\t")
            if tab < 0:
                continue
            tedid = line[:tab]
            mo = AFRE.match(tedid)
            if not mo:
                continue
            uni = mo.group(1)
            if uni not in keep:
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 14:
                continue
            cath = f[13]
            # taxid from the proteome-tax_id-XXXXX field (col 12) if present
            taxid = ""
            mt = re.search(r"tax_id-(\d+)", f[12]) if len(f) > 12 else None
            if mt:
                taxid = mt.group(1)
            org = f[19] if len(f) > 19 else ""
            w.write(f"{uni}\t{tedid}\t{f[2]}\t{f[3]}\t{cath}\t{taxid}\t{org}\n")
            kept += 1
            per_uni[uni] += 1
    print(
        f"scanned {n:,}, kept {kept:,}, distinct uniprot with TED {len(per_uni):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
