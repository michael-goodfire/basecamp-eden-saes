"""Regulatory-motif span layer (``reg|<type>``) for the annotation panel v2.

Surfaces the verified per-instance regulatory spans that the ``annot_v2`` build
(#38) already deposits in each genome's ``spans.jsonl`` as discrete, coordinate-
resolved spans in the standard span-TSV format consumed by
:mod:`basecamp_eden_saes.autointerp.join_cover`.

No fresh motif scan: the ``annot_v2`` ``spans.jsonl`` already carries per-instance
``regulatory`` (rbs / promoter_-10 / promoter_-35 / terminator), ``operon``
(operon / operon_internal) and ``replication`` (dnaA_box / oriC / ter) spans with
0-based half-open ``[nt_start, nt_end)`` contig coordinates and a reading strand.
We map the plan's six featured types to ``reg|<key>`` keys and re-emit them; the
dense methylation-motif ('motif' layer: GATC / CCWGG / GANTC) tracks are excluded
(compositional, millions of sites, not discrete regulatory elements).

Output per genome: ``<out>/<accession>.tsv`` with lines
``reg|<key>\\t<contig>\\t<strand>\\t<start>\\t<end>``.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

# (layer, ann_type) in annot_v2 spans.jsonl  ->  reg| key
TYPE_MAP: dict[tuple[str, str], str] = {
    ("regulatory", "rbs"): "rbs",
    ("regulatory", "promoter_-10"): "promoter_-10",
    ("regulatory", "promoter_-35"): "promoter_-35",
    ("regulatory", "terminator"): "terminator",
    ("operon", "operon"): "operon",
    ("operon", "operon_internal"): "operon_internal",
    ("replication", "dnaA_box"): "dnaa_box",
    ("replication", "oriC"): "oriC",
    ("replication", "ter"): "ter",
}

# id -> human label for names.json
LABELS: dict[str, str] = {
    "reg|rbs": "ribosome-binding site (Shine-Dalgarno)",
    "reg|promoter_-10": "promoter -10 box (Pribnow)",
    "reg|promoter_-35": "promoter -35 box",
    "reg|terminator": "rho-independent terminator",
    "reg|operon": "operon (co-transcribed gene run)",
    "reg|operon_internal": "operon-internal region",
    "reg|dnaa_box": "DnaA box (replication-origin motif)",
    "reg|oriC": "replication origin (oriC)",
    "reg|ter": "replication terminus (ter)",
}


def convert_accession(spans_jsonl: str | Path, out_tsv: str | Path) -> Counter:
    """Reformat one genome's annot_v2 ``spans.jsonl`` into a ``reg|`` span TSV."""
    counts: Counter = Counter()
    with open(out_tsv, "w") as w:
        for line in open(spans_jsonl):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = TYPE_MAP.get((r.get("layer"), r.get("ann_type")))
            if key is None:
                continue
            ann = f"reg|{key}"
            w.write(f"{ann}\t{r['contig']}\t{r['strand']}\t{int(r['nt_start'])}\t{int(r['nt_end'])}\n")
            counts[ann] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the reg| regulatory-span layer from annot_v2 spans.jsonl.")
    ap.add_argument("--annot-v2", required=True, help="v1 bundle layers/annot_v2 dir")
    ap.add_argument("--out", required=True, help="output layer dir (one <acc>.tsv per genome)")
    ap.add_argument("--labels-out", default="", help="optional labels json path")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    n = 0
    for d in sorted(glob.glob(str(Path(args.annot_v2) / "GCF_*"))):
        sj = Path(d) / "spans.jsonl"
        if not sj.exists():
            continue
        acc = Path(d).name
        c = convert_accession(sj, out / f"{acc}.tsv")
        total.update(c)
        n += 1
    print(f"regulatory layer: {n} genomes")
    for k in sorted(total):
        print(f"  {k}: {total[k]}")
    if args.labels_out:
        json.dump(LABELS, open(args.labels_out, "w"), indent=0)
        print(f"wrote labels -> {args.labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
