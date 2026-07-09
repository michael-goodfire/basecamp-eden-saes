"""Build ``labels.json`` (``{ann_key: human_label}``) for the pfam + cath layers.

Pfam labels come from the panel's ``ann_meta.json`` (``ann_label`` map); CATH
superfamily names are harvested from the built CATH ``domains.tsv`` files.
"""

from __future__ import annotations

import argparse
import glob
import json


def main(argv: list[str] | None = None) -> int:
    """Merge Pfam and CATH names into a single labels map."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann-meta", required=True, help="panel ann_meta.json (pfam labels)")
    ap.add_argument("--cath-tsv-dir", required=True, help="dir of CATH domains.tsv files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    labels: dict[str, str] = {}
    meta = json.load(open(args.ann_meta))
    for ann, lab in meta.get("ann_label", {}).items():
        labels[ann] = lab
    for f in glob.glob(f"{args.cath_tsv_dir}/*.tsv"):
        with open(f) as fh:
            fh.readline()
            for line in fh:
                v = line.rstrip("\n").split("\t")
                if len(v) >= 6:
                    ann = "cath|" + v[4]
                    if ann not in labels and v[5]:
                        labels[ann] = v[5]
    json.dump(labels, open(args.out, "w"))
    n_cath = sum(k.startswith("cath|") for k in labels)
    n_pfam = sum(k.startswith("pfam|") for k in labels)
    print(f"labels: {len(labels)} ({n_cath} cath, {n_pfam} pfam)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
