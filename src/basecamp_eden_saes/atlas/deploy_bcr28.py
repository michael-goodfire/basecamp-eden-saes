"""Assemble the final bcr28 viewer data dir from #3's finalized viewer + the
recomputed (overlap-metric) association layer + the per-exemplar locus fix.

Runs AFTER #3 finalizes its bcr28 viewer (exemplars / logos / structures /
continuous tracks). To avoid racing on #3's in-place files, this copies #3's
viewer into our artifact dir and edits the COPY:

  1. overwrite rates.json / span_rates.json / index_rows.json with the recomputed
     overlap-metric versions (from build_associations);
  2. per feature JSON: replace ``detected`` with the recomputed entries (span-level
     overlap + position-level lift), and rewrite every exemplar's ``prot`` + ``genes``
     to the specific CDS copy at its locus (atlas.exemplar_loci -- fixes the
     aggregated-coordinate gene-track + structure-paint bug);
  3. update names.json for the new layers (rfam / mge / reg / ec / ko / cog) and drop
     GO from the featured set.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

from .exemplar_loci import GenomeCDS, fix_exemplar

NEW_CLASSES = ("rfam", "mge", "reg", "ec", "ko", "cog")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src-viewer", required=True, help="#3 finalized bcr28 viewer dir")
    ap.add_argument("--assoc-dir", required=True, help="build_associations output")
    ap.add_argument("--panel", required=True, help="ncbi_dataset/data dir (GFFs) for the locus fix")
    ap.add_argument("--names-add", nargs="*", default=[], help="label jsons to fold into names.json")
    ap.add_argument("--out", required=True, help="final viewer dir (created)")
    ap.add_argument("--drop-go", action="store_true")
    a = ap.parse_args(argv)

    src, out = Path(a.src_viewer), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"copying {src} -> {out}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)

    # 1. swap association sidecars
    detected = json.load(open(Path(a.assoc_dir) / "detected_by_feature.json"))
    for name in ("rates.json", "span_rates.json", "index_rows.json"):
        s = Path(a.assoc_dir) / name
        if s.exists():
            shutil.copy2(s, out / name)
    print(f"detected features: {len(detected)}")

    # 3. names.json
    names_path = out / "names.json"
    names = json.load(open(names_path)) if names_path.exists() else {}
    if a.drop_go:
        names.pop("go", None)
        names.pop("pfam2go", None)
    for lf in a.names_add:
        if not os.path.exists(lf):
            continue
        j = json.load(open(lf))
        for ann, nm in j.items():
            cls, _, rest = ann.partition("|")
            names.setdefault(cls, {})[rest] = nm
    json.dump(names, open(names_path, "w"))

    # 2. per feature JSON: detected + exemplar locus fix
    gcache: dict[str, GenomeCDS] = {}

    def gcds(acc):
        if acc not in gcache:
            gcache[acc] = GenomeCDS(f"{a.panel}/{acc}/genomic.gff")
        return gcache[acc]

    n_files = n_fixed_ex = 0
    for fp in sorted(glob.glob(str(out / "feature" / "latent_*.json"))):
        fid = str(int(Path(fp).stem.split("_")[1]))
        d = json.load(open(fp))
        d["detected"] = detected.get(fid, [])
        if a.drop_go:
            d["top_annotations"] = [e for e in (d.get("top_annotations") or [])
                                    if e.get("class") != "go"]
        for ex in (d.get("exemplars") or []):
            acc = ex.get("acc")
            if not acc:
                continue
            gff = f"{a.panel}/{acc}/genomic.gff"
            if not os.path.exists(gff):
                continue
            try:
                fix_exemplar(ex, gcds(acc))
                n_fixed_ex += 1
            except Exception as e:  # never let one exemplar abort the deploy
                print(f"  warn: {acc} exemplar fix failed: {e}")
        with open(fp, "w") as fh:
            json.dump(d, fh)
        n_files += 1
        if n_files % 5000 == 0:
            print(f"  {n_files} feature files...")
    print(f"rewrote {n_files} feature files; fixed {n_fixed_ex} exemplars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
