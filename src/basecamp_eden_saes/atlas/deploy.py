"""Apply the staged corrected atlas layer, then assemble the standalone viewer.

Two steps, exposed as subcommands of one CLI (``atlas.deploy``):

``apply``    Apply the staged corrected TED layer to the canonical viewer in place.
             Backs up every canonical file it overwrites (changed feature files +
             index_rows.json + rates.json per dict) to ``<backup-root>/<dict>/`` for
             rollback, then copies the staged files over the canonical location.
             af_cache and unchanged feature files are untouched.

``assemble`` Compose one standalone viewer under ``<app>/viewer``:
               - front-end (js/css/index.html/dicts.json) from ``--viewer-src``
                 (the repo's viewer/: canonical #53 JS + vendored NGL)
               - the three atlas dicts hardlinked from ``--atlas-root``
               - names.json from ``--atlas-root``
               - af_cache/ hardlinked from ``--af-cache`` (AlphaFold DB cache)
               - struct_index.json + struct_coverage.json from ``--struct``
             Hardlinks (cp -al) avoid duplicating ~100 GB; dest + artifacts share
             one filesystem. Also writes ``<app>/manifest.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

from .. import config

# Canonical on-cluster defaults (overridable via CLI).
DEFAULT_CANON = "/mnt/data/artifacts/silico/eden_feature_viewer_canonical/viewer"
DEFAULT_VIEWER_SRC = str(Path(__file__).resolve().parents[3] / "viewer")

DICTS = ("og2", "bcr_k16", "bcr_k64")


# --------------------------------------------------------------------------- #
# Step: apply staged corrected layer onto the canonical atlas
# --------------------------------------------------------------------------- #
def apply_staged_layer(*, staging_root: str, canon: str, backup_root: str,
                       dicts=DICTS) -> None:
    """Copy staged corrected files over the canonical viewer, backing up originals."""
    for dct in dicts:
        stage = f"{staging_root}/{dct}"
        backup = f"{backup_root}/{dct}"
        os.makedirs(f"{backup}/feature", exist_ok=True)
        changed = sorted(glob.glob(f"{stage}/feature/latent_*.json"))
        # backup index_rows + rates (once) + each changed feature file
        for name in ["index_rows.json", "rates.json"]:
            src = f"{canon}/{dct}/{name}"
            if os.path.exists(src) and not os.path.exists(f"{backup}/{name}"):
                shutil.copy2(src, f"{backup}/{name}")
        n_bk = 0
        for fp in changed:
            base = os.path.basename(fp)
            canon_fp = f"{canon}/{dct}/feature/{base}"
            bk_fp = f"{backup}/feature/{base}"
            if os.path.exists(canon_fp) and not os.path.exists(bk_fp):
                shutil.copy2(canon_fp, bk_fp)
                n_bk += 1
        # overwrite canonical with staged
        for fp in changed:
            shutil.copy2(fp, f"{canon}/{dct}/feature/{os.path.basename(fp)}")
        shutil.copy2(f"{stage}/index_rows.json", f"{canon}/{dct}/index_rows.json")
        shutil.copy2(f"{stage}/rates.json", f"{canon}/{dct}/rates.json")
        print(f"{dct}: backed up {n_bk} feature files + index_rows + rates; "
              f"applied {len(changed)} corrected feature files")
    print("DONE deploy apply")


# --------------------------------------------------------------------------- #
# Step: assemble the standalone viewer
# --------------------------------------------------------------------------- #
def hardlink_tree(src, dst) -> None:
    """cp -al src dst (recursive hardlink); falls back to copy across filesystems."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    r = subprocess.run(["cp", "-al", src, dst], capture_output=True, text=True)
    if r.returncode != 0:
        # cross-device or hardlink unsupported -> full copy
        print(f"  cp -al failed ({r.stderr.strip()}); falling back to cp -a", flush=True)
        subprocess.run(["cp", "-a", src, dst], check=True)


def assemble_viewer(*, app: str, viewer_src: str, atlas_root: str, af_cache: str,
                    struct: str, dicts=DICTS) -> None:
    """Compose the standalone viewer (front-end + atlas dicts + af_cache + struct)."""
    vdst = Path(app) / "viewer"
    vdst.mkdir(parents=True, exist_ok=True)
    # front-end js is copied fresh; wipe the vendored js dir so removed files don't linger
    if (vdst / "js").exists():
        shutil.rmtree(vdst / "js")

    # 1) front-end (canonical #53 JS + vendored NGL + shell)
    src = Path(viewer_src)
    (vdst / "js" / "vendor").mkdir(parents=True, exist_ok=True)
    for f in (src / "js").glob("*.js"):
        shutil.copy(f, vdst / "js" / f.name)
    for f in (src / "js" / "vendor").glob("*"):
        shutil.copy(f, vdst / "js" / "vendor" / f.name)
    for f in ["index.html", "styles.css", "dicts.json"]:
        if (src / f).exists():
            shutil.copy(src / f, vdst / f)
    shutil.copy(f"{atlas_root}/names.json", vdst / "names.json")

    # 2) atlas dicts (hardlink; all three dense)
    for d in dicts:
        print(f"hardlinking atlas dict {d} ...", flush=True)
        hardlink_tree(f"{atlas_root}/{d}", str(vdst / d))

    # 3) af_cache (hardlink AFDB + ESMFold2 CIFs)
    print("hardlinking af_cache ...", flush=True)
    hardlink_tree(af_cache, str(vdst / "af_cache"))

    # 4) structure index + coverage
    shutil.copy(f"{struct}/struct_index.json", vdst / "struct_index.json")
    shutil.copy(f"{struct}/struct_coverage.json", vdst / "struct_coverage.json")

    # manifest: the report (index.html) is the entrypoint; it links to viewer/index.html
    with open(Path(app) / "manifest.json", "w") as fh:
        json.dump({"type": "static", "entrypoint": "index.html"}, fh)

    n_cif = len(list((vdst / "af_cache").glob("*.cif")))
    with open(vdst / "struct_coverage.json") as fh:
        cov = json.load(fh)
    print(f"deployed viewer -> {vdst}", flush=True)
    print(f"  af_cache CIFs: {n_cif}", flush=True)
    for d in dicts:
        print(f"  {d}: {cov[d]['coverage_pct']}% structure coverage", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_apply = sub.add_parser("apply", help="apply the staged corrected layer onto the canonical atlas")
    p_apply.add_argument("--staging-root", required=True, help="staging tree root (<root>/<dict>/)")
    p_apply.add_argument("--backup-root", required=True, help="dir to back up overwritten canonical files")
    p_apply.add_argument("--canon", default=DEFAULT_CANON, help="canonical viewer dir")

    p_asm = sub.add_parser("assemble", help="assemble the standalone viewer")
    p_asm.add_argument("--app", required=True, help="destination root (viewer is written under <app>/viewer)")
    p_asm.add_argument("--atlas-root", required=True,
                       help="dir with og2/bcr_k64/bcr_k16 subdirs (all dense) + names.json")
    p_asm.add_argument("--struct", required=True,
                       help="dir with struct_index.json + struct_coverage.json")
    p_asm.add_argument("--viewer-src", default=DEFAULT_VIEWER_SRC,
                       help="front-end viewer/ dir (default: this repo's viewer/)")
    p_asm.add_argument("--af-cache", default=None,
                       help="AlphaFold DB structure cache dir; "
                            "defaults to config.data_paths().af_cache")
    args = ap.parse_args(argv)

    if args.command == "apply":
        apply_staged_layer(staging_root=args.staging_root, canon=args.canon,
                           backup_root=args.backup_root)
    elif args.command == "assemble":
        af_cache = args.af_cache if args.af_cache is not None else str(config.data_paths().af_cache)
        assemble_viewer(app=args.app, viewer_src=args.viewer_src,
                        atlas_root=args.atlas_root, af_cache=af_cache, struct=args.struct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
