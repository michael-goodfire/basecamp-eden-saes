"""Per-feature GLOBAL peak activation across the panel (for peak-firing cover).

The peak-firing cover rule counts a span as covered by feature f only if f fires
inside it at ``>= fraction * peak[f]``, where ``peak[f]`` is f's maximum activation
value anywhere in the code store. This module computes ``peak`` (shape [F]).

Per-genome ``build`` writes a partial max; ``merge`` takes the element-wise max
across partials.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from . import metrics as M


def build_genome(accession: str, codes_dir: str | Path, n_features: int = M.F_DEFAULT) -> np.ndarray:
    peak = np.zeros(n_features, np.float32)
    for fn in glob.glob(str(Path(codes_dir) / accession / "*.npz")):
        with np.load(fn) as z:
            idx = z["indices"]
            val = z["values"].astype(np.float32)
        if idx.size:
            np.maximum.at(peak, idx, val)
    return peak


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-feature global peak activation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--acc", required=True)
    b.add_argument("--codes-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--F", type=int, default=M.F_DEFAULT)
    m = sub.add_parser("merge")
    m.add_argument("--partials", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        peak = build_genome(a.acc, a.codes_dir, n_features=a.F)
        np.savez_compressed(a.out, peak=peak)
        print(f"{a.acc}: peak max={peak.max():.3f} nonzero={int((peak>0).sum())} -> {a.out}")
        return 0
    peak = None
    files = sorted(glob.glob(a.partials))
    for f in files:
        with np.load(f) as z:
            p = z["peak"]
        peak = p.copy() if peak is None else np.maximum(peak, p)
    np.savez_compressed(a.out, peak=peak)
    print(f"merged {len(files)} partials -> {a.out}; peak max={peak.max():.3f} nonzero={int((peak>0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
