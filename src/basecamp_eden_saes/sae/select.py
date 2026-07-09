"""Apply the selection gates to a model's sweep and pick the best SAE.

Gates (on the held-out slice):
  * dead-feature fraction < 10%  -- HARD gate (separates a usable ef from an
    over-provisioned one).
  * L0 in [k/2, 2k]              -- sane sparsity band around the target k.
Among configs passing both gates, pick the highest variance-explained (VE).
If none pass, report it as a finding (over-provisioned vs under-trained) rather
than silently picking a failing config.

Writes ``selection.json`` next to each input ``sweep_results.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEAD_MAX = 0.10


def select_one(results_path: Path) -> dict:
    """Apply the gates to one ``sweep_results.json`` and return the selection dict."""
    data = json.loads(results_path.read_text())
    configs = data["configs"]
    rows = []
    for label, m in configs.items():
        k = m["k"]
        l0 = m["l0"]
        dead = m["dead_fraction"]
        l0_ok = (k / 2.0) <= l0 <= (2.0 * k)
        dead_ok = dead < DEAD_MAX
        rows.append(
            {
                "label": label,
                "ef": m["ef"],
                "k": k,
                "d_sae": m["d_sae"],
                "variance_explained": m["variance_explained"],
                "l0": l0,
                "dead_fraction": dead,
                "coherence_mean": m.get("coherence_mean"),
                "l0_ok": l0_ok,
                "dead_ok": dead_ok,
                "pass": l0_ok and dead_ok,
                "checkpoint": m.get("checkpoint"),
            }
        )
    passing = [r for r in rows if r["pass"]]
    best = max(passing, key=lambda r: r["variance_explained"]) if passing else None

    # diagnosis when nothing passes
    diagnosis = None
    if best is None:
        n_dead_fail = sum(1 for r in rows if not r["dead_ok"])
        n_l0_fail = sum(1 for r in rows if not r["l0_ok"])
        if n_dead_fail >= n_l0_fail and n_dead_fail > 0:
            diagnosis = (
                "No config clears the <10% dead-feature gate: dictionaries are "
                "over-provisioned for this model/layer (too many features for the "
                "effective rank) and/or under-trained. Lower ef or train longer."
            )
        else:
            diagnosis = (
                "L0 out of the [k/2, 2k] band for all configs: check the BatchTopK "
                "threshold / k selection at eval."
            )

    rows.sort(key=lambda r: (r["ef"], r["k"]))
    return {
        "model": data["model"],
        "layer": data["layer"],
        "best": best["label"] if best else None,
        "best_config": best,
        "rationale": (
            f"Highest held-out VE ({best['variance_explained']:.4f}) among "
            f"{len(passing)} configs passing both gates (dead<10%, L0 in [k/2,2k])."
            if best
            else diagnosis
        ),
        "dead_max": DEAD_MAX,
        "n_passing": len(passing),
        "configs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``bes-select-sae``."""
    p = argparse.ArgumentParser(description="Select the best SAE from a sweep.")
    p.add_argument("results", nargs="+", help="one or more sweep_results.json paths")
    args = p.parse_args(argv)
    for path in args.results:
        p_ = Path(path)
        sel = select_one(p_)
        out = p_.parent / "selection.json"
        out.write_text(json.dumps(sel, indent=2))
        logger.info(
            "[%s] best=%s (%d passing) -> %s",
            sel["model"],
            sel["best"],
            sel["n_passing"],
            out,
        )
        for r in sel["configs"]:
            flag = "PASS" if r["pass"] else ("dead" if not r["dead_ok"] else "L0")
            logger.info(
                "  %10s  VE=%.4f  L0=%6.1f  dead=%6.2f%%  [%s]",
                r["label"],
                r["variance_explained"],
                r["l0"],
                r["dead_fraction"] * 100,
                flag,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
