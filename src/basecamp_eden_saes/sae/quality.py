"""SAE sweep quality analysis: throughput extrapolation + convergence check.

Two independent, dependency-light analyses used to close out a sweep:

* **Throughput extrapolation** (:func:`extrapolate` / :func:`summarize`) takes a
  benchmark's per-token cost measurements and projects the wall-clock and peak
  activation-disk cost of the planned ``tokens x epochs`` sweep for both the
  pre-harvest and on-the-fly backends, applying the harvesting-strategy decision
  rule.

* **Convergence check** (:func:`check_model` + :func:`main`) reads each trained
  run's ``losses/global_variance_explained`` and
  ``features/percent_dead_features`` history from W&B, fits a slope over the last
  ``--tail-frac`` of training, and flags a run as plateaued only if the
  variance-explained curve is essentially flat there. ``wandb`` is imported
  lazily inside :func:`main`, so importing this module never requires it.

Throughput cost model (per token), measured by the benchmark:
  f_fwd    EDEN partial-forward seconds/token  (Arm A one-time; Arm B every epoch)
  s_sae    one BatchTopK SAE step seconds/token (identical for both arms)
  t_disk   disk-read seconds/token             (Arm A train phase only)

For a grid of G SAEs over ``tokens`` x ``epochs`` (shared activations: one
forward / one disk-read feeds the whole grid each epoch):
  Arm A (pre-harvest): tokens*f_fwd  +  epochs*tokens*(t_disk + G*s_sae)
  Arm B (on-the-fly):                   epochs*tokens*(f_fwd  + G*s_sae)
Peak activation disk (Arm A): tokens * d_model * bytes_per_elt  (Arm B ~ 0).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BYTES_PER_ELT = 2  # bf16

VE_KEY = "losses/global_variance_explained"
DEAD_KEY = "features/percent_dead_features"


# --- throughput extrapolation ------------------------------------------------


def _per_token_costs(bench: dict, d_model: int = 4096) -> dict:
    arm_a = bench["arm_a"]
    arm_b = bench["arm_b"]
    scaling = bench.get("scaling", {}).get("runs", [])

    f_fwd = 1.0 / arm_a["harvest"]["forward_tokens_per_sec"]
    t_disk_plus_sae = 1.0 / arm_a["tokens_per_sec"]
    otf_fwd_plus_sae = 1.0 / arm_b["tokens_per_sec"]

    # s_sae from the scaling slope: per_batch_sae_seconds grows ~linearly in N.
    s_sae = None
    if len(scaling) >= 2:
        batch_tokens = scaling[0]["tokens"] / max(scaling[0]["timed_steps"], 1)
        xs = [r["n_saes"] for r in scaling]
        ys = [r["per_batch_sae_seconds"] for r in scaling]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        slope = (
            sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
            if denom
            else ys[0] / xs[0]
        )
        s_sae = slope / batch_tokens  # seconds/token for one SAE
    if s_sae is None:
        # fall back: derive from arm_b (1/T_otf = f_fwd + s_sae)
        s_sae = max(otf_fwd_plus_sae - f_fwd, 0.0)

    t_disk = max(t_disk_plus_sae - s_sae, 0.0)
    return {
        "f_fwd": f_fwd,
        "s_sae": s_sae,
        "t_disk": t_disk,
        "otf_fwd_plus_sae": otf_fwd_plus_sae,
        "disk_plus_sae": t_disk_plus_sae,
        "fwd_tokens_per_sec": 1.0 / f_fwd,
        "sae_tokens_per_sec": 1.0 / s_sae if s_sae > 0 else float("inf"),
    }


def extrapolate(
    bench: dict,
    tokens: float = 2e9,
    epochs: int = 3,
    grid_size: int = 6,
    d_model: int = 4096,
    disk_headroom_tb: float = 18.0,
) -> dict:
    """Project sweep wall-clock + disk cost for both backends and pick a winner."""
    c = _per_token_costs(bench, d_model)
    f, s, td = c["f_fwd"], c["s_sae"], c["t_disk"]

    arm_a_harvest = tokens * f
    arm_a_train = epochs * tokens * (td + grid_size * s)
    arm_a_total = arm_a_harvest + arm_a_train
    arm_b_total = epochs * tokens * (f + grid_size * s)

    peak_disk_tb = tokens * d_model * BYTES_PER_ELT / 1e12

    arm_a_feasible = peak_disk_tb <= disk_headroom_tb
    if not arm_a_feasible:
        winner = "on_the_fly"
        reason = (
            f"pre-harvest infeasible: {peak_disk_tb:.1f} TB > "
            f"{disk_headroom_tb} TB headroom"
        )
    elif arm_a_total < arm_b_total:
        winner = "pre_harvest"
        reason = (
            f"pre-harvest lower wall-clock ({arm_a_total/3600:.2f} h vs "
            f"{arm_b_total/3600:.2f} h) and fits disk ({peak_disk_tb:.1f} TB)"
        )
    else:
        winner = "on_the_fly"
        reason = (
            f"on-the-fly lower wall-clock ({arm_b_total/3600:.2f} h vs "
            f"{arm_a_total/3600:.2f} h)"
        )

    return {
        "tokens": tokens,
        "epochs": epochs,
        "grid_size": grid_size,
        "costs": c,
        "arm_a_pre_harvest": {
            "harvest_hours": arm_a_harvest / 3600,
            "train_hours": arm_a_train / 3600,
            "total_hours": arm_a_total / 3600,
            "peak_disk_tb": peak_disk_tb,
            "feasible": arm_a_feasible,
        },
        "arm_b_on_the_fly": {
            "total_hours": arm_b_total / 3600,
            "peak_disk_tb": 0.0,
        },
        "decision": {"winner": winner, "reason": reason},
    }


def summarize(bench: dict, **kw: Any) -> dict:
    """Convenience: extrapolate at 2B and 5B and return both + the decision."""
    out = {
        "at_2B_3ep": extrapolate(bench, tokens=2e9, **kw),
        "at_5B_3ep": extrapolate(bench, tokens=5e9, **kw),
    }
    return out


# --- convergence check (W&B) -------------------------------------------------


def _tail_slope(
    steps, vals, tail_frac: float
) -> tuple[float | None, float | None]:
    """Linear slope (per step) + total change over the last tail_frac of steps."""
    steps = np.asarray(steps, dtype=float)
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(steps) & np.isfinite(vals)
    steps, vals = steps[ok], vals[ok]
    if len(steps) < 4:
        return None, None
    order = np.argsort(steps)
    steps, vals = steps[order], vals[order]
    cut = steps.max() - tail_frac * (steps.max() - steps.min())
    sel = steps >= cut
    if sel.sum() < 3:
        sel = np.ones_like(steps, dtype=bool)
    s, v = steps[sel], vals[sel]
    slope = float(np.polyfit(s, v, 1)[0])
    total_change = float(v[-1] - v[0])
    return slope, total_change


def check_model(
    api: Any,
    entity: str,
    project: str,
    model: str,
    layer: int,
    tail_frac: float,
    ve_tol: float,
) -> dict:
    """Per-run tail-slope / plateau flags for one model's finished sweep runs."""
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": f"^{model}-l{layer}-ef"}},
    )
    out = {}
    for r in runs:
        if r.state == "running":
            continue
        hist = r.history(
            keys=[VE_KEY, DEAD_KEY, "current_step"], samples=10000, pandas=True
        )
        if hist is None or len(hist) == 0:
            continue
        step_col = "current_step" if "current_step" in hist else "_step"
        ve_slope, ve_change = (None, None)
        dead_slope, dead_change = (None, None)
        if VE_KEY in hist:
            ve_slope, ve_change = _tail_slope(hist[step_col], hist[VE_KEY], tail_frac)
        if DEAD_KEY in hist:
            dead_slope, dead_change = _tail_slope(
                hist[step_col], hist[DEAD_KEY], tail_frac
            )
        plateaued = ve_change is not None and abs(ve_change) < ve_tol
        out[r.name] = {
            "state": r.state,
            "ve_tail_slope": ve_slope,
            "ve_tail_change": ve_change,
            "dead_tail_slope": dead_slope,
            "dead_tail_change": dead_change,
            "plateaued": plateaued,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the W&B convergence / plateau check."""
    ap = argparse.ArgumentParser(description="Check SAE sweep runs plateaued in W&B.")
    ap.add_argument("--project", default="basecamp-eden-saes")
    ap.add_argument("--models", nargs="+", default=["og2", "bcr"])
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument(
        "--tail-frac",
        type=float,
        default=0.10,
        help="fraction of steps at the end to fit",
    )
    ap.add_argument(
        "--ve-tol",
        type=float,
        default=5e-3,
        help="max |VE change| over the tail to count as flat",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import wandb

    api = wandb.Api()
    entity = api.default_entity
    per_run: dict = {}
    for m in args.models:
        per_run.update(
            check_model(
                api, entity, args.project, m, args.layer, args.tail_frac, args.ve_tol
            )
        )

    ve_slopes = [
        v["ve_tail_slope"] for v in per_run.values() if v["ve_tail_slope"] is not None
    ]
    result = {
        "project": args.project,
        "models": args.models,
        "layer": args.layer,
        "tail_frac": args.tail_frac,
        "ve_tol": args.ve_tol,
        "n_runs": len(per_run),
        "all_plateaued": bool(per_run)
        and all(v["plateaued"] for v in per_run.values()),
        "ve_tail_slope_mean": float(np.mean(ve_slopes)) if ve_slopes else None,
        "runs": per_run,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    logger.info(
        "all_plateaued=%s mean_ve_tail_slope=%s (%d runs) -> %s",
        result["all_plateaued"],
        result["ve_tail_slope_mean"],
        len(per_run),
        args.out,
    )
    for name, v in sorted(per_run.items()):
        logger.info(
            "  %16s  VE_tail=%s  dead_tail=%s  plateaued=%s",
            name,
            v["ve_tail_change"],
            v["dead_tail_change"],
            v["plateaued"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
