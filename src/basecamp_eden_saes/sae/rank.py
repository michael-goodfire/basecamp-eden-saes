"""Activation health + effective-rank diagnostic for EDEN-7B residuals.

Streams a sample of corpus windows through EDEN's partial forward, capturing the
residual stream at several layers in a single pass, and answers two questions
that gate an SAE sweep:

1. **Health** -- are the activations degenerate? Per-token L2-norm distribution,
   fraction non-finite (NaN/Inf) / zero-norm, mean pairwise cosine between random
   token pairs (collapse check), and the per-coordinate variance distribution
   (dead-dimension check).

2. **Effective rank** -- how many real directions do the activations span? A
   streaming centered covariance ``C = E[(x-mu)(x-mu)^T]`` (d x d) is accumulated
   over all sampled tokens, then eigendecomposed. From the eigenspectrum we report
   the participation ratio ``PR = (sum lambda)^2 / sum lambda^2``, the number of
   components for 90/95/99% of variance, and the stable rank ``sum lambda / max lambda``.

A same-trace isotropic Gaussian baseline is accumulated through the identical
covariance path as a full-rank (~d_model) reference, so the real spectrum is read
against both d_model and an empirical full-rank control.

The deliverable is a decision: a health go/no-go and, if healthy, a recommended
SAE expansion factor ``ef`` (``d_sae = ef * d_model``) sized from the measured
effective rank.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from basecamp_eden_saes.corpus.build import CorpusReader
from basecamp_eden_saes.harvest.eden import EdenPartialForward

logger = logging.getLogger(__name__)


class LayerStats:
    """Streaming accumulator for one layer's health + covariance statistics."""

    def __init__(
        self,
        d_model: int,
        device: str,
        reservoir: int = 8192,
        norm_sample_cap: int = 500_000,
        seed: int = 42,
    ):
        self.d = d_model
        self.device = device
        self.dtype = torch.float64
        # centered covariance via sum and sum-of-outer-products
        self.sum_x = torch.zeros(d_model, dtype=self.dtype, device=device)
        self.sum_xxT = torch.zeros((d_model, d_model), dtype=self.dtype, device=device)
        self.count = 0
        # health counters
        self.n_nan = 0
        self.n_inf = 0
        self.n_zero_norm = 0
        # per-token norm moments + min/max + a capped sample for the histogram
        self.norm_sum = 0.0
        self.norm_sq_sum = 0.0
        self.norm_min = float("inf")
        self.norm_max = 0.0
        self.norm_sample: list[np.ndarray] = []
        self.norm_sample_n = 0
        self.norm_sample_cap = norm_sample_cap
        # reservoir of raw vectors for the pairwise-cosine collapse check
        self.reservoir = reservoir
        self.vec_sample: torch.Tensor | None = None
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def update(self, acts: torch.Tensor) -> None:
        """Fold a [N, d] batch of token activations (any dtype) into the stats."""
        x = acts.to(torch.float32)
        n = x.shape[0]
        if n == 0:
            return
        finite = torch.isfinite(x)
        self.n_nan += int(torch.isnan(x).any(dim=1).sum().item())
        self.n_inf += int(torch.isinf(x).any(dim=1).sum().item())
        # keep only fully-finite rows for the numeric accumulators
        row_finite = finite.all(dim=1)
        xf = x[row_finite]
        if xf.shape[0] == 0:
            self.count += 0
            return
        norms = torch.linalg.vector_norm(xf, dim=1)
        self.n_zero_norm += int((norms == 0).sum().item())
        self.norm_sum += float(norms.sum().item())
        self.norm_sq_sum += float((norms * norms).sum().item())
        self.norm_min = min(self.norm_min, float(norms.min().item()))
        self.norm_max = max(self.norm_max, float(norms.max().item()))
        # capped norm sample for histogram
        if self.norm_sample_n < self.norm_sample_cap:
            self.norm_sample.append(norms.detach().cpu().numpy())
            self.norm_sample_n += norms.numel()
        # covariance accumulators (float64)
        xd = xf.to(self.dtype)
        self.sum_x += xd.sum(0)
        self.sum_xxT += xd.t() @ xd
        self.count += xf.shape[0]
        # reservoir of raw vectors (first `reservoir` tokens) for cosine check
        if self.vec_sample is None:
            self.vec_sample = xf[: self.reservoir].detach().clone()
        elif self.vec_sample.shape[0] < self.reservoir:
            need = self.reservoir - self.vec_sample.shape[0]
            self.vec_sample = torch.cat(
                [self.vec_sample, xf[:need].detach().clone()], 0
            )

    @torch.no_grad()
    def finalize(self, var_floor_frac: float = 1e-3) -> dict[str, Any]:
        """Eigendecompose the accumulated covariance and return the summary dict."""
        mu = self.sum_x / self.count
        cov = self.sum_xxT / self.count - torch.outer(mu, mu)
        # symmetrize for numerical safety, then eigendecompose
        cov = 0.5 * (cov + cov.t())
        evals = torch.linalg.eigvalsh(cov)  # ascending
        evals = torch.clamp(evals, min=0.0)
        evals_desc = torch.flip(evals, dims=[0])
        total = float(evals_desc.sum().item())
        lam = evals_desc / max(total, 1e-30)
        cumvar = torch.cumsum(lam, dim=0)

        def rank_at(frac: float) -> int:
            return int((cumvar < frac).sum().item()) + 1

        pr = float(
            (total**2) / float((evals_desc * evals_desc).sum().item() + 1e-30)
        )
        lam_max = float(evals_desc[0].item())
        stable_rank = float(total / (lam_max + 1e-30))

        per_coord_var = torch.diagonal(cov).clamp(min=0.0)
        med_var = float(per_coord_var.median().item())
        dead_frac = float(
            (per_coord_var < var_floor_frac * med_var).float().mean().item()
        )

        # pairwise cosine on the raw reservoir (collapse check) + centered version
        cos_raw, cos_centered = self._pairwise_cosine(mu)

        mean_norm = self.norm_sum / max(self.count, 1)
        var_norm = max(self.norm_sq_sum / max(self.count, 1) - mean_norm**2, 0.0)

        return {
            "num_tokens": int(self.count),
            "n_nan_tokens": int(self.n_nan),
            "n_inf_tokens": int(self.n_inf),
            "n_zero_norm_tokens": int(self.n_zero_norm),
            "nonfinite_frac": (self.n_nan + self.n_inf)
            / max(self.count + self.n_nan + self.n_inf, 1),
            "norm_mean": mean_norm,
            "norm_std": float(var_norm**0.5),
            "norm_min": self.norm_min,
            "norm_max": self.norm_max,
            "mean_pairwise_cosine_raw": cos_raw,
            "mean_pairwise_cosine_centered": cos_centered,
            "participation_ratio": pr,
            "stable_rank": stable_rank,
            "rank_90": rank_at(0.90),
            "rank_95": rank_at(0.95),
            "rank_99": rank_at(0.99),
            "lambda_max": lam_max,
            "total_variance": total,
            "per_coord_var_median": med_var,
            "dead_coord_frac": dead_frac,
            # arrays for figures
            "eigenvalues": evals_desc.detach().cpu().numpy(),
            "cumvar": cumvar.detach().cpu().numpy(),
            "per_coord_var": per_coord_var.detach().cpu().numpy(),
            "norm_sample": (
                np.concatenate(self.norm_sample) if self.norm_sample else np.array([])
            ),
            "mu": mu.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def _pairwise_cosine(self, mu: torch.Tensor) -> tuple[float, float]:
        if self.vec_sample is None or self.vec_sample.shape[0] < 2:
            return float("nan"), float("nan")
        v = self.vec_sample.to(torch.float32)

        def mean_offdiag_cos(m: torch.Tensor) -> float:
            nrm = torch.linalg.vector_norm(m, dim=1, keepdim=True)
            nrm = torch.clamp(nrm, min=1e-12)
            u = m / nrm
            g = u @ u.t()
            n = g.shape[0]
            off = (g.sum() - torch.diagonal(g).sum()) / (n * (n - 1))
            return float(off.item())

        cos_raw = mean_offdiag_cos(v)
        cos_centered = mean_offdiag_cos(v - mu.to(torch.float32))
        return cos_raw, cos_centered


@torch.no_grad()
def gaussian_baseline(
    d_model: int,
    total_variance: float,
    num_tokens: int,
    device: str,
    batch: int = 16384,
    seed: int = 0,
) -> dict[str, Any]:
    """Isotropic Gaussian with matched trace, run through the covariance path.

    per-coordinate variance = total_variance / d_model, so the Gaussian's trace
    equals the real data's. With num_tokens >> d_model its spectrum is the
    Marchenko-Pastur full-rank reference (PR ~ d_model).
    """
    std = (total_variance / d_model) ** 0.5
    gen = torch.Generator(device=device).manual_seed(seed)
    sum_xxT = torch.zeros((d_model, d_model), dtype=torch.float64, device=device)
    sum_x = torch.zeros(d_model, dtype=torch.float64, device=device)
    count = 0
    while count < num_tokens:
        n = min(batch, num_tokens - count)
        x = torch.randn(n, d_model, generator=gen, device=device) * std
        xd = x.to(torch.float64)
        sum_x += xd.sum(0)
        sum_xxT += xd.t() @ xd
        count += n
    mu = sum_x / count
    cov = sum_xxT / count - torch.outer(mu, mu)
    cov = 0.5 * (cov + cov.t())
    evals = torch.clamp(torch.linalg.eigvalsh(cov), min=0.0)
    evals_desc = torch.flip(evals, dims=[0])
    total = float(evals_desc.sum().item())
    lam = evals_desc / max(total, 1e-30)
    cumvar = torch.cumsum(lam, dim=0)
    pr = float((total**2) / float((evals_desc * evals_desc).sum().item() + 1e-30))

    def rank_at(frac: float) -> int:
        return int((cumvar < frac).sum().item()) + 1

    return {
        "participation_ratio": pr,
        "rank_90": rank_at(0.90),
        "rank_95": rank_at(0.95),
        "rank_99": rank_at(0.99),
        "stable_rank": float(total / (float(evals_desc[0].item()) + 1e-30)),
        "eigenvalues": evals_desc.detach().cpu().numpy(),
        "cumvar": cumvar.detach().cpu().numpy(),
        "num_tokens": int(count),
    }


def health_gate(
    s: dict[str, Any], collapse_thresh: float = 0.9, dead_thresh: float = 0.5
) -> dict[str, Any]:
    """Apply the health gate to one layer's finalized stats."""
    checks = {
        "finite": s["nonfinite_frac"] == 0.0,
        "nonzero_norm": s["n_zero_norm_tokens"] == 0,
        "not_collapsed": (s["mean_pairwise_cosine_centered"] < collapse_thresh),
        "variance_spread": s["dead_coord_frac"] < dead_thresh,
    }
    return {"checks": checks, "pass": all(checks.values())}


def recommend_ef(R: float, d_model: int) -> dict[str, Any]:
    """Map an effective rank R to an SAE expansion-factor recommendation.

    Two anchors are reported transparently:
      * 1-4x R sizing -> implied ef range ef in [R/d_model, 4R/d_model].
      * the ratio R/d_model -> whether the LM-default ef=16 is justified.
    The data spanning a near-full subspace (high ratio) justifies a large
    overcomplete dictionary (ef=16); a low-rank subspace anchors ef lower so the
    dictionary does not train mostly dead.
    """
    ratio = R / d_model
    ef_lo = max(1, int(np.ceil(1.0 * R / d_model)))
    ef_hi = max(ef_lo, int(np.ceil(4.0 * R / d_model)))
    if ratio >= 0.5:
        rec, reason = (
            "16",
            "effective rank spans >=50% of d_model; LM-default ef=16 stands",
        )
    elif ratio >= 0.25:
        rec, reason = "8", "effective rank is moderate; anchor below default"
    else:
        span = f"{ef_lo}" if ef_lo == ef_hi else f"{ef_lo}-{ef_hi}"
        rec, reason = span, "low effective rank; size the dictionary near the rank"
    return {
        "effective_rank": R,
        "ratio_to_d_model": ratio,
        "d_sae_1x_to_4x": [int(round(R)), int(round(4 * R))],
        "implied_ef_1x_to_4x": [ef_lo, ef_hi],
        "recommended_ef": rec,
        "reason": reason,
    }


def run(
    corpus: str,
    model_path: str,
    layers: list[int],
    n_tokens: int,
    fwd_windows: int,
    out_dir: str,
    hook_layer: int = 24,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, Any]:
    """Stream the corpus, accumulate per-layer stats, and write the rank report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reader = CorpusReader(corpus)
    eden = EdenPartialForward(model_path, layer=hook_layer, device=device)
    d_model = eden.d_model
    accs = {ell: LayerStats(d_model, device, seed=seed) for ell in layers}

    t0 = time.time()
    done = 0
    target_done = False
    for windows in reader.iter_batches(fwd_windows, shuffle=True, seed=seed):
        multi = eden.tokens_from_windows_multi(windows, layers)
        for ell in layers:
            accs[ell].update(multi[ell])
        done = accs[hook_layer].count
        if done % (fwd_windows * 50) < fwd_windows:
            rate = done / max(time.time() - t0, 1e-9)
            logger.info(
                "[capture] %.2fM / %.1fM tokens (%.1fk tok/s)",
                done / 1e6,
                n_tokens / 1e6,
                rate / 1e3,
            )
        if done >= n_tokens:
            target_done = True
            break
    if not target_done:
        logger.info("[capture] corpus exhausted at %.2fM tokens", done / 1e6)

    results: dict[str, Any] = {
        "model_path": model_path,
        "corpus": corpus,
        "d_model": d_model,
        "hook_layer": hook_layer,
        "layers": layers,
        "n_tokens_target": n_tokens,
        "seed": seed,
        "layer_stats": {},
        "gate": {},
        "baseline": {},
        "ef": {},
    }
    arrays: dict[str, Any] = {}
    for ell in layers:
        s = accs[ell].finalize()
        # split scalars from arrays
        for k in ("eigenvalues", "cumvar", "per_coord_var", "norm_sample", "mu"):
            arrays[f"layer{ell}_{k}"] = s.pop(k)
        results["layer_stats"][str(ell)] = s
        results["gate"][str(ell)] = health_gate(s)

    # Gaussian full-rank baseline matched to the hook layer's trace
    hook_s = results["layer_stats"][str(hook_layer)]
    base = gaussian_baseline(
        d_model,
        hook_s["total_variance"],
        min(hook_s["num_tokens"], 1_000_000),
        device,
        seed=0,
    )
    for k in ("eigenvalues", "cumvar"):
        arrays[f"baseline_{k}"] = base.pop(k)
    results["baseline"] = base

    # ef recommendation from the hook layer's effective rank (PR + 95% rank)
    R_pr = hook_s["participation_ratio"]
    R_95 = float(hook_s["rank_95"])
    results["ef"] = {
        "from_participation_ratio": recommend_ef(R_pr, d_model),
        "from_rank_95": recommend_ef(R_95, d_model),
    }

    with open(out / "rank_stats.json", "w") as fh:
        json.dump(results, fh, indent=2)
    np.savez(out / "rank_arrays.npz", **arrays)
    logger.info("rank report: %s", json.dumps(results, indent=2))
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the EDEN residual health + effective-rank check."""
    p = argparse.ArgumentParser(
        description="EDEN residual health + effective-rank check."
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", required=True, help="EDEN model path")
    p.add_argument("--layers", type=int, nargs="+", default=[16, 24, 28])
    p.add_argument("--hook-layer", type=int, default=24)
    p.add_argument("--n-tokens", type=float, default=5e6)
    p.add_argument("--fwd-windows", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    run(
        corpus=args.corpus,
        model_path=args.model,
        layers=list(args.layers),
        n_tokens=int(args.n_tokens),
        fwd_windows=args.fwd_windows,
        out_dir=args.out,
        hook_layer=args.hook_layer,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
