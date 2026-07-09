"""DDP entrypoint for the EDEN layer-L BatchTopK SAE ef/k sweep (one model).

Launched under torchrun with one process per H100. Each rank:
  1. loads the (frozen, bf16) EDEN partial forward to layer L,
  2. streams a *disjoint* shard of corpus windows (data parallel),
  3. fits normalizer stats once on rank 0 and broadcasts them,
  4. builds the 9-config (ef x k) BatchTopK SAE grid (identical weights across
     ranks via broadcast),
  5. trains all 9 SAEs off one shared forward per batch, all-reducing per-SAE
     grads across ranks (see basecamp_eden_saes.sae.trainer),
  6. evaluates each SAE on a held-out window split and writes per-config metrics
     + canonical SAE checkpoints (rank 0).

Run one model at a time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from basecamp_eden_saes import config
from basecamp_eden_saes.corpus.build import CorpusReader
from basecamp_eden_saes.harvest.eden import EdenPartialForward
from basecamp_eden_saes.harvest.streaming import (
    StreamingActivationDataset,
    build_sae,
    estimate_normalizer_stats,
)
from basecamp_eden_saes.sae.ddp_metrics import all_reduce_global_sae_metrics
from basecamp_eden_saes.sae.trainer import (
    SAEConfig,
    SweepTrainer,
    assert_shared_forward,
    broadcast_module,
    build_aux_fn,
    build_lr_fn,
    eval_sae_chunked,
    find_resume_step,
)

# Canonical goodfire-core SAE metric computer (the logging-free half that
# SAEWandBCallback itself composes). We log its standard-namespace dict
# (losses/global_variance_explained, features/percent_dead_features,
# features/l0_sparsity, features/histogram) to each config's own W&B run,
# because the SweepTrainer runs all 9 SAEs concurrently in one process and the
# global-`wandb.run` SAEWandBCallback can target only one run at a time.
from goodfire_core.saes.callbacks import SAEMetricsCallback

logger = logging.getLogger(__name__)


def log0(rank: int, *a) -> None:
    """Log a message on rank 0 only."""
    if rank == 0:
        logger.info(" ".join(str(x) for x in a))


def _smoke_metric_check(configs, dataset, world, device, rank) -> None:
    """Precondition: the v3 full-batch metric reduction works.

    Runs one shared-forward batch through the first SAE, then compares this
    rank's *local-shard* variance-explained (from the loss object) against the
    all-reduced *global* VE. The all-reduce is collective, so every rank calls
    it. Confirms (a) the global value is finite, (b) it is identical across ranks
    (the reduction actually ran), and (c) it differs from the single-rank value.
    """
    cfg = configs[0]
    it = dataset.training_iterator(device=device)
    it.set_max_steps(1)
    batch = next(iter(it.iter_epoch()))
    acts = batch.acts
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        res = cfg.sae(acts, sparsity_weight=cfg.aux_fn(0))
    local_ve = float(res.global_variance_explained)
    overrides = all_reduce_global_sae_metrics(cfg.sae, acts, res, world, device)
    global_ve = overrides["losses/global_variance_explained"]
    # cross-rank identity of the reduced global value
    t = torch.tensor([global_ve], dtype=torch.float64, device=device)
    if world > 1:
        gathered = [torch.zeros_like(t) for _ in range(world)]
        dist.all_gather(gathered, t)
        max_dev = max(float((gathered[0] - g).abs().max()) for g in gathered)
    else:
        max_dev = 0.0
    log0(
        rank,
        f"[metric-fix smoke] {cfg.label}: local_ve(rank0 shard)={local_ve:.5f} "
        f"global_ve={global_ve:.5f} (|d|={abs(global_ve - local_ve):.2e}); "
        f"cross-rank global max-dev={max_dev:.2e} (expect 0); "
        f"L0_global={overrides['features/l0_sparsity']:.2f}",
    )
    it.cleanup()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``bes-train-sae`` (DDP ef/k sweep for one model)."""
    paths = config.data_paths()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(paths.models))
    p.add_argument("--layer", type=int, default=28)
    p.add_argument(
        "--corpus", default=None, help="corpus dir (default: config.data_paths().corpus)"
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--tokens", type=float, default=2e9, help="global training token budget"
    )
    p.add_argument("--efs", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument(
        "--batch-tokens",
        type=int,
        default=8192,
        help="per-rank SAE batch tokens (v3: 8192 -> global 65536 on 8 GPUs)",
    )
    p.add_argument(
        "--fwd-windows", type=int, default=16, help="windows per EDEN forward"
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--aux-k", type=int, default=512)
    p.add_argument("--aux-loss-weight", type=float, default=1.0 / 32.0)
    p.add_argument("--lr-warmup-frac", type=float, default=0.05)
    p.add_argument(
        "--lr-cooldown-frac",
        type=float,
        default=0.1,
        help="only used when --lr-decay linear",
    )
    p.add_argument(
        "--lr-decay",
        choices=["cosine", "linear"],
        default="cosine",
        help="v3 default cosine: warmup then cosine anneal to ~0",
    )
    p.add_argument("--aux-warmup-frac", type=float, default=0.1)
    p.add_argument(
        "--save-every",
        type=int,
        default=5000,
        help="mid-run resumable checkpoint cadence (rank 0); 0 disables",
    )
    p.add_argument(
        "--keep-last",
        type=int,
        default=2,
        help="resumable checkpoints kept per config (plus the final dictionary)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest common mid-run checkpoint under <out>/resume",
    )
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument(
        "--held-out-windows", type=int, default=4096, help="windows reserved for eval"
    )
    p.add_argument("--eval-tokens", type=int, default=524_288)
    p.add_argument("--norm-stats-tokens", type=int, default=2_000_000)
    p.add_argument("--max-steps", type=int, default=0, help="override step count (smoke)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb-project", default="basecamp-eden-saes")
    p.add_argument(
        "--metrics-log-every",
        type=int,
        default=50,
        help="canonical goodfire-core SAE metrics logged every N steps",
    )
    p.add_argument(
        "--metrics-eval-every",
        type=int,
        default=500,
        help="feature-density histogram logged every N steps",
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    corpus = str(Path(args.corpus)) if args.corpus else str(paths.corpus)

    # --- distributed setup -------------------------------------------------
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_dist = world > 1
    if is_dist:
        # Generous timeout: rank 0 runs the held-out eval of all 9 SAEs solo
        # while ranks 1-7 wait at the final barrier; the default 10-min NCCL
        # collective timeout could trip on the full run's larger eval.
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed)

    model_path = paths.models[args.model]
    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    log0(rank, f"[{args.model}] world={world} layer={args.layer} model={model_path}")
    log0(
        rank,
        f"[{args.model}] efs={args.efs} ks={args.ks} "
        f"batch_tokens(per-rank)={args.batch_tokens}",
    )

    # --- EDEN forward + corpus --------------------------------------------
    eden = EdenPartialForward(model_path, layer=args.layer, device=device, max_seq=4096)
    reader = CorpusReader(corpus)
    n_windows = len(reader)

    # deterministic train / held-out window split (held-out = last H windows)
    H = min(args.held_out_windows, n_windows // 10)
    all_idx = np.arange(n_windows)
    held_windows = all_idx[n_windows - H :]
    train_windows = all_idx[: n_windows - H]
    log0(
        rank,
        f"[{args.model}] windows: train={len(train_windows)} held_out={len(held_windows)}",
    )

    # --- step budget -------------------------------------------------------
    global_batch = args.batch_tokens * world
    projected_full_steps = int(args.tokens // global_batch)
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = projected_full_steps
    log0(
        rank,
        f"[{args.model}] global_batch={global_batch} total_steps={total_steps} "
        f"(full {args.tokens:.0f}-token budget -> {projected_full_steps} steps)",
    )

    # --- normalizer stats (rank 0 computes, broadcast to all) -------------
    t = time.time()
    if rank == 0:
        stats = estimate_normalizer_stats(
            eden,
            reader,
            n_tokens=args.norm_stats_tokens,
            fwd_windows=args.fwd_windows,
            seed=args.seed,
            window_subset=train_windows,  # train-only: no held-out leak
        )
        mean = stats["per_coord_mean"].to(device)
        meta = torch.tensor(
            [stats["mean_norm"], stats["std_norm"]], dtype=torch.float64, device=device
        )
    else:
        mean = torch.zeros(eden.d_model, device=device)
        meta = torch.zeros(2, dtype=torch.float64, device=device)
    if is_dist:
        dist.broadcast(mean, src=0)
        dist.broadcast(meta, src=0)
    stats = {
        "per_coord_mean": mean.float().cpu(),
        "mean_norm": float(meta[0].item()),
        "std_norm": float(meta[1].item()),
    }
    norm_threshold = stats["mean_norm"] + 4.0 * stats["std_norm"]
    log0(
        rank,
        f"[{args.model}] normalizer: mean_norm={stats['mean_norm']:.2f} "
        f"std_norm={stats['std_norm']:.2f} outlier_thr={norm_threshold:.2f} "
        f"({time.time()-t:.1f}s)",
    )
    if not np.isfinite(stats["mean_norm"]) or not np.isfinite(stats["std_norm"]):
        raise RuntimeError("non-finite normalizer stats")

    # --- build SAE grid (identical across ranks) --------------------------
    dataset = StreamingActivationDataset(
        reader,
        eden,
        sae_batch_tokens=args.batch_tokens,
        fwd_windows=args.fwd_windows,
        seed=args.seed,
        window_subset=train_windows,
        rank=rank,
        world_size=world,
    )

    # shared-forward precondition (smoke): one EDEN forward per produced batch
    if args.smoke and rank == 0:
        nfwd = assert_shared_forward(dataset, n_batches=3)
        log0(
            rank,
            f"[{args.model}] assert_shared_forward: {nfwd} EDEN forwards for 3 batches "
            f"(>=1 forward/batch from producer only)",
        )

    # mid-run resumable checkpoints live under <out>/resume/<label>/step_<n>.pt
    ckpt_dir = out_dir / "resume"
    labels = [f"ef{ef}_k{k}" for ef in args.efs for k in args.ks]
    resume_step = find_resume_step(ckpt_dir, labels) if args.resume else 0
    if resume_step > 0:
        log0(
            rank,
            f"[{args.model}] RESUME from step {resume_step}/{total_steps} "
            f"(ckpt_dir={ckpt_dir})",
        )
    elif args.resume:
        log0(
            rank,
            f"[{args.model}] --resume set but no common checkpoint found; "
            f"training from scratch",
        )

    configs: list[SAEConfig] = []
    wandb_runs = {}
    use_wandb = (not args.no_wandb) and rank == 0
    if use_wandb:
        import wandb
    for ef in args.efs:
        for k in args.ks:
            label = f"ef{ef}_k{k}"
            sae = build_sae(eden.d_model, ef, k, stats, device=device, aux_k=args.aux_k)
            broadcast_module(sae, src=0)  # identical init across ranks
            run = None
            if use_wandb:
                run = wandb.init(
                    project=args.wandb_project,
                    name=f"{args.model}-l{args.layer}-{label}",
                    group=f"{args.model}-l{args.layer}",
                    job_type="sae-sweep",
                    reinit="create_new",  # 9 concurrent run handles in one process
                    config={
                        "model": args.model,
                        "layer": args.layer,
                        "ef": ef,
                        "k": k,
                        "d_model": eden.d_model,
                        "d_sae": ef * eden.d_model,
                        "aux_k": args.aux_k,
                        "lr": args.lr,
                        "total_steps": total_steps,
                        "global_batch_tokens": global_batch,
                        "tokens": args.tokens,
                        "aux_loss_weight": args.aux_loss_weight,
                        "corpus": corpus,
                    },
                )
                wandb_runs[label] = run
            opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
            # Resume: every rank loads the same rank-0-written checkpoint file
            # (shared FS), so weights + optimizer state stay identical across
            # ranks -- equivalent to broadcasting fresh init, but from step N.
            if resume_step > 0:
                rpath = ckpt_dir / label / f"step_{resume_step}.pt"
                rck = torch.load(str(rpath), map_location=device, weights_only=False)
                sae.load_state_dict(rck["state_dict"])
                opt.load_state_dict(rck["optimizer_state"])
                log0(
                    rank,
                    f"[{args.model}] {label}: resumed weights+opt from {rpath.name}",
                )
            # One canonical metric computer per SAE (rank-0 logging only). Its
            # FeatureStatisticsTracker + the SAE's last_activated_at buffer drive
            # the standard features/* metrics; on_train_begin builds the tracker.
            metrics_cb = None
            if rank == 0:
                metrics_cb = SAEMetricsCallback(
                    log_every_steps=args.metrics_log_every,
                    eval_every_steps=args.metrics_eval_every,
                    dead_feature_threshold=int(1e6),
                    model=sae,
                )
                metrics_cb.on_train_begin(sae=sae)
            configs.append(
                SAEConfig(
                    label=label,
                    ef=ef,
                    k=k,
                    sae=sae,
                    opt=opt,
                    lr_fn=build_lr_fn(
                        args.lr,
                        total_steps,
                        args.lr_warmup_frac,
                        args.lr_cooldown_frac,
                        decay=args.lr_decay,
                    ),
                    aux_fn=build_aux_fn(
                        args.aux_loss_weight, total_steps, args.aux_warmup_frac
                    ),
                    wandb_run=run,
                    metrics_cb=metrics_cb,
                )
            )
    log0(
        rank,
        f"[{args.model}] built {len(configs)} SAEs: "
        + ", ".join(f"{c.label}(d_sae={c.ef*eden.d_model})" for c in configs),
    )

    # --- metric-fix precondition (smoke): the all-reduced global VE is computed
    # across ranks (identical on every rank) and differs from a single rank's
    # local-shard VE. Confirms the v3 full-batch metric reduction before the run.
    if args.smoke:
        _smoke_metric_check(configs, dataset, world, device, rank)

    # --- train -------------------------------------------------------------
    trainer = SweepTrainer(
        dataset,
        configs,
        total_steps=total_steps,
        device=device,
        rank=rank,
        world_size=world,
        clip_grad_max_norm=args.clip,
        norm_threshold=norm_threshold,
        log_every_steps=args.metrics_log_every,
        hist_every_steps=args.metrics_eval_every,
        save_every_steps=args.save_every,
        ckpt_dir=ckpt_dir,
        keep_last=args.keep_last,
        start_step=resume_step,
    )
    log0(
        rank,
        f"[{args.model}] training {total_steps} steps "
        f"(start_step={resume_step}, save_every={args.save_every} -> {ckpt_dir})...",
    )
    hist = trainer.fit()
    log0(rank, f"[{args.model}] train done: {hist}")

    # --- grad-sync precondition (smoke): a SAE's grads identical across ranks
    if args.smoke and is_dist:
        cfg = configs[0]
        g = next(pp.grad for pp in cfg.sae.parameters() if pp.grad is not None).clone()
        gathered = [torch.zeros_like(g) for _ in range(world)]
        dist.all_gather(gathered, g)
        max_dev = max(
            float((gathered[0] - gathered[i]).abs().max()) for i in range(world)
        )
        log0(
            rank,
            f"[{args.model}] grad-sync check: max cross-rank grad dev = {max_dev:.3e} "
            f"(expect 0)",
        )

    # --- held-out eval + checkpoints (rank 0) -----------------------------
    # Free optimizer state + grads first: not needed at eval, and the 9
    # dictionaries' Adam states (~22 GB) otherwise crowd out the eval codes.
    for cfg in configs:
        cfg.opt = None
        for p_ in cfg.sae.parameters():
            p_.grad = None
    torch.cuda.empty_cache()

    if rank == 0:
        results = {}
        for cfg in configs:
            # Save the dictionary FIRST -- it is the deliverable, and saving
            # before eval means an eval failure never loses a trained SAE
            # (metrics can be recomputed offline from the checkpoint).
            ck_path = out_dir / f"{cfg.label}.pt"
            cfg.sae.save_checkpoint(str(ck_path))
            m = eval_sae_chunked(
                cfg.sae,
                eden,
                reader,
                held_windows,
                n_tokens=args.eval_tokens,
                fwd_windows=args.fwd_windows,
                device=device,
            )
            dens = m.pop("density")
            # density histogram (log-spaced firing-fraction bins) for the report
            nz = dens[dens > 0]
            hist_edges = np.logspace(-7, 0, 36)
            hist_counts = np.histogram(np.clip(nz, 1e-7, 1.0), bins=hist_edges)[
                0
            ].tolist()
            m.update(
                {
                    "ef": cfg.ef,
                    "k": cfg.k,
                    "d_sae": cfg.ef * eden.d_model,
                    "checkpoint": str(ck_path),
                    "density_hist_edges": hist_edges.tolist(),
                    "density_hist_counts": hist_counts,
                    "n_dead": int((dens == 0).sum()),
                }
            )
            results[cfg.label] = m
            np.save(out_dir / f"{cfg.label}_density.npy", dens)
            # write results incrementally so partial metrics survive a crash
            (out_dir / "sweep_results.json").write_text(
                json.dumps(
                    {
                        "model": args.model,
                        "model_path": model_path,
                        "layer": args.layer,
                        "corpus": corpus,
                        "tokens": args.tokens,
                        "total_steps": total_steps,
                        "global_batch_tokens": global_batch,
                        "world_size": world,
                        "train_history": hist,
                        "normalizer": {
                            k: v for k, v in stats.items() if k != "per_coord_mean"
                        },
                        "configs": results,
                    },
                    indent=2,
                )
            )
            if cfg.wandb_run is not None:
                cfg.wandb_run.log(
                    {
                        f"heldout/{kk}": vv
                        for kk, vv in m.items()
                        if isinstance(vv, (int, float))
                    }
                )
                cfg.wandb_run.finish()
            log0(
                rank,
                f"[{args.model}] {cfg.label}: VE={m['variance_explained']:.4f} "
                f"L0={m['l0']:.1f} dead={m['dead_fraction']*100:.2f}% "
                f"coh={m['coherence_mean']:.4f}",
            )
        summary = {
            "model": args.model,
            "model_path": model_path,
            "layer": args.layer,
            "corpus": corpus,
            "tokens": args.tokens,
            "total_steps": total_steps,
            "global_batch_tokens": global_batch,
            "world_size": world,
            "train_history": hist,
            "normalizer": {k: v for k, v in stats.items() if k != "per_coord_mean"},
            "configs": results,
        }
        (out_dir / "sweep_results.json").write_text(json.dumps(summary, indent=2))
        log0(rank, f"[{args.model}] wrote {out_dir/'sweep_results.json'}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
