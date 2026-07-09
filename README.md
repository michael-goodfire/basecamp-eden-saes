# basecamp-eden-saes

Sparse-autoencoder (SAE) engine and feature viewer for the **EDEN-7B** DNA
foundation model. This repository takes an EDEN checkpoint end-to-end: build an
OpenGenome2 training corpus, harvest activations, train a grid of BatchTopK SAEs,
select the best one, ground its features in genome annotations (Pfam/GO, CATH/TED
folds, genomic regulatory layers), quantify how far a feature's detection
generalizes beyond sequence similarity, and browse the result in a standalone
feature-atlas viewer.

It is the packaged, documented form of a research thread; see
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the source experiments.

## Package layout

`src/basecamp_eden_saes/`

| module | what it does |
| --- | --- |
| `config.py` | resolves the external data inputs (models, corpus, bundle, code store, backgrounds, structure cache) from `EDEN_SAES_*` env vars with on-cluster defaults |
| `corpus/` | OpenGenome2 5B-token corpus build: GTDB isolate + metagenome sampling, byte-level DNA tokenization, low-complexity filtering |
| `harvest/` | activation harvesting — on-the-fly streaming, on-disk, and the full-sparse per-nucleotide code store |
| `sae/` | BatchTopK SAE training with a shared-forward expansion×k grid sweep, and best-SAE selection |
| `annotations/` | RefSeq annotation-panel build (per-nt tracks, Pfam/GO domains, v2 genomic layers) and panel loaders |
| `autointerp/` | annotation-grounded span metrics (recall, matched-negative precision fold, circular-shift null) over the code store, plus the reproduction check |
| `structure/` | CATH-S95 + TED fold layers and AlphaFold-DB structure coverage |
| `analysis/` | structure-beyond-sequence generalization (both-strand rejoin, home vs sequence-divergent recall, recall-vs-identity decay) |
| `atlas/` | canonical per-feature atlas builder (fold-granularity canonicalization, dual span/position enrichment) |

The standalone feature-viewer front-end (HTML/CSS/JS) lives under [`viewer/`](viewer/).

## Installation

The base install (corpus post-processing, autointerp span metrics, structure and
generalization analysis, and the **deterministic reproduction check**) needs only
open dependencies:

```bash
uv sync
```

The GPU pipeline (corpus tokenization, activation harvest, SAE training) needs the
`engine` extra, which depends on **goodfire-core**, a *private* Goodfire package:

```bash
uv sync --extra engine        # also: --extra annotations --extra structure --extra viz --extra all
```

`goodfire-core` is pinned (see `[tool.uv.sources]` in `pyproject.toml`) to the
revision the harvest / SAE-training experiments ran against. Resolving it needs
read access to `github.com/goodfire-ai/goodfire-core`:

- **On a Silico cluster:** nothing extra — the deploy-key git URL rewrite is
  pre-provisioned, so `uv sync --extra engine` just works.
- **Elsewhere:** you need your own GitHub access to that private repo (or a
  replicated deploy-key + SSH `insteadOf` rewrite). CI cannot resolve the
  dependency without one of these credentials.

## Data inputs (referenced, not committed)

Large artifacts are referenced by path, resolved in `config.py` and overridable
via environment variables. The on-cluster defaults are the paths the source thread
produced them at.

| input | default location | override |
| --- | --- | --- |
| EDEN models (OG2 / BCR) | `/mnt/data/shared/models/{Eden-7B-OG2-286B,EDEN-7B-BCR}` | `EDEN_SAES_MODEL_OG2` / `EDEN_SAES_MODEL_BCR` |
| OpenGenome2 corpus (5B tok) | `/mnt/data/artifacts/silico/basecamp-eden-saes/datasets/og2-gtdb-metag-bcr4096-5b` | `EDEN_SAES_CORPUS` |
| trained SAE checkpoints | `/mnt/data/artifacts/silico/basecamp-eden-saes/saes/{bcr-l28-v3,og2-l28}` | `EDEN_SAES_SAE_CHECKPOINTS` |
| annotation bundle (`eden_annotation_panel_v1`) | `/mnt/data/artifacts/silico/eden_annotation_panel_v1` | `EDEN_SAES_BUNDLE` |
| full-sparse code store (740 GB) | `/mnt/data/artifacts/silico/basecamp-eden-saes/code_store/{bcr,og2}` | `EDEN_SAES_CODE_STORE` |
| matched-negative backgrounds | `.../exp_01kwxjy7zmf30b1tb9k58pspfp/metrics` (per-dict `bg.npz`) | `EDEN_SAES_BG_ROOT` |
| panel genome FASTA (exact GC) | `.../exp_01kwd6629qfvsvdjda4kfyft2z/panel/genomes` | `EDEN_SAES_PANEL_GENOMES` |
| AlphaFold structure cache (~50 GB) | `.../exp_01kx09bzv8fjqvh3k37t6c9x47/af_cache` | `EDEN_SAES_AF_CACHE` |

The **annotation bundle** is the documented single annotation input: a 152-genome
RefSeq panel with per-nt tracks, Pfam/GO domains, CATH-S95 + TED fold span layers,
and their matched-negative backgrounds. Its `MANIFEST.json` documents every layer.
Panel genome FASTA is reconstructable from `panel/accessions.txt` via the NCBI
Datasets CLI if the default path is unavailable.

## Pipeline order

1. **Corpus** — `bes-build-corpus` builds the 5B-token OpenGenome2 corpus.
2. **Harvest** — `bes-harvest` (on-disk) or the on-the-fly path feeds SAE training;
   `bes-harvest-codes` writes the full-sparse per-nucleotide code store.
3. **SAE** — `bes-train-sae` runs the expansion×k grid; `bes-select-sae` picks the
   best SAE by held-out variance-explained under dead-feature / L0-band gates.
4. **Annotations** — build the RefSeq panel layers (see `annotations/`).
5. **Autointerp** — `bes-span-cover` + `bes-span-reduce` compute per-annotation
   recall / precision-fold / null over the code store.
6. **Structure** — build CATH/TED fold layers and AlphaFold coverage (`structure/`).
7. **Analysis** — `bes-rejoin` + `bes-generalization` measure
   structure-beyond-sequence generalization.
8. **Atlas** — `bes-build-atlas` canonicalizes fold granularity and assembles the
   per-feature atlas the viewer serves.

## Reproduction check

`bes-reproduce` re-derives the pinned feature-atlas metrics directly from the
annotation bundle span layers and the full-sparse code store, and compares them to
the recorded values (±2%). See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

```bash
# one partial per genome (SLURM array), then reduce + compare
bes-reproduce partial --acc GCF_000005845.2 --out /tmp/repro
bes-reproduce reduce --partials '/tmp/repro/*.npz' --out reproduction.json
```

## Tests

```bash
uv run pytest        # unit tests for the metric primitives and config
```

## License

Proprietary — Goodfire. This repository depends on the private `goodfire-core`
package and is not intended for external distribution.
