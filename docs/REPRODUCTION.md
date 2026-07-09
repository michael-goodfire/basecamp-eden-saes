# Reproduction check

`bes-reproduce` re-derives the pinned feature-atlas metrics **directly from the
annotation bundle span layers and the full-sparse code store** — no re-inference,
no re-training — and compares them to the values recorded in the deployed feature
atlas (source experiments #53 / #50 / #40).

## How to run

```bash
# one partial per genome (SLURM array over the 152-genome panel), then reduce
for acc in $(cat "$BUNDLE/panel/accessions.txt"); do
  bes-reproduce partial --acc "$acc" --out /path/to/partials
done
bes-reproduce reduce --partials '/path/to/partials/*.npz' --out reproduction.json
```

`partial` reads, for the two target features, the annotation's spans from the
bundle (`layers/{cath,ted}/spans/<acc>.tsv`), the per-position codes from the code
store (`code_store/bcr/bcr_ef8_k64/<acc>/<contig>.{plus,minus}.npz`), the exact
per-span GC from the panel FASTA, and — for the lift feature — the feature's
panel-wide firing total (a full per-genome scan). `reduce` sums the partials,
applies the matched-negative background (`bg.npz`), and compares.

## Metrics and the pinned bar

| metric | feature / annotation | recorded | reproduced | source |
| --- | --- | --- | --- | --- |
| recall (`covered/n_spans`) | 2044 / `cath\|3.40.1090.10` | 0.2434 | **0.2434** | #40/#53 |
| span_fold (`covered/E_bg`) | 2044 / `cath\|3.40.1090.10` | 702.956 | **702.9557** | #40/#53 |
| recall (`covered/n_spans`) | 27629 / `ted\|1.10.760.10` | 0.752 | **0.7518** | #50/#53 |
| lift (`PPV/prior`) | 2044 / `cath\|3.40.1090.10` | 244.3 | 377.83 (see below) | #53 |

The **pass/fail bar** is the three span-level metrics (recall + matched-negative
span_fold), recomputed end-to-end from the named inputs and matching to ±2%
(`recall` and `span_fold` for 2044 match to 4–5 significant figures; `recall` for
27629 matches). `bes-reproduce reduce` exits 0 when these gated metrics pass.

`span_fold` is pinned only for feature 2044: its CATH layer shares the #38 Pfam
matched-negative background, which this repo takes as input. The TED layer used a
separate covered-universe background (`structure/background.py`), so only `recall`
(which is background-independent) is gated for the TED-layer target 27629.

## Why `lift` is reported but not gated

The repository carries `lift` in **two places, on two different activation
bases**, and neither is part of the pass/fail bar:

- **`bes-reproduce`** recomputes `lift` from the canonical #38 code store —
  **377.8** for feature 2044 — reported for transparency alongside the gated span
  metrics.
- **The viewer** (`atlas/enrichment.py` → the served feature atlas, rendered by
  `viewer/js/main.js`) embeds the *historical* `lift` = **244.3**, which rests on
  the earlier/sparser #42 atlas position-rate basis (see below). This value is
  preserved as-is so the deployed atlas matches the source thread; it is **not**
  recomputed from the canonical #38 inputs and is flagged as the superseded #42
  basis in `viewer/README.md`.

The two numbers are consistent once you know they use different harvests: the
position-level firing rate differs between the #42 (sparse) and #38 (dense)
activation bases, so `PPV/prior` differs. The span-level metrics (recall,
span_fold), which #40 computed from the #38 store, reproduce exactly.

`lift = PPV / prior` is a **position-level** effect size:
`PPV = obs / fire_count` (fraction of a feature's firings that land inside the
annotation) and `prior = A / total` (fraction of all positions inside the
annotation). The annotation position count `A` (39,522) reproduces exactly from
the bundle spans, and `total` reproduces to 0.06%.

The recorded viewer value 244.3, however, rests on the **#42 atlas position-rate
basis** (`pos_rate` ≈ 6.9%, `fire_count` = 362,085) — an earlier, *sparser*
activation harvest than the #38 full-sparse code store this repository takes as its
`code_store` input. Recomputing the position basis directly from the #38 code store
gives an in-annotation firing rate of ≈46% (`obs` = 18,183, `fire_count` ≈ 1.56M),
hence `lift` ≈ 377.8. The two harvests use different activation thresholds, so the
position-level rate — and therefore `lift` — differs, while the span-level metrics
(which #40 computed from the #38 code store) reproduce exactly.

`lift` is therefore reported for transparency but excluded from the pass/fail bar:
it is not derivable from the named inputs (bundle + #38 code store). The #42 atlas
that produced the recorded position-rate table is listed in the packaging
reconciliation as superseded (see `docs/PROVENANCE.md`) and is not a documented
data input. To reproduce the historical 244.3 exactly, run the packaged
`atlas/enrichment.py` (`embed_enrichment`) on the #42 position-rate table
(`rates.json`) + `A` + `fire_count` + `total`; that validates the packaged lift
*formula*, but consumes the superseded position-rate artifact rather than the #38
code store.
