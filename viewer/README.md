# EDEN feature viewer (standalone front-end)

A static, dependency-free browser app for exploring the per-feature atlas of the
EDEN SAE dictionaries (`og2`, `bcr_k64`, `bcr_k16`). It shows, for each SAE
feature, its detected annotations (structural folds, protein families, genomic
regulatory layers), the dual enrichment metrics, per-exemplar genome tracks, and
an interactive 3D view of the AlphaFold structure painted by feature activation.

No build step: it is plain ES modules + one vendored 3D library, served as static
files.

## Layout

```
viewer/
  index.html            shell (loads js/main.js as a module)
  styles.css            styles
  dicts.json            dictionary catalog (id/label/model/headline) shown in the picker
  js/
    main.js             app entry: routing, index table, feature panel wiring
    data.js             fetch + cache of the per-dict JSON (index_rows/rates/features)
    tracks.js           genome / annotation track rendering
    logo.js             sequence-logo rendering
    af.js               AlphaFold 3D structure view (drives the vendored NGL lib)
    vendor/
      ngl.js            vendored NGL molecular-graphics library (~1.3 MB)
  README.md             this file
```

The JS is the **canonical front-end from #53** (dual-enrichment table + span_rates
wiring) and supersedes #49's viewer wiring; #49 contributes the shell
(`index.html`, `styles.css`, `dicts.json`) and the vendored `js/vendor/ngl.js`.

## Data the viewer reads

The front-end fetches static JSON that the assembler places **next to** these
files, one subtree per dictionary (`<dict>/`) plus a few top-level sidecars. The
per-feature JSON lives at `<dict>/feature/latent_<NNNNN>.json`.

### Per-feature JSON

Each feature carries a `detected[]` array (the ranked annotations for that
feature). Every entry has:

- `id`   — annotation id, e.g. `ted|1.10.760.10`, `cath|3.40.50.720`,
  `pfam|PF00012`, or a genomic-layer type.
- `class` — one of `ted`, `cath`, `pfam`, or a genomic class.
- `pretty` — display label.
- `recall` — fraction of the annotation's spans the feature covers.
- position-level `fold` — the capped (999×) Haldane circular-shift ratio.

and, embedded by `atlas.enrichment`, the **dual enrichment metrics**:

- `span_fold` — span-level gate number = `cover_rate / matched_bg_rate`.
- `cover_rate` — fraction of the fold's spans covered (= `recall`).
- `matched_bg_rate` — expected covered fraction under the matched-negative background.
- `pos_rate`, `bg_rate` — in-fold and background per-position firing rates
  (the numerator/denominator behind a capped `fold`).
- `lift` = `ppv / prior` — the position-level effect size that replaces the capped
  `fold` in the display, where `ppv` = P(in annotation | feature fires) (bounded
  `[0, 1]`) and `prior` = P(in annotation) = annotation positions / total positions.
- `ppv`, `prior` — the two components of `lift`.

A legacy `top_annotations[]` array may also be present and is augmented with the
same fields where a matching annotation id can be joined.

### Per-dictionary sidecars (`<dict>/`)

- `index_rows.json` — one row per feature for the index table (`feature_id`,
  `top_annotation`, `top_class`, `top_fold`, `top_recall`, …).
- `rates.json` — `{feature: {ann_id: [pos_rate, bg_rate]}}`; source of the
  position-level rates + `lift`.
- `span_rates.json` — `{feature: {ann_id: [precision_fold, cover_rate, bg_rate]}}`;
  source of the span-level metrics (ted/cath from #40, pfam/genomic from #38).

### Top-level sidecars

- `dicts.json` — dictionary catalog (shipped with the shell).
- `names.json` — human-readable names, copied from the atlas root.
- `struct_index.json` — per-feature -> structure (UniProt/CIF) index.
- `struct_coverage.json` — per-dict structure-coverage percentages.
- `af_cache/` — AlphaFold-DB + ESMFold CIF structures (hardlinked, ~50 GB).

## How it is assembled and served

The viewer is assembled by `basecamp_eden_saes.atlas.deploy`:

```
python -m basecamp_eden_saes.atlas.deploy assemble \
    --app <dest> \
    --atlas-root <dir with og2/bcr_k64/bcr_k16 + names.json> \
    --struct <dir with struct_index.json + struct_coverage.json> \
    [--viewer-src <this viewer/ dir>]   # defaults to this repo's viewer/
    [--af-cache <AlphaFold cache dir>]  # defaults to config.data_paths().af_cache
```

`assemble` copies this front-end into `<dest>/viewer/`, hardlinks the per-dict
atlas trees and `af_cache/` (via `cp -al`, to avoid duplicating ~100 GB on the
shared filesystem), copies the structure sidecars, and writes `<dest>/manifest.json`.

The per-dict atlas JSON is produced upstream by the other `atlas` modules
(`rebuild` -> `span_rates` -> `enrichment`, then `deploy apply`). Serve `<dest>`
(or `<dest>/viewer`) with any static file server; open `index.html`.

## Note on the `lift` metric (superseded #42 basis)

Each detected annotation entry carries a position-level `lift` = PPV / prior
(`atlas/enrichment.py`). The `lift` values embedded in the *deployed* atlas — and
therefore shown in this viewer (e.g. feature 2044 / `cath|3.40.1090.10` reads
**244.3**) — rest on the earlier, sparser **#42 atlas position-rate basis**, which
the packaging reconciliation lists as *superseded*. They are preserved as-is so
the deployed atlas matches the source thread.

They are **not** recomputed from the canonical #38 code store this repository takes
as input. Recomputing `lift` from the canonical inputs (`bes-reproduce`) gives a
different value on the denser #38 activation basis (377.8 for feature 2044). The
span-level metrics shown in the viewer (`recall`, `span_fold`) *do* derive from the
#38 store and reproduce exactly. See `docs/REPRODUCTION.md` for the full account.
