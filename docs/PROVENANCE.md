# Provenance

This repository is the packaged, rewritten form of a Silico research thread that
built and interpreted sparse autoencoders for the EDEN-7B DNA foundation model.
The code here is a clean re-synthesis of the thread's experiment code — it is not a
copy of the experiment worktrees.

## Source experiments

| # | role in this repo | module(s) |
| --- | --- | --- |
| 16 | OpenGenome2 corpus build + activation-harvest engine | `corpus/`, `harvest/` |
| 26 | BatchTopK SAE training (EDEN-BCR layer 28, expansion×k grid) + selection | `sae/` |
| 31 | base RefSeq annotation panel (per-nt tracks, Pfam/GO domains) | `annotations/` |
| 38 | annotation panel v2 (genomic layers) + full-sparse code store + span autointerp | `annotations/`, `harvest/code_store.py`, `autointerp/` |
| 40 | CATH-S95 + TED fold layers and span enrichment | `structure/` |
| 49 | AlphaFold-DB structure coverage (+ deferred ESMFold gap-fold) | `structure/` |
| 50 | structure-beyond-sequence generalization (corrected both-strand rejoin) | `analysis/` |
| 53 | canonical per-feature atlas + standalone viewer | `atlas/`, `viewer/` |

## Reconciliation across overlapping experiments

Where experiments overlapped, the latest/corrected version was taken:

- **Viewer / atlas:** experiment 53 is canonical. It supersedes the earlier viewer
  rebuilds (42, 52, 47) and 49's viewer wiring. 49 still contributes the viewer
  shell (`index.html`, `styles.css`, `dicts.json`) and the vendored 3D library
  (`ngl.js`); 53's `viewer_src/js` (dual span/position enrichment table,
  `span_rates.json`) is the shipped front-end logic.
- **Structure-beyond-sequence:** experiment 50's corrected both-strand rejoin
  supersedes 41's plus-only rejoin (which silently recorded minus-strand genes as
  non-firing). Only 50 is shipped.
- **Autointerp / annotations:** experiment 38's v2 per-span metrics and code store
  supersede 31's original per-position pass. 31 is kept only for the base-panel
  build code (per-nt tracks, Pfam/GO domains) that 38 consumes read-only.

## Referenced (not shipped) experiments

- **57** built the consolidated annotation bundle `eden_annotation_panel_v1`, the
  documented single annotation input. Referenced by path (see the README).
- **41 / 42 / 47 / 52** are superseded rebuilds; their corrected outputs live in
  experiments 50 / 53. **35** supplied the spatial-clustering z-scores used by the
  optional clustering-link analysis (`analysis/aggregate.py`), referenced by path.

## Internal library dependency

`goodfire-core` (private Goodfire repository) is imported, never vendored. It is
pinned in `pyproject.toml` to the revision the harvest / SAE-training experiments
ran against (`b4c2083…`). The autointerp/analysis/atlas metric code does not depend
on goodfire-core, so the deterministic reproduction check runs without it.
