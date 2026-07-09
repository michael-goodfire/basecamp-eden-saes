"""Central configuration for data inputs referenced (not committed) by this package.

The model weights, the OpenGenome2 corpus, the annotation bundle, the 740 GB
code store, the per-dictionary matched-negative backgrounds, and the ~50 GB
AlphaFold structure cache are large external artifacts. They are resolved here
from environment variables, falling back to the canonical on-cluster paths they
were produced at. Override any of them with the matching ``EDEN_SAES_*`` env var
(or by constructing :class:`DataPaths` directly) to point at a local copy.

Nothing in this module reads a path at import time; resolution happens when you
call :func:`data_paths` or one of the accessors, so importing the package never
touches the filesystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- canonical on-cluster defaults (the artifacts the source thread produced) ---
_DEFAULT_MODELS = {
    "og2": "/mnt/data/shared/models/Eden-7B-OG2-286B",
    "bcr": "/mnt/data/shared/models/EDEN-7B-BCR",
}
_DEFAULT_CORPUS = (
    "/mnt/data/artifacts/silico/basecamp-eden-saes/datasets/og2-gtdb-metag-bcr4096-5b"
)
# Trained SAE checkpoints, one subdir per model layer (bcr-l28-v3, og2-l28).
_DEFAULT_SAE_CHECKPOINTS = "/mnt/data/artifacts/silico/basecamp-eden-saes/saes"
_DEFAULT_ANNOTATION_BUNDLE = "/mnt/data/artifacts/silico/eden_annotation_panel_v1"
# The 740 GB full-sparse code store (stable shared path; the per-model subdirs are
# code_store/{bcr,og2}).
_DEFAULT_CODE_STORE = "/mnt/data/artifacts/silico/basecamp-eden-saes/code_store"
# Per-dict matched-negative backgrounds (experiment #38 metrics; not relocated to
# the stable root, so still referenced at their produced location).
_DEFAULT_BG_ROOT = (
    "/mnt/data/artifacts/silico/experiments/_flat/"
    "exp_01kwxjy7zmf30b1tb9k58pspfp/metrics"
)
# Panel genome FASTA (experiment #31; used only for exact per-span GC). The bundle
# documents these are reconstructable from panel/accessions.txt via the NCBI
# Datasets CLI if this path is unavailable.
_DEFAULT_PANEL_GENOMES = (
    "/mnt/data/artifacts/silico/experiments/_flat/"
    "exp_01kwd6629qfvsvdjda4kfyft2z/panel/genomes"
)
# ~50 GB AlphaFold-DB structure cache (experiment #49 artifacts).
_DEFAULT_AF_CACHE = (
    "/mnt/data/artifacts/silico/experiments/_flat/"
    "exp_01kx09bzv8fjqvh3k37t6c9x47/af_cache"
)

# Code-store subdirectory + background file per dictionary name. The dictionary
# name identifies (model, expansion, k): the three shipped SAEs are the ef8 grid.
_DICT_CODE_SUBDIR = {
    "og2": "og2/og2_ef8_k64",
    "bcr_k64": "bcr/bcr_ef8_k64",
    "bcr_k16": "bcr/bcr_ef8_k16",
}
_DICT_BG_SUBDIR = {"og2": "og2", "bcr_k64": "bcr_k64", "bcr_k16": "bcr_k16"}
_DICT_MODEL = {"og2": "og2", "bcr_k64": "bcr", "bcr_k16": "bcr"}
# SAE-checkpoint subdirectory (one per model layer) under the SAE-checkpoints root.
_DICT_SAE_SUBDIR = {"og2": "og2-l28", "bcr_k64": "bcr-l28-v3", "bcr_k16": "bcr-l28-v3"}

DICTS = ("og2", "bcr_k64", "bcr_k16")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class DataPaths:
    """Resolved locations of the external data inputs.

    All fields default to the canonical on-cluster paths and are overridable via
    the corresponding ``EDEN_SAES_*`` environment variable.
    """

    models: dict[str, str]
    corpus: Path
    sae_checkpoints: Path
    annotation_bundle: Path
    code_store: Path
    bg_root: Path
    panel_genomes: Path
    af_cache: Path

    # --- per-dictionary accessors ---
    def model_path(self, dict_name: str) -> str:
        """HuggingFace/local path of the EDEN model backing ``dict_name``."""
        return self.models[_DICT_MODEL[dict_name]]

    def code_dir(self, dict_name: str) -> Path:
        """Directory of the full-sparse code store for ``dict_name``."""
        return self.code_store / _DICT_CODE_SUBDIR[dict_name]

    def bg_npz(self, dict_name: str) -> Path:
        """Matched-negative background ``bg.npz`` for ``dict_name``."""
        return self.bg_root / _DICT_BG_SUBDIR[dict_name] / "bg.npz"

    def sae_checkpoint(self, dict_name: str, label: str) -> Path:
        """Trained SAE checkpoint path, e.g. ``sae_checkpoint('bcr_k64', 'ef8_k64')``."""
        return self.sae_checkpoints / _DICT_SAE_SUBDIR[dict_name] / f"{label}.pt"

    # --- annotation bundle sub-paths ---
    def bundle_accessions(self) -> Path:
        return self.annotation_bundle / "panel" / "accessions.txt"

    def bundle_spans(self, layer: str, accession: str) -> Path:
        """Span TSV for a fold layer (``cath`` or ``ted``) and genome accession."""
        return self.annotation_bundle / "layers" / layer / "spans" / f"{accession}.tsv"

    def genome_fasta(self, accession: str) -> Path:
        """Panel genome FASTA (for exact per-span GC)."""
        return self.panel_genomes / accession / "genomic.fna"


def data_paths() -> DataPaths:
    """Resolve :class:`DataPaths` from the environment, with on-cluster defaults."""
    models = {
        "og2": _env("EDEN_SAES_MODEL_OG2", _DEFAULT_MODELS["og2"]),
        "bcr": _env("EDEN_SAES_MODEL_BCR", _DEFAULT_MODELS["bcr"]),
    }
    return DataPaths(
        models=models,
        corpus=Path(_env("EDEN_SAES_CORPUS", _DEFAULT_CORPUS)),
        sae_checkpoints=Path(_env("EDEN_SAES_SAE_CHECKPOINTS", _DEFAULT_SAE_CHECKPOINTS)),
        annotation_bundle=Path(_env("EDEN_SAES_BUNDLE", _DEFAULT_ANNOTATION_BUNDLE)),
        code_store=Path(_env("EDEN_SAES_CODE_STORE", _DEFAULT_CODE_STORE)),
        bg_root=Path(_env("EDEN_SAES_BG_ROOT", _DEFAULT_BG_ROOT)),
        panel_genomes=Path(_env("EDEN_SAES_PANEL_GENOMES", _DEFAULT_PANEL_GENOMES)),
        af_cache=Path(_env("EDEN_SAES_AF_CACHE", _DEFAULT_AF_CACHE)),
    )


def accessions(paths: DataPaths | None = None) -> list[str]:
    """Read the 152-genome panel accession list from the annotation bundle."""
    paths = paths or data_paths()
    with open(paths.bundle_accessions()) as fh:
        return [ln.strip() for ln in fh if ln.strip()]
