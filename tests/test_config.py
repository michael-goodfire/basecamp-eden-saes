"""Unit tests for data-path resolution and env overrides."""

from __future__ import annotations

from basecamp_eden_saes import config


def test_defaults_and_dict_mapping():
    p = config.data_paths()
    # per-dictionary code-store subdir and model routing
    assert p.code_dir("bcr_k64").as_posix().endswith("code_store/bcr/bcr_ef8_k64")
    assert p.code_dir("og2").as_posix().endswith("code_store/og2/og2_ef8_k64")
    assert p.model_path("bcr_k64") == p.models["bcr"]
    assert p.model_path("og2") == p.models["og2"]
    # SAE checkpoint routes to the per-model layer subdir
    assert p.sae_checkpoint("bcr_k64", "ef8_k64").as_posix().endswith(
        "saes/bcr-l28-v3/ef8_k64.pt"
    )
    assert p.sae_checkpoint("og2", "ef8_k64").as_posix().endswith("saes/og2-l28/ef8_k64.pt")
    # background per dict
    assert p.bg_npz("bcr_k16").as_posix().endswith("metrics/bcr_k16/bg.npz")


def test_env_override(monkeypatch):
    monkeypatch.setenv("EDEN_SAES_BUNDLE", "/custom/bundle")
    monkeypatch.setenv("EDEN_SAES_CODE_STORE", "/custom/codes")
    p = config.data_paths()
    assert p.annotation_bundle.as_posix() == "/custom/bundle"
    assert p.code_dir("bcr_k64").as_posix() == "/custom/codes/bcr/bcr_ef8_k64"
    assert p.bundle_spans("cath", "GCF_1").as_posix() == "/custom/bundle/layers/cath/spans/GCF_1.tsv"


def test_dicts_constant():
    assert set(config.DICTS) == {"og2", "bcr_k64", "bcr_k16"}
