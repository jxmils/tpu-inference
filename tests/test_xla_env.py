# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path

import pytest


def _load_xla_env():
    root = Path(__file__).resolve().parents[1] / "tpu_inference" / "xla_env.py"
    spec = importlib.util.spec_from_file_location("xla_env", root)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


apply_xla_dump_from_env = _load_xla_env().apply_xla_dump_from_env


@pytest.fixture
def clean_xla_env(monkeypatch):
    monkeypatch.delenv("XLA_DUMP_TO", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)


def test_apply_xla_dump_from_env_merges_flag(monkeypatch, clean_xla_env, tmp_path):
    monkeypatch.setenv("XLA_DUMP_TO", str(tmp_path))
    apply_xla_dump_from_env()
    assert f"--xla_dump_to={tmp_path}" in os.environ["XLA_FLAGS"]


def test_apply_xla_dump_from_env_preserves_existing_flags(
        monkeypatch, clean_xla_env, tmp_path):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
    monkeypatch.setenv("XLA_DUMP_TO", str(tmp_path))
    apply_xla_dump_from_env()
    flags = os.environ["XLA_FLAGS"]
    assert "--xla_gpu_enable_triton_gemm=false" in flags
    assert f"--xla_dump_to={tmp_path}" in flags


def test_apply_xla_dump_from_env_idempotent(monkeypatch, clean_xla_env, tmp_path):
    monkeypatch.setenv("XLA_DUMP_TO", str(tmp_path))
    apply_xla_dump_from_env()
    first = os.environ["XLA_FLAGS"]
    apply_xla_dump_from_env()
    assert os.environ["XLA_FLAGS"] == first


def test_apply_xla_dump_from_env_noop_without_env(monkeypatch, clean_xla_env):
    apply_xla_dump_from_env()
    assert "XLA_FLAGS" not in os.environ
