"""Shared fixtures: config assembly against the fake server and adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))  # so 'fake_adapter' is importable by config

from fake_server import FakeServer  # noqa: E402


@pytest.fixture
def fake_server():
    with FakeServer() as server:
        yield server


@pytest.fixture
def prompt_dir() -> Path:
    return REPO_ROOT / "configs" / "prompts"


@pytest.fixture
def write_run_config(tmp_path: Path, prompt_dir: Path):
    """Factory writing a complete run config pointing at the fake server."""

    def _write(
        *,
        base_url: str,
        datasets: list[dict[str, Any]],
        engine: dict[str, Any] | None = None,
        prompts: dict[str, Any] | None = None,
        modes: dict[str, Any] | None = None,
        models: list[dict[str, Any]] | None = None,
        name: str = "test-run",
    ) -> Path:
        model = {
            "id": "fake-model",
            "model_name": "test/model",
            "endpoint": {
                "base_url": base_url,
                "api_key": "test-key",
                "batch": {"enabled": True, "path": "/v1/chat/completions/batch", "group_size": 4},
            },
            "sampling": {"max_tokens_default": 64, "max_tokens_cap": 512, "max_tokens_floor": 16},
            "limits": {"max_parallel_batches": 2, "context_window": 4096},
        }
        config: dict[str, Any] = {
            "name": name,
            "seed": 1234,
            "engine": {
                "output_root": str(tmp_path / "runs"),
                "data_root": str(tmp_path / "data"),
                "tokenizer": {"backend": "heuristic", "safety_margin_tokens": 0},
                "retry": {
                    "max_attempts": 3,
                    "initial_backoff_s": 0.01,
                    "max_backoff_s": 0.05,
                    "rate_limit_backoff_s": 0.01,
                    "jitter": 0.0,
                    "recovery": {"enabled": False},
                },
                "reporting": {"include_sample_sheets": True},
                **(engine or {}),
            },
            "modes": modes or {},
            "prompts": {
                "template_dirs": [str(prompt_dir)],
                "bindings": {"generation": "gen_freeform_v1", "selection": "sel_mcq_letter_v1"},
                **(prompts or {}),
            },
            "models": models or [model],
            "datasets": datasets,
        }
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def fake_dataset():
    """Factory for a dataset entry backed by the test adapter."""

    def _dataset(dataset_id: str = "fake", **options: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": dataset_id,
            "impl": "fake_adapter:FakeAdapter",
            "sample_size": int(options.pop("sample_size", 8)),
            "options": options,
        }
        return entry

    return _dataset
