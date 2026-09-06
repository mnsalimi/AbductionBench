"""Configuration layering, interpolation, overrides and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from abductionbench.core.config import ConfigError, load_run_config


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_extends_and_override(tmp_path: Path, prompt_dir: Path):
    _write(
        tmp_path / "base.yaml",
        {
            "engine": {"concurrency": {"max_parallel_tasks": 7}, "limits": {"input_token_budget": 999}},
            "prompts": {"template_dirs": [str(prompt_dir)], "bindings": {"generation": "gen_freeform_v1"}},
        },
    )
    run = _write(
        tmp_path / "run.yaml",
        {
            "extends": ["base.yaml"],
            "name": "layered",
            "engine": {"limits": {"input_token_budget": 42}},
            "models": [
                {"id": "m", "model_name": "test/model", "endpoint": {"base_url": "http://x"}}
            ],
            "datasets": [{"id": "d", "impl": "fake_adapter:FakeAdapter"}],
        },
    )
    config = load_run_config(run)
    assert config.name == "layered"
    # inherited from base
    assert config.engine.concurrency.max_parallel_tasks == 7
    # overridden by the run file
    assert config.engine.limits.input_token_budget == 42
    assert config.datasets[0].sample_size == 300  # schema default


def test_env_interpolation_and_missing_env(tmp_path: Path, prompt_dir: Path):
    payload = {
        "prompts": {"template_dirs": [str(prompt_dir)], "bindings": {"generation": "gen_freeform_v1"}},
        "models": [
            {
                "id": "m",
                "model_name": "test/model",
                "endpoint": {"base_url": "${env:TEST_URL|http://fallback}", "api_key": "${env:TEST_KEY}"},
            }
        ],
        "datasets": [],
    }
    run = _write(tmp_path / "run.yaml", payload)
    os.environ.pop("TEST_URL", None)
    os.environ.pop("TEST_KEY", None)
    with pytest.raises(ConfigError, match="TEST_KEY"):
        load_run_config(run)
    os.environ["TEST_KEY"] = "secret"
    try:
        config = load_run_config(run)
        assert config.models[0].endpoint.base_url == "http://fallback"
        assert config.models[0].endpoint.api_key == "secret"
        os.environ["TEST_URL"] = "http://explicit/"
        config = load_run_config(run)
        assert config.models[0].endpoint.base_url == "http://explicit"  # trailing slash stripped
    finally:
        os.environ.pop("TEST_KEY", None)
        os.environ.pop("TEST_URL", None)


def test_dotted_set_override_and_filters(tmp_path: Path, prompt_dir: Path):
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {"template_dirs": [str(prompt_dir)], "bindings": {"generation": "gen_freeform_v1"}},
            "models": [
                {"id": "a", "model_name": "t/a", "endpoint": {"base_url": "http://a"}},
                {"id": "b", "model_name": "t/b", "endpoint": {"base_url": "http://b"}},
            ],
            "datasets": [
                {"id": "d1", "impl": "fake_adapter:FakeAdapter"},
                {"id": "d2", "impl": "fake_adapter:FakeAdapter"},
            ],
            "dataset_defaults": {"sample_size": 11},
        },
    )
    config = load_run_config(
        run,
        overrides=["engine.limits.input_token_budget=123", "seed=7"],
        dataset_filter=["d2"],
        model_filter=["b"],
    )
    assert config.engine.limits.input_token_budget == 123
    assert config.seed == 7
    assert [d.id for d in config.datasets] == ["d2"]
    assert [m.id for m in config.models] == ["b"]
    assert config.datasets[0].sample_size == 11  # dataset_defaults applied


def test_unknown_filter_and_duplicate_ids(tmp_path: Path, prompt_dir: Path):
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {"template_dirs": [str(prompt_dir)], "bindings": {"generation": "gen_freeform_v1"}},
            "models": [{"id": "a", "model_name": "t/a", "endpoint": {"base_url": "http://a"}}],
            "datasets": [
                {"id": "d", "impl": "fake_adapter:FakeAdapter"},
                {"id": "d", "impl": "fake_adapter:FakeAdapter"},
            ],
        },
    )
    with pytest.raises(ConfigError, match="unknown model ids"):
        load_run_config(run, model_filter=["zzz"])
    with pytest.raises(ConfigError, match="duplicate dataset ids"):
        load_run_config(run)


def test_model_file_reference_with_inline_override(tmp_path: Path, prompt_dir: Path):
    _write(
        tmp_path / "model.yaml",
        {
            "model": {
                "id": "ref",
                "model_name": "test/model",
                "endpoint": {
                    "base_url": "http://ref",
                    "batch": {"group_size": 8, "path": "/v1/chat/completions/batch"},
                },
            }
        },
    )
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {"template_dirs": [str(prompt_dir)], "bindings": {"generation": "gen_freeform_v1"}},
            "models": [{"file": "model.yaml", "id": "renamed"}],
            "datasets": [],
        },
    )
    config = load_run_config(run)
    assert config.models[0].id == "renamed"
    assert config.models[0].endpoint.batch.group_size == 8
    assert config.models[0].endpoint.resolved_batch_base_url() == "http://ref"


def test_batch_url_and_key_fallbacks(tmp_path: Path, prompt_dir: Path):
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {"template_dirs": [str(prompt_dir)], "bindings": {"generation": "gen_freeform_v1"}},
            "models": [
                {
                    "id": "m",
                    "model_name": "test/model",
                    "endpoint": {
                        "base_url": "http://gateway",
                        "api_key": "shared",
                        "batch": {"base_url": "http://tunnel", "api_key": "own"},
                    },
                }
            ],
            "datasets": [],
        },
    )
    endpoint = load_run_config(run).models[0].endpoint
    assert endpoint.resolved_batch_base_url() == "http://tunnel"
    assert endpoint.resolved_batch_api_key() == "own"


def test_nested_interpolation_fallback(tmp_path: Path, prompt_dir: Path, monkeypatch):
    """A fallback may itself be a placeholder: ${env:A|${env:B}}."""
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {"template_dirs": [str(prompt_dir)], "bindings": {"generation": "gen_freeform_v1"}},
            "models": [
                {
                    "id": "m",
                    "model_name": "test/model",
                    "endpoint": {
                        "base_url": "http://x",
                        "api_key": "${env:ABENCH_ADMIN_KEY|${env:ABENCH_API_KEY}}",
                    },
                }
            ],
            "datasets": [],
        },
    )
    monkeypatch.delenv("ABENCH_ADMIN_KEY", raising=False)
    monkeypatch.setenv("ABENCH_API_KEY", "shared-key")
    assert load_run_config(run).models[0].endpoint.api_key == "shared-key"
    monkeypatch.setenv("ABENCH_ADMIN_KEY", "admin-key")
    assert load_run_config(run).models[0].endpoint.api_key == "admin-key"


def test_dataset_force_overrides_dataset_config(tmp_path: Path, prompt_dir: Path):
    """`dataset_defaults` yields to a dataset's own value; `dataset_force` wins."""
    (tmp_path / "ds.yaml").write_text(
        yaml.safe_dump(
            {"dataset": {"id": "d", "impl": "fake_adapter:FakeAdapter", "sample_size": 300}}
        ),
        encoding="utf-8",
    )
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {
                "template_dirs": [str(prompt_dir)],
                "bindings": {"generation": "gen_freeform_v1"},
            },
            "models": [{"id": "m", "model_name": "t/m", "endpoint": {"base_url": "http://x"}}],
            "datasets": [{"file": "ds.yaml"}],
            "dataset_defaults": {"sample_size": 50, "seed": 7},
            "dataset_force": {"sample_size": 12},
        },
    )
    config = load_run_config(run)
    assert config.datasets[0].sample_size == 12  # forced over the dataset's own 300
    assert config.datasets[0].seed == 7          # default still fills what was unset


def test_datasets_glob_expands_and_skips_templates(tmp_path: Path, prompt_dir: Path):
    directory = tmp_path / "ds"
    directory.mkdir()
    for name in ("a", "b"):
        (directory / f"{name}.yaml").write_text(
            yaml.safe_dump({"dataset": {"id": name, "impl": "fake_adapter:FakeAdapter"}}),
            encoding="utf-8",
        )
    (directory / "_TEMPLATE.yaml").write_text(
        yaml.safe_dump({"dataset": {"id": "tpl", "impl": "x:Y"}}), encoding="utf-8"
    )
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {
                "template_dirs": [str(prompt_dir)],
                "bindings": {"generation": "gen_freeform_v1"},
            },
            "models": [{"id": "m", "model_name": "t/m", "endpoint": {"base_url": "http://x"}}],
            "datasets_glob": ["ds/*.yaml"],
        },
    )
    config = load_run_config(run)
    assert sorted(d.id for d in config.datasets) == ["a", "b"]  # _TEMPLATE ignored


def test_set_keeps_yaml_boolean_words_as_strings(tmp_path: Path, prompt_dir: Path):
    """`--set ...resume_policy=off` must not become the boolean False.

    YAML 1.1 reads bare `off`/`no`/`yes` as booleans, which would fail
    validation for a field whose allowed values include the literal "off".
    """
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {
                "template_dirs": [str(prompt_dir)],
                "bindings": {"generation": "gen_freeform_v1"},
            },
            "models": [{"id": "m", "model_name": "t/m", "endpoint": {"base_url": "http://x"}}],
            "datasets": [],
        },
    )
    config = load_run_config(run, overrides=["engine.checkpoint.resume_policy=off"])
    assert config.engine.checkpoint.resume_policy == "off"
    # Genuine booleans still parse as booleans.
    config = load_run_config(run, overrides=["engine.checkpoint.enabled=false"])
    assert config.engine.checkpoint.enabled is False


def test_inherited_paths_resolve_against_the_declaring_config(tmp_path: Path, prompt_dir: Path):
    """A `datasets_glob`/`template_dirs` declared in a parent must still resolve.

    The child config lives in a different directory, so resolving only against
    the child would break every inherited relative path.
    """
    parent_dir = tmp_path / "base"
    child_dir = tmp_path / "elsewhere" / "runs"
    (parent_dir / "ds").mkdir(parents=True)
    child_dir.mkdir(parents=True)
    (parent_dir / "ds" / "a.yaml").write_text(
        yaml.safe_dump({"dataset": {"id": "a", "impl": "fake_adapter:FakeAdapter"}}),
        encoding="utf-8",
    )
    _write(
        parent_dir / "parent.yaml",
        {
            "prompts": {
                "template_dirs": [str(prompt_dir)],
                "bindings": {"generation": "gen_freeform_v1"},
            },
            "datasets_glob": ["ds/*.yaml"],
        },
    )
    child = _write(
        child_dir / "child.yaml",
        {
            "extends": [str(parent_dir / "parent.yaml")],
            "name": "child",
            "models": [{"id": "m", "model_name": "t/m", "endpoint": {"base_url": "http://x"}}],
        },
    )
    config = load_run_config(child)
    assert [d.id for d in config.datasets] == ["a"]


def test_unmatched_glob_reports_where_it_looked(tmp_path: Path, prompt_dir: Path):
    run = _write(
        tmp_path / "run.yaml",
        {
            "prompts": {
                "template_dirs": [str(prompt_dir)],
                "bindings": {"generation": "gen_freeform_v1"},
            },
            "models": [{"id": "m", "model_name": "t/m", "endpoint": {"base_url": "http://x"}}],
            "datasets_glob": ["nowhere/*.yaml"],
        },
    )
    with pytest.raises(ConfigError, match="searched relative to"):
        load_run_config(run)


def test_run_ids_are_stamped_in_the_configured_timezone(tmp_path):
    """A run id names a directory and a backup folder that people read."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from abductionbench.core.engine import EvaluationEngine

    class _Stub:
        """Just enough engine to exercise the run-id stamping."""

        def __init__(self, zone):
            from abductionbench.core.config import EngineConfig

            self.engine_cfg = EngineConfig(run_id_timezone=zone)

        _run_id_tz = EvaluationEngine._run_id_tz
        _default_run_id = EvaluationEngine._default_run_id

    tehran = _Stub("Asia/Tehran")._default_run_id("full")
    utc = _Stub("UTC")._default_run_id("full")
    assert tehran.endswith("_full") and utc.endswith("_full")

    # Tehran is UTC+03:30, so the two stamps differ by 3h30m.
    fmt = "%Y%m%d-%H%M%S"
    delta = datetime.strptime(tehran.split("_")[0], fmt) - datetime.strptime(
        utc.split("_")[0], fmt
    )
    assert 3 * 3600 + 25 * 60 <= delta.total_seconds() <= 3 * 3600 + 35 * 60

    # And it really is Tehran's wall clock, not an arbitrary offset.
    expected = datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y%m%d-%H%M")
    assert tehran.startswith(expected)
    assert datetime.now(timezone.utc).strftime("%Y%m%d-%H%M") in utc


def test_an_unusable_timezone_falls_back_to_utc_rather_than_failing(caplog):
    """A run that cannot start is worse than one named in the wrong timezone."""
    from datetime import timezone

    from abductionbench.core.config import EngineConfig
    from abductionbench.core.engine import EvaluationEngine

    class _Stub:
        def __init__(self):
            self.engine_cfg = EngineConfig(run_id_timezone="Mars/Olympus_Mons")

        _run_id_tz = EvaluationEngine._run_id_tz

    assert _Stub()._run_id_tz() is timezone.utc
