"""Fresh-clone configs and private token-import paths; no live credentials."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "setup_drive_remote.sh"
TOKEN = {"access_token": "test-access-only", "refresh_token": "test-refresh-only"}


@pytest.fixture
def import_env(tmp_path):
    # Fake rclone consumes the write probe without contacting any remote.
    binary = tmp_path / "bin" / "rclone"
    binary.parent.mkdir()
    binary.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = rcat ]; then cat >/dev/null; fi\n"
        "if [ -n \"$FAIL_DRIVE_READ\" ] && [ \"$1\" = lsd ]; then exit 7; fi\n"
        "exit 0\n"
    )
    binary.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{binary.parent}{os.pathsep}{os.environ['PATH']}",
        "RCLONE_CONFIG": str(tmp_path / "private" / "rclone.conf"),
        "DRIVE_FOLDER_ID": "test-folder",
        "DRIVE_REMOTE_NAME": "gdrive",
    }


@pytest.mark.parametrize("source", ["stdin", "file"])
def test_existing_token_import_is_private_and_not_printed(import_env, tmp_path, source):
    args = ["bash", str(SCRIPT)]
    payload = json.dumps(TOKEN)
    if source == "stdin":
        args += ["--token-stdin"]
    else:
        token_file = tmp_path / "token.json"
        token_file.write_text(payload)
        args += ["--token-file", str(token_file)]
    result = subprocess.run(args, input=payload, text=True, capture_output=True, env=import_env)
    assert result.returncode == 0, result.stderr
    config = Path(import_env["RCLONE_CONFIG"])
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert "root_folder_id = test-folder" in config.read_text()
    assert TOKEN["refresh_token"] in config.read_text()
    assert "read  OK" in result.stdout and "write OK" in result.stdout
    assert all(secret not in result.stdout + result.stderr for secret in TOKEN.values())


def test_bare_access_token_is_rejected_before_config_is_written(import_env):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--token-stdin"],
        input=json.dumps({"access_token": "test-access"}),
        text=True, capture_output=True, env=import_env,
    )
    assert result.returncode == 2
    assert "refresh_token" in result.stderr
    assert not Path(import_env["RCLONE_CONFIG"]).exists()


def test_failed_read_does_not_claim_ready(import_env):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--token-stdin"], input=json.dumps(TOKEN),
        text=True, capture_output=True, env={**import_env, "FAIL_DRIVE_READ": "1"},
    )
    assert result.returncode == 1
    assert "read FAILED" in result.stderr
    assert "Ready" not in result.stdout


def test_import_preserves_other_remotes_and_backs_up_replaced_section(import_env):
    config = Path(import_env["RCLONE_CONFIG"])
    config.parent.mkdir()
    config.write_text("[other]\ntype = local\n\n[gdrive]\ntype = drive\ntoken = old\n")
    config.chmod(0o600)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--token-stdin"], input=json.dumps(TOKEN),
        text=True, capture_output=True, env=import_env,
    )
    assert result.returncode == 0, result.stderr
    assert "[other]" in config.read_text()
    assert config.read_text().count("[gdrive]") == 1
    assert list(config.parent.glob("rclone.conf.bak.*"))


def test_coworker_configs_enable_reasoning_and_keep_judge_out_of_evaluation(monkeypatch):
    monkeypatch.setenv("ABENCH_API_KEY", "test-key")
    monkeypatch.setenv("ABENCH_ADMIN_KEY", "test-key")
    monkeypatch.setenv("ABENCH_JUDGE_MODEL", "gpt-oss-20b-local")
    config = load_run_config(ROOT / "configs/runs/coworker.yaml")
    assert config.engine.sync.enabled
    assert config.engine.sync.remote_path == "gdrive:"
    assert config.engine.reasoning_judge.enabled
    assert config.engine.reasoning_judge.max_tokens == 2048
    assert config.engine.reasoning_judge.model == config.engine.judge.model
    assert [model.id for model in config.evaluated_models()] == ["qwen3-5-2b-local"]
    assert set(config.modes.prompt_modes) == {"io", "cot"}


def test_smoke_can_plan_both_kinds_without_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("ABENCH_API_KEY", "test-key")
    monkeypatch.setenv("ABENCH_ADMIN_KEY", "test-key")
    monkeypatch.setenv("ABENCH_JUDGE_MODEL", "gpt-oss-20b-local")
    config = load_run_config(
        ROOT / "configs/runs/coworker_smoke.yaml",
        overrides=[
            "engine.sync.enabled=false", f"engine.output_root={tmp_path}/runs",
            f"engine.data_root={tmp_path}/data", "engine.tokenizer.backend=heuristic",
        ],
    )
    assert [dataset.id for dataset in config.datasets] == ["smoke"]
    result = asyncio.run(EvaluationEngine(config, dry_run=True, offline=True).run())
    assert not result.skipped_datasets
    assert len(result.tasks) == 4
    assert {task.identity.task_kind for task in result.tasks} == {"generation", "selection"}
    assert sum(task.n_planned for task in result.tasks) == 8
