"""A run directory is shared, so the config written into it must carry no secret.

`engine.sync` uploads the whole run directory, `run_config.resolved.yaml`
included. That file is provenance and it is meant to be safe to hand to
anyone.

It was not. Redaction walked the paths it knew about -- every model endpoint,
its batch override, the two header bags -- and was correct about all of them.
The enumeration was the defect: `engine.simulator.api_key` was added later,
holds a live credential, and was never on the list, so it was written in
plaintext into every run directory and uploaded from there. Its own docstring
says "read from the environment, never written to a config file or a log".

So these tests assert the property, not the paths: no value anywhere in the
document looks like a credential. A new secret field is covered the day it is
added rather than the day someone remembers to extend a list.
"""

from __future__ import annotations

import re

import yaml

from abductionbench.core.config import RunConfig, dump_resolved

#: Shapes of real credentials this project handles.
_CREDENTIAL_SHAPES = (
    re.compile(r"sk-[A-Za-z0-9-]{12,}"),      # OpenAI / OpenRouter
    re.compile(r"hf_[A-Za-z0-9]{12,}"),       # Hugging Face
    re.compile(r"vllm-[A-Za-z0-9]{12,}"),     # this box's vLLM servers
)

SIMULATOR_KEY = "sk-or-v1-000000000000000000000000000000000000000000000000"
MODEL_KEY = "sk-model-000000000000000000000000"
DATASET_KEY = "sk-or-v1-111111111111111111111111111111111111111111111111"
HF = "hf_000000000000000000000000"


def _config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "t",
            "prompts": {},
            "engine": {
                "simulator": {
                    "enabled": True,
                    "base_url": "https://openrouter.ai/api",
                    "model": "openai/gpt-4o-mini",
                    "api_key": SIMULATOR_KEY,
                    "by_dataset": {"vivabench": {"api_key": DATASET_KEY}},
                }
            },
            "models": [
                {
                    "id": "m",
                    "model_name": "local/model",
                    "endpoint": {
                        "base_url": "http://127.0.0.1:18000",
                        "api_key": MODEL_KEY,
                        "headers": {"Authorization": f"Bearer {MODEL_KEY}"},
                        "batch": {"api_key": MODEL_KEY},
                    },
                }
            ],
            "datasets": [],
        }
    )


def _dumped(tmp_path) -> tuple[str, dict]:
    path = tmp_path / "run_config.resolved.yaml"
    dump_resolved(_config(), path)
    text = path.read_text()
    return text, yaml.safe_load(text)


def test_no_value_anywhere_in_the_document_looks_like_a_credential(tmp_path):
    """The property that matters, checked over the whole file rather than by path."""
    text, _ = _dumped(tmp_path)
    for shape in _CREDENTIAL_SHAPES:
        found = shape.search(text)
        assert found is None, f"credential-shaped value survived redaction: {found.group()[:12]}..."


def test_the_simulator_key_is_redacted(tmp_path):
    """The field that was actually leaking, named explicitly so it stays fixed."""
    _, data = _dumped(tmp_path)
    assert data["engine"]["simulator"]["api_key"] == "***redacted***"


def test_a_per_dataset_simulator_override_is_redacted_too(tmp_path):
    """The same hole one level down: vivabench gets its own key."""
    _, data = _dumped(tmp_path)
    assert data["engine"]["simulator"]["by_dataset"]["vivabench"]["api_key"] == "***redacted***"


def test_the_model_paths_that_already_worked_still_work(tmp_path):
    _, data = _dumped(tmp_path)
    endpoint = data["models"][0]["endpoint"]
    assert endpoint["api_key"] == "***redacted***"
    assert endpoint["batch"]["api_key"] == "***redacted***"
    assert endpoint["headers"]["Authorization"] == "***redacted***"


def test_settings_that_merely_mention_tokens_are_not_redacted(tmp_path):
    """Redaction must not eat the provenance this file exists for.

    `max_tokens`, `max_tokens_cap` and `tiktoken_encoding` are settings. A
    substring rule on "token" or "key" would destroy them, which is why the
    match is exact names plus `_token`-style suffixes.
    """
    _, data = _dumped(tmp_path)
    sampling = data["models"][0]["sampling"]
    assert isinstance(sampling["max_tokens_cap"], int)
    assert isinstance(sampling["max_tokens_default"], int)
    assert data["engine"]["tokenizer"]["tiktoken_encoding"] == "cl100k_base"
    assert data["engine"]["simulator"]["max_tokens"] == 512


def test_an_absent_credential_stays_null_rather_than_becoming_a_redaction(tmp_path):
    """"Not set" and "set but hidden" must stay distinguishable."""
    config = _config()
    config.models[0].endpoint.batch.api_key = None
    path = tmp_path / "c.yaml"
    dump_resolved(config, path)
    data = yaml.safe_load(path.read_text())
    assert data["models"][0]["endpoint"]["batch"]["api_key"] is None


def test_redact_false_still_writes_the_real_values(tmp_path):
    """The flag has one caller and one purpose; it must keep working."""
    path = tmp_path / "c.yaml"
    dump_resolved(_config(), path, redact=False)
    assert SIMULATOR_KEY in path.read_text()
