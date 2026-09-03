"""Configuration schema and loader.

Every knob the engine has is declared here and set from YAML.  Nothing about a
run -- endpoints, batch group sizes, retry policy, sampling, prompt templates,
which datasets, sample size, token budgets -- is hardcoded anywhere else.

Composition rules
-----------------
* A run config may ``extends`` one or more other YAML files; later files
  override earlier ones, and the run file itself overrides all of them
  (deep merge for mappings, replace for scalars/lists).
* String values support interpolation:
  ``${env:VAR}``, ``${env:VAR|fallback}``, ``${file:/path/to/secret}`` and
  ``${cmd:shell command}`` (stdout, stripped).  This keeps API keys and
  rotating tunnel URLs out of the config files.
* CLI ``--set dotted.path=value`` overrides are applied last.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError

__all__ = [
    "ConcurrencyConfig",
    "BatchingConfig",
    "RetryConfig",
    "EndpointRecoveryConfig",
    "TimeoutConfig",
    "CheckpointConfig",
    "TokenizerConfig",
    "LimitsConfig",
    "LoggingConfig",
    "JudgeConfig",
    "SyncConfig",
    "EngineConfig",
    "BatchEndpointConfig",
    "EndpointConfig",
    "ModelSamplingConfig",
    "ModelLimitsConfig",
    "ModelConfig",
    "PromptConfig",
    "DatasetConfig",
    "RunConfig",
    "load_run_config",
    "load_yaml",
    "interpolate",
]


# --------------------------------------------------------------------------- #
# YAML loading, interpolation, merging
# --------------------------------------------------------------------------- #

_INTERP_START = re.compile(r"\$\{(env|file|cmd):")


def _resolve_token(kind: str, spec: str) -> str:
    """Resolve one ``${kind:spec}`` placeholder.

    For ``env``, ``spec`` is ``NAME`` or ``NAME|fallback``; the fallback is
    itself interpolated, and only when the variable is actually unset -- so
    ``${env:ADMIN_KEY|${env:API_KEY}}`` does not require ``API_KEY`` to exist
    when ``ADMIN_KEY`` does.
    """
    if kind == "env":
        name, sep, fallback = spec.partition("|")
        value = os.environ.get(name.strip())
        if value is not None:
            return value
        if not sep:
            raise ConfigError(
                f"environment variable {name.strip()!r} is referenced in config "
                "but not set (use ${env:NAME|fallback} to allow a default)"
            )
        return _interpolate_str(fallback)
    if kind == "file":
        path = Path(_interpolate_str(spec).strip()).expanduser()
        if not path.is_file():
            raise ConfigError(f"config references missing file {path}")
        return path.read_text(encoding="utf-8").strip()
    if kind == "cmd":
        command = _interpolate_str(spec).strip()
        try:
            out = subprocess.run(  # noqa: S602 - command comes from trusted local config
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        except subprocess.SubprocessError as exc:  # pragma: no cover - env specific
            raise ConfigError(f"config command failed: {command!r}: {exc}") from exc
        return out.stdout.strip()
    raise ConfigError(f"unknown interpolation kind {kind!r}")


def _interpolate_str(text: str) -> str:
    """Substitute every placeholder in ``text``, honouring nested braces.

    A regex cannot do this correctly: ``${env:A|${env:B}}`` needs brace
    matching, so placeholders are located by scanning and resolved recursively.
    """
    out: list[str] = []
    position = 0
    while True:
        match = _INTERP_START.search(text, position)
        if match is None:
            out.append(text[position:])
            return "".join(out)
        out.append(text[position : match.start()])
        kind = match.group(1)
        # Find the '}' that closes this '${', skipping nested placeholders.
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth:
            raise ConfigError(f"unterminated placeholder in config value {text!r}")
        out.append(_resolve_token(kind, text[match.end() : index]))
        position = index + 1


def interpolate(value: Any) -> Any:
    """Recursively resolve ``${env:...}`` / ``${file:...}`` / ``${cmd:...}``.

    Keeps secrets and rotating tunnel URLs out of the YAML files themselves.
    """
    if isinstance(value, str):
        return _interpolate_str(value)
    if isinstance(value, dict):
        return {k: interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file (no interpolation, no merging)."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {p} must contain a mapping at the top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins.  Lists are replaced, not merged."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


#: YAML 1.1 reads these bare words as booleans, which silently breaks a --set
#: whose value is a real choice name (``resume_policy=off``).
_YAML_BOOL_WORDS = {"on", "off", "yes", "no", "y", "n"}


def _apply_dotted(target: dict[str, Any], dotted: str, raw_value: str) -> None:
    """Apply ``a.b.c=value`` onto a nested dict, parsing the value as YAML.

    ``off`` and friends are kept as strings: YAML would turn them into
    booleans, so ``--set engine.checkpoint.resume_policy=off`` would fail
    validation even though ``off`` is exactly one of the allowed values.
    """
    keys = dotted.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"--set {dotted}: {key!r} is not a mapping")
    stripped = raw_value.strip()
    if stripped.lower() in _YAML_BOOL_WORDS and not (
        stripped.startswith(("'", '"'))
    ):
        node[keys[-1]] = stripped
        return
    node[keys[-1]] = yaml.safe_load(raw_value)


def _resolve_relative(path: str | Path, anchor: Path) -> Path:
    """Resolve ``path`` relative to the config file that mentioned it, then CWD."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    anchored = (anchor.parent / candidate).resolve()
    if anchored.exists():
        return anchored
    return candidate.resolve()


def _resolve_against_sources(
    path: str | Path, anchor: Path, sources: Sequence[str]
) -> Path:
    """Resolve a relative path against any config file in the ``extends`` chain.

    A key like ``datasets_glob`` or ``template_dirs`` is often declared in a
    *parent* config and inherited by a child that lives in a different
    directory.  Resolving only against the child would break the inherited
    value, so every file that contributed to this configuration is tried --
    deepest (the child) first, then its parents, then the working directory.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    anchors = [anchor, *(Path(source) for source in reversed(list(sources)))]
    for source in anchors:
        resolved = (source.parent / candidate).resolve()
        if resolved.exists():
            return resolved
    return candidate.resolve()


def load_layered(
    path: str | Path,
    *,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a YAML file plus everything it ``extends``, deep-merged.

    ``extends`` may be a single path or a list; paths are resolved relative to
    the file that declares them.  Cycles raise :class:`ConfigError`.
    """
    p = Path(path).resolve()
    seen = _seen or set()
    if p in seen:
        raise ConfigError(f"circular 'extends' involving {p}")
    seen = seen | {p}

    raw = load_yaml(p)
    parents = raw.pop("extends", []) or []
    if isinstance(parents, (str, Path)):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        merged = deep_merge(merged, load_layered(_resolve_relative(parent, p), _seen=seen))
    merged = deep_merge(merged, raw)
    merged.setdefault("_source_files", [])
    merged["_source_files"] = [*merged["_source_files"], str(p)]
    return merged


# --------------------------------------------------------------------------- #
# Engine configuration
# --------------------------------------------------------------------------- #


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConcurrencyConfig(_Base):
    """How much work runs at once."""

    #: (dataset x model x template) units evaluated in parallel.
    max_parallel_tasks: int = Field(2, ge=1)
    #: Global ceiling on in-flight batch calls across all models.
    max_parallel_batches_global: int = Field(8, ge=1)
    #: Threads used for adapter scoring (scoring is sync, possibly CPU-bound).
    scoring_workers: int = Field(4, ge=1)


class BatchingConfig(_Base):
    """How samples are packed into native batch calls.

    vLLM's ``/v1/chat/completions/batch`` shares one sampling-parameter set
    across the whole call, so only samples with an identical sampling signature
    can travel together.  ``max_tokens_quantum`` rounds adapter-supplied output
    budgets up to a common grid so per-sample budgets do not shatter batches
    into singletons.
    """

    max_tokens_quantum: int = Field(256, ge=1)
    #: Group samples of similar input length together (shorter head-of-line
    #: blocking within a batch call, since a batch returns only when its
    #: slowest conversation finishes).
    sort_by_input_tokens: bool = True
    #: On an invalid-request/context-length failure, split the batch and retry
    #: the halves to isolate the offending sample instead of failing all of it.
    bisect_on_invalid_request: bool = True
    #: Also split a batch whose retries were exhausted by timeouts/5xx.  A large
    #: batch can exceed a proxy's request ceiling (Cloudflare quick tunnels cut
    #: requests off at ~100s with a 524) or simply queue behind a busy server;
    #: halving it makes each call finish sooner, and is strictly better than
    #: failing every sample in the batch.
    bisect_on_timeout: bool = True
    #: Fall back to single-sample (non-batch) calls when the batch endpoint is
    #: unavailable for a model (e.g. no pass-through route configured).
    fallback_to_single: bool = True
    #: Hard ceiling regardless of a model's own ``group_size``.
    max_group_size: int = Field(64, ge=1)


class EndpointRecoveryConfig(_Base):
    """Recovery behaviour for tunnels that drop or rotate their URL."""

    enabled: bool = True
    probe_path: str = "/v1/models"
    probe_timeout_s: float = Field(30.0, gt=0)
    #: Re-run the model's ``discovery`` commands after a connection error, so a
    #: restarted Cloudflare quick tunnel (new URL) is picked up automatically.
    rediscover_on_connection_error: bool = True
    #: Wait between recovery probes.
    probe_interval_s: float = Field(15.0, gt=0)
    max_probe_attempts: int = Field(20, ge=1)


class RetryConfig(_Base):
    """Retry policy, per error class."""

    max_attempts: int = Field(5, ge=1)
    initial_backoff_s: float = Field(2.0, ge=0)
    max_backoff_s: float = Field(120.0, ge=0)
    backoff_multiplier: float = Field(2.0, ge=1.0)
    #: Multiplicative jitter fraction; 0.3 means +/-30%.
    jitter: float = Field(0.3, ge=0, le=1)
    #: Dedicated (longer) wait for HTTP 429.
    rate_limit_backoff_s: float = Field(20.0, ge=0)
    #: Error classes that are retried at all.
    retry_error_classes: list[str] = Field(
        default_factory=lambda: ["transient", "rate_limit", "unknown", "protocol"]
    )
    recovery: EndpointRecoveryConfig = Field(default_factory=EndpointRecoveryConfig)
    #: Re-issue a sample whose response came back empty *because the token
    #: budget ran out* (a reasoning model can spend its whole ``max_tokens`` on
    #: hidden chain-of-thought and return ``content: null``).  Only that sample
    #: is retried, with a multiplied budget -- far cheaper than raising the
    #: budget for every sample in the dataset.
    escalate_empty_responses: bool = True
    empty_budget_multiplier: float = Field(2.0, gt=1.0)
    max_empty_escalations: int = Field(1, ge=0)
    #: Also re-issue a sample whose answer was *cut off* at the budget
    #: (``finish_reason="length"`` with partial content).  Off by default: a
    #: verbose model would double the cost of every long-form dataset.  Turn it
    #: on for datasets where a truncated answer is unscorable (a symbolic
    #: equation, a label list) rather than merely shorter.
    escalate_truncated_responses: bool = False


class TimeoutConfig(_Base):
    """HTTP timeouts.  ``read_s`` must accommodate a whole batch generation."""

    connect_s: float = Field(30.0, gt=0)
    read_s: float = Field(1800.0, gt=0)
    write_s: float = Field(120.0, gt=0)
    pool_s: float = Field(60.0, gt=0)


class CheckpointConfig(_Base):
    """Incremental persistence and resume."""

    enabled: bool = True
    #: ``strict``  -- resume only records whose prompt fingerprint still matches
    #:                (template/sampling/model change invalidates them);
    #: ``sample_id`` -- resume by sample id regardless of prompt changes;
    #: ``off``      -- always re-run everything.
    resume_policy: Literal["strict", "sample_id", "off"] = "strict"
    #: fsync the records file after every append (safest) or every N appends.
    fsync_every: int = Field(1, ge=1)
    #: Persist the exact request/response JSON of each batch call for audit.
    store_raw_payloads: bool = True
    #: Keep at most this many raw payload files per task (0 = unlimited).
    max_raw_payloads: int = Field(0, ge=0)


class TokenizerConfig(_Base):
    """How input tokens are counted for budget enforcement.

    ``auto`` tries the model's own HF tokenizer (exact), then ``tiktoken``
    (close), then a character heuristic (rough but dependency-free).
    """

    backend: Literal["auto", "hf", "tiktoken", "heuristic", "endpoint"] = "auto"
    hf_model: str | None = None
    tiktoken_encoding: str = "cl100k_base"
    chars_per_token: float = Field(3.6, gt=0)
    #: Per-message overhead added to account for chat-template scaffolding.
    per_message_overhead: int = Field(4, ge=0)
    #: Safety margin added to every estimate before comparing to the budget.
    safety_margin_tokens: int = Field(64, ge=0)


class LimitsConfig(_Base):
    """Input-size policy, applied uniformly by the engine.

    The value of the budget itself is configuration -- the engine has no
    dataset-specific knowledge of who needs what.
    """

    #: Maximum *input* tokens per sample (output budget is separate).
    input_token_budget: int = Field(16000, ge=1)
    #: What to do when a rendered prompt exceeds the budget:
    #: ``resample`` asks the adapter for a replacement item from the same split,
    #: ``skip`` records the sample as skipped, ``fail`` aborts the task.
    on_oversize: Literal["resample", "skip", "fail"] = "resample"
    #: Ceiling on replacement draws before giving up on a task.
    max_resample_attempts: int = Field(200, ge=0)
    #: If more than this fraction of a dataset's sample is oversize, the task is
    #: reported as unusable rather than resampled indefinitely.
    oversize_abort_fraction: float = Field(0.5, gt=0, le=1)


class LoggingConfig(_Base):
    level: str = "INFO"
    #: Machine-readable JSONL event log next to the human-readable log.
    json_log: bool = True
    #: ``auto`` shows a rich progress display only on a TTY.
    progress: Literal["auto", "on", "off"] = "auto"
    #: Truncate long strings in the human log at this many characters.
    log_text_clip: int = Field(400, ge=0)


class JudgeConfig(_Base):
    """Optional LLM-as-judge stage available to adapters that need it.

    Off by default.  The judge prompt is a normal versioned template, so judged
    metrics are as configurable as the main prompts.
    """

    enabled: bool = False
    #: Model id (from the run's model list, or its own endpoint config file).
    model: str | None = None
    template: str = "judge_binary_v1"
    max_tokens: int = Field(512, ge=1)
    temperature: float = Field(0.0, ge=0)
    group_size: int = Field(8, ge=1)
    #: Cache judge verdicts on disk so re-scoring does not re-spend tokens.
    cache: bool = True


class SyncConfig(_Base):
    """Incremental off-box backup of a run's artifacts (see ``core/sync.py``).

    Runs in a background thread that shells out to ``rclone``, so it cannot slow
    down or break inference.  ``remote_path`` is anything rclone understands --
    ``gdrive:AbductionBench``, ``s3:bucket/prefix``, or a plain local path.
    """

    enabled: bool = False
    #: rclone destination. ``<remote>:<path>`` for a configured remote, or a
    #: local/mounted path (which is also how this is tested without credentials).
    remote_path: str = ""
    #: Upload into ``<remote_path>/<run-id>/`` so runs never overwrite each other.
    per_run_subdir: bool = True
    #: Seconds between incremental uploads.  60s keeps the remote within a
    #: minute of the local state while adding negligible load.
    interval_s: float = Field(60.0, gt=0)
    rclone_binary: str = "rclone"
    transfers: int = Field(4, ge=1)
    checkers: int = Field(8, ge=1)
    timeout_s: int = Field(300, ge=10)
    #: e.g. "8M" to cap upload bandwidth; empty means unlimited.
    bandwidth_limit: str = ""
    #: Glob patterns to leave out.  Raw per-batch payloads are the bulky part of
    #: a run, so a slow link can drop them while still backing up every record,
    #: metric and log.
    exclude: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    #: If the destination is unusable at startup (e.g. credentials not set up
    #: yet), re-check this often and begin uploading once it works.  0 disables
    #: the re-check, leaving the run permanently un-backed-up.
    preflight_retry_s: float = Field(300.0, ge=0)


class ReportingConfig(_Base):
    """Excel/CSV/Markdown outputs."""

    excel_filename: str = "abductionbench_results.xlsx"
    #: Also write the full per-sample grid into the workbook (can be large).
    include_sample_sheets: bool = True
    #: Cap rows per sample sheet to keep the workbook openable.
    max_sample_rows_per_sheet: int = Field(5000, ge=1)
    #: Long-form CSV of every metric value.
    write_csv: bool = True
    #: Per-task markdown run documentation.
    write_run_documentation: bool = True
    #: Clip response text stored in the workbook.
    response_clip_chars: int = Field(2000, ge=0)


class EngineConfig(_Base):
    """Everything that is not model-, dataset- or prompt-specific."""

    output_root: Path = Path("runs")
    data_root: Path = Path("data")
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #


class BatchEndpointConfig(_Base):
    """Where a model's *native batch* endpoint lives.

    This is deliberately a separate base URL from the gateway: vLLM's batch
    route is not part of the OpenAI API, and in this deployment it is reachable
    either on the model's own tunnel (``https://<model-tunnel>/v1/chat/completions/batch``)
    or on a gateway pass-through path (``https://<gateway>/<model-dir>/v1/chat/completions/batch``).
    Both topologies are expressible here without touching code.
    """

    enabled: bool = True
    #: Defaults to the model's chat base URL when omitted.
    base_url: str | None = None
    path: str = "/v1/chat/completions/batch"
    #: How many samples of the same dataset go into one batch call *for this
    #: model*.  This is the per-model "batch group size".
    group_size: int = Field(8, ge=1)
    #: Separate key if the batch route authenticates differently (a model's own
    #: tunnel accepts only its own vLLM key, while the gateway accepts virtual
    #: per-user keys).
    api_key: str | None = None
    #: Extra headers for the batch route.
    headers: dict[str, str] = Field(default_factory=dict)


class DiscoveryConfig(_Base):
    """Commands that print the *current* base URLs.

    Cloudflare quick-tunnel URLs rotate whenever the tunnel restarts.  When a
    connection error happens, the engine re-runs these commands and picks up the
    new URL instead of failing the whole run.
    """

    base_url_command: str | None = None
    batch_base_url_command: str | None = None


class EndpointConfig(_Base):
    """OpenAI-compatible endpoint used for discovery and non-batched calls."""

    base_url: str
    api_key: str | None = None
    models_path: str = "/v1/models"
    chat_path: str = "/v1/chat/completions"
    headers: dict[str, str] = Field(default_factory=dict)
    #: Verify TLS.  Quick tunnels present valid certs, so keep this on.
    verify_tls: bool = True
    batch: BatchEndpointConfig = Field(default_factory=BatchEndpointConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.rstrip("/")

    def resolved_batch_base_url(self) -> str:
        return (self.batch.base_url or self.base_url).rstrip("/")

    def resolved_batch_api_key(self) -> str | None:
        return self.batch.api_key or self.api_key


class ModelSamplingConfig(_Base):
    """Default decoding parameters for a model (per-sample overrides allowed)."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    max_tokens_default: int = Field(512, ge=1)
    #: Upper bound applied to adapter-requested budgets.
    max_tokens_cap: int = Field(4096, ge=1)
    #: Lower bound; reasoning models need headroom or ``content`` comes back
    #: ``null`` because the hidden chain-of-thought consumed the budget.
    max_tokens_floor: int = Field(64, ge=1)
    stop: list[str] = Field(default_factory=list)
    #: Vendor-specific extras passed straight through to the request body.
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelLimitsConfig(_Base):
    max_parallel_batches: int = Field(2, ge=1)
    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    #: Model context window; used together with the requested output budget to
    #: reject prompts that cannot possibly fit.
    context_window: int | None = None


class ModelConfig(_Base):
    """One model under evaluation."""

    id: str
    #: The exact string the server expects in the request's ``model`` field.
    model_name: str
    description: str = ""
    #: Marks reasoning models whose ``content`` may be ``null`` while their
    #: chain-of-thought lands in ``reasoning``; affects only telemetry and the
    #: empty-response warning, never scoring.
    reasoning_model: bool = False
    endpoint: EndpointConfig
    sampling: ModelSamplingConfig = Field(default_factory=ModelSamplingConfig)
    limits: ModelLimitsConfig = Field(default_factory=ModelLimitsConfig)
    #: Free-form notes surfaced in the run documentation.
    notes: dict[str, Any] = Field(default_factory=dict)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.id).strip("-")


# --------------------------------------------------------------------------- #
# Prompt configuration
# --------------------------------------------------------------------------- #


class PromptConfig(_Base):
    """Where prompt templates come from and which one each task kind uses.

    ``bindings`` maps a ``task_kind`` (declared by the adapter, e.g.
    ``"generation"``) to a template id.  ``dataset_overrides`` lets one dataset
    use a different template without touching the adapter, and
    ``template_variants`` lets a single run evaluate several templates against
    the same data (each becomes its own task and its own row in the grid).
    """

    template_dirs: list[Path] = Field(default_factory=lambda: [Path("configs/prompts")])
    bindings: dict[str, str] = Field(default_factory=dict)
    dataset_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    #: Named alternative binding sets, all evaluated in the same run.
    #: ``{"v2": {"generation": "gen_cot_v2"}}`` -> a second task per dataset.
    template_variants: dict[str, dict[str, str]] = Field(default_factory=dict)
    #: Prepend/append blocks available to templates as ``{{ preamble }}`` etc.
    shared_blocks: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_bindings(self) -> PromptConfig:
        if not self.bindings:
            raise ConfigError(
                "prompts.bindings must map at least one task_kind to a template id"
            )
        return self


# --------------------------------------------------------------------------- #
# Dataset + run configuration
# --------------------------------------------------------------------------- #


class DatasetConfig(_Base):
    """A dataset entry in a run.

    The engine knows only these generic fields.  Anything dataset-specific goes
    into ``options`` and is handed to the child adapter untouched.
    """

    id: str
    #: ``"package.module:ClassName"`` -- resolved at runtime, so the core never
    #: imports adapter code directly.
    impl: str
    enabled: bool = True
    sample_size: int = Field(300, ge=1)
    #: Falls back to the run seed when unset.
    seed: int | None = None
    #: Per-dataset overrides of the engine's input-size policy.
    input_token_budget: int | None = None
    #: Per-dataset override of a model's batch group size, when a dataset's
    #: items are unusually long.
    batch_group_size: int | None = None
    #: Per-dataset template binding overrides (``task_kind -> template_id``).
    prompt_bindings: dict[str, str] = Field(default_factory=dict)
    #: Opaque adapter options (data paths, subtask selection, metric switches).
    options: dict[str, Any] = Field(default_factory=dict)
    #: Metric name used for the headline table; falls back to the adapter's
    #: declared primary metric.
    primary_metric: str | None = None
    notes: str = ""


class RunConfig(_Base):
    """A complete, self-contained description of one evaluation run."""

    name: str = "run"
    description: str = ""
    #: Global determinism seed; datasets inherit it unless they override.
    seed: int = 20260903
    engine: EngineConfig = Field(default_factory=EngineConfig)
    prompts: PromptConfig
    models: list[ModelConfig] = Field(default_factory=list)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    #: Provenance: which files this config was assembled from.
    source_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> RunConfig:
        if not self.models:
            raise ConfigError("run config must list at least one model")
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ConfigError(f"duplicate model ids in run config: {ids}")
        ds_ids = [d.id for d in self.datasets]
        if len(ds_ids) != len(set(ds_ids)):
            raise ConfigError(f"duplicate dataset ids in run config: {ds_ids}")
        if self.engine.judge.enabled and not self.engine.judge.model:
            raise ConfigError("engine.judge.enabled requires engine.judge.model")
        return self

    def enabled_datasets(self) -> list[DatasetConfig]:
        return [d for d in self.datasets if d.enabled]

    def model_by_id(self, model_id: str) -> ModelConfig:
        for model in self.models:
            if model.id == model_id:
                return model
        raise ConfigError(f"unknown model id {model_id!r}")


# --------------------------------------------------------------------------- #
# Top-level loader
# --------------------------------------------------------------------------- #


def _load_referenced_list(
    entries: list[Any],
    *,
    anchor: Path,
    key: str,
) -> list[dict[str, Any]]:
    """Normalize a list that may contain inline mappings and/or file references.

    Accepted element forms::

        - configs/models/foo.yaml                    # path string
        - {file: configs/models/foo.yaml}            # explicit reference
        - {file: configs/models/foo.yaml, group_size: 16}  # reference + overrides
        - {id: foo, model_name: ...}                 # fully inline

    A referenced file may wrap its content under a top-level ``key`` (e.g.
    ``model:`` / ``dataset:``) or provide the mapping directly.
    """
    out: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, (str, Path)):
            entry = {"file": str(entry)}
        if not isinstance(entry, dict):
            raise ConfigError(f"invalid {key} entry: {entry!r}")
        entry = dict(entry)
        ref = entry.pop("file", None)
        if ref is None:
            out.append(entry)
            continue
        loaded = load_layered(_resolve_relative(ref, anchor))  # noqa: E501 - per-entry anchor
        loaded.pop("_source_files", None)
        body = loaded.get(key, loaded)
        if not isinstance(body, dict):
            raise ConfigError(f"{ref}: expected a mapping under {key!r}")
        out.append(deep_merge(body, entry))
    return out


def load_run_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
    dataset_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
) -> RunConfig:
    """Load, compose, interpolate and validate a run configuration.

    Parameters
    ----------
    path:
        Run YAML file.  May ``extends`` engine defaults and reference model /
        dataset config files.
    overrides:
        ``dotted.path=value`` strings applied after merging (CLI ``--set``).
    dataset_filter, model_filter:
        Restrict the run to these ids (CLI ``--datasets`` / ``--models``).
    """
    anchor = Path(path).resolve()
    raw = load_layered(anchor)
    sources = raw.pop("_source_files", [])

    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(f"--set expects dotted.path=value, got {override!r}")
        dotted, _, value = override.partition("=")
        _apply_dotted(raw, dotted.strip(), value)

    # 'run:' wrapper is optional.
    body = raw.get("run", raw) if isinstance(raw.get("run"), dict) else raw
    body = dict(body)
    for extra_key in ("engine", "prompts"):
        if extra_key not in body and extra_key in raw:
            body[extra_key] = raw[extra_key]

    # `datasets_glob` expands to dataset config files, so a run config does not
    # have to be edited every time a dataset is added.  Files whose name starts
    # with "_" are skipped (templates), and the order is sorted for determinism.
    globs = body.pop("datasets_glob", []) or []
    if isinstance(globs, (str, Path)):
        globs = [globs]
    if globs:
        discovered: list[str] = []
        for pattern in globs:
            pattern_path = Path(str(pattern))
            # The pattern may have been declared by any file in the extends
            # chain, so try each of their directories (child first).
            roots = (
                [Path("/")]
                if pattern_path.is_absolute()
                else [anchor.parent, *(Path(src).parent for src in reversed(sources))]
            )
            matches: list[Path] = []
            for root in roots:
                matches = sorted(root.glob(str(pattern_path)))
                if matches:
                    break
            discovered.extend(
                str(match) for match in matches if not match.name.startswith("_")
            )
        if not discovered:
            raise ConfigError(
                f"datasets_glob matched no files: {globs} (searched relative to "
                f"{[str(anchor.parent), *[str(Path(s).parent) for s in reversed(sources)]]})"
            )
        body["datasets"] = [*(body.get("datasets") or []), *discovered]

    body["models"] = _load_referenced_list(body.get("models", []), anchor=anchor, key="model")
    body["datasets"] = _load_referenced_list(
        body.get("datasets", []), anchor=anchor, key="dataset"
    )

    # `dataset_defaults` supplies values a dataset's own config may override;
    # `dataset_force` overrides the dataset's config (useful for smoke runs, e.g.
    # forcing sample_size: 12 across every dataset in one line).
    dataset_defaults = body.pop("dataset_defaults", {}) or {}
    if dataset_defaults:
        body["datasets"] = [deep_merge(dataset_defaults, d) for d in body["datasets"]]
    dataset_force = body.pop("dataset_force", {}) or {}
    if dataset_force:
        body["datasets"] = [deep_merge(d, dataset_force) for d in body["datasets"]]

    if model_filter:
        wanted = set(model_filter)
        body["models"] = [m for m in body["models"] if m.get("id") in wanted]
        missing = wanted - {m.get("id") for m in body["models"]}
        if missing:
            raise ConfigError(f"--models refers to unknown model ids: {sorted(missing)}")
    if dataset_filter:
        wanted = set(dataset_filter)
        body["datasets"] = [d for d in body["datasets"] if d.get("id") in wanted]
        missing = wanted - {d.get("id") for d in body["datasets"]}
        if missing:
            raise ConfigError(f"--datasets refers to unknown dataset ids: {sorted(missing)}")

    # Prompt template directories are resolved relative to the config file that
    # names them (falling back to the working directory), so a run config can be
    # invoked from anywhere.
    prompts_block = body.get("prompts")
    if isinstance(prompts_block, dict) and prompts_block.get("template_dirs"):
        prompts_block["template_dirs"] = [
            str(_resolve_against_sources(entry, anchor, sources))
            for entry in prompts_block["template_dirs"]
        ]

    body = interpolate(body)
    body["source_files"] = sources
    try:
        return RunConfig.model_validate(body)
    except Exception as exc:  # pydantic ValidationError or ConfigError
        raise ConfigError(f"invalid run configuration ({anchor}): {exc}") from exc


def dump_resolved(config: RunConfig, path: str | Path, *, redact: bool = True) -> None:
    """Write the fully-resolved config next to a run's outputs (provenance).

    API keys are redacted by default so run directories are safe to share.
    """
    data = config.model_dump(mode="json")
    if redact:
        for model in data.get("models", []):
            endpoint = model.get("endpoint", {})
            for field_name in ("api_key",):
                if endpoint.get(field_name):
                    endpoint[field_name] = "***redacted***"
            batch = endpoint.get("batch", {})
            if batch.get("api_key"):
                batch["api_key"] = "***redacted***"
            for header_bag in (endpoint.get("headers", {}), batch.get("headers", {})):
                for key in list(header_bag):
                    if "auth" in key.lower() or "key" in key.lower():
                        header_bag[key] = "***redacted***"
    Path(path).write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
