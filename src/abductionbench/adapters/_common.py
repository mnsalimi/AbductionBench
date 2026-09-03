"""Shared plumbing for child adapters.

Nothing here is specific to one benchmark: fetching sources (git, HTTP, Hugging
Face), reading the file formats these datasets ship in, choosing a split, and a
few text helpers.  Adapters import from here instead of re-implementing
downloads and parsing.

All fetchers are **idempotent and cache-aware**: they no-op when the data is
already materialized, and they raise
:class:`~abductionbench.core.adapter.SkippedDataset` (not a bare exception) when
a source cannot be obtained, so one broken dataset is reported and skipped
rather than stopping a run.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..core.adapter import SkippedDataset

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_git_repo",
    "ensure_download",
    "ensure_hf_snapshot",
    "load_hf_dataset",
    "extract_archive",
    "read_json",
    "read_jsonl",
    "read_csv_rows",
    "read_parquet_rows",
    "read_gzip_jsonl",
    "read_lines",
    "read_text",
    "find_files",
    "first_existing",
    "pick_split_file",
    "SPLIT_PREFERENCE",
    "normalize_whitespace",
    "clip_words",
    "as_list",
    "letter_labels",
    "stable_id",
]

#: Split preference mandated for the suite: evaluate the test split when one
#: exists, else validation, else train.
SPLIT_PREFERENCE = ("test", "dev", "validation", "valid", "val", "train")

_GIT_TIMEOUT = 900
_HTTP_TIMEOUT = 300


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #


def ensure_git_repo(
    url: str,
    dest: Path,
    *,
    offline: bool = False,
    depth: int = 1,
    branch: str | None = None,
) -> Path:
    """Shallow-clone ``url`` into ``dest`` (no-op when already present).

    Falls back to GitHub's codeload zip when ``git clone`` fails (some repos in
    the suite are large or have broken LFS pointers).
    """
    dest = Path(dest)
    if (dest / ".git").exists() or (dest.exists() and any(dest.iterdir())):
        return dest
    if offline:
        raise SkippedDataset(f"offline mode: {url} is not materialized at {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    command = ["git", "clone", "--quiet", "--depth", str(depth)]
    if branch:
        command += ["--branch", branch]
    command += [url, str(dest)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=_GIT_TIMEOUT)
        return dest
    except (subprocess.SubprocessError, OSError) as exc:
        stderr = getattr(exc, "stderr", b")") or b""
        logger.warning("git clone failed for %s (%s); trying codeload zip", url, stderr[:200])
        shutil.rmtree(dest, ignore_errors=True)

    zip_url = _codeload_url(url, branch)
    if zip_url is None:
        raise SkippedDataset(f"cannot clone {url} and it is not a GitHub URL")
    try:
        archive = ensure_download(zip_url, dest.parent / f"{dest.name}.zip", offline=offline)
    except SkippedDataset:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SkippedDataset(f"cannot fetch {url}: {exc}") from exc
    extracted = extract_archive(archive, dest.parent / f"{dest.name}__unzip")
    inner = [p for p in extracted.iterdir() if p.is_dir()]
    source = inner[0] if len(inner) == 1 else extracted
    shutil.move(str(source), str(dest))
    shutil.rmtree(extracted, ignore_errors=True)
    return dest


def _codeload_url(url: str, branch: str | None) -> str | None:
    match = re.match(r"https?://github\.com/([^/]+)/([^/.]+)", url)
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    for candidate in ([branch] if branch else []) + ["main", "master"]:
        return f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{candidate}"
    return None


def ensure_download(url: str, dest: Path, *, offline: bool = False) -> Path:
    """Download ``url`` to ``dest`` once; returns the local path."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        raise SkippedDataset(f"offline mode: {url} is not cached at {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=_HTTP_TIMEOUT) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise SkippedDataset(f"cannot download {url}: {exc}") from exc
    tmp.replace(dest)
    return dest


def ensure_hf_snapshot(
    repo_id: str,
    dest: Path,
    *,
    offline: bool = False,
    repo_type: str = "dataset",
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    """Download a Hugging Face repo snapshot into ``dest``."""
    dest = Path(dest)
    marker = dest / ".snapshot_complete"
    if marker.exists():
        return dest
    if offline:
        raise SkippedDataset(f"offline mode: HF repo {repo_id} is not cached at {dest}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(dest),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as exc:  # noqa: BLE001
        raise SkippedDataset(f"cannot download HF {repo_type} {repo_id}: {exc}") from exc
    marker.write_text("ok", encoding="utf-8")
    return dest


def load_hf_dataset(repo_id: str, *, split: str | None = None, **kwargs: Any) -> Any:
    """``datasets.load_dataset`` wrapped so failures become skips."""
    try:
        from datasets import load_dataset

        return load_dataset(repo_id, split=split, token=os.environ.get("HF_TOKEN") or None, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise SkippedDataset(f"cannot load HF dataset {repo_id}: {exc}") from exc


def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract a zip/tar archive into ``dest`` (idempotent)."""
    archive, dest = Path(archive), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(dest)  # noqa: S202 - trusted benchmark archives
        else:
            raise SkippedDataset(f"{archive} is neither a zip nor a tar archive")
    except SkippedDataset:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SkippedDataset(f"cannot extract {archive}: {exc}") from exc
    return dest


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> Any:
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise SkippedDataset(f"{path} is not valid JSON: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank and unparseable lines."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(read_text(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("%s:%d is not valid JSON; skipped", path, number)
    return rows


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a parquet file into plain dicts (numpy arrays become lists)."""
    import pandas as pd

    frame = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        rows.append({key: _plain(value) for key, value in record.items()})
    return rows


def _plain(value: Any) -> Any:
    """Convert numpy/pandas scalars and arrays into plain Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a gzip-compressed JSONL file."""
    import gzip

    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_lines(path: Path) -> list[str]:
    """Read a text file as a list of stripped, non-empty lines."""
    return [line.strip() for line in read_text(path).splitlines() if line.strip()]


def read_csv_rows(path: Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    """Read a CSV/TSV into dicts, sniffing the delimiter when not given."""
    text = read_text(path)
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]


def find_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    """All files under ``root`` matching any glob pattern, sorted."""
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(root).rglob(pattern))
    return sorted({p for p in found if p.is_file()})


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def pick_split_file(
    candidates: Sequence[Path],
    *,
    preference: Sequence[str] = SPLIT_PREFERENCE,
) -> tuple[Path, str] | None:
    """Choose one file by split preference (test → validation → train).

    Matching is on the file *name*, so ``test.jsonl``, ``dev-00000.parquet`` and
    ``data_train.json`` all resolve.  Returns ``(path, split_name)``.
    """
    for split in preference:
        for path in candidates:
            if re.search(rf"(^|[^a-z]){split}([^a-z]|$)", path.name.lower()):
                return path, split
    return None


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", str(text))).strip()


def clip_words(text: str, limit: int) -> str:
    """Trim to ``limit`` words, marking the cut (used only where documented)."""
    words = str(text).split()
    if len(words) <= limit:
        return str(text)
    return " ".join(words[:limit]) + f" … [truncated, {len(words) - limit} words omitted]"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def letter_labels(count: int, start: str = "A") -> list[str]:
    first = ord(start)
    return [chr(first + index) for index in range(count)]


def stable_id(*parts: Any) -> str:
    """Readable, filesystem-safe id from arbitrary parts."""
    joined = "-".join(str(part) for part in parts if part is not None and str(part) != "")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", joined)[:120]


def iter_dicts(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield dicts from JSON that may be a list, a dict of lists, or nested."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
