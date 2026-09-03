"""Generic, task-shaped metric primitives.

These are building blocks, not dataset knowledge: string normalization, exact
match, token overlap, ROUGE-L / BLEU-style overlap, ranking metrics, set
metrics, and answer extraction that honours a prompt template's declared
``output_contract``.  Child adapters compose them into the metric their dataset
actually defines and remain free to implement anything bespoke.

Everything here is pure Python (no scipy/nltk), deterministic, and safe on
degenerate input (empty strings, ``None``, mismatched lengths) -- a scorer must
never raise on a weird model response.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "normalize_text",
    "normalize_answer",
    "exact_match",
    "contains_match",
    "token_f1",
    "token_precision_recall_f1",
    "rouge_l",
    "bleu",
    "jaccard",
    "set_prf",
    "hits_at_k",
    "mean_reciprocal_rank",
    "spearman",
    "kendall_tau",
    "numeric_match",
    "extract_choice_label",
    "extract_answer_span",
    "extract_first_number",
    "mean",
    "std",
    "confidence_interval_95",
    "summarize_numeric",
    "brier_score",
]

_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
_ARTICLES = {"a", "an", "the"}


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #


def normalize_text(text: Any) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(text: Any, *, drop_articles: bool = True) -> str:
    """SQuAD-style answer normalization (also drops leading articles)."""
    normalized = normalize_text(text)
    if drop_articles and normalized:
        tokens = [t for t in normalized.split() if t not in _ARTICLES]
        normalized = " ".join(tokens)
    return normalized


def _tokens(text: Any) -> list[str]:
    return normalize_answer(text).split()


# --------------------------------------------------------------------------- #
# string-level matching
# --------------------------------------------------------------------------- #


def exact_match(prediction: Any, reference: Any) -> float:
    """1.0 when normalized strings are identical."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def any_exact_match(prediction: Any, references: Iterable[Any]) -> float:
    """1.0 when the prediction exactly matches any acceptable reference."""
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(ref) for ref in references))


def contains_match(prediction: Any, reference: Any) -> float:
    """1.0 when the normalized reference appears inside the normalized prediction.

    Useful for free-form generation where the model wraps the right answer in
    prose (e.g. "the most likely diagnosis is <gold>").
    """
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not ref:
        return 0.0
    return float(ref in pred)


def token_precision_recall_f1(prediction: Any, reference: Any) -> tuple[float, float, float]:
    """Bag-of-tokens precision/recall/F1 (SQuAD F1)."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        # Both empty counts as a match; one empty counts as a miss.
        value = float(pred_tokens == ref_tokens)
        return value, value, value
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def token_f1(prediction: Any, reference: Any) -> float:
    return token_precision_recall_f1(prediction, reference)[2]


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: Any, reference: Any, *, beta: float = 1.2) -> dict[str, float]:
    """ROUGE-L (LCS-based) precision/recall/F-measure.

    Implemented directly to avoid a heavyweight dependency; matches the standard
    definition with the usual ``beta`` weighting recall over precision.
    """
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    beta_sq = beta * beta
    f_measure = ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)
    return {"precision": precision, "recall": recall, "f": f_measure}


def bleu(prediction: Any, reference: Any, *, max_n: int = 4, smooth: bool = True) -> float:
    """Sentence-level BLEU with add-one smoothing (single reference)."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    log_sum = 0.0
    usable = 0
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        if not pred_ngrams:
            continue
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        if smooth:
            precision = (overlap + 1) / (total + 1)
        elif overlap == 0:
            return 0.0
        else:
            precision = overlap / total
        log_sum += math.log(precision)
        usable += 1
    if not usable:
        return 0.0
    geometric_mean = math.exp(log_sum / usable)
    brevity = math.exp(min(0.0, 1 - len(ref_tokens) / len(pred_tokens)))
    return geometric_mean * brevity


# --------------------------------------------------------------------------- #
# set and ranking metrics
# --------------------------------------------------------------------------- #


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def set_prf(predicted: Iterable[Any], gold: Iterable[Any]) -> dict[str, float]:
    """Precision/recall/F1 over sets (e.g. sets of abduced facts or rules)."""
    pred_set, gold_set = set(predicted), set(gold)
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def hits_at_k(ranked: Sequence[Any], gold: Any, k: int) -> float:
    """1.0 when ``gold`` appears in the top ``k`` of a ranked list."""
    return float(gold in list(ranked)[:k])


def mean_reciprocal_rank(ranked: Sequence[Any], gold: Any) -> float:
    for position, item in enumerate(ranked, start=1):
        if item == gold:
            return 1.0 / position
    return 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (0.0 for degenerate input)."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    rank_a, rank_b = _ranks(list(a)), _ranks(list(b))
    mean_a, mean_b = mean(rank_a), mean(rank_b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(rank_a, rank_b, strict=True))
    den = math.sqrt(sum((x - mean_a) ** 2 for x in rank_a)) * math.sqrt(
        sum((y - mean_b) ** 2 for y in rank_b)
    )
    return num / den if den else 0.0


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-a; 0.0 for degenerate input."""
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = (a[i] - a[j]) > 0
            sign_b = (b[i] - b[j]) > 0
            if a[i] == a[j] or b[i] == b[j]:
                continue
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def numeric_match(prediction: Any, reference: Any, *, rel_tol: float = 0.05) -> float:
    """1.0 when two numbers agree within a relative tolerance."""
    try:
        pred = float(prediction)
        ref = float(reference)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(pred) or math.isnan(ref):
        return 0.0
    if ref == 0:
        return float(abs(pred) <= rel_tol)
    return float(abs(pred - ref) / abs(ref) <= rel_tol)


def brier_score(probability: float, outcome: float) -> float:
    """Squared error of a probabilistic judgement (lower is better)."""
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return 1.0
    p = min(1.0, max(0.0, p))
    return (p - float(outcome)) ** 2


# --------------------------------------------------------------------------- #
# answer extraction (template-contract aware)
# --------------------------------------------------------------------------- #


def extract_answer_span(text: str | None, contract: dict[str, Any] | None = None) -> str:
    """Pull the answer out of a response, guided by the template's contract.

    Recognized ``output_contract`` keys (all optional):

    ``answer_regex``
        Regex with either one group or a named ``answer`` group; first match wins.
    ``answer_prefix``
        Literal marker (e.g. ``"Final answer:"``); text after its last
        occurrence is returned.
    ``strip_markdown``
        Remove surrounding ``**`` / backticks (default ``True``).

    With no contract -- or no match -- the whole trimmed response is returned, so
    a scorer always gets *something* to compare.
    """
    if not text:
        return ""
    contract = contract or {}
    body = text.strip()

    regex = contract.get("answer_regex")
    if regex:
        match = re.search(regex, body, flags=re.IGNORECASE | re.DOTALL)
        if match:
            groups = match.groupdict()
            candidate = groups.get("answer") if "answer" in groups else None
            if candidate is None:
                candidate = match.group(1) if match.groups() else match.group(0)
            body = (candidate or "").strip()

    prefix = contract.get("answer_prefix")
    if prefix:
        lowered = body.lower()
        marker = str(prefix).lower()
        position = lowered.rfind(marker)
        if position >= 0:
            body = body[position + len(marker) :].strip()
            body = body.lstrip(":").strip()

    if contract.get("strip_markdown", True):
        body = body.strip().strip("`").strip()
        body = re.sub(r"^\*+|\*+$", "", body).strip()
    return body


def extract_choice_label(
    text: str | None,
    labels: Sequence[str],
    contract: dict[str, Any] | None = None,
) -> str | None:
    """Find which multiple-choice label a response selected.

    Strategy, in order:

    1. apply the template's ``output_contract`` (regex/prefix) and test whether
       what remains starts with a label;
    2. look for an explicit "answer is X" style statement;
    3. look for a standalone label token (``A``, ``(B)``, ``C.``) scanning from
       the end of the response, since models usually conclude with the answer.

    Returns ``None`` when nothing matches, so the caller can record
    ``parse_ok=False`` rather than scoring a guess.
    """
    if not text:
        return None
    label_list = [str(label) for label in labels]
    if not label_list:
        return None
    escaped = "|".join(re.escape(label) for label in sorted(label_list, key=len, reverse=True))

    span = extract_answer_span(text, contract)
    for candidate in (span, text):
        if not candidate:
            continue
        head = re.match(rf"^\(?\[?({escaped})\)?\]?\s*[).:,-]?\s*", candidate.strip(),
                        flags=re.IGNORECASE)
        if head:
            return _canonical_label(head.group(1), label_list)

    statement = re.findall(
        rf"(?:answer|choice|option|label|select(?:ed)?)\D{{0,20}}?\b({escaped})\b",
        text,
        flags=re.IGNORECASE,
    )
    if statement:
        return _canonical_label(statement[-1], label_list)

    standalone = re.findall(rf"(?<![A-Za-z0-9])\(?({escaped})\)?(?![A-Za-z0-9])", text)
    if standalone:
        return _canonical_label(standalone[-1], label_list)
    return None


def _canonical_label(found: str, labels: Sequence[str]) -> str | None:
    for label in labels:
        if label.lower() == found.lower():
            return label
    return None


def extract_first_number(text: str | None) -> float | None:
    """First number in a response (handles ``1,234.5``, ``-3``, ``2e3``)."""
    if not text:
        return None
    match = re.search(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values if v is not None and not _is_nan(v)]
    return sum(items) / len(items) if items else 0.0


def std(values: Iterable[float]) -> float:
    items = [float(v) for v in values if v is not None and not _is_nan(v)]
    if len(items) < 2:
        return 0.0
    average = sum(items) / len(items)
    variance = sum((v - average) ** 2 for v in items) / (len(items) - 1)
    return math.sqrt(variance)


def confidence_interval_95(values: Iterable[float]) -> float:
    """Half-width of the normal-approximation 95% CI of the mean."""
    items = [float(v) for v in values if v is not None and not _is_nan(v)]
    if len(items) < 2:
        return 0.0
    return 1.96 * std(items) / math.sqrt(len(items))


def summarize_numeric(values: Iterable[float]) -> dict[str, float]:
    items = [float(v) for v in values if v is not None and not _is_nan(v)]
    if not items:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0.0, "ci95": 0.0}
    return {
        "mean": mean(items),
        "std": std(items),
        "min": min(items),
        "max": max(items),
        "n": float(len(items)),
        "ci95": confidence_interval_95(items),
    }


def macro_average(per_group: dict[str, Iterable[float]]) -> float:
    """Unweighted mean of per-group means (macro average)."""
    group_means = [mean(values) for values in per_group.values()]
    return mean(group_means)


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def aggregate_mean_metrics(
    score_metrics: Sequence[dict[str, float]],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Mean of every metric key present in a sequence of per-sample metrics.

    Keys absent from a given sample are skipped for that sample (not treated as
    zero), so optional sub-metrics do not silently drag an average down.
    """
    buckets: dict[str, list[float]] = {}
    for metrics in score_metrics:
        for key, value in (metrics or {}).items():
            if _is_nan(value):
                continue
            buckets.setdefault(key, []).append(float(value))
    return {f"{prefix}{key}": mean(values) for key, values in sorted(buckets.items())}
