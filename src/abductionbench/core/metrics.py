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
    "extract_choice_labels",
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
    """Kendall's tau-b; 0.0 for degenerate input.

    tau-b rather than tau-a because ties are normal in the data this would be
    used on -- judge scores and rank positions repeat constantly -- and tau-a's
    denominator (every pair, tied or not) caps the coefficient below 1 whenever
    they do, so a perfect but tied ordering cannot score 1.
    """
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
    # tau-b's denominator: sqrt((C+D+ties_in_a) * (C+D+ties_in_b)). Dividing by
    # C+D alone -- which is what this did while calling itself tau-a -- is
    # Goodman-Kruskal gamma, which ignores ties entirely and reads systematically
    # higher than either tau.
    ties_a = sum(
        1 for i in range(n) for j in range(i + 1, n) if a[i] == a[j] and b[i] != b[j]
    )
    ties_b = sum(
        1 for i in range(n) for j in range(i + 1, n) if b[i] == b[j] and a[i] != a[j]
    )
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return (concordant - discordant) / denominator if denominator else 0.0


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


#: What a model writes when a multi-select question has no applicable option.
#: Also what :meth:`DatasetAdapter._reduce_bov` writes when every per-hypothesis
#: question was answered no, so an all-no BOV item is an empty *answer* rather
#: than an unparseable one.
EMPTY_SELECTION = {"none", "nothing", "no option", "no options", "n/a", "-"}


def extract_choice_labels(
    text: str | None,
    labels: Sequence[str],
    contract: dict[str, Any] | None = None,
) -> list[str] | None:
    """Every label a multi-select response chose, in the order it named them.

    The single-choice sibling, :func:`extract_choice_label`, falls back to
    scanning the whole response for a label token.  That fallback is wrong here:
    a chain-of-thought answer mentions half the options while thinking, so
    harvesting labels from the reasoning would credit the model for choices it
    talked itself out of.  A *set* is therefore read only from the answer line
    (the contract's span, or the last non-empty line when there is none).

    Returns ``[]`` for an explicit empty selection ("none"), and ``None`` when
    nothing could be read at all -- which the caller records as a parse failure
    rather than as "the model selected nothing".
    """
    if not text:
        return None
    label_list = [str(label) for label in labels]
    if not label_list:
        return None
    escaped = "|".join(re.escape(label) for label in sorted(label_list, key=len, reverse=True))

    span = extract_answer_span(text, contract)
    if not span:
        lines = [line for line in text.splitlines() if line.strip()]
        span = lines[-1] if lines else ""
    span = span.strip()
    if not span:
        return None

    found = re.findall(
        rf"(?<![A-Za-z0-9])\(?({escaped})\)?(?![A-Za-z0-9])", span, flags=re.IGNORECASE
    )
    out: list[str] = []
    for raw in found:
        canonical = _canonical_label(raw, label_list)
        if canonical and canonical not in out:
            out.append(canonical)
    if out:
        return out
    stripped = re.sub(r"[^\w\s/-]", " ", span).strip().lower()
    return [] if stripped in EMPTY_SELECTION else None


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


#: What a per-sample metric that has no value is written as, where a blank
#: would read as "not computed" -- reasoning_anchoring_point when the model
#: never considered its answer, or the index was unusable. It is not a number,
#: so every mean skips it (see _is_nan), and it is carried through a resume
#: unchanged by stored_metrics().
MISSING_METRIC = "None"


def stored_metrics(raw: dict[str, Any] | None) -> dict[str, Any]:
    """A record's metrics as numbers, keeping MISSING_METRIC and dropping the rest."""
    out: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if value == MISSING_METRIC:
            out[key] = MISSING_METRIC
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
    return out


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


#: Metrics that are read together, so they must be averaged over the same rows.
#:
#: A mean is only comparable with another mean when both were taken over the
#: same samples. These columns are not read one at a time -- a reader divides
#: one by another, or checks that a ratio column agrees with the two it came
#: from -- and each was being averaged over whatever subset happened to have
#: it. Measured on one abd task: `observations_total` came from 148 samples,
#: `observations_used` from 58 and `observation_coverage` from its own set, so
#: the sheet said used/total = 18.69/89.99 = 0.208 while the coverage column,
#: computed per sample and then averaged, said 0.346. Both were arithmetically
#: right and they described different populations.
#:
#: DELIBERATELY NARROW. A sample is dropped from a whole group only when the
#: group is one a formula spans, because every metric excluded from an average
#: is information thrown away: `total_steps` came from 56 samples and
#: `useful_steps` from 40, so binding them costs `total_steps` 16 rows. That is
#: the right price for making the ratio mean something, and the wrong price for
#: a column nobody divides. Metrics whose normalized form is computed
#: per-sample -- prior knowledge, uncertainty, directionality -- are not here:
#: their ratio never crosses two averages.
CO_AVERAGED_METRIC_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "reasoning_observations_total",
            "reasoning_observations_used",
            "reasoning_observation_coverage",
        }
    ),
    frozenset(
        {
            "reasoning_total_steps",
            "reasoning_useful_steps",
            "reasoning_useless_steps",
            "reasoning_useful_step_fraction",
            "reasoning_useless_step_fraction",
        }
    ),
)


def _drop_partial_groups(metrics: dict[str, float]) -> set[str]:
    """Keys this sample must sit out because a metric beside them is missing.

    A sample carrying NONE of a group is untouched -- an io task has no
    reasoning metrics at all and is not what this is about. Only a sample
    holding some of a group but not all of it is excluded, and then from the
    whole group, so every column in it is averaged over identical rows.
    """
    excluded: set[str] = set()
    for group in CO_AVERAGED_METRIC_GROUPS:
        present = {
            key for key in group if key in metrics and not _is_nan(metrics.get(key))
        }
        if present and present != group:
            excluded |= present
    return excluded


def aggregate_mean_metrics(
    score_metrics: Sequence[dict[str, float]],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Mean of every metric key present in a sequence of per-sample metrics.

    Keys absent from a given sample are skipped for that sample (not treated as
    zero), so optional sub-metrics do not silently drag an average down.

    Metrics listed together in ``CO_AVERAGED_METRIC_GROUPS`` are an exception:
    a sample missing any one of them sits out the whole group, so the columns a
    reader combines are means over the same rows.
    """
    buckets: dict[str, list[float]] = {}
    for metrics in score_metrics:
        excluded = _drop_partial_groups(metrics or {})
        for key, value in (metrics or {}).items():
            if key in excluded or _is_nan(value):
                continue
            buckets.setdefault(key, []).append(float(value))
    return {f"{prefix}{key}": mean(values) for key, values in sorted(buckets.items())}
