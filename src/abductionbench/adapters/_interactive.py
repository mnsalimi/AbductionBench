"""Shared machinery for benchmarks whose data is delivered interactively.

An interactive benchmark is a loop, not a prompt: the model asks for evidence
and the benchmark answers from the case, the cluster or the simulator.  Three
pieces are the same whichever benchmark it is, so they live here:

* :func:`parse_action` -- reading the model's turn as ``{"action", "query"}``.
  Benchmarks in this suite converge on that shape because their own harnesses
  ask for it, and a model that answers in prose still has to be understood.
* :class:`EvidenceStore` -- a flattened, searchable view of one item's findings,
  so a free-text request ("check her potassium", "CT abdomen") can be matched to
  the keys the case actually holds.
* :class:`InteractiveMixin` -- the adapter-side glue: turn accounting, per
  category limits, and the transcript that ends up in the record.

Deliberately *lexical*, not model-driven.  Several of these benchmarks map a
request to a finding using an LLM (VivaBench's ``LLMMapper``) or an embedding
index.  Doing that here would put a second model inside the evaluation of the
first: the score would then depend on how well the examiner model understood the
request, and two runs of the same system could disagree because the examiner
did.  A deterministic matcher is reproducible and its failures are visible in
the transcript, which is the trade this suite wants.  Each adapter says so in
its own caveats.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Action", "parse_action", "EvidenceStore", "InteractiveMixin", "flatten"]

#: Words that carry no discriminating information in a request.  This list is
#: load-bearing, not cosmetic: several of these benchmarks key their evidence by
#: the full question text ("Do you have a cough?"), so leaving auxiliaries in
#: makes every "do you have X" question match every other one -- and an
#: environment that answers a question about fever with the patient's cough has
#: stopped being an environment.
_STOPWORDS = frozenset(
    """a an the of for to and or in on at is are was were be been being am i we you he she it they
    do does did done have has had having would could should will shall may might must can
    like want please check order perform give show tell ask about get me us my your his her their
    this that these those there here any some more much many with without into from by
    what when where why which who whom whose how whether if then than so as also just
    patient case result results finding findings test tests value values""".split()
)

#: A request has to cover at least this much of its own content, or the match is
#: a coincidence rather than an answer.  Calibrated against the real thing: at
#: 0.34 with an uncapped denominator, "do you sweat a lot at night?" missed a
#: finding filed as "sweating", because the words that did not matter counted
#: against the words that did.
_MIN_QUERY_COVERAGE = 0.25
#: Words beyond this many are extra detail, not extra requirements.
_COVERAGE_DENOMINATOR_CAP = 4


@dataclass(slots=True)
class Action:
    """One parsed model turn."""

    action: str
    query: Any = ""
    reasoning: str = ""
    raw: str = ""
    parsed: bool = True

    @property
    def query_text(self) -> str:
        if isinstance(self.query, str):
            return self.query
        return json.dumps(self.query, ensure_ascii=False)


def _json_blobs(text: str) -> list[str]:
    """Every balanced ``{...}`` span in the text, outermost first."""
    blobs: list[str] = []
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                blobs.append(text[start : index + 1])
    return blobs


def parse_action(text: str, *, actions: tuple[str, ...]) -> Action:
    """Read one model turn as an action plus a query.

    Three readings, in order of how much they trust the model: strict JSON, any
    JSON object embedded in prose (models fence it in markdown constantly), and
    finally a keyword scan for an action name followed by the rest of the line.
    The last one exists because an episode is expensive -- 300 cases times
    several turns -- and throwing an episode away over a missing brace measures
    formatting, not reasoning.
    """
    stripped = (text or "").strip()
    if not stripped:
        return Action(action="", raw=text or "", parsed=False)

    fenced = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
    for candidate in [fenced, *_json_blobs(fenced)]:
        try:
            blob = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(blob, dict):
            continue
        # Benchmarks disagree on the field names: VivaBench uses
        # action/query, EvoClinician uses action_type/action_text. Both are
        # read, because the prompts are the benchmarks' own and this adapter
        # layer does not get to rename their fields.
        name = str(
            blob.get("action") or blob.get("action_type") or ""
        ).strip().lower()
        if name in actions:
            query = blob.get("query")
            if query in (None, ""):
                query = blob.get("action_text") or blob.get("content") or ""
            return Action(
                action=name,
                query=query,
                reasoning=str(blob.get("reasoning", "")),
                raw=stripped,
            )

    lowered = fenced.lower()
    for name in actions:
        match = re.search(rf'\b{re.escape(name)}\b\W{{0,4}}(.*)', lowered)
        if match:
            return Action(action=name, query=match.group(1).strip()[:400], raw=stripped)
    return Action(action="", query=stripped[:400], raw=stripped, parsed=False)


def flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten nested case data into ``dotted.key -> readable value``.

    Empty and null findings are dropped: a case that records ``"smoking": null``
    means the finding was never established, and offering it as an answer would
    invent evidence the case does not have.
    """
    out: dict[str, str] = {}
    if value is None or value == "" or value == [] or value == {}:
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, child))
    elif isinstance(value, list):
        rendered = ", ".join(str(v) for v in value if v not in (None, ""))
        if rendered:
            out[prefix] = rendered
    else:
        out[prefix] = str(value)
    return out


#: Words are folded to their first few characters before being compared, so a
#: request and a record can use different forms of the same word: "biopsy and
#: histopathology" has to reach a finding filed under "histological findings",
#: and "how long has the pain been present" has to reach "presented with pain".
#: Four characters is short, and it does merge unrelated words occasionally --
#: which is why a match must still cover a third of the request (see
#: ``_MIN_QUERY_COVERAGE``) before anything is disclosed.
_FOLD_TO = 4


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {
        (w if len(w) <= _FOLD_TO else w[:_FOLD_TO])
        for w in words
        if w not in _STOPWORDS and len(w) > 2
    }


@dataclass(slots=True)
class EvidenceStore:
    """One item's findings, grouped into the categories a request can name."""

    #: category -> {dotted key: rendered finding}
    categories: dict[str, dict[str, str]] = field(default_factory=dict)
    #: keys already disclosed, so the transcript never repeats itself.
    revealed: set[str] = field(default_factory=set)

    def search(self, category: str, query: str, limit: int = 6) -> list[tuple[str, str]]:
        """Findings in ``category`` whose key or value matches the request."""
        wanted = _tokens(query)
        if not wanted:
            return []
        scored: list[tuple[float, str, str]] = []
        for key, value in self.categories.get(category, {}).items():
            key_tokens = _tokens(key.replace(".", " ").replace("_", " "))
            shared = wanted & key_tokens
            weight = 1.0
            if not shared:
                # A request can also name the finding's content ("murmur"), not
                # only its key ("heart_sounds"). That is weaker evidence -- but
                # not when the key is only an index, which is how a case stored
                # as prose is held: there the sentence *is* the finding.
                shared = wanted & _tokens(value)
                weight = 1.0 if not key_tokens else 0.5
            if not shared:
                continue
            # Scored against the request, so a long key cannot dilute a precise
            # question, and a vague question cannot claim a precise finding.
            # The denominator is capped: a request that names the right things
            # is a good match however many other words it used, and models are
            # wordy.
            coverage = len(shared) / min(len(wanted), _COVERAGE_DENOMINATOR_CAP)
            if coverage < _MIN_QUERY_COVERAGE:
                continue
            scored.append((weight * coverage, key, value))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [(key, value) for _score, key, value in scored[:limit]]

    def reveal(self, category: str, query: str, limit: int = 6) -> str:
        """The examiner's answer to one request.

        Findings are reported per *object*, not per leaf.  A lab result is
        stored flattened as ``potassium.name``, ``potassium.value``,
        ``potassium.units``, ``potassium.flag``; disclosing those as four lines
        would make the model reassemble a result that the case records as one.
        """
        hits = self.search(category, query, limit)
        if not hits:
            return ""
        fields = self.categories.get(category, {})
        seen: list[str] = []
        for key, _value in hits:
            parent = key.rsplit(".", 1)[0] if "." in key else key
            if parent not in seen:
                seen.append(parent)
        lines: list[str] = []
        for parent in seen[:limit]:
            # Direct children only. Grouping by prefix alone would sweep an
            # entire subtree into one finding the first time a request matched a
            # shallow key like ``history.chief_complaint``.
            depth = parent.count(".")
            group: dict[str, str] = {}
            for child, value in fields.items():
                if child == parent:
                    group.setdefault("value", value)
                    self.revealed.add(child)
                elif child.startswith(f"{parent}.") and child.count(".") == depth + 1:
                    group[child.rsplit(".", 1)[-1]] = value
                    self.revealed.add(child)
            label = str(group.pop("name", "")) or parent.rsplit(".", 1)[-1].replace("_", " ")
            body = self._render(group)
            if re.fullmatch(r"[a-z]*\s?\d+", label.strip()):
                # A prose case is keyed by sentence index; the index is
                # bookkeeping, not something to read back to the model.
                lines.append(f"- {body}" if body else "")
            else:
                lines.append(f"- {label}: {body}" if body else f"- {label}")
        lines = [line for line in lines if line]
        return "\n".join(lines)

    @staticmethod
    def _render(group: dict[str, str]) -> str:
        """One finding's fields as a clinician would read them back."""
        if not group:
            return ""
        # Drop bookkeeping that means nothing to the reader, and lead with the
        # value the request was actually about.
        ordered: list[str] = []
        for key in ("value", "description", "report", "present", "result", "impression"):
            if key in group:
                ordered.append(str(group.pop(key)))
        units = group.pop("units", "")
        if units and ordered:
            ordered[0] = f"{ordered[0]} {units}"
        for key in ("flag", "reference_range", "duration", "context", "severity", "onset",
                    "note", "modality", "region", "system"):
            value = group.pop(key, "")
            if value not in ("", None):
                ordered.append(f"{key.replace('_', ' ')}: {value}")
        for key, value in group.items():
            if value not in ("", None):
                ordered.append(f"{key.replace('_', ' ')}: {value}")
        return "; ".join(part for part in ordered if part)

    def coverage(self, keys: set[str]) -> float:
        """Fraction of the case's diagnostically relevant keys the model asked for."""
        if not keys:
            return 0.0
        return len(self.revealed & keys) / len(keys)


class InteractiveMixin:
    """Turn accounting shared by the interactive adapters.

    Counts are per category because the benchmarks are: VivaBench allows ten
    history questions and five investigations, and a model that spends twenty
    turns on history has done something the benchmark meant to prevent.
    """

    #: category -> how many requests of that kind an episode may make.
    category_limits: dict[str, int] = {}

    def limit_note(self, state: dict[str, Any], category: str) -> str:
        used = state.setdefault("counts", {}).get(category, 0)
        limit = self.category_limits.get(category)
        if limit is None or used < limit:
            return ""
        return (
            f"\nLimit on {category.replace('_', ' ')} reached. "
            "Please proceed to working up the patient."
        )

    def bump(self, state: dict[str, Any], category: str) -> None:
        counts = state.setdefault("counts", {})
        counts[category] = counts.get(category, 0) + 1

    def over_limit(self, state: dict[str, Any], category: str) -> bool:
        limit = self.category_limits.get(category)
        if limit is None:
            return False
        return state.setdefault("counts", {}).get(category, 0) > limit
