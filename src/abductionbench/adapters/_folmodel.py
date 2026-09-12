"""Deciding whether two first-order formulas say the same thing.

ABD ships, with every instance, the *finite worlds* the hypothesis is about:
a domain of named individuals and the full extension of every predicate over
it.  That is unusual and it is worth exploiting.  Logical equivalence of
first-order formulas is undecidable in general, but equivalence **over a given
finite structure** is not: evaluate both formulas at every individual of every
world and see whether they ever disagree.  Two formulas that pick out the same
individuals in every world the model was shown do the same explanatory work,
whatever their syntax -- ``(and (P x) (Q x))`` and ``(and (Q x) (P x))``,
``(not (not (P x)))`` and ``(P x)``, a renamed bound variable, a contrapositive.

So the check here is a model checker, not a string comparison and not a
language model.  It gives a definite yes or no for anything it can parse and
evaluate, and says ``None`` -- undecidable -- only when it cannot: a malformed
s-expression, an unknown predicate, or a formula whose quantifier nesting would
cost more evaluations than the guard allows.  Those cases, and only those, are
what an LLM judge is asked about.

The syntax is the release's own: ``(and A B)``, ``(or A B)``, ``(not A)``,
``(implies A B)``, ``(iff A B)``, ``(forall v A)``, ``(exists v A)``,
``(= a b)`` and predicate application ``(S x y)``.
"""

from __future__ import annotations

import re
from typing import Any

#: Beyond this many (world, assignment) evaluations a comparison is abandoned
#: as undecidable rather than left to run.  A 3-deep quantifier nest over an
#: 11-element domain is ~1.3k assignments per individual per world, which is
#: cheap; a 6-deep one is not, and no ABD gold needs it.
MAX_EVALUATIONS = 400_000

CONNECTIVES = {"and", "or", "not", "implies", "iff", "forall", "exists", "="}

_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
_PAIR_RE = re.compile(r"[^\s(),]+")
#: Two of the release's 600 golds serialise a bound variable as ``Var(y)``
#: rather than ``y`` -- a bug in its writer, not a different syntax. Repaired
#: on the way in, for the gold and the candidate alike, so those instances are
#: scored rather than silently handed to the judge.
_VAR_WRAPPER_RE = re.compile(r"\bVar\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")


class FormulaError(Exception):
    """The formula could not be parsed, or could not be evaluated."""


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def parse(text: str) -> Any:
    """Parse one s-expression into nested lists of strings."""
    tokens = _TOKEN_RE.findall(_VAR_WRAPPER_RE.sub(r"\1", text or ""))
    if not tokens:
        raise FormulaError("empty formula")
    node, position = _parse_from(tokens, 0)
    if position != len(tokens):
        # Trailing junk means the answer was not a single formula.
        raise FormulaError("more than one expression")
    return node


def _parse_from(tokens: list[str], position: int) -> tuple[Any, int]:
    if position >= len(tokens):
        raise FormulaError("unexpected end of formula")
    token = tokens[position]
    if token == ")":
        raise FormulaError("unbalanced parentheses")
    if token != "(":
        return token, position + 1
    position += 1
    items: list[Any] = []
    while position < len(tokens) and tokens[position] != ")":
        node, position = _parse_from(tokens, position)
        items.append(node)
    if position >= len(tokens):
        raise FormulaError("unbalanced parentheses")
    if not items:
        raise FormulaError("empty parentheses")
    return items, position + 1


# --------------------------------------------------------------------------- #
# worlds
# --------------------------------------------------------------------------- #


def build_world(world: dict[str, Any]) -> tuple[list[str], dict[str, set[tuple[str, ...]]]]:
    """Turn one ABD ``trainWorlds`` entry into a domain and predicate extensions.

    The release writes a unary extension as a list of individuals and a binary
    one as a list of ``"(a0, a1)"`` strings, under a ``true`` key beside the
    complementary ``false`` key.  Only ``true`` is read: the domain is closed,
    so anything not listed true is false.
    """
    domain = [str(element) for element in (world.get("domain") or [])]
    if not domain:
        raise FormulaError("world has no domain")
    extensions: dict[str, set[tuple[str, ...]]] = {}
    for name, extension in (world.get("predicates") or {}).items():
        entries = extension.get("true") if isinstance(extension, dict) else extension
        tuples: set[tuple[str, ...]] = set()
        for entry in entries or []:
            if isinstance(entry, (list, tuple)):
                tuples.add(tuple(str(part) for part in entry))
            else:
                parts = _PAIR_RE.findall(str(entry))
                if parts:
                    tuples.add(tuple(parts))
        extensions[str(name)] = tuples
    return domain, extensions


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


class _Budget:
    """A shared count, so a runaway comparison stops instead of hanging."""

    __slots__ = ("left",)

    def __init__(self, left: int) -> None:
        self.left = left

    def spend(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise FormulaError("evaluation budget exhausted")


def evaluate(
    node: Any,
    domain: list[str],
    extensions: dict[str, set[tuple[str, ...]]],
    binding: dict[str, str],
    budget: _Budget,
) -> bool:
    budget.spend()
    if isinstance(node, str):
        # A bare atom: only a nullary predicate makes sense here.
        if node in extensions:
            return () in extensions[node]
        raise FormulaError(f"unknown atom {node!r}")
    head, rest = node[0], node[1:]
    if not isinstance(head, str):
        raise FormulaError("formula head is not a symbol")
    head_lower = head.lower()

    if head_lower == "not":
        if len(rest) != 1:
            raise FormulaError("not takes one argument")
        return not evaluate(rest[0], domain, extensions, binding, budget)
    if head_lower == "and":
        return all(evaluate(part, domain, extensions, binding, budget) for part in rest)
    if head_lower == "or":
        return any(evaluate(part, domain, extensions, binding, budget) for part in rest)
    if head_lower == "implies":
        if len(rest) != 2:
            raise FormulaError("implies takes two arguments")
        return (not evaluate(rest[0], domain, extensions, binding, budget)) or evaluate(
            rest[1], domain, extensions, binding, budget
        )
    if head_lower == "iff":
        if len(rest) != 2:
            raise FormulaError("iff takes two arguments")
        return evaluate(rest[0], domain, extensions, binding, budget) == evaluate(
            rest[1], domain, extensions, binding, budget
        )
    if head_lower in {"forall", "exists"}:
        if len(rest) != 2:
            raise FormulaError(f"{head_lower} takes a variable and a body")
        variable = rest[0]
        if not isinstance(variable, str):
            raise FormulaError("quantified variable is not a symbol")
        shadowed = binding.get(variable)
        try:
            results = []
            for element in domain:
                binding[variable] = element
                results.append(evaluate(rest[1], domain, extensions, binding, budget))
                if head_lower == "exists" and results[-1]:
                    return True
                if head_lower == "forall" and not results[-1]:
                    return False
            return head_lower == "forall"
        finally:
            if shadowed is None:
                binding.pop(variable, None)
            else:
                binding[variable] = shadowed
    if head_lower in {"=", "eq", "equal"}:
        if len(rest) != 2:
            raise FormulaError("= takes two arguments")
        return _term(rest[0], binding) == _term(rest[1], binding)
    if head in extensions:
        return tuple(_term(argument, binding) for argument in rest) in extensions[head]
    raise FormulaError(f"unknown predicate {head!r}")


def _term(node: Any, binding: dict[str, str]) -> str:
    if not isinstance(node, str):
        raise FormulaError("a term must be a symbol")
    return binding.get(node, node)


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def extensionally_equal(
    candidate: str,
    gold: str,
    worlds: list[dict[str, Any]],
    *,
    free_variable: str = "x",
    max_evaluations: int = MAX_EVALUATIONS,
) -> bool | None:
    """Do the two formulas pick out the same individuals in every world?

    ``True``/``False`` is a decision; ``None`` means the check does not apply --
    the candidate could not be parsed, names something the world does not
    define, or would cost more than the evaluation budget. A caller should send
    exactly those to a judge and nothing else.
    """
    if not worlds:
        return None
    try:
        candidate_node = parse(candidate)
        gold_node = parse(gold)
    except FormulaError:
        return None
    budget = _Budget(max_evaluations)
    try:
        for world in worlds:
            domain, extensions = build_world(world)
            for element in domain:
                binding = {free_variable: element}
                left = evaluate(candidate_node, domain, extensions, dict(binding), budget)
                right = evaluate(gold_node, domain, extensions, dict(binding), budget)
                if left != right:
                    return False
    except FormulaError:
        return None
    except RecursionError:
        return None
    return True
