"""The 50 hidden target functions of Alien Abduction, and their test suites.

Alien Abduction (arXiv:2608.03388) hides a small pure Python function behind a
Game Master and asks the model to recover it from input/output evidence.  Its
authors state that "the source code and test instances will be released upon
acceptance"; at the time of writing no release exists, so this module rebuilds
the target set from what the paper *does* publish.

What is taken from the paper, not invented here:

* **The five domains and their signatures** (Table 2): ``int -> int``,
  ``(int, int) -> int``, ``str -> str``, ``List[int] -> int`` and
  ``(bool, bool) -> bool``, ten functions each, fifty in all.
* **The names of all fifty functions** (Table 3).  Every function below is named
  exactly as the paper names it, in the paper's order, so a per-target
  comparison against the published results is meaningful.
* **The evidence construction** (Section 4.1): each target is paired with test
  cases whose inputs come from "type-aware pools that mix edge cases (e.g.,
  zero, negatives, empty strings and lists) with random values", with every
  output "computed by executing the target function".

What the paper does **not** publish is the fifty *bodies* -- Table 3 is a list
of names.  Each body here is therefore this suite's reading of the paper's name,
written to be the shortest function that the name straightforwardly describes.
Where a name admits more than one honest reading the choice is recorded in the
function's own docstring, and :data:`AMBIGUOUS` lists those targets so the run
documentation can say plainly which results rest on an interpretation.  The
targets are what the paper calls "primitive": short, pure, deterministic, and
standard-library only.

Two of the Boolean targets (``b_without_a`` and ``difference_negative``) reduce
to the same truth table from different derivations.  That is left as it is: on
two Boolean inputs only sixteen functions exist and ten names must land among
them, so a collision is a property of the published name list rather than
something to design away by renaming a target the paper named.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import string
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Target", "TARGETS", "BY_NAME", "DOMAINS", "AMBIGUOUS",
    "test_cases", "reveal_order", "render_input",
    "materialize", "load_materialized", "fingerprint", "GENERATOR_VERSION",
]

# --------------------------------------------------------------------------- #
# Number: (x: int) -> int
# --------------------------------------------------------------------------- #


def difference_from_reversed_absolute(x: int) -> int:
    """abs(x) minus the number its decimal digits spell backwards."""
    magnitude = abs(x)
    return magnitude - int(str(magnitude)[::-1])


def smallest_divisor_ge_two_abs(x: int) -> int:
    """The smallest divisor >= 2 of abs(x).

    Reading: 0 and 1 have no divisor >= 2 that is meaningful here (every integer
    divides 0), so they are returned unchanged; a prime returns itself.
    """
    magnitude = abs(x)
    if magnitude < 2:
        return magnitude
    divisor = 2
    while divisor * divisor <= magnitude:
        if magnitude % divisor == 0:
            return divisor
        divisor += 1
    return magnitude


def distance_from_square_of_three(x: int) -> int:
    """How far x is from 9."""
    return abs(x - 9)


def add_seven_if_negative(x: int) -> int:
    """Add 7 to a negative x; leave any other x alone."""
    return x + 7 if x < 0 else x


def max_with_negative_self(x: int) -> int:
    """The larger of x and -x."""
    return max(x, -x)


def times_four_mod_nine(x: int) -> int:
    """Four times x, modulo nine."""
    return (x * 4) % 9


def integer_average_with_ten(x: int) -> int:
    """The integer average of x and ten."""
    return (x + 10) // 2


def modulo_of_cube_by_eleven(x: int) -> int:
    """x cubed, modulo eleven."""
    return (x**3) % 11


def fibonacci_index_small(x: int) -> int:
    """The Fibonacci number at index abs(x), for small indices.

    Reading: "small" is taken as the bound that keeps the target primitive, so
    the index is clamped to 20 (F(0) = 0, F(1) = 1).  Without a clamp the
    function is unbounded and its outputs stop being evidence a reader can use.
    """
    index = min(abs(x), 20)
    previous, current = 0, 1
    for _ in range(index):
        previous, current = current, previous + current
    return previous


def absolute_modulus_gap_eleven(x: int) -> int:
    """How far abs(x) is from the nearest multiple of eleven.

    Reading: "gap" is the distance to the nearest multiple in either direction,
    which is what makes the name say something ``abs(x) % 11`` does not.
    """
    remainder = abs(x) % 11
    return min(remainder, 11 - remainder)


# --------------------------------------------------------------------------- #
# Number Pairs: (a: int, b: int) -> int
# --------------------------------------------------------------------------- #


def last_digit_sum(a: int, b: int) -> int:
    """The last decimal digit of a + b."""
    return abs(a + b) % 10


def zero_if_opposite_else_product(a: int, b: int) -> int:
    """Zero when a and b are opposites; otherwise their product."""
    return 0 if a == -b else a * b


def absolute_sum_minus_absolute_difference(a: int, b: int) -> int:
    """abs(a + b) - abs(a - b)."""
    return abs(a + b) - abs(a - b)


def sum_if_negative(a: int, b: int) -> int:
    """a + b when that sum is negative, otherwise zero.

    Reading: the condition is on the sum, mirroring ``add_seven_if_negative``,
    where the test is likewise on the value the function is about.
    """
    total = a + b
    return total if total < 0 else 0


def difference_squared(a: int, b: int) -> int:
    """The square of a - b."""
    return (a - b) ** 2


def sum_after_incrementing_first(a: int, b: int) -> int:
    """Add one to a, then add b."""
    return (a + 1) + b


def manhattan_to_pair(a: int, b: int) -> int:
    """The Manhattan distance from the origin to the point (a, b)."""
    return abs(a) + abs(b)


def sum_plus_larger_abs(a: int, b: int) -> int:
    """a + b plus whichever of them has the larger magnitude."""
    return a + b + max(abs(a), abs(b))


def triple_sum_minus_product(a: int, b: int) -> int:
    """Three times the sum, less the product."""
    return 3 * (a + b) - a * b


def product_mod_seven(a: int, b: int) -> int:
    """The product of a and b, modulo seven."""
    return (a * b) % 7


# --------------------------------------------------------------------------- #
# String: (text: str) -> str
# --------------------------------------------------------------------------- #


def rot13_lowercase_only(text: str) -> str:
    """ROT13 the lowercase letters; leave every other character alone."""
    out = []
    for char in text:
        if "a" <= char <= "z":
            out.append(chr((ord(char) - 97 + 13) % 26 + 97))
        else:
            out.append(char)
    return "".join(out)


def keep_whitespace_only(text: str) -> str:
    """Keep the whitespace and discard everything else."""
    return "".join(char for char in text if char.isspace())


def mirror_with_pipe(text: str) -> str:
    """The text, a pipe, then the text backwards."""
    return text + "|" + text[::-1]


def strip_and_lowercase(text: str) -> str:
    """Strip the outer whitespace and lowercase what is left."""
    return text.strip().lower()


def repeat_three_times(text: str) -> str:
    """The text three times over."""
    return text * 3


def remove_whitespace(text: str) -> str:
    """The text with every whitespace character removed."""
    return "".join(char for char in text if not char.isspace())


def hex_code_points(text: str) -> str:
    """Each character's code point in lowercase hex, space separated."""
    return " ".join(format(ord(char), "x") for char in text)


def take_last_three(text: str) -> str:
    """The last three characters (all of them, if there are fewer)."""
    return text[-3:]


def letters_only(text: str) -> str:
    """Keep the alphabetic characters and discard everything else."""
    return "".join(char for char in text if char.isalpha())


def prepend_hash(text: str) -> str:
    """A '#' in front of the text."""
    return "#" + text


# --------------------------------------------------------------------------- #
# List: (items: List[int]) -> int
# --------------------------------------------------------------------------- #


def sum_excluding_max(items: list[int]) -> int:
    """The sum with one occurrence of the largest value left out."""
    if not items:
        return 0
    return sum(items) - max(items)


def sum_mod_three(items: list[int]) -> int:
    """The sum, modulo three."""
    return sum(items) % 3


def count_adjacent_opposite_pairs(items: list[int]) -> int:
    """How many neighbouring pairs have opposite signs (zero counts as neither)."""
    return sum(1 for left, right in zip(items, items[1:], strict=False) if left * right < 0)


def sum_smaller_of_neighbors(items: list[int]) -> int:
    """For each neighbouring pair, add the smaller of the two."""
    return sum(min(left, right) for left, right in zip(items, items[1:], strict=False))


def count_distinct_values(items: list[int]) -> int:
    """How many different values appear."""
    return len(set(items))


def sum_of_positive_squares(items: list[int]) -> int:
    """The sum of the squares of the values above zero."""
    return sum(value * value for value in items if value > 0)


def sum_negative_values(items: list[int]) -> int:
    """The sum of the values below zero."""
    return sum(value for value in items if value < 0)


def sum_neighbors_products(items: list[int]) -> int:
    """The sum of the products of each neighbouring pair."""
    return sum(left * right for left, right in zip(items, items[1:], strict=False))


def sum_palindrome_values(items: list[int]) -> int:
    """The sum of the values whose digits read the same backwards.

    Reading: the sign is not a digit, so -121 is a palindrome and -12 is not.
    """
    total = 0
    for value in items:
        digits = str(abs(value))
        if digits == digits[::-1]:
            total += value
    return total


def count_odd_values(items: list[int]) -> int:
    """How many of the values are odd."""
    return sum(1 for value in items if value % 2 != 0)


# --------------------------------------------------------------------------- #
# Logic: (a: bool, b: bool) -> bool
# --------------------------------------------------------------------------- #


def both_are_true_identity(a: bool, b: bool) -> bool:
    """True when both are true."""
    return a and b


def b_without_a(a: bool, b: bool) -> bool:
    """b with a taken out of it: true when b holds and a does not."""
    return b and not a


def a_requires_b(a: bool, b: bool) -> bool:
    """a implies b: false only when a holds without b."""
    return (not a) or b


def second_is_majority(a: bool, b: bool) -> bool:
    """Whether the majority of the two votes is b's.

    Reading: with two voters the majority is only decided when they agree, and
    a tie goes to the second voter -- so on either branch the answer is b.
    """
    return b if a != b else b


def bool_from_any_tuple(a: bool, b: bool) -> bool:
    """The truth value of the tuple holding the two arguments.

    Reading: taken literally, and a non-empty tuple is always truthy -- so this
    target is the constant True regardless of its arguments.
    """
    return bool((a, b))


def a_or_b_via_if(a: bool, b: bool) -> bool:
    """a, or else b -- written as a conditional."""
    return True if a else b


def difference_negative(a: bool, b: bool) -> bool:
    """Whether a - b is negative, counting False as 0 and True as 1."""
    return (int(a) - int(b)) < 0


def nor_result(a: bool, b: bool) -> bool:
    """Neither one."""
    return not (a or b)


def all_or_none(a: bool, b: bool) -> bool:
    """Both, or neither."""
    return a == b


def reverse_implication(a: bool, b: bool) -> bool:
    """b implies a: false only when b holds without a."""
    return a or (not b)


# --------------------------------------------------------------------------- #
# the target table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Target:
    """One hidden function, as the Game Master holds it."""

    name: str
    domain: str
    signature: str
    fn: Callable[..., Any]

    @property
    def source(self) -> str:
        """The function's own source, which is what an episode reveals at the end."""
        return inspect.getsource(self.fn).strip()

    @property
    def arity(self) -> int:
        return 2 if self.domain in ("Number Pairs", "Logic") else 1


#: domain -> (signature, the ten functions the paper lists for it, in its order)
DOMAINS: dict[str, tuple[str, tuple[Callable[..., Any], ...]]] = {
    "Number": (
        "(x: int) -> int",
        (
            difference_from_reversed_absolute,
            smallest_divisor_ge_two_abs,
            distance_from_square_of_three,
            add_seven_if_negative,
            max_with_negative_self,
            times_four_mod_nine,
            integer_average_with_ten,
            modulo_of_cube_by_eleven,
            fibonacci_index_small,
            absolute_modulus_gap_eleven,
        ),
    ),
    "Number Pairs": (
        "(a: int, b: int) -> int",
        (
            last_digit_sum,
            zero_if_opposite_else_product,
            absolute_sum_minus_absolute_difference,
            sum_if_negative,
            difference_squared,
            sum_after_incrementing_first,
            manhattan_to_pair,
            sum_plus_larger_abs,
            triple_sum_minus_product,
            product_mod_seven,
        ),
    ),
    "String": (
        "(text: str) -> str",
        (
            rot13_lowercase_only,
            keep_whitespace_only,
            mirror_with_pipe,
            strip_and_lowercase,
            repeat_three_times,
            remove_whitespace,
            hex_code_points,
            take_last_three,
            letters_only,
            prepend_hash,
        ),
    ),
    "List": (
        "(items: List[int]) -> int",
        (
            sum_excluding_max,
            sum_mod_three,
            count_adjacent_opposite_pairs,
            sum_smaller_of_neighbors,
            count_distinct_values,
            sum_of_positive_squares,
            sum_negative_values,
            sum_neighbors_products,
            sum_palindrome_values,
            count_odd_values,
        ),
    ),
    "Logic": (
        "(a: bool, b: bool) -> bool",
        (
            both_are_true_identity,
            b_without_a,
            a_requires_b,
            second_is_majority,
            bool_from_any_tuple,
            a_or_b_via_if,
            difference_negative,
            nor_result,
            all_or_none,
            reverse_implication,
        ),
    ),
}

TARGETS: tuple[Target, ...] = tuple(
    Target(name=fn.__name__, domain=domain, signature=signature, fn=fn)
    for domain, (signature, functions) in DOMAINS.items()
    for fn in functions
)

BY_NAME: dict[str, Target] = {target.name: target for target in TARGETS}

#: Targets whose published name admits more than one honest body.  Named here
#: so the run documentation can say which results rest on a reading rather than
#: leaving the reader to infer it from the source.
AMBIGUOUS: dict[str, str] = {
    "smallest_divisor_ge_two_abs": "0 and 1 have no divisor >= 2; both are returned unchanged",
    "fibonacci_index_small": "'small' read as a clamp: the index is min(abs(x), 20)",
    "absolute_modulus_gap_eleven": "'gap' read as distance to the nearest multiple, not abs(x) % 11",
    "last_digit_sum": "read as the last digit of the sum, not the sum of the last digits",
    "sum_if_negative": "the negativity test is on the sum, as in add_seven_if_negative",
    "second_is_majority": "a tie among two voters goes to the second, so the answer is b",
    "bool_from_any_tuple": "read literally: a non-empty tuple is truthy, so the target is constant True",
    "sum_palindrome_values": "the sign is not a digit, so -121 counts and -12 does not",
}


# --------------------------------------------------------------------------- #
# evidence: the per-target test suite
# --------------------------------------------------------------------------- #

#: Integers the paper's "edge cases" clause names, plus the ones that separate
#: the arithmetic targets from each other (multiples of 9, 10 and 11 pull
#: times_four_mod_nine, last_digit_sum and absolute_modulus_gap_eleven apart).
_EDGE_INTS = (0, 1, -1, 2, -2, 3, -3, 7, -7, 9, -9, 10, -10, 11, -11, 12, 21, 99, -99, 100, -100, 121, 1000, -1000)

_EDGE_STRINGS = (
    "",
    " ",
    "   ",
    "a",
    "Z",
    "abc",
    "ABC",
    "Hello, World!",
    "  padded  ",
    "123",
    "\t\n",
    "aA bB cC",
    "mixed CASE 42",
    "ab",
    "racecar",
    "!@#$%",
    "a b",
    "The Quick Brown Fox",
)

_EDGE_LISTS: tuple[list[int], ...] = (
    [],
    [0],
    [1],
    [-1],
    [0, 0],
    [1, 1],
    [1, -1],
    [-1, 1],
    [5, 5, 5],
    [3, -3, 3, -3],
    [121, 12, -121],
    [2, 4, 6, 8],
    [1, 3, 5, 7],
    [-2, -4, -6],
    [10, 0, -10],
    [7],
)

_ALPHABET = string.ascii_letters + string.digits + "   .,!?"


def _seeded(target: Target) -> random.Random:
    """A generator fixed by the target's name, so every mode sees one suite.

    The paper draws its targets and cases "with a fixed seed" precisely so that
    "the same 50 targets are used across the modes"; keying off the name rather
    than the run seed extends that to every run of this harness as well.
    """
    digest = hashlib.sha256(target.name.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _input_pool(target: Target, rng: random.Random, count: int) -> tuple[list[tuple[Any, ...]], int]:
    """Type-aware inputs for one target, and how many of them are edge cases.

    The edge cases come first, which is bookkeeping rather than an order the
    model ever sees: :func:`reveal_order` is what decides what gets revealed
    first, and it needs to know which is which.
    """
    domain = target.domain
    if domain == "Logic":
        # Two booleans admit four distinct inputs and no more, so the suite is
        # the whole input space -- and all of it is an edge case. Asking for a
        # hundred would only repeat it.
        space = [(a, b) for a in (False, True) for b in (False, True)]
        return space, len(space)

    pool: list[tuple[Any, ...]] = []
    seen: set[str] = set()

    def add(args: tuple[Any, ...]) -> None:
        key = repr(args)
        if key not in seen:
            seen.add(key)
            pool.append(args)

    if domain == "Number":
        for value in _EDGE_INTS:
            add((value,))
        edges = len(pool)
        while len(pool) < count:
            add((rng.randint(-1000, 1000),))
    elif domain == "Number Pairs":
        edges = 0
        for left in _EDGE_INTS[:8]:
            for right in _EDGE_INTS[:8]:
                add((left, right))
                if len(pool) >= count // 2:
                    break
            if len(pool) >= count // 2:
                break
        edges = len(pool)
        while len(pool) < count:
            add((rng.randint(-100, 100), rng.randint(-100, 100)))
    elif domain == "String":
        for value in _EDGE_STRINGS:
            add((value,))
        edges = len(pool)
        while len(pool) < count:
            length = rng.randint(1, 12)
            add(("".join(rng.choice(_ALPHABET) for _ in range(length)),))
    elif domain == "List":
        for value in _EDGE_LISTS:
            add((list(value),))
        edges = len(pool)
        while len(pool) < count:
            length = rng.randint(0, 8)
            add(([rng.randint(-20, 20) for _ in range(length)],))
    else:  # pragma: no cover - the five domains above are the whole table
        raise ValueError(f"no input pool for domain {domain!r}")
    return pool[:count], min(edges, count)


def test_cases(target: Target, count: int = 100) -> list[tuple[tuple[Any, ...], Any]]:
    """The target's suite: ``[(args, expected), ...]``, outputs computed by running it.

    Section 4.1 gives this suite both of its roles -- it is the evidence the
    passive and single-turn modes reveal, and it is the held-out suite every
    submitted hypothesis is verified against.  One hundred cases per target,
    except in Logic, where the input space itself holds only four.
    """
    rng = _seeded(target)
    pool, _edges = _input_pool(target, rng, count)
    return [(args, target.fn(*args)) for args in pool]


#: How much of a revealed batch should be edge cases.  The paper does not fix a
#: ratio, but the batch it prints in Table 6 is ten pairs of which every one is
#: an edge case, and the reason is plain: a batch of ten random four-digit
#: integers separates almost none of the Number targets from each other, so an
#: unweighted draw would be measuring the draw rather than the model.
_EDGE_SHARE = 0.6


def reveal_order(target: Target, count: int = 100) -> list[int]:
    """The order this target's cases are revealed in, as indices into its suite.

    One order per target, fixed by its name, and used by both passive modes: the
    single-turn batch is this order's prefix and the sequential mode walks it
    one pair per turn, so the two modes differ in how the evidence arrives
    rather than in what it is.  Edge cases are drawn preferentially early
    because that is what makes the early evidence discriminating.
    """
    rng = _seeded(target)
    pool, edges = _input_pool(target, rng, count)
    # A second, independent stream: drawing the order from the same generator
    # that built the pool would make the order depend on how many random inputs
    # the pool happened to need.
    rng = random.Random(f"{target.name}:reveal")
    edge_indices = list(range(edges))
    rest_indices = list(range(edges, len(pool)))
    rng.shuffle(edge_indices)
    rng.shuffle(rest_indices)
    order: list[int] = []
    while edge_indices or rest_indices:
        if edge_indices and (not rest_indices or rng.random() < _EDGE_SHARE):
            order.append(edge_indices.pop())
        else:
            order.append(rest_indices.pop())
    return order


def render_input(args: tuple[Any, ...]) -> str:
    """One input as the Game Master writes it: ``-10`` for one argument, ``2, 1`` for two."""
    return ", ".join(repr(arg) for arg in args)


# --------------------------------------------------------------------------- #
# materialization -- the generated benchmark, written out as data
# --------------------------------------------------------------------------- #

#: Bumped whenever a target body or the evidence construction changes, so a
#: materialized copy from an older version is detected rather than trusted.
GENERATOR_VERSION = "1.0"

TARGETS_FILE = "targets.json"
CASES_FILE = "test_cases.jsonl"
MANIFEST_FILE = "MANIFEST.json"


def fingerprint(count: int = 100) -> str:
    """Identifies exactly which benchmark the generator currently produces.

    Covers the generator version, the requested suite size and every target's
    name *and source*, so editing one function body invalidates the copy on
    disk.  Without this a materialized dataset would silently outlive the code
    that describes it, and a run would score one benchmark against another's
    reference.
    """
    digest = hashlib.sha256()
    digest.update(f"{GENERATOR_VERSION}:{count}".encode())
    for target in TARGETS:
        digest.update(target.name.encode())
        digest.update(target.source.encode())
    return digest.hexdigest()[:16]


def materialize(root: Any, count: int = 100) -> dict[str, Any]:
    """Write the generated benchmark into ``root`` and return its manifest.

    Alien Abduction has nothing to download -- its authors released no code or
    test instances -- so where every other dataset here caches a clone or a
    snapshot, this one writes what it generated.  Doing so is not a convenience:
    it is what makes the benchmark *inspectable*.  A reader can diff the suites,
    a run can be audited against the exact evidence it used, and the data
    directory stops being empty for a reason no one can see from the outside.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    targets_payload = [
        {
            "name": target.name,
            "domain": target.domain,
            "signature": target.signature,
            "arity": target.arity,
            "source": target.source,
            "ambiguous_reading": AMBIGUOUS.get(target.name, ""),
        }
        for target in TARGETS
    ]
    (root / TARGETS_FILE).write_text(
        json.dumps(targets_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with open(root / CASES_FILE, "w", encoding="utf-8") as handle:
        for target in TARGETS:
            cases = test_cases(target, count)
            handle.write(
                json.dumps(
                    {
                        "name": target.name,
                        "domain": target.domain,
                        "cases": [[list(args), output] for args, output in cases],
                        "reveal_order": reveal_order(target, count),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "fingerprint": fingerprint(count),
        "requested_cases_per_target": count,
        "targets": len(TARGETS),
        "domains": {
            domain: {"signature": signature, "targets": len(functions)}
            for domain, (signature, functions) in DOMAINS.items()
        },
        "ambiguous_readings": AMBIGUOUS,
        "provenance": (
            "Generated, not downloaded. The benchmark's authors released no code or test "
            "instances ('will be released upon acceptance'), so the 50 target names, their "
            "signatures, the five domains and the evidence construction are taken from the "
            "paper (arXiv:2608.03388, Table 2, Table 3, Section 4.1) and the function bodies "
            "are this suite's reading of those published names. Regenerate with "
            "`python tools/alien_abduction_data.py`."
        ),
    }
    (root / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_materialized(root: Any, count: int = 100) -> list[dict[str, Any]]:
    """Read the written benchmark back, regenerating it first if it is absent or stale."""
    root = Path(root)
    manifest_path = root / MANIFEST_FILE
    current = fingerprint(count)
    stale = True
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stale = manifest.get("fingerprint") != current
        except (json.JSONDecodeError, OSError):
            stale = True
    if stale or not (root / CASES_FILE).is_file():
        materialize(root, count)

    items: list[dict[str, Any]] = []
    with open(root / CASES_FILE, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            blob = json.loads(line)
            items.append(
                {
                    "target": BY_NAME[blob["name"]],
                    # Back to tuples: an input is a fixed argument list, and JSON
                    # has only one sequence type. Leaving them as lists would make
                    # two modes' suites compare unequal while being the same data.
                    "cases": [(tuple(args), output) for args, output in blob["cases"]],
                    "order": list(blob["reveal_order"]),
                }
            )
    return items
