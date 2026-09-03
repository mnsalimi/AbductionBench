"""Normalizing mathematical expressions before symbolic comparison.

Models answer physics questions in the notation they were trained on -- LaTeX
(``\\dfrac{a}{b}``, ``\\sqrt{x}``), unicode (``ε₀``, ``·``, ``√``), and implicit
multiplication (``mu G M_m`` for ``mu*G*M_m``) -- no matter how firmly the prompt
asks for Python syntax.  Comparing such an answer with SymPy fails to parse and
the item is scored as undecidable, which measures the notation rather than the
physics: in a live run this turned an answer with 0.95 token overlap against the
reference into a zero.

This module converts those surface forms into something SymPy can parse, and
parses with implicit multiplication enabled.  It never changes the *meaning* of
an expression: every rule is a notation rewrite, and anything unrecognized is
left alone so the comparison still reports itself as undecidable rather than
guessing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

__all__ = ["normalize_math", "parse_math", "equal_expressions", "equal_up_to_scale"]

#: Greek letters and other symbols models use in place of ASCII names.
_UNICODE_NAMES = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda_", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "Gamma", "Δ": "Delta",
    "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Π": "Pi", "Σ": "Sigma",
    "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega", "ℏ": "hbar",
}
#: Unicode sub/superscript digits, which models use for indices (n₀, ε₀).
_SUBSCRIPTS = {"₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
               "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9"}
_SUPERSCRIPTS = {"²": "**2", "³": "**3", "⁴": "**4", "½": "*0.5"}

#: PhysGym's reference equations are written for numpy/python evaluation
#: (``np.pi``, ``np.arctan``, ``math.sqrt``); SymPy needs the bare names.
_MODULE_PREFIXES = (
    (r"\b(?:np|numpy|math|sp|sympy)\s*\.\s*", ""),
    (r"\barctan\b", "atan"),
    (r"\barcsin\b", "asin"),
    (r"\barccos\b", "acos"),
    (r"\barctan2\b", "atan2"),
    (r"\bpower\s*\(", "Pow("),
    (r"\babs\s*\(", "Abs("),
)

#: LaTeX Greek commands (``\eta``, ``\Delta``) as opposed to the unicode forms
#: handled by ``_UNICODE_NAMES``.  Longest names first so ``\theta`` is not
#: matched as ``\the`` + ``ta``.
_LATEX_GREEK = tuple(
    sorted(
        (
            "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta",
            "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu",
            "xi", "pi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon",
            "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta",
            "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
            "hbar", "ell", "infty",
        ),
        key=len,
        reverse=True,
    )
)

_LATEX_COMMANDS = (
    (r"\\?[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))"),
    (r"\\?sqrt\s*\[\s*([^\]]*)\s*\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))"),
    (r"\\?sqrt\s*\{([^{}]*)\}", r"((\1)**0.5)"),
    (r"\\?sqrt\s*\(([^()]*)\)", r"((\1)**0.5)"),
    (r"√\s*\{([^{}]*)\}", r"((\1)**0.5)"),
    (r"√\s*\[([^\]]*)\]", r"((\1)**0.5)"),
    (r"√\s*\(([^()]*)\)", r"((\1)**0.5)"),
    (r"\\exp\s*\{([^{}]*)\}", r"exp(\1)"),
    (r"\\(left|right|,|;|:|!|quad|qquad|displaystyle|mathrm|text|mathbf|bm)", " "),
    (r"\\cdot|·|×|✕", "*"),
    (r"\\div|÷", "/"),
    (r"\\pi", "pi"),
    (r"\\times", "*"),
    (r"\\?\bln\b", "log"),
)


def normalize_math(text: str) -> str:
    """Rewrite LaTeX/unicode notation into SymPy-parseable text."""
    if not text:
        return ""
    body = text.strip().strip("`$ ").replace("\n", " ")
    # Strip a leading "<var> =" so only the right-hand side remains.
    body = re.sub(r"^[^=]{0,40}=\s*", "", body) if body.count("=") == 1 else body
    body = body.replace("$", " ")
    for name in _LATEX_GREEK:
        # "lambda" is a Python keyword, so it gets the same suffix the unicode
        # mapping uses; "infty" becomes SymPy's oo.
        target = {"lambda": "lambda_", "infty": "oo"}.get(name, name)
        body = re.sub(rf"\\{name}\b", target, body)
    for pattern, replacement in _MODULE_PREFIXES:
        body = re.sub(pattern, replacement, body)
    # Apply the whole rule set repeatedly: a nested construct such as
    # \frac{\sqrt{3} a}{b} only becomes matchable once the inner \sqrt has been
    # rewritten, and the brace-free patterns cannot see through nesting.
    for _ in range(4):
        before = body
        for pattern, replacement in _LATEX_COMMANDS:
            body = re.sub(pattern, replacement, body)
        if body == before:
            break
    for source, target in {**_UNICODE_NAMES, **_SUBSCRIPTS, **_SUPERSCRIPTS}.items():
        body = body.replace(source, target)
    body = body.replace("^", "**").replace("{", "(").replace("}", ")")
    body = re.sub(r"[\[\]]", "", body)
    # Unicode minus/dashes, and any backslash a rule did not consume.
    body = body.replace("\u2212", "-").replace("\u2013", "-").replace("\\", "")
    # A comma between symbols is a LaTeX thin space (\,) whose backslash is
    # gone; treat it as multiplication.  Commas inside a real function call are
    # not used by these datasets' references.
    body = re.sub(r"(?<=[A-Za-z0-9_)])\s*,\s*(?=[A-Za-z(])", "*", body)
    # "epsilon_0E_0" is two symbols juxtaposed, which SymPy would read as one.
    body = re.sub(r"(?<=_\d)(?=[A-Za-z])", "*", body)
    # A prime is notation for a related quantity ("t'"); rename it so the
    # expression parses and is *decided* (as a mismatch) instead of being
    # reported as unreadable.
    body = re.sub(r"([A-Za-z][A-Za-z0-9_]*)'", r"\1_prime", body)
    body = re.sub(r"\s+", " ", body).strip().rstrip(".,;:")
    # Drop a trailing unbalanced ')' left by prose like "...)."
    while body.count(")") > body.count("("):
        body = body[: body.rfind(")")] + body[body.rfind(")") + 1 :]
    return body.strip()


def parse_math(text: str, symbols: Sequence[str]) -> Any | None:
    """Parse an expression with implicit multiplication; ``None`` if impossible.

    Implicit multiplication matters because models write ``mu G M_m`` rather
    than ``mu*G*M_m``; without it that parses as an unknown function call.
    """
    normalized = normalize_math(text)
    if not normalized:
        return None
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:  # pragma: no cover - sympy is a declared dependency
        return None
    local = {name: sympy.Symbol(name) for name in symbols if name}
    # Function and constant names must be bound, or implicit multiplication
    # turns `sqrt(x)` into `sqrt * (x)`.
    for name in (
        "sqrt", "exp", "log", "sin", "cos", "tan", "atan", "asin", "acos",
        "sinh", "cosh", "tanh", "atan2", "Abs", "Pow",
    ):
        local[name] = getattr(sympy, name, None) or sympy.Function(name)
    local["pi"] = sympy.pi
    local.setdefault("E", sympy.E)
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        return parse_expr(
            normalized, local_dict=local, transformations=transformations, evaluate=True
        )
    except Exception:  # noqa: BLE001 - unparseable model output
        return None


def equal_expressions(candidate: str, reference: str, symbols: Sequence[str]) -> bool | None:
    """``True``/``False``, or ``None`` when the comparison cannot be decided."""
    left = parse_math(candidate, symbols)
    right = parse_math(reference, symbols)
    if left is None or right is None:
        return None
    try:
        import sympy

        return bool(sympy.simplify(left - right) == 0)
    except Exception:  # noqa: BLE001 - simplification blew up
        return None


def equal_up_to_scale(candidate: str, reference: str, symbols: Sequence[str]) -> bool | None:
    """Equality up to a non-zero scalar factor (for expressions equal to zero)."""
    left = parse_math(candidate, symbols)
    right = parse_math(reference, symbols)
    if left is None or right is None:
        return None
    try:
        import sympy

        if sympy.simplify(left - right) == 0:
            return True
        if right == 0:
            return bool(sympy.simplify(left) == 0)
        ratio = sympy.simplify(left / right)
        return bool(ratio.is_number and ratio != 0)
    except Exception:  # noqa: BLE001
        return None
