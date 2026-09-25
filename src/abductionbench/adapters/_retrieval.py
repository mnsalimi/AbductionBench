"""BM25 retrieval over normalised words, for environments that answer from a
fixed inventory of recorded questions (ddxplus).

Words are lower-cased, stop words dropped, and each word reduced to its stem
with the Snowball (Porter2) English stemmer, so "coughing", "coughs" and
"cough" are one term. Scoring is Okapi BM25 (k1 = 1.5, b = 0.75) with the
term statistics -- document frequency, average length -- taken from the WHOLE
inventory, not from the few questions one patient has answers for: how rare a
word is is a property of the questionnaire.

A score is only meaningful relative to its query, so it is normalised by the
query's own ceiling: the sum of the IDFs of its (known) terms, which is what a
question containing every query term once, at average length, would score.
A normalised score of 1.0 is "every word of the question found"; the default
threshold of 0.5 asks for at least half of the question's weight.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

import snowballstemmer

K1 = 1.5
B = 0.75
#: Normalised-score cut-off (see the module docstring).
DEFAULT_THRESHOLD = 0.5

#: A standard English stop-word list (the function words of the NLTK list),
#: plus the words every questionnaire item shares ("do you have ...").
STOPWORDS = frozenset(
    """i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
    she her hers herself it its itself they them their theirs themselves what which who whom this
    that these those am is are was were be been being have has had having do does did doing a an
    the and but if or because as until while of at by for with about against between into through
    during before after above below to from up down in out on off over under again further then
    once here there when where why how all any both each few more most other some such no nor not
    only own same so than too very s t can will just don should now would could ever feel felt
    any anything""".split()
)

_stemmer = snowballstemmer.stemmer("english")
_WORD = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def terms(text: str) -> list[str]:
    """Normalised terms of ``text``: lower-cased, stop words out, stemmed."""
    return [_stem(w) for w in _WORD.findall((text or "").lower()) if w not in STOPWORDS]


class BM25:
    """Okapi BM25 with collection statistics from a fixed inventory."""

    def __init__(self, inventory: Iterable[str], *, k1: float = K1, b: float = B):
        docs = [terms(text) for text in inventory]
        self.k1, self.b = k1, b
        self.n = max(1, len(docs))
        self.avgdl = (sum(len(d) for d in docs) / self.n) or 1.0
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        # The non-negative ("+1") form of the Robertson-Sparck Jones IDF.
        self.idf = {t: math.log((self.n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def score(self, query_terms: list[str], text: str) -> float:
        doc = terms(text)
        if not doc:
            return 0.0
        tf = Counter(doc)
        norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
        total = 0.0
        for term in set(query_terms):
            f = tf.get(term, 0)
            if f:
                total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / (f + norm)
        return total

    def ceiling(self, query_terms: list[str]) -> float:
        return sum(self.idf.get(t, 0.0) for t in set(query_terms))

    def search(
        self,
        query: str,
        documents: Iterable[str],
        *,
        limit: int = 3,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list[tuple[str, float]]:
        """The ``limit`` best documents at or above ``threshold``, best first,
        with their normalised scores."""
        q = terms(query)
        ceiling = self.ceiling(q)
        if ceiling <= 0:
            return []
        scored = [(doc, self.score(q, doc) / ceiling) for doc in documents]
        kept = [(doc, s) for doc, s in scored if s >= threshold]
        kept.sort(key=lambda row: (-row[1], row[0]))
        return kept[:limit]
