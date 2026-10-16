"""Small lexical helpers shared by the heuristic (offline) components."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    tell explain describe give list please pls plz know want need does mean thing things
    stuff hey hi hello kindly could would something anything info information""".split()
)

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(word: str) -> str:
    """Very light suffix stripping so 'embeddings' matches 'embedding'."""
    for suffix in ("ings", "ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def keywords(text: str) -> set[str]:
    return {stem(w) for w in tokenize(text) if w not in STOPWORDS and len(w) > 1}


def keyword_overlap(query: str, text: str) -> float:
    """Fraction of the query's keywords that appear in `text`."""
    q = keywords(query)
    if not q:
        return 0.0
    return len(q & keywords(text)) / len(q)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if len(s.strip()) > 15]
