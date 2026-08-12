"""
Signal H: constraint / instruction importance.

Detects phrases that change the meaning of a query as a filter or
requirement - comparators, quantifiers, inclusion/exclusion language.
These are matched as multi-word phrases first (so "greater than"
scores as a unit rather than two independently weak words), then the
phrase's score is copied onto every token inside it.

Numbers, currency amounts, and percentages near constraint language
are also boosted, since "greater than $10,000" is only meaningful with
the number attached - the number and the comparator must survive
compression together.
"""
import re
from typing import List
from tokenizer.tokenizer import WordToken

CONSTRAINT_PHRASES = [
    "at least", "at most", "no more than", "no less than",
    "greater than", "less than", "more than", "fewer than",
    "before", "after", "only", "exactly", "must", "should",
    "required", "excluding", "including", "must not", "shouldn't",
    "between", "up to", "within", "not exceeding",
]
# longest phrases first so multi-word phrases match before their
# single-word substrings do
CONSTRAINT_PHRASES.sort(key=lambda p: -len(p.split()))

_NUMBER_RE = re.compile(r"^\$?\d[\d,]*\.?\d*%?$")


def score_constraint(tokens: List[WordToken], text: str) -> List[float]:
    lower_text = text.lower()
    scores = [0.0] * len(tokens)

    for phrase in CONSTRAINT_PHRASES:
        for m in re.finditer(re.escape(phrase), lower_text):
            s, e = m.start(), m.end()
            for t in tokens:
                if t.start_char >= s and t.end_char <= e:
                    scores[t.index] = max(scores[t.index], 1.0)

    for i, t in enumerate(tokens):
        if _NUMBER_RE.match(t.text):
            window = tokens[max(0, i - 2):i]
            near_constraint = any(scores[w.index] > 0 for w in window)
            is_quant_entity = t.ent_type in ("MONEY", "PERCENT", "CARDINAL", "QUANTITY")
            if near_constraint or is_quant_entity:
                scores[t.index] = max(scores[t.index], 0.9)
            else:
                scores[t.index] = max(scores[t.index], 0.5)

    return scores
