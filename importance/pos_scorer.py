"""
Signal E: part-of-speech based content-vs-function word estimate.

This is an engineered prior, not a learned weighting: content-bearing
POS categories (nouns, proper nouns, verbs, numbers, adjectives) get a
higher base score than function words (determiners, adpositions,
conjunctions). Function words are deliberately NOT forced to 0 - e.g.
"without" is tagged ADP/SCONJ but must survive compression. That's why
negation and constraint detection (Signals B/H) exist as separate,
higher-weighted overrides rather than folding everything into POS
alone.
"""
from typing import List
from tokenizer.tokenizer import WordToken

POS_BASE_SCORES = {
    "PROPN": 0.90, "NOUN": 0.80, "VERB": 0.75, "NUM": 0.75,
    "ADJ": 0.60, "ADV": 0.40, "AUX": 0.30, "PRON": 0.30,
    "PART": 0.30, "ADP": 0.20, "SCONJ": 0.25, "CCONJ": 0.20,
    "DET": 0.10, "INTJ": 0.15, "SYM": 0.30, "X": 0.30,
    "PUNCT": 0.0, "SPACE": 0.0,
}
DEFAULT_POS_SCORE = 0.35


def score_pos(tokens: List[WordToken]) -> List[float]:
    return [POS_BASE_SCORES.get(t.pos, DEFAULT_POS_SCORE) for t in tokens]
