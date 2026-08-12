"""
Signal C: named-entity importance.

Uses spaCy's statistical NER model. Different entity types get
different base weights because, e.g., a MONEY or DATE value is almost
always load-bearing for a query's meaning, while a WORK_OF_ART entity
is somewhat less likely to be critical on average. These per-type
weights are themselves configurable (ENTITY_TYPE_WEIGHTS below) rather
than a single flat "is an entity" flag.
"""
from typing import List
from tokenizer.tokenizer import WordToken

ENTITY_TYPE_WEIGHTS = {
    "MONEY": 1.0, "PERCENT": 1.0, "DATE": 0.95, "TIME": 0.9,
    "CARDINAL": 0.85, "QUANTITY": 0.9, "ORDINAL": 0.6,
    "PERSON": 0.9, "ORG": 0.9, "GPE": 0.85, "LOC": 0.8,
    "PRODUCT": 0.9, "EVENT": 0.7, "WORK_OF_ART": 0.6,
    "LAW": 0.8, "LANGUAGE": 0.6, "NORP": 0.6, "FAC": 0.6,
}
DEFAULT_ENTITY_WEIGHT = 0.7


def score_entities(tokens: List[WordToken]) -> List[float]:
    scores = []
    for t in tokens:
        if t.ent_type:
            scores.append(ENTITY_TYPE_WEIGHTS.get(t.ent_type, DEFAULT_ENTITY_WEIGHT))
        else:
            scores.append(0.0)
    return scores
