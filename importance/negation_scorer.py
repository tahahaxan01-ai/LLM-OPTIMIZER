"""
Signal B (negation half) + critical protection input: negation and
critical logical-connective detection.

Combines a curated negation/connective word list (fast, catches
contractions like "don't"/"shouldn't" and connectives like "rather")
with spaCy's dependency-parse relations - dep_ == "neg" for negation,
and dep_ == "mark" for subordinating conjunctions ("unless",
"although", "provided", etc.) that the word list alone might miss.

Negated tokens AND critical logical connectives get the maximum raw
signal score. Dropping a negator can invert a sentence's meaning
entirely - "Do not delete the database." collapsing into "Delete the
database." if "not" is removed. The same is true for exception/
contrast/joint-condition connectives: dropping "unless" from
"excluding X unless Y" silently removes the exception clause, and
dropping "rather" from "escalate rather than reject" removes the
contrast between two alternatives, leaving an ambiguous instruction.

The stopword half of Signal B (generic filler words like "please",
"kindly", "could") is handled implicitly: it is exactly what the POS
scorer (function-word categories) and the TF-IDF scorer (very common,
low-IDF words) already push toward low scores, so it is not
duplicated here.
"""
from typing import List
from tokenizer.tokenizer import WordToken

NEGATION_WORDS = {
    "not", "no", "never", "without", "cannot", "can't", "don't",
    "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "mustn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "hadn't", "neither", "nor", "none", "nothing", "nobody", "nowhere",
}

# Exception / conditional-exclusion connectives: drop these and an
# "excluding X unless Y" carve-out silently becomes an unconditional
# exclusion.
EXCEPTION_WORDS = {
    "unless", "except", "excepting", "excluding", "besides", "barring",
    "provided", "otherwise", "notwithstanding",
}

# Contrastive "pick-this-not-that" connectives: drop these and two
# alternatives collapse into an ambiguous list instead of a choice.
CONTRAST_WORDS = {
    "rather", "instead", "whereas", "although", "though", "despite",
    "regardless", "nonetheless", "nevertheless",
}

# Joint-condition adverbs: drop these and an "A AND B together" test
# reads as two independent, optional checks instead of one combined one.
JOINT_CONDITION_WORDS = {
    "simultaneously", "concurrently", "jointly", "together",
}

CRITICAL_CONNECTIVES = EXCEPTION_WORDS | CONTRAST_WORDS | JOINT_CONDITION_WORDS

# Comparative/threshold prepositions: only protected when adjacent to
# a numeric/date/money entity, since "under", "before", "over" etc.
# are extremely common in non-threshold senses and blanket-protecting
# them would gut compression. Adjacent to a number, though, they set
# the direction of a threshold ("under $50" vs "over $50") and
# dropping them makes the threshold ambiguous or backwards.
THRESHOLD_PREPOSITIONS = {
    "under", "over", "above", "below", "within", "exceeding",
}
THRESHOLD_ADJACENT_ENTITIES = {"MONEY", "QUANTITY", "PERCENT", "DATE", "TIME", "CARDINAL"}


def _is_threshold_preposition(t: WordToken, tokens: List[WordToken]) -> bool:
    if t.text.lower() not in THRESHOLD_PREPOSITIONS:
        return False
    for neighbor in tokens:
        if abs(neighbor.index - t.index) == 1 and neighbor.ent_type in THRESHOLD_ADJACENT_ENTITIES:
            return True
    return False


def score_negation(tokens: List[WordToken]) -> List[float]:
    scores = [0.0] * len(tokens)
    for t in tokens:
        text_lower = t.text.lower()
        if (
            text_lower in NEGATION_WORDS
            or t.dep == "neg"
            or text_lower in CRITICAL_CONNECTIVES
            or t.dep == "mark"
            or _is_threshold_preposition(t, tokens)
        ):
            scores[t.index] = 1.0
    return scores