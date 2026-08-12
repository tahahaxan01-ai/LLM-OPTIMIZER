"""Normalization utilities shared by every signal scorer."""
from typing import List


def min_max_normalize(values: List[float]) -> List[float]:
    """Rescale a list of raw signal scores to the 0.0-1.0 range.

    If every value is identical, the signal carries no discriminating
    information for this prompt (e.g. no entities were found at all,
    so every raw entity score is 0). In that case we return 0.0 for
    everyone rather than an arbitrary constant like 0.5, so an
    uninformative signal cannot silently inflate every token's final
    score.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
