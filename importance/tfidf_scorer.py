"""
Signal A: statistical informativeness (TF-IDF style).

True TF-IDF needs a corpus to estimate document frequency. Since we
are scoring a single incoming prompt with no corpus of "other prompts"
to compare against, we approximate IDF using general-English word
frequency statistics from the `wordfreq` library (a Zipf frequency
scale, roughly 1 = extremely rare, 7 = extremely common). This is a
standard, well-understood substitute for corpus-derived IDF, and it is
explicitly documented here as an approximation - not "the true
importance of a token" (see README, section "scientific caveats").

TF is computed within the current prompt only (a word repeated in the
prompt gets a small boost).
"""
from typing import List
from wordfreq import zipf_frequency
from tokenizer.tokenizer import WordToken


def _idf_approx(word: str) -> float:
    z = zipf_frequency(word.lower(), "en")
    if z == 0.0:
        # Not in the frequency list at all -> treat as maximally rare
        # (often a proper noun, technical term, or typo).
        return 1.0
    return max(0.0, min(1.0, (7.0 - z) / 7.0))


def score_tfidf(tokens: List[WordToken]) -> List[float]:
    words = [t.text.lower() for t in tokens if not (t.is_punct or t.is_space)]
    n = max(1, len(words))
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    scores = []
    for t in tokens:
        if t.is_punct or t.is_space:
            scores.append(0.0)
            continue
        w = t.text.lower()
        tf = counts.get(w, 1) / n
        idf = _idf_approx(w)
        scores.append(tf * idf)
    return scores
