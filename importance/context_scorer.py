"""
Signal G: contextual / structural importance.

A full implementation would compare a token's contextual embedding
(e.g. from BERT) against its embedding in isolation, to measure how
much its meaning depends on surrounding words ("bank" in "river bank"
vs "bank account"). That requires a second heavyweight model pass per
token, which is expensive to run for every token in every prompt.

As a lightweight proxy for this prototype, we use the token's position
in the dependency parse tree: tokens that many other tokens depend on
(large subtree, many children) are structurally more central to the
sentence than leaves are. This is cheap - it reuses the parse spaCy
already computed during tokenization - and correlates reasonably well
with "removing this breaks the sentence's structure."

Upgrade path: once the semantic scorer's embedding model is already
loaded (see semantic_scorer.py), replace `score_context` with a true
in-context-vs-isolated embedding comparison, since the marginal cost
of one more embedding call per token becomes acceptable when the
infrastructure is already in memory.
"""
from typing import List
from tokenizer.tokenizer import WordToken


def score_context(tokens: List[WordToken], doc) -> List[float]:
    # tokenize_words() (tokenizer.py) drops spaCy's whitespace tokens, so
    # `tokens` and `doc` can have different lengths/indices. Rebuild the
    # same non-space ordering here rather than trusting doc token `.i`
    # positions directly, so scores line up with `tokens` by position.
    real_doc_tokens = [t for t in doc if not t.is_space]
    n = max(1, len(real_doc_tokens))
    scores = [0.0] * len(tokens)
    for idx, t in enumerate(real_doc_tokens):
        subtree_size = sum(1 for st in t.subtree if not st.is_space)
        n_children = sum(1 for c in t.children if not c.is_space)
        centrality = (0.6 * (subtree_size / n)) + (0.4 * min(1.0, n_children / 3.0))
        if idx < len(scores):
            scores[idx] = min(1.0, centrality)
    return scores
