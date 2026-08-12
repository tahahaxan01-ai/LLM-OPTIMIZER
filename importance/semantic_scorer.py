"""
Signal F: ablation-based semantic importance.

Method:
  1. Embed the full prompt.
  2. For each candidate span (token or phrase), embed the prompt with
     that span removed.
  3. importance = 1 - cosine_similarity(original, modified)

Efficiency: naive token-by-token ablation needs N+1 forward passes for
an N-token prompt. Two optimizations are used:

  * Batching - every ablated variant is embedded in a single encode()
    call (SentenceTransformer batches internally) instead of one call
    per token.
  * Span-level ablation for long prompts - above
    config.semantic_token_ablation_limit real tokens, we ablate spaCy
    noun chunks plus any remaining single tokens not covered by a
    chunk, instead of every individual token. This keeps the number of
    forward passes roughly proportional to the number of "meaningful
    spans" rather than raw token count, while still covering the whole
    prompt.

If sentence-transformers (or its model weights) is unavailable - e.g.
no network access to download the model - a degraded fallback is used:
a simple bag-of-words count-vector "embedding" with cosine similarity.
This fallback is explicitly logged/returned as such; it is NOT a
substitute for real contextual embeddings, it only exists so the rest
of the pipeline stays runnable offline (see README limitations).
"""
from typing import List, Tuple
import numpy as np
from tokenizer.tokenizer import WordToken

_MODEL = None
_MODEL_FAILED = False
LAST_RUN_USED_FALLBACK = False


def _get_model(model_name: str):
    global _MODEL, _MODEL_FAILED
    if _MODEL is not None or _MODEL_FAILED:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(model_name)
    except Exception:
        _MODEL_FAILED = True
        _MODEL = None
    return _MODEL


def _bow_embed(texts: List[str]) -> np.ndarray:
    vocab = {}
    for t in texts:
        for w in t.lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
    mat = np.zeros((len(texts), max(1, len(vocab))))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            mat[i, vocab[w]] += 1.0
    return mat


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 1.0
    return float(np.dot(a, b) / denom)


def _get_spans(tokens: List[WordToken], doc, limit: int) -> List[Tuple[int, int]]:
    """Return list of (start_char, end_char) spans to ablate."""
    real = [t for t in tokens if not (t.is_punct or t.is_space)]
    if len(real) <= limit:
        return [(t.start_char, t.end_char) for t in real]

    spans = []
    covered = set()
    for chunk in doc.noun_chunks:
        spans.append((chunk.start_char, chunk.end_char))
        covered.update(range(chunk.start, chunk.end))
    for i, t in enumerate(doc):
        if i not in covered and not (t.is_punct or t.is_space):
            spans.append((t.idx, t.idx + len(t.text)))
    return spans


def score_semantic(tokens: List[WordToken], text: str, doc, config) -> List[float]:
    global LAST_RUN_USED_FALLBACK
    model = _get_model(config.semantic_model_name)
    use_real_model = model is not None
    LAST_RUN_USED_FALLBACK = not use_real_model

    spans = _get_spans(tokens, doc, config.semantic_token_ablation_limit)
    if not spans:
        return [0.0 for _ in tokens]

    variants = [text]
    for (s, e) in spans:
        variant = (text[:s] + text[e:]).strip()
        variant = " ".join(variant.split())
        variants.append(variant if variant else " ")

    if use_real_model:
        embeddings = model.encode(variants, batch_size=min(32, len(variants)))
    else:
        embeddings = _bow_embed(variants)

    original_vec = embeddings[0]
    span_importance = []
    for i in range(1, len(variants)):
        sim = _cosine(original_vec, embeddings[i])
        span_importance.append(max(0.0, 1.0 - sim))

    # distribute span importance back onto individual word tokens
    token_scores = [0.0] * len(tokens)
    for (s, e), imp in zip(spans, span_importance):
        for t in tokens:
            if t.start_char >= s and t.end_char <= e:
                token_scores[t.index] = max(token_scores[t.index], imp)
    return token_scores
