"""
Signal D: technical / domain-specific term detection.

Robust technical-term detection is an open problem, so this prototype
combines several *weak, independent* heuristics rather than relying on
a single hardcoded list (a list alone can never generalize to a brand
new library/framework name that didn't exist when the list was
written):

  1. Seed vocabulary hit (small, explicitly weak signal - weight 0.35)
  2. Acronym / initialism shape, e.g. "JWT", "RAG", "API" (weight 0.25)
  3. CamelCase / mixed alnum shape, e.g. "FastAPI", "PostgreSQL",
     "GPT4" (weight 0.20)
  4. Statistical rarity (wordfreq) for capitalized, non-sentence-
     initial tokens that are NOT already flagged as a named entity -
     i.e. "unusual capitalized words that aren't people/places" often
     turn out to be library/product/technology names (weight 0.20)

This keeps the detector swappable: `embedding_similarity_hook` is an
explicit extension point for later replacing/augmenting these
heuristics with a similarity-to-technical-centroid score from
embeddings, without changing the scorer's public interface.
"""
import re
from typing import List, Optional, Callable
from wordfreq import zipf_frequency
from tokenizer.tokenizer import WordToken

SEED_TECH_TERMS = {
    "api", "sdk", "jwt", "oauth", "rest", "graphql", "grpc", "http", "https",
    "json", "xml", "yaml", "sql", "nosql", "orm", "cli", "gui", "ui", "ux",
    "fastapi", "flask", "django", "react", "vue", "angular", "node", "npm",
    "pytorch", "tensorflow", "keras", "sklearn", "pandas", "numpy",
    "transformer", "rag", "llm", "langchain", "llama", "gpt", "bert",
    "postgresql", "mysql", "mongodb", "redis", "kubernetes", "docker",
    "aws", "azure", "gcp", "ci", "cd", "git", "github", "gitlab",
    "microservice", "webhook", "endpoint", "middleware", "async", "await",
    "regex", "token", "embedding", "vector", "database", "backend", "frontend",
}

_ACRONYM_RE = re.compile(r"^[A-Z]{2,6}$")
_CAMEL_RE = re.compile(r"^[A-Za-z]*[a-z][A-Z][A-Za-z]*$")
_ALNUM_MIX_RE = re.compile(r"^[A-Za-z]+\d+[A-Za-z0-9]*$")


def embedding_similarity_hook(word: str, encode_fn: Optional[Callable] = None) -> float:
    """Extension point: if an embedding encoder is supplied, score
    `word` by its cosine similarity to a centroid embedding of
    SEED_TECH_TERMS. Returns 0.0 with no encoder, so this stays
    optional and cheap by default."""
    if encode_fn is None:
        return 0.0
    import numpy as np
    ref_vecs = encode_fn(list(SEED_TECH_TERMS))
    centroid = ref_vecs.mean(axis=0)
    word_vec = encode_fn([word])[0]
    denom = np.linalg.norm(centroid) * np.linalg.norm(word_vec)
    if denom < 1e-9:
        return 0.0
    sim = float(np.dot(centroid, word_vec) / denom)
    return max(0.0, sim)


def score_technical(tokens: List[WordToken], encode_fn: Optional[Callable] = None) -> List[float]:
    scores = []
    for i, t in enumerate(tokens):
        if t.is_punct or t.is_space:
            scores.append(0.0)
            continue
        text = t.text
        lower = text.lower()

        seed_hit = 1.0 if lower in SEED_TECH_TERMS else 0.0
        acronym_hit = 1.0 if _ACRONYM_RE.match(text) else 0.0
        camel_hit = 1.0 if (_CAMEL_RE.match(text) or _ALNUM_MIX_RE.match(text)) else 0.0

        rarity_hit = 0.0
        if i > 0 and text[:1].isupper() and not t.ent_type:
            z = zipf_frequency(lower, "en")
            if z == 0.0 or z < 3.0:
                rarity_hit = 1.0

        emb_hit = embedding_similarity_hook(lower, encode_fn) if encode_fn else 0.0

        score = (0.35 * seed_hit + 0.25 * acronym_hit +
                 0.20 * camel_hit + 0.20 * rarity_hit)
        if encode_fn:
            score = 0.7 * score + 0.3 * emb_hit
        scores.append(min(1.0, score))
    return scores
