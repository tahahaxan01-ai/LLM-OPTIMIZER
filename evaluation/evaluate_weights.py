"""
Optional weight-search module.

We are explicitly NOT searching for weights that maximize token
removal - that objective is trivially "maximized" by removing every
token. Instead we optimize a composite objective:

    Objective = c1 * TokenReductionScore
              + c2 * SemanticPreservationScore
              + c3 * InformationPreservationScore

For a candidate weight configuration and a removal threshold:

  * TokenReductionScore - fraction of tokens that would fall below the
    threshold (i.e. become eligible for removal in the next
    compression stage). Higher = more compression opportunity found.
  * SemanticPreservationScore - cosine similarity between the
    embedding of the original prompt and the embedding of the prompt
    with below-threshold, non-protected tokens removed. Higher =
    meaning preserved.
  * InformationPreservationScore - fraction of PROTECTED tokens
    (negations, entities, numbers/dates, constraints) that remain
    after the simulated removal. This is checked explicitly and
    separately from semantic similarity, because embedding similarity
    can stay high even when a single critical number or negation is
    dropped.

Default coefficients favor preservation over raw reduction
(c1=0.3, c2=0.35, c3=0.35), since a compression module that reads
fluently but silently drops a "not" or a dollar figure is worse than
one that compresses less aggressively.
"""
import itertools
import os
import random
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importance.scorer import TokenImportanceScorer
from importance.config import ScoringConfig
from importance.semantic_scorer import _get_model, _bow_embed, _cosine


def _simulate_compression(text: str, scorer: TokenImportanceScorer, threshold: float):
    results = scorer.score(text)
    kept = [r for r in results if r.final_score >= threshold or r.protected]
    kept_text = " ".join(r.token for r in kept)
    protected_total = [r for r in results if r.protected]
    kept_indices = {r.index for r in kept}
    protected_kept = [r for r in protected_total if r.index in kept_indices]
    reduction = 1.0 - (len(kept) / max(1, len(results)))
    info_preservation = (len(protected_kept) / len(protected_total)) if protected_total else 1.0
    return kept_text, reduction, info_preservation


def _semantic_similarity(original: str, compressed: str, config: ScoringConfig) -> float:
    model = _get_model(config.semantic_model_name)
    if model is not None:
        vecs = model.encode([original, compressed])
    else:
        vecs = _bow_embed([original, compressed])
    return _cosine(vecs[0], vecs[1])


def evaluate_config(prompts: List[str], config: ScoringConfig, threshold: float,
                     coeffs=(0.3, 0.35, 0.35)) -> float:
    scorer = TokenImportanceScorer(config=config)
    c1, c2, c3 = coeffs
    total = 0.0
    for p in prompts:
        compressed_text, reduction, info_preservation = _simulate_compression(p, scorer, threshold)
        sem_sim = _semantic_similarity(p, compressed_text, config)
        total += c1 * reduction + c2 * sem_sim + c3 * info_preservation
    return total / max(1, len(prompts))


def grid_search(prompts: List[str], weight_grid: Dict[str, List[float]],
                 threshold: float = 0.4, coeffs=(0.3, 0.35, 0.35)) -> Tuple[Dict, float]:
    keys = list(weight_grid.keys())
    best_score, best_weights = -1.0, None
    for combo in itertools.product(*[weight_grid[k] for k in keys]):
        base = ScoringConfig().weights.copy()
        for k, v in zip(keys, combo):
            base[k] = v
        cfg = ScoringConfig(weights=base)
        score = evaluate_config(prompts, cfg, threshold, coeffs)
        if score > best_score:
            best_score, best_weights = score, base
    return best_weights, best_score


def random_search(prompts: List[str], n_trials: int = 20, threshold: float = 0.4,
                   coeffs=(0.3, 0.35, 0.35), seed: int = 42) -> Tuple[Dict, float]:
    rng = random.Random(seed)
    keys = list(ScoringConfig().weights.keys())
    best_score, best_weights = -1.0, None
    for _ in range(n_trials):
        raw = {k: rng.uniform(0.02, 0.3) for k in keys}
        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}
        cfg = ScoringConfig(weights=weights)
        score = evaluate_config(prompts, cfg, threshold, coeffs)
        if score > best_score:
            best_score, best_weights = score, weights
    return best_weights, best_score


# Bayesian optimization (e.g. via scikit-optimize or Optuna) is a
# reasonable next step once grid/random search establish which weight
# ranges are worth exploring more finely. It is intentionally not
# implemented here to avoid adding a heavyweight dependency for
# marginal benefit at this prototype stage - grid/random search over 8
# weights with a handful of test prompts is already cheap.

if __name__ == "__main__":
    from tests.test_importance import TEST_PROMPTS

    grid = {
        "semantic": [0.15, 0.20, 0.30],
        "entity": [0.10, 0.15, 0.20],
    }
    best_w, best_s = grid_search(TEST_PROMPTS[:6], grid)
    print("Best weights (partial grid):", best_w)
    print("Best objective score:", best_s)
