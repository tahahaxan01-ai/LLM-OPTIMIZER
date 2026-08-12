"""
Configuration for the LLMify token-importance scoring system.

All weights, thresholds, and protection parameters live here so that
experiments can be run by editing this file (or by constructing a
ScoringConfig with overrides) instead of hunting through the codebase.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ScoringConfig:
    # --- Signal weights --------------------------------------------------
    # These are hyperparameters, not ground truth. See README.md,
    # section "Why these initial weights", for the reasoning behind the
    # starting values, and evaluation/evaluate_weights.py for a way to
    # search over alternatives instead of trusting these blindly.
    weights: Dict[str, float] = field(default_factory=lambda: {
        "tfidf": 0.15,
        "entity": 0.15,
        "technical": 0.10,
        "pos": 0.10,
        "semantic": 0.20,
        "context": 0.08,
        "constraint": 0.12,
        "negation": 0.10,
    })

    # --- Ranking thresholds ------------------------------------------------
    high_threshold: float = 0.70
    medium_threshold: float = 0.40

    # --- Critical-token protection ------------------------------------------
    # final = combined + (1 - combined) * protection_strength
    # This pulls protected tokens toward 1.0 without hard-clipping them
    # to exactly 1.0, so relative ordering among protected tokens is
    # preserved and the effect stays a configurable "nudge" instead of
    # an unconditional override (see section 6 of the spec).
    protection_strength: float = 0.85

    # --- Semantic scorer behaviour -------------------------------------------
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Prompts with more real (non-punct) tokens than this fall back to
    # span-level (noun-chunk) ablation instead of token-by-token
    # ablation, to keep the number of embedding calls bounded on long
    # prompts. See importance/semantic_scorer.py.
    semantic_token_ablation_limit: int = 25

    def validate(self):
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Sum of signal weights must be > 0")


DEFAULT_CONFIG = ScoringConfig()
