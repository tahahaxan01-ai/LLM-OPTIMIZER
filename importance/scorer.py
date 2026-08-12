"""
Top-level orchestrator: TokenImportanceScorer.

Combines all eight signals into a single, explainable final importance
score per word/span, applies critical-token protection, and can map
the result onto actual LLM subword tokens via the tokenizer module.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable

from tokenizer.tokenizer import get_nlp, tokenize_words, WordToken, align_to_llm_tokens
from importance.config import ScoringConfig, DEFAULT_CONFIG
from importance.normalizer import min_max_normalize
from importance.tfidf_scorer import score_tfidf
from importance.entity_scorer import score_entities
from importance.technical_scorer import score_technical
from importance.pos_scorer import score_pos
from importance.semantic_scorer import score_semantic
from importance.context_scorer import score_context
from importance.constraint_scorer import score_constraint
from importance.negation_scorer import score_negation


@dataclass
class TokenScoreResult:
    token: str
    index: int
    tfidf_score: float
    entity_score: float
    technical_score: float
    pos_score: float
    semantic_score: float
    context_score: float
    constraint_score: float
    negation_score: float
    raw_combined_score: float
    protected: bool
    protection_reason: Optional[str]
    final_score: float
    rank_label: str
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TokenImportanceScorer:
    def __init__(self, config: ScoringConfig = None, technical_encode_fn: Optional[Callable] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        self.technical_encode_fn = technical_encode_fn
        self.nlp = get_nlp()

    def score(self, text: str) -> List[TokenScoreResult]:
        doc = self.nlp(text)
        tokens = tokenize_words(text)

        raw_tfidf = score_tfidf(tokens)
        raw_entity = score_entities(tokens)
        raw_technical = score_technical(tokens, self.technical_encode_fn)
        raw_pos = score_pos(tokens)
        raw_semantic = score_semantic(tokens, text, doc, self.config)
        raw_context = score_context(tokens, doc)
        raw_constraint = score_constraint(tokens, text)
        raw_negation = score_negation(tokens)

        norm_tfidf = min_max_normalize(raw_tfidf)
        norm_entity = min_max_normalize(raw_entity)
        norm_technical = min_max_normalize(raw_technical)
        norm_pos = min_max_normalize(raw_pos)
        norm_semantic = min_max_normalize(raw_semantic)
        norm_context = min_max_normalize(raw_context)
        norm_constraint = min_max_normalize(raw_constraint)
        norm_negation = min_max_normalize(raw_negation)

        w = self.config.weights
        total_weight = sum(w.values())
        results = []
        for i, t in enumerate(tokens):
            combined = (
                w["tfidf"] * norm_tfidf[i] +
                w["entity"] * norm_entity[i] +
                w["technical"] * norm_technical[i] +
                w["pos"] * norm_pos[i] +
                w["semantic"] * norm_semantic[i] +
                w["context"] * norm_context[i] +
                w["constraint"] * norm_constraint[i] +
                w["negation"] * norm_negation[i]
            )
            combined = combined / total_weight if total_weight > 0 else combined

            protected, reason = self._protection_reason(t, raw_negation[i], raw_constraint[i])
            final = combined
            if protected:
                final = combined + (1.0 - combined) * self.config.protection_strength
            final = max(0.0, min(1.0, final))

            reasons = self._build_reasons(
                norm_tfidf[i], norm_entity[i], norm_technical[i], norm_pos[i],
                norm_semantic[i], norm_context[i], norm_constraint[i], norm_negation[i],
            )

            results.append(TokenScoreResult(
                token=t.text, index=t.index,
                tfidf_score=round(norm_tfidf[i], 3),
                entity_score=round(norm_entity[i], 3),
                technical_score=round(norm_technical[i], 3),
                pos_score=round(norm_pos[i], 3),
                semantic_score=round(norm_semantic[i], 3),
                context_score=round(norm_context[i], 3),
                constraint_score=round(norm_constraint[i], 3),
                negation_score=round(norm_negation[i], 3),
                raw_combined_score=round(combined, 3),
                protected=protected,
                protection_reason=reason,
                final_score=round(final, 3),
                rank_label=self._label(final),
                reasons=reasons,
            ))
        return results

    def score_with_llm_tokens(self, text: str):
        """Returns (word_level_results, llm_subword_tokens_with_scores)."""
        word_results = self.score(text)
        word_tokens = tokenize_words(text)
        scores_only = [r.final_score for r in word_results]
        subwords = align_to_llm_tokens(word_tokens, scores_only, text)
        return word_results, subwords

    def _protection_reason(self, t: WordToken, neg_raw: float, constraint_raw: float):
        if neg_raw >= 1.0:
            return True, "negation"
        if constraint_raw >= 1.0:
            return True, "constraint"
        if t.ent_type in ("DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "QUANTITY"):
            return True, "number_or_date_entity"
        if t.ent_type in ("PERSON", "ORG", "GPE", "PRODUCT", "LOC"):
            return True, "named_entity"
        return False, None

    def _label(self, score: float) -> str:
        if score >= self.config.high_threshold:
            return "HIGH"
        if score >= self.config.medium_threshold:
            return "MEDIUM"
        return "LOW"

    def _build_reasons(self, tfidf, entity, technical, pos, semantic, context, constraint, negation):
        reasons = []
        if negation >= 0.9:
            reasons.append("negation word (protected - flips meaning if removed)")
        if constraint >= 0.9:
            reasons.append("constraint/instruction phrase")
        if entity >= 0.7:
            reasons.append("named entity")
        if technical >= 0.6:
            reasons.append("technical term")
        if semantic >= 0.6:
            reasons.append("large semantic shift when ablated")
        if tfidf >= 0.6:
            reasons.append("statistically rare/informative word")
        if pos >= 0.7:
            reasons.append("content-bearing part of speech")
        if context >= 0.6:
            reasons.append("structurally central in sentence")
        if not reasons:
            if pos <= 0.2 and tfidf <= 0.2:
                reasons.append("grammatical/function word")
            else:
                reasons.append("low signal across measured features")
        return reasons
