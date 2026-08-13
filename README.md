# LLM-OPTIMIZER

**A two-stage pipeline for reducing LLM inference cost: intelligent prompt compression followed by task-aware model routing.**

## Overview

LLM-OPTIMIZER addresses a core inefficiency in LLM-powered applications: sending long, verbose prompts to expensive, high-capability models regardless of whether the task actually requires that level of capability. This project tackles the problem from two angles that work together as a single pipeline —

1. **Compress** the prompt by removing low-signal tokens while guaranteeing that words critical to meaning (negations, constraints, exceptions) are never dropped.
2. **Route** the compressed prompt to the cheapest model tier capable of handling it, escalating to stronger models only when the task's complexity genuinely demands it.

The result is a system designed to cut token volume and cost on high-traffic LLM applications without degrading instruction fidelity or output quality.

---

## Stage 1: Token Importance Scoring & Compression

At the core of the compression stage is a multi-signal scoring engine that evaluates every token in a prompt across **8 independent signals** and combines them into a single, explainable importance score:

| Signal | What it measures |
|---|---|
| TF-IDF | Statistical rarity/informativeness of a word |
| Named Entity Recognition | Whether a token is a person, organization, date, money value, etc. |
| Technical Term Detection | Domain-specific/technical vocabulary |
| Part-of-Speech | Content-bearing vs. function words |
| Semantic Ablation | How much meaning shifts when a token is removed |
| Context Centrality | How structurally central a token is within the sentence |
| Constraint Detection | Instruction/rule-bearing phrases |
| Negation & Critical Connectives | Words that flip or gate meaning (`not`, `unless`, `rather than`, `simultaneously`) |

### Critical-token protection

A key design decision: certain tokens are **never** eligible for removal, regardless of their raw score. This includes:
- Negation words (`not`, `never`, `without`) — dropping these can invert a sentence's meaning entirely (e.g. *"Do not delete the database"* → *"Delete the database"*).
- Exception/conditional connectives (`unless`, `except`, `provided that`) — these gate an instruction's scope; dropping them silently removes exception clauses.
- Contrastive connectives (`rather than`, `instead of`) — these distinguish between two alternatives; dropping them collapses a choice into an ambiguous list.
- Joint-condition adverbs (`simultaneously`, `jointly`) — these signal that multiple conditions must hold *together*, not independently.
- Threshold prepositions adjacent to numeric/date/money entities (`under $50`, `within 6 months`) — these set the direction of a threshold.
- Fixed multi-word phrases (`rather than`, `in which case`) — protected as a unit, since word-by-word scoring alone can miss phrases where only one word individually scores high.
- Named entities and numeric/date values.

This protection layer was iteratively refined by stress-testing the scorer against long, constraint-heavy prompts and identifying specific failure modes where naive compression corrupted meaning — a process documented and validated against a 50-prompt test suite spanning fraud detection, legal, technical, and multi-clause conditional prompts.

Compression uses a **percentile-based cutoff** (keep the top N% of non-protected tokens by score) rather than a fixed absolute threshold, since a fixed cutoff becomes effectively stricter as prompt length increases and scores dilute across more competing tokens.

**Result:** 30–40% token reduction on long, complex prompts while preserving every instruction, rule, and exception the original prompt contained.

---

## Stage 2: Task-Aware Model Routing

The second stage takes the compressed prompt (and the token-level scores already computed by Stage 1) and decides which model should handle it.

### Task classification
The compressed prompt is classified into one of several task categories (code generation, reasoning/math, data analysis, creative writing, structured extraction, multi-step/agentic workflows, conversational Q&A) using keyword and pattern matching.

### Complexity scoring
Rather than routing on category alone, a **complexity score** is computed directly from Stage 1's signals — the density of protected constraint tokens, negation tokens, technical terms, and prompt length. This means a short, simple prompt stays on a lightweight model even if its category would normally suggest otherwise, while a prompt dense with rules and exceptions gets escalated even if its category is typically "cheap."

### Routing logic
- Each task category has a **baseline model tier** (fast / balanced / max) chosen for typical cost-efficiency.
- The router **escalates** — never downgrades — above that baseline when complexity signals indicate the task is precision-critical (high constraint/negation density, low classification confidence, or unusually long prompts).
- Very high complexity scores jump directly to the top tier, rather than climbing incrementally, since a prompt that dense with protected tokens isn't a candidate for a partial upgrade.

**Result:** routine, high-volume tasks are handled by cost-efficient models, while nothing precision-critical is ever silently routed to an underpowered model.

---

## Architecture

Raw Prompt
│
▼
┌─────────────────────────┐
│ Token Importance Scorer │ (8 signals + protection layer)
└─────────────────────────┘
│
▼
Compressed Prompt + Per-Token Scores
│
▼
┌─────────────────────────┐
│ Task Classifier │ (category + complexity_score)
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│ Model Router │ (tier selection + escalation)
└─────────────────────────┘
│
▼
Selected Model


---

## Tech Stack

- **Python 3.12**
- **spaCy** — dependency parsing, POS tagging, NER
- **Custom scoring engine** — TF-IDF, semantic ablation, rule-based constraint/negation detection

---

## Status

Actively in development as part of a fintech internship project. Prompt compression stage is functional and validated; model routing stage is under active development by a collaborating teammate.

## Future Work

- Abstractive compression pass (rewriting rather than pure extraction) for further reduction beyond what token-dropping alone can achieve
- Clause-level (dependency-subtree) pruning instead of word-level dropping, for cleaner grammatical output
- Filler-phrase collapsing pre-pass
- Confidence-calibrated routing using real usage data
