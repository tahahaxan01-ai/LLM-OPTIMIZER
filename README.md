# LLMify — Token-Level Importance Scoring Module

This is the **importance-scoring stage** of the Query Compression component.
It does not compress, delete tokens, or rewrite prompts. It only measures,
for every word/span in a prompt, how important that token is — and exposes
*why* it received that score. Compression is a later stage that will consume
these scores.

## 1. Recommended technology (chosen before writing code)

| Need | Choice | Why |
|---|---|---|
| Word tokenization, POS, dependency parse, NER | **spaCy** (`en_core_web_sm`) | One lightweight model gives POS + dependency tree + NER in a single pass — no need for three separate libraries. `en_core_web_sm` is ~13MB, fast enough for interactive use. |
| Statistical rarity (TF-IDF-like signal) | **`wordfreq`** | True TF-IDF needs a background corpus for IDF. `wordfreq` gives calibrated general-English frequency (Zipf scale) without having to ship/maintain a corpus. Combined with in-prompt term frequency, it approximates TF-IDF. |
| Semantic importance | **sentence-transformers** (`all-MiniLM-L6-v2`) | Small (~80MB), fast, good general-purpose sentence embeddings — enough for ablation-based importance without the cost of a full LLM embedding call. |
| LLM subword tokenization | **tiktoken** | This is what actually determines an LLM's real token count/cost — needed to correctly map word-level scores onto real tokens (Section 7 of the spec). |
| Weight search | plain grid/random search (stdlib `itertools`/`random`) | 8 weights, small prompt sets — a heavyweight optimizer (e.g. Optuna) is unjustified at this stage. |

**Not used:** Hugging Face `transformers` directly (sentence-transformers already wraps a suitable model at a fraction of the complexity), spaCy's large/trf models (too heavy for a prototype), and no LLM API calls anywhere in this module — scoring must work offline and cheaply, since the whole point is to *reduce* API usage downstream.

## 2. Architecture

```
llmify/
├── importance/
│   ├── scorer.py          orchestrates all 8 signals -> final score
│   ├── tfidf_scorer.py     Signal A - statistical informativeness
│   ├── entity_scorer.py    Signal C - named entities
│   ├── technical_scorer.py Signal D - technical terms
│   ├── pos_scorer.py       Signal E - part of speech
│   ├── semantic_scorer.py  Signal F - ablation-based semantic importance
│   ├── constraint_scorer.py Signal H - constraints/instructions
│   ├── negation_scorer.py  Signal B (negation half) - protection
│   ├── context_scorer.py   Signal G - structural/contextual centrality
│   ├── normalizer.py       shared 0-1 normalization
│   └── config.py           weights, thresholds, protection strength
├── tokenizer/
│   └── tokenizer.py        spaCy word tokens + tiktoken subword alignment
├── evaluation/
│   └── evaluate_weights.py grid/random search over signal weights
├── tests/
│   └── test_importance.py  20 test prompts + sanity assertions
├── main.py                 CLI
└── requirements.txt
```

Every scorer is a standalone function taking `WordToken`s (and sometimes the
spaCy `doc` or raw text) and returning one raw score per token — independently
testable and independently replaceable.

## 3. Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 4. Usage

```bash
python main.py "Do not delete the production database."
python main.py --demo                       # runs all 20 built-in test prompts
python main.py "your prompt" --json-out out.json
python -m evaluation.evaluate_weights       # example grid search run
python tests/test_importance.py             # sanity tests
```

## 5. How a token gets its score (Section 17 of the spec)

**A. How a token receives its weight.** Every word token is scored independently
by 8 signal functions (A–H below), each producing a raw number. Each signal's
raw scores across the *current prompt* are min-max normalized to [0,1]
(`normalizer.py`) — normalization is per-prompt, not global, since "important
relative to this prompt" is what matters for compression. The 8 normalized
scores are combined with configurable weights (`config.py`) into a combined
score, then critical-token protection (below) is applied to get the final
score in [0,1].

**B. Why each signal is used.**
- *TF-IDF (statistical)* — flags words that are unusually rare/specific
  rather than generic filler, independent of grammar or meaning.
- *Entity* — named things (people, orgs, dates, money) are usually essential
  to a query's factual content.
- *Technical* — domain terminology (`FastAPI`, `JWT`) is exactly the kind of
  token that a naive compressor would wrongly discard as "just another word."
- *POS* — a cheap structural prior: nouns/verbs generally carry more content
  than determiners/prepositions.
- *Semantic (ablation)* — the only signal that directly measures "how much
  does removing this change the meaning of the whole prompt," rather than
  inferring importance from surface features.
- *Context* — captures that importance isn't purely a property of a single
  word in isolation; a word's structural role in the sentence matters too.
- *Constraint* — instruction/filter language (`greater than`, `only`,
  `excluding`) changes what a correct answer looks like, so it needs its own
  dedicated detector rather than relying on POS/TF-IDF, which would
  under-rate common words like "before" or "only."
- *Negation* — the single highest-consequence category: dropping "not" can
  invert meaning entirely, which no other signal reliably catches on its own.

**C. How the initial weights were selected.** They are engineering priors, not
fitted values (config.py):
`{tfidf: 0.15, entity: 0.15, technical: 0.10, pos: 0.10, semantic: 0.20,
context: 0.08, constraint: 0.12, negation: 0.10}`.
Semantic got the largest single weight (0.20) because it's the most direct
measurement of meaning-change, entity and TF-IDF are next because they're
reliable and cheap, and context got the smallest weight (0.08) because it's
the least direct proxy (dependency centrality standing in for true
in-context-vs-isolated embedding comparison). These are explicitly flagged
in `config.py` as hyperparameters to be revisited, not ground truth.

**D. How the weights can later be optimized.**
`evaluation/evaluate_weights.py` implements grid search and random search
over the 8 weights against a composite objective:
`Objective = c1·TokenReductionScore + c2·SemanticPreservationScore + c3·InformationPreservationScore`
computed by *simulating* a compression at a given threshold (keep tokens
above threshold or protected, drop the rest) and measuring: how much was
removed, how similar the compressed prompt's embedding is to the original,
and what fraction of protected tokens survived. The objective is explicitly
not "maximize removal" — the default coefficients (0.3 / 0.35 / 0.35)
weight preservation over reduction. Bayesian optimization (Optuna/
scikit-optimize) is a documented next step once grid/random search narrow
down promising ranges — not implemented now to avoid adding a heavy
dependency for a prototype-stage problem with only 8 parameters.

**E. How semantic importance is calculated.** Ablation: embed the full
prompt, embed the prompt with a token/span removed, importance = 1 − cosine
similarity. For efficiency, all ablated variants are embedded in **one**
batched `encode()` call rather than one call per token. For prompts longer
than `config.semantic_token_ablation_limit` (default 25 real tokens), the
scorer switches from per-token ablation to per-span ablation using spaCy
noun chunks (plus any leftover single tokens), so cost scales with the
number of meaningful phrases rather than raw token count. If the embedding
model can't be loaded (no network), a bag-of-words fallback is used and
flagged — this is a degraded approximation, not real semantic scoring; on a
machine with network access, install `sentence-transformers` and the real
model is used automatically (this sandbox couldn't download the ~80MB model
weights from Hugging Face, so the demo output below used the fallback).

**F. How critical tokens are protected.** `scorer.py::_protection_reason`
flags a token as protected if it's a negation, part of a matched constraint
phrase, or a DATE/TIME/MONEY/PERCENT/CARDINAL/QUANTITY/PERSON/ORG/GPE/
PRODUCT/LOC entity. Protected tokens get
`final = combined + (1 - combined) * protection_strength` (default strength
0.85) — a strong pull toward 1.0 that still respects relative differences
between protected tokens, rather than hard-clipping everything protected to
exactly 1.0.

**G. How word-level spans are mapped to LLM subword tokens.**
`tokenizer/tokenizer.py::align_to_llm_tokens` tokenizes the raw string with
`tiktoken` (`cl100k_base`) independently of spaCy's word tokenization, then
for each subword token finds the spaCy word token with the largest character-offset
overlap and copies that word's final score onto it. This is necessary because
`1 word ≠ 1 LLM token` — e.g. "authentication" can split into multiple BPE
pieces — and the compression stage will need per-subword-token scores to
make real token-budget decisions.

**H. How this will be used for ~30% token compression (future stage).**
The next stage will pick a score threshold (or target compression ratio),
keep every protected token regardless of score, and drop non-protected
tokens below the threshold — most likely operating on the LLM-subword-token
view produced by `align_to_llm_tokens`, since that's what actually
determines the token count sent to Gemini/other APIs.

**I. Limitations of this first version.**
- TF-IDF/IDF is approximated from general word frequency, not a real corpus
  of past prompts (would improve with logged prompt history).
- Technical-term detection is heuristic (seed list + shape rules), not a
  learned classifier — it will miss unfamiliar terms with no distinctive
  shape.
- Context scoring uses dependency-tree centrality as a proxy for true
  contextual (in-sentence-vs-isolated) meaning, which a contextual embedding
  model would capture better but at higher cost.
- Semantic ablation with the bag-of-words fallback is noticeably weaker than
  with real sentence embeddings — always prefer running with
  `sentence-transformers` installed and network access to its model weights.
- Entity/technical detection both depend on `en_core_web_sm`, a small model;
  a larger spaCy model or a technical-domain-tuned NER would catch more.

**J. Upgrading to a trained ML token-classification model.**
Every signal in this file is a hand-built feature. The natural next step is
to treat `(tfidf, entity, technical, pos, semantic, context, constraint,
negation)` as an 8-dimensional feature vector per token and train a small
supervised model (e.g. logistic regression or a shallow token-tagging
transformer) on human-labeled importance judgments or on outcomes from the
compression stage (did removing this token hurt downstream task
performance?), replacing the fixed linear combination with a learned one —
without changing any of the individual signal extractors.

## 6. Scientific caveats (Section 16 of the spec)

- TF-IDF/word-frequency is *one statistical signal contributing to* the
  estimate — not "the true importance of a token."
- Nothing here assumes "attention equals importance"; no attention weights
  are used or implied anywhere in this module.
- The final score is an engineered, weighted combination of independently
  fallible signals, not a learned ground truth.

## 7. Example output (generated by running this code, not hand-written)

```
$ python main.py "Do not delete the production database."

Token           TF-IDF  Entity  Tech    POS  Semantic Context Constr Negate  Final  Rank
Do                0.23    0.00  0.00   0.38     0.52    0.00   0.00   0.00   0.18   LOW
not               0.11    0.00  0.00   0.38     0.52    0.00   0.00   1.00   0.89   HIGH   (protected: negation)
delete            1.00    0.00  0.00   0.94     0.52    1.00   0.00   0.00   0.43   MEDIUM
the               0.00    0.00  0.00   0.12     0.52    0.00   0.00   0.00   0.12   LOW
production        0.67    0.00  0.00   1.00     0.52    0.00   0.00   0.00   0.30   LOW
database          0.97    0.00  1.00   1.00     1.00    0.48   0.00   0.00   0.58   MEDIUM
.                 0.00    0.00  0.00   0.00     0.00    0.00   0.00   0.00   0.00   LOW
```

"not" correctly outranks every function word and is flagged protected, even
though on raw POS/TF-IDF signals alone it looks unremarkable — exactly the
failure mode Section 2/6 warns against.
