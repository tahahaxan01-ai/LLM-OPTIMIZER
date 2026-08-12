"""
Word-level tokenization (spaCy) and alignment to LLM subword tokens
(tiktoken).

IMPORTANT: "1 word == 1 LLM token" is false, and this module keeps two
separate representations to deal with that honestly:

  * WordToken   - the linguistic word/span level, produced by spaCy.
                  All importance signals operate at this level, because
                  POS tags, dependency parses, and named entities are
                  only meaningful for whole words, not arbitrary
                  byte-pair fragments.

  * SubwordToken - after the final importance score is computed per
                  WordToken, `align_to_llm_tokens` maps each score onto
                  the actual tokenizer output (tiktoken) that an LLM
                  API would see, by overlapping character offsets. A
                  word that becomes multiple subword tokens (e.g.
                  "authentication" often splits into several BPE
                  pieces) has its single word-level importance score
                  copied onto every subword token it produced, because
                  the next stage (compression) will need to make
                  keep/drop decisions at the subword-token level to be
                  directly usable against an LLM's real token budget.
"""
from dataclasses import dataclass
from typing import List, Optional
import spacy
import tiktoken

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run:\n"
                "    python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


@dataclass
class WordToken:
    index: int
    text: str
    start_char: int
    end_char: int
    lemma: str
    pos: str            # coarse POS (spaCy .pos_)
    tag: str             # fine-grained tag (spaCy .tag_)
    dep: str
    ent_type: str         # '' if not part of a named entity
    is_stop: bool
    is_punct: bool
    is_space: bool


def tokenize_words(text: str) -> List[WordToken]:
    nlp = get_nlp()
    doc = nlp(text)
    tokens = []
    for t in doc:
        if t.is_space:
            continue
        tokens.append(WordToken(
            index=len(tokens),
            text=t.text,
            start_char=t.idx,
            end_char=t.idx + len(t.text),
            lemma=t.lemma_,
            pos=t.pos_,
            tag=t.tag_,
            dep=t.dep_,
            ent_type=t.ent_type_,
            is_stop=t.is_stop,
            is_punct=t.is_punct,
            is_space=t.is_space,
        ))
    return tokens


@dataclass
class SubwordToken:
    index: int
    token_id: int
    text: str
    start_char: int
    end_char: int
    importance_score: Optional[float] = None
    source_word_index: Optional[int] = None


_ENC = None


def get_encoder():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def tokenize_subwords(text: str) -> List[SubwordToken]:
    enc = get_encoder()
    ids = enc.encode(text)
    subwords = []
    cursor = 0
    for i, tid in enumerate(ids):
        piece = enc.decode([tid])
        # tiktoken pieces can include a leading space, so search from
        # the current cursor to recover character offsets.
        start = text.find(piece, cursor)
        if start == -1:
            start = cursor  # fallback for rare unicode edge cases
        end = start + len(piece)
        subwords.append(SubwordToken(
            index=i, token_id=tid, text=piece,
            start_char=start, end_char=end,
        ))
        cursor = end
    return subwords


def align_to_llm_tokens(word_tokens: List[WordToken],
                         word_scores: List[float],
                         text: str) -> List[SubwordToken]:
    """Map word-level importance scores onto real LLM subword tokens
    by character-offset overlap. Each subword token inherits the score
    of whichever word token it overlaps most with."""
    subwords = tokenize_subwords(text)
    for sw in subwords:
        best_overlap = 0
        best_word_idx = None
        for wt in word_tokens:
            overlap = min(sw.end_char, wt.end_char) - max(sw.start_char, wt.start_char)
            if overlap > best_overlap:
                best_overlap = overlap
                best_word_idx = wt.index
        if best_word_idx is not None:
            sw.source_word_index = best_word_idx
            sw.importance_score = word_scores[best_word_idx]
        else:
            sw.importance_score = 0.0
    return subwords
