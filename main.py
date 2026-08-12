"""
CLI entry point.

Usage (run from inside the llmify/ directory):
    python main.py "Your prompt here"
    python main.py --demo                      # runs the 20 built-in test prompts
    python main.py "Your prompt" --json-out out.json
"""
import json
import argparse

from importance.scorer import TokenImportanceScorer
from importance.config import DEFAULT_CONFIG
from tests.test_importance import TEST_PROMPTS


def print_table(prompt: str, results):
    print(f"\nOriginal Prompt:\n\"{prompt}\"\n")
    header = (f"{'Token':<18}{'TF-IDF':>8}{'Entity':>8}{'Tech':>8}{'POS':>8}"
              f"{'Semantic':>10}{'Context':>9}{'Constr':>8}{'Negate':>8}{'Final':>8}  Rank")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.token:<18}{r.tfidf_score:>8.2f}{r.entity_score:>8.2f}{r.technical_score:>8.2f}"
              f"{r.pos_score:>8.2f}{r.semantic_score:>10.2f}{r.context_score:>9.2f}"
              f"{r.constraint_score:>8.2f}{r.negation_score:>8.2f}{r.final_score:>8.2f}  {r.rank_label}")


def compress_prompt(results, keep_ratio: float):
    """Keep every protected token, plus the top `keep_ratio` fraction of the
    rest by score (dropping the bottom (1-keep_ratio) fraction of
    non-protected tokens). A percentile-based cutoff scales with prompt
    length, instead of an absolute score threshold which gets stricter
    in effect as scores dilute across more competing tokens in longer
    prompts (see conversation: a fixed 0.40 cutoff dropped "fraudulent",
    "patterns", "analyze" etc. from a 45-token prompt)."""
    non_protected = [r for r in results if not r.protected]
    n_keep = round(len(non_protected) * keep_ratio)
    ranked_non_protected = sorted(non_protected, key=lambda r: -r.final_score)
    keep_set = {r.index for r in ranked_non_protected[:n_keep]} | {r.index for r in results if r.protected}
    kept = [r for r in results if r.index in keep_set]
    dropped = [r for r in results if r.index not in keep_set]
    compressed_text = " ".join(r.token for r in kept)
    reduction = 1.0 - (len(kept) / max(1, len(results)))
    return compressed_text, kept, dropped, reduction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--demo", action="store_true", help="Run the 20 built-in test prompts")
    parser.add_argument("--json-out", default=None, help="Path to save full JSON output")
    parser.add_argument("--keep-ratio", type=float, default=0.6,
                         help="Fraction of non-protected tokens to keep, by score rank (default: 0.6)")
    args = parser.parse_args()

    scorer = TokenImportanceScorer(config=DEFAULT_CONFIG)
    keep_ratio = args.keep_ratio

    prompts = TEST_PROMPTS if (args.demo or not args.prompt) else [args.prompt]

    all_output = []
    for p in prompts:
        results = scorer.score(p)
        print_table(p, results)
        ranked = sorted(results, key=lambda r: -r.final_score)
        print("\nRanking (high -> low):")
        for r in ranked:
            print(f"  {r.rank_label:<6} {r.token:<20} {r.final_score:.2f}  ({'; '.join(r.reasons)})")

        compressed_text, kept, dropped, reduction = compress_prompt(results, keep_ratio)
        print(f"\nCompressed Prompt (keep-ratio={keep_ratio:.2f}, {reduction:.0%} tokens removed):")
        print(f"\"{compressed_text}\"")
        if dropped:
            print(f"Dropped: {', '.join(r.token for r in dropped)}")

        all_output.append({
            "prompt": p,
            "tokens": [r.to_dict() for r in results],
            "compressed_prompt": compressed_text,
            "reduction": round(reduction, 3),
        })

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_output, f, indent=2)
        print(f"\nSaved JSON output to {args.json_out}")


if __name__ == "__main__":
    main()
