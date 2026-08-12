"""
20 test prompts covering the categories required by the spec, plus
lightweight sanity assertions per signal. These check the *direction*
of each signal (e.g. negation words score higher than stopwords) - they
are not exhaustive correctness proofs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importance.scorer import TokenImportanceScorer
from importance.config import DEFAULT_CONFIG

TEST_PROMPTS = [
    "What is the capital of France?",                                                                  # 1 general
    "Please explain how to implement JWT authentication in a FastAPI application using Python.",         # 2 technical
    "I am building a multi-tenant SaaS platform and need a detailed explanation of how to design a scalable database schema that supports per-tenant isolation, audit logging, and soft deletes.",  # 3 long
    "Schedule the meeting for January 15, 2026 at 3pm.",                                                 # 4 dates
    "I need exactly 5 examples and no more than 10 lines each.",                                         # 5 numbers
    "Find all transactions greater than $10,000 last quarter.",                                          # 6 monetary
    "Can you summarize the biography of Marie Curie?",                                                   # 7 names
    "What is the weather like in Lahore, Pakistan today?",                                                # 8 locations
    "Do not delete the production database.",                                                             # 9 negation
    "The report must be at least 5 pages, excluding the appendix, and submitted before Friday.",          # 10 constraints
    "Explain the difference between supervised and unsupervised learning in machine learning.",           # 11 AI/ML
    "Write a Python function that reverses a linked list.",                                                # 12 programming
    "Analyze Tesla's revenue growth in Q2 2026 compared to Q2 2025.",                                      # 13 finance
    "Draft a business proposal for expanding into the Southeast Asian market.",                            # 14 business
    "First install the dependencies, then configure the environment variables, and finally run the server.",  # 15 multi-step
    "Return the answer strictly as a JSON object with keys 'name' and 'age'.",                              # 16 output format
    "Deploy the microservice using Kubernetes and PostgreSQL with a Redis cache.",                          # 17 tech terminology
    "Please kindly could you possibly just go ahead and maybe help me with this if that's okay.",           # 18 redundant
    "Cancel my order.",                                                                                     # 19 short
    "Given the current macroeconomic environment, including elevated interest rates and persistent inflation, evaluate whether a mid-cap technology company with $50M in annual revenue should prioritize debt financing or equity financing for a $10M expansion, assuming the expansion must be completed within 18 months and must not increase debt-to-equity ratio above 1.5.",  # 20 complex long
]


def test_negation_is_protected():
    scorer = TokenImportanceScorer(config=DEFAULT_CONFIG)
    results = scorer.score("Do not delete the production database.")
    not_token = next(r for r in results if r.token.lower() == "not")
    the_token = next(r for r in results if r.token.lower() == "the")
    assert not_token.final_score > the_token.final_score
    assert not_token.protected is True


def test_stopwords_score_lower_than_technical_terms():
    scorer = TokenImportanceScorer(config=DEFAULT_CONFIG)
    results = scorer.score("Please explain how to implement JWT authentication in FastAPI.")
    please = next(r for r in results if r.token.lower() == "please")
    jwt = next(r for r in results if r.token.lower() == "jwt")
    assert jwt.final_score > please.final_score


def test_constraint_numbers_protected():
    scorer = TokenImportanceScorer(config=DEFAULT_CONFIG)
    results = scorer.score("Find all transactions greater than $10,000 last quarter.")
    money_tokens = [r for r in results if "10,000" in r.token or r.token == "$"]
    assert any(r.final_score >= DEFAULT_CONFIG.medium_threshold for r in money_tokens)


def test_all_demo_prompts_run_without_error():
    scorer = TokenImportanceScorer(config=DEFAULT_CONFIG)
    for p in TEST_PROMPTS:
        results = scorer.score(p)
        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.final_score <= 1.0


if __name__ == "__main__":
    test_negation_is_protected()
    test_stopwords_score_lower_than_technical_terms()
    test_constraint_numbers_protected()
    test_all_demo_prompts_run_without_error()
    print("All sanity tests passed.")
