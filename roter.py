import re

def extract_features(query: str):
    text = query.lower()

    features = {
        "task": "general",
        "difficulty": "easy",
        "reasoning": "low",
        "context_length": "short",
        "tool_required": False,
        "coding": False
    }

    # -------------------------
    # TASK DETECTION
    # -------------------------

    math_keywords = [
        "solve", "equation", "integral", "derivative",
        "matrix", "probability", "theorem", "proof",
        "algebra", "calculus", "geometry"
    ]

    coding_keywords = [
        "python", "javascript", "java", "c++",
        "code", "function", "bug", "debug",
        "api", "sql", "react", "fastapi"
    ]

    reasoning_keywords = [
        "reason", "logic", "deduce", "infer",
        "prove", "why", "analyze"
    ]

    if any(word in text for word in coding_keywords):
        features["task"] = "coding"
        features["coding"] = True

    elif any(word in text for word in math_keywords):
        features["task"] = "math"

    elif any(word in text for word in reasoning_keywords):
        features["task"] = "reasoning"

    # -------------------------
    # CONTEXT LENGTH
    # -------------------------

    word_count = len(query.split())

    if word_count < 50:
        features["context_length"] = "short"

    elif word_count < 300:
        features["context_length"] = "medium"

    else:
        features["context_length"] = "long"

    # -------------------------
    # REASONING LEVEL
    # -------------------------

    high_reasoning = [
        "prove",
        "derive",
        "optimize",
        "analyze",
        "compare",
        "explain why",
        "step by step",
        "design"
    ]

    medium_reasoning = [
        "solve",
        "calculate",
        "debug",
        "explain"
    ]

    if any(x in text for x in high_reasoning):
        features["reasoning"] = "high"

    elif any(x in text for x in medium_reasoning):
        features["reasoning"] = "medium"

    # -------------------------
    # DIFFICULTY
    # -------------------------

    hard_words = [
        "advanced",
        "complex",
        "proof",
        "theorem",
        "optimization",
        "architecture",
        "distributed",
        "concurrent"
    ]

    medium_words = [
        "debug",
        "calculate",
        "implement",
        "compare",
        "algorithm"
    ]

    if any(x in text for x in hard_words):
        features["difficulty"] = "hard"

    elif any(x in text for x in medium_words):
        features["difficulty"] = "medium"

    # -------------------------
    # TOOL REQUIREMENT
    # -------------------------

    tool_keywords = [
        "latest",
        "current",
        "today",
        "weather",
        "stock price",
        "news",
        "search online"
    ]

    if any(x in text for x in tool_keywords):
        features["tool_required"] = True

    return features

query = """
Prove that there are infinitely many prime numbers.
"""

# ============================================================
# TEST DATA — 50 QUERIES
# ============================================================

test_queries = [

    # ---------- MATH ----------
    "What is 15 + 27?",
    "Solve 2x + 5 = 15.",
    "Calculate the area of a circle with radius 5.",
    "Solve the quadratic equation x^2 - 5x + 6 = 0.",
    "Find the derivative of x^3 + 2x^2 - 5x.",
    "Calculate the integral of x^2 sin(x).",
    "Solve this system of linear equations using matrices.",
    "Prove that there are infinitely many prime numbers.",
    "Derive the gradient descent update rule.",
    "Prove the central limit theorem.",

    # ---------- CODING ----------
    "Write Python code to print Hello World.",
    "Write a Python function to reverse a list.",
    "Create a JavaScript function that validates an email address.",
    "Explain what this Python function does.",
    "Debug this Python function because it returns the wrong result.",
    "Implement binary search in C++.",
    "Create a FastAPI endpoint for uploading images.",
    "Optimize this SQL query for a database with 10 million records.",
    "Debug a distributed asynchronous Python application.",
    "Design the architecture for a scalable microservices application.",

    # ---------- REASONING ----------
    "If John is taller than Mike and Mike is taller than Sam, who is tallest?",
    "Explain why the sky appears blue.",
    "Compare nuclear energy and solar energy.",
    "Analyze the advantages and disadvantages of remote work.",
    "A farmer has 17 sheep and all but 9 die. How many are left?",
    "If all A are B and some B are C, can we conclude some A are C?",
    "Analyze this logical argument and determine whether it is valid.",
    "Solve this complex logic puzzle step by step.",
    "Design an optimal strategy for allocating limited resources.",
    "Prove whether this logical statement is always true.",

    # ---------- KNOWLEDGE / GENERAL ----------
    "What is the capital of Japan?",
    "Who wrote Hamlet?",
    "What is photosynthesis?",
    "Explain how a CPU works.",
    "What is machine learning?",
    "Explain the difference between supervised and unsupervised learning.",
    "Explain how transformer neural networks work.",
    "Explain the architecture of a modern operating system.",
    "Compare transformers and recurrent neural networks.",
    "Explain quantum entanglement and its implications.",

    # ---------- CURRENT / TOOL ----------
    "What is the weather today?",
    "What is the current Bitcoin price?",
    "Who won the latest Formula 1 race?",
    "What are the latest AI news stories?",
    "Search online for the latest NVIDIA GPU.",

    # ---------- WRITING / OTHER ----------
    "Write a short birthday message.",
    "Summarize this paragraph for me.",
    "Translate this English sentence into Urdu.",
    "Compare Python and Java and explain which one I should learn.",
    "Analyze and design a distributed AI system that can handle millions of users."
]


# ============================================================
# RUN TEST
# ============================================================

for i, query in enumerate(test_queries, start=1):

    result = extract_features(query)

    print("=" * 80)
    print(f"QUERY {i}")
    print(f"Text: {query}")
    print("-" * 80)
    print(result)

print("=" * 80)
print(f"TOTAL QUERIES TESTED: {len(test_queries)}")