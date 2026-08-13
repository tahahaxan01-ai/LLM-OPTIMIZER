import re
import random
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================
# 1. MODELS
# ============================================================

MODEL_METADATA = {

    "nemotron_ultra": {
        "math": 95,
        "coding": 85,
        "reasoning": 95,
        "knowledge": 90,
        "writing": 80,
        "summarization": 85,
        "translation": 80,
        "general": 90,

        "max_difficulty": 3,
        "max_context": 3,

        "tool_support": True,
        "structured_output": True,
        "multilingual": True,

        "cost": 10
    },

    "nemotron_super": {
        "math": 88,
        "coding": 80,
        "reasoning": 88,
        "knowledge": 85,
        "writing": 78,
        "summarization": 82,
        "translation": 78,
        "general": 85,

        "max_difficulty": 3,
        "max_context": 3,

        "tool_support": True,
        "structured_output": True,
        "multilingual": True,

        "cost": 7
    },

    "north_mini_code": {
        "math": 60,
        "coding": 95,
        "reasoning": 78,
        "knowledge": 65,
        "writing": 55,
        "summarization": 65,
        "translation": 55,
        "general": 65,

        "max_difficulty": 3,
        "max_context": 3,

        "tool_support": True,
        "structured_output": True,
        "multilingual": False,

        "cost": 5
    },

    "laguna_s": {
        "math": 65,
        "coding": 92,
        "reasoning": 80,
        "knowledge": 65,
        "writing": 60,
        "summarization": 65,
        "translation": 55,
        "general": 68,

        "max_difficulty": 3,
        "max_context": 3,

        "tool_support": True,
        "structured_output": True,
        "multilingual": False,

        "cost": 6
    },

    "laguna_xs": {
        "math": 50,
        "coding": 82,
        "reasoning": 65,
        "knowledge": 60,
        "writing": 55,
        "summarization": 62,
        "translation": 50,
        "general": 60,

        "max_difficulty": 2,
        "max_context": 2,

        "tool_support": False,
        "structured_output": True,
        "multilingual": False,

        "cost": 2
    },

    "gemma_26b": {
        "math": 78,
        "coding": 72,
        "reasoning": 78,
        "knowledge": 86,
        "writing": 85,
        "summarization": 86,
        "translation": 82,
        "general": 85,

        "max_difficulty": 2,
        "max_context": 2,

        "tool_support": False,
        "structured_output": True,
        "multilingual": True,

        "cost": 3
    },

    "gpt_oss_20b": {
        "math": 82,
        "coding": 82,
        "reasoning": 85,
        "knowledge": 82,
        "writing": 80,
        "summarization": 82,
        "translation": 78,
        "general": 83,

        "max_difficulty": 3,
        "max_context": 3,

        "tool_support": True,
        "structured_output": True,
        "multilingual": True,

        "cost": 4
    },

    "ling_tiny": {
        "math": 55,
        "coding": 50,
        "reasoning": 55,
        "knowledge": 65,
        "writing": 65,
        "summarization": 67,
        "translation": 65,
        "general": 65,

        "max_difficulty": 1,
        "max_context": 1,

        "tool_support": False,
        "structured_output": False,
        "multilingual": True,

        "cost": 1
    }
}


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(query: str):

    text = query.lower()

    features = {
        "task": "general",
        "difficulty": "easy",
        "reasoning": "low",
        "context_length": "short",
        "tool_required": False,
        "coding": False,
        "structured_output": False,
        "multilingual": False
    }

    # --------------------------------------------------------
    # TASK DETECTION
    # --------------------------------------------------------

    coding_keywords = [
        "python", "javascript", "typescript", "java", "c++",
        "code", "function", "bug", "debug", "api",
        "sql", "react", "fastapi", "algorithm",
        "program", "database", "compiler", "github",
        "docker", "backend", "frontend"
    ]

    math_keywords = [
        "equation", "integral", "derivative",
        "matrix", "probability", "theorem",
        "algebra", "calculus", "geometry",
        "differentiate", "integrate", "gradient",
        "prime", "polynomial"
    ]

    writing_keywords = [
        "write an essay", "write a letter", "write an email",
        "write a story", "rewrite", "grammar",
        "paraphrase", "birthday message"
    ]

    summarization_keywords = [
        "summarize", "summary", "shorten this"
    ]

    translation_keywords = [
        "translate", "translation"
    ]

    reasoning_keywords = [
        "logic", "deduce", "infer",
        "logical argument", "logic puzzle"
    ]

    knowledge_keywords = [
        "what is", "who is", "who wrote",
        "explain what", "define"
    ]

    # Coding first because "solve this code" etc.
    if any(x in text for x in coding_keywords):
        features["task"] = "coding"
        features["coding"] = True

    elif any(x in text for x in translation_keywords):
        features["task"] = "translation"
        features["multilingual"] = True

    elif any(x in text for x in summarization_keywords):
        features["task"] = "summarization"

    elif any(x in text for x in writing_keywords):
        features["task"] = "writing"

    elif any(x in text for x in math_keywords):
        features["task"] = "math"

    elif any(x in text for x in reasoning_keywords):
        features["task"] = "reasoning"

    elif any(x in text for x in knowledge_keywords):
        features["task"] = "knowledge"

    # Detect obvious mathematical notation
    math_patterns = [
        r"\d+\s*[\+\-\*/]\s*\d+",
        r"[a-z]\s*\^\s*\d+",
        r"\d*[a-z]\s*[\+\-]\s*\d+",
        r"\bsolve\s+.*=",
    ]

    if any(re.search(pattern, text) for pattern in math_patterns):
        if not features["coding"]:
            features["task"] = "math"

    # --------------------------------------------------------
    # CONTEXT LENGTH
    # --------------------------------------------------------

    word_count = len(query.split())

    if word_count < 50:
        features["context_length"] = "short"

    elif word_count < 300:
        features["context_length"] = "medium"

    else:
        features["context_length"] = "long"

    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    high_reasoning_keywords = [
        "prove",
        "derive",
        "optimize",
        "analyze",
        "design",
        "step by step",
        "architecture",
        "complex",
        "reason through"
    ]

    medium_reasoning_keywords = [
        "solve",
        "calculate",
        "debug",
        "compare",
        "explain",
        "implement"
    ]

    if any(x in text for x in high_reasoning_keywords):
        features["reasoning"] = "high"

    elif any(x in text for x in medium_reasoning_keywords):
        features["reasoning"] = "medium"

    # --------------------------------------------------------
    # DIFFICULTY
    # --------------------------------------------------------

    hard_keywords = [
        "advanced",
        "complex",
        "proof",
        "prove",
        "theorem",
        "optimization",
        "architecture",
        "distributed",
        "concurrent",
        "microservices",
        "research",
        "derive"
    ]

    medium_keywords = [
        "debug",
        "calculate",
        "implement",
        "compare",
        "algorithm",
        "explain",
        "database",
        "api"
    ]

    if any(x in text for x in hard_keywords):
        features["difficulty"] = "hard"

    elif any(x in text for x in medium_keywords):
        features["difficulty"] = "medium"

    # --------------------------------------------------------
    # TOOLS
    # --------------------------------------------------------

    tool_keywords = [
        "latest",
        "current",
        "today",
        "weather",
        "stock price",
        "bitcoin price",
        "news",
        "search online",
        "browse",
        "live data"
    ]

    if any(x in text for x in tool_keywords):
        features["tool_required"] = True

    # --------------------------------------------------------
    # STRUCTURED OUTPUT
    # --------------------------------------------------------

    structured_keywords = [
        "json",
        "xml",
        "csv",
        "yaml",
        "schema",
        "machine readable",
        "structured output"
    ]

    if any(x in text for x in structured_keywords):
        features["structured_output"] = True

    # --------------------------------------------------------
    # MULTILINGUAL
    # --------------------------------------------------------

    multilingual_keywords = [
        "urdu",
        "spanish",
        "french",
        "german",
        "arabic",
        "chinese",
        "japanese",
        "hindi",
        "translate"
    ]

    if any(x in text for x in multilingual_keywords):
        features["multilingual"] = True

    return features


# ============================================================
# 3. CONVERT TEXT LEVELS TO NUMBERS
# ============================================================

DIFFICULTY_TO_NUMBER = {
    "easy": 1,
    "medium": 2,
    "hard": 3
}

CONTEXT_TO_NUMBER = {
    "short": 1,
    "medium": 2,
    "long": 3
}

REASONING_TO_NUMBER = {
    "low": 1,
    "medium": 2,
    "high": 3
}


# ============================================================
# 4. SCORE EACH MODEL
#
# THIS IS USED OFFLINE TO CREATE OUR SYNTHETIC LABELS.
# ============================================================

def calculate_model_score(features, model):

    metadata = MODEL_METADATA[model]

    task = features["task"]

    # Some metadata doesn't have every possible task.
    task_score = metadata.get(task, metadata["general"])

    difficulty = DIFFICULTY_TO_NUMBER[features["difficulty"]]
    context = CONTEXT_TO_NUMBER[features["context_length"]]
    reasoning = REASONING_TO_NUMBER[features["reasoning"]]

    # --------------------------------------------------------
    # HARD FILTERS
    # --------------------------------------------------------

    if metadata["max_difficulty"] < difficulty:
        return -9999

    if metadata["max_context"] < context:
        return -9999

    if features["tool_required"] and not metadata["tool_support"]:
        return -9999

    if features["structured_output"] and not metadata["structured_output"]:
        return -9999

    if features["multilingual"] and not metadata["multilingual"]:
        return -9999

    # --------------------------------------------------------
    # BASE PERFORMANCE
    # --------------------------------------------------------

    score = task_score

    # Strong reasoning query should favor strong reasoning models
    if reasoning == 3:
        score += metadata["reasoning"] * 0.30

    elif reasoning == 2:
        score += metadata["reasoning"] * 0.15

    else:
        score += metadata["reasoning"] * 0.05

    # --------------------------------------------------------
    # CODING SPECIALIZATION
    # --------------------------------------------------------

    if features["coding"]:
        score += metadata["coding"] * 0.30

    # --------------------------------------------------------
    # COST PENALTY
    # --------------------------------------------------------

    # Bigger penalty for easy tasks.
    # We don't want Ultra answering "2 + 2".
    if difficulty == 1:
        cost_weight = 3.0

    elif difficulty == 2:
        cost_weight = 1.5

    else:
        cost_weight = 0.5

    score -= metadata["cost"] * cost_weight

    return score


# ============================================================
# 5. GET OPTIMAL MODEL FROM METADATA
#
# Used to label synthetic training data.
# ============================================================

def choose_best_model_from_metadata(features):

    scores = {}

    for model in MODEL_METADATA:
        scores[model] = calculate_model_score(
            features,
            model
        )

    best_model = max(
        scores,
        key=scores.get
    )

    return best_model


# ============================================================
# 6. SYNTHETIC TRAINING DATA
# ============================================================

TASKS = [
    "math",
    "coding",
    "reasoning",
    "knowledge",
    "writing",
    "summarization",
    "translation",
    "general"
]

DIFFICULTIES = [
    "easy",
    "medium",
    "hard"
]

REASONING_LEVELS = [
    "low",
    "medium",
    "high"
]

CONTEXT_LEVELS = [
    "short",
    "medium",
    "long"
]


def generate_training_data(number_of_samples=10000):

    rows = []

    for _ in range(number_of_samples):

        task = random.choice(TASKS)

        features = {

            "task": task,

            "difficulty": random.choice(
                DIFFICULTIES
            ),

            "reasoning": random.choice(
                REASONING_LEVELS
            ),

            "context_length": random.choice(
                CONTEXT_LEVELS
            ),

            "tool_required": random.choice(
                [True, False]
            ),

            "coding": task == "coding",

            "structured_output": random.choice(
                [True, False]
            ),

            "multilingual": (
                task == "translation"
                or random.random() < 0.10
            )
        }

        best_model = choose_best_model_from_metadata(
            features
        )

        row = features.copy()

        row["best_model"] = best_model

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 7. TRAIN DECISION TREE
# ============================================================

def train_router():

    dataset = generate_training_data(
        number_of_samples=10000
    )

    X = dataset.drop(
        columns=["best_model"]
    )

    y = dataset["best_model"]

    categorical_features = [
        "task",
        "difficulty",
        "reasoning",
        "context_length"
    ]

    boolean_features = [
        "tool_required",
        "coding",
        "structured_output",
        "multilingual"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "boolean",
                "passthrough",
                boolean_features
            )
        ]
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    )

    router = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            decision_tree
        )
    ])

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    router.fit(
        X_train,
        y_train
    )

    predictions = router.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Router test accuracy: {accuracy:.4f}"
    )

    return router


# ============================================================
# 8. ROUTE A QUERY
# ============================================================

def route_query(query, router):

    # Step 1
    # Query -> features
    features = extract_features(query)

    # Step 2
    # Convert to dataframe because sklearn expects columns
    input_data = pd.DataFrame(
        [features]
    )

    # Step 3
    # Features -> Decision Tree -> Model
    selected_model = router.predict(
        input_data
    )[0]

    return {
        "query": query,
        "features": features,
        "selected_model": selected_model
    }


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print(
        "\nTraining LLM Router...\n"
    )

    router = train_router()

    print(
        "\nRouter ready."
    )

    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        query = input(
            "Enter query: "
        )

        if query.lower() in {
            "exit",
            "quit"
        }:
            break

        result = route_query(
            query,
            router
        )

        print(
            "\nExtracted Features:"
        )

        for key, value in result[
            "features"
        ].items():

            print(
                f"  {key}: {value}"
            )

        print(
            "\nSelected LLM:"
        )

        print(
            f"  {result['selected_model']}"
        )

        print(
            "\n" + "=" * 60 + "\n"
        )