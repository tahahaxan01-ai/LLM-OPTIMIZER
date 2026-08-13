import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from importance.config import DEFAULT_CONFIG
from importance.scorer import TokenImportanceScorer
from llm_router import choose_best_model_from_metadata, extract_features
from main import compress_prompt as compress_scored_tokens


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL_MAP = {
    "nemotron_ultra": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "nemotron_super": "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "north_mini_code": "qwen/qwen-2.5-coder-32b-instruct",
    "laguna_s": "meta-llama/llama-3.1-8b-instruct:free",
    "laguna_xs": "google/gemma-2-9b-it:free",
    "gemma_26b": "google/gemma-3-27b-it:free",
    "gpt_oss_20b": "openai/gpt-oss-20b",
    "ling_tiny": "google/gemma-2-9b-it:free",
}


def load_env_file() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


load_env_file()
print(
    "OpenRouter key loaded:",
    bool(os.getenv("OPENROUTER_API_KEY")),
    "length:",
    len(os.getenv("OPENROUTER_API_KEY", "")),
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    keep_ratio: float = Field(default=0.6, ge=0.1, le=1.0)


class ChatResponse(BaseModel):
    reply: str
    original_prompt: str
    compressed_prompt: str
    selected_router_model: str
    selected_openrouter_model: str
    features: Dict[str, Any]
    reduction: float
    dropped_tokens: List[str]


app = FastAPI(title="Prompt Compressor Router Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_scorer() -> TokenImportanceScorer:
    return TokenImportanceScorer(config=DEFAULT_CONFIG)


def get_openrouter_model(internal_model: str) -> str:
    env_key = f"OPENROUTER_MODEL_{internal_model.upper()}"
    return os.getenv(env_key, DEFAULT_MODEL_MAP.get(internal_model, "openai/gpt-oss-20b"))


def compress_prompt(prompt: str, keep_ratio: float) -> Dict[str, Any]:
    results = get_scorer().score(prompt)
    compressed_text, kept, dropped, reduction = compress_scored_tokens(results, keep_ratio)

    return {
        "compressed_prompt": compressed_text,
        "kept_tokens": [token.token for token in kept],
        "dropped_tokens": [token.token for token in dropped],
        "reduction": round(reduction, 3),
    }


def route_original_prompt(prompt: str) -> Dict[str, Any]:
    features = extract_features(prompt)
    selected_model = choose_best_model_from_metadata(features)

    return {
        "features": features,
        "selected_model": selected_model,
        "openrouter_model": get_openrouter_model(selected_model),
    }


def call_openrouter(model: str, compressed_prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return (
            "OpenRouter is not connected yet. Add OPENROUTER_API_KEY to your "
            "environment, restart the backend, and this route will call the selected model."
        )
    api_key = api_key.strip()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer the user from the compressed prompt you receive.",
            },
            {
                "role": "user",
                "content": compressed_prompt,
            },
        ],
    }

    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Content-Type", "application/json")
    request.add_header("HTTP-Referer", os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"))
    request.add_header("X-Title", os.getenv("OPENROUTER_APP_NAME", "Prompt Compressor Router Chat"))

    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"OpenRouter HTTP {exc.code}: {error_body}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter request failed with {exc.code}: {error_body}",
        ) from exc
    except (URLError, TimeoutError) as exc:
        print(f"OpenRouter connection failed: {exc}")
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"Unexpected OpenRouter response: {data}") from exc


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    original_prompt = request.message.strip()
    if not original_prompt:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    compression = compress_prompt(original_prompt, request.keep_ratio)
    route = route_original_prompt(original_prompt)
    print(
        "Routing request:",
        {
            "selected_router_model": route["selected_model"],
            "selected_openrouter_model": route["openrouter_model"],
            "compressed_prompt": compression["compressed_prompt"],
        },
    )
    reply = call_openrouter(
        model=route["openrouter_model"],
        compressed_prompt=compression["compressed_prompt"],
    )

    return ChatResponse(
        reply=reply,
        original_prompt=original_prompt,
        compressed_prompt=compression["compressed_prompt"],
        selected_router_model=route["selected_model"],
        selected_openrouter_model=route["openrouter_model"],
        features=route["features"],
        reduction=compression["reduction"],
        dropped_tokens=compression["dropped_tokens"],
    )
