"""Configuration layer. Supports three provider modes:
- 'local' (default): sentence-transformers embeddings, Ollama cheap lane, Gemini free tier big lane
- 'original': SIE + Alibaba Cloud (the hackathon setup, needs paid keys)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── Provider mode ──────────────────────────────────────────────────────────
PROVIDER = os.getenv("PROVIDER", "local")  # "local" or "original"

# ── Gemini free tier ───────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
LARGE_MODEL_PRICING_URL = "https://ai.google.dev/pricing"
LARGE_MODEL_PRICING_RETRIEVAL_DATE = "October 2026"
LARGE_MODEL_INPUT_PRICE_PER_1M = 0.075   # $/1M input tokens
LARGE_MODEL_OUTPUT_PRICE_PER_1M = 0.30   # $/1M output tokens (includes thought tokens)
COST_LABEL = "Proxy Cost (not billed)"

# ── Ollama (local) ─────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ── Optional Mixture-of-Specialists (Domain-routed small models) ───────────
ENABLE_SPECIALISTS = os.getenv("ENABLE_SPECIALISTS", "false").lower() in ("true", "1", "yes")
SPECIALIST_MODELS = {
    "code": os.getenv("SPECIALIST_CODE_MODEL", "qwen2.5-coder:7b"),
    "math": os.getenv("SPECIALIST_MATH_MODEL", "deepseek-math:7b"),
    "general": OLLAMA_MODEL,
}

# ── Local embeddings ───────────────────────────────────────────────────────
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Original hackathon backends (only used when PROVIDER=original) ─────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://ws-217y1bpliyzcf5nl.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")

SIE_BASE_URL = os.getenv("SIE_BASE_URL", "https://api.superlinked.com/v1").rstrip("/")
SIE_API_KEY = os.getenv("SIE_API_KEY", "")
SIE_EMBED_MODEL = os.getenv("SIE_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
SIE_CHAT_MODEL = os.getenv("SIE_CHAT_MODEL", "Qwen/Qwen3.5-4B")
SIE_SMART_MODEL = os.getenv("SIE_SMART_MODEL", "Qwen/Qwen3.6-27B")

# ── Data paths ─────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
EXEMPLARS_PATH = DATA_DIR / "exemplars.jsonl"
LOG_PATH = DATA_DIR / "log.jsonl"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
JUDGE_LOG_PATH = DATA_DIR / "judge.jsonl"
REPLAY_LOG_PATH = DATA_DIR / "replay.jsonl"
HOLDOUT_PATH = DATA_DIR / "holdout.jsonl"
EVAL_PATH = DATA_DIR / "eval_200.jsonl"

# ── Routes ─────────────────────────────────────────────────────────────────
TEXT_ROUTES = ["simple", "reasoning"]

if PROVIDER == "local":
    # Free stack: Ollama for cheap, Gemini free tier for expensive
    MODEL_FOR_ROUTE = {
        "simple": OLLAMA_MODEL,
        "reasoning": GEMINI_MODEL,
        # Visual routes not supported in local mode
        "vision": GEMINI_MODEL,
        "imagegen": "disabled",
        "videogen": "disabled",
    }
    BACKEND_FOR_ROUTE = {
        "simple": "ollama",
        "reasoning": "gemini",
        "vision": "gemini",
        "imagegen": "disabled",
        "videogen": "disabled",
    }
else:
    # Original hackathon setup
    MODEL_FOR_ROUTE = {
        "simple": os.getenv("SIMPLE_MODEL", SIE_CHAT_MODEL if SIE_API_KEY else "qwen-flash"),
        "reasoning": os.getenv("REASONING_MODEL", "qwen-max" if DASHSCOPE_API_KEY else SIE_SMART_MODEL),
        "vision": os.getenv("VISION_MODEL", "qwen-vl-max" if DASHSCOPE_API_KEY else SIE_SMART_MODEL),
        "imagegen": os.getenv("IMAGEGEN_MODEL", "z-image-turbo"),
        "videogen": "happyhorse-1.1-t2v",
    }
    BACKEND_FOR_ROUTE = {
        "simple": "sie" if SIE_API_KEY else "modelstudio",
        "reasoning": "modelstudio" if DASHSCOPE_API_KEY else "sie",
        "vision": "modelstudio" if DASHSCOPE_API_KEY else "sie",
        "imagegen": "modelstudio",
        "videogen": "modelstudio",
    }

# Visual models use the DashScope NATIVE api on dashscope-intl (Singapore),
# not the workspace compatible-mode host. Image gen is sync; video is async
# (X-DashScope-Async: enable, then poll /tasks/{task_id}).
DASHSCOPE_NATIVE_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
IMAGEGEN_ENDPOINT = f"{DASHSCOPE_NATIVE_BASE}/services/aigc/multimodal-generation/generation"
VIDEOGEN_ENDPOINT = f"{DASHSCOPE_NATIVE_BASE}/services/aigc/video-generation/video-synthesis"

# ── Cost weight for routing (λ parameter for quality-vs-cost tradeoff) ─────
COST_WEIGHT = float(os.getenv("COST_WEIGHT", "0.0"))
