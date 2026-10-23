"""Proxy and estimated pricing for models ($/1M tokens).
Supports both local provider (simulated proxy pricing) and original hackathon backends.
"""

from router import config

# model -> ($/1M input tokens, $/1M output tokens)
PRICES = {
    # Local & Free stack (Simulated Proxy Rates based on standard hosted inference)
    "llama3.2:3b": (0.04, 0.16),
    "qwen2.5:3b": (0.04, 0.16),
    "qwen2.5:1.5b": (0.02, 0.08),
    "gemma-4-26b-a4b-it": (0.075, 0.30),
    "gemini-3-flash-preview": (0.075, 0.30),
    "gemini-3.8-flash": (0.075, 0.30),
    "gemini-3.5-flash-lite": (0.075, 0.30),
    "gemini-3.5-flash": (0.075, 0.30),
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-1.5-flash": (0.075, 0.30),
    # Original hackathon Alibaba Model Studio pricing
    "qwen-flash": (0.05, 0.40),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    "qwen-vl-max": (0.80, 3.20),
    # Original hackathon SIE lanes
    "Qwen/Qwen3.5-4B": (0.04, 0.16),
    "Qwen/Qwen3.6-27B": (0.40, 1.60),
}

if config.PROVIDER == "local":
    COUNTERFACTUAL_MODEL = config.GEMINI_MODEL
else:
    COUNTERFACTUAL_MODEL = "qwen-max"

# $ per generation for visual models (original mode only)
FLAT = {
    "z-image-turbo": 0.01,
    "qwen-image-3.0-pro": 0.06,
    "wan2.7-image": 0.04,
    "happyhorse-1.1-t2v": 0.20,
}


def flat_cost(model: str) -> float:
    return FLAT.get(model, 0.05)


def estimate(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, PRICES.get(COUNTERFACTUAL_MODEL, (0.075, 0.30)))
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def counterfactual(tokens_in: int, tokens_out: int) -> float:
    return estimate(COUNTERFACTUAL_MODEL, tokens_in, tokens_out)


calculate_cost = estimate
