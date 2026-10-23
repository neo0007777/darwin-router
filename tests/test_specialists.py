"""Tests for Mixture-of-Specialists (Domain-Specific Small Models) routing and fallback."""

import httpx
import numpy as np
import pytest

from router import config
from router.index import ExemplarIndex
from router.proxy import dispatch


def test_category_voting_returns_expected_domain():
    idx = ExemplarIndex()
    # Mock vecs: 3 code exemplars, 1 math exemplar
    # Normalizing a 4-dim space
    e_code1 = [1.0, 0.0, 0.0, 0.0]
    e_code2 = [0.95, 0.05, 0.0, 0.0]
    e_math = [0.0, 1.0, 0.0, 0.0]

    idx.add("code snippet 1", "simple", "bootstrap", e_code1, persist=False, category="code")
    idx.add("code snippet 2", "simple", "bootstrap", e_code2, persist=False, category="code")
    idx.add("math equation", "simple", "bootstrap", e_math, persist=False, category="math")

    # Query very close to code
    q_code = [0.99, 0.01, 0.0, 0.0]
    label, margin, category = idx.route(q_code, return_category=True)
    assert label == "simple"
    assert category == "code"

    # Query with default return_category=False returns 2-tuple (label, margin)
    result = idx.route(q_code, return_category=False)
    assert len(result) == 2
    assert result[0] == "simple"

    # Query close to math
    q_math = [0.01, 0.99, 0.0, 0.0]
    label_m, margin_m, category_m = idx.route(q_math, return_category=True)
    assert label_m == "simple"
    assert category_m == "math"


def test_specialist_config_mapping():
    assert "code" in config.SPECIALIST_MODELS
    assert "math" in config.SPECIALIST_MODELS
    assert "general" in config.SPECIALIST_MODELS
    assert config.SPECIALIST_MODELS["general"] == config.OLLAMA_MODEL


@pytest.mark.anyio
async def test_specialist_dispatch_graceful_fallback(monkeypatch):
    """When a specialist model is missing in Ollama (404), dispatch must fall back
    silently to the default OLLAMA_MODEL without raising an exception."""
    from router import proxy

    calls = []

    async def mock_post(url, json=None, **kwargs):
        requested_model = json.get("model")
        calls.append(requested_model)
        req = httpx.Request("POST", url)
        if requested_model == "qwen2.5-coder:7b":
            # Simulate Ollama 404 for uninstalled model
            response = httpx.Response(404, json={"error": "model 'qwen2.5-coder:7b' not found"}, request=req)
            raise httpx.HTTPStatusError("Model Not Found", request=req, response=response)
        
        # Base model succeeds
        return httpx.Response(200, json={
            "id": "chatcmpl-mock",
            "choices": [{"message": {"role": "assistant", "content": "hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }, request=req)

    monkeypatch.setattr(proxy.http_client, "post", mock_post)

    body = {"messages": [{"role": "user", "content": "def test(): pass"}]}
    resp, model, backend = await dispatch("simple", body, "def test(): pass", model_override="qwen2.5-coder:7b")

    # Verify both attempts were made: first specialist, then fallback
    assert calls == ["qwen2.5-coder:7b", config.OLLAMA_MODEL]
    assert "(specialist fallback)" in model
    assert backend == "ollama"
    assert resp["choices"][0]["message"]["content"] == "hello world"
