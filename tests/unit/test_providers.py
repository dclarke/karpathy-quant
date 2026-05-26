"""Tests for gemini provider helpers + init."""

import json
import os
from unittest.mock import patch

import pytest

from karpathy_quant.providers.base import Hypothesis, HypothesisRequest
from karpathy_quant.providers import gemini as gemini_mod


# -- fixtures / helpers --

def _basic_request(**overrides):
    defaults = dict(
        symbol="AAPL",
        ohlcv_stats={"close": {"mean": 150.0}, "volume": {"mean": 5000000}},
        n_hypotheses=3,
    )
    defaults.update(overrides)
    return HypothesisRequest(**defaults)


VALID_JSON = json.dumps([
    {"code": "def get_feature(df): return df['close']", "theory": "price is everything", "direction": "long"},
    {"code": "def get_feature(df): return df['volume']", "theory": "volume matters", "direction": "short"},
])

# same thing but wrapped in markdown fences
FENCED_JSON = "```json\n" + VALID_JSON + "\n```"


# ===== _build_user_prompt tests =====

class TestBuildUserPrompt:

    def test_basic_prompt_contains_symbol(self):
        req = _basic_request()
        prompt = gemini_mod._build_user_prompt(req)
        assert "AAPL" in prompt
        assert "3 trading signal" in prompt
        # should not have saturated/hof sections
        assert "AVOID" not in prompt
        assert "shown promise" not in prompt

    def test_with_saturated_theories(self):
        req = _basic_request(saturated_theories=["mean reversion on mondays", "gap fill nonsense"])
        prompt = gemini_mod._build_user_prompt(req)
        assert "AVOID" in prompt
        assert "mean reversion on mondays" in prompt
        assert "gap fill nonsense" in prompt

    def test_with_hall_of_fame(self):
        req = _basic_request(hall_of_fame=["momentum breakout", "volume spike entry"])
        prompt = gemini_mod._build_user_prompt(req)
        assert "shown promise" in prompt
        assert "momentum breakout" in prompt

    def test_with_both_saturated_and_hof(self):
        req = _basic_request(
            saturated_theories=["bad idea 1"],
            hall_of_fame=["good idea 1"],
        )
        prompt = gemini_mod._build_user_prompt(req)
        assert "AVOID" in prompt
        assert "shown promise" in prompt

    def test_saturated_truncated_to_20(self):
        # only last 20 should appear
        theories = [f"theory_{i}" for i in range(30)]
        req = _basic_request(saturated_theories=theories)
        prompt = gemini_mod._build_user_prompt(req)
        assert "theory_10" in prompt  # 30-20=10, so theory_10 is first kept
        assert "theory_0" not in prompt

    def test_hof_truncated_to_10(self):
        hof = [f"approach_{i}" for i in range(15)]
        req = _basic_request(hall_of_fame=hof)
        prompt = gemini_mod._build_user_prompt(req)
        assert "approach_5" in prompt
        assert "approach_0" not in prompt

    def test_ohlcv_stats_serialized(self):
        req = _basic_request(ohlcv_stats={"close": {"mean": 42.5}})
        prompt = gemini_mod._build_user_prompt(req)
        assert "42.5" in prompt

    def test_ends_with_json_instruction(self):
        req = _basic_request()
        prompt = gemini_mod._build_user_prompt(req)
        assert "JSON array" in prompt


# ===== _parse_response tests =====

class TestParseResponse:

    def test_valid_json(self):
        results = gemini_mod._parse_response(VALID_JSON)
        assert len(results) == 2
        assert results[0].theory == "price is everything"
        assert results[1].direction == "short"

    def test_markdown_fenced_json(self):
        results = gemini_mod._parse_response(FENCED_JSON)
        assert len(results) == 2

    def test_filters_items_without_get_feature(self):
        data = json.dumps([
            {"code": "def something(df): pass", "theory": "nope"},
            {"code": "def get_feature(df): return df['close']", "theory": "yes"},
        ])
        results = gemini_mod._parse_response(data)
        assert len(results) == 1
        assert results[0].theory == "yes"

    def test_empty_code_skipped(self):
        data = json.dumps([
            {"code": "", "theory": "empty code"},
            {"code": "def get_feature(df): ...", "theory": "ok"},
        ])
        results = gemini_mod._parse_response(data)
        assert len(results) == 1

    def test_direction_defaults_long(self):
        data = json.dumps([
            {"code": "def get_feature(df): return 1", "theory": "no dir given"},
        ])
        results = gemini_mod._parse_response(data)
        assert results[0].direction == "long"

    def test_returns_hypothesis_objects(self):
        results = gemini_mod._parse_response(VALID_JSON)
        for h in results:
            assert isinstance(h, Hypothesis)

    def test_empty_array(self):
        assert gemini_mod._parse_response("[]") == []

    def test_whitespace_around_json(self):
        padded = "   \n" + VALID_JSON + "\n  "
        results = gemini_mod._parse_response(padded)
        assert len(results) == 2

    def test_missing_code_key(self):
        # code key missing entirely -> defaults to ""
        data = json.dumps([{"theory": "no code at all"}])
        assert gemini_mod._parse_response(data) == []


# ===== Provider.__init__ tests =====

class TestGeminiProviderInit:

    def test_defaults(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key-123"}):
            p = gemini_mod.GeminiProvider()
        assert p._model == "gemini-2.5-flash"
        assert p._api_key == "env-key-123"

    def test_custom_model_and_key(self):
        p = gemini_mod.GeminiProvider(model="gemini-2.5-pro", api_key="my-key")
        assert p._model == "gemini-2.5-pro"
        assert p._api_key == "my-key"

    def test_api_key_arg_overrides_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}):
            p = gemini_mod.GeminiProvider(api_key="from-arg")
        assert p._api_key == "from-arg"

    def test_no_key_anywhere(self):
        with patch.dict(os.environ, {}, clear=True):
            p = gemini_mod.GeminiProvider()
        assert p._api_key == ""
