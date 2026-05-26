"""Gemini provider for hypothesis generation."""

from __future__ import annotations

import json
import os
from typing import Any

from .base import Hypothesis, HypothesisRequest, Provider

_SYSTEM_PROMPT = """\
You are a quantitative trading researcher. You generate trading signal hypotheses \
as executable Python code.

Each hypothesis must define a function `get_feature(df)` that takes a pandas DataFrame \
with columns [open, high, low, close, volume] and a DatetimeIndex, and returns a \
pandas Series of float values. Values > 0.5 indicate a trade entry signal.

Available in the execution environment: pd (pandas), np (numpy), and the DataFrame \
columns as standalone variables (close, high, low, open, volume).

Rules:
- Use only pandas/numpy operations on OHLCV data
- Return a Series with the same index as the input DataFrame
- The signal should be based on a clear, testable market hypothesis
- Avoid lookahead bias: only use data available up to the current bar
- Each hypothesis should be meaningfully different from others
"""


def _build_user_prompt(request: HypothesisRequest) -> str:
    parts = [
        f"Generate {request.n_hypotheses} trading signal hypotheses for {request.symbol}.",
        "",
        f"OHLCV summary stats: {json.dumps(request.ohlcv_stats, default=str)}",
    ]

    if request.saturated_theories:
        parts.append("")
        parts.append(
            "AVOID these theory families (already tested, didn't work):\n- "
            + "\n- ".join(request.saturated_theories[-20:])
        )

    if request.hall_of_fame:
        parts.append("")
        parts.append(
            "These approaches have shown promise (build on them or try variations):\n- "
            + "\n- ".join(request.hall_of_fame[-10:])
        )

    parts.append("")
    parts.append(
        "Respond with a JSON array of objects, each with keys: "
        '"code" (the Python code string), "theory" (1-2 sentence explanation), '
        '"direction" ("long" or "short"). '
        "IMPORTANT: In the code field, use \\\\n for newlines — do NOT use literal "
        "newlines inside JSON string values. "
        "Return ONLY the JSON array, no markdown fences or other text."
    )
    return "\n".join(parts)


class GeminiProvider(Provider):

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

    async def generate(self, request: HypothesisRequest) -> list[Hypothesis]:
        from google import genai

        client = genai.Client(api_key=self._api_key)
        user_prompt = _build_user_prompt(request)

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )

        return _parse_response(response.text or "")


def _fix_json(text: str) -> str:
    """Best-effort cleanup of LLM JSON output."""
    import re
    s = text.strip()
    # strip markdown fences
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1])
    # remove // comments
    s = re.sub(r"//[^\n]*", "", s)
    # strip trailing commas
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # escape literal newlines/tabs inside JSON strings
    result = []
    in_string = False
    i = 0
    while i < len(s):
        ch = s[i]
        # check for escaped character (skip next char)
        if in_string and ch == '\\' and i + 1 < len(s):
            result.append(ch)
            result.append(s[i + 1])
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _parse_response(text: str) -> list[Hypothesis]:
    cleaned = _fix_json(text)

    raw: list[dict[str, Any]] = json.loads(cleaned)
    results: list[Hypothesis] = []

    for item in raw:
        code = item.get("code", "")
        theory = item.get("theory", "")
        direction = item.get("direction", "long")
        if code and "get_feature" in code:
            results.append(Hypothesis(code=code, theory=theory, direction=direction))

    return results
