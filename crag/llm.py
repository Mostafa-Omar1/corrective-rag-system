"""Thin wrapper around the Claude API returning schema-validated JSON.

When no API key is configured `get_llm()` returns None and every component
falls back to a deterministic heuristic, so the app still works offline.
"""
from __future__ import annotations

import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class ClaudeLLM:
    def __init__(self, model: str, effort: str = "low", api_key: str | None = None):
        self.model = model
        self.effort = effort
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def json(self, system: str, prompt: str, schema: dict, max_tokens: int = 4000) -> dict:
        """Call Claude and return the JSON object that matches `schema`."""
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                # On a safety-classifier decline, re-run on Anthropic's recommended model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Invalid Anthropic API key.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the Anthropic API, try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API.") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The model response was truncated (max_tokens).")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned invalid JSON.") from exc


def get_llm(model: str, effort: str = "low", api_key: str | None = None) -> ClaudeLLM | None:
    key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if not key:
        return None
    return ClaudeLLM(model, effort, api_key=api_key)
