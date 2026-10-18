"""Thin wrapper around the Anthropic SDK: one function that returns a validated
Pydantic object plus a usage record (tokens, cost, latency) for tracking."""

import time
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from agent.config import MODEL

T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMRefusal(Exception):
    pass


@lru_cache(maxsize=1)
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def structured(
    system: str,
    prompt: str,
    output: type[T],
    *,
    step: str,
    model: str = MODEL,
    effort: str = "medium",
    max_tokens: int = 16000,
) -> tuple[T, dict]:
    """Ask the model for an answer that matches the `output` schema."""
    start = time.perf_counter()
    response = client().messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=output,
        output_config={"effort": effort},
    )
    latency = time.perf_counter() - start
    if response.stop_reason == "refusal":
        raise LLMRefusal(f"{step}: model declined the request")
    if response.parsed_output is None:
        raise ValueError(f"{step}: no structured output (stop_reason={response.stop_reason})")

    usage = response.usage
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    record = {
        "step": step,
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6,
        "latency_s": round(latency, 2),
    }
    return response.parsed_output, record
