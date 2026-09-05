"""Model factory — any OpenAI-compatible provider from a spec string.

Groq, Moonshot and OpenRouter all implement OpenAI's chat completions API
including `tools`, so `ChatOpenAI(base_url=...)` reaches all of them with no extra
dependency.

Spec format: "<provider>:<model>" or bare "<model>" (defaults to openai).
    gpt-4.1
    openai:gpt-5-mini
    groq:llama-3.3-70b-versatile
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from config import MAX_OUTPUT_TOKENS

load_dotenv()


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str
    base_url: str | None
    signup: str


PROVIDERS: dict[str, Provider] = {
    "openai": Provider("openai", "OPENAI_API_KEY", None,
                       "https://platform.openai.com/api-keys"),
    "groq": Provider("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
                     "https://console.groq.com/keys"),
    "moonshot": Provider("moonshot", "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1",
                         "https://platform.moonshot.ai/console/api-keys"),
    "openrouter": Provider("openrouter", "OPENROUTER_API_KEY",
                           "https://openrouter.ai/api/v1", "https://openrouter.ai/keys"),
}


class MissingKeyError(RuntimeError):
    """The provider implied by a spec has no key configured."""


def parse_spec(spec: str) -> tuple[Provider, str]:
    if ":" in spec:
        head, _, tail = spec.partition(":")
        if head in PROVIDERS:
            return PROVIDERS[head], tail
    return PROVIDERS["openai"], spec


def available(spec: str) -> bool:
    provider, _ = parse_spec(spec)
    return bool(os.environ.get(provider.key_env, "").strip())


def build_model(spec: str, **kwargs: Any):
    """Build a chat model for `spec`.

    `temperature` is passed only when the caller asks: OpenAI's reasoning models
    reject an explicit temperature while Groq's Llama models accept it.
    """
    from langchain_openai import ChatOpenAI

    provider, model = parse_spec(spec)
    key = os.environ.get(provider.key_env, "").strip()
    if not key:
        raise MissingKeyError(
            f"{spec!r} needs {provider.key_env}. Add it to .env "
            f"(get one at {provider.signup})."
        )

    init: dict[str, Any] = {"model": model, "api_key": key, **kwargs}
    if provider.base_url:
        init["base_url"] = provider.base_url
    init.setdefault("timeout", 180.0)
    # A 429 must pace the run, not end it: a four-specialist orchestrator on a
    # low TPM tier will hit the ceiling mid-run, and the SDK honours retry-after.
    init.setdefault("max_retries", 8)
    # Cap output explicitly. Left unset, a provider reserves the model's entire
    # output budget per request, and Groq then refuses the call outright against
    # its output-tokens-per-minute allowance — "Request too large ... on output
    # tokens per minute" before a single token is generated. It is also a real
    # cost control: output is priced several times higher than input, so an
    # uncapped call is an unbounded one.
    init.setdefault("max_tokens", MAX_OUTPUT_TOKENS)
    return ChatOpenAI(**init)
