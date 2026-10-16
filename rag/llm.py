"""LLM client protocol and the OpenAI implementation.

`ChatEngine`, `build_index` and the conflict pre-scan depend only on the `LLM`
protocol, so tests inject fakes and never touch the network.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, Sequence

from rag.config import EMBED_MODEL

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when a model call fails or returns unusable output."""


class LLM(Protocol):
    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIClient:
    """Thin wrapper over the openai SDK (verified against openai==3.31.0).

    chat_json: chat.completions.create with temperature 0 and a strict
    json_schema response_format; one retry on API/parse/refusal errors.
    embed: one batched embeddings.create call, results ordered by `index`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        embed_model: str = EMBED_MODEL,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout, max_retries=2)
        self._client = client
        self.embed_model = embed_model

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format=response_format,
                )
                message = resp.choices[0].message
                if getattr(message, "refusal", None):
                    raise LLMError("model refused the request")
                data = json.loads(message.content or "")
                if not isinstance(data, dict):
                    raise LLMError("model output is not a JSON object")
                return data
            except Exception as exc:  # API error, JSON error or refusal
                last_error = exc
                log.warning(
                    "chat_json %s attempt %d failed: %s",
                    schema_name,
                    attempt + 1,
                    type(exc).__name__,
                )
        raise LLMError(f"chat_json {schema_name} failed") from last_error

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        try:
            resp = self._client.embeddings.create(model=self.embed_model, input=texts)
        except Exception as exc:
            raise LLMError("embedding request failed") from exc
        items = sorted(resp.data, key=lambda d: d.index)
        return [list(d.embedding) for d in items]
