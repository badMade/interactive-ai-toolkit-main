"""Generic OpenAI compatible provider adapter."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..core import LLMRequest, LLMResponse
from .openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    name = "OpenAICompatible"

    def endpoint(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **kwargs: Any) -> str:
        base = self._config.base_url.rstrip("/")
        if operation == "embeddings" or request.extra.get("operation") == "embeddings":
            suffix = request.extra.get("embeddings_path", "/v1/embeddings")
        else:
            suffix = request.extra.get("chat_path", "/v1/chat/completions")
        return f"{base}{suffix}"

    def parse_response(self, payload: Dict[str, Any], *, request: LLMRequest) -> LLMResponse:
        return super().parse_response(payload, request=request)
