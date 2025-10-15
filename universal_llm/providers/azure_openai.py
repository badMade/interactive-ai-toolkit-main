"""Azure OpenAI provider adapter."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..core import LLMRequest, LLMResponse
from .openai import OpenAIProvider
from .base import ProviderConfig


class AzureOpenAIProvider(OpenAIProvider):
    name = "AzureOpenAI"

    def headers(self, request: LLMRequest) -> Dict[str, str]:
        headers = super().headers(request)
        # Azure expects the key via `api-key` header.
        headers.pop("Authorization", None)
        headers["api-key"] = self.require_api_key()
        return headers

    def endpoint(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **kwargs: Any) -> str:
        deployment = request.extra.get("deployment") or self._config.deployment or request.model
        api_version = request.extra.get("api_version") or kwargs.get("api_version") or "2024-02-15-preview"
        base = self._config.base_url.rstrip("/")
        if operation == "embeddings" or request.extra.get("operation") == "embeddings":
            return f"{base}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
        return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    def parse_response(self, payload: Dict[str, Any], *, request: LLMRequest) -> LLMResponse:
        # Azure OpenAI mirrors the OpenAI schema, so reuse parent implementation.
        return super().parse_response(payload, request=request)
