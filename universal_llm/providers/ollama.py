"""Ollama provider adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import LLMRequest, LLMResponse, Message
from .base import Capability, Provider


class OllamaProvider(Provider):
    name = "Ollama"
    capabilities = Capability.CHAT | Capability.STREAM | Capability.EMBEDDINGS

    def headers(self, request: LLMRequest) -> Dict[str, str]:  # noqa: D401 - simple header mapping
        headers = {"Content-Type": "application/json"}
        if self._config.extra_headers:
            headers.update(self._config.extra_headers)
        return headers

    def build_payload(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        if operation == "embeddings":
            embedding_input = request.extra.get("input")
            if embedding_input is None and request.messages:
                embedding_input = [self._message_to_text(message) for message in request.messages]
            payload = {"model": request.model, "input": embedding_input}
            payload.update(kwargs)
            return payload

        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [self._format_message(message) for message in request.messages],
            "stream": stream,
        }
        options: Dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
        payload.update(request.extra)
        payload.update(kwargs)
        return payload

    def endpoint(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **_: Any) -> str:
        if operation == "embeddings" or request.extra.get("operation") == "embeddings":
            return f"{self._config.base_url.rstrip('/')}/api/embed"
        return f"{self._config.base_url.rstrip('/')}/api/chat"

    def parse_response(self, payload: Dict[str, Any], *, request: LLMRequest) -> LLMResponse:
        if "embeddings" in payload:
            message = Message(role="assistant", content="")
            return LLMResponse(message=message, tool_calls=[], usage={}, raw=payload)
        message_payload = payload.get("message", {})
        role = message_payload.get("role", "assistant")
        content = message_payload.get("content", "")
        message = Message(role=role, content=content)
        usage = payload.get("eval_count", {})
        return LLMResponse(message=message, tool_calls=[], usage={"eval_count": usage}, raw=payload)

    @staticmethod
    def _format_message(message: Message) -> Dict[str, Any]:
        return {"role": message.role, "content": OllamaProvider._message_to_text(message)}

    @staticmethod
    def _message_to_text(message: Message) -> str:
        if isinstance(message.content, list):
            parts: List[str] = []
            for part in message.content:
                value = part.data.get("text") if isinstance(part.data, dict) else None
                if value:
                    parts.append(value)
            return "\n".join(parts)
        return str(message.content)
