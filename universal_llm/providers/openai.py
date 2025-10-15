"""OpenAI provider adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import LLMRequest, LLMResponse, Message, ToolCall
from .base import Capability, Provider, ProviderConfig


class OpenAIProvider(Provider):
    name = "OpenAI"
    capabilities = Capability.CHAT | Capability.STREAM | Capability.TOOLS | Capability.JSON_MODE | Capability.EMBEDDINGS

    def build_payload(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": request.model}
        payload.update({k: v for k, v in {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }.items() if v is not None})
        payload.update(request.extra)
        payload.update(kwargs)

        if operation == "embeddings":
            embedding_input = payload.pop("input", None) or request.extra.get("input")
            if embedding_input is None and request.messages:
                embedding_input = [self._message_to_plain_text(msg) for msg in request.messages]
            payload = {"model": request.model, "input": embedding_input}
            return payload

        payload["messages"] = [self._format_message(message) for message in request.messages]
        if request.tools:
            payload["tools"] = [self._format_tool(tool) for tool in request.tools]
        if stream:
            payload["stream"] = True
        return payload

    def endpoint(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **_: Any) -> str:
        if operation == "embeddings" or request.extra.get("operation") == "embeddings":
            return f"{self._config.base_url.rstrip('/')}/v1/embeddings"
        return f"{self._config.base_url.rstrip('/')}/v1/chat/completions"

    def parse_response(self, payload: Dict[str, Any], *, request: LLMRequest) -> LLMResponse:
        if "data" in payload and isinstance(payload["data"], list):
            # Embedding response format
            usage = payload.get("usage", {})
            message = Message(role="assistant", content="")
            return LLMResponse(message=message, tool_calls=[], usage={**usage, "embeddings": payload.get("data", [])}, raw=payload)

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("unexpected OpenAI response without choices")
        choice = choices[0]
        message_payload = choice.get("message") or {}
        message = Message(role=message_payload.get("role", "assistant"), content=message_payload.get("content", ""))
        tool_calls = self._parse_tool_calls(message_payload.get("tool_calls"))
        usage = payload.get("usage", {})
        return LLMResponse(message=message, tool_calls=tool_calls, usage=usage, raw=payload)

    @staticmethod
    def _format_message(message: Message) -> Dict[str, Any]:
        if isinstance(message.content, list):
            serialized: List[Dict[str, Any]] = []
            for part in message.content:
                if hasattr(part, "model_dump"):
                    serialized.append(part.model_dump())  # type: ignore[attr-defined]
                else:  # pragma: no cover - pydantic v1 fallback
                    serialized.append(part.dict())  # type: ignore[attr-defined]
            content = serialized
        else:
            content = message.content
        return {"role": message.role, "content": content}

    @staticmethod
    def _format_tool(tool: Any) -> Dict[str, Any]:
        parameters = tool.parameters or {}
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            },
        }

    @staticmethod
    def _parse_tool_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> List[ToolCall]:
        if not tool_calls:
            return []
        normalized: List[ToolCall] = []
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            arguments = function.get("arguments")
            normalized.append(
                ToolCall(
                    id=call.get("id") if isinstance(call, dict) else None,
                    name=function.get("name", ""),
                    arguments=arguments or {},
                    raw=call if isinstance(call, dict) else None,
                )
            )
        return normalized

    @staticmethod
    def _message_to_plain_text(message: Message) -> str:
        if isinstance(message.content, list):
            texts = []
            for part in message.content:
                value = part.data.get("text") if isinstance(part.data, dict) else None
                if value:
                    texts.append(value)
            return "\n".join(texts)
        return str(message.content)
