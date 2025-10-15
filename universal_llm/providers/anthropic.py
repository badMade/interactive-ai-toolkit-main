"""Anthropic Claude provider adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import LLMRequest, LLMResponse, Message, ToolCall
from .base import Capability, Provider


class AnthropicProvider(Provider):
    name = "Anthropic"
    capabilities = Capability.CHAT | Capability.STREAM | Capability.JSON_MODE | Capability.TOOLS

    def headers(self, request: LLMRequest) -> Dict[str, str]:
        headers = {
            "x-api-key": self.require_api_key(),
            "content-type": "application/json",
            "anthropic-version": request.extra.get("anthropic_version", "2023-06-01"),
        }
        if self._config.extra_headers:
            headers.update(self._config.extra_headers)
        return headers

    def build_payload(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        if operation == "embeddings":
            raise RuntimeError("Anthropic does not provide embeddings via this adapter")
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [self._format_message(message) for message in request.messages if message.role != "system"],
            "max_tokens": request.max_tokens or 1024,
        }
        system_messages = [message for message in request.messages if message.role == "system"]
        if system_messages:
            payload["system"] = "\n".join(self._extract_text(message) for message in system_messages)
        if request.tools:
            payload["tools"] = [self._format_tool(tool) for tool in request.tools]
        if stream:
            payload["stream"] = True
        payload.update({k: v for k, v in {
            "temperature": request.temperature,
            "top_p": request.top_p,
        }.items() if v is not None})
        payload.update(request.extra)
        payload.update(kwargs)
        return payload

    def endpoint(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **_: Any) -> str:
        return f"{self._config.base_url.rstrip('/')}/v1/messages"

    def parse_response(self, payload: Dict[str, Any], *, request: LLMRequest) -> LLMResponse:
        content = payload.get("content", [])
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        message = Message(role="assistant", content="\n".join(text_parts))
        tool_calls = self._parse_tool_calls(content)
        usage = payload.get("usage", {})
        return LLMResponse(message=message, tool_calls=tool_calls, usage=usage, raw=payload)

    @staticmethod
    def _format_message(message: Message) -> Dict[str, Any]:
        return {"role": message.role, "content": AnthropicProvider._extract_text(message)}

    @staticmethod
    def _extract_text(message: Message) -> str:
        if isinstance(message.content, list):
            fragments: List[str] = []
            for part in message.content:
                value = part.data.get("text") if isinstance(part.data, dict) else None
                if value:
                    fragments.append(value)
            return "\n".join(fragments)
        return str(message.content)

    @staticmethod
    def _format_tool(tool: Any) -> Dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.parameters or {},
        }

    @staticmethod
    def _parse_tool_calls(content: List[Dict[str, Any]]) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "tool_use":
                continue
            tool_calls.append(
                ToolCall(
                    id=part.get("id"),
                    name=part.get("name", ""),
                    arguments=part.get("input", {}),
                    raw=part,
                )
            )
        return tool_calls
