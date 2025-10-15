"""Google Vertex AI / Gemini provider adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import LLMRequest, LLMResponse, Message, ToolCall
from .base import Capability, Provider


class VertexProvider(Provider):
    name = "VertexAI"
    capabilities = Capability.CHAT | Capability.STREAM | Capability.TOOLS | Capability.JSON_MODE | Capability.EMBEDDINGS

    def headers(self, request: LLMRequest) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        if self._config.extra_headers:
            headers.update(self._config.extra_headers)
        return headers

    def build_payload(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        if operation == "embeddings":
            embedding_input = request.extra.get("input")
            if embedding_input is None and request.messages:
                embedding_input = [self._extract_text(message) for message in request.messages]
            texts: List[str] = []
            if isinstance(embedding_input, str):
                texts = [embedding_input]
            elif isinstance(embedding_input, (list, tuple)):
                texts = [str(item) for item in embedding_input]
            elif embedding_input is not None:
                texts = [str(embedding_input)]

            formatted_inputs = [{"parts": [{"text": text}]} for text in texts if text]
            payload: Dict[str, Any] = {"model": request.model}
            if not formatted_inputs:
                payload["content"] = {"parts": [{"text": ""}]}
            elif len(formatted_inputs) == 1:
                payload["content"] = formatted_inputs[0]
            else:
                payload["contents"] = formatted_inputs
            payload.update(kwargs)
            return payload

        contents = [self._format_message(message) for message in request.messages]
        payload: Dict[str, Any] = {
            "model": request.model,
            "contents": contents,
        }
        if request.tools:
            payload["tools"] = [self._format_tool(tool) for tool in request.tools]
        if stream:
            payload["stream"] = True
        payload.update({k: v for k, v in {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_tokens,
        }.items() if v is not None})
        payload.update(request.extra)
        payload.update(kwargs)
        return payload

    def endpoint(self, request: LLMRequest, *, stream: bool = False, operation: Optional[str] = None, **_: Any) -> str:
        if operation == "embeddings" or request.extra.get("operation") == "embeddings":
            return f"{self._config.base_url.rstrip('/')}/v1beta/models/{request.model}:embedContent"
        verb = ":streamGenerateContent" if stream else ":generateContent"
        return f"{self._config.base_url.rstrip('/')}/v1beta/models/{request.model}{verb}"

    def parse_response(self, payload: Dict[str, Any], *, request: LLMRequest) -> LLMResponse:
        if "embeddings" in payload:
            message = Message(role="assistant", content="")
            return LLMResponse(message=message, tool_calls=[], usage={}, raw=payload)

        candidates = payload.get("candidates", [])
        if not candidates:
            raise RuntimeError("unexpected Vertex response without candidates")
        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts") if isinstance(candidate.get("content"), dict) else candidate.get("content", [])
        text = self._parts_to_text(content_parts)
        message = Message(role="assistant", content=text)
        tool_calls = self._parse_tool_calls(candidate.get("content", {}))
        usage = payload.get("usageMetadata", {})
        return LLMResponse(message=message, tool_calls=tool_calls, usage=usage, raw=payload)

    @staticmethod
    def _format_message(message: Message) -> Dict[str, Any]:
        if message.role == "system":
            return {"role": "user", "parts": [{"text": VertexProvider._extract_text(message)}]}
        return {"role": message.role, "parts": [{"text": VertexProvider._extract_text(message)}]}

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
            "function_declarations": [{
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters or {},
            }]
        }

    @staticmethod
    def _parts_to_text(parts: Any) -> str:
        if not parts:
            return ""
        if isinstance(parts, list):
            return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if isinstance(parts, dict):
            return parts.get("text", "")
        return str(parts)

    @staticmethod
    def _parse_tool_calls(content: Any) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        parts = content.get("parts") if isinstance(content, dict) else []
        for part in parts:
            if not isinstance(part, dict) or part.get("functionCall") is None:
                continue
            function_call = part["functionCall"]
            tool_calls.append(
                ToolCall(
                    id=None,
                    name=function_call.get("name", ""),
                    arguments=function_call.get("args", {}),
                    raw=part,
                )
            )
        return tool_calls
