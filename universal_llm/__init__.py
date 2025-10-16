"""Unified LLM facade with provider auto-selection.

The :class:`LLM` class exposes a tiny surface area for synchronous and
asynchronous chat completions, streaming responses, and embeddings across
multiple providers.  Configuration is resolved in the following order:

1. Explicit dictionaries passed to :class:`LLM`.
2. A YAML/JSON file referenced via the ``UNIVERSAL_LLM_CONFIG`` environment variable.
3. Environment variables for each provider (for example ``OPENAI_API_KEY``).

The YAML/JSON configuration supports the shape::

    default_provider: openai
    model_overrides:
      my-custom-model: openai_compatible
    prefix_overrides:
      claude-: anthropic
    providers:
      openai:
        base_url: https://api.openai.com
        api_key: ${OPENAI_API_KEY}
        organization: ${OPENAI_ORG}
      azure_openai:
        base_url: https://contoso.openai.azure.com
        api_key: ${AZURE_OPENAI_KEY}
        deployment: gpt-4o
        rate_limit_per_minute: 600

Environment variables fallbacks include:

``OPENAI_API_KEY``
    Token for OpenAI or compatible services.
``OPENAI_BASE_URL``
    Overrides the default ``https://api.openai.com`` endpoint.
``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_KEY`` / ``AZURE_OPENAI_DEPLOYMENT``
    Azure OpenAI connection details.
``ANTHROPIC_API_KEY``
    Anthropic Claude API key; ``ANTHROPIC_BASE_URL`` optionally overrides the
    endpoint.
``GOOGLE_API_KEY`` or ``VERTEX_API_KEY``
    OAuth bearer token for Vertex AI Generative Language APIs.  ``VERTEX_BASE_URL``
    may override the default ``https://generativelanguage.googleapis.com``.
``OLLAMA_BASE_URL``
    Base URL for a locally running Ollama daemon.

The facade keeps provider clients lazy-initialized so that importing the
module never performs network I/O.  Sync and async entrypoints share the
same request normalization pipeline to keep behavior consistent.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Tuple, Union

try:  # pragma: no cover - optional dependency for YAML parsing
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .core import LLMRequest, LLMResponse, Message, ToolSpec, ContentPart
from .providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    VertexProvider,
)
from .providers.base import Provider, ProviderConfig, RetryConfig

__all__ = ["LLM"]


_PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "azure_openai": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "vertex": VertexProvider,
    "ollama": OllamaProvider,
    "openai_compatible": OpenAICompatibleProvider,
}

_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com",
    "azure_openai": "",
    "anthropic": "https://api.anthropic.com",
    "vertex": "https://generativelanguage.googleapis.com",
    "ollama": "http://127.0.0.1:11434",
    "openai_compatible": "https://api.openai.com",
}

_DEFAULT_PREFIX_HINTS = {
    "gpt-": "openai",
    "o1-": "openai",
    "text-": "openai",
    "claude-": "anthropic",
    "claude-3-5-sonnet": "anthropic",
    "opus-": "anthropic",
    "sonnet-": "anthropic",
    "gemini-": "vertex",
    "models/gemini": "vertex",
    "ollama:": "ollama",
}


@dataclass
class _ResolvedConfig:
    providers: Dict[str, Dict[str, Any]]
    default_provider: Optional[str]
    prefix_overrides: Dict[str, str]
    model_overrides: Dict[str, str]


class LLM:
    """Provider-agnostic facade consolidating synchronous and async workflows."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        raw_config = config or self._load_configuration()
        self._config = self._normalize_config(raw_config)
        self._provider_instances: Dict[str, Provider] = {}

    # Public sync API --------------------------------------------------
    def chat(
        self,
        model: str,
        messages: Iterable[Union[Message, Dict[str, Any]]],
        *,
        tools: Optional[Iterable[Union[ToolSpec, Dict[str, Any]]]] = None,
        provider: Optional[str] = None,
        **parameters: Any,
    ) -> LLMResponse:
        request, provider_args = self._build_request(model, messages, tools, parameters)
        provider_instance = self._select_provider(model, provider)
        return provider_instance.chat(request, **provider_args)

    def stream(
        self,
        model: str,
        messages: Iterable[Union[Message, Dict[str, Any]]],
        *,
        tools: Optional[Iterable[Union[ToolSpec, Dict[str, Any]]]] = None,
        provider: Optional[str] = None,
        **parameters: Any,
    ) -> Iterator[LLMResponse]:
        request, provider_args = self._build_request(model, messages, tools, parameters)
        provider_instance = self._select_provider(model, provider)
        return provider_instance.stream(request, **provider_args)

    def embed(
        self,
        model: str,
        inputs: Iterable[Union[Message, Dict[str, Any], str]],
        *,
        provider: Optional[str] = None,
        **parameters: Any,
    ) -> LLMResponse:
        messages = self._inputs_to_messages(inputs)
        request, provider_args = self._build_request(model, messages, tools=None, parameters=parameters)
        provider_instance = self._select_provider(model, provider)
        return provider_instance.embed(request, **provider_args)

    # Public async API -------------------------------------------------
    async def achat(
        self,
        model: str,
        messages: Iterable[Union[Message, Dict[str, Any]]],
        *,
        tools: Optional[Iterable[Union[ToolSpec, Dict[str, Any]]]] = None,
        provider: Optional[str] = None,
        **parameters: Any,
    ) -> LLMResponse:
        request, provider_args = self._build_request(model, messages, tools, parameters)
        provider_instance = self._select_provider(model, provider)
        return await provider_instance.achat(request, **provider_args)

    async def astream(
        self,
        model: str,
        messages: Iterable[Union[Message, Dict[str, Any]]],
        *,
        tools: Optional[Iterable[Union[ToolSpec, Dict[str, Any]]]] = None,
        provider: Optional[str] = None,
        **parameters: Any,
    ) -> AsyncIterator[LLMResponse]:
        request, provider_args = self._build_request(model, messages, tools, parameters)
        provider_instance = self._select_provider(model, provider)
        return provider_instance.astream(request, **provider_args)

    async def aembed(
        self,
        model: str,
        inputs: Iterable[Union[Message, Dict[str, Any], str]],
        *,
        provider: Optional[str] = None,
        **parameters: Any,
    ) -> LLMResponse:
        messages = self._inputs_to_messages(inputs)
        request, provider_args = self._build_request(model, messages, tools=None, parameters=parameters)
        provider_instance = self._select_provider(model, provider)
        return await provider_instance.aembed(request, **provider_args)

    # Internal helpers -------------------------------------------------
    def _build_request(
        self,
        model: str,
        messages: Iterable[Union[Message, Dict[str, Any]]],
        tools: Optional[Iterable[Union[ToolSpec, Dict[str, Any]]]],
        parameters: Dict[str, Any],
    ) -> Tuple[LLMRequest, Dict[str, Any]]:
        params = dict(parameters)
        temperature = params.pop("temperature", None)
        top_p = params.pop("top_p", None)
        max_tokens = params.pop("max_tokens", None)
        presence_penalty = params.pop("presence_penalty", None)
        frequency_penalty = params.pop("frequency_penalty", None)
        extra = params.pop("extra", {})
        provider_args = params.pop("provider_args", {})
        # Any remaining keyword arguments are forwarded via the request ``extra`` field.
        extra.update(params)

        normalized_messages = [self._ensure_message(message) for message in messages]
        normalized_tools = [self._ensure_tool(tool) for tool in tools] if tools else None
        request = LLMRequest(
            model=model,
            messages=normalized_messages,
            tools=normalized_tools,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra=extra,
        )
        return request, provider_args

    def _inputs_to_messages(self, inputs: Iterable[Union[Message, Dict[str, Any], str]]) -> List[Message]:
        messages: List[Message] = []
        for item in inputs:
            if isinstance(item, Message):
                messages.append(item)
            elif isinstance(item, str):
                messages.append(Message(role="user", content=item))
            elif isinstance(item, dict):
                messages.append(self._ensure_message(item))
            else:
                raise TypeError(f"Unsupported embedding input: {type(item)!r}")
        return messages

    def _ensure_message(self, message: Union[Message, Dict[str, Any]]) -> Message:
        if isinstance(message, Message):
            return message
        content = message.get("content")
        if isinstance(content, list):
            parts = [self._ensure_content_part(part) for part in content]
        else:
            parts = content
        return Message(role=message.get("role", "user"), content=parts)

    @staticmethod
    def _ensure_content_part(part: Union[ContentPart, Dict[str, Any]]) -> ContentPart:
        if isinstance(part, ContentPart):
            return part
        return ContentPart(type=part.get("type", "text"), data=part.get("data", {}))

    @staticmethod
    def _ensure_tool(tool: Union[ToolSpec, Dict[str, Any]]) -> ToolSpec:
        if isinstance(tool, ToolSpec):
            return tool
        return ToolSpec(name=tool["name"], description=tool.get("description"), parameters=tool.get("parameters", {}))

    def _select_provider(self, model: str, override: Optional[str]) -> Provider:
        provider_name = override or self._config.model_overrides.get(model)
        if provider_name is None:
            for prefix, name in {**_DEFAULT_PREFIX_HINTS, **self._config.prefix_overrides}.items():
                if model.startswith(prefix):
                    provider_name = name
                    break
        if provider_name is None:
            provider_name = self._config.default_provider or "openai"
        provider_name = provider_name.lower()
        if provider_name not in self._config.providers:
            raise RuntimeError(f"No configuration for provider '{provider_name}'")
        if provider_name not in self._provider_instances:
            self._provider_instances[provider_name] = self._instantiate_provider(provider_name)
        return self._provider_instances[provider_name]

    def _instantiate_provider(self, provider_name: str) -> Provider:
        provider_cls = _PROVIDER_CLASSES.get(provider_name)
        if provider_cls is None:
            raise RuntimeError(f"Unknown provider '{provider_name}'")
        settings = self._config.providers[provider_name]
        provider_config = self._build_provider_config(provider_name, settings)
        return provider_cls(provider_config)

    def _build_provider_config(self, provider_name: str, settings: Dict[str, Any]) -> ProviderConfig:
        base_url = settings.get("base_url") or _DEFAULT_BASE_URLS.get(provider_name)
        if not base_url:
            raise RuntimeError(f"Provider '{provider_name}' requires a base_url")
        api_key = settings.get("api_key")
        organization = settings.get("organization")
        deployment = settings.get("deployment")
        extra_headers = settings.get("extra_headers")
        timeout = float(settings.get("timeout", 30.0))
        rate_limit = settings.get("rate_limit_per_minute")
        retry_settings = settings.get("retry") or {}
        retry = RetryConfig(
            attempts=int(retry_settings.get("attempts", 3)),
            backoff_factor=float(retry_settings.get("backoff_factor", 2.0)),
            min_backoff=float(retry_settings.get("min_backoff", 0.5)),
            max_backoff=float(retry_settings.get("max_backoff", 10.0)),
            jitter=float(retry_settings.get("jitter", 0.1)),
        )
        return ProviderConfig(
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            deployment=deployment,
            extra_headers=extra_headers,
            timeout=timeout,
            rate_limit_per_minute=int(rate_limit) if rate_limit else None,
            retry=retry,
        )

    # Configuration ----------------------------------------------------
    def _load_configuration(self) -> Dict[str, Any]:
        config_path = os.getenv("UNIVERSAL_LLM_CONFIG")
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Configured UNIVERSAL_LLM_CONFIG file not found: {config_path}")
            return self._read_config_file(path)
        return self._env_configuration()

    def _read_config_file(self, path: Path) -> Dict[str, Any]:
        content = path.read_text()
        # Try JSON first for environments that do not bundle PyYAML.
        try:
            return self._expand_env(json.loads(content))
        except json.JSONDecodeError:
            if yaml is None:
                raise RuntimeError("YAML configuration requires the optional PyYAML dependency")
            data = yaml.safe_load(content)  # type: ignore[no-any-return]
            return self._expand_env(data)

    def _env_configuration(self) -> Dict[str, Any]:
        providers: Dict[str, Dict[str, Any]] = {}
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            providers["openai"] = {
                "api_key": openai_key,
                "base_url": os.getenv("OPENAI_BASE_URL", _DEFAULT_BASE_URLS["openai"]),
                "organization": os.getenv("OPENAI_ORG"),
            }
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        if azure_key:
            providers["azure_openai"] = {
                "api_key": azure_key,
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            }
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            providers["anthropic"] = {
                "api_key": anthropic_key,
                "base_url": os.getenv("ANTHROPIC_BASE_URL", _DEFAULT_BASE_URLS["anthropic"]),
            }
        vertex_key = os.getenv("VERTEX_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if vertex_key:
            providers["vertex"] = {
                "api_key": vertex_key,
                "base_url": os.getenv("VERTEX_BASE_URL", _DEFAULT_BASE_URLS["vertex"]),
            }
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            providers["ollama"] = {
                "base_url": ollama_url,
                "api_key": None,
            }
        return {"providers": providers}

    def _normalize_config(self, config: Dict[str, Any]) -> _ResolvedConfig:
        providers_raw = config.get("providers", {})
        providers: Dict[str, Dict[str, Any]] = {}
        for name, settings in providers_raw.items():
            providers[name.lower()] = dict(settings)
            providers[name.lower()].setdefault("base_url", _DEFAULT_BASE_URLS.get(name.lower(), ""))
        default_provider = config.get("default_provider")
        prefix_overrides = {k: v for k, v in config.get("prefix_overrides", {}).items()}
        model_overrides = {k: v for k, v in config.get("model_overrides", {}).items()}
        return _ResolvedConfig(
            providers=providers,
            default_provider=default_provider.lower() if isinstance(default_provider, str) else default_provider,
            prefix_overrides={k: v.lower() for k, v in prefix_overrides.items()},
            model_overrides={k: v.lower() for k, v in model_overrides.items()},
        )

    def _expand_env(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._expand_env(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._expand_env(item) for item in value]
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value
