"""Provider abstractions for the universal LLM facade."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from ..core import (
    AsyncHttpClient,
    LLMRequest,
    LLMResponse,
    RateLimiter,
    RetryConfig,
    SyncHttpClient,
    with_retry,
    with_retry_async,
)


class Capability(IntFlag):
    """Features that a provider can expose."""

    CHAT = auto()
    STREAM = auto()
    TOOLS = auto()
    JSON_MODE = auto()
    EMBEDDINGS = auto()


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration shared by concrete providers."""

    base_url: str
    api_key: Optional[str]
    organization: Optional[str] = None
    deployment: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None
    timeout: float = 30.0
    rate_limit_per_minute: Optional[int] = None
    retry: RetryConfig = RetryConfig()


class Provider(ABC):
    """Abstract provider API that concrete implementations must satisfy."""

    name: str
    capabilities: Capability

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._sync_client = SyncHttpClient(timeout=config.timeout)
        self._async_client = AsyncHttpClient(timeout=config.timeout)
        self._rate_limiter: Optional[RateLimiter]
        if config.rate_limit_per_minute:
            rate = config.rate_limit_per_minute / 60.0
            self._rate_limiter = RateLimiter(rate=rate, capacity=config.rate_limit_per_minute)
        else:
            self._rate_limiter = None

    @abstractmethod
    def build_payload(self, request: LLMRequest, *, stream: bool = False, **kwargs: Any) -> Dict[str, Any]:
        """Translate the normalized request into the provider specific payload."""

    @abstractmethod
    def parse_response(self, payload: Dict[str, Any], *, request: LLMRequest) -> LLMResponse:
        """Normalize the provider response into :class:`LLMResponse`."""

    @abstractmethod
    def endpoint(self, request: LLMRequest, *, stream: bool = False, **kwargs: Any) -> str:
        """Return the absolute endpoint URL for the given request."""

    def headers(self, request: LLMRequest) -> Dict[str, str]:
        headers: Dict[str, str] = {"Authorization": f"Bearer {self.require_api_key()}"}
        if self._config.organization:
            headers["OpenAI-Organization"] = self._config.organization
        if self._config.extra_headers:
            headers.update(self._config.extra_headers)
        return headers

    def require_api_key(self) -> str:
        if not self._config.api_key:
            raise RuntimeError(f"{self.name} requires an API key")
        return self._config.api_key

    def _maybe_rate_limit(self) -> None:
        if self._rate_limiter:
            self._rate_limiter.acquire()

    async def _maybe_rate_limit_async(self) -> None:
        if self._rate_limiter:
            await self._rate_limiter.acquire_async()

    # Sync entrypoints -------------------------------------------------
    def chat(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        self._maybe_rate_limit()
        payload = self.build_payload(request, stream=False, **kwargs)
        call = with_retry(self._config.retry)(self._sync_client.request)
        response_payload = call("POST", self.endpoint(request), headers=self.headers(request), json_body=payload)
        return self.parse_response(response_payload, request=request)

    def stream(self, request: LLMRequest, **kwargs: Any) -> Iterator[LLMResponse]:
        if Capability.STREAM not in self.capabilities:
            raise RuntimeError(f"{self.name} does not support streaming")
        self._maybe_rate_limit()
        payload = self.build_payload(request, stream=True, **kwargs)
        stream = self._sync_client.stream(
            "POST",
            self.endpoint(request, stream=True),
            headers=self.headers(request),
            json_body=payload,
        )
        for chunk in stream:
            if not chunk:
                continue
            yield self.parse_response(chunk, request=request)

    def embed(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        if Capability.EMBEDDINGS not in self.capabilities:
            raise RuntimeError(f"{self.name} does not support embeddings")
        self._maybe_rate_limit()
        payload = self.build_payload(request, stream=False, operation="embeddings", **kwargs)
        call = with_retry(self._config.retry)(self._sync_client.request)
        response_payload = call(
            "POST",
            self.endpoint(request, stream=False, operation="embeddings", **kwargs),
            headers=self.headers(request),
            json_body=payload,
        )
        return self.parse_response(response_payload, request=request)

    # Async entrypoints ------------------------------------------------
    async def achat(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        await self._maybe_rate_limit_async()
        payload = self.build_payload(request, stream=False, **kwargs)
        call = with_retry_async(self._config.retry)(self._async_client.request)
        response_payload = await call("POST", self.endpoint(request), headers=self.headers(request), json_body=payload)
        return self.parse_response(response_payload, request=request)

    async def astream(self, request: LLMRequest, **kwargs: Any) -> AsyncIterator[LLMResponse]:
        if Capability.STREAM not in self.capabilities:
            raise RuntimeError(f"{self.name} does not support streaming")
        await self._maybe_rate_limit_async()
        payload = self.build_payload(request, stream=True, **kwargs)
        async for chunk in self._async_client.stream(
            "POST",
            self.endpoint(request, stream=True),
            headers=self.headers(request),
            json_body=payload,
        ):
            if not chunk:
                continue
            yield self.parse_response(chunk, request=request)

    async def aembed(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        if Capability.EMBEDDINGS not in self.capabilities:
            raise RuntimeError(f"{self.name} does not support embeddings")
        await self._maybe_rate_limit_async()
        payload = self.build_payload(request, stream=False, operation="embeddings", **kwargs)
        call = with_retry_async(self._config.retry)(self._async_client.request)
        response_payload = await call(
            "POST",
            self.endpoint(request, stream=False, operation="embeddings", **kwargs),
            headers=self.headers(request),
            json_body=payload,
        )
        return self.parse_response(response_payload, request=request)
