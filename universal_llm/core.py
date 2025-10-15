"""Core data structures and helpers for the universal LLM facade.

This module provides the pydantic models that describe LLM requests and
responses.  It also ships with utility helpers for resilient HTTP
communication, safe JSON parsing, retry/backoff semantics, and
cooperative rate limiting primitives that can be reused by provider
implementations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

try:
    from pydantic import BaseModel, Field, validator
    try:  # pragma: no cover - optional in pydantic<2
        from pydantic import ConfigDict
    except ImportError:  # pragma: no cover - legacy branch
        ConfigDict = None  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "pydantic is required for universal_llm. Please install it via the project's\n"
        "requirements.txt or add it as a dependency."
    ) from exc

try:  # pragma: no cover - optional dependency
    import httpx
except Exception:  # pragma: no cover - fallback path if httpx is unavailable
    httpx = None  # type: ignore


_LOGGER = logging.getLogger(__name__)


class ImmutableModel(BaseModel):
    """Base class that freezes models for both Pydantic v1 and v2."""

    if "ConfigDict" in globals() and ConfigDict is not None:  # pragma: no cover - version specific
        model_config = ConfigDict(frozen=True)  # type: ignore[assignment]
    else:  # pragma: no cover - version specific
        class Config:
            frozen = True
            allow_mutation = False


class ContentPart(ImmutableModel):
    """A multimodal content fragment within a message."""

    type: str
    data: Dict[str, Any] = Field(default_factory=dict)

class Message(ImmutableModel):
    """Represents a single conversational turn."""

    role: str
    content: Union[str, List[ContentPart]]

    @validator("role")
    def _role_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("message role must not be empty")
        return value


class ToolSpec(ImmutableModel):
    """Describes a tool/function that the LLM can invoke."""

    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator("name")
    def _name_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("tool name must not be empty")
        return value


class ToolCall(ImmutableModel):
    """A tool invocation emitted by a model."""

    id: Optional[str] = None
    name: str
    arguments: Union[str, Dict[str, Any]]
    raw: Optional[Dict[str, Any]] = None

class LLMRequest(ImmutableModel):
    """A normalized request across all providers."""

    model: str
    messages: List[Message]
    tools: Optional[List[ToolSpec]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class LLMResponse(ImmutableModel):
    """A normalized response across all providers."""

    message: Message
    tool_calls: List[ToolCall] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None



def safe_json(data: Union[str, bytes, Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    """Parse JSON content without raising unexpected exceptions.

    The helper gracefully handles common error scenarios.  If JSON parsing
    fails, an empty dictionary is returned to keep downstream code robust
    while logging the issue for observability.
    """

    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"values": data}
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if not data:
            return {}
        return json.loads(data)
    except json.JSONDecodeError:  # pragma: no cover - defensive branch
        _LOGGER.debug("failed to decode json payload", exc_info=True)
        return {}


_T = TypeVar("_T")


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retrying transient failures."""

    attempts: int = 3
    backoff_factor: float = 2.0
    min_backoff: float = 0.5
    max_backoff: float = 10.0
    jitter: float = 0.1
    retriable: Tuple[type, ...] = (Exception,)


def _sleep(duration: float) -> None:
    time.sleep(duration)


async def _asleep(duration: float) -> None:
    await asyncio.sleep(duration)


def with_retry(config: RetryConfig) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator applying retry/backoff to a synchronous function."""

    def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            last_error: Optional[Exception] = None
            for attempt in range(1, config.attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retriable as error:  # type: ignore[misc]
                    last_error = error
                    if attempt == config.attempts:
                        raise
                    delay = _compute_backoff(config, attempt)
                    _LOGGER.debug("retrying %s in %.2fs", func.__name__, delay, exc_info=error)
                    _sleep(delay)
            assert last_error is not None  # pragma: no cover - for mypy only
            raise last_error

        return wrapper

    return decorator


def with_retry_async(config: RetryConfig) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Decorator applying retry/backoff to an async function."""

    def decorator(func: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            last_error: Optional[Exception] = None
            for attempt in range(1, config.attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retriable as error:  # type: ignore[misc]
                    last_error = error
                    if attempt == config.attempts:
                        raise
                    delay = _compute_backoff(config, attempt)
                    _LOGGER.debug("retrying %s in %.2fs", func.__name__, delay, exc_info=error)
                    await _asleep(delay)
            assert last_error is not None  # pragma: no cover - for mypy only
            raise last_error

        return wrapper

    return decorator


def _compute_backoff(config: RetryConfig, attempt: int) -> float:
    base_delay = config.min_backoff * math.pow(config.backoff_factor, attempt - 1)
    if config.jitter:
        base_delay += random.uniform(0, config.jitter)
    return min(base_delay, config.max_backoff)


class RateLimiter:
    """Token bucket rate limiter supporting sync and async flows."""

    def __init__(self, rate: float, capacity: Optional[int] = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate
        self._capacity = capacity or max(1, int(rate))
        self._tokens = float(self._capacity)
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self._async_lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                remaining = (tokens - self._tokens) / self._rate
            _sleep(remaining)

    async def acquire_async(self, tokens: float = 1.0) -> None:
        while True:
            async with self._async_lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                remaining = (tokens - self._tokens) / self._rate
            await _asleep(remaining)


def _yield_sse_events(lines: Iterator[str]) -> Iterator[Dict[str, Any]]:
    """Convert an iterator of SSE lines into parsed JSON payloads."""

    data_lines: List[str] = []
    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if not line:
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            if data.strip() == "[DONE]":
                break
            parsed = safe_json(data)
            if parsed:
                yield parsed
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        data = "\n".join(data_lines)
        if data.strip() != "[DONE]":
            parsed = safe_json(data)
            if parsed:
                yield parsed


async def _async_yield_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[Dict[str, Any]]:
    """Asynchronous companion to :func:`_yield_sse_events`."""

    data_lines: List[str] = []
    async for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if not line:
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            if data.strip() == "[DONE]":
                break
            parsed = safe_json(data)
            if parsed:
                yield parsed
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        data = "\n".join(data_lines)
        if data.strip() != "[DONE]":
            parsed = safe_json(data)
            if parsed:
                yield parsed


class SyncHttpClient:
    """Light-weight HTTP client wrapper that gracefully handles dependencies."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def request(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
                params: Optional[Dict[str, Any]] = None, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if httpx is not None:  # pragma: no cover - depends on environment
            response = httpx.request(method, url, headers=headers, params=params, json=json_body, timeout=self._timeout)
            response.raise_for_status()
            return safe_json(response.text)
        # Fallback to urllib to avoid hard dependency.
        from urllib import request as urllib_request
        from urllib.error import HTTPError
        from urllib.parse import urlencode

        data_bytes: Optional[bytes] = None
        req_headers = headers or {}
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(params)}"
        if json_body is not None:
            data_bytes = json.dumps(json_body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        req = urllib_request.Request(full_url, data=data_bytes, headers=req_headers, method=method.upper())
        try:
            with urllib_request.urlopen(req, timeout=self._timeout) as response:  # type: ignore[arg-type]
                return safe_json(response.read())
        except HTTPError as error:
            payload = error.read()
            raise RuntimeError(f"HTTP {error.code} calling {url}: {payload}") from error

    def stream(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
               params: Optional[Dict[str, Any]] = None, json_body: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        if httpx is None:
            raise RuntimeError("Streaming requests require the optional 'httpx' dependency")
        with httpx.stream(method, url, headers=headers, params=params, json=json_body, timeout=self._timeout) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                yield from _yield_sse_events(response.iter_lines())
            else:
                payload = response.read()
                parsed = safe_json(payload)
                if parsed:
                    yield parsed


class AsyncHttpClient:
    """Asynchronous companion for :class:`SyncHttpClient`."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def request(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
                      params: Optional[Dict[str, Any]] = None, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if httpx is None:
            raise RuntimeError("Async HTTP requests require the optional 'httpx' dependency")
        async with httpx.AsyncClient(timeout=self._timeout) as client:  # pragma: no cover - network
            response = await client.request(method, url, headers=headers, params=params, json=json_body)
            response.raise_for_status()
            return safe_json(response.text)

    async def stream(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
                     params: Optional[Dict[str, Any]] = None, json_body: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        if httpx is None:
            raise RuntimeError("Streaming requests require the optional 'httpx' dependency")
        async with httpx.AsyncClient(timeout=self._timeout) as client:  # pragma: no cover - network
            async with client.stream(method, url, headers=headers, params=params, json=json_body) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    async for chunk in _async_yield_sse_events(response.aiter_lines()):
                        yield chunk
                else:
                    payload = await response.aread()
                    parsed = safe_json(payload)
                    if parsed:
                        yield parsed
