"""
llm/client.py
==============

HTTP client for talking to a local llama.cpp server (or any other
OpenAI-compatible chat completion endpoint).

llama.cpp's built-in server (`llama-server`) exposes an OpenAI-compatible
API at `/v1/chat/completions`, supporting both streaming (SSE,
`stream: true`) and non-streaming responses. This module wraps that API
with:

    * A simple synchronous, generator-based streaming interface
      (`stream_chat`) that yields text deltas as they arrive, so the UI
      can render tokens live.
    * A non-streaming convenience method (`chat`) for cases (like the
      planner) where we want the full response at once.
    * Robust error handling for connection failures, timeouts, and
      malformed responses -- since a local server may not be running yet,
      or may be mid-restart, this needs to fail informatively rather than
      crashing the whole TUI.

Only the `requests` library is used (no `openai` SDK dependency), to keep
the install footprint minimal for Termux / low-memory devices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import requests

from config import ModelConfig


class LLMConnectionError(Exception):
    """Raised when the LLM server cannot be reached at all."""


class LLMResponseError(Exception):
    """Raised when the LLM server responds, but with an error or malformed body."""


@dataclass
class ChatChunk:
    """A single piece of a streamed chat response."""

    delta: str = ""
    finished: bool = False
    finish_reason: Optional[str] = None
    # Populated only on the final chunk, if the server reports usage stats.
    usage: Optional[Dict[str, int]] = None


@dataclass
class ChatResponse:
    """A complete (non-streamed) chat response."""

    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw: Optional[Dict[str, Any]] = None


class LLMClient:
    """Thin client around an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self._session = requests.Session()

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @property
    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return f"{base}/v1/chat/completions"

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": stream,
        }

    # ------------------------------------------------------------------
    # Non-streaming chat
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Send a chat request and return the full response at once."""
        payload = self._build_payload(messages, stream=False, temperature=temperature, max_tokens=max_tokens)

        try:
            resp = self._session.post(
                self._endpoint,
                headers=self._headers,
                json=payload,
                timeout=self.config.request_timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to LLM server at {self.config.base_url}. "
                "Is llama.cpp server running? "
                f"(underlying error: {exc})"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(f"Request to LLM server timed out after {self.config.request_timeout}s") from exc

        if resp.status_code != 200:
            raise LLMResponseError(f"LLM server returned HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM server returned invalid JSON: {exc}") from exc

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"Unexpected response structure from LLM server: {exc}; raw={data}") from exc

        usage = data.get("usage")
        return ChatResponse(content=content, finish_reason=finish_reason, usage=usage, raw=data)

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[ChatChunk, None, None]:
        """Stream a chat completion, yielding ChatChunk objects as tokens arrive.

        The llama.cpp / OpenAI streaming format sends Server-Sent-Events
        lines like:

            data: {"choices": [{"delta": {"content": "Hello"}, ...}]}
            data: {"choices": [{"delta": {"content": " world"}, ...}]}
            data: [DONE]

        This generator parses each line, extracts the incremental content,
        and yields it. On the final chunk (finish_reason present, or
        "[DONE]" received) it yields a ChatChunk with finished=True.
        """
        payload = self._build_payload(messages, stream=True, temperature=temperature, max_tokens=max_tokens)

        try:
            resp = self._session.post(
                self._endpoint,
                headers=self._headers,
                json=payload,
                timeout=self.config.request_timeout,
                stream=True,
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to LLM server at {self.config.base_url}. "
                "Is llama.cpp server running? "
                f"(underlying error: {exc})"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(f"Request to LLM server timed out after {self.config.request_timeout}s") from exc

        if resp.status_code != 200:
            # Try to read the body for a useful error message even though
            # we requested streaming -- error responses are usually small
            # and non-streamed anyway.
            body = resp.text[:500]
            raise LLMResponseError(f"LLM server returned HTTP {resp.status_code}: {body}")

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()

                if data_str == "[DONE]":
                    yield ChatChunk(delta="", finished=True, finish_reason="stop")
                    return

                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    # Skip malformed SSE lines rather than aborting the
                    # whole stream -- some servers emit keep-alive comments.
                    continue

                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta_obj = choice.get("delta", {})
                delta_text = delta_obj.get("content") or ""
                finish_reason = choice.get("finish_reason")

                usage = obj.get("usage")
                is_finished = finish_reason is not None
                yield ChatChunk(delta=delta_text, finished=is_finished, finish_reason=finish_reason, usage=usage)

                if is_finished:
                    return
        except requests.exceptions.ChunkedEncodingError as exc:
            raise LLMResponseError(f"Stream interrupted: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(f"Streaming request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the server appears reachable, False otherwise.

        Tries a few common llama.cpp server health endpoints before
        falling back to a lightweight chat request.

        Timeout is intentionally short (2s): detecting "nothing is
        listening on this port" is normally near-instant at the TCP
        level, and this method can be called periodically in the
        background (see ui/tui.py's _recheck_connection_worker) -- a long
        timeout here mostly matters for a server that's genuinely slow to
        respond while up, which is rare for a /health endpoint.
        """
        base = self.config.base_url.rstrip("/")
        for suffix in ("/health", "/v1/models"):
            try:
                resp = self._session.get(base + suffix, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                continue
        return False

    # ------------------------------------------------------------------
    # Model auto-detection
    # ------------------------------------------------------------------
    def detect_model(self) -> Optional[str]:
        """Query the server's /v1/models endpoint to discover the model it
        actually has loaded, instead of assuming a hardcoded name.

        llama.cpp, Ollama, LM Studio, vLLM, and text-generation-webui all
        expose this OpenAI-compatible endpoint and return at least one
        entry under "data" with an "id" field naming the loaded model.
        Returns None (rather than raising) if the endpoint is unreachable
        or returns something unexpected -- callers should treat that as
        "could not auto-detect" and fall back to whatever was already
        configured, not as a hard failure.
        """
        base = self.config.base_url.rstrip("/")
        try:
            resp = self._session.get(f"{base}/v1/models", headers=self._headers, timeout=5.0)
        except requests.exceptions.RequestException:
            return None

        if resp.status_code != 200:
            return None

        try:
            data = resp.json()
            entries = data.get("data") or []
            if not entries:
                return None
            model_id = entries[0].get("id")
            return str(model_id) if model_id else None
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError, TypeError):
            return None
