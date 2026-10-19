"""Streaming client for OpenAI-compatible chat completions.

Backends:
  * **openai** — local vLLM, ALB ingress (`/v1/chat/completions`)
  * **runpod** — serverless vLLM (`/openai/v1/chat/completions`)

RunPod: https://docs.runpod.io/serverless/vllm/openai-compatibility

Do not send a ``model`` field to RunPod unless it matches the worker template;
the worker already has ``MODEL_NAME`` configured. Use ``--attack-model`` only to
pick prompt files under ``attack_data/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Literal, Optional

Backend = Literal["auto", "openai", "runpod"]

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'requests' package is required. Install it with 'uv sync' (or 'uv pip install requests')."
    ) from exc

RUNPOD_API_HOST = "api.runpod.ai"
RUNPOD_OPENAI_V1_SUFFIX = "/openai/v1"
CHAT_PATH = "/chat/completions"
LOCAL_CHAT_PATH = "/v1/chat/completions"

EMPTY_COMPLETION_HINT = (
    "Empty completion (no streamed tokens) with HTTP 200. "
    "On RunPod this usually means the worker job failed or was throttled while the "
    "gateway still returned success — check RunPod job logs, reduce N×λ, or omit "
    "--model so the worker uses its template MODEL_NAME."
)


class EmptyCompletionError(RuntimeError):
    """HTTP succeeded but the response stream contained no tokens."""


def _empty_completion_message(result: "RequestResult", stream_snippet: str = "") -> str:
    parts = [EMPTY_COMPLETION_HINT]
    if result.http_status is not None:
        parts.append(f"http_status={result.http_status}")
    if result.finish_reason:
        parts.append(f"finish_reason={result.finish_reason!r}")
    if stream_snippet:
        parts.append(f"stream_snippet={stream_snippet[:300]!r}")
    return " ".join(parts)


def default_endpoint() -> str:
    import os

    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    if endpoint_id:
        return f"https://{RUNPOD_API_HOST}/v2/{endpoint_id}{RUNPOD_OPENAI_V1_SUFFIX}"
    return "http://localhost:8000"


def default_api_key() -> Optional[str]:
    import os

    for name in ("RUNPOD_API_KEY", "OPENAI_API_KEY", "VLLM_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def api_key_source() -> Optional[str]:
    """Return which env var supplied the default API key (for diagnostics)."""
    import os

    for name in ("RUNPOD_API_KEY", "OPENAI_API_KEY", "VLLM_API_KEY"):
        if os.environ.get(name, "").strip():
            return name
    return None


def resolve_backend(endpoint: str, backend: Backend = "auto") -> Backend:
    if backend != "auto":
        return backend
    return "runpod" if RUNPOD_API_HOST in endpoint else "openai"


def resolve_chat_completions_url(endpoint: str, backend: Backend = "auto") -> str:
    """Return the full URL for POST chat completions."""
    base = endpoint.rstrip("/")
    resolved = resolve_backend(base, backend)

    if resolved == "runpod":
        if not base.endswith(RUNPOD_OPENAI_V1_SUFFIX):
            if "/v2/" in base and "/openai" not in base:
                base += RUNPOD_OPENAI_V1_SUFFIX
            elif base.endswith("/openai"):
                base += "/v1"
        return base + CHAT_PATH

    if base.endswith(RUNPOD_OPENAI_V1_SUFFIX) or base.endswith("/v1"):
        return base + CHAT_PATH
    return base + LOCAL_CHAT_PATH


def resolve_runpod_base(endpoint_id: str) -> str:
    return f"https://{RUNPOD_API_HOST}/v2/{endpoint_id.strip()}{RUNPOD_OPENAI_V1_SUFFIX}"


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _api_error_message(err: Any) -> str:
    if isinstance(err, dict):
        return str(err.get("message", err))
    return str(err)


def _raise_api_error(err: Any) -> None:
    raise RuntimeError(f"API error: {_api_error_message(err)}")


def _extract_delta_text(choice: dict) -> Optional[str]:
    delta = choice.get("delta") or {}
    text = delta.get("content") or delta.get("text")
    if text:
        return text
    message = choice.get("message") or {}
    return message.get("content")


def _parse_sse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return None
    return json.loads(payload)


def _iter_sse_chunks(response: requests.Response) -> Iterator[dict]:
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        chunk = _parse_sse_line(raw)
        if chunk is not None:
            yield chunk


def _backoff_seconds(attempt: int, response: Optional[requests.Response] = None) -> float:
    """Sleep duration before retry; honours ``Retry-After`` when present."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(2.0**attempt, 30.0)


_RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})


def _check_non_stream_json(response: requests.Response) -> None:
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "application/json" not in content_type or "text/event-stream" in content_type:
        return
    body = response.json()
    if body.get("error"):
        _raise_api_error(body["error"])
    raise RuntimeError(f"Unexpected JSON response (expected SSE): {str(body)[:500]}")


@dataclass
class RequestResult:
    """Latency and output statistics for one streamed chat completion."""

    index: int
    is_attack: bool
    pair_index: int
    success: bool
    model: Optional[str] = None
    error: Optional[str] = None
    http_status: Optional[int] = None
    ttft_s: Optional[float] = None
    total_s: Optional[float] = None
    decode_s: Optional[float] = None
    output_tokens: int = 0
    output_chars: int = 0
    usage_completion_tokens: Optional[int] = None
    usage_prompt_tokens: Optional[int] = None
    itl_mean_s: Optional[float] = None
    itl_p50_s: Optional[float] = None
    itl_p95_s: Optional[float] = None
    itl_max_s: Optional[float] = None
    decode_throughput_tok_s: Optional[float] = None
    finish_reason: Optional[str] = None
    prompt_preview: str = ""
    generated_preview: str = ""
    generated_text: str = ""
    inter_token_latencies_s: list[float] = field(default_factory=list)

    def to_dict(self, include_itls: bool = False, include_full_text: bool = False) -> dict:
        data = asdict(self)
        if not include_itls:
            data.pop("inter_token_latencies_s", None)
        if not include_full_text:
            data.pop("generated_text", None)
        return data


@dataclass
class ChatClient:
    """OpenAI-compatible streaming chat client (local vLLM or RunPod serverless)."""

    endpoint: str = field(default_factory=default_endpoint)
    model: Optional[str] = None
    api_key: Optional[str] = None
    backend: Backend = "auto"
    timeout_s: float = 600.0
    max_retries: int = 5  # retries after transient / rate-limit errors
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    def __post_init__(self) -> None:
        self.backend = resolve_backend(self.endpoint, self.backend)
        if self.api_key is None:
            self.api_key = default_api_key()

    @property
    def url(self) -> str:
        return resolve_chat_completions_url(self.endpoint, self.backend)

    @property
    def sends_model_field(self) -> bool:
        # Never advertise a model on RunPod: the worker uses its template
        # MODEL_NAME, and a mismatched model can fail the job while HTTP stays 200.
        return bool(self.model) and self.backend != "runpod"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _openai_v1_base(self) -> str:
        return self.url[: -len(CHAT_PATH)]

    def list_models(self) -> list[str]:
        resp = requests.get(
            f"{self._openai_v1_base()}/models",
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if "id" in m]

    def _build_payload(self, content: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }
        if self.sends_model_field:
            payload["model"] = self.model
        if self.backend != "runpod":
            payload["stream_options"] = {"include_usage": True}
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def _apply_chunk(self, chunk: dict, result: RequestResult, token_times: list[float], text_parts: list[str]) -> None:
        if chunk.get("error"):
            _raise_api_error(chunk["error"])

        usage = chunk.get("usage")
        if usage:
            result.usage_completion_tokens = usage.get("completion_tokens")
            result.usage_prompt_tokens = usage.get("prompt_tokens")

        choices = chunk.get("choices") or []
        if not choices:
            return

        choice = choices[0]
        piece = _extract_delta_text(choice)
        if piece:
            token_times.append(time.perf_counter())
            text_parts.append(piece)

        if choice.get("finish_reason"):
            result.finish_reason = choice["finish_reason"]
        if chunk.get("model"):
            result.model = chunk["model"]

    def _finalize_metrics(
        self,
        result: RequestResult,
        start: float,
        token_times: list[float],
        text_parts: list[str],
        preview_chars: int,
        stream_snippet: str = "",
    ) -> None:
        result.total_s = time.perf_counter() - start
        full_text = "".join(text_parts)

        if not full_text and not token_times:
            result.success = False
            result.error = _empty_completion_message(result, stream_snippet)
            return

        result.success = True
        if token_times:
            result.ttft_s = token_times[0] - start
            itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            result.inter_token_latencies_s = itls
            if itls:
                result.itl_mean_s = sum(itls) / len(itls)
                result.itl_p50_s = _percentile(itls, 50)
                result.itl_p95_s = _percentile(itls, 95)
                result.itl_max_s = max(itls)
            result.decode_s = token_times[-1] - token_times[0]
            if result.decode_s and len(token_times) > 1:
                result.decode_throughput_tok_s = (len(token_times) - 1) / result.decode_s

        if result.model is None and self.sends_model_field:
            result.model = self.model
        result.output_tokens = result.usage_completion_tokens or len(token_times)
        result.output_chars = len(full_text)
        result.generated_preview = full_text[:preview_chars]
        result.generated_text = full_text

    def stream_chat(
        self,
        content: str,
        index: int = 0,
        is_attack: bool = False,
        pair_index: int = 0,
        preview_chars: int = 200,
    ) -> RequestResult:
        result = RequestResult(
            index=index,
            is_attack=is_attack,
            pair_index=pair_index,
            success=False,
            model=self.model,
            prompt_preview=content[:preview_chars],
        )
        token_times: list[float] = []
        text_parts: list[str] = []
        start = time.perf_counter()
        attempts = max(1, int(self.max_retries) + 1)
        last_exc: Optional[Exception] = None

        for attempt in range(attempts):
            token_times = []
            text_parts = []
            stream_snippet = ""
            try:
                with requests.post(
                    self.url,
                    json=self._build_payload(content),
                    headers=self._headers(),
                    stream=True,
                    timeout=self.timeout_s,
                ) as resp:
                    result.http_status = resp.status_code
                    resp.raise_for_status()
                    _check_non_stream_json(resp)
                    for chunk in _iter_sse_chunks(resp):
                        if not stream_snippet:
                            stream_snippet = json.dumps(chunk, ensure_ascii=False)[:300]
                        self._apply_chunk(chunk, result, token_times, text_parts)
                self._finalize_metrics(
                    result, start, token_times, text_parts, preview_chars, stream_snippet
                )
                if result.success:
                    return result
                if not token_times and not text_parts and attempt + 1 < attempts:
                    last_exc = EmptyCompletionError(result.error or "empty stream")
                    time.sleep(_backoff_seconds(attempt))
                    continue
                return result
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
                resp = exc.response
                if resp is not None:
                    result.http_status = resp.status_code
                    if (
                        resp.status_code in _RETRYABLE_HTTP_STATUS
                        and attempt + 1 < attempts
                    ):
                        time.sleep(_backoff_seconds(attempt, resp))
                        continue
                break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(_backoff_seconds(attempt))
                    continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break

        result.success = False
        result.error = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown error"
        result.total_s = time.perf_counter() - start
        resp_obj = getattr(last_exc, "response", None) if last_exc else None
        if resp_obj is not None and result.http_status is None:
            result.http_status = getattr(resp_obj, "status_code", None)
        return result
