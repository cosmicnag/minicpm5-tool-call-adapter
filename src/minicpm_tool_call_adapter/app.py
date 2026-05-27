from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .parser import MiniCPM5Parser, build_openai_tool_calls

DEFAULT_LOG_PATH = Path("minicpm5_adapter.log")


@dataclass
class ProxyConfig:
    listen_host: str = "127.0.0.1"
    listen_port: int = 8000
    upstream_url: str | None = None
    llama_bin: str = "/usr/bin/llama-server"
    model_path: str | None = os.environ.get("MINICPM5_MODEL_PATH")
    template_path: str | None = os.environ.get("MINICPM5_TEMPLATE_PATH")
    served_model_name: str = "minicpm5-1b"
    log_path: str = str(DEFAULT_LOG_PATH)
    upstream_timeout_s: float = 240.0
    request_timeout_s: float = 600.0
    cache_k: str = "q8_0"
    cache_v: str = "q8_0"
    context_length: int = 131072
    extra_args: list[str] = field(default_factory=list)
    chat_template_kwargs: str | None = None
    set_thinking: bool = False


class UpstreamManager:
    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.upstream_url = config.upstream_url
        self.owns_process = False

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _build_command(self, port: int) -> list[str]:
        cmd = [
            self.config.llama_bin,
            "--port",
            str(port),
            "-m",
            self.config.model_path,
            "--alias",
            self.config.served_model_name,
            "--jinja",
            "--chat-template-file",
            self.config.template_path,
            "--cache-type-k",
            self.config.cache_k,
            "--cache-type-v",
            self.config.cache_v,
            "--parallel",
            "1",
            "-fa",
            "on",
            "-c",
            str(self.config.context_length),
        ]
        if self.config.chat_template_kwargs:
            cmd += ["--chat-template-kwargs", self.config.chat_template_kwargs]
        if self.config.set_thinking:
            cmd += ["--reasoning", "on"]
        cmd.extend(self.config.extra_args)
        return cmd

    async def start(self) -> str:
        if self.upstream_url:
            await self._wait_until_ready(self.upstream_url)
            return self.upstream_url

        if not self.config.model_path:
            raise ValueError("MINICPM5_MODEL_PATH or --model-path is required unless --upstream-url is set")
        if not self.config.template_path:
            raise ValueError(
                "MINICPM5_TEMPLATE_PATH or --template-path is required unless --upstream-url is set"
            )
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"MiniCPM5 model not found: {self.config.model_path}")
        if not Path(self.config.template_path).exists():
            raise FileNotFoundError(
                f"MiniCPM5 chat template not found: {self.config.template_path}"
            )

        port = self._free_port()
        self.upstream_url = f"http://127.0.0.1:{port}"
        self.owns_process = True
        log_path = Path(self.config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(log_path, "a", encoding="utf-8", buffering=1)

        env = os.environ.copy()
        self.process = subprocess.Popen(
            self._build_command(port),
            stdout=self.log_handle,
            stderr=self.log_handle,
            env=env,
            text=True,
        )

        try:
            await self._wait_until_ready(self.upstream_url)
        except Exception:
            self.stop()
            raise

        return self.upstream_url

    async def _wait_until_ready(self, base_url: str) -> None:
        timeout_at = time.monotonic() + self.config.upstream_timeout_s
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < timeout_at:
                if self.process and self.process.poll() is not None:
                    raise RuntimeError(f"Upstream llama-server exited with code {self.process.returncode}")
                try:
                    resp = await client.get(f"{base_url}/v1/models")
                    if resp.status_code < 500:
                        return
                except Exception as exc:  # pragma: no cover - startup race
                    last_error = exc
                await asyncio.sleep(0.5)
        raise TimeoutError(f"Timed out waiting for upstream at {base_url}") from last_error

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self.log_handle:
            try:
                self.log_handle.close()
            finally:
                self.log_handle = None


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    blocked = {
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    return {k: v for k, v in headers.items() if k.lower() not in blocked}


def _json_or_text(payload: bytes) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return payload.decode("utf-8", errors="replace")


def _tool_calls_from_choice(
    completion_id: str, choice_index: int, message: dict[str, Any], tools: Any, parser: MiniCPM5Parser
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    content = message.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)

    parsed = parser.detect_and_parse(content, tools)
    existing_tool_calls = message.get("tool_calls") or []
    if not isinstance(existing_tool_calls, list):
        existing_tool_calls = []

    converted = build_openai_tool_calls(
        completion_id,
        choice_index,
        parsed.tool_calls,
        start_index=len(existing_tool_calls),
    )

    rewritten = dict(message)
    rewritten["role"] = rewritten.get("role", "assistant")
    rewritten["content"] = parsed.normal_text or None
    if existing_tool_calls or converted:
        rewritten["tool_calls"] = existing_tool_calls + converted
        rewritten.pop("function_call", None)
    if converted:
        return rewritten, converted
    return rewritten, []


def rewrite_completion_response(data: dict[str, Any], tools: Any, parser: MiniCPM5Parser) -> dict[str, Any]:
    choices = data.get("choices") or []
    completion_id = str(data.get("id") or f"chatcmpl-{int(time.time())}")
    for i, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue
        rewritten, tool_calls = _tool_calls_from_choice(completion_id, i, message, tools, parser)
        choice["message"] = rewritten
        if tool_calls:
            choice["finish_reason"] = "tool_calls"
    return data


def _chunk(
    *,
    completion_id: str,
    model: str,
    choice_index: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": choice_index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


async def _sse_stream_from_completion(data: dict[str, Any]) -> Any:
    completion_id = str(data.get("id") or f"chatcmpl-{int(time.time())}")
    model = str(data.get("model") or "minicpm5-1b")
    choices = data.get("choices") or []
    last_choice_index = len(choices) - 1
    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue

        role_delta = {"role": message.get("role", "assistant")}
        yield f"data: {json.dumps(_chunk(completion_id=completion_id, model=model, choice_index=idx, delta=role_delta), ensure_ascii=False)}\n\n"

        content = message.get("content")
        if isinstance(content, str) and content:
            content_delta = {"content": content}
            yield f"data: {json.dumps(_chunk(completion_id=completion_id, model=model, choice_index=idx, delta=content_delta), ensure_ascii=False)}\n\n"

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            tool_delta = {"tool_calls": tool_calls}
            yield f"data: {json.dumps(_chunk(completion_id=completion_id, model=model, choice_index=idx, delta=tool_delta), ensure_ascii=False)}\n\n"

        finish_reason = choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
        usage = data.get("usage") if idx == last_choice_index else None
        yield f"data: {json.dumps(_chunk(completion_id=completion_id, model=model, choice_index=idx, delta={}, finish_reason=finish_reason, usage=usage), ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def create_app(config: ProxyConfig) -> FastAPI:
    parser = MiniCPM5Parser()
    manager = UpstreamManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        upstream_url = await manager.start()
        app.state.upstream_url = upstream_url
        app.state.client = httpx.AsyncClient(base_url=upstream_url, timeout=config.request_timeout_s)
        app.state.parser = parser
        try:
            yield
        finally:
            client: httpx.AsyncClient | None = getattr(app.state, "client", None)
            if client is not None:
                await client.aclose()
            manager.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "upstream": getattr(app.state, "upstream_url", None)}

    @app.get("/v1/models")
    async def models() -> Response:
        client: httpx.AsyncClient = app.state.client
        resp = await client.get("/v1/models")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers=_response_headers(resp.headers),
        )

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def passthrough(path: str, request: Request) -> Response:
        if path == "chat/completions" and request.method.upper() == "POST":
            return await chat_completions(request)

        client: httpx.AsyncClient = app.state.client
        upstream_method = request.method.upper()
        body = await request.body()
        resp = await client.request(
            upstream_method,
            f"/v1/{path}",
            content=body,
            params=dict(request.query_params),
            headers={k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}},
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers=_response_headers(resp.headers),
        )

    async def chat_completions(request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        payload = await request.json()
        stream = bool(payload.get("stream"))
        upstream_payload = dict(payload)
        upstream_payload["stream"] = False

        resp = await client.post(
            "/v1/chat/completions",
            json=upstream_payload,
            headers={k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}},
        )
        if resp.status_code >= 400:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
                headers=_response_headers(resp.headers),
            )

        data = resp.json()
        rewritten = rewrite_completion_response(data, payload.get("tools"), app.state.parser)
        if stream:
            return StreamingResponse(
                _sse_stream_from_completion(rewritten),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return JSONResponse(content=rewritten, status_code=resp.status_code)

    return app


def build_config_from_args(args: argparse.Namespace) -> ProxyConfig:
    extra_args = shlex.split(args.llama_extra_args or "") if args.llama_extra_args else []
    chat_template_kwargs = args.chat_template_kwargs or os.environ.get("MINICPM5_CHAT_TEMPLATE_KWARGS")
    return ProxyConfig(
        listen_host=args.host,
        listen_port=args.port,
        upstream_url=args.upstream_url,
        llama_bin=args.llama_bin,
        model_path=args.model_path,
        template_path=args.template_path,
        served_model_name=args.served_model_name,
        log_path=args.log_path,
        upstream_timeout_s=args.upstream_timeout_s,
        request_timeout_s=args.request_timeout_s,
        cache_k=args.cache_k,
        cache_v=args.cache_v,
        context_length=args.context_length,
        extra_args=extra_args,
        chat_template_kwargs=chat_template_kwargs,
        set_thinking=args.reasoning,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniCPM5 XML tool-call proxy")
    parser.add_argument("--host", default=os.environ.get("MINICPM5_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--upstream-url",
        default=os.environ.get("MINICPM5_UPSTREAM_URL"),
        help="Use an already-running llama-server instead of spawning one",
    )
    parser.add_argument("--llama-bin", default=os.environ.get("MINICPM5_LLAMABIN", "/usr/bin/llama-server"))
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MINICPM5_MODEL_PATH"),
        help="Path to the MiniCPM5 GGUF (required unless --upstream-url is set)",
    )
    parser.add_argument(
        "--template-path",
        default=os.environ.get("MINICPM5_TEMPLATE_PATH"),
        help="Path to the MiniCPM5 Jinja chat template (required unless --upstream-url is set)",
    )
    parser.add_argument("--served-model-name", default=os.environ.get("MINICPM5_MODEL_NAME", "minicpm5-1b"))
    parser.add_argument("--log-path", default=os.environ.get("MINICPM5_LOG_PATH", str(DEFAULT_LOG_PATH)))
    parser.add_argument(
        "--upstream-timeout-s",
        type=float,
        default=float(os.environ.get("MINICPM5_UPSTREAM_TIMEOUT_S", "240")),
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.environ.get("MINICPM5_REQUEST_TIMEOUT_S", "600")),
    )
    parser.add_argument("--cache-k", default=os.environ.get("MINICPM5_CACHE_K", "q8_0"))
    parser.add_argument("--cache-v", default=os.environ.get("MINICPM5_CACHE_V", "q8_0"))
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(os.environ.get("MINICPM5_CONTEXT_LENGTH", "131072")),
    )
    parser.add_argument(
        "--llama-extra-args",
        default=os.environ.get("MINICPM5_LLAMA_EXTRA_ARGS", ""),
        help="Extra llama-server args appended verbatim",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default=os.environ.get("MINICPM5_CHAT_TEMPLATE_KWARGS", ""),
        help="JSON string passed to --chat-template-kwargs",
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("MINICPM5_REASONING", "0") in {"1", "true", "yes", "on"},
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)
    app = create_app(config)
    uvicorn.run(app, host=config.listen_host, port=config.listen_port, log_level="info")


if __name__ == "__main__":
    main(sys.argv[1:])
