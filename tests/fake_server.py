"""A minimal stand-in for a vLLM OpenAI-compatible server, for tests.

Implements the three routes the engine uses -- ``GET /v1/models``,
``POST /v1/chat/completions`` and vLLM's ``POST /v1/chat/completions/batch`` --
with the same quirks the real server has (aggregate batch usage, ``content:
null`` when the budget is exhausted, whole-batch failure on one bad
conversation) plus deliberate fault injection so retries, bisection and
endpoint recovery can be exercised over real HTTP.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeServerState:
    """Knobs the tests flip to shape server behaviour."""

    def __init__(self) -> None:
        self.model_name = "test/model"
        self.api_key: str | None = None
        self.batch_enabled = True
        #: Number of upcoming requests that should fail with this status.
        self.fail_next: list[int] = []
        #: Conversations whose first user message contains this marker make the
        #: whole batch fail with 400 (a "poison" sample).
        self.poison_marker: str | None = None
        #: Return ``content: null`` for conversations containing this marker.
        self.empty_marker: str | None = None
        #: Reply function: (conversation, max_tokens) -> content string.
        self.responder: Callable[[list[dict[str, Any]], int], str] = (
            lambda conv, max_tokens: f"echo:{conv[-1]['content'][:80]}"
        )
        self.batch_calls: list[int] = []       # size of every batch call served
        self.single_calls = 0
        self.requests = 0
        self.lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    state: FakeServerState

    def log_message(self, *args: Any) -> None:  # silence stderr noise
        return

    # -- helpers ------------------------------------------------------- #

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        if not self.state.api_key:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.state.api_key}"

    def _consume_failure(self) -> int | None:
        with self.state.lock:
            self.state.requests += 1
            if self.state.fail_next:
                return self.state.fail_next.pop(0)
        return None

    # -- routes -------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if not self._auth_ok():
            self._json(401, {"error": "bad key"})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._json(
                200,
                {"object": "list", "data": [{"id": self.state.model_name, "object": "model"}]},
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        if not self._auth_ok():
            self._json(401, {"error": "bad key"})
            return

        forced = self._consume_failure()
        if forced is not None:
            self._json(forced, {"error": {"message": f"injected failure {forced}"}})
            return

        path = self.path.rstrip("/")
        if path == "/v1/chat/completions/batch":
            self._batch(payload)
        elif path == "/v1/chat/completions":
            self._single(payload)
        else:
            self._json(404, {"error": "not found"})

    def _content_for(self, conversation: list[dict[str, Any]], max_tokens: int) -> str | None:
        text = " ".join(str(message.get("content", "")) for message in conversation)
        if self.state.empty_marker and self.state.empty_marker in text:
            return None
        return self.state.responder(conversation, max_tokens)

    def _batch(self, payload: dict[str, Any]) -> None:
        if not self.state.batch_enabled:
            self._json(404, {"error": {"message": "route not found"}})
            return
        conversations = payload.get("messages") or []
        if not isinstance(conversations, list) or not conversations:
            self._json(400, {"error": {"message": "messages must be a non-empty list"}})
            return
        if not isinstance(conversations[0], list):
            self._json(400, {"error": {"message": "batch expects a list of conversations"}})
            return
        max_tokens = int(payload.get("max_tokens") or 16)

        # Real vLLM fails the WHOLE call if any conversation is invalid.
        if self.state.poison_marker:
            for conversation in conversations:
                text = " ".join(str(m.get("content", "")) for m in conversation)
                if self.state.poison_marker in text:
                    self._json(
                        400,
                        {
                            "error": {
                                "message": (
                                    "This model's maximum context length is 4096 tokens, "
                                    "however you requested more. Please reduce the length."
                                )
                            }
                        },
                    )
                    return

        with self.state.lock:
            self.state.batch_calls.append(len(conversations))

        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for index, conversation in enumerate(conversations):
            content = self._content_for(conversation, max_tokens)
            prompt_tokens += sum(len(str(m.get("content", "")).split()) for m in conversation)
            completion_tokens += len((content or "").split())
            choices.append(
                {
                    "index": index,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning": None if content else "thought too long",
                    },
                    "finish_reason": "stop" if content else "length",
                }
            )
        self._json(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": payload.get("model") or self.state.model_name,
                "choices": choices,
                # Aggregate over the whole call, exactly like vLLM.
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def _single(self, payload: dict[str, Any]) -> None:
        conversation = payload.get("messages") or []
        if conversation and isinstance(conversation[0], list):
            self._json(400, {"error": {"message": "single endpoint got a batch payload"}})
            return
        with self.state.lock:
            self.state.single_calls += 1
        content = self._content_for(conversation, int(payload.get("max_tokens") or 16))
        self._json(
            200,
            {
                "id": "chatcmpl-fake-single",
                "object": "chat.completion",
                "model": payload.get("model") or self.state.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop" if content else "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


class FakeServer:
    """Context manager running :class:`_Handler` on an ephemeral port."""

    def __init__(self) -> None:
        self.state = FakeServerState()
        handler = type("BoundHandler", (_Handler,), {"state": self.state})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
