"""A stand-in for api.anthropic.com that records exactly what it was sent.

The gateway's whole claim is about what crosses the wire. Asserting on the
gateway's own view of that would be asserting that it believes itself; this
server is the independent witness, and the tests read what *it* received.

    POST /v1/messages        answers, and records the request body
    GET  /__received         every request body it has seen, newest last
    POST /__reset            forget them

Started by the tests. Not part of the shipped gateway.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(docs_url=None, redoc_url=None)

RECEIVED: list[dict[str, Any]] = []


@app.get("/__received")
def received() -> list[dict[str, Any]]:
    return RECEIVED


@app.post("/__reset")
def reset() -> dict[str, int]:
    count = len(RECEIVED)
    RECEIVED.clear()
    return {"cleared": count}


@app.post("/v1/messages")
async def messages(request: Request) -> Any:
    raw = await request.body()
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = None

    RECEIVED.append(
        {
            "raw": raw.decode("utf-8", "replace"),
            "json": payload,
            "headers": {k.lower(): v for k, v in request.headers.items()},
        }
    )

    if payload and payload.get("stream"):
        return StreamingResponse(_sse(), media_type="text/event-stream")

    return JSONResponse(
        {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": (payload or {}).get("model", "mock"),
            "content": [{"type": "text", "text": "mock reply"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    )


async def _sse() -> Any:
    events = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_mock", "role": "assistant", "content": []}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "mock "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "stream"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    ]
    for name, data in events:
        yield f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
