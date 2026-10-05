"""The Kryptos model gateway: redact the prompt on its way to the model.

A local HTTP proxy that Claude Code (or anything speaking the Anthropic
Messages API) points at with one environment variable:

    export ANTHROPIC_BASE_URL=http://127.0.0.1:8787

Requests arrive here, the latest user message is redacted, and the rewritten
request goes upstream. The model answers the redacted version, so you get a
normal reply instead of a refusal -- which is the thing a ``UserPromptSubmit``
hook cannot do, because a hook is consulted about a request while a proxy owns
one.

    Claude Code  --->  this gateway  --->  api.anthropic.com
                         |
                         +-- kryptos_pii_local.redact()   (this machine)

What it redacts
---------------
**Only the latest user message, and only its text blocks.** Everything else in
the request is forwarded byte for byte: the system prompt, tool definitions,
assistant turns, earlier user turns, and tool results -- including file
contents Claude read.

That boundary is deliberate and it is narrower than "no PII reaches the model".
A Claude Code request carries the whole conversation, every tool definition and
the contents of files it has read. Running all of that through a PII detector
would redact identifiers out of source code, names out of diffs and addresses
out of test fixtures, and the agent would stop working. So v1 protects what the
user typed, and claims nothing more. See the SECURITY BOUNDARY section in
README.md.

Fail closed
-----------
Any failure -- unparseable body, detector error, a redaction that does not
verify -- returns an error to the client and forwards nothing. The original
request is never sent "just this once" after an error, because that is exactly
the request someone was trying to protect.

Local only
----------
Imports ``kryptos_pii_local``. There is no import of ``kryptos_pii_client`` and
no code path to a Kryptos service; the only outbound connection this process
makes is to the configured Anthropic upstream.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.environ.get("KRYPTOS_GATEWAY_UPSTREAM", "https://api.anthropic.com").rstrip("/")

# Long, because a model turn is long. The read timeout has to outlast the
# slowest completion or streaming replies get cut off mid-answer.
TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=float(os.environ.get("KRYPTOS_GATEWAY_READ_TIMEOUT", "600")),
    write=60.0,
    pool=10.0,
)

# Hop-by-hop headers and anything describing a body we are about to change.
# Everything else -- x-api-key, authorization, anthropic-version,
# anthropic-beta, the client's own user-agent -- is forwarded untouched, because
# authentication and API versioning are not ours to reinterpret.
STRIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
}

STRIP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
}


class RedactionFailed(Exception):
    """The request could not be made safe, so it must not be sent."""


def _sdk() -> Any:
    try:
        import kryptos_pii_local
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "kryptos-gateway needs the local detector.\n"
            f"  interpreter: {sys.executable}\n"
            "  fix: pip install kryptos-pii-local && kryptos-pii-model download"
        ) from exc
    return kryptos_pii_local


sdk = _sdk()


def redact_text(text: str) -> tuple[str, list[str]]:
    """Redact one string. Returns the safe text and the categories found.

    Raises :class:`RedactionFailed` rather than returning the input when
    anything goes wrong. A redactor that returns its input on error is a
    redactor that forwards PII on error.
    """
    try:
        result = sdk.redact(text)
    except Exception as exc:  # noqa: BLE001 - every failure is the same decision
        raise RedactionFailed(f"{type(exc).__name__}: {exc}") from exc

    if not result.findings:
        return text, []

    safe = result.text
    if safe is None:
        raise RedactionFailed("the detector reported findings but produced no redacted text")

    return safe, sorted({f.type for f in result.findings})


def redact_latest_user_message(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Rewrite the newest user turn in an Anthropic Messages request.

    Only that turn, and within it only ``text`` blocks: a ``tool_result`` block
    carries output the agent needs intact, and rewriting it would corrupt the
    agent's view of its own tools.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise RedactionFailed("request has no 'messages' array")

    index = None
    for i in range(len(messages) - 1, -1, -1):
        entry = messages[i]
        if isinstance(entry, dict) and entry.get("role") == "user":
            index = i
            break
    if index is None:
        return payload, []

    message = messages[index]
    content = message.get("content")
    found: list[str] = []

    if isinstance(content, str):
        safe, types = redact_text(content)
        found += types
        new_content: Any = safe

    elif isinstance(content, list):
        new_content = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                # tool_result, image, document: forwarded untouched by design.
                new_content.append(block)
                continue
            text = block.get("text")
            if not isinstance(text, str):
                new_content.append(block)
                continue
            safe, types = redact_text(text)
            found += types
            new_content.append({**block, "text": safe})

    else:
        raise RedactionFailed(
            f"latest user message has content of type {type(content).__name__}"
        )

    if not found:
        return payload, []

    # Rebuild rather than mutate: the caller's payload is the thing we would
    # fall back to, and a half-redacted object is worse than none.
    rebuilt = dict(payload)
    rebuilt["messages"] = [
        {**m, "content": new_content} if i == index else m for i, m in enumerate(messages)
    ]
    return rebuilt, sorted(set(found))


def _log(message: str) -> None:
    """Operational logging only.

    Nothing here ever receives prompt text or a matched value. Counts and
    category names are the most specific thing that may be printed.
    """
    print(f"kryptos-gateway: {message}", file=sys.stderr, flush=True)


app = FastAPI(title="Kryptos model gateway", docs_url=None, redoc_url=None)


@app.get("/kryptos/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "upstream": UPSTREAM,
        "model_installed": sdk.model_installed(),
        "redacts": "latest user message, text blocks only",
    }


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy(path: str, request: Request) -> Response:
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in STRIP_REQUEST_HEADERS
    }

    # Only the Messages API carries prompts. Everything else -- token counting,
    # model listing -- is forwarded as-is.
    if request.method == "POST" and path.endswith("v1/messages") and body:
        try:
            import json

            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise RedactionFailed("request body is not a JSON object")
            payload, found = redact_latest_user_message(payload)
        except RedactionFailed as exc:
            _log(f"BLOCKED: {exc}")
            return _refuse(str(exc))
        except ValueError as exc:
            _log(f"BLOCKED: body is not valid JSON ({type(exc).__name__})")
            return _refuse("request body is not valid JSON")

        if found:
            body = json.dumps(payload).encode()
            _log(f"redacted latest user message: {', '.join(found)}")

    url = f"{UPSTREAM}/{path}"
    client = httpx.AsyncClient(timeout=TIMEOUT)
    upstream_request = client.build_request(
        request.method, url, headers=headers, content=body, params=request.query_params
    )

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        _log(f"upstream error: {type(exc).__name__}")
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"kryptos-gateway: upstream {type(exc).__name__}"},
            },
        )

    async def stream() -> Any:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        headers={
            k: v for k, v in upstream.headers.items() if k.lower() not in STRIP_RESPONSE_HEADERS
        },
        media_type=upstream.headers.get("content-type"),
    )


def _refuse(detail: str) -> JSONResponse:
    """Fail closed, in the shape the client already knows how to display."""
    return JSONResponse(
        status_code=400,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "kryptos-gateway refused to forward this request because it "
                    f"could not be redacted: {detail}. "
                    "The request was not sent upstream."
                ),
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    """``kryptos-gateway``: run the proxy."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kryptos-gateway",
        description="Redact the latest user message on its way to the Anthropic API.",
    )
    parser.add_argument("--host", default=os.environ.get("KRYPTOS_GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("KRYPTOS_GATEWAY_PORT", "8787"))
    )
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args(argv)

    import uvicorn

    _log(f"upstream {UPSTREAM}")
    _log(f"point Claude Code at it:  export ANTHROPIC_BASE_URL=http://{args.host}:{args.port}")
    if not sdk.model_installed():
        _log("WARNING: no checkpoint installed; every request will fail closed.")
        _log("  fix: kryptos-pii-model download")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
