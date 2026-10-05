"""The model gateway, tested against a mock Anthropic server.

Every assertion about safety is made on what the *mock server received*, not on
what the gateway returned or believes it did. The gateway's claim is about what
crosses the wire, so the witness has to be on the far side of the wire.

    pytest client  --->  gateway  --->  mock Anthropic
                                          records the body
                                          tests read it back

Both servers run as real uvicorn processes on real ports, because the parts
most likely to break -- header forwarding, SSE passthrough, body rewriting --
do not exist when you call the ASGI app in-process.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROXY_DIR = REPO / "integrations" / "proxy"

EMAIL = "john@example.com"
PHONE = "+1-202-555-0198"
SSN = "529-41-7836"
PASSWORD = "hunter2xyzQ!"


def _interpreter() -> str | None:
    candidates = [
        os.environ.get("KRYPTOS_PII_PYTHON"),
        str(Path.home() / "Desktop" / "kryptos" / "kryptos-test" / ".venv" / "bin" / "python"),
        str(Path.home() / ".kryptos-pii" / "bin" / "python"),
        shutil.which("python3"),
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", "import kryptos_pii_local, fastapi, uvicorn, httpx"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return candidate
    return None


PYTHON = _interpreter()
needs_stack = pytest.mark.skipif(
    PYTHON is None, reason="no interpreter with kryptos_pii_local + fastapi + uvicorn + httpx"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(port: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"nothing listening on {port} after {timeout}s")


class Stack:
    """A mock Anthropic server with the gateway pointed at it."""

    def __init__(self, upstream_env: str | None = None) -> None:
        assert PYTHON
        self.mock_port = free_port()
        self.gateway_port = free_port()

        self.mock = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "mock_anthropic:app",
             "--host", "127.0.0.1", "--port", str(self.mock_port), "--log-level", "error"],
            cwd=str(PROXY_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        wait_for(self.mock_port)

        upstream = upstream_env or f"http://127.0.0.1:{self.mock_port}"
        self.gateway = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "gateway:app",
             "--host", "127.0.0.1", "--port", str(self.gateway_port), "--log-level", "error"],
            cwd=str(PROXY_DIR),
            env={**os.environ, "KRYPTOS_GATEWAY_UPSTREAM": upstream},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        wait_for(self.gateway_port)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    def send(self, payload: dict, **kwargs) -> "httpx.Response":  # type: ignore[name-defined]
        import httpx

        return httpx.post(
            f"{self.base}/v1/messages",
            json=payload,
            headers={
                "x-api-key": "sk-ant-test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120,
            **kwargs,
        )

    def received(self) -> list[dict]:
        import httpx

        return httpx.get(f"http://127.0.0.1:{self.mock_port}/__received", timeout=30).json()

    def last_outbound(self) -> str:
        """The exact bytes the mock Anthropic server was sent."""
        rows = self.received()
        assert rows, "the mock server received nothing"
        return rows[-1]["raw"]

    def close(self) -> None:
        for proc in (self.gateway, self.mock):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()


@pytest.fixture(scope="module")
def stack():
    if PYTHON is None:
        pytest.skip("no suitable interpreter")
    s = Stack()
    try:
        yield s
    finally:
        s.close()


def ask(text: str) -> dict:
    return {
        "model": "claude-opus-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": text}],
    }


# --- the end-to-end guarantee --------------------------------------------


@needs_stack
def test_end_to_end_raw_pii_never_reaches_the_upstream(stack: Stack) -> None:
    """The headline test (requirement 11).

    Raw PII in -> redacted placeholder on the wire -> normal answer back.
    """
    response = stack.send(ask(f"Email is {EMAIL}. Summarize this."))

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "mock reply"

    outbound = stack.last_outbound()
    assert EMAIL not in outbound, "raw email reached the upstream"
    assert "[EMAIL]" in outbound
    assert "Summarize this." in outbound, "non-PII content was not preserved"


# --- one test per PII type (requirement 7) --------------------------------


@needs_stack
@pytest.mark.parametrize(
    "raw, placeholder",
    [
        (EMAIL, "[EMAIL]"),
        (PHONE, "[PHONE]"),
        (SSN, "[SSN]"),
        (PASSWORD, "[PASSWORD]"),
    ],
    ids=["email", "phone", "ssn", "password"],
)
def test_each_pii_type_is_replaced(stack: Stack, raw: str, placeholder: str) -> None:
    label = {"[PASSWORD]": "password"}.get(placeholder, "value")
    stack.send(ask(f"The {label} is {raw}. Summarize."))
    outbound = stack.last_outbound()
    assert raw not in outbound
    assert placeholder in outbound


@needs_stack
def test_multiple_pii_types_in_one_prompt(stack: Stack) -> None:
    stack.send(ask(f"Email {EMAIL}, phone {PHONE}, ssn {SSN}. Summarize."))
    outbound = stack.last_outbound()
    for raw in (EMAIL, PHONE, SSN):
        assert raw not in outbound
    for placeholder in ("[EMAIL]", "[PHONE]", "[SSN]"):
        assert placeholder in outbound


@needs_stack
def test_a_clean_prompt_is_forwarded_unchanged(stack: Stack) -> None:
    prompt = "Explain what PostgreSQL indexes are."
    response = stack.send(ask(prompt))
    assert response.status_code == 200
    assert prompt in stack.last_outbound()


@needs_stack
def test_a_multiline_prompt_keeps_its_shape(stack: Stack) -> None:
    prompt = f"Customer notes:\n\n- email: {EMAIL}\n- phone: {PHONE}\n\nSummarize the notes."
    stack.send(ask(prompt))
    outbound = json.loads(stack.last_outbound())
    text = outbound["messages"][-1]["content"]

    assert EMAIL not in text and PHONE not in text
    assert text.count("\n") == prompt.count("\n"), "line structure was not preserved"
    assert "Summarize the notes." in text


# --- what must NOT be touched (requirement 3) ------------------------------


@needs_stack
def test_system_tools_history_and_tool_results_are_preserved_exactly(stack: Stack) -> None:
    """The scope boundary, asserted rather than documented.

    Everything except the latest user message goes upstream byte for byte --
    including PII in it, which is the documented limitation, not a bug.
    """
    payload = {
        "model": "claude-opus-5",
        "max_tokens": 64,
        "system": f"You help with support. Escalate to supervisor@example.com.",
        "tools": [{"name": "lookup", "description": f"Finds a customer by email like {EMAIL}",
                   "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {"role": "user", "content": f"earlier turn with {PHONE}"},
            {"role": "assistant", "content": f"I saw {SSN} in the record."},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": f"row: {EMAIL}"},
                {"type": "text", "text": f"Now summarize. My own email is {EMAIL}."},
            ]},
        ],
    }
    stack.send(payload)
    outbound = json.loads(stack.last_outbound())

    # Untouched, exactly as sent.
    assert outbound["system"] == payload["system"]
    assert outbound["tools"] == payload["tools"]
    assert outbound["messages"][0] == payload["messages"][0]
    assert outbound["messages"][1] == payload["messages"][1]
    assert outbound["messages"][2]["content"][0] == payload["messages"][2]["content"][0]

    # The one thing that is rewritten: the text block of the latest user turn.
    assert outbound["messages"][2]["content"][1]["text"] == "Now summarize. My own email is [EMAIL]."


# --- streaming (requirement 2) --------------------------------------------


@needs_stack
def test_streaming_passes_through_and_is_still_redacted(stack: Stack) -> None:
    import httpx

    payload = {**ask(f"Email is {EMAIL}. Summarize."), "stream": True}
    chunks: list[str] = []
    with httpx.stream(
        "POST", f"{stack.base}/v1/messages", json=payload,
        headers={"x-api-key": "k", "anthropic-version": "2023-06-01"}, timeout=120,
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        for line in response.iter_lines():
            chunks.append(line)

    body = "\n".join(chunks)
    assert "message_start" in body and "message_stop" in body
    assert "mock " in body and "stream" in body, "SSE deltas did not arrive intact"
    assert EMAIL not in stack.last_outbound()


# --- headers (requirement 2) ----------------------------------------------


@needs_stack
def test_authentication_and_version_headers_are_forwarded(stack: Stack) -> None:
    stack.send(ask("hello"))
    headers = stack.received()[-1]["headers"]
    assert headers.get("x-api-key") == "sk-ant-test-key"
    assert headers.get("anthropic-version") == "2023-06-01"


# --- fail closed (requirement 5) ------------------------------------------


@needs_stack
def test_a_malformed_body_is_refused_and_not_forwarded(stack: Stack) -> None:
    import httpx

    before = len(stack.received())
    response = httpx.post(
        f"{stack.base}/v1/messages",
        content=b"{not json at all",
        headers={"content-type": "application/json", "x-api-key": "k"},
        timeout=60,
    )
    assert response.status_code == 400
    assert "refused to forward" in response.json()["error"]["message"]
    assert len(stack.received()) == before, "a malformed request was forwarded anyway"


@needs_stack
def test_a_request_without_messages_is_refused(stack: Stack) -> None:
    before = len(stack.received())
    response = stack.send({"model": "claude-opus-5", "max_tokens": 10})
    assert response.status_code == 400
    assert len(stack.received()) == before


@needs_stack
def test_content_of_an_unexpected_type_is_refused(stack: Stack) -> None:
    before = len(stack.received())
    response = stack.send(
        {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": 12345}]}
    )
    assert response.status_code == 400
    assert len(stack.received()) == before


def test_a_detector_failure_blocks_rather_than_forwarding() -> None:
    """Unit-level, because making the real detector fail mid-flight is not
    something to arrange with a live process."""
    if PYTHON is None:
        pytest.skip("no suitable interpreter")
    script = """
import sys, json
sys.path.insert(0, %r)
import gateway

class Boom:
    def redact(self, text):
        raise RuntimeError("detector exploded")
    def model_installed(self):
        return True

gateway.sdk = Boom()
payload = {"messages": [{"role": "user", "content": "Email is john@example.com"}]}
try:
    gateway.redact_latest_user_message(payload)
    print("FORWARDED")
except gateway.RedactionFailed as exc:
    print("BLOCKED")
""" % str(PROXY_DIR)
    out = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True, timeout=120)
    assert "BLOCKED" in out.stdout, out.stderr


def test_a_redaction_that_produces_no_text_blocks() -> None:
    """Findings but no redacted text must not fall back to the original."""
    if PYTHON is None:
        pytest.skip("no suitable interpreter")
    script = """
import sys
sys.path.insert(0, %r)
import gateway

class Finding:
    type = "email"

class Result:
    findings = [Finding()]
    text = None          # the failure being simulated

class Broken:
    def redact(self, text):
        return Result()
    def model_installed(self):
        return True

gateway.sdk = Broken()
try:
    gateway.redact_text("Email is john@example.com")
    print("FORWARDED")
except gateway.RedactionFailed:
    print("BLOCKED")
""" % str(PROXY_DIR)
    out = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True, timeout=120)
    assert "BLOCKED" in out.stdout, out.stderr


# --- health ---------------------------------------------------------------


@needs_stack
def test_health_reports_the_scope_it_actually_covers(stack: Stack) -> None:
    import httpx

    body = httpx.get(f"{stack.base}/kryptos/health", timeout=30).json()
    assert body["status"] == "ok"
    assert body["redacts"] == "latest user message, text blocks only"


@needs_stack
def test_a_compressed_upstream_response_survives_the_proxy(stack: Stack) -> None:
    """Regression: the gateway relays the raw body, so content-encoding has to
    travel with it.

    Stripping that header handed Claude Code gzip bytes it believed were plain
    text, and it died with "JSON Parse error: Unrecognized token". The mock did
    not compress, so only the live API caught it. Now the mock can.
    """
    import httpx

    response = httpx.post(
        f"{stack.base}/v1/messages/gzip",
        json={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "k", "accept-encoding": "gzip"},
        timeout=60,
    )
    assert response.status_code == 200
    # httpx decodes using content-encoding. If the gateway dropped the header
    # this raises or yields mojibake instead of JSON.
    assert response.json()["content"][0]["text"] == "gzipped reply"
