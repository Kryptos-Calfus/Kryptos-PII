"""The PII Protection SDK.

Two ways to call the same extension, and the choice is a privacy decision rather
than a performance one:

    from kryptos_pii_sdk import Kryptos
    pii = Kryptos(api_key=os.environ["KRYPTOS_API_KEY"])
    pii.redact("Call Priya on 9812345678")       # text goes to Kryptos

    from kryptos_pii_sdk import local
    local.redact("Call Priya on 9812345678")     # text stays on this machine

The hosted client talks to the orchestrator, so the call is authenticated,
scoped, metered and audited like any other extension execution. The local
runtime imports the detector directly and needs no API key at all -- section 8
of CLAUDE.md is explicit that purely local execution must not require one.

Both return the same :class:`Result`, so moving between them is a one-line
change and never a rewrite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__version__ = "0.1.0"

DEFAULT_BASE_URL = os.environ.get("KRYPTOS_BASE_URL", "https://api.kryptos.ai")
OPERATIONS = ("detect", "redact", "mask", "tokenize", "block", "audit")


class PIIError(RuntimeError):
    """The extension could not reach a decision."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class Finding:
    type: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None
    detector: str | None = None


@dataclass
class Result:
    """One decision. ``text`` is the content as it should now be used."""

    decision: str
    risk: str
    findings: list[Finding] = field(default_factory=list)
    transformed_content: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    vault: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str | None:
        """The content to pass on: rewritten when the action rewrote it."""
        return self.transformed_content

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @property
    def found_anything(self) -> bool:
        return bool(self.findings)

    def detokenize(self, text: str) -> str:
        """Put the original values back. Only works for a tokenize result whose
        vault you were given, which hosted calls do not return by default."""
        for token, original in self.vault.items():
            text = text.replace(token, original)
        return text

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Result:
        return cls(
            decision=payload["decision"],
            risk=payload.get("risk", "none"),
            findings=[
                Finding(
                    type=f["type"],
                    start=f.get("start"),
                    end=f.get("end"),
                    confidence=f.get("confidence"),
                    detector=f.get("detector"),
                )
                for f in payload.get("findings", [])
            ],
            transformed_content=payload.get("transformed_content"),
            reason_codes=payload.get("reason_codes", []),
            latency_ms=payload.get("latency_ms"),
            metadata=payload.get("metadata", {}),
            vault=payload.get("vault", {}),
        )


class Kryptos:
    """Hosted execution through the Kryptos control plane.

    The API key authenticates the account; the extension must be installed on it
    and the key must carry the matching ``pii:<operation>`` scope, or the call is
    refused. That refusal is the platform working, not a bug in this client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
    ) -> None:
        key = api_key or os.environ.get("KRYPTOS_API_KEY")
        if not key:
            raise PIIError(
                "No API key. Pass api_key=... or set KRYPTOS_API_KEY. "
                "For key-free execution that keeps text on this machine, use "
                "kryptos_pii_sdk.local instead."
            )
        self._key = key
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _call(self, operation: str, content: str, config: dict[str, Any] | None) -> Result:
        import httpx

        if operation not in OPERATIONS:
            raise PIIError(f"'{operation}' is not a PII operation: {list(OPERATIONS)}")
        body: dict[str, Any] = {"content": content}
        if config:
            body["config"] = config
        try:
            response = httpx.post(
                f"{self._base}/api/v1/pii-protection/{operation}",
                json=body,
                headers={"authorization": f"Bearer {self._key}", "content-type": "application/json"},
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001 - any transport failure is one condition here
            raise PIIError(f"Could not reach Kryptos at {self._base}: {exc}") from exc

        if response.status_code >= 400:
            error = {}
            try:
                error = response.json().get("error", {})
            except Exception:  # noqa: BLE001 - a non-JSON error body is still an error
                pass
            raise PIIError(
                error.get("message", f"Request failed with {response.status_code}"),
                status=response.status_code,
                code=error.get("code"),
            )
        return Result.from_payload(response.json())

    def detect(self, content: str, **config: Any) -> Result:
        return self._call("detect", content, config)

    def redact(self, content: str, **config: Any) -> Result:
        return self._call("redact", content, config)

    def mask(self, content: str, **config: Any) -> Result:
        return self._call("mask", content, config)

    def tokenize(self, content: str, **config: Any) -> Result:
        return self._call("tokenize", content, config)

    def block(self, content: str, **config: Any) -> Result:
        return self._call("block", content, config)


class _Local:
    """In-process execution. No network, no API key, no text leaving the host."""

    def _call(self, operation: str, content: str, config: dict[str, Any] | None) -> Result:
        from kryptos_pii.engine import run

        outcome = run(content, operation=operation, config=config or {})
        payload = outcome.result.model_dump(mode="json")
        payload["vault"] = outcome.vault
        return Result.from_payload(payload)

    def detect(self, content: str, **config: Any) -> Result:
        return self._call("detect", content, config)

    def redact(self, content: str, **config: Any) -> Result:
        return self._call("redact", content, config)

    def mask(self, content: str, **config: Any) -> Result:
        return self._call("mask", content, config)

    def tokenize(self, content: str, **config: Any) -> Result:
        return self._call("tokenize", content, config)

    def block(self, content: str, **config: Any) -> Result:
        return self._call("block", content, config)


local = _Local()


def protect(text: str, **config: Any) -> str:
    """The one-liner from CLAUDE.md section 28: redact locally, return the text."""
    return local.redact(text, **config).text or text


__all__ = ["Finding", "Kryptos", "PIIError", "Result", "local", "protect"]
