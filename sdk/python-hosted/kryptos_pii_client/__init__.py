"""The hosted PII client: Kryptos does the masking.

    pip install kryptos-pii-client

    from kryptos_pii_client import Kryptos

    pii = Kryptos()                       # reads KRYPTOS_API_KEY
    pii.redact("Call Priya on 9812345678").text

One dependency, no model, no checkpoint, nothing to download. Text is sent to
the Kryptos control plane over TLS, which authenticates the key, checks its
scopes, runs the extension, meters the call and writes the audit record.

If the text must not leave the machine, this is the wrong package: install
``kryptos-pii-local`` instead. The two return the same :class:`Result`, so
switching is one import line. That is the only difference that matters, and it
is a privacy decision rather than a performance one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__version__ = "0.4.0"

DEFAULT_BASE_URL = os.environ.get("KRYPTOS_BASE_URL", "https://api.kryptos.ai")
EXTENSION = "pii-protection"
OPERATIONS = ("detect", "redact", "mask", "tokenize", "block", "audit")


class PIIError(RuntimeError):
    """The extension could not reach a decision.

    ``status`` and ``code`` are set when Kryptos refused the call rather than
    failing to answer it -- a missing scope, a revoked key, an extension that is
    not installed. Those are the platform working, not a bug in this client.
    """

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class Finding:
    """One detected span. The matched text is deliberately absent."""

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
        """Put the original values back.

        The hosted service does not return the token-to-value mapping by
        default -- holding a pile of customer plaintext would be a worse
        liability than the problem this extension solves -- so this works only
        for a deployment that has explicitly opted in.
        """
        if not self.vault:
            raise PIIError(
                "This result carries no token mapping, so it cannot be reversed. "
                "The hosted service does not return the vault by default. Use "
                "kryptos-pii-local if you need reversible tokenization."
            )
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

    The key must carry the matching ``pii:<operation>`` scope and the extension
    must be installed on the account, or the call is refused.
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
                "For execution that keeps text on this machine and needs no key, "
                "install kryptos-pii-local instead."
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
                f"{self._base}/api/v1/{EXTENSION}/{operation}",
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
        """Report what is in the text without changing it."""
        return self._call("detect", content, config)

    def redact(self, content: str, **config: Any) -> Result:
        """Replace personal information with its category, e.g. ``[EMAIL]``."""
        return self._call("redact", content, config)

    def mask(self, content: str, **config: Any) -> Result:
        """Replace personal information with asterisks, preserving length."""
        return self._call("mask", content, config)

    def tokenize(self, content: str, **config: Any) -> Result:
        """Replace personal information with stable, reversible tokens."""
        return self._call("tokenize", content, config)

    def block(self, content: str, **config: Any) -> Result:
        """Refuse the text outright when it carries personal information."""
        return self._call("block", content, config)

    def audit(self, content: str, **config: Any) -> Result:
        """Record what is in the text, changing nothing."""
        return self._call("audit", content, config)


__all__ = ["DEFAULT_BASE_URL", "Finding", "Kryptos", "PIIError", "Result"]
