"""The local PII SDK: the text never leaves this machine.

    pip install kryptos-pii-local
    kryptos-pii-model download          # the checkpoint, once

    from kryptos_pii_local import redact

    redact("Call Priya on 9812345678").text
    # 'Call [PERSON_NAME] on [PHONE]'

No API key, no network, no control plane. The detector runs in this process
against a checkpoint on this disk, which is the only arrangement that can
honestly promise the text stays put.

Six operations, explained in full in ``docs/operations.md``:

    detect     find it, change nothing
    audit      find it, change nothing, ever
    redact     'priya@acme.com' -> '[EMAIL]'
    mask       'priya@acme.com' -> '**************'
    tokenize   'priya@acme.com' -> '<EMAIL_8ae616f5f64a>', and back again
    block      refuse the text outright

If you would rather Kryptos did the masking -- no checkpoint to host, calls
metered and audited in your dashboard -- install ``kryptos-pii-client``
instead. Both return the same :class:`Result`, so switching is one import line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__version__ = "0.4.0"

OPERATIONS = ("detect", "redact", "mask", "tokenize", "block", "audit")


class PIIError(RuntimeError):
    """The extension could not reach a decision."""


class ModelMissing(PIIError):
    """No checkpoint on this machine, so there is nothing to detect with.

    Raised rather than quietly falling back to regex. A PII filter that silently
    stops detecting most of what it should is worse than one that is plainly
    down: you would never find out.
    """


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
        """The content to pass on: rewritten when the operation rewrote it."""
        return self.transformed_content

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @property
    def found_anything(self) -> bool:
        return bool(self.findings)

    def detokenize(self, text: str) -> str:
        """Put the original values back into a tokenized round trip.

        Only meaningful for a ``tokenize`` result. The mapping never left this
        process, which is what makes the round trip safe.
        """
        if not self.vault:
            raise PIIError(
                "This result carries no token mapping. Only tokenize() produces one."
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


# --- the checkpoint ------------------------------------------------------


def model_path() -> str:
    """Where the detector will load the checkpoint from."""
    from kryptos_pii import model

    return str(model.model_dir())


def model_installed() -> bool:
    """Whether a usable checkpoint is on this machine."""
    from kryptos_pii import model

    return model.is_installed()


def download_model(repo_id: str | None = None, **kwargs: Any) -> str:
    """Fetch the checkpoint. Equivalent to ``kryptos-pii-model download``.

    Resumable and cached, so calling it when the checkpoint is already present
    does no network work.
    """
    from kryptos_pii import model

    return str(model.download(repo_id or model.DEFAULT_REPO, **kwargs))


def _require_model() -> None:
    if not model_installed():
        raise ModelMissing(
            f"No PII detection checkpoint at {model_path()}.\n"
            "Fetch it once with:  kryptos-pii-model download\n"
            "Or, in code:         kryptos_pii_local.download_model()\n"
            "To run without it, pass detector_mode='regex' -- materially lower "
            "recall, for environments that cannot host the checkpoint."
        )


# --- the operations ------------------------------------------------------


def run(text: str, operation: str = "redact", **config: Any) -> Result:
    """Execute one operation. The six wrappers below are the usual way in."""
    if operation not in OPERATIONS:
        raise PIIError(f"'{operation}' is not a PII operation: {list(OPERATIONS)}")
    if config.get("detector_mode", "hybrid") != "regex":
        _require_model()

    from kryptos_pii.engine import run as _run

    outcome = _run(text, operation=operation, config=config)
    payload = outcome.result.model_dump(mode="json")
    # The vault stays here, in the process that did the tokenizing. That is the
    # only place it is safe, and locally it is also the only place it exists.
    payload["vault"] = outcome.vault
    return Result.from_payload(payload)


def detect(text: str, **config: Any) -> Result:
    """Report what is in the text.

    Defers to the configured ``action``, so this can still rewrite. When you
    need a guarantee that nothing changes, use :func:`audit`.
    """
    return run(text, "detect", **config)


def audit(text: str, **config: Any) -> Result:
    """Report what is in the text and never rewrite it, whatever the config."""
    return run(text, "audit", **config)


def redact(text: str, **config: Any) -> Result:
    """Replace each value with its category: ``priya@acme.com`` -> ``[EMAIL]``.

    The usual choice. The sentence still reads, so whatever consumes the text
    next -- usually a model -- can still work with it.
    """
    return run(text, "redact", **config)


def mask(text: str, **config: Any) -> Result:
    """Replace each value with asterisks, preserving length.

    Use it when the layout matters: a fixed-width export, an aligned diff, a
    terminal table. The masked text says nothing about what was removed.
    """
    return run(text, "mask", **config)


def tokenize(text: str, **config: Any) -> Result:
    """Replace each value with a stable, reversible token.

    The round trip this exists for::

        r = tokenize("Email priya@acme.com")
        reply = model(r.text)        # the model never sees the address
        final = r.detokenize(reply)  # the address is back in the answer
    """
    return run(text, "tokenize", **config)


def block(text: str, **config: Any) -> Result:
    """Refuse the text when it carries personal information.

    Clean text still passes: with nothing found there is nothing to block over,
    and the decision is ``allow``.
    """
    return run(text, "block", **config)


def protect(text: str, **config: Any) -> str:
    """Redact and hand back the string. The one-liner, when you want no ceremony.

    Returns the original text unchanged when nothing was found.
    """
    return redact(text, **config).text or text


__all__ = [
    "Finding",
    "ModelMissing",
    "PIIError",
    "Result",
    "audit",
    "block",
    "detect",
    "download_model",
    "mask",
    "model_installed",
    "model_path",
    "protect",
    "redact",
    "run",
    "tokenize",
]
