"""Findings in, a structured decision out.

The split is deliberate: :mod:`detector` answers "what is here", this module
answers "what should happen to it". Only the second half is policy, and keeping
it apart is what lets the same detector serve the hosted API, the SDK and the
MCP server without any of them re-deciding anything.

Every outcome is deterministic. Nothing here consults a model, and the same text
with the same configuration always produces the same decision and the same
reason codes (CLAUDE.md Rule 4).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Any

from kryptos_pii.contract import Decision, ExtensionResult, Finding, Risk
from kryptos_pii.detector import DEFAULT_THRESHOLD, Detection, detect
from kryptos_pii.taxonomy import category_of, is_high_risk, reason_codes_for

EXTENSION_NAME = "pii-protection"
EXTENSION_VERSION = "0.3.0"

OPERATIONS = ("detect", "redact", "mask", "tokenize", "block", "audit")

# The operation decides what happens to the content; the orchestrator's policy
# engine may still escalate it, which is why this never returns the final word.
_OPERATION_DECISION = {
    "detect": Decision.LOG,
    "audit": Decision.LOG,
    "redact": Decision.REDACT,
    "mask": Decision.MASK,
    "tokenize": Decision.TOKENIZE,
    "block": Decision.BLOCK,
}

DEFAULT_CONFIG: dict[str, Any] = {
    "detector_mode": "hybrid",
    "action": "redact",
    "threshold": DEFAULT_THRESHOLD,
    "min_confidence": 0.0,
}


@dataclass(frozen=True)
class Outcome:
    result: ExtensionResult
    # token -> original value. Returned to the caller in local and SDK use, and
    # dropped by the hosted service, which must not become a store of plaintext.
    vault: dict[str, str]


def _token_secret() -> bytes:
    """Keyed so tokens cannot be reversed by guessing the value.

    A deployment that wants tokens stable across restarts sets the variable; the
    default is a fresh per-process key, which is the safer thing to forget.
    """
    configured = os.environ.get("KRYPTOS_PII_TOKEN_SECRET")
    return configured.encode("utf-8") if configured else os.urandom(32)


_SECRET = _token_secret()


def _token_for(value: str, label: str) -> str:
    digest = hmac.new(_SECRET, value.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return f"<{label.upper()}_{digest}>"


def _risk_of(detections: list[Detection]) -> Risk:
    if not detections:
        return Risk.NONE
    # Risk is assessed on the category, so a specific type inherits it: an
    # 'aadhaar' finding is high risk because 'government_id' is, exactly as it
    # was when the detector only ever said 'government_id'.
    if any(is_high_risk(d.type) for d in detections):
        return Risk.HIGH
    return Risk.MEDIUM if len(detections) > 1 else Risk.LOW


def _rewrite(text: str, detections: list[Detection], action: str) -> tuple[str, dict[str, str]]:
    """Apply the action to every detected span. Spans are non-overlapping and in
    order, so one left-to-right pass is correct and offsets stay meaningful."""
    out: list[str] = []
    vault: dict[str, str] = {}
    cursor = 0
    for detection in detections:
        original = text[detection.start : detection.end]
        if action == "redact":
            replacement = f"[{detection.type.upper()}]"
        elif action == "mask":
            # Length is preserved so the shape of a record survives masking.
            replacement = "*" * len(original)
        elif action == "tokenize":
            replacement = _token_for(original, detection.type)
            vault[replacement] = original
        else:
            replacement = original
        out.append(text[cursor : detection.start])
        out.append(replacement)
        cursor = detection.end
    out.append(text[cursor:])
    return "".join(out), vault


def resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_CONFIG, **(config or {})}
    if merged["detector_mode"] not in ("hybrid", "regex"):
        raise ValueError(f"Unknown detector_mode '{merged['detector_mode']}'")
    if merged["action"] not in OPERATIONS:
        raise ValueError(f"Unknown action '{merged['action']}'")
    threshold = float(merged["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    merged["threshold"] = threshold
    merged["min_confidence"] = float(merged["min_confidence"])
    return merged


def run(
    text: str,
    *,
    operation: str = "detect",
    config: dict[str, Any] | None = None,
) -> Outcome:
    """Execute one operation over ``text``.

    ``operation`` is what the caller asked for; ``config['action']`` is what the
    installation configured. The operation wins when it names a transform, so a
    caller can POST /v1/pii/redact without reconfiguring the installation, and
    the configured action applies to the generic 'detect' entry point.
    """
    started = time.perf_counter()
    settings = resolve_config(config)
    if operation not in OPERATIONS:
        raise ValueError(f"'{operation}' is not an operation this extension declares")

    detections = [
        d
        for d in detect(
            text,
            detector_mode=settings["detector_mode"],
            threshold=settings["threshold"],
        )
        if d.confidence >= settings["min_confidence"]
    ]

    # 'detect' defers to whatever the installation configured, which is how one
    # endpoint serves an installation set to redact and another set to block.
    # 'audit' never defers: it records that personal data was present and leaves
    # the content alone, so it stays usable as a pure observation mode.
    if operation == "detect":
        action = settings["action"]
    elif operation == "audit":
        action = "audit"
    else:
        action = operation
    decision = _OPERATION_DECISION[action]
    if not detections:
        # Nothing found is an allow, whatever the configured action was: there is
        # nothing to redact and nothing to block over.
        decision = Decision.ALLOW
        action = "detect"

    transformed: str | None = None
    vault: dict[str, str] = {}
    if detections and action in ("redact", "mask", "tokenize"):
        transformed, vault = _rewrite(text, detections, action)

    findings = [
        Finding(
            type=d.type,
            category=category_of(d.type),
            start=d.start,
            end=d.end,
            action=decision,
            confidence=d.confidence,
            detector=d.detector,
        )
        for d in detections
    ]
    # Both the specific code and its category, so a policy can match
    # PII_AADHAAR for precision or PII_GOVERNMENT_ID for breadth, and a policy
    # written before this change keeps firing.
    reason_codes = sorted({code for d in detections for code in reason_codes_for(d.type)})
    if detections and operation == "block":
        reason_codes.append("PII_PRESENT_BLOCKED")

    result = ExtensionResult(
        decision=decision,
        risk=_risk_of(detections),
        findings=findings,
        transformed_content=transformed,
        reason_codes=reason_codes,
        extension=EXTENSION_NAME,
        extension_version=EXTENSION_VERSION,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        metadata={
            "detector_mode": settings["detector_mode"],
            "threshold": settings["threshold"],
            "applied_action": action,
            "finding_count": len(detections),
        },
    )
    return Outcome(result=result, vault=vault)
