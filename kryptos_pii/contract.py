"""The Kryptos extension contract, as this extension sees it.

These models are a deliberate copy of ``kryptos.contract`` rather than an import
of it. The orchestrator and the extension are separate repositories and separate
deployables; an import would couple their release cycles and let orchestrator
internals leak into a security component that is supposed to be replaceable.

What keeps the copy honest is ``contract_version``: the orchestrator sends it on
every call and the extension refuses a version it does not implement, so a drift
between the two is a loud error rather than a silently wrong decision.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0"


class Decision(StrEnum):
    ALLOW = "allow"
    LOG = "log"
    REDACT = "redact"
    MASK = "mask"
    TOKENIZE = "tokenize"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


class Risk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RequestSource(StrEnum):
    AGENT_PROMPT = "agent_prompt"
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RENDERED_OUTPUT = "rendered_output"
    MEMORY = "memory"
    DOCUMENT = "document"
    OTHER = "other"


class ExecutionRequest(BaseModel):
    """What the orchestrator sends. Extra keys are ignored rather than rejected,
    so a newer orchestrator can add a field without breaking this version."""

    model_config = ConfigDict(extra="ignore")

    request_id: str
    user_id: str
    extension: str
    operation: str
    content: str = ""
    source: RequestSource = RequestSource.OTHER
    agent_id: str | None = None
    session_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    """One span this extension decided is personal information.

    There is no ``value`` field, and that is the point: findings travel into the
    orchestrator's audit trail, which must never hold the matched content.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    start: int | None = None
    end: int | None = None
    action: Decision = Decision.LOG
    confidence: float | None = None
    detector: str | None = None


class ExtensionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    risk: Risk = Risk.NONE
    findings: list[Finding] = Field(default_factory=list)
    transformed_content: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    extension: str
    extension_version: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["ok", "degraded", "down"]
    extension: str
    version: str
    contract_version: str = CONTRACT_VERSION
