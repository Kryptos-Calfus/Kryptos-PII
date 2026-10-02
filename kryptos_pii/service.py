"""The hosted extension runtime.

Two endpoints are the whole contract with the orchestrator:

    POST /v1/execute   an ExecutionRequest in, an ExtensionResult out
    GET  /health       a HealthReport

Everything else here exists for people rather than for the orchestrator: the
ergonomic ``/v1/pii/{operation}`` routes are what the SDK and the MCP server
call, and they are a thin translation into the same :func:`engine.run`. No route
decides anything of its own (CLAUDE.md Rule 7).

What this service never does is keep content. Nothing is written to disk, no
request body is logged, and the tokenization vault is returned to the caller in
the response rather than retained -- a hosted PII filter that accumulated
plaintext would be a worse liability than the problem it solves.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from kryptos_pii.contract import (
    CONTRACT_VERSION,
    ExecutionRequest,
    ExtensionResult,
    HealthReport,
)
from kryptos_pii.detector import DEFAULT_MODEL_DIR, ModelUnavailable, checkpoint_for
from kryptos_pii.engine import EXTENSION_NAME, EXTENSION_VERSION, OPERATIONS, run

# The hosted service returns tokens but not the mapping unless the deployment
# opts in. Off by default: the mapping is plaintext, and the privacy boundary
# this extension sells is that plaintext does not travel.
RETURN_VAULT = os.environ.get("KRYPTOS_PII_RETURN_VAULT", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    # Load the checkpoint before the first request rather than during it, so a
    # cold start shows up here and not as an orchestrator timeout.
    try:
        checkpoint_for().load()
        app.state.model_ready = True
        app.state.model_error = None
    except (ModelUnavailable, ImportError) as exc:
        app.state.model_ready = False
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Kryptos PII Protection",
    version=EXTENSION_VERSION,
    description="Detects and removes personal information. A Kryptos platform extension.",
    lifespan=lifespan,
)


class OperationBody(BaseModel):
    content: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


def _check_contract(version: str | None) -> None:
    if version and version.split(".")[0] != CONTRACT_VERSION.split(".")[0]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This extension implements contract {CONTRACT_VERSION}; "
                f"the caller asked for {version}"
            ),
        )


def _execute(content: str, operation: str, config: dict[str, Any]) -> tuple[ExtensionResult, dict[str, str]]:
    if operation not in OPERATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{operation}' is not an operation this extension declares: {list(OPERATIONS)}",
        )
    try:
        outcome = run(content, operation=operation, config=config)
    except ModelUnavailable as exc:
        # 503, not a quiet regex fallback: the caller's failure policy decides
        # whether to fail open or closed, and it can only do that if it is told.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return outcome.result, outcome.vault


@app.post("/v1/execute", response_model=ExtensionResult)
def execute(
    request: ExecutionRequest,
    x_kryptos_contract_version: str | None = Header(default=None),
) -> ExtensionResult:
    """The orchestrator's entry point. One request, one decision."""
    _check_contract(x_kryptos_contract_version)
    result, _vault = _execute(request.content, request.operation, request.config)
    return result


@app.post("/v1/pii/{operation}")
def execute_operation(operation: str, body: OperationBody) -> dict[str, Any]:
    """The direct entry point, for the SDK, the MCP server and curl.

    Same engine, same decision. It returns the tokenization vault when the
    deployment allows it, which the orchestrator path never does.
    """
    result, vault = _execute(body.content, operation, body.config)
    payload = result.model_dump(mode="json")
    if vault and RETURN_VAULT:
        payload["vault"] = vault
    return payload


@app.get("/health", response_model=HealthReport)
def health() -> HealthReport:
    ready = getattr(app.state, "model_ready", False)
    return HealthReport(
        status="ok" if ready else "degraded",
        extension=EXTENSION_NAME,
        version=EXTENSION_VERSION,
        model_dir=str(DEFAULT_MODEL_DIR),
        model_loaded=ready,
        detail=getattr(app.state, "model_error", None),
    )
