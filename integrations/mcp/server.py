"""PII Protection as an MCP server.

This is how Claude reaches the extension: Claude Desktop, Claude Code or any
other MCP client gets four tools and can mask text before it is pasted into a
prompt, written to a file or sent to another tool.

It is an adapter and nothing more (CLAUDE.md Rule 7). Every tool call ends up in
the same :func:`kryptos_pii.engine.run` that the hosted API and the SDK use, so
there is exactly one detector and one set of decisions to reason about.

Transport is stdio, which is what Claude Desktop launches.

    # local: text never leaves the machine, no API key
    uv run python integrations/mcp/server.py

    # hosted: Kryptos does the masking
    KRYPTOS_PII_MODE=hosted KRYPTOS_API_KEY=kr_live_... \
        uv run python integrations/mcp/server.py

Claude Desktop configuration lives in integrations/mcp/README.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "sdk" / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mcp.server.fastmcp import FastMCP  # noqa: E402

MODE = os.environ.get("KRYPTOS_PII_MODE", "local").lower()

mcp = FastMCP("kryptos-pii")


def _client():  # noqa: ANN202 - either SDK surface; both expose the same methods
    import kryptos_pii_sdk

    if MODE == "hosted":
        return kryptos_pii_sdk.Kryptos()
    return kryptos_pii_sdk.local


def _summarise(result: Any, *, include_text: bool) -> dict[str, Any]:
    """What Claude gets back.

    The findings carry types, offsets and confidence but never the matched
    values. That is not an oversight: this tool's output goes into a model's
    context, and putting the PII back into the transcript would undo the
    masking it was asked to perform.
    """
    payload: dict[str, Any] = {
        "decision": result.decision,
        "risk": result.risk,
        "found": len(result.findings),
        "types": sorted({f.type for f in result.findings}),
        "reason_codes": result.reason_codes,
        "mode": MODE,
    }
    if include_text:
        payload["text"] = result.text
    return payload


@mcp.tool()
def pii_detect(text: str) -> dict[str, Any]:
    """Report whether text contains personal information, without changing it.

    Use this to decide whether something is safe to send on. It returns the
    categories found and how many, never the values themselves.
    """
    return _summarise(_client().detect(text), include_text=False)


@mcp.tool()
def pii_redact(text: str) -> dict[str, Any]:
    """Replace personal information with its category, e.g. [EMAIL].

    This is the one to use before putting user-supplied text into a prompt, a
    log, a commit message or an issue. ``text`` in the result is the safe
    version; use it in place of the original.
    """
    return _summarise(_client().redact(text), include_text=True)


@mcp.tool()
def pii_mask(text: str) -> dict[str, Any]:
    """Replace personal information with asterisks, preserving length.

    Prefer this to redaction when the shape of a record matters, such as a
    fixed-width export or a screenshot-like transcript.
    """
    return _summarise(_client().mask(text), include_text=True)


@mcp.tool()
def pii_tokenize(text: str) -> dict[str, Any]:
    """Replace personal information with stable, reversible tokens.

    Use this when the text has to survive a round trip -- a model rewrites it
    and the real values go back afterwards. The token-to-value mapping is
    deliberately NOT returned to the model: it stays with the process that did
    the tokenizing, which is the only place it is safe.
    """
    return _summarise(_client().tokenize(text), include_text=True)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
