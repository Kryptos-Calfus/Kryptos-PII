"""PII Protection as an MCP server — the engine behind the Claude plugin.

Claude gets four tools and can mask text before it is pasted into a prompt,
written to a file, put in a commit message or sent to another tool.

This is an adapter and nothing more (CLAUDE.md Rule 7). Every tool call ends up
in the same :func:`kryptos_pii.engine.run` that the hosted API and both SDKs
use, so there is exactly one detector and one set of decisions to reason about.

Transport is stdio, which is what Claude Code and Claude Desktop launch.

    KRYPTOS_PII_MODE=local    text never leaves the machine, no API key (default)
    KRYPTOS_PII_MODE=hosted   Kryptos does the masking; needs KRYPTOS_API_KEY

Install instructions are in README.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Running from a repository checkout rather than an installed plugin: put the
# two SDK packages on the path. Skipped entirely when the directories are not
# there, which is the case for every real installation.
_REPO = Path(__file__).resolve().parents[2]
for _candidate in (_REPO / "sdk" / "python-local", _REPO / "sdk" / "python-hosted", _REPO):
    _is_source_tree = (_candidate / "kryptos_pii_local").exists() or (_candidate / "kryptos_pii").exists()
    if _is_source_tree and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

MODE = os.environ.get("KRYPTOS_PII_MODE", "local").strip().lower() or "local"


def _fail(message: str) -> None:
    """Stop with an actionable message.

    An MCP server that fails silently shows up in Claude as a tool that is just
    missing, with nothing to act on. These go to stderr, where Claude Code
    surfaces them under /plugin.
    """
    print(f"kryptos-pii: {message}", file=sys.stderr)
    raise SystemExit(1)


# The server class was renamed in mcp 2.0: FastMCP became MCPServer. Both are
# in the wild -- the person installing this picks their own interpreter and
# their own pin -- and the three methods used here are identical across the two,
# so support both rather than making the plugin care.
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
    except ImportError as exc:
        import importlib.util

        if importlib.util.find_spec("mcp") is None:
            _fail(
                "the 'mcp' package is not installed for this interpreter.\n"
                f"  interpreter: {sys.executable}\n"
                "  fix: pip install mcp kryptos-pii-local\n"
                "  then set the plugin's 'Python interpreter' option to that interpreter."
            )
        _fail(
            "the installed 'mcp' package has neither MCPServer (2.x) nor FastMCP (1.x).\n"
            f"  interpreter: {sys.executable}\n"
            f"  import error: {exc}\n"
            "  fix: pip install -U mcp"
        )


class _Hosted:
    """Masking happens at Kryptos. Text leaves the machine."""

    def __init__(self) -> None:
        try:
            from kryptos_pii_client import Kryptos
        except ImportError:
            _fail(
                "hosted mode needs the client SDK.\n"
                f"  interpreter: {sys.executable}\n"
                "  fix: pip install kryptos-pii-client"
            )
        if not os.environ.get("KRYPTOS_API_KEY"):
            _fail(
                "hosted mode needs an API key.\n"
                "  fix: set the plugin's 'Kryptos API key' option, or switch the "
                "'Where masking runs' option back to local."
            )
        self._client = Kryptos()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _Local:
    """Masking happens in this process. No network, no API key, no text leaving."""

    def __init__(self) -> None:
        try:
            import kryptos_pii_local
        except ImportError:
            _fail(
                "local mode needs the local SDK.\n"
                f"  interpreter: {sys.executable}\n"
                "  fix: pip install kryptos-pii-local && kryptos-pii-model download\n"
                "  then set the plugin's 'Python interpreter' option to that interpreter."
            )
        self._sdk = kryptos_pii_local

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sdk, name)


_client: Any = None


def client() -> Any:
    """Built on first use, so an unreachable backend is a tool error rather
    than a server that refuses to start."""
    global _client
    if _client is None:
        _client = _Hosted() if MODE == "hosted" else _Local()
    return _client


mcp = _Server("kryptos-pii")


def _summarise(result: Any, *, include_text: bool) -> dict[str, Any]:
    """What Claude gets back.

    Findings carry types, offsets and counts but never the matched values. That
    is not an oversight: this output goes into a model's context, and putting
    the personal information back there would undo the masking it was just
    asked to perform.
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
        payload["text"] = result.text if result.text is not None else "(nothing found; use the original text)"
    return payload


@mcp.tool()
def pii_detect(text: str) -> dict[str, Any]:
    """Report whether text contains personal information, without changing it.

    Use this to decide whether something is safe to send on. Returns the
    categories found and how many, never the values themselves.
    """
    return _summarise(client().detect(text), include_text=False)


@mcp.tool()
def pii_redact(text: str) -> dict[str, Any]:
    """Replace personal information with its category, e.g. [EMAIL].

    The one to reach for by default. Use it before putting user-supplied text
    into a prompt, a log, a commit message or an issue. The sentence still
    reads afterwards, so the redacted text stays usable. ``text`` in the result
    is the safe version; use it in place of the original.
    """
    return _summarise(client().redact(text), include_text=True)


@mcp.tool()
def pii_mask(text: str) -> dict[str, Any]:
    """Replace personal information with asterisks, preserving length.

    Prefer this to redaction when the layout matters: a fixed-width export, an
    aligned table, a diff that has to stay lined up. The masked text says
    nothing about what was removed.
    """
    return _summarise(client().mask(text), include_text=True)


@mcp.tool()
def pii_tokenize(text: str) -> dict[str, Any]:
    """Replace personal information with stable, reversible tokens.

    Use this when the text has to survive a round trip: you rewrite it and the
    real values go back afterwards. The token-to-value mapping is deliberately
    NOT returned to you. It stays with the process that did the tokenizing,
    which is the only place it is safe, and that process puts the values back.
    """
    return _summarise(client().tokenize(text), include_text=True)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
