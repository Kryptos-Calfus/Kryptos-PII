"""Stop a prompt carrying personal information before the model ever sees it.

This runs as a Claude Code ``UserPromptSubmit`` hook, which fires after the user
hits enter and before the prompt is sent to the model. Claude Code hands us the
prompt on stdin as JSON and reads our decision from stdout.

Why this blocks rather than rewrites
------------------------------------
``UserPromptSubmit`` cannot modify the prompt. There is no field for it: a hook
may let the prompt through, append context to it, or block it outright. So the
only way to keep personal information away from the model is to not send the
prompt at all.

That is a stronger guarantee than rewriting would have been. There is no
transformed payload to get subtly wrong and no path where a half-redacted
prompt still goes out. Either the prompt is clean and proceeds untouched, or it
is stopped and the user is shown a redacted version to resubmit.

The skill and the MCP tools in this plugin are a different mechanism for a
different problem: text Claude is already handling and is about to write
somewhere. They are not what keeps the prompt clean, and they cannot be --
by the time a tool runs, the model has read the prompt.

Fail closed
-----------
Every failure here blocks. A missing SDK, a missing checkpoint, an unparseable
event, an unexpected exception: all of them stop the prompt. A PII filter that
fails open is worse than no filter, because the user believes they are covered
and nobody finds out otherwise.

No prompt leaves this machine. The hook imports ``kryptos_pii_local`` and never
``kryptos_pii_client``; there is no code path here that sends text to the
hosted API, by construction rather than by configuration.

What it can send, when ``KRYPTOS_API_KEY`` is set, is the *decision*: a verdict,
the kinds of thing found, how many, and how long it took. That is the local
runtime reporting to the control plane (CLAUDE.md section 23), the same call the
local SDKs make, and it is what puts your Claude Code prompts on the Kryptos
console. The prompt, the matched values and a hash of either are not in it --
a hash of a short prompt is a guessable prompt.

Reporting is best effort and comes after the decision. The console being
unreachable must never decide whether a prompt is safe.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# Where to record what happened, when the operator asks. Diagnostics carry the
# sanitized prompt and the finding types; the raw prompt and the matched values
# are never written, which is the whole point of the log existing.
DIAGNOSTIC_LOG = os.environ.get("KRYPTOS_PII_HOOK_LOG")

# An escape hatch that has to be deliberate. Set it and the hook reports but
# does not block -- the monitor mode the platform offers everywhere else. It is
# off unless explicitly set, and it says so in the block message so nobody runs
# in it by accident for a month.
ENFORCE = os.environ.get("KRYPTOS_PII_HOOK_ENFORCE", "true").strip().lower() != "false"

# Where to report decisions, if anywhere. Both must be set: a key with no URL
# has nowhere to go, and a URL with no key would be refused.
CONSOLE_URL = os.environ.get("KRYPTOS_BASE_URL", "").strip().rstrip("/")
CONSOLE_KEY = os.environ.get("KRYPTOS_API_KEY", "").strip()
# A person sitting at Claude Code, as the console will label it.
AGENT_ID = os.environ.get("KRYPTOS_AGENT_ID", "claude-code").strip() or "claude-code"

# The session Claude Code is in, filled from the hook event when it carries one,
# so a whole conversation groups into one session in the console.
SESSION_ID: str | None = None


def _report(decision: str, *, would: str | None = None, **facts: Any) -> None:
    """Record on the console that a prompt was checked here.

    Metadata only, and never on the critical path: every failure is swallowed,
    because a control plane that cannot be reached is not a reason to hold up
    somebody's prompt. The detection already happened, locally, either way.
    """
    if not (CONSOLE_URL and CONSOLE_KEY):
        return
    try:
        import urllib.error
        import urllib.request

        types = list(facts.get("types") or [])
        codes = [f"PII_{kind.upper()}" for kind in types]
        if facts.get("error"):
            # A filter that failed is the most important thing this can report.
            codes = [f"HOOK_{str(facts['error']).upper()}"]
        body = json.dumps(
            {
                "extension": "pii-protection",
                "operation": "redact",
                "decision": decision,
                "would_decision": would or decision,
                "risk": facts.get("risk") or ("high" if types else "none"),
                "reason_codes": codes,
                "finding_types": types,
                "finding_count": int(facts.get("found") or 0),
                # Deliberately no content_hash: a prompt is short enough that a
                # hash of one is a prompt anybody can recover by guessing.
                "content_bytes": int(facts.get("bytes") or 0),
                "latency_ms": float(facts.get("latency_ms") or 0.0),
                "agent_id": AGENT_ID,
                "session_id": SESSION_ID,
            }
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - the URL is the operator's own
            f"{CONSOLE_URL}/api/v1/executions/report",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {CONSOLE_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310
            response.read()
    except Exception as exc:  # noqa: BLE001 - reporting must never break the hook
        _log({"report_failed": f"{type(exc).__name__}: {exc}"})


def _log(payload: dict[str, Any]) -> None:
    """Append one diagnostic record. Never raises: a broken log must not
    become a broken hook."""
    if not DIAGNOSTIC_LOG:
        return
    try:
        record = {"ts": time.time(), **payload}
        with open(DIAGNOSTIC_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - diagnostics are best effort by definition
        pass


def _emit(decision: dict[str, Any]) -> None:
    """Hand Claude Code the decision and stop."""
    json.dump(decision, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()
    raise SystemExit(0)


def allow(reason: str, **diagnostics: Any) -> None:
    """Let the prompt through untouched.

    Deliberately silent. Anything written to stdout by this hook on a non-block
    path would be injected into the model's context, and a PII filter that
    narrates itself into every turn is one people switch off.
    """
    _log({"decision": "allow", "reason": reason, **diagnostics})
    _report("allow", **diagnostics)
    raise SystemExit(0)


def block(message: str, **diagnostics: Any) -> None:
    """Stop the prompt. It never reaches the model."""
    _log({"decision": "block", **diagnostics})
    if not ENFORCE:
        _log({"decision": "would_block", "note": "monitor mode", **diagnostics})
        # Monitor mode, reported exactly as the platform models it elsewhere:
        # what happened, and what would have happened under enforcement.
        _report("allow", would="block", **diagnostics)
        print(
            "kryptos-pii: would have blocked this prompt, but "
            "KRYPTOS_PII_HOOK_ENFORCE=false is set.",
            file=sys.stderr,
        )
        raise SystemExit(0)
    _report("block", **diagnostics)
    _emit(
        {
            "decision": "block",
            "reason": message,
            # Without this Claude Code appends "Original prompt: ..." to the
            # block notice, putting every value we just removed back on screen
            # and into the session transcript. Blocking a prompt and then
            # printing it is not blocking it.
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "suppressOriginalPrompt": True,
            },
        }
    )


# Where the prompt lives in the event, most current first.
#
# ``prompt`` is what Claude Code sends; the schema for this event is
# ``{hook_event_name, prompt, source, ...}``. ``user_input`` is accepted because
# the published hook documentation named that field, so a build matching the
# docs rather than the binary stays protected rather than silently unprotected.
PROMPT_FIELDS = ("prompt", "user_input")


def prompt_from(event: dict[str, Any]) -> str:
    """The submitted prompt, or a block if the event does not contain one.

    The distinction this draws is the one that let raw PII through before: a
    *missing* field and an *empty* prompt are not the same thing. Reading an
    unknown payload with ``.get()`` turns "I do not understand this event" into
    "the user submitted nothing", and the second is allowed. So an event with
    no recognised prompt field is a block -- if the shape changes under us
    again, prompts stop rather than flow unchecked.
    """
    for field in PROMPT_FIELDS:
        if field not in event:
            continue
        value = event[field]
        if not isinstance(value, str):
            block(
                "Kryptos PII protection could not read the prompt, so it was "
                "not sent.\n"
                "\n"
                f"  the '{field}' field is {type(value).__name__}, not a string\n"
                "\n"
                "This is a plugin bug. Report it with your Claude Code version.",
                error="prompt_field_wrong_type",
                field=field,
                field_type=type(value).__name__,
            )
        return value

    block(
        "Kryptos PII protection could not find the prompt in this event, so it "
        "was not sent.\n"
        "\n"
        f"  looked for: {', '.join(PROMPT_FIELDS)}\n"
        f"  the event carried: {', '.join(sorted(event)) or '(nothing)'}\n"
        "\n"
        "Claude Code may have changed the hook payload. The prompt was held "
        "back rather than sent unchecked. Please report this with your Claude "
        "Code version.",
        error="no_prompt_field",
        # The keys only. Their values are the prompt, which is the thing we
        # are refusing to leak.
        event_keys=sorted(event),
    )
    raise AssertionError("unreachable: block() exits")  # pragma: no cover


def _sdk() -> Any:
    """The local detector, or a block explaining how to install it."""
    try:
        import kryptos_pii_local
    except ImportError as exc:
        block(
            "Kryptos PII protection could not run, so this prompt was not sent.\n"
            "\n"
            f"  the local SDK is not installed for {sys.executable}\n"
            f"  ({exc})\n"
            "\n"
            "Fix it with:\n"
            "  pip install kryptos-pii-local && kryptos-pii-model download\n"
            "\n"
            "Then point the hook at that interpreter by setting "
            "KRYPTOS_PII_PYTHON to its path.",
            error="sdk_missing",
            interpreter=sys.executable,
        )
    return kryptos_pii_local  # type: ignore[name-defined]


def main() -> int:
    raw_event = sys.stdin.read()

    try:
        event = json.loads(raw_event)
    except (ValueError, TypeError):
        block(
            "Kryptos PII protection could not read the prompt event, so this "
            "prompt was not sent. This is a bug in the plugin; report it with "
            "the Claude Code version.",
            error="unparseable_event",
        )

    # Claude Code names the conversation; carrying it through means a whole
    # session groups into one row in the console rather than scattering.
    global SESSION_ID
    identifier = event.get("session_id")
    SESSION_ID = identifier if isinstance(identifier, str) and identifier else None

    prompt = prompt_from(event)
    size = len(prompt.encode("utf-8"))

    if not prompt.strip():
        # The field was there and held nothing. An empty prompt cannot carry
        # PII, and this is the only case that is allowed without inspection.
        allow("empty prompt", found=0, bytes=size)

    sdk = _sdk()

    started = time.perf_counter()
    try:
        result = sdk.redact(prompt)
    except sdk.ModelMissing as exc:
        block(
            "Kryptos PII protection could not run, so this prompt was not sent.\n"
            "\n"
            f"  no detection checkpoint on this machine\n"
            f"  ({exc})\n"
            "\n"
            "Fix it with:\n"
            "  kryptos-pii-model download",
            error="checkpoint_missing",
            interpreter=sys.executable,
        )
    except Exception as exc:  # noqa: BLE001 - every failure is the same decision
        block(
            "Kryptos PII protection failed, so this prompt was not sent.\n"
            "\n"
            f"  {type(exc).__name__}: {exc}\n"
            "\n"
            "The prompt was held back rather than sent unchecked. Retry, or "
            "set KRYPTOS_PII_HOOK_ENFORCE=false to run unprotected while you "
            "investigate.",
            error="detector_failed",
            interpreter=sys.executable,
        )

    took_ms = round((time.perf_counter() - started) * 1000, 2)

    if not result.findings:
        allow("no findings", found=0, bytes=size, latency_ms=took_ms)

    # ``text`` is the transformed content and is None when nothing changed --
    # but we only get here when something was found, so it is set. The fallback
    # exists so a future change cannot turn this into a None in a user's face.
    sanitized = result.text or prompt
    types = sorted({f.type for f in result.findings})

    _log(
        {
            "found": len(result.findings),
            "types": types,
            "risk": result.risk,
            # The sanitized prompt, which is safe to keep. The raw prompt and
            # the matched values are deliberately absent.
            "sanitized": sanitized,
        }
    )

    block(
        f"Blocked: this prompt contains personal information "
        f"({', '.join(types)}), so it was not sent to the model.\n"
        "\n"
        "Here it is with the values removed. Check it and send this instead:\n"
        "\n"
        f"{sanitized}\n",
        found=len(result.findings),
        types=types,
        sanitized=sanitized,
        risk=str(result.risk),
        bytes=size,
        latency_ms=took_ms,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - last line of fail-closed defence
        # Nothing should reach here. If something does, the prompt still must
        # not go out on the assumption that it was checked.
        json.dump(
            {
                "decision": "block",
                "reason": (
                    "Kryptos PII protection crashed, so this prompt was not sent.\n"
                    f"\n  {type(exc).__name__}: {exc}\n"
                ),
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        raise SystemExit(0)
