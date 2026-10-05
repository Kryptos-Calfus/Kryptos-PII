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

Nothing leaves this machine. The hook imports ``kryptos_pii_local`` and never
``kryptos_pii_client``; there is no code path here that can reach the hosted
API, by construction rather than by configuration.
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
    raise SystemExit(0)


def block(message: str, **diagnostics: Any) -> None:
    """Stop the prompt. It never reaches the model."""
    _log({"decision": "block", **diagnostics})
    if not ENFORCE:
        _log({"decision": "would_block", "note": "monitor mode", **diagnostics})
        print(
            "kryptos-pii: would have blocked this prompt, but "
            "KRYPTOS_PII_HOOK_ENFORCE=false is set.",
            file=sys.stderr,
        )
        raise SystemExit(0)
    _emit({"decision": "block", "reason": message})


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

    prompt = prompt_from(event)

    if not prompt.strip():
        # The field was there and held nothing. An empty prompt cannot carry
        # PII, and this is the only case that is allowed without inspection.
        allow("empty prompt", found=0)

    sdk = _sdk()

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

    if not result.findings:
        allow("no findings", found=0)

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
