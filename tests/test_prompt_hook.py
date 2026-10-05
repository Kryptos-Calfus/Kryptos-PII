"""The pre-model hook: does personal information reach the model or not.

These tests run the hook the way Claude Code runs it -- as a subprocess, JSON
on stdin, decision on stdout -- rather than importing it. The contract being
tested is the process contract, and importing the module would skip the part
most likely to break.

The central idea is :func:`model_bound_text`. A ``UserPromptSubmit`` hook cannot
rewrite a prompt; it can only let it through or stop it. So what the model ends
up seeing is a function of the decision:

    block  -> nothing at all
    allow  -> the prompt, byte for byte

Every ordering test below is an assertion about that function. It is the thing
that would actually be wrong if the hook were wired in after the model instead
of before it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPO_HOOK = REPO / "integrations" / "claude-plugin" / "hooks" / "precheck.py"

# The plugin as Claude Code actually installed it, which is a different copy of
# the file on a different path with no repository around it. Requirement: test
# the installed path, not only the working tree.
INSTALLED_ROOT = Path.home() / ".claude" / "plugins" / "cache" / "kryptos" / "kryptos-pii"


def installed_hook() -> Path | None:
    """The newest installed copy of the hook, if the plugin is installed."""
    if not INSTALLED_ROOT.is_dir():
        return None
    versions = sorted(p for p in INSTALLED_ROOT.iterdir() if p.is_dir())
    for version in reversed(versions):
        candidate = version / "hooks" / "precheck.py"
        if candidate.exists():
            return candidate
    return None


# The interpreter that has the SDK. The repo's own venv does not: the detector
# is installed, but kryptos_pii_local lives in sdk/python-local and is not on
# the path here. Prefer an interpreter that can genuinely import it.
def _interpreter() -> str | None:
    candidates = [
        os.environ.get("KRYPTOS_PII_PYTHON"),
        str(Path.home() / "Desktop" / "kryptos" / "kryptos-test" / ".venv" / "bin" / "python"),
        str(Path.home() / ".kryptos-pii" / "bin" / "python"),
        shutil.which("python3"),
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", "import kryptos_pii_local"], capture_output=True
        )
        if probe.returncode == 0:
            return candidate
    return None


PYTHON = _interpreter()
needs_sdk = pytest.mark.skipif(PYTHON is None, reason="no interpreter with kryptos_pii_local")

CANARY = "KRYPTOS_RAW_CANARY_9f7a31c842e14d9b"

SENSITIVE_PROMPT = f"""Customer: {CANARY}
Email: kryptos-canary-847291@example.com
Phone: +1-202-555-0198
SSN: 847-29-1046"""

# Every raw value that must never survive to the model, or to a diagnostic log.
RAW_VALUES = (
    CANARY,
    "kryptos-canary-847291@example.com",
    "+1-202-555-0198",
    "847-29-1046",
)

CLEAN_PROMPT = "Refactor the retry loop in engine.py to use exponential backoff."


# Captured from a real `claude -p` run against a throwaway UserPromptSubmit
# hook, not transcribed from the documentation. That distinction is the whole
# reason this fixture exists: the published docs name the field `user_input`,
# Claude Code sends `prompt`, and a suite built on the documented shape passed
# every test while the hook let raw PII through to the model.
def event_fixture(prompt: str) -> dict:
    return {
        "session_id": "42a010e9-d3f3-4246-a35b-27acdb7072ff",
        "transcript_path": str(REPO / "transcript.jsonl"),
        "cwd": str(REPO),
        "prompt_id": "a03cdde8-7f8a-4163-85a3-13856a426eb8",
        "permission_mode": "default",
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
        "source": "user",
    }


def run_hook(
    prompt: str,
    *,
    hook: Path | None = None,
    env: dict[str, str] | None = None,
    python: str | None = None,
    field: str = "prompt",
    extra_event: dict | None = None,
) -> dict:
    """Invoke the hook as Claude Code does and return its decision.

    An empty dict means the hook said nothing, which is how a hook allows a
    prompt through: silence on stdout, because anything printed there would be
    injected into the model's context.
    """
    event = {**event_fixture(prompt), **(extra_event or {})}
    if field != "prompt":
        event[field] = event.pop("prompt")
    proc = subprocess.run(
        [python or PYTHON, str(hook or REPO_HOOK)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=180,
    )
    assert proc.returncode == 0, f"hook exited {proc.returncode}: {proc.stderr}"
    out = proc.stdout.strip()
    return json.loads(out) if out else {}


def model_bound_text(decision: dict, prompt: str) -> str:
    """What the model would receive, given the hook's decision.

    A blocked prompt is never sent, so the model sees nothing. An allowed
    prompt is sent unchanged -- the hook has no way to alter it.
    """
    if decision.get("decision") == "block":
        return ""
    return prompt


# --- the ordering guarantee ----------------------------------------------


@needs_sdk
def test_the_raw_canary_never_reaches_the_model() -> None:
    """The regression test. If this fails, PII is reaching the model."""
    decision = run_hook(SENSITIVE_PROMPT)
    reaching_model = model_bound_text(decision, SENSITIVE_PROMPT)

    assert CANARY not in reaching_model
    for value in RAW_VALUES:
        assert value not in reaching_model, f"{value!r} reached the model"


@needs_sdk
def test_a_prompt_with_pii_is_blocked_outright() -> None:
    decision = run_hook(SENSITIVE_PROMPT)
    assert decision.get("decision") == "block"
    assert model_bound_text(decision, SENSITIVE_PROMPT) == ""


@needs_sdk
def test_the_block_message_carries_no_raw_value() -> None:
    """The reason is shown to the user and may be recorded. Putting the values
    back into it would undo the block it is explaining."""
    decision = run_hook(SENSITIVE_PROMPT)
    reason = decision.get("reason", "")

    assert reason, "a block must explain itself"
    for value in RAW_VALUES:
        assert value not in reason, f"the block message leaked {value!r}"


@needs_sdk
def test_the_block_message_offers_the_sanitized_prompt() -> None:
    """Blocking without showing the user what to send instead is a dead end."""
    reason = run_hook(SENSITIVE_PROMPT).get("reason", "")
    assert "[EMAIL]" in reason
    assert "[SSN]" in reason
    assert "[PHONE]" in reason


@needs_sdk
def test_a_clean_prompt_passes_through_untouched() -> None:
    """A filter that blocks everything is not protection, it is an outage."""
    decision = run_hook(CLEAN_PROMPT)
    assert decision == {}
    assert model_bound_text(decision, CLEAN_PROMPT) == CLEAN_PROMPT


@needs_sdk
def test_an_empty_prompt_is_allowed() -> None:
    assert run_hook("   ") == {}


# --- fail closed ----------------------------------------------------------


def test_a_missing_sdk_blocks_rather_than_passing_the_prompt_through() -> None:
    """The failure that matters. An interpreter without the SDK must stop the
    prompt, not wave it through as if it had been checked."""
    bare = sys.executable  # the repo venv: no kryptos_pii_local on its path
    probe = subprocess.run([bare, "-c", "import kryptos_pii_local"], capture_output=True)
    if probe.returncode == 0:
        pytest.skip("this interpreter has the SDK, so it cannot test the missing case")

    decision = run_hook(SENSITIVE_PROMPT, python=bare)
    assert decision.get("decision") == "block"
    assert model_bound_text(decision, SENSITIVE_PROMPT) == ""
    assert "pip install kryptos-pii-local" in decision.get("reason", "")


def test_an_unparseable_event_blocks() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_HOOK)],
        input="not json at all",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["decision"] == "block"


# --- diagnostics ----------------------------------------------------------


@needs_sdk
def test_diagnostics_record_the_sanitized_payload_and_never_the_raw_one(
    tmp_path: pytest.TempPathFactory,
) -> None:
    log = Path(str(tmp_path)) / "hook.jsonl"
    run_hook(SENSITIVE_PROMPT, env={"KRYPTOS_PII_HOOK_LOG": str(log)})

    assert log.exists(), "diagnostic mode wrote nothing"
    body = log.read_text()

    for value in RAW_VALUES:
        assert value not in body, f"the diagnostic log leaked {value!r}"

    records = [json.loads(line) for line in body.splitlines() if line.strip()]
    sanitized = [r for r in records if "sanitized" in r]
    assert sanitized, "no sanitized payload was recorded"
    assert "[EMAIL]" in sanitized[-1]["sanitized"]
    assert sorted(sanitized[-1]["types"]) == ["email", "password", "phone", "ssn"]


# --- the installed plugin, not the working copy ---------------------------


@needs_sdk
@pytest.mark.skipif(installed_hook() is None, reason="plugin not installed")
def test_the_installed_plugin_blocks_the_canary_too() -> None:
    """The working copy passing proves nothing about what Claude Code runs."""
    hook = installed_hook()
    assert hook is not None
    decision = run_hook(SENSITIVE_PROMPT, hook=hook)

    assert decision.get("decision") == "block"
    assert model_bound_text(decision, SENSITIVE_PROMPT) == ""
    assert CANARY not in json.dumps(decision)


@pytest.mark.skipif(installed_hook() is None, reason="plugin not installed")
def test_the_installed_plugin_registers_the_hook_on_the_right_event() -> None:
    """A hook on the wrong event would run after the model had already read
    the prompt, which is the bug this whole file exists to prevent."""
    hook = installed_hook()
    assert hook is not None
    manifest = json.loads((hook.parent / "hooks.json").read_text())

    assert "UserPromptSubmit" in manifest["hooks"], manifest["hooks"].keys()
    entry = manifest["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    # Exec form: the script is an argument, so a plugin root containing a space
    # cannot split the command into pieces.
    invocation = " ".join([entry["command"], *entry.get("args", [])])
    assert invocation.endswith("hooks/precheck.sh"), invocation


# --- the payload shape itself --------------------------------------------
#
# These exist because the hook once read a field Claude Code does not send.
# Every prompt looked empty, every prompt was allowed, and the suite passed:
# the tests fed the same wrong field the code read. What was missing was any
# test that the hook understands the shape it is actually given.


@needs_sdk
def test_the_real_prompt_field_with_pii_is_blocked() -> None:
    """The exact event Claude Code sends, with PII in it."""
    decision = run_hook(SENSITIVE_PROMPT, field="prompt")
    assert decision.get("decision") == "block"
    assert model_bound_text(decision, SENSITIVE_PROMPT) == ""


@needs_sdk
def test_the_real_prompt_field_without_pii_is_allowed() -> None:
    decision = run_hook(CLEAN_PROMPT, field="prompt")
    assert decision == {}
    assert model_bound_text(decision, CLEAN_PROMPT) == CLEAN_PROMPT


@needs_sdk
def test_the_legacy_user_input_field_still_works() -> None:
    """Kept so a build matching the documentation stays protected rather than
    silently unprotected."""
    decision = run_hook(SENSITIVE_PROMPT, field="user_input")
    assert decision.get("decision") == "block"


@needs_sdk
def test_an_event_with_no_recognised_prompt_field_blocks() -> None:
    """The regression for the actual bug.

    A missing field must not read as an empty prompt. If Claude Code renames
    the field again, prompts stop; they do not flow unchecked.
    """
    event = {"hook_event_name": "UserPromptSubmit", "source": "user"}
    proc = subprocess.run(
        [PYTHON, str(REPO_HOOK)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        timeout=180,
    )
    decision = json.loads(proc.stdout)
    assert decision["decision"] == "block"
    assert "looked for" in decision["reason"]


@needs_sdk
def test_a_prompt_field_of_the_wrong_type_blocks() -> None:
    proc = subprocess.run(
        [PYTHON, str(REPO_HOOK)],
        input=json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": {"a": 1}}),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert json.loads(proc.stdout)["decision"] == "block"


@needs_sdk
def test_a_genuinely_empty_prompt_is_allowed() -> None:
    """Present and empty is the one case that passes without inspection.
    Absent is not the same thing, and is tested above."""
    assert run_hook("", field="prompt") == {}
    assert run_hook("   \n ", field="prompt") == {}


@needs_sdk
def test_no_raw_value_survives_into_output_whichever_field_carried_it() -> None:
    for field in ("prompt", "user_input"):
        decision = run_hook(SENSITIVE_PROMPT, field=field)
        blob = json.dumps(decision)
        for value in RAW_VALUES:
            assert value not in blob, f"{value!r} leaked via {field}"
