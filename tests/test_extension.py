"""What the PII extension must keep doing.

These run in ``detector_mode="regex"``, so they need no checkpoint and finish in
milliseconds. That is deliberate: the deterministic stages are the ones that
carry the high-risk categories (keys, cards, labelled identifiers), so they are
the ones that must never regress silently. The classifier's own recall is
measured by ``finetune/evaluate.py`` against a held-out split, which is the
right tool for that and the wrong tool for a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kryptos_pii.classify import classify, luhn_valid
from kryptos_pii.contract import Decision, ExtensionResult, Risk
from kryptos_pii.engine import resolve_config, run

REGEX = {"detector_mode": "regex"}


# --- classification -------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("priya@acme.com", "email"),
        ("9812345678", "phone"),            # Indian mobile, not an account
        ("+91 98123 45678", "phone"),
        ("0123456789", "financial"),        # 10 digits but not a mobile prefix
        ("4111 1111 1111 1111", "payment_card"),
        ("123456789012345", "financial"),   # 15 digits, fails Luhn
        ("ABCDE1234F", "government_id"),
        ("123-45-6789", "government_id"),
        ("192.168.1.4", "network_address"),
        ("01/02/1990", "date_of_birth"),
        ("Priya Sharma", "person_name"),
        ("742 Evergreen Terrace", "address"),
    ],
)
def test_categories(value: str, expected: str) -> None:
    assert classify(value) == expected


def test_luhn_separates_a_card_from_a_long_number() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("123456789012345")


# --- deterministic detection ---------------------------------------------


def test_api_keys_and_cards_never_depend_on_the_model() -> None:
    text = "key sk-abcdefghijklmnop1234 and card 4111 1111 1111 1111"
    result = run(text, operation="redact", config=REGEX).result
    assert "sk-abcdefghijklmnop1234" not in result.transformed_content
    assert "4111 1111 1111 1111" not in result.transformed_content
    assert set(result.reason_codes) == {"PII_SECRET", "PII_PAYMENT_CARD"}


def test_a_field_label_identifies_its_value() -> None:
    text = "DOB: 01/02/1990, Aadhaar: 1234 5678 9012, password: Hunter2!xyz"
    result = run(text, operation="redact", config=REGEX).result
    assert result.transformed_content == (
        "DOB: [DATE_OF_BIRTH], Aadhaar: [GOVERNMENT_ID], password: [SECRET]"
    )


def test_a_label_in_prose_is_not_a_finding() -> None:
    """The field stage runs ahead of the model, so its precision is load-bearing."""
    for text in (
        "Account is closed and the server is slow.",
        "Contact 5 people today. The build is fine.",
    ):
        assert run(text, operation="detect", config=REGEX).result.findings == []


def test_a_value_span_stops_at_the_value() -> None:
    """A span that runs on would delete text that is not personal."""
    result = run("acct 123456789012345 and ip 10.2.3.4", operation="redact", config=REGEX).result
    assert result.transformed_content == "acct [FINANCIAL] and ip [NETWORK_ADDRESS]"


# --- actions --------------------------------------------------------------


def test_each_action_produces_its_own_decision() -> None:
    text = "email priya@acme.com"
    for operation, decision in (
        ("detect", Decision.REDACT),   # falls through to the configured action
        ("redact", Decision.REDACT),
        ("mask", Decision.MASK),
        ("tokenize", Decision.TOKENIZE),
        ("block", Decision.BLOCK),
        ("audit", Decision.LOG),
    ):
        assert run(text, operation=operation, config=REGEX).result.decision is decision


def test_audit_records_without_changing_anything() -> None:
    """Observation mode: it must stay safe to turn on in front of live traffic."""
    result = run("email priya@acme.com", operation="audit", config=REGEX).result
    assert result.decision is Decision.LOG
    assert result.transformed_content is None
    assert result.findings


def test_masking_preserves_length() -> None:
    result = run("email priya@acme.com", operation="mask", config=REGEX).result
    assert result.transformed_content == "email " + "*" * len("priya@acme.com")


def test_tokens_are_reversible_only_with_the_vault() -> None:
    outcome = run("email priya@acme.com", operation="tokenize", config=REGEX)
    masked = outcome.result.transformed_content
    assert "priya@acme.com" not in masked
    restored = masked
    for token, original in outcome.vault.items():
        restored = restored.replace(token, original)
    assert restored == "email priya@acme.com"


def test_clean_text_is_allowed_whatever_the_action() -> None:
    result = run("The deploy finished at noon.", operation="block", config=REGEX).result
    assert result.decision is Decision.ALLOW
    assert result.risk is Risk.NONE
    assert result.transformed_content is None


def test_high_risk_categories_raise_the_risk() -> None:
    assert run("card 4111 1111 1111 1111", operation="detect", config=REGEX).result.risk is Risk.HIGH


# --- contract -------------------------------------------------------------


def test_findings_never_carry_the_matched_text() -> None:
    """Findings reach the orchestrator's audit trail; content must not."""
    result = run("email priya@acme.com", operation="detect", config=REGEX).result
    for finding in result.findings:
        assert "priya@acme.com" not in finding.model_dump_json()


def test_the_result_satisfies_the_contract() -> None:
    payload = run("email priya@acme.com", operation="redact", config=REGEX).result.model_dump(mode="json")
    assert ExtensionResult.model_validate(payload)


def test_the_manifest_declares_what_the_engine_implements() -> None:
    manifest = yaml.safe_load((ROOT / "extension.yaml").read_text())
    from kryptos_pii.engine import OPERATIONS

    assert set(manifest["operations"]) == set(OPERATIONS)
    assert manifest["default_config"] == resolve_config(None)
    assert set(manifest["scopes"]) == {f"pii:{op}" for op in OPERATIONS}


def test_unknown_configuration_is_refused() -> None:
    with pytest.raises(ValueError):
        run("x", operation="detect", config={"detector_mode": "telepathy"})
    with pytest.raises(ValueError):
        run("x", operation="detect", config={"threshold": 5})
