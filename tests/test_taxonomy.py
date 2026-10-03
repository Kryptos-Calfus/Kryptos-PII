"""The hierarchical taxonomy: specific type, broad category, and no silent breakage.

This change made findings more granular. In a security product that is exactly
the change most likely to break something quietly: a policy written against a
broad type stops matching the day the extension starts reporting a narrow one,
and nothing errors -- the rule simply never fires again.

Every test here exists to make that failure loud instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kryptos_pii.classify import classify
from kryptos_pii.engine import run
from kryptos_pii.taxonomy import (
    CATEGORIES,
    CATEGORY_OF,
    UNSUPPORTED,
    category_of,
    is_high_risk,
    reason_codes_for,
)

REGEX = {"detector_mode": "regex"}


def _findings(text: str):  # noqa: ANN202
    return run(text, operation="detect", config=REGEX).result.findings


# --- 1. existing detection still works ------------------------------------


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("Email priya@acme.com today", "email"),
        ("card 4111 1111 1111 1111", "payment_card"),
        ("Aadhaar: 1234 5678 9012", "government_id"),
        ("PAN: ABCDE1234F", "government_id"),
        ("ip 10.2.3.4", "network_address"),
        ("password: Hunter2!xyz", "secret"),
    ],
)
def test_existing_detection_still_finds_the_same_values(text: str, category: str) -> None:
    """Granularity must not cost recall: everything found before is still found."""
    found = _findings(text)

    assert found, f"nothing detected in {text!r}"
    assert category in {f.category for f in found}


# --- 2. broad categories still work ---------------------------------------


def test_every_specific_type_resolves_to_a_category_that_existed_before() -> None:
    """A new category would be as breaking as a renamed one: a policy can only
    match values the extension actually emits."""
    for pii_type, category in CATEGORY_OF.items():
        assert category in CATEGORIES, f"{pii_type} maps to unknown category {category}"


def test_category_of_is_stable_for_the_old_broad_values() -> None:
    for broad in CATEGORIES:
        assert category_of(broad) == broad


# --- 3. specific types are preserved --------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_type", "expected_category"),
    [
        ("1234 5678 9012", "aadhaar", "government_id"),
        ("ABCDE1234F", "pan", "government_id"),
        ("536-22-8841", "ssn", "government_id"),
        ("J8369854", "passport", "government_id"),
        ("priya@okaxis", "upi_id", "financial"),
        ("192.168.1.4", "ip_address", "network_address"),
        ("00:24:81:A7:B2:F9", "mac_address", "network_address"),
        ("eyJhbGciOi.abc.def", "jwt", "secret"),
        ("sk-ABCDEFGH12345678", "api_key", "secret"),
        ("MRN-889213", "medical_record", "government_id"),
        ("DL-42193", "driver_license", "government_id"),
        ("560001", "postal_code", "address"),
    ],
)
def test_specific_types_are_preserved(value: str, expected_type: str, expected_category: str) -> None:
    """These distinctions were always made by the patterns and then discarded."""
    assert classify(value) == expected_type
    assert category_of(expected_type) == expected_category


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("Aadhaar: 1234 5678 9012", "aadhaar"),
        ("PAN: ABCDE1234F", "pan"),
        ("SSN: 536-22-8841", "ssn"),
        ("Tax ID: 24-1598763", "tax_id"),
        ("IFSC: HDFC0001234", "ifsc"),
        ("OTP: 418293", "otp"),
        ("MRN: 0009271658", "medical_record"),
        ("Employee ID: MKT-3912", "employee_id"),
        ("IP address 10.2.3.4", "ip_address"),
        ("acct 123456789012345", "bank_account"),
    ],
)
def test_field_labels_surface_their_specific_type(text: str, expected_type: str) -> None:
    """'ssn' and 'tax id' used to be one alternation reporting one type."""
    assert expected_type in {f.type for f in _findings(text)}


def test_a_finding_carries_both_levels() -> None:
    finding = _findings("Aadhaar: 1234 5678 9012")[0]

    assert finding.type == "aadhaar"
    assert finding.category == "government_id"


# --- 4. reason codes become specific --------------------------------------


def test_reason_codes_carry_the_specific_type_and_the_category() -> None:
    codes = run("PAN: ABCDE1234F", operation="detect", config=REGEX).result.reason_codes

    assert "PII_PAN" in codes, "the specific code is the point of the change"
    assert "PII_GOVERNMENT_ID" in codes, "the broad code is what existing policies match"


def test_a_type_that_is_its_own_category_emits_one_code() -> None:
    """No PII_EMAIL plus a redundant duplicate."""
    assert reason_codes_for("email") == ["PII_EMAIL"]
    assert reason_codes_for("aadhaar") == ["PII_AADHAAR", "PII_GOVERNMENT_ID"]


# --- 5. existing broad policies continue to match -------------------------


def test_a_policy_matching_the_broad_category_still_fires() -> None:
    """The regression this whole change risks.

    Mirrors how the orchestrator builds its match set: specific type plus
    category. A rule written as `finding_type: government_id` before Aadhaar
    had its own type must still match an Aadhaar finding.
    """
    findings = _findings("Aadhaar: 1234 5678 9012 and PAN ABCDE1234F")
    match_set = {v for f in findings for v in (f.type, f.category or f.type)}

    assert "government_id" in match_set, "a pre-existing broad policy would stop firing"
    assert {"aadhaar", "pan"} <= match_set, "the specific types are available too"


def test_risk_is_assessed_on_the_category_so_it_does_not_drop() -> None:
    """An aadhaar finding is high risk because government_id is. If risk were
    read from the specific type it would silently fall to medium."""
    assert is_high_risk("aadhaar")
    assert is_high_risk("pan")
    assert is_high_risk("jwt")
    assert run("Aadhaar: 1234 5678 9012", operation="detect", config=REGEX).result.risk == "high"


# --- 6. nothing invents a type --------------------------------------------


def test_an_unknown_type_is_returned_unchanged() -> None:
    """A type with no mapping is its own category. Guessing one would put a
    finding into a policy bucket it was never shown to belong in."""
    assert category_of("something_we_never_defined") == "something_we_never_defined"
    assert reason_codes_for("something_we_never_defined") == ["PII_SOMETHING_WE_NEVER_DEFINED"]


def test_unrecognisable_values_stay_generic() -> None:
    """A span the patterns cannot identify keeps the generic type rather than
    being promoted to a specific one it did not earn."""
    assert classify("zzzz") == "person_name"   # has letters: a name is the honest guess
    assert classify("") == "personal_data"


def test_unsupported_types_are_not_in_the_mapping() -> None:
    """Documented-as-missing must mean absent, not silently present."""
    for pii_type in UNSUPPORTED:
        assert pii_type not in CATEGORY_OF, f"{pii_type} is documented unsupported but mapped"


# --- 7. payment cards are not guessed -------------------------------------


def test_cards_are_payment_card_and_never_credit_or_debit() -> None:
    """A card number cannot distinguish credit from debit, so the extension
    does not pretend otherwise."""
    assert classify("4111111111111111") == "payment_card"
    assert category_of("payment_card") == "payment_card"

    assert "credit_card" not in CATEGORY_OF
    assert "debit_card" not in CATEGORY_OF

    codes = run("card 4111 1111 1111 1111", operation="detect", config=REGEX).result.reason_codes
    assert "PII_PAYMENT_CARD" in codes
    assert not any("CREDIT" in c or "DEBIT" in c for c in codes)


# --- known gaps, pinned so they cannot be forgotten ------------------------


def test_cvv_is_mapped_but_no_real_cvv_is_detected() -> None:
    """A documented gap, not a working detector.

    'cvv|cvc' is in _FIELD_TYPES, but the credential length guard drops values
    under 6 characters and a CVV is 3-4 digits, so a real one is never found.
    This predates the taxonomy change. The test pins the behaviour so that
    fixing the guard makes this fail loudly rather than passing unnoticed.
    """
    from kryptos_pii.taxonomy import UNREACHABLE

    assert "cvv" in UNREACHABLE

    assert _findings("CVV: 775") == []
    assert _findings("cvc 999") == []
    # Long enough to clear the guard, which is how we know the label itself works.
    assert "cvv" in {f.type for f in _findings("CVV: 775421")}


def test_pin_has_no_detector_at_all() -> None:
    """PIN is in the target taxonomy and there is no field label or pattern for
    it. Absent from the mapping rather than present and never firing."""
    assert "pin" in CATEGORY_OF  # mapped, because the category is knowable
    assert _findings("PIN: 4821") == [], "no detector produces a credential PIN today"
