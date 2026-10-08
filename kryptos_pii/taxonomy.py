"""Specific PII types, and the broad category each one belongs to.

A finding now carries both:

    {"type": "aadhaar", "category": "government_id", "start": 42, "end": 56}

The detector has always told an Aadhaar number from a PAN from an SSN -- they
are separate patterns -- and then threw the distinction away by reporting all
three as ``government_id``. This module is where that information stops being
discarded.

**The category is the type this extension used to report, unchanged.** That is
deliberate and it is the whole backwards-compatibility story: a policy written
against ``finding_type: government_id`` keeps matching an Aadhaar number,
because the finding still carries ``government_id`` as its category, and the
orchestrator matches on both. Granularity is added beside the old value, never
on top of it. In a security product the alternative -- a policy that quietly
stops matching after an upgrade -- is a failure mode, not a migration cost.

Every type maps to a category, and a category maps to itself, so
``category_of`` is total and never invents anything: a type nothing recognises
comes back unchanged rather than being forced into a bucket it did not earn.
"""

from __future__ import annotations

# --- broad categories: exactly the values this extension reported before ---

PERSON_NAME = "person_name"
EMAIL = "email"
PHONE = "phone"
ADDRESS = "address"
DATE_OF_BIRTH = "date_of_birth"
GOVERNMENT_ID = "government_id"
FINANCIAL = "financial"
PAYMENT_CARD = "payment_card"
SECRET = "secret"
NETWORK_ADDRESS = "network_address"
GENERIC = "personal_data"

CATEGORIES = frozenset(
    {
        PERSON_NAME, EMAIL, PHONE, ADDRESS, DATE_OF_BIRTH, GOVERNMENT_ID,
        FINANCIAL, PAYMENT_CARD, SECRET, NETWORK_ADDRESS, GENERIC,
    }
)

# --- specific type -> broad category --------------------------------------
#
# Only types an existing detector can actually produce appear here. Nothing is
# listed because the taxonomy would look tidier with it: a type with no detector
# behind it is a promise the extension cannot keep. See UNSUPPORTED below for
# what was deliberately left out and why.

CATEGORY_OF: dict[str, str] = {
    # Government and institutional identifiers. Each has its own pattern in
    # classify(), or its own field label in the detector, or both.
    "aadhaar": GOVERNMENT_ID,
    "pan": GOVERNMENT_ID,
    "ssn": GOVERNMENT_ID,
    "passport": GOVERNMENT_ID,
    "driver_license": GOVERNMENT_ID,
    "tax_id": GOVERNMENT_ID,
    "medical_record": GOVERNMENT_ID,
    "health_insurance": GOVERNMENT_ID,
    "employee_id": GOVERNMENT_ID,
    # Money. payment_card is its own category rather than a type under
    # financial, because that is what it was before and policies may match it.
    "upi_id": FINANCIAL,
    "ifsc": FINANCIAL,
    "bank_account": FINANCIAL,
    # Credentials.
    "password": SECRET,
    "api_key": SECRET,
    "jwt": SECRET,
    "otp": SECRET,
    "pin": SECRET,
    "cvv": SECRET,
    # Network.
    "ip_address": NETWORK_ADDRESS,
    "mac_address": NETWORK_ADDRESS,
    # Contact.
    "postal_code": ADDRESS,
}

# A category is also a usable type: it is what a detector reports when it knows
# the kind of thing but not which kind. 'government_id' with no further
# evidence stays 'government_id' rather than being guessed into 'pan'.
CATEGORY_OF.update({category: category for category in CATEGORIES})

ALL_TYPES = frozenset(CATEGORY_OF)

# Categories whose exposure is treated as high risk. Unchanged from before:
# these are categories, so a specific type inherits the risk of its category
# and no risk assessment shifts because of this change.
HIGH_RISK_CATEGORIES = frozenset({SECRET, PAYMENT_CARD, FINANCIAL, GOVERNMENT_ID})

# Types named in the target taxonomy that are NOT implemented, and why. Kept in
# code rather than in a document so it stays honest: anything added here must
# either gain a detector or stay out of the mapping above.
UNSUPPORTED: dict[str, str] = {
    "voter_id": "no pattern exists; India EPIC format not implemented",
    "iban": "no pattern or mod-97 checksum implemented",
    "swift": "no pattern implemented",
    "vin": "no pattern or checksum implemented",
    "license_plate": "no pattern implemented",
    "gps_coordinates": "no pattern implemented",
    "device_id": "indistinguishable from session_id/cookie_id without a label",
    "session_id": "indistinguishable from device_id/cookie_id without a label",
    "cookie_id": "indistinguishable from device_id/session_id without a label",
    "access_token": "indistinguishable from refresh_token without a label",
    "refresh_token": "indistinguishable from access_token without a label",
    "private_key": "no pattern implemented; -----BEGIN blocks are not matched",
    "student_id": "no pattern or field label exists",
    "patient_id": "no pattern; 'mrn' maps to medical_record instead",
    "professional_license": "no pattern or field label exists",
    "credit_card": "a card number cannot be told from a debit card; use payment_card",
    "debit_card": "a card number cannot be told from a credit card; use payment_card",
    "username": "the 'user name' field label exists but the type is out of scope for now",
}


# Mapped and correct, but no input the detector accepts can currently produce
# them. Recorded rather than quietly left in the mapping, so the gap is a known
# fact instead of a surprise in production.
UNREACHABLE: dict[str, str] = {
    "cvv": (
        "the 'cvv|cvc' field label exists, but the credential length guard in "
        "detector._labelled_spans drops values shorter than 6 characters and a "
        "CVV is 3-4 digits. Real CVVs are therefore never detected. Predates "
        "this change; fixing it means a length rule per type, not per category."
    ),
}


def category_of(pii_type: str) -> str:
    """The broad category for a type.

    Total by construction. An unrecognised type is returned unchanged rather
    than mapped to a default, because inventing a category for something the
    detector did not identify would put a finding into a policy bucket it was
    never shown to belong in.
    """
    return CATEGORY_OF.get(pii_type, pii_type)


def reason_codes_for(pii_type: str, category: str | None = None) -> list[str]:
    """The reason codes one finding contributes: the specific type, and the
    category when it differs.

    Both are emitted so a policy can match ``PII_AADHAAR`` for precision or
    ``PII_GOVERNMENT_ID`` for breadth, and an existing policy written against
    the broad code keeps firing.

    ``category`` is for findings whose category this module cannot know: a
    customer's own pattern, declared as belonging to one of the categories
    above. A custom type named ``claim_number`` and declared ``government_id``
    emits ``PII_CLAIM_NUMBER`` and ``PII_GOVERNMENT_ID``, so an existing policy
    rule written against the category covers it from the moment it is added.
    """
    category = category or category_of(pii_type)
    codes = [f"PII_{pii_type.upper()}"]
    if category != pii_type:
        codes.append(f"PII_{category.upper()}")
    return codes


def is_high_risk(pii_type: str) -> bool:
    return category_of(pii_type) in HIGH_RISK_CATEGORIES


__all__ = [
    "ALL_TYPES",
    "CATEGORIES",
    "CATEGORY_OF",
    "HIGH_RISK_CATEGORIES",
    "UNREACHABLE",
    "UNSUPPORTED",
    "category_of",
    "is_high_risk",
    "reason_codes_for",
]
