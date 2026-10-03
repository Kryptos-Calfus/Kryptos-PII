"""What kind of personal information a detected span is.

The fine-tuned checkpoint answers one question -- is this span personal -- and
deliberately not a second one, so the category has to come from the shape of the
matched text. This module is that step, and it is kept separate from
``finetune.common.guess_type`` on purpose: that function orders its patterns for
labelling training data, where a ten-digit run is far more often an account
number, and reusing it at inference mislabels Indian mobile numbers as financial
data. Reason codes drive policy, so the category has to be right here even
though the mask/no-mask decision does not depend on it.

Order matters and runs most-specific first. Where a shape is genuinely ambiguous
a checksum settles it rather than a guess (CLAUDE.md section 11): a 16-digit run
that passes Luhn is a card, one that fails is an account number.
"""

from __future__ import annotations

import re

PERSON = "person_name"
GENERIC = "personal_data"

# The three arms of the old single _SECRET pattern, kept apart so the kind of
# credential survives into the finding. The alternation and the ordering are
# unchanged; only the result is more specific.
_API_KEY = re.compile(r"^(?:sk-[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")
_JWT = re.compile(r"^eyJ[A-Za-z0-9_-]{6,}\.")
# A password is recognised by shape alone: letters, digits and a symbol with no
# whitespace. It is the weakest of the three and stays last.
_PASSWORD = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*_])[^\s]{8,}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_MAC = re.compile(r"^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$")
_DATE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}-\d{2}-\d{2}$|^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$")
# Government and membership identifiers with a fixed, recognisable layout.
# One pattern per kind: the layouts were always distinct, they were just being
# reported under one name.
_PAN = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")              # India PAN
_SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")              # US SSN
_PASSPORT = re.compile(r"^[A-Z]\d{7,8}$")
# Prefixed membership and record numbers. The prefix names the kind, which is
# why these can be told apart at all.
_PREFIXED_ID = re.compile(r"^(?P<prefix>POL|MBR|EMP|INS|DL|MRN)[-\s]?\w+", re.IGNORECASE)
_PREFIX_TYPE = {
    "DL": "driver_license",
    "MRN": "medical_record",
    "EMP": "employee_id",
    "POL": "health_insurance",
    "INS": "health_insurance",
    "MBR": "health_insurance",
}
_AADHAAR = re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$")
_UPI = re.compile(r"^[\w.-]{3,}@(?:ok\w+|paytm|ybl|upi|axl|ibl)$", re.I)
_ADDRESS = re.compile(r"^\d{1,5}[A-Za-z]?\s+[A-Z]|^(?:Flat|House|H\.?\s?No|Plot|Door)\b", re.I)
# A phone number may carry a country code, spaces, hyphens or brackets.
_PHONE_SHAPE = re.compile(r"^\+?[\d][\d\s()\-]{7,18}$")

# Every type classify() can return, specific and broad alike. The single
# source of truth for which of these is a category lives in taxonomy.py.
from kryptos_pii.taxonomy import ALL_TYPES  # noqa: E402,F401  (re-exported)


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def luhn_valid(number: str) -> bool:
    """The card checksum. A 16-digit run that fails this is not a card."""
    digits = [int(c) for c in _digits(number)]
    if len(digits) < 12:
        return False
    total, parity = 0, len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _numeric_type(value: str) -> str | None:
    """Classify a run that is only digits and separators.

    This is where the training-time heuristic went wrong, so the cases are
    spelled out rather than ordered by luck.
    """
    digits = _digits(value)
    length = len(digits)
    if not digits:
        return None

    # An explicit country code means a phone, whatever the length.
    if value.strip().startswith("+") and 10 <= length <= 15:
        return "phone"
    if _AADHAAR.match(value.strip()):
        return "aadhaar"
    if length == 10:
        # Indian mobile numbers start 6-9; a 10-digit run starting 0-5 is far
        # more likely an account or reference number.
        return "phone" if digits[0] in "6789" else "financial"
    if 11 <= length <= 13 and digits.startswith(("91", "1", "44", "61")):
        return "phone"
    if 13 <= length <= 19:
        # Luhn is what separates a real card from a long account or reference
        # number of the same length. A card carries more risk, so it is worth
        # its own type and its own reason code.
        return "payment_card" if luhn_valid(digits) else "financial"
    if length in (5, 6):
        return "postal_code"  # PIN / ZIP; category is still 'address'
    if 9 <= length <= 18:
        return "financial"
    return None


def classify(value: str) -> str:
    """The specific type of one detected span.

    Returns the narrowest type the evidence supports -- ``aadhaar`` rather than
    ``government_id`` -- and falls back to the broad category when the shape
    says what kind of thing it is but not which kind. Use
    ``taxonomy.category_of`` to get the broad category for any result.

    The order of the checks is unchanged. Only the names returned are narrower.
    """
    text = value.strip()
    if not text:
        return GENERIC

    if _EMAIL.match(text):
        return "email"
    if _UPI.match(text):
        return "upi_id"
    if _IPV4.match(text):
        return "ip_address"
    if _MAC.match(text):
        return "mac_address"
    if _API_KEY.match(text):
        return "api_key"
    if _JWT.match(text):
        return "jwt"
    if _PASSWORD.match(text):
        return "password"
    if _DATE.match(text):
        return "date_of_birth"
    if _PAN.match(text):
        return "pan"
    if _SSN.match(text):
        return "ssn"
    if _PASSPORT.match(text):
        return "passport"
    prefixed = _PREFIXED_ID.match(text)
    if prefixed:
        # The prefix is the evidence. Anything else keeps the broad category.
        return _PREFIX_TYPE.get(prefixed.group("prefix").upper(), "government_id")

    if _PHONE_SHAPE.match(text) or re.fullmatch(r"[\d\s()\-]+", text):
        numeric = _numeric_type(text)
        if numeric:
            return numeric

    if _ADDRESS.match(text):
        return "address"
    if re.search(r"[A-Za-zÀ-ÿऀ-ॿ஀-௿]", text):
        return PERSON
    return GENERIC
