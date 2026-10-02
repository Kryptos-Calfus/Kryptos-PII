"""Shared by training and inference: how text is split into pieces, and the question LAYA answers.

Regex here knows no PII formats. It only proposes pieces that *could* be PII; the
fine-tuned LAYA model decides, in context, whether each piece is personal information.

The proposer has two jobs and only two:
  1. RECALL  - every real PII value must overlap at least one candidate, or LAYA never sees it.
  2. BOUNDARIES - a candidate should not swallow the words around the value. `Contact Sarah`
     as one span means masking it also destroys "Contact", and in training it teaches the
     model that "Contact" is personal.
Deciding whether a candidate is actually PII is LAYA's job, never this module's.
"""

import re

# Words trimmed from the edges of a capitalised run. These are lead-ins and field labels,
# not a PII vocabulary: the list deliberately excludes anything that is also a common given
# name (Grace, Rose, Mark, Victoria, Jordan, May, June, Bill, Art, Dawn, Faith, Hope).
EDGE_WORDS = set("""
A An The This That These Those It Its He She They We You I My Our Your His Her Their Me Us Him Them
Hi Hello Hey Dear Dearest Greetings Namaste Bonjour Hola Ciao Salut Mera Meri Main Aapka
Please Kindly Thanks Thank Cheers Regards Sincerely Best Warm Yours Attn Ref Note Following
Regarding About Attached Enclosed Reach Contact Call Email Mail Phone Mobile Landline Telephone
Customer Client Patient Employee Candidate Applicant Beneficiary Holder Account Accounts
Name Full First Last Middle Username User Login Password Temporary Recovery Token Bearer Key
Subject From To Cc Bcc Sent Date Sender Recipient Team Support Sales Billing Legal Payroll
Insurance Member Policy Group Order Ticket Invoice Reference Transaction Receipt Confirmation
Passport Voter Vehicle Registration Licence License Driver Drivers National Social Security Tax
Medical Record Records Lab Discharge Summary Diagnosis Medication Emergency Relationship
Deliver Delivery Ship Shipping Shipment Courier Transfer Wire Refund Payment Paid Charged
Bank Branch Checking Savings Routing Card Credit Debit Cheque Check Balance Amount Total
Address Home House Flat Apartment Unit Floor Building Office Room Block Sector Street Road
Server Device Build Release Version Sprint Deploy Deployment Error Code Project Product Model
Nationality Blood Age Gender Occupation Designation Department Division Branchwise
Yes No Ok Okay Sorry Also And But Or If So Then When While Here There Now Today Tomorrow
""".lower().split())

# A lowercase word that a cue phrase marks as a name: "yo its arjun,", "hi this is neha".
LOWER_NAME_CUE = re.compile(
    r"(?:^|[,.!?;\n]\s*|\b(?:it'?s|its|this is|i am|i'?m|am|hey|hi|hello|yo|naam|call me|myself)\s+)"
    r"([a-z][a-z'-]{2,}(?:\s+[a-z][a-z'-]{2,})?)"
    r"(?=\s+(?:here|hu|hoon|hai|bol)\b|\s*,|\s*\.|$)",
    re.I,
)

PIECE_RES = [
    # Email / handle first: it must win over the bare-digit and capitalised-run patterns.
    re.compile(r"[^\s,;()<>\[\]]+@[^\s,;()<>\[\]]+"),
    # A number glued to its capitalised run, optionally ending in a year: keeps
    # "742 Evergreen Terrace" and "12 June 2026" as one span rather than splitting off the number.
    re.compile(r"\b\d{1,5}[A-Za-z]?\s+(?:[A-ZÀ-Þ][\w'À-ɏ-]*[ \t]*){1,4}(?:\d{4}\b)?"),
    # Anything with a digit, joining groups split by single spaces/hyphens ("98765 43210").
    re.compile(r"[^\s,;()<>\[\]]*\d[^\s,;()<>\[\]]*(?:[ -]\d[^\s,;()<>\[\]]*)*"),
    # Runs of 1-3 capitalised words (Latin incl. accents).
    re.compile(r"\b[A-ZÀ-Þ][\w'À-ɏ-]*(?:[ \t]+[A-ZÀ-Þ][\w'À-ɏ-]*){0,2}"),
    # The value right after "Key: " or "key is " (usernames, passwords, lowercase handles).
    re.compile(r"(?<=: )[^\s,;]+|(?<=\bis )[^\s,;]+"),
    # Non-Latin script words (Devanagari, Tamil), up to two.
    re.compile(r"[ऀ-ॿ஀-௿]+(?:\s[ऀ-ॿ஀-௿]+)?"),
    # Lowercase names that only a cue phrase reveals: "yo its arjun", "sneha here".
    LOWER_NAME_CUE,
    re.compile(r"\b[a-z]{3,}(?=\s+here\b)"),
]
TRAIL = ".:!?'\",;"
LEAD = "\"'(<[ \t"

CONTEXT_CHARS = 120  # text on each side of a piece that LAYA sees; see set_context_chars()


def set_context_chars(n: int) -> None:
    """Override the context window (experiments in train.py sweep this)."""
    global CONTEXT_CHARS
    CONTEXT_CHARS = int(n)


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    """Shrink a span off surrounding punctuation, possessives and lead-in/label words."""
    while s < e and text[s] in LEAD:
        s += 1
    while e > s and text[e - 1] in TRAIL + " \t":
        e -= 1
    if text[s:e].endswith(("'s", "’s")):
        e -= 2
    # Drop leading/trailing label words, but never everything: a bare "Contact" may still be
    # proposed on its own, and LAYA is free to call it not-PII.
    words = text[s:e].split()
    while len(words) > 1 and words[0].strip(TRAIL).lower() in EDGE_WORDS:
        s += len(words[0]) + (len(text[s:e]) - len(text[s:e].lstrip()))
        while s < e and text[s] in " \t":
            s += 1
        words = text[s:e].split()
    while len(words) > 1 and words[-1].strip(TRAIL).lower() in EDGE_WORDS:
        e -= len(words[-1])
        while e > s and text[e - 1] in " \t":
            e -= 1
        words = text[s:e].split()
    return s, e


def pieces(text: str) -> list[tuple[int, int]]:
    """Non-overlapping candidate spans, earlier patterns first."""
    out: list[tuple[int, int]] = []
    for rx in PIECE_RES:
        for m in rx.finditer(text):
            s, e = m.span(1) if rx.groups else m.span()
            s, e = _trim(text, s, e)
            if e - s < 2 or any(s < b and a < e for a, b in out):
                continue
            out.append((s, e))
    return sorted(out)


def state_for(text: str, s: int, e: int, context_chars: int | None = None) -> dict:
    n = CONTEXT_CHARS if context_chars is None else context_chars
    return {"text": text[max(0, s - n):e + n], "span": text[s:e]}


def label_pieces(text: str, pii_spans) -> list[tuple[int, int, str]]:
    """(start, end, type) for every piece; type is NOT_PII when nothing overlaps.

    Gold spans may be (start, end) or (start, end, type); untyped ones default to "person".
    Binary training derives its label as `type != NOT_PII`.
    """
    return [(s, e, _type_at(s, e, pii_spans)) for s, e in pieces(text)]


def spans_of(text: str, values: list[str]) -> list[tuple[int, int]]:
    """Every case-insensitive occurrence of each labelled value."""
    low, out = text.lower(), []
    for v in values:
        i = low.find(v.lower())
        while i >= 0:
            out.append((i, i + len(v)))
            i = low.find(v.lower(), i + 1)
    return out


def training_pieces(text: str, pii_spans) -> list[tuple[int, int, str]]:
    """Candidates for TRAINING: the regex proposals plus any gold span the regex missed.

    Without this, a PII value the extractor fails to propose never becomes a training
    example, so the model can never learn it - the regex's blind spots become the model's.
    Inference still uses pieces() alone; this only widens what the model is taught on.
    """
    out = label_pieces(text, pii_spans)
    for span in pii_spans:
        a, b = _trim(text, span[0], span[1])
        if b - a >= 2 and not any(a < e and s < b for s, e, _ in out):
            out.append((a, b, span[2] if len(span) > 2 else "person"))
    return sorted(out)


# ---------------------------------------------------------------- PII types
# The same architecture as the yes/no question: regex proposes a span, LAYA decides what it
# is. "not_pii" is one of the options, so the type model also does the detection - it does
# not need the binary model in front of it.

NOT_PII = "not_pii"
TYPES = ["person", "email", "phone", "address", "id", "financial", "secret", "date", "network", NOT_PII]

# Option text shares a fixed token budget (head_max_len=192), so these stay terse on purpose.
TYPE_QUESTION = {
    "kind": {
        "type": "choice",
        "instructions": "In this text, what kind of information is the value of 'span'?",
        "criteria": {
            "person": "a person's name",
            "email": "an email address",
            "phone": "a personal phone number",
            "address": "a home address, street, flat or postal code",
            "id": "a government or membership ID: Aadhaar, PAN, SSN, passport, licence, MRN, employee or policy number",
            "financial": "a card number, bank account, UPI ID or CVV",
            "secret": "a password, API key, token, OTP or username",
            "date": "a person's date of birth",
            "network": "an IP or MAC address",
            NOT_PII: "not personal: a place, company, product, ticket, order, amount, time or ordinary word",
        },
    }
}

# Shapes used ONLY to type the hand-labelled real training texts, whose labels list values
# without a category. Never used at inference: the model assigns the type there.
_TYPE_SHAPES = [
    ("email", re.compile(r"^[^@\s]+@[^@\s]+$")),
    ("network", re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$|^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$")),
    ("date", re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}-\d{2}-\d{2}$|^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$")),
    ("secret", re.compile(r"^(?:sk-|ghp_|AKIA|eyJ)|[!@#$%^&*]")),
    ("id", re.compile(r"^[A-Z]{5}\d{4}[A-Z]$|^\d{3}-\d{2}-\d{4}$|^(?:POL|MBR|EMP|INS|DL)[-\d]|^[A-Z]\d{7,8}$")),
    ("financial", re.compile(r"^(?:\d{4}[\s-]?){3,4}\d{0,4}$|^\d{9,18}$")),
    ("phone", re.compile(r"^\+?[\d\s()-]{8,}$")),
    ("address", re.compile(r"^\d{1,5}[A-Za-z]?\s+[A-Z]|^(?:Flat|House|H\.?No)\b", re.I)),
]


def guess_type(value: str) -> str:
    """Best-effort category for a hand-labelled value (training data only)."""
    v = value.strip()
    for label, rx in _TYPE_SHAPES:
        if rx.search(v):
            return label
    if re.search(r"^\d{5,6}$", v):
        return "address"          # bare PIN / ZIP code
    if re.search(r"[A-Za-z]", v):
        return "person"
    return "id"


def typed_spans_of(text: str, values: list[str]) -> list[tuple[int, int, str]]:
    return [(a, b, guess_type(text[a:b])) for a, b in spans_of(text, values)]


def _type_at(s: int, e: int, pii_spans) -> str:
    for span in pii_spans:
        a, b = span[0], span[1]
        if s < b and a < e:
            return span[2] if len(span) > 2 else "person"
    return NOT_PII


QUESTION = {
    "pii": {
        "type": "noul",
        "instructions": "Is the value of 'span' personal information about a specific individual in this text "
                        "(name, contact detail, ID number, account, password, or birth date)?",
        "criteria": {
            "false": "span is not personal: a place, company, product, ticket, amount, time, or ordinary word.",
            "true": "span identifies or belongs to a specific person.",
        },
    }
}
