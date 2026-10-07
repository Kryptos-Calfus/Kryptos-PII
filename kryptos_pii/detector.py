"""Detection: regex proposes, the fine-tuned model decides.

This is the same two-stage design the repository was built around, and it is
kept exactly as the training pipeline assumed it:

    1. ``finetune.common.pieces`` proposes candidate spans. It knows no PII
       formats -- its only jobs are recall and span boundaries.
    2. The LAYA checkpoint in ``finetune/laya-pii`` reads each candidate in
       context and answers whether it is personal information. The checkpoint
       reports val F1 0.988 at threshold 0.5 with a 120-character context
       window, so those are the defaults here; changing the context without
       retraining invalidates the measurement.

Two rules follow from CLAUDE.md and are enforced rather than documented:

* No LLM call. Detection is the local checkpoint, never a model API (section 11).
* The model is the decision, not a suggestion. If the checkpoint is missing, the
  hybrid detector fails loudly instead of quietly degrading to regex -- a PII
  filter that silently stops detecting is worse than one that is plainly down.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from kryptos_pii.candidates import QUESTION, pieces, set_context_chars, state_for
from kryptos_pii.classify import classify, luhn_valid
from kryptos_pii.model import model_dir
from kryptos_pii.taxonomy import HIGH_RISK_CATEGORIES, category_of

# Resolved once per process: an explicit KRYPTOS_PII_MODEL_DIR, the checkpoint
# in a repository checkout, or the one 'kryptos-pii-model download' fetched.
# See kryptos_pii.model for the search order.
DEFAULT_MODEL_DIR = model_dir()
DEFAULT_THRESHOLD = 0.5
DEFAULT_CONTEXT_CHARS = 120

# How many candidate spans go to the model at once. Peak memory scales with this
# rather than with document length, so it is the knob that decides whether a
# large document is slow or fatal. 64 keeps a batch's inputs well under a
# megabyte at the default context window while still amortising per-call
# overhead; raise it on a machine with headroom.
MAX_BATCH_SPANS = max(1, int(os.environ.get("KRYPTOS_PII_MAX_BATCH_SPANS", "64")))

# The checkpoint answers yes/no, so it cannot name a category. ``classify`` is
# that step: a deterministic read of the matched text, used to label the finding
# and pick a redaction placeholder. It never decides whether something is
# masked -- that is always the model's answer.

# Values the model is not consulted about, because their shape is already proof.
# A live key or card number must never depend on a neural decision.
_CERTAIN = [
    ("secret", re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")),
    ("email", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    # Only when Luhn passes, which is checked below -- the pattern alone would
    # also match any other 13-19 digit run.
    ("payment_card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
]

# "Key: value" fields, where the label is the evidence.
#
# This stage exists because the classifier is measurably weak exactly here: a
# bare "123456789012345" after the word "acct" scores 0.001, because nothing in
# the digits themselves is personal and the model was not taught to lean on the
# label. The repository's own demo already solved this with field regex, and
# CLAUDE.md section 11 puts deterministic patterns ahead of the model for
# precisely this reason. The key stays visible; only the value is a finding.
# The label already names the kind of identifier, which is the only reason
# these can be told apart at all. The value is now that specific type rather
# than the broad category it used to collapse into; taxonomy.category_of maps
# it back, so nothing that matched before stops matching.
_FIELD_TYPES = {
    r"a/?c|acct|account(?: number| no\.?)?|bank account": "bank_account",
    r"ifsc|routing(?: number)?|sort code": "ifsc",
    r"upi(?: id)?": "upi_id",
    r"cvv|cvc": "cvv",
    r"otp|verification code|2fa": "otp",
    r"password|passcode|pwd": "password",
    r"api[ _-]?key|secret|token": "api_key",
    # A user name is a credential-adjacent value, but USERNAME as a type is out
    # of scope for this change, so this one keeps the broad category.
    r"(?:login )?user ?name|user id": "secret",
    r"aadhaar|aadhar": "aadhaar",
    r"pan(?: number)?": "pan",
    r"passport(?: number| no\.?)?": "passport",
    r"driver'?s? licen[cs]e(?: number| no\.?)?|dl no\.?": "driver_license",
    r"mrn|medical record number": "medical_record",
    r"(?:insurance )?member id|policy(?: number| no\.?)?": "health_insurance",
    r"employee id|emp id": "employee_id",
    # Split from one entry: 'ssn' and 'tax id' were a single alternation
    # reporting one type, and they name two different identifiers.
    r"ssn": "ssn",
    r"(?:tax|national) id(?: number)?": "tax_id",
    r"ip(?: address)?": "ip_address",
    r"mac(?: address)?": "mac_address",
    r"dob|date of birth": "date_of_birth",
    r"phone|mobile|contact(?: number)?|cell": "phone",
}

# The value after the label: either quoted/delimited, or a run up to punctuation.
_FIELD_RES = [
    (
        label,
        re.compile(
            # The value is ONE token, plus any further groups that are pure
            # digits -- that is what joins "1234 5678 9012" without letting the
            # span run on into the rest of the sentence.
            rf"\b(?:{key})\b\s*(?::|=|is|-)?\s+(?P<value>[^\s,;]+(?:[ -]\d{{2,}}){{0,3}})",
            re.IGNORECASE,
        ),
    )
    for key, label in _FIELD_TYPES.items()
]

# Kept as a name other modules import. These are categories, so a specific
# type inherits its category's risk and no risk assessment changes.
HIGH_RISK_TYPES = HIGH_RISK_CATEGORIES


@dataclass(frozen=True)
class Detection:
    start: int
    end: int
    type: str
    confidence: float
    detector: str


class ModelUnavailable(RuntimeError):
    """The checkpoint this detector was configured to use is not loadable."""


class _Checkpoint:
    """Loads the checkpoint once per process and scores candidates with it.

    Loading is ~850 MB and several seconds, so it happens behind a lock on first
    use and is warmed at startup by the service.
    """

    def __init__(self, model_dir: Path, context_chars: int) -> None:
        self._dir = Path(model_dir)
        self._context_chars = context_chars
        self._agent = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self._dir / "model.safetensors").exists()

    def load(self):  # noqa: ANN201 - a laya agent; the package is optional at import time
        if self._agent is not None:
            return self._agent
        with self._lock:
            if self._agent is not None:
                return self._agent
            if not self.available:
                raise ModelUnavailable(
                    f"No fine-tuned checkpoint at {self._dir}.\n"
                    "Fetch the published one with: kryptos-pii-model download\n"
                    "Or train your own with finetune/train.py, or run with "
                    "detector_mode=regex, which has materially lower recall."
                )
            import laya

            set_context_chars(self._context_chars)
            agent = laya.load(str(self._dir))
            # One throwaway call so the first real request is not paying for
            # lazy weight materialisation.
            agent.predict(state_for("warmup Rahul", 7, 12), QUESTION)
            self._agent = agent
            return agent

    def score(self, text: str, spans: list[tuple[int, int]]) -> list[float]:
        """Score every candidate span, in bounded batches.

        Candidate count grows with document length -- roughly 60 per KB of
        prose -- so a 500 KB document proposes around 28,000 spans. Handing all
        of them to the model in one call made peak memory a function of input
        size, which is how a large document turns into an OOM rather than a slow
        request.

        Batching is lossless here, and that is a property of the design rather
        than a hope: ``state_for`` already gives each span its own 120-character
        window, so a span's input does not depend on which other spans travel
        with it. Where the batch boundary falls cannot change a score.

        This is why the fix is batching and not chunking the text. Splitting the
        document would cut context at the chunk edges and change what the model
        sees near them; splitting the candidate list changes nothing.
        """
        if not spans:
            return []
        agent = self.load()
        out: list[float] = []
        for start in range(0, len(spans), MAX_BATCH_SPANS):
            window = spans[start : start + MAX_BATCH_SPANS]
            results = agent.predict_batch([state_for(text, s, e) for s, e in window], QUESTION)
            out += [float(r["answers"]["pii"]["noul"]) for r in results]
        return out


_checkpoints: dict[tuple[str, int], _Checkpoint] = {}
_checkpoints_lock = threading.Lock()


def checkpoint_for(model_dir: Path | str = DEFAULT_MODEL_DIR, context_chars: int = DEFAULT_CONTEXT_CHARS) -> _Checkpoint:
    key = (str(model_dir), context_chars)
    with _checkpoints_lock:
        if key not in _checkpoints:
            _checkpoints[key] = _Checkpoint(Path(model_dir), context_chars)
        return _checkpoints[key]


def _labelled_spans(text: str) -> list[Detection]:
    """Values a field label already identifies. Deterministic, no model call."""
    out: list[Detection] = []
    for label, rx in _FIELD_RES:
        for match in rx.finditer(text):
            start, end = match.span("value")
            # Sentence punctuation and closing delimiters are not part of the
            # value. Brackets matter as much as full stops: behind the gateway
            # this text is often JSON tool arguments, and a span that swallows
            # the closing `"}` turns a valid call into an invalid one.
            while end > start and text[end - 1] in ".:!?'\")]}>":
                end -= 1
            value = text[start:end]
            # Without a separator the label may just be prose ("account is
            # closed", "contact 5 people"), so the value has to look like one:
            # four or more digits for anything numeric, and a secret has to be
            # non-trivial. This is what keeps the stage precise enough to run
            # ahead of the model.
            digits = sum(c.isdigit() for c in value)
            if category_of(label) == "secret":
                if len(value) < 6 or value.lower() in ("closed", "expired", "reset"):
                    continue
            elif digits < 4:
                continue
            out.append(
                Detection(start=start, end=end, type=label, confidence=1.0, detector="field-regex")
            )
    return out


def _certain_spans(text: str) -> list[Detection]:
    out: list[Detection] = []
    for label, rx in _CERTAIN:
        for match in rx.finditer(text):
            if label == "payment_card" and not luhn_valid(match.group()):
                continue
            out.append(
                Detection(
                    start=match.start(),
                    end=match.end(),
                    type=label,
                    confidence=1.0,
                    detector="regex",
                )
            )
    return out


def detect(
    text: str,
    *,
    detector_mode: str = "hybrid",
    threshold: float = DEFAULT_THRESHOLD,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[Detection]:
    """Spans of ``text`` that are personal information, left to right.

    ``hybrid`` is regex candidates judged by the checkpoint. ``regex`` skips the
    model and keeps only the shapes that are self-evident -- far lower recall,
    offered for CPU-only deployments that cannot host the checkpoint, and named
    in the result's metadata so nobody mistakes it for the full detector.
    """
    if not text:
        return []

    found = _certain_spans(text) + _labelled_spans(text)

    if detector_mode != "regex":
        candidates = [
            (s, e)
            for s, e in pieces(text)
            # A span already settled by shape does not need a model opinion.
            if not any(s < d.end and d.start < e for d in found)
        ]
        scores = checkpoint_for(model_dir, context_chars).score(text, candidates)
        for (start, end), probability in zip(candidates, scores):
            if probability >= threshold:
                found.append(
                    Detection(
                        start=start,
                        end=end,
                        type=classify(text[start:end]),
                        confidence=round(probability, 4),
                        detector="laya-pii",
                    )
                )

    # Overlaps would corrupt the rewrite, which walks the spans in order.
    ordered = sorted(found, key=lambda d: (d.start, -(d.end - d.start)))
    kept: list[Detection] = []
    for detection in ordered:
        if kept and detection.start < kept[-1].end:
            continue
        kept.append(detection)
    return kept
