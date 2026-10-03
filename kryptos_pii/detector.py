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

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from kryptos_pii.candidates import QUESTION, pieces, set_context_chars, state_for
from kryptos_pii.classify import classify, luhn_valid
from kryptos_pii.model import model_dir

# Resolved once per process: an explicit KRYPTOS_PII_MODEL_DIR, the checkpoint
# in a repository checkout, or the one 'kryptos-pii-model download' fetched.
# See kryptos_pii.model for the search order.
DEFAULT_MODEL_DIR = model_dir()
DEFAULT_THRESHOLD = 0.5
DEFAULT_CONTEXT_CHARS = 120

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
_FIELD_TYPES = {
    r"a/?c|acct|account(?: number| no\.?)?|bank account": "financial",
    r"ifsc|routing(?: number)?|sort code": "financial",
    r"upi(?: id)?": "financial",
    r"cvv|cvc": "secret",
    r"otp|verification code|2fa": "secret",
    r"password|passcode|pwd": "secret",
    r"api[ _-]?key|secret|token": "secret",
    r"(?:login )?user ?name|user id": "secret",
    r"aadhaar|aadhar": "government_id",
    r"pan(?: number)?": "government_id",
    r"passport(?: number| no\.?)?": "government_id",
    r"driver'?s? licen[cs]e(?: number| no\.?)?|dl no\.?": "government_id",
    r"mrn|medical record number": "government_id",
    r"(?:insurance )?member id|policy(?: number| no\.?)?": "government_id",
    r"employee id|emp id": "government_id",
    r"(?:tax|national) id(?: number)?|ssn": "government_id",
    r"ip(?: address)?": "network_address",
    r"mac(?: address)?": "network_address",
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

HIGH_RISK_TYPES = frozenset({"secret", "payment_card", "financial", "government_id"})


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
        if not spans:
            return []
        agent = self.load()
        results = agent.predict_batch([state_for(text, s, e) for s, e in spans], QUESTION)
        return [float(r["answers"]["pii"]["noul"]) for r in results]


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
            # Sentence punctuation is not part of the value.
            while end > start and text[end - 1] in ".:!?'\"":
                end -= 1
            value = text[start:end]
            # Without a separator the label may just be prose ("account is
            # closed", "contact 5 people"), so the value has to look like one:
            # four or more digits for anything numeric, and a secret has to be
            # non-trivial. This is what keeps the stage precise enough to run
            # ahead of the model.
            digits = sum(c.isdigit() for c in value)
            if label == "secret":
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
