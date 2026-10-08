"""Customer-supplied regular expressions, as a deterministic detector stage.

Every deployment has identifiers the extension cannot know about: an internal
employee badge, a claim number, a customer reference with a company prefix. This
is where a customer teaches the detector about them without forking it or
waiting for a release.

Three properties make that safe enough to accept from configuration:

* **Deterministic.** A custom pattern is a shape, so it is treated exactly like
  the built-in shape stage (``_certain_spans``): matched, never sent to the
  model, always the same answer.
* **Bounded.** A pattern is compiled and checked before it is ever run, and
  every run carries a time budget. A regular expression is a program, and a
  configuration field that accepts a program accepts a way to hang the service;
  :data:`MATCH_TIMEOUT_SECONDS` is what makes that a 422 instead of an outage.
* **Named, not categorised by guesswork.** A custom type does not inherit a
  category it did not earn. It belongs to ``personal_data`` unless the customer
  says which of the taxonomy's categories it is, and that choice is what decides
  the risk level and the reason codes.

Nothing here knows what a finding means. It knows where one is.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# A regular expression can take time exponential in the length of its input.
# `regex` is the one engine in reach that can be told to stop, so matching goes
# through it and the dependency is declared rather than hoped for.
import regex

from kryptos_pii.taxonomy import ALL_TYPES, CATEGORIES, GENERIC

# Budgets. Each one turns an unbounded cost into a refusal.
#
# A pattern count and a pattern length cap the work a single configuration can
# ask for. The per-pattern timeout caps what one pathological pattern can do to
# one request. The match cap stops a pattern that matches almost everything from
# turning a document into a million findings.
MAX_PATTERNS = 24
MAX_PATTERN_CHARS = 400
MATCH_TIMEOUT_SECONDS = 0.25
TOTAL_TIMEOUT_SECONDS = 1.0
MAX_MATCHES_PER_PATTERN = 512

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")

# A quantified group that itself contains a quantifier: (a+)+, (\d*\s*)+. This
# is the shape behind most catastrophic backtracking, and it is cheap to refuse
# at configuration time rather than discovering it under load. The timeout is
# the real defence; this is the one that produces an error a customer can act
# on, at the moment they paste the pattern in.
_NESTED_QUANTIFIER = re.compile(r"\([^()]*[*+}][^()]*\)\s*[*+{]")

# The span a match contributes. A pattern may name a group `value`, which is how
# a customer matches on a label but reports only what follows it:
#
#     (?i)badge\s*#?\s*(?P<value>[A-Z]{2}-\d{6})
#
# The label stays readable; only the identifier is a finding.
VALUE_GROUP = "value"


class CustomPatternError(ValueError):
    """A custom pattern was rejected, or could not be run inside its budget.

    A subclass of ``ValueError`` on purpose: the hosted service already turns a
    ValueError from the engine into a 422 naming the problem, which is what a
    customer needs here. A pattern the platform will not run must fail loudly --
    silently dropping it would leave a customer believing their identifiers were
    being detected.
    """


@dataclass(frozen=True)
class CustomPattern:
    name: str
    category: str
    source: str
    confidence: float
    compiled: Any  # regex.Pattern; untyped because the module ships no stubs

    @property
    def uses_value_group(self) -> bool:
        return VALUE_GROUP in self.compiled.groupindex


@dataclass(frozen=True)
class CustomMatch:
    """One span a custom pattern claimed.

    Deliberately not a ``Detection``: this module is imported by the detector,
    and handing back the detector's own type would be a circular import for no
    gain. The detector converts.
    """

    start: int
    end: int
    type: str
    category: str
    confidence: float


def _spec_error(index: int, problem: str) -> CustomPatternError:
    return CustomPatternError(f"custom_patterns[{index}]: {problem}")


def normalise_patterns(specs: Any) -> list[dict[str, Any]]:
    """Check the shape of the configuration and fill in its defaults.

    Separate from compilation so the same validation runs whether a pattern is
    about to be used or is merely being saved. Returns a canonical list, which
    is also what makes the compilation cache key stable.
    """
    if specs in (None, ""):
        return []
    if not isinstance(specs, list):
        raise CustomPatternError("custom_patterns must be a list of pattern objects")
    if len(specs) > MAX_PATTERNS:
        raise CustomPatternError(
            f"custom_patterns accepts at most {MAX_PATTERNS} patterns; {len(specs)} were given"
        )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise _spec_error(index, "each pattern must be an object with 'name' and 'pattern'")

        unknown = sorted(set(spec) - {"name", "pattern", "category", "ignore_case", "confidence"})
        if unknown:
            raise _spec_error(index, f"unknown keys {unknown}")

        name = spec.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise _spec_error(
                index,
                "'name' must be lowercase letters, digits and underscores, 2-40 characters "
                "(it becomes the finding type and the PII_<NAME> reason code)",
            )
        if name in ALL_TYPES:
            # Redefining 'email' would emit PII_EMAIL for something the built-in
            # detector never found, and a policy written against that code would
            # fire on a shape nobody reviewed. Names stay distinct from the
            # taxonomy so a reason code means one thing.
            raise _spec_error(index, f"'{name}' is a built-in type; choose a different name")
        if name in seen:
            raise _spec_error(index, f"duplicate pattern name '{name}'")
        seen.add(name)

        pattern = spec.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise _spec_error(index, "'pattern' must be a non-empty regular expression")
        if len(pattern) > MAX_PATTERN_CHARS:
            raise _spec_error(
                index, f"'pattern' is {len(pattern)} characters; the limit is {MAX_PATTERN_CHARS}"
            )
        if _NESTED_QUANTIFIER.search(pattern):
            raise _spec_error(
                index,
                "'pattern' nests a quantifier inside a quantified group, which backtracks "
                "catastrophically on some inputs. Rewrite it without the nesting",
            )

        category = spec.get("category", GENERIC)
        if category not in CATEGORIES:
            raise _spec_error(
                index,
                f"'category' must be one of {sorted(CATEGORIES)}; it decides the risk level "
                "and the broad reason code this finding carries",
            )

        confidence = spec.get("confidence", 1.0)
        if not isinstance(confidence, int | float) or isinstance(confidence, bool):
            raise _spec_error(index, "'confidence' must be a number between 0 and 1")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise _spec_error(index, "'confidence' must be between 0 and 1")

        ignore_case = spec.get("ignore_case", False)
        if not isinstance(ignore_case, bool):
            raise _spec_error(index, "'ignore_case' must be true or false")

        out.append(
            {
                "name": name,
                "pattern": pattern,
                "category": category,
                "ignore_case": ignore_case,
                "confidence": confidence,
            }
        )
    return out


def _compile(specs: list[dict[str, Any]]) -> tuple[CustomPattern, ...]:
    compiled: list[CustomPattern] = []
    for index, spec in enumerate(specs):
        flags = regex.IGNORECASE if spec["ignore_case"] else 0
        try:
            expression = regex.compile(spec["pattern"], flags)
        except regex.error as exc:
            raise _spec_error(index, f"'pattern' is not a valid regular expression: {exc}") from exc
        # A pattern that matches the empty string matches at every offset, which
        # is a finding per character and a rewrite that destroys the text.
        if expression.match(""):
            raise _spec_error(index, "'pattern' matches the empty string, so it matches everywhere")
        compiled.append(
            CustomPattern(
                name=spec["name"],
                category=spec["category"],
                source=spec["pattern"],
                confidence=spec["confidence"],
                compiled=expression,
            )
        )
    return tuple(compiled)


@lru_cache(maxsize=256)
def _compile_cached(canonical: str) -> tuple[CustomPattern, ...]:
    return _compile(json.loads(canonical))


def compile_patterns(specs: Any) -> tuple[CustomPattern, ...]:
    """Validate and compile, reusing the compilation across requests.

    An installation's patterns do not change between calls, and compiling two
    dozen expressions on every request would be a tax on the fast path that
    CLAUDE.md section 32 asks us not to pay. The cache key is the canonical
    configuration, so a changed pattern compiles again and an unchanged one
    never does.
    """
    normalised = normalise_patterns(specs)
    if not normalised:
        return ()
    return _compile_cached(json.dumps(normalised, sort_keys=True, separators=(",", ":")))


def find(text: str, patterns: tuple[CustomPattern, ...]) -> list[CustomMatch]:
    """Every span the custom patterns claim, in no particular order.

    Ordering and overlap resolution belong to the detector, which has to settle
    custom spans against built-in ones anyway.

    Two budgets, not one. Per pattern, so a single bad expression is attributed
    to the pattern that caused it; and across the whole stage, because the
    per-pattern budget multiplied by :data:`MAX_PATTERNS` is longer than the
    timeout the orchestrator gives this extension, and a stage that outlives its
    own call is an outage with extra steps.
    """
    if not text or not patterns:
        return []

    out: list[CustomMatch] = []
    remaining = TOTAL_TIMEOUT_SECONDS
    for pattern in patterns:
        matched = 0
        budget = min(MATCH_TIMEOUT_SECONDS, remaining)
        if budget <= 0:
            raise CustomPatternError(
                f"custom patterns exceeded their {TOTAL_TIMEOUT_SECONDS}s total budget on this "
                f"input before reaching '{pattern.name}'. Simplify the patterns above it"
            )
        started = time.perf_counter()
        try:
            for match in pattern.compiled.finditer(text, timeout=budget):
                start, end = (
                    match.span(VALUE_GROUP) if pattern.uses_value_group else match.span()
                )
                # A `value` group that did not participate in the match reports
                # (-1, -1); an empty span contributes nothing to a rewrite.
                if start < 0 or end <= start:
                    continue
                out.append(
                    CustomMatch(
                        start=start,
                        end=end,
                        type=pattern.name,
                        category=pattern.category,
                        confidence=pattern.confidence,
                    )
                )
                matched += 1
                if matched >= MAX_MATCHES_PER_PATTERN:
                    break
        except TimeoutError as exc:
            raise CustomPatternError(
                f"custom pattern '{pattern.name}' exceeded its {round(budget, 3)}s budget "
                "on this input. Rewrite it to backtrack less -- anchor it, or replace a "
                "wildcard with a character class"
            ) from exc
        remaining -= time.perf_counter() - started
    return out


__all__ = [
    "MATCH_TIMEOUT_SECONDS",
    "MAX_PATTERNS",
    "MAX_PATTERN_CHARS",
    "TOTAL_TIMEOUT_SECONDS",
    "CustomMatch",
    "CustomPattern",
    "CustomPatternError",
    "compile_patterns",
    "find",
    "normalise_patterns",
]
