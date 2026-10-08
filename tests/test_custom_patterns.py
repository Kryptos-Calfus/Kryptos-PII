"""Customer-supplied patterns: what they must do, and what they must never do.

Two halves. The first is the feature -- a customer's own identifier is found,
labelled, categorised and acted on like any other finding. The second is the
blast radius, which is the part that matters: this is a configuration field that
accepts a program, so every one of these tests is about what the platform
refuses to run.

All of it runs in ``detector_mode="regex"``, so no checkpoint is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kryptos_pii.custom import (
    MAX_PATTERNS,
    CustomPatternError,
    compile_patterns,
    normalise_patterns,
)
from kryptos_pii.detector import detect
from kryptos_pii.engine import resolve_config, run

CLAIM = {"name": "claim_number", "pattern": r"KR-[0-9]{8}", "category": "government_id"}
BADGE = {"name": "badge", "pattern": r"badge\s*#?\s*(?P<value>[A-Z]{2}-\d{4})", "ignore_case": True}


def config(*patterns: dict) -> dict:
    return {"detector_mode": "regex", "custom_patterns": list(patterns)}


# --- the feature ----------------------------------------------------------


def test_a_custom_pattern_finds_what_the_built_in_detector_cannot() -> None:
    text = "Claim KR-10482093 was reopened."
    result = run(text, operation="redact", config=config(CLAIM)).result

    assert result.transformed_content == "Claim [CLAIM_NUMBER] was reopened."
    (finding,) = result.findings
    assert (finding.type, finding.category, finding.detector) == (
        "claim_number",
        "government_id",
        "custom-regex",
    )
    assert text[finding.start : finding.end] == "KR-10482093"


def test_the_declared_category_decides_risk_and_the_broad_reason_code() -> None:
    """The category is why a policy written before the pattern existed covers it.

    A rule matching PII_GOVERNMENT_ID, or finding_type government_id, fires on a
    claim number from the moment the pattern is added -- nobody has to go back
    and widen the policy.
    """
    result = run("Claim KR-10482093", operation="detect", config=config(CLAIM)).result

    assert result.risk == "high"  # government_id is a high-risk category
    assert set(result.reason_codes) == {"PII_CLAIM_NUMBER", "PII_GOVERNMENT_ID"}


def test_an_undeclared_category_is_not_guessed() -> None:
    result = run("Ref XQ-99 here", operation="detect", config=config(
        {"name": "internal_ref", "pattern": r"XQ-\d{2}"}
    )).result

    (finding,) = result.findings
    assert finding.category == "personal_data"
    assert result.risk == "low"


def test_a_value_group_reports_only_the_value() -> None:
    """Matching on the label and redacting only what follows it."""
    result = run("Please check Badge #AB-1234 today", operation="redact", config=config(BADGE)).result

    assert result.transformed_content == "Please check Badge #[BADGE] today"


def test_custom_patterns_run_in_hybrid_mode_too() -> None:
    """They are shapes, not candidates, so the classifier never sees them.

    Asserted through detect() rather than run() so no checkpoint is loaded: the
    custom stage produces its finding before the model stage is reached.
    """
    patterns = compile_patterns([CLAIM])
    found = detect("KR-10482093", detector_mode="regex", custom_patterns=patterns)

    assert [(d.type, d.detector) for d in found] == [("claim_number", "custom-regex")]


def test_a_custom_pattern_wins_an_overlapping_span() -> None:
    """The customer said what this text is; a built-in guess does not overrule it."""
    patterns = [{"name": "support_alias", "pattern": r"help@[a-z.]+", "category": "email"}]
    result = run("Write to help@acme.com", operation="detect", config=config(*patterns)).result

    (finding,) = result.findings
    assert finding.type == "support_alias"


def test_findings_still_never_carry_the_matched_text() -> None:
    result = run("Claim KR-10482093", operation="detect", config=config(CLAIM)).result
    assert "KR-10482093" not in result.model_dump_json()


def test_the_result_names_which_patterns_fired() -> None:
    result = run("Claim KR-10482093", operation="detect", config=config(CLAIM, BADGE)).result

    assert result.metadata["custom_patterns"] == 2
    assert result.metadata["custom_matches"] == ["claim_number"]


def test_confidence_is_reported_and_filtered_on() -> None:
    loose = {"name": "maybe_ref", "pattern": r"Z\d{3}", "confidence": 0.4}
    settings = config(loose) | {"min_confidence": 0.5}

    assert run("Z123", operation="detect", config=config(loose)).result.findings
    assert not run("Z123", operation="detect", config=settings).result.findings


# --- what the platform refuses -------------------------------------------


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ({"name": "Claim Number", "pattern": "x"}, "'name' must be lowercase"),
        ({"name": "email", "pattern": "x"}, "built-in type"),
        ({"name": "ok_name", "pattern": ""}, "non-empty"),
        ({"name": "ok_name", "pattern": "("}, "not a valid regular expression"),
        ({"name": "ok_name", "pattern": r"a?"}, "matches the empty string"),
        ({"name": "ok_name", "pattern": r"(\d+\s*)+$"}, "nests a quantifier"),
        ({"name": "ok_name", "pattern": "x", "category": "invented"}, "'category' must be one of"),
        ({"name": "ok_name", "pattern": "x", "confidence": 2}, "between 0 and 1"),
        ({"name": "ok_name", "pattern": "x", "colour": "red"}, "unknown keys"),
        ({"name": "ok_name", "pattern": "x" * 401}, "the limit is"),
    ],
)
def test_a_pattern_the_platform_will_not_run_is_refused_with_a_reason(
    spec: dict, fragment: str
) -> None:
    """Refused loudly, and never silently dropped.

    Dropping it would leave a customer believing their identifiers were being
    detected, which is the one failure mode a PII filter must not have.
    """
    with pytest.raises(CustomPatternError) as caught:
        compile_patterns([spec])
    assert fragment in str(caught.value)


def test_duplicate_names_are_refused() -> None:
    with pytest.raises(CustomPatternError, match="duplicate"):
        compile_patterns([CLAIM, CLAIM])


def test_the_pattern_count_is_capped() -> None:
    many = [{"name": f"ref_{i}", "pattern": f"A{i}B"} for i in range(MAX_PATTERNS + 1)]
    with pytest.raises(CustomPatternError, match="at most"):
        compile_patterns(many)


def test_a_catastrophic_pattern_times_out_instead_of_hanging() -> None:
    """The nesting heuristic does not catch everything, so the budget must.

    `(a|aa)+$` has no nested quantifier for the validator to see and still
    backtracks exponentially. It is accepted at configuration time and stopped
    at match time, which is why both defences exist.
    """
    evil = {"name": "pathological", "pattern": r"(a|aa)+$"}
    with pytest.raises(CustomPatternError, match="budget"):
        run("a" * 5000 + "b", operation="detect", config=config(evil))


def test_configuration_is_validated_when_it_is_saved() -> None:
    """resolve_config compiles, so a bad pattern fails on save rather than on
    the first request whose text happens to reach it."""
    with pytest.raises(CustomPatternError):
        resolve_config(config({"name": "ok_name", "pattern": "["}))


def test_an_empty_configuration_stays_empty() -> None:
    assert normalise_patterns(None) == []
    assert resolve_config(None)["custom_patterns"] == []


# --- the manifest promises exactly this ----------------------------------


def test_the_manifest_describes_the_field_the_engine_implements() -> None:
    manifest = yaml.safe_load((ROOT / "extension.yaml").read_text())
    schema = manifest["configuration_schema"]["custom_patterns"]

    assert schema["type"] == "array"
    assert schema["max_items"] == MAX_PATTERNS
    assert set(schema["items"]["required"]) == {"name", "pattern"}
    # The categories offered are the ones the taxonomy actually has: an enum
    # listing a category the engine would reject is a promise to a customer the
    # extension cannot keep.
    from kryptos_pii.taxonomy import CATEGORIES

    assert set(schema["items"]["properties"]["category"]["enum"]) == set(CATEGORIES)


def test_the_manifest_requires_an_api_key_for_hosted_execution() -> None:
    manifest = yaml.safe_load((ROOT / "extension.yaml").read_text())
    assert manifest["requires_api_key"] is True
    hosted = [c for c in manifest["delivery"] if c["execution_mode"] == "hosted_api"]
    assert hosted and all(c["requires_api_key"] for c in hosted)
