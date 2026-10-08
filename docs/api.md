# PII Protection API

Two surfaces, one engine.

## Through the Kryptos platform

The normal path. The orchestrator authenticates the key, checks the scope,
applies policy, records usage and writes the audit event.

```http
POST /api/v1/pii-protection/redact
Authorization: Bearer kr_live_xxx
Content-Type: application/json

{"content": "Call Priya on 9812345678"}
```

```json
{
  "decision": "redact",
  "risk": "medium",
  "findings": [
    {"type": "person_name", "start": 5, "end": 10, "confidence": 1.0, "detector": "laya-pii"},
    {"type": "phone", "start": 14, "end": 24, "confidence": 0.952, "detector": "laya-pii"}
  ],
  "transformed_content": "Call [PERSON_NAME] on [PHONE]",
  "reason_codes": ["PII_PERSON_NAME", "PII_PHONE"],
  "extension": "pii-protection",
  "extension_version": "0.4.0"
}
```

The key needs the matching scope (`pii:redact` here). A key without it is
refused with `insufficient_scope`, and the extension must be installed on the
account or the call is a 404.

## Directly against the extension runtime

What the orchestrator itself calls, and what you call when you self-host.

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/v1/execute` | The contract endpoint. An `ExecutionRequest` in, an `ExtensionResult` out. |
| `POST` | `/v1/pii/{operation}` | Ergonomic form: `{"content": "...", "config": {...}}`. |
| `GET`  | `/health` | `HealthReport`, including whether the checkpoint loaded. |

## Operations

| Operation | Decision | Content |
| --------- | -------- | ------- |
| `detect`   | the configured action | rewritten if that action rewrites |
| `redact`   | `redact`   | spans replaced with `[TYPE]` |
| `mask`     | `mask`     | spans replaced with `*`, length preserved |
| `tokenize` | `tokenize` | spans replaced with stable reversible tokens |
| `block`    | `block`    | unchanged; the caller is expected to stop |
| `audit`    | `log`      | unchanged, findings recorded |

Text with no personal information always returns `allow`, whatever the
operation: there is nothing to redact and nothing to block over.

## Configuration

| Key | Default | Meaning |
| --- | ------- | ------- |
| `detector_mode` | `hybrid` | `hybrid` runs the classifier over regex candidates. `regex` keeps only self-evident shapes and labelled fields — materially lower recall, for deployments that cannot host the checkpoint. |
| `action` | `redact` | What `detect` does when it finds something. |
| `threshold` | `0.5` | Classifier probability at or above which a span is personal. The checkpoint was evaluated at 0.5; moving it invalidates that measurement until you re-run `finetune/evaluate.py`. |
| `min_confidence` | `0.0` | Drop findings below this before acting. |
| `custom_patterns` | `[]` | Your own identifiers, as regular expressions. See below. |

### Custom patterns

Every deployment has identifiers this detector cannot know about. Declare them
and they run as a deterministic stage beside the built-in shapes, in both
detector modes, and are never sent to the classifier:

```json
{
  "content": "Claim KR-10482093 was filed by Badge #AB-1234.",
  "config": {
    "custom_patterns": [
      {"name": "claim_number", "pattern": "KR-[0-9]{8}", "category": "government_id"},
      {"name": "employee_badge", "pattern": "badge\\s*#?\\s*(?P<value>[A-Z]{2}-\\d{4})", "ignore_case": true}
    ]
  }
}
```

```
Claim [CLAIM_NUMBER] was filed by Badge #[EMPLOYEE_BADGE].
```

| Field | Required | Meaning |
| ----- | -------- | ------- |
| `name` | yes | Becomes the finding type and the `PII_<NAME>` reason code. Lowercase letters, digits and underscores, 2–40 characters. Must not be one of the built-in types. |
| `pattern` | yes | A Python regular expression, at most 400 characters. Name a group `value` to report only part of the match, which is how the label above stays readable while the identifier is redacted. |
| `category` | no | One of the taxonomy's categories (`government_id`, `secret`, `payment_card`, …). It decides the risk level and the broad reason code, so a policy rule written against `government_id` covers a new claim-number pattern the moment you add it. Defaults to `personal_data`. |
| `ignore_case` | no | Match without regard to case. Default `false`. |
| `confidence` | no | Reported on every finding the pattern produces and compared against `min_confidence`. Default `1.0`. |

Where a custom pattern and a built-in detector claim the same text, yours wins.

**What is refused, and why.** A regular expression is a program, so a field that
accepts one accepts a way to hang the service. Patterns are compiled when the
configuration is saved and rejected if they are malformed, match the empty
string, nest a quantifier inside a quantified group, or exceed the limits (24
patterns, 400 characters each). Matching then runs under a budget — 0.25s per
pattern and 1s for the whole stage — and a pattern that exceeds it returns
`422` naming the pattern rather than stalling the request. Your failure policy
decides what that refusal means; for a transforming operation the default is to
fail closed.

## Errors

| Status | Meaning |
| ------ | ------- |
| `422` | The configuration or operation is not valid. |
| `503` | The checkpoint is not loadable. The extension does **not** silently fall back to regex — your failure policy decides whether that fails open or closed. |

## What is never returned

Findings carry a type, offsets and a confidence, never the matched text. They
travel into the orchestrator's audit trail, and putting the values there would
defeat the point of the extension. The tokenization vault is returned only to a
direct caller, and only when the deployment sets `KRYPTOS_PII_RETURN_VAULT`.
