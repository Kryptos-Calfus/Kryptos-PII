# The six operations

PII Protection finds personal information in text and then does one of six
things about it. Detection is identical in every case — the same regex
candidates, the same classifier, the same findings. The operation only decides
what happens to the text afterwards.

Throughout this page the input is:

```text
Email priya@acme.com, card 4111111111111111
```

## At a glance

| Operation | What comes back | Reversible | Use it when |
| --- | --- | --- | --- |
| `detect` | the text, unchanged | — | you want to decide for yourself |
| `audit` | the text, unchanged | — | you want a record, never a change |
| `redact` | `Email [EMAIL], card [PAYMENT_CARD]` | no | a model still has to understand the text |
| `mask` | `Email **************, card ****************` | no | the shape of the record matters |
| `tokenize` | `Email <EMAIL_8ae616f5f64a>, card <PAYMENT_CARD_1f2cd9a4b7e0>` | **yes** | the real values have to come back afterwards |
| `block` | nothing; the request is refused | — | this text must not move at all |

Three families, which is the useful way to hold it in your head:

```text
observe          rewrite                      refuse
detect, audit    redact, mask, tokenize       block
```

Every one of them returns the same structured findings — type, offsets,
confidence — and none of them ever returns the matched value. That is
deliberate: findings travel into audit logs and model context, and putting the
personal information back there would undo the work.

---

## observe

### `detect`

Finds everything and changes nothing. You get the categories, where they are,
and how confident the detector was.

```python
r = pii.detect("Email priya@acme.com, card 4111111111111111")

r.found_anything   # True
r.decision         # 'log'
r.reason_codes     # ['PII_EMAIL', 'PII_PAYMENT_CARD']
r.text             # None — nothing was rewritten
[f.type for f in r.findings]          # ['email', 'payment_card']
[(f.start, f.end) for f in r.findings] # [(6, 20), (27, 43)]
```

**One sharp edge.** `detect` defers to whatever action the *installation* is
configured with. On an installation configured `action: redact`, calling
`detect` redacts. This is what lets one endpoint serve a fleet whose policies
differ, but it means `detect` is not a guarantee that nothing changes.

If you need that guarantee, use `audit`.

### `audit`

Same detection, and it never rewrites, whatever the installation is configured
to do. That is the entire difference from `detect`.

```python
r = pii.audit(text)
r.text   # None. Always. Regardless of configuration.
```

Reach for it when the point is the record rather than the result: a scanner
walking a corpus, a monitor-mode rollout measuring what *would* happen, a
compliance sweep that must not mutate what it reads.

---

## rewrite

All three replace the detected spans in place and leave the rest of the text
untouched. They differ in what they put there, and that choice has consequences.

### `redact` — the category stays readable

```text
Email [EMAIL], card [PAYMENT_CARD]
```

The value is gone; the *kind* of thing it was is still there. Use this when
something downstream still has to make sense of the sentence — most often a
model. `Email [EMAIL]` still reads as a sentence about an email address.
`Email ********` reads as noise.

Length changes, so offsets into the original no longer line up.

### `mask` — the shape stays intact

```text
Email **************, card ****************
```

Character-for-character the same length as the original. Use it when something
depends on the layout: a fixed-width export, a terminal table, a screenshot, a
diff that must stay aligned.

The cost is that the masked text says nothing about what was removed.

### `tokenize` — the value can come back

```text
Email <EMAIL_8ae616f5f64a>, card <PAYMENT_CARD_1f2cd9a4b7e0>
```

The only reversible operation, and the reason it exists is the round trip:

```python
r = pii.tokenize("Email priya@acme.com")

reply = model(r.text)        # the model never sees the address
final = r.detokenize(reply)  # the address is back in the answer
```

The token is an HMAC of the value, so the same value yields the same token
within a run and the token cannot be reversed by guessing. The mapping lives in
`result.vault`.

**Where the vault lives is the whole security story.** Locally it never leaves
your process. The hosted service does *not* return it by default — a hosted PII
filter that accumulated a pile of customer plaintext would be a worse liability
than the problem it solves. So hosted `tokenize` gives you tokens you cannot
reverse unless the deployment has explicitly opted in. If you need the round
trip, run locally.

By default the HMAC key is fresh per process, so tokens are not stable across
restarts. Set `KRYPTOS_PII_TOKEN_SECRET` if you need them to be.

---

## refuse

### `block`

Detection runs, and if anything is found the decision is `block`. There is no
transformed text, because the point is that the text does not move.

```python
r = pii.block(text)
r.blocked        # True
r.reason_codes   # ['PII_EMAIL', 'PII_PAYMENT_CARD', 'PII_PRESENT_BLOCKED']
r.text           # None
```

Clean text passes: when nothing is found, every operation — `block` included —
returns `allow`. You cannot block on an empty finding list, because there is
nothing to block over.

Note that `block` here is the *extension's* decision, not the last word. The
orchestrator's policy engine combines decisions from every installed extension
and takes the strongest applicable one, so another extension can escalate an
`allow` to a block. The ladder is:

```text
allow < log < redact < require_approval < block
```

---

## Choosing

Most integrations want `redact`. It is the default action for a reason: the text
stays usable, nothing leaks, and nothing has to be kept anywhere.

```text
Does the real value have to come back later?
├── yes ─────────────────────────────────> tokenize (run it locally)
└── no
    │
    Does something downstream read this text?
    ├── a model, a person, a log ────────> redact
    └── a fixed-width format ────────────> mask

Should this text move at all?
└── no ──────────────────────────────────> block

Just looking?
├── and a config change must not surprise me ──> audit
└── and I want the installation's action ──────> detect
```

## What every result carries

| Member | Meaning |
| --- | --- |
| `.text` | The content to pass on. `None` when the operation did not rewrite. |
| `.decision` | `allow`, `log`, `redact`, `mask`, `tokenize` or `block`. |
| `.blocked` | True when the decision was `block`. |
| `.found_anything` | True when anything was detected. |
| `.findings` | Type, offsets and confidence per span. Never the matched value. |
| `.reason_codes` | `PII_<TYPE>` per category found, plus `PII_PRESENT_BLOCKED`. What policies match on. |
| `.risk` | `none`, `low`, `medium` or `high`. High when a secret, card or government id was found. |
| `.vault` | Token to original value, for `tokenize`. Local by default. |
| `.latency_ms` | How long detection took. |
