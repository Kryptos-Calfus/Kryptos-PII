# kryptos-pii-client

Mask personal information by calling Kryptos. Nothing to download, no model to
host — one dependency and an API key.

> **Not on PyPI yet.** `pip install kryptos-pii-client` will fail with *No
> matching distribution found* until it is published. Until then, download the
> wheel from the extension page and install the file. It is self-contained —
> `httpx` is its only dependency and that is on PyPI.

**Python 3.10 or newer.**

```bash
pip install ./kryptos_pii_client-0.3.0-py3-none-any.whl
export KRYPTOS_API_KEY=kr_live_xxxxxxxx
```

```python
from kryptos_pii_client import Kryptos

pii = Kryptos()

r = pii.redact("Call Priya on 9812345678, email priya@acme.com")

r.text           # 'Call [PERSON_NAME] on [PHONE], email [EMAIL]'
r.reason_codes   # ['PII_EMAIL', 'PII_PERSON_NAME', 'PII_PHONE']
r.risk           # 'medium'
```

Get a key from the Kryptos dashboard under **API keys**. It needs the matching
`pii:<operation>` scope — a key issued for `pii:redact` cannot call `tokenize`,
and a key never gains access to an extension installed after it was issued. A
refusal there is the platform working, not a bug in this client.

## Where your text goes

This package sends the text to Kryptos over TLS. It is held in memory for the
duration of the request and is not written to disk or logs; findings carry
offsets and types, never the matched values.

If the text must not leave your environment at all, this is the wrong package —
install [`kryptos-pii-local`](../python-local/README.md), which runs the detector
in your own process and needs no key. Both return the same `Result`, so
switching is one import line.

## The six operations

Detection is identical in every case. The operation only decides what happens to
the text afterwards. Input: `Email priya@acme.com, card 4111111111111111`

| Call | Result | Reversible | Use it when |
| --- | --- | --- | --- |
| `pii.detect(t)` | text unchanged | — | you want to decide for yourself |
| `pii.audit(t)` | text unchanged, always | — | you want a record, never a change |
| `pii.redact(t)` | `Email [EMAIL], card [PAYMENT_CARD]` | no | a model still has to read the text |
| `pii.mask(t)` | `Email **************, card ****************` | no | the shape of the record matters |
| `pii.tokenize(t)` | `Email <EMAIL_8ae616f5f64a>, …` | see below | the real values must come back |
| `pii.block(t)` | refused | — | this text must not move at all |

Most integrations want `redact`: the sentence still reads, so whatever consumes
the text next — usually a model — can still work with it.

`detect` has one sharp edge: it defers to the action your *installation* is
configured with, so on an installation set to redact, `detect` redacts. Use
`audit` when you need a guarantee that nothing changes.

Full explanation of all six, with the trade-offs: **[docs/operations.md](../../docs/operations.md)**.

### tokenize is not reversible here by default

The hosted service does not return the token-to-value mapping. A hosted PII
filter that accumulated a pile of customer plaintext would be a worse liability
than the problem it solves, so the vault is dropped unless the deployment has
explicitly opted in.

You get tokens you cannot reverse. If you need the round trip, run
`kryptos-pii-local`, where the mapping never leaves your process.

## Configuration per call

Any configuration key the installation accepts can be overridden for one call:

```python
pii.redact(text, threshold=0.7)             # stricter: fewer, surer findings
pii.detect(text, detector_mode="regex")     # skip the classifier
```

## Result

| Member | Meaning |
| --- | --- |
| `.text` | The content to pass on. `None` when the operation did not rewrite. |
| `.decision` | `allow`, `log`, `redact`, `mask`, `tokenize` or `block`. |
| `.blocked` | True when the decision was `block`. |
| `.found_anything` | True when anything was detected. |
| `.findings` | Type, offsets and confidence per span. Never the matched value. |
| `.reason_codes` | `PII_<TYPE>` per category found. What policies match on. |
| `.risk` | `none`, `low`, `medium`, `high`. High for secrets, cards, government ids. |
| `.latency_ms` | How long detection took. |

## Errors

`PIIError` carries `.status` and `.code` when Kryptos refused the call rather
than failing to answer it:

```python
from kryptos_pii_client import PIIError

try:
    r = pii.tokenize(text)
except PIIError as exc:
    exc.status   # 403
    exc.code     # 'scope_denied'
```

## Pointing at your own deployment

```bash
export KRYPTOS_BASE_URL=https://kryptos.internal.example.com
```

Or per client: `Kryptos(base_url="https://…", timeout=30.0)`.
