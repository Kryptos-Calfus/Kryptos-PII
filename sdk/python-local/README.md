# kryptos-pii-local

Mask personal information on your own machine. No API key, no network, no
control plane — the text never leaves the process.

> **Not on PyPI yet.** `pip install kryptos-pii-local` will fail with *No
> matching distribution found* until it is published. Until then, download the
> offline bundle from the extension page — it holds this package and the
> extension it wraps — and install from that.

**Python 3.12 or newer.** The macOS system `python3` is 3.9 and will refuse the
install.

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# from the unzipped bundle
pip install --find-links . kryptos-pii-local

kryptos-pii-model download          # the checkpoint, once (~850 MB)
```

`--find-links .` resolves the two Kryptos wheels out of the directory; the
ordinary dependencies still come from your package index. From a checkout,
`pip install ../.. ../python-local` does the same thing.

```python
from kryptos_pii_local import redact

r = redact("Call Priya on 9812345678, email priya@acme.com")

r.text           # 'Call [PERSON_NAME] on [PHONE], email [EMAIL]'
r.reason_codes   # ['PII_EMAIL', 'PII_PERSON_NAME', 'PII_PHONE']
r.risk           # 'medium'
```

Or with no ceremony at all:

```python
from kryptos_pii_local import protect

protect("Call Priya on 9812345678")   # 'Call [PERSON_NAME] on [PHONE]'
```

## The checkpoint

The detector is a fine-tuned classifier, and its weights are too large to ship
inside a wheel. So installing takes seconds and the model arrives in a second,
explicit step:

```bash
kryptos-pii-model download     # fetch it
kryptos-pii-model status       # is it here, and how big
kryptos-pii-model path         # where it will be loaded from
```

Three locations are searched, first one with weights wins:

1. `KRYPTOS_PII_MODEL_DIR` — an explicit path, always wins
2. `<repo>/finetune/laya-pii` — the training output, if you are in a checkout
3. `~/.cache/kryptos/pii/<repo>` — what `download` writes

Without a checkpoint, every call raises `ModelMissing` rather than quietly
falling back to regex. A PII filter that silently stops detecting most of what
it should is worse than one that is plainly down — you would never find out.

If you genuinely cannot host the checkpoint, ask for regex explicitly and accept
the materially lower recall:

```python
redact(text, detector_mode="regex")
```

## The six operations

Detection is identical in every case. The operation only decides what happens to
the text afterwards. Input: `Email priya@acme.com, card 4111111111111111`

| Call | Result | Reversible | Use it when |
| --- | --- | --- | --- |
| `detect(t)` | text unchanged | — | you want to decide for yourself |
| `audit(t)` | text unchanged, always | — | you want a record, never a change |
| `redact(t)` | `Email [EMAIL], card [PAYMENT_CARD]` | no | a model still has to read the text |
| `mask(t)` | `Email **************, card ****************` | no | the shape of the record matters |
| `tokenize(t)` | `Email <EMAIL_8ae616f5f64a>, …` | **yes** | the real values must come back |
| `block(t)` | refused | — | this text must not move at all |

Most integrations want `redact`: the sentence still reads, so whatever consumes
the text next can still work with it.

`detect` has one sharp edge — it defers to the configured `action`, so on a
configuration set to redact, `detect` redacts. Use `audit` when you need a
guarantee that nothing changes.

Full explanation of all six, with the trade-offs: **[docs/operations.md](https://github.com/Kryptos-Calfus/Kryptos-PII/blob/main/docs/operations.md)**.

### The round trip

`tokenize` is the only reversible one, and the round trip is why it exists:

```python
from kryptos_pii_local import tokenize

r = tokenize("Email priya@acme.com about the refund")

reply = model(r.text)        # the model never sees the address
final = r.detokenize(reply)  # the address is back in the answer
```

The mapping is in `r.vault` and never leaves this process. Tokens are an HMAC of
the value, so they cannot be reversed by guessing. The key is fresh per process
by default; set `KRYPTOS_PII_TOKEN_SECRET` if you need tokens stable across
restarts.

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
| `.vault` | Token to original value, for `tokenize`. |
| `.latency_ms` | How long detection took. |

Findings never carry the matched value. That is deliberate: they travel into
logs and model context, and putting the personal information back there would
undo the work.

## Hosted instead?

If you would rather Kryptos did the masking — nothing to download, calls metered
and audited in your dashboard — install
[`kryptos-pii-client`](https://pypi.org/project/kryptos-pii-client/). Both packages return the same
`Result`, so switching is one import line.

The difference is not performance. It is where the text goes.

| | `kryptos-pii-local` | `kryptos-pii-client` |
| --- | --- | --- |
| Text leaves your environment | no | yes |
| API key | not used | required |
| You host the checkpoint | yes | no |
| Usage and audit in the dashboard | no | yes |
| Reversible `tokenize` | yes | only if the deployment opts in |
