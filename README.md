# PII Protection

A Kryptos platform extension. It finds personal information in text and removes
it before it reaches a model, a tool, a log or a ticket.

Detection is two stages, and neither of them is an LLM:

1. **Regex proposes.** `finetune/common.py` knows no PII formats. Its only jobs
   are recall and span boundaries — every real value must overlap a candidate,
   and a candidate must not swallow the words around it.
2. **A fine-tuned classifier decides.** The LAYA checkpoint in `finetune/laya-pii`
   reads each candidate in context and answers whether it is personal. It
   reports val F1 0.988 at threshold 0.5 with a 120-character window.

Ahead of both sits a deterministic stage for the things that must never depend
on a neural decision: API keys, Luhn-valid card numbers, and labelled fields
(`Aadhaar: …`, `password: …`). That stage exists because the classifier is
measurably weak exactly there — a bare account number after the word `acct`
scores 0.001, because nothing in the digits is personal and the label is the
only evidence.

## Ways to use it

The choice between these is a privacy decision, not a performance one. Each is
declared in `extension.yaml`, and the Kryptos control plane renders its install
instructions and serves its download from that declaration.

| | How | Text leaves your environment |
| --- | --- | --- |
| **Hosted API** | `POST /api/v1/pii-protection/redact` with a Kryptos API key | yes |
| **`kryptos-pii-client`** | hosted SDK — a thin client, one dependency | yes |
| **`kryptos-pii-local`** | local SDK — the detector runs in your process | no |
| **Claude plugin** | four tools and a skill in Claude Code | no, in its default mode |
| **Local runtime** | `uvicorn kryptos_pii.service:app` in your own cluster | no |

All of them call the same `kryptos_pii.engine.run`. The adapters translate; they
do not decide. That is the point of the extension contract — a second detector
hiding behind one of these would be a second thing to audit.

```python
from kryptos_pii_local import protect
protect("Call Priya on 9812345678")     # 'Call [PERSON_NAME] on [PHONE]'
```

Neither SDK is on PyPI yet, and both ship as downloads from the extension page
until they are. `scripts/build_artifacts.py` builds them; the local one is a
bundle of two wheels, because it depends on `kryptos-pii` and that has no index
to resolve from yet. Python 3.12 or newer for anything local, 3.10 for the
hosted client.

## Operations

Six, and the detection is identical in all of them — only what happens to the
text afterwards differs.

| | Result for `Email priya@acme.com` | Reversible |
| --- | --- | --- |
| `detect` | unchanged; returns findings | — |
| `audit` | unchanged, whatever the configuration | — |
| `redact` | `Email [EMAIL]` | no |
| `mask` | `Email **************` | no |
| `tokenize` | `Email <EMAIL_8ae616f5f64a>` | **yes** |
| `block` | refused | — |

Text with nothing in it always returns `allow`, whatever the operation.

**[docs/operations.md](docs/operations.md)** explains each one, what it costs
you, and how to choose. Read it before wiring one in — `redact`, `mask` and
`tokenize` all mean "take the PII out", and which you want depends on
differences the names do not carry.

## The checkpoint

Local execution needs the fine-tuned checkpoint, which is ~850 MB and therefore
not inside any wheel:

```bash
kryptos-pii-model download     # fetch it into ~/.cache/kryptos/pii
kryptos-pii-model status       # is it here
kryptos-pii-model path         # where it will be loaded from
```

An explicit `KRYPTOS_PII_MODEL_DIR` wins; otherwise a checkout's
`finetune/laya-pii` is used, and failing that the download cache.

## Layout

```
extension.yaml          the manifest the Kryptos registry publishes
kryptos_pii/            the installable package: everything inference needs
  candidates.py         how text is split into candidate spans
  detector.py           regex candidates + the fine-tuned classifier
  classify.py           what category a detected span is
  engine.py             findings -> a deterministic decision
  service.py            POST /v1/execute, GET /health
  contract.py           the contract, copied not imported, and version-checked
  model.py              where the checkpoint is, and how to fetch it
sdk/python-hosted/      kryptos-pii-client: calls the Kryptos API
sdk/python-local/       kryptos-pii-local: runs the detector in your process
integrations/
  claude-plugin/        the Claude Code plugin, and the MCP server behind it
scripts/
  build_artifacts.py    builds every download extension.yaml promises
  convert_checkpoint.py stores the checkpoint in bf16, the precision it runs in
  compare_checkpoints.py does a changed checkpoint decide anything differently
  publish_model.py      uploads the checkpoint to the Hugging Face Hub
RELEASING.md            how the packages and the checkpoint get published
docs/operations.md      what each of the six operations does
tests/                  27 tests, no checkpoint needed
finetune/               the dataset, training, evaluation and the checkpoint
                        (common.py is now an alias for kryptos_pii.candidates,
                        so training scripts import it exactly as before)
```

`contract.py` is a deliberate copy of the orchestrator's contract rather than an
import of it. The two are separate repositories and separate deployables; an
import would couple their release cycles. What keeps the copy honest is
`contract_version`, which the orchestrator sends on every call and this
extension refuses when it does not match.

## Run it

```bash
uv sync
uv run pytest                                   # 27 tests, no model needed
HF_HUB_OFFLINE=1 uv run python -m uvicorn kryptos_pii.service:app --port 8080
```

Deploy with `helm-charts/kryptos` (the `pii` subchart), then build the
downloads and register everything:

```bash
uv run python scripts/build_artifacts.py        # wheels + the plugin archive

kryptos registry publish --manifest extension.yaml \
  --endpoint http://kryptos-pii.kryptos.svc.cluster.local
```

Publishing uploads each artifact the manifest declares and pins its sha256, so
the extension page's download buttons and the Claude marketplace's integrity
check both describe the bytes that actually shipped. A declared artifact that
has not been built stops the publish rather than shipping a button that 404s.

## What it does not do

* **It does not call an LLM.** CLAUDE.md section 11 is explicit: detection is
  regex plus a local checkpoint, and there is no model API on the path. The
  three-engine comparison demo that once sat here (LAYA vs Presidio vs
  GPT-4o-mini) has been removed along with its dependencies; `git log` has it if
  you want it back.
* **It does not fall back silently.** No checkpoint means 503, not a quiet drop
  to regex. `detector_mode: regex` is available, has materially lower recall,
  and names itself in every result's metadata.
* **It does not keep your text.** Nothing is written to disk or logs. Findings
  carry types, offsets and confidence, never the matched value — they travel
  into the orchestrator's audit trail, where content must never go.
* **It is not perfect recall.** The classifier misses things; `10.2.3.4` with no
  label scores 0.003. Measure before you trust a threshold:
  `uv run python finetune/evaluate.py`.
