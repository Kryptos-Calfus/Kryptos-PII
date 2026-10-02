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

## Four ways to use it

| | How | Text leaves your environment |
| --- | --- | --- |
| **Hosted API** | `POST /api/v1/pii-protection/redact` with a Kryptos API key | yes |
| **Python SDK** | `Kryptos(api_key=...)` hosted, or `local` in-process | your choice |
| **Local runtime** | `uvicorn kryptos_pii.service:app` in your own cluster | no |
| **Claude (MCP)** | four tools in Claude Desktop or Claude Code | not in local mode |

All four call the same `kryptos_pii.engine.run`. The adapters translate; they do
not decide. That is the point of the extension contract — a second detector
hiding behind one of these would be a second thing to audit.

```python
from kryptos_pii_sdk import protect
protect("Call Priya on 9812345678")     # 'Call [PERSON_NAME] on [PHONE]'
```

## Operations

`detect`, `redact`, `mask`, `tokenize`, `block`, `audit`.

```
in   Priya Sharma, 9812345678, priya@acme.com, PAN ABCDE1234F, card 4111 1111 1111 1111
out  [PERSON_NAME], [PHONE], [EMAIL], PAN [GOVERNMENT_ID], card [PAYMENT_CARD]
```

`audit` never rewrites — it records that personal data was present and leaves the
content alone, so it is safe to turn on in front of live traffic. Text with
nothing in it always returns `allow`, whatever the operation.

## Layout

```
extension.yaml          the manifest the Kryptos registry publishes
kryptos_pii/
  detector.py           regex candidates + the fine-tuned classifier
  classify.py           what category a detected span is
  engine.py             findings -> a deterministic decision
  service.py            POST /v1/execute, GET /health
  contract.py           the contract, copied not imported, and version-checked
sdk/python/             the SDK: hosted and local
integrations/mcp/       the MCP server for Claude
tests/                  27 tests, no checkpoint needed
finetune/               the dataset, training, evaluation and the checkpoint
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

Deploy with `helm-charts/kryptos` (the `pii` subchart), then register it:

```bash
kryptos registry publish --manifest extension.yaml \
  --endpoint http://kryptos-pii.kryptos.svc.cluster.local
```

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
