---
license: apache-2.0
library_name: laya
base_model: answerdotai/ModernBERT-large
pipeline_tag: token-classification
tags:
  - pii
  - privacy
  - redaction
  - anonymization
  - security
language:
  - en
---

# laya-pii

A fine-tuned LAYA classifier that decides, in context, whether a span of text is
personal information. It is the detector behind
Kryptos PII Protection. The code that proposes spans for it and reads its
scores is in
[Kryptos-Calfus/Kryptos-PII](https://github.com/Kryptos-Calfus/Kryptos-PII).

It answers one question per candidate span:

> Is the value of 'span' personal information about a specific individual in
> this text (name, contact detail, ID number, account, password, or birth date)?

## What it is not

**It is not a span finder.** It never reads raw text looking for PII. Something
upstream proposes candidate spans and this model rules on each one. On its own
it will find nothing.

That split is deliberate. Proposal is a recall problem a regex solves well and
cheaply; deciding whether `Grace` is a person or a word is a context problem a
regex cannot solve at all. `kryptos_pii.candidates` is the proposer this model
was trained against.

**It is not a classifier of PII type.** It answers yes or no. The category in a
Kryptos finding (`email`, `payment_card`, …) comes from a deterministic read of
the matched text, not from this model.

## Use it

```bash
pip install kryptos-pii-local
kryptos-pii-model download
```

```python
from kryptos_pii_local import redact

redact("Call Priya on 9812345678").text   # 'Call [PERSON_NAME] on [PHONE]'
```

That wires up the proposer, this model, the category pass and the masking. To
drive the checkpoint directly:

```python
import laya
from kryptos_pii.candidates import QUESTION, state_for, set_context_chars

set_context_chars(120)                      # what it was trained with
agent = laya.load("path/to/laya-pii")
text = "Call Priya on 9812345678"
scores = agent.predict_batch([state_for(text, 5, 10)], QUESTION)
scores[0]["answers"]["pii"]["noul"]         # P(personal information)
```

## Training

| | |
| --- | --- |
| Base | `answerdotai/ModernBERT-large` |
| Parameters | 421M |
| Precision | bfloat16 |
| Size | 843 MB |
| Context window | 120 characters each side of the span |
| Threshold | 0.5 |
| Validation F1 | 0.988 (piece level, at threshold 0.5) |
| Training data | 6000 synthetic examples, 3 epochs, epoch 2 selected |

The context window is not a tuning knob. The model was trained to see 120
characters on each side, and changing it at inference invalidates the reported
F1 until you re-measure.

Stored in bfloat16 because that is the precision it computes in
(`amp_dtype: bf16`); fp32 storage carried precision that every forward pass
discarded. Converting changed no detection on the held-out set.

## Measured behaviour

On a 111-case held-out set, end to end through the Kryptos pipeline, matching a
prediction to a true span by character overlap:

| | |
| --- | --- |
| Precision | 0.910 |
| Recall | 0.955 |
| F1 | 0.932 |

This is a different and harsher number than the 0.988 above: that one scores the
model's own yes/no decisions on proposed pieces, this one scores the whole
system against real spans and is what a user actually experiences.

## Limits

Read these before trusting it with anything that matters.

- **English only.** It has not been evaluated on any other language.
- **Recall is not 1.0.** It misses things. An unlabelled `10.2.3.4` scores
  0.003. Run `finetune/evaluate.py` against your own data before choosing a
  threshold; do not take 0.5 on faith for your domain.
- **The misses are its own.** On the 111-case held-out set every one of the 9
  missed values *was* proposed as a candidate and scored below threshold here:
  three IPv4 addresses, two spelled-out dates, an Indian mobile number, a Tamil
  name, a postcode and a street address. Raising recall means retraining or
  moving the threshold, not improving the proposer. It also cannot see what the
  proposer never offers, but on this set that was not the limiting factor.
- **Values whose shape is proof should not reach it.** API keys, Luhn-valid card
  numbers and labelled fields (`Aadhaar: …`, `password: …`) are matched
  deterministically in Kryptos and never put to a neural vote. A live credential
  must not depend on a probability.
- **Synthetic training data.** It was fine-tuned on generated examples. Names,
  formats and contexts outside that distribution are where it will be weakest,
  and non-Western names deserve your own evaluation rather than this one.
- **It is a detector, not a guarantee.** Masking output from an imperfect
  detector is risk reduction, not compliance.

## Licence

Apache-2.0. The base model carries its own licence from Answer.AI.
