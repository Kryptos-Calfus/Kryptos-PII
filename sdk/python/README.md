# kryptos-pii-sdk

```bash
pip install kryptos-pii-sdk
```

## Hosted — Kryptos does the masking

```python
import os
from kryptos_pii_sdk import Kryptos

pii = Kryptos(api_key=os.environ["KRYPTOS_API_KEY"])

result = pii.redact("Call Priya on 9812345678")
result.text          # 'Call [PERSON_NAME] on [PHONE]'
result.decision      # 'redact'
result.reason_codes  # ['PII_PERSON_NAME', 'PII_PHONE']
```

Every call is authenticated, scoped, metered and audited by the control plane.
The key needs the matching `pii:<operation>` scope.

## Local — text never leaves the machine

```python
from kryptos_pii_sdk import local, protect

protect("Call Priya on 9812345678")   # 'Call [PERSON_NAME] on [PHONE]'

result = local.tokenize("Email priya@acme.com")
result.text                      # 'Email <EMAIL_8ae616f5f64a>'
result.detokenize(result.text)   # 'Email priya@acme.com'
```

No API key, no network. This needs the extension installed locally
(`pip install kryptos-pii` plus the checkpoint); section 8 of the platform spec
is explicit that purely local execution must not require a credential.

## Choosing

The two clients return the same `Result`, so moving between them is one line.
The difference is not performance, it is where the text goes:

| | Hosted | Local |
| --- | --- | --- |
| Text leaves your environment | yes | no |
| API key | required | not used |
| Usage, audit, policy in the dashboard | yes | decision only, reported without content |
| You host the 1.6 GB checkpoint | no | yes |

## Result

| Member | Meaning |
| ------ | ------- |
| `.text` | The content to pass on. `None` when the action did not rewrite. |
| `.decision` | `allow`, `log`, `redact`, `mask`, `tokenize` or `block`. |
| `.blocked` | True when the decision was `block`. |
| `.findings` | Type, offsets and confidence per span. Never the matched value. |
| `.reason_codes` | `PII_<TYPE>`, what policies match on. |
| `.vault` | Token to original value, for `tokenize`. Local only by default. |

`PIIError` carries `.status` and `.code` when the platform refused the call.
