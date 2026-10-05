# Kryptos model gateway

A local proxy that redacts the latest user message on its way to the Anthropic
API. Claude receives `[EMAIL]` instead of `john@example.com` and **answers
normally** — no block, no resubmitting.

```text
Claude Code  ──ANTHROPIC_BASE_URL──>  kryptos-gateway  ──>  api.anthropic.com
                                            │
                                            └── kryptos_pii_local.redact()
                                                (this machine, no network)
```

This is what a `UserPromptSubmit` hook cannot do. A hook is *consulted about* a
request and may only allow or veto it; a proxy *owns* the request and can
rewrite it.

## Run it

```bash
pip install kryptos-pii-local 'kryptos-pii[serve]' httpx
kryptos-pii-model download          # once, ~850 MB

python integrations/proxy/gateway.py --port 8787
```

Then point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude
```

That is the whole integration. Your API key, your model choice and every other
setting are unchanged — the gateway forwards `x-api-key`, `authorization`,
`anthropic-version` and `anthropic-beta` untouched.

Check it is alive:

```bash
curl -s http://127.0.0.1:8787/kryptos/health
```

| Variable | Default | |
| --- | --- | --- |
| `KRYPTOS_GATEWAY_UPSTREAM` | `https://api.anthropic.com` | point elsewhere for tests or a gateway of your own |
| `KRYPTOS_GATEWAY_PORT` | `8787` | |
| `KRYPTOS_GATEWAY_HOST` | `127.0.0.1` | loopback on purpose |
| `KRYPTOS_GATEWAY_READ_TIMEOUT` | `600` | must outlast your slowest completion |

## SECURITY BOUNDARY

Read this before relying on it.

### What it protects

**The latest user message, and only its `text` blocks.** That is what you type
into Claude Code. Email, phone, SSN, passwords and the other categories the
detector supports are replaced before the request leaves your machine.

### What it does NOT protect

Everything else in the request is forwarded **byte for byte**:

| Not redacted | Why |
| --- | --- |
| `system` prompt | yours, and rewriting it would change the agent's instructions |
| `tools` definitions | a renamed parameter breaks tool calling |
| assistant messages | the model's own prior output |
| **earlier user messages** | already sent on a previous turn; redacting now would rewrite history the model has seen |
| **`tool_result` blocks** | this is file contents, command output, database rows |

That last row is the one that matters. **If Claude reads a file containing
personal information, that file's contents go to the API unredacted.** The same
is true of command output and anything else a tool returns.

This is a deliberate v1 scope, not an oversight. A Claude Code request carries
the entire conversation, every tool definition and the contents of every file
read. Running all of it through a PII detector would strip identifiers out of
source code, names out of diffs and addresses out of fixtures, and the agent
would stop working.

So: **this gateway protects what you type. It does not protect what Claude
reads.** Do not describe it as "no PII reaches the model".

### Fail closed

Any failure — unparseable body, a detector error, findings with no redacted
text, an unexpected content type — returns `400` and **forwards nothing**. The
original request is never sent after an error, because that is precisely the
request someone wanted protected.

### Logging

Stderr carries category names and counts only (`redacted latest user message:
email, phone`). Prompt text and matched values are never written anywhere.

### Local only

The gateway imports `kryptos_pii_local`. There is no import of
`kryptos_pii_client` and no code path to a Kryptos service. The single outbound
connection this process makes is to `KRYPTOS_GATEWAY_UPSTREAM`.

## Use it with the hook

The gateway and the `UserPromptSubmit` hook are complementary and can run
together:

| | Gateway | Hook |
| --- | --- | --- |
| Result | redacts, Claude answers | blocks, you resubmit |
| Covers | any client using `ANTHROPIC_BASE_URL` | Claude Code only |
| If it fails | request refused | prompt refused |

Keeping the hook on is a backstop: it still fails closed if the gateway is not
running, or if someone forgets `ANTHROPIC_BASE_URL`. Running both means a prompt
with PII is stopped by the hook before the gateway ever sees it — so if you want
the gateway's redact-and-answer behaviour, turn the hook off for that project
with `KRYPTOS_PII_HOOK_ENFORCE=false`, and accept that you are then relying on
the gateway alone.

## Tests

```bash
pytest tests/test_gateway.py
```

They start a mock Anthropic server and the gateway as real processes, and assert
on **what the mock server received** — not on what the gateway says it did. The
claim is about what crosses the wire, so the witness is on the far side of it.
