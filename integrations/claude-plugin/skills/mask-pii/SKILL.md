---
name: mask-pii
description: Use before personal information would leave a safe place — pasting user data, logs, support tickets, database rows or CSV samples into a prompt, a commit message, an issue, a bug report, a test fixture or a file that gets committed. Also when the user asks to redact, mask, anonymize, scrub, de-identify or tokenize text, or asks whether something contains PII. Provides pii_detect, pii_redact, pii_mask and pii_tokenize.
---

# Masking personal information

Four tools, backed by a detector that runs where the plugin is configured to run
it. Detection is identical in all four; the tool only decides what happens to
the text.

## Which tool

| Tool | Result | Use it when |
| --- | --- | --- |
| `pii_redact` | `priya@acme.com` → `[EMAIL]` | **the default.** The sentence still reads afterwards |
| `pii_mask` | `priya@acme.com` → `**************` | the layout matters — fixed-width, aligned columns, a diff |
| `pii_tokenize` | `priya@acme.com` → `<EMAIL_8ae616f5f64a>` | the real values have to come back afterwards |
| `pii_detect` | nothing changes | you only need to know whether anything is in there |

When in doubt, `pii_redact`.

## When to reach for this

Call a masking tool *before* the text lands somewhere it will persist or be
read by something else:

- Sample rows, query results or log lines going into a prompt, an issue or a
  commit message
- A support ticket, email or transcript the user pastes in for analysis
- Test fixtures and seed data built from something real
- A bug report or stack trace that carries request bodies
- Anything the user describes as customer data, user data or production data

You do not need it for code, configuration, documentation or synthetic examples
that were never real.

## Using the result

The tool returns `text`, which is the safe version. Use that string from then
on — do not go back to the original, and do not reconstruct the values you were
shown before masking.

The result also carries `types`, `found` and `reason_codes` so you can tell the
user what was removed:

> I redacted 2 email addresses and a phone number before putting this in the
> issue.

The tools deliberately never return the matched values. Findings carry
categories and offsets only, because this output goes into the conversation,
and putting the personal information back here would undo the masking.

## The round trip

`pii_tokenize` is the only reversible one. Use it when the text must come back
intact — the user wants a rewritten email that still has the real address in it,
for example. Work on the tokenized text and leave the tokens alone; the plugin
puts the real values back on its side. Never guess at what a token stood for.

## If a tool fails

The server reports plainly when it cannot run: a missing checkpoint in local
mode, a missing API key in hosted mode, the wrong Python interpreter. Pass the
message on to the user rather than falling back to masking the text yourself
with a regex — a half-working PII filter is worse than a visibly broken one,
because nobody finds out.
