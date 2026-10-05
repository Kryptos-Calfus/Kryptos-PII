# Kryptos PII Protection — Claude plugin

Gives Claude four tools so it can strip personal information before text reaches
a prompt, a file, a commit message or another tool.

| Tool | What it does |
| --- | --- |
| `pii_detect` | Reports categories and counts. Changes nothing. |
| `pii_redact` | Replaces values with `[TYPE]`. The usual one. |
| `pii_mask` | Replaces values with `*`, preserving length. |
| `pii_tokenize` | Replaces values with reversible tokens. |

The plugin also installs a skill, so Claude reaches for these on its own when
customer data is about to go somewhere it shouldn't — you do not have to ask
every time.

No tool ever returns a matched value to Claude. Results carry types, offsets and
counts, because the output goes into the conversation, and putting the PII back
there would undo the masking.

> The extension has six operations in total; the plugin exposes four. `block` is
> a policy outcome for a gateway to enforce, not something a model should ask
> for, and `audit` differs from `detect` only in a configuration detail that
> does not exist here. See [docs/operations.md](../../docs/operations.md).

## Install

**1. Install the Python side**, into whichever interpreter you'll point the
plugin at. It needs **Python 3.12 or newer** — the macOS system `python3` is 3.9
and will refuse.

```bash
python3.12 -m venv ~/.kryptos-pii && source ~/.kryptos-pii/bin/activate

pip install kryptos-pii-local mcp
kryptos-pii-model download          # the checkpoint, ~850 MB, once
```

That downloads [`sathvik-17/kryptos-pii`](https://huggingface.co/sathvik-17/kryptos-pii)
from the Hugging Face Hub. No account and no API key are needed for either step.

For hosted masking instead, `pip install kryptos-pii-client mcp` — no checkpoint
to download.

Note the interpreter you used (`~/.kryptos-pii/bin/python` above). Step 3 asks
for it, and pointing the plugin at the wrong one is the usual failure.

An air-gapped machine can install the same two wheels from the offline bundle on
the extension page with `pip install --find-links . kryptos-pii-local mcp`, and
copy the checkpoint in by hand to a directory named in `KRYPTOS_PII_MODEL_DIR`.

**2. Add the marketplace and install the plugin** in Claude Code:

```text
/plugin marketplace add Kryptos-Calfus/Kryptos-PII
/plugin install kryptos-pii@kryptos
```

From a local checkout, point at the directory instead — the repository root,
which is where the marketplace manifest lives:

```text
/plugin marketplace add ./
/plugin install kryptos-pii@kryptos
```

Both work from a terminal too, as `claude plugin marketplace add …` and
`claude plugin install …`.

**3. Answer the four configuration prompts** Claude Code shows when the plugin
is enabled:

| Option | What to put |
| --- | --- |
| Where masking runs | `local` (default) or `hosted` |
| Python interpreter | `python3`, or the full path to the venv python you installed into |
| Kryptos API key | hosted mode only |
| Kryptos URL | hosted mode only, if you self-host |

The interpreter is the one people get wrong. If `pip install` went into a
virtualenv, give the plugin that venv's `bin/python`, not bare `python3`.

## Claude Desktop

Claude Desktop does not install Claude Code plugins. Point it at the same server
by hand, in
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kryptos-pii": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["/absolute/path/to/pii/integrations/claude-plugin/server.py"],
      "env": { "KRYPTOS_PII_MODE": "local" }
    }
  }
}
```

## local or hosted

`local` is the default, and it is the right default for a tool whose whole job
is to stop data escaping: the detector runs in the server process, nothing goes
over the network, and no API key is used.

`hosted` sends the text to Kryptos instead, so the calls appear in your usage
and audit trail and you host no checkpoint. Every result names which mode
produced it, so you can always tell which one answered.

One consequence worth knowing: in hosted mode `pii_tokenize` returns tokens that
cannot be reversed, because the hosted service does not keep the mapping. The
round trip works only in local mode.

## When something is wrong

The server writes plain diagnostics to stderr, which Claude Code shows under
`/plugin`. The three that actually happen:

- **the 'mcp' package is not installed for this interpreter** — the Python
  interpreter option points somewhere without the packages. The message prints
  the interpreter it tried; `pip install` into that one, or change the option.
- **No PII detection checkpoint at …** — run `kryptos-pii-model download`.
- **hosted mode needs an API key** — fill in the API key option, or switch the
  mode option back to `local`.
