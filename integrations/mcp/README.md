# PII Protection for Claude (MCP)

Gives Claude four tools so it can strip personal information before text reaches
a prompt, a file, a commit message or another tool.

| Tool | What it does |
| ---- | ------------ |
| `pii_detect` | Reports categories and counts. Changes nothing. |
| `pii_redact` | Replaces values with `[TYPE]`. The usual one. |
| `pii_mask` | Replaces values with `*`, preserving length. |
| `pii_tokenize` | Replaces values with reversible tokens. |

No tool ever returns a matched value to the model. The result carries types,
offsets and counts, because putting the PII back into the transcript would undo
the masking Claude just asked for.

## Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kryptos-pii": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/pii", "python", "integrations/mcp/server.py"],
      "env": { "KRYPTOS_PII_MODE": "local", "HF_HUB_OFFLINE": "1" }
    }
  }
}
```

## Claude Code

```bash
claude mcp add kryptos-pii -- uv run --directory /absolute/path/to/pii python integrations/mcp/server.py
```

## Local or hosted

`KRYPTOS_PII_MODE=local` (the default) runs the detector in-process: nothing
leaves the machine and no API key is used. This is the right default for a tool
whose whole job is to stop data escaping.

`KRYPTOS_PII_MODE=hosted` with `KRYPTOS_API_KEY` sends the text to Kryptos
instead, so the calls appear in your usage and audit trail. Every result names
which mode produced it.

## Install

```bash
uv sync --extra mcp
```

Local mode also needs the checkpoint at `finetune/laya-pii` (or
`KRYPTOS_PII_MODEL_DIR`). Without it the server starts but every call returns an
error rather than quietly dropping to regex.
