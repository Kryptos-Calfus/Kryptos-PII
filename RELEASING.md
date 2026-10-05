# Releasing

Two things ship: the Python packages to PyPI, and the checkpoint to the Hugging
Face Hub. Do the checkpoint first — the SDK is inert without it, and a working
`pip install` that then fails at `kryptos-pii-model download` is worse than no
install at all.

Both are effectively irreversible. PyPI will not let you re-upload a version,
and the Hub keeps LFS history, so a 1.6 GB blob pushed once stays in the
repository forever. Everything below is arranged so the irreversible step is
last and boring.

## Names

Checked available on 2026-10-03, not yet reserved:

| | |
| --- | --- |
| PyPI | `kryptos-pii`, `kryptos-pii-local`, `kryptos-pii-client`, `kryptos` |
| Hub | `kryptos/laya-pii` (the `kryptos` org does not exist yet) |

Re-checked 2026-10-05: all four PyPI names still return 404, and the Hub org is
still unregistered. The names are the perishable part. Reserving them costs one
upload each.

**Create the Hub org before step 1.** `DEFAULT_REPO` in `kryptos_pii/model.py`
is `kryptos/laya-pii`; push without the org and the weights land under your
personal namespace while every installed copy of `kryptos-pii-model download`
keeps asking for an id that does not exist.

## 1. The checkpoint

```bash
# Convert, if it is still fp32. Inference already runs in bf16, so this is
# removing waste rather than trading quality.
python scripts/convert_checkpoint.py finetune/laya-pii

# Prove it decides identically. Exits non-zero if anything moved.
python scripts/compare_checkpoints.py <fp32 copy> finetune/laya-pii

# Check the upload without doing it. Refuses fp32 weights and a missing card.
python scripts/publish_model.py --repo kryptos/laya-pii

huggingface-cli login
python scripts/publish_model.py --repo kryptos/laya-pii --push
```

Then verify as a stranger would, on a machine with no checkpoint and no
`KRYPTOS_PII_MODEL_DIR` set:

```bash
kryptos-pii-model download && kryptos-pii-model status
```

If that works, change the default in `kryptos_pii/model.py` (`DEFAULT_REPO`) to
the published id, and drop the "not on the Hub yet" caveats from the READMEs.

## 2. The packages

Three distributions, in dependency order — `kryptos-pii-local` is unusable until
`kryptos-pii` is on the index.

Build them into a directory of their own. `scripts/build_artifacts.py` is for
the manifest's download channels, not for PyPI: it stages the `kryptos-pii` and
`kryptos-pii-local` wheels inside `dist/_bundle/`, zips them into the offline
bundle and deletes the staging directory, so after it runs there is no loose
wheel for twine to check.

```bash
rm -rf dist/pypi
for pkg in . sdk/python-hosted sdk/python-local; do
  uv build --wheel --sdist --out-dir dist/pypi "$pkg"
done

python -m twine check dist/pypi/*      # twine is in the dev group
```

TestPyPI first. It is free, it catches a broken README or classifier, and its
namespace is separate so nothing you do there spends a real name:

```bash
python -m twine upload --repository testpypi dist/pypi/*
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple kryptos-pii
```

Then the real index, in this order — uploading `kryptos-pii-local` before
`kryptos-pii` publishes a package that cannot resolve:

```bash
python -m twine upload dist/pypi/kryptos_pii-0.3.0*
python -m twine upload dist/pypi/kryptos_pii_client-0.3.0*
python -m twine upload dist/pypi/kryptos_pii_local-0.3.0*
```

Authenticate with an API token, not a password: `__token__` as the username and
a `pypi-…` token as the password, or put it in `~/.pypirc`. Scope the first
token to the whole account, because a project-scoped token cannot create a
project that does not exist yet; replace it with per-project tokens afterwards.

Check from a clean 3.12 environment on a machine that has never seen this repo:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install kryptos-pii-local && kryptos-pii-model download
python -c "from kryptos_pii_local import redact; print(redact('Call Priya on 9812345678').text)"
```

## 3. Simplify the instructions

Publishing is only finished when the docs stop apologising for it. Once both are
live, in `extension.yaml` and the four READMEs:

- `pip install --find-links . kryptos-pii-local` becomes `pip install kryptos-pii-local`
- the "Not on PyPI yet" notes come out
- the offline bundle stays, demoted to what it is: the air-gapped path

Keep the Python version note. 3.12+ for anything local is still true, and the
macOS system `python3` is still 3.9.

## 4. Republish the extension

```bash
python scripts/build_artifacts.py
kryptos registry publish --manifest extension.yaml --endpoint <runtime url>
```

A published version is immutable, so a changed manifest needs a new version.
Bump `version:` in `extension.yaml`, the three `pyproject.toml` files, both SDK
`__version__`s, `kryptos_pii/engine.py`, and the plugin's `plugin.json` — the
test suite checks they agree.

## Before any of it

```bash
pytest                                   # 95 tests, no checkpoint needed
python scripts/build_artifacts.py        # everything the manifest promises
claude plugin validate ./integrations/claude-plugin
```
