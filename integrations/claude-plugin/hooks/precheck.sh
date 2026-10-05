#!/bin/sh
# Find the interpreter that has the Kryptos detector, and run the pre-model
# check with it.
#
# This wrapper exists because a Claude Code hook is a command line, not a
# Python entry point, and the interpreter holding kryptos-pii-local is rarely
# the one on PATH -- on macOS the system python3 is 3.9 and the SDK needs 3.12.
#
# Order, first that works wins:
#
#   1. $KRYPTOS_PII_PYTHON        an explicit choice, which always wins
#   2. ~/.kryptos-pii/bin/python  the venv the plugin README tells people to make
#   3. python3                    for the case where it is already the right one
#
# "Works" means the interpreter can import the SDK, not merely that the file
# exists. A venv that was deleted or half-built should fall through to the next
# candidate rather than produce a confusing import error.
#
# Finding nothing is a block, not a warning: exit 2 stops the prompt, and the
# stderr text below is what the user is shown. Failing open here would mean an
# unprotected prompt going out while the user believed otherwise.

set -u

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CHECK="$HERE/precheck.py"

for candidate in \
    "${KRYPTOS_PII_PYTHON:-}" \
    "$HOME/.kryptos-pii/bin/python" \
    python3
do
    [ -n "$candidate" ] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import kryptos_pii_local' >/dev/null 2>&1; then
        exec "$candidate" "$CHECK"
    fi
done

cat >&2 <<'MESSAGE'
kryptos-pii: this prompt was not sent, because PII protection could not run.

  No Python interpreter on this machine has kryptos-pii-local installed.
  Tried: $KRYPTOS_PII_PYTHON, ~/.kryptos-pii/bin/python, python3

  Fix it with:

    python3.12 -m venv ~/.kryptos-pii
    ~/.kryptos-pii/bin/pip install kryptos-pii-local
    ~/.kryptos-pii/bin/kryptos-pii-model download

  Or point the hook at an interpreter that already has it:

    export KRYPTOS_PII_PYTHON=/path/to/bin/python
MESSAGE
exit 2
