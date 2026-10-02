"""Run the encoder-depth and context-size experiments at reduced scale.

    HF_HUB_OFFLINE=1 uv run python finetune/sweep.py context
    HF_HUB_OFFLINE=1 uv run python finetune/sweep.py layers

Each configuration trains from the same base checkpoint on the same data and is scored on
the same validation split, so the numbers are comparable to each other. They are NOT
comparable to a full run: the scale is cut deliberately so a sweep finishes in minutes.
Validation only - the test split is not touched here.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Logs only - the runs pass --no-save, so no 1.6 GB checkpoint is written per configuration.
LOGS = Path(os.environ.get("SWEEP_DIR", Path(tempfile.gettempdir()) / "laya-pii-sweep"))

GRIDS = {"context": ("--context", [60, 120, 200, 300]),
         "layers": ("--layers", [4, 6, 8, 12])}
FIXED = ["--synth", "2500", "--epochs", "1", "--no-sweep", "--no-hard", "--errors", "0", "--no-save"]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "context"
    flag, values = GRIDS[which]
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"logs: {LOGS}")
    results = []
    for v in values:
        print(f"\n===== {flag} {v} =====", flush=True)
        cmd = [sys.executable, str(ROOT / "finetune" / "train.py"), flag, str(v)] + FIXED
        log = LOGS / f"{which}-{v}.log"
        with log.open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
        text = log.read_text()
        val = [l for l in text.splitlines() if "VALIDATION" in l]
        test = [l for l in text.splitlines() if "value  P/R/F1" in l]
        results.append((v, val[-1].strip() if val else "FAILED", test[-1].strip() if test else ""))
        print(results[-1][1], flush=True)

    print(f"\n===== {which} sweep summary (validation picks the winner) =====")
    for v, val, test in results:
        print(f"  {flag} {v:>4}: {val}")
    print("\nReduced scale (2500 synth, 1 epoch): use this to rank configurations, "
          "then re-run the winner at full scale.")


if __name__ == "__main__":
    main()
