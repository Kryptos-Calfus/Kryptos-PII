"""Does a changed checkpoint decide anything differently?

    python scripts/compare_checkpoints.py finetune/laya-pii.fp32 finetune/laya-pii

Runs both checkpoints over the held-out set and reports precision, recall, F1,
and -- the part that actually matters -- whether the span sets are identical and
whether any span crossed the decision threshold.

Aggregate F1 can sit still while individual decisions churn underneath it, so a
matching F1 is reported but is not the test. Two checkpoints are interchangeable
when they mask the same characters.

Use this before publishing any checkpoint change: bf16, quantization, a retrain.
It is the difference between "the numbers look fine" and knowing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "finetune" / "test_set.json"

# Each checkpoint is scored in its own process: the detector caches a loaded
# checkpoint per directory for the life of the process, and two ~1 GB models
# resident at once is how a laptop starts swapping mid-measurement.
WORKER = """
import json, os, sys
sys.path.insert(0, %(root)r)
os.environ["KRYPTOS_PII_MODEL_DIR"] = sys.argv[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from kryptos_pii.detector import detect
from kryptos_pii.candidates import spans_of

cases = json.load(open(%(test_set)r))
threshold = float(sys.argv[2])
tp = fp = fn = 0
scores = {}
for case in cases:
    text = case["text"]
    truth = set(spans_of(text, case["pii"]))
    got = set()
    for d in detect(text, detector_mode="hybrid", threshold=threshold):
        got.add((d.start, d.end))
        scores["%%s|%%d|%%d" %% (case["source"], d.start, d.end)] = round(d.confidence, 6)
    matched = set()
    for g in got:
        hit = next((t for t in truth if not (g[1] <= t[0] or t[1] <= g[0])), None)
        if hit:
            tp += 1
            matched.add(hit)
        else:
            fp += 1
    fn += len(truth - matched)

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
print(json.dumps({"tp": tp, "fp": fp, "fn": fn, "precision": precision,
                  "recall": recall, "f1": f1, "scores": scores}))
"""


def score(model_dir: Path, threshold: float) -> dict:
    source = WORKER % {"root": str(ROOT), "test_set": str(TEST_SET)}
    done = subprocess.run(
        [sys.executable, "-c", source, str(model_dir.resolve()), str(threshold)],
        capture_output=True,
        text=True,
        check=False,  # the failure is reported with the worker's stderr below
        cwd=ROOT,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    if done.returncode != 0:
        raise SystemExit(f"Scoring {model_dir} failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_checkpoints")
    parser.add_argument("baseline", type=Path, help="the checkpoint you trust")
    parser.add_argument("candidate", type=Path, help="the one you changed")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    print(f"scoring baseline  {args.baseline}")
    a = score(args.baseline, args.threshold)
    print(f"scoring candidate {args.candidate}\n")
    b = score(args.candidate, args.threshold)

    print(f"{'':11}{'tp':>5}{'fp':>5}{'fn':>5}{'prec':>9}{'recall':>9}{'f1':>9}")
    for name, d in (("baseline", a), ("candidate", b)):
        print(
            f"{name:11}{d['tp']:5}{d['fp']:5}{d['fn']:5}"
            f"{d['precision']:9.4f}{d['recall']:9.4f}{d['f1']:9.4f}"
        )

    sa, sb = a["scores"], b["scores"]
    same = set(sa) == set(sb)
    print(f"\nspans detected   baseline {len(sa)}, candidate {len(sb)}")
    print(f"identical spans  {same}")

    shared = set(sa) & set(sb)
    crossed = [k for k in shared if min(sa[k], sb[k]) < args.threshold <= max(sa[k], sb[k])]
    if shared:
        deltas = sorted(abs(sa[k] - sb[k]) for k in shared)
        print(f"confidence delta max {deltas[-1]:.5f}, median {deltas[len(deltas) // 2]:.5f}")
    print(f"crossed {args.threshold}   {len(crossed)}")

    for label, missing in (("only baseline", set(sa) - set(sb)), ("only candidate", set(sb) - set(sa))):
        if missing:
            print(f"{label}: {sorted(missing)[:5]}")

    interchangeable = same and not crossed
    print(
        "\nVERDICT: interchangeable -- same characters masked."
        if interchangeable
        else "\nVERDICT: decisions changed. Do not publish this as a drop-in replacement."
    )
    return 0 if interchangeable else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
