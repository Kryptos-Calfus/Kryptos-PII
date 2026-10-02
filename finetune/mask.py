"""Mask one piece of text with the fine-tuned model and show why it decided that way.

    HF_HUB_OFFLINE=1 uv run python finetune/mask.py "Call Priya on 9812345678"
    echo "..." | HF_HUB_OFFLINE=1 uv run python finetune/mask.py
    HF_HUB_OFFLINE=1 uv run python finetune/mask.py            # interactive, one text per line

Every candidate is listed with P(PII), so a wrong output tells you which half is at fault:
a value missing from the candidate list is an extractor problem, a value listed with the
wrong probability is a model problem.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "finetune"), str(ROOT / "evals")]

import laya  # noqa: E402

import common  # noqa: E402
from common import pieces  # noqa: E402
from train import OUT_DIR, make_scorer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("text", nargs="*", help="text to mask; omit to read stdin or go interactive")
p.add_argument("--checkpoint", default=str(OUT_DIR))
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument("--context", type=int, default=120, help="must match what the checkpoint trained with")
p.add_argument("--quiet", action="store_true", help="print only the masked text")
args = p.parse_args()

common.set_context_chars(args.context)
if not (Path(args.checkpoint) / "model.safetensors").exists():
    sys.exit(f"No checkpoint at {args.checkpoint}. Run finetune/train.py first.")

agent = laya.load(args.checkpoint)
score = make_scorer(agent)


def run(text):
    t0 = time.perf_counter()
    ps = pieces(text)
    probs = score(text, ps)
    out, cur = [], 0
    for (s, e), prob in zip(ps, probs):
        if prob >= args.threshold:
            out.append(text[cur:s]); out.append("[PII]"); cur = e
    masked = "".join(out) + text[cur:]
    ms = (time.perf_counter() - t0) * 1000

    if args.quiet:
        print(masked)
        return
    print(f"\n  in  {text}")
    print(f"  out {masked}")
    print(f"  {len(ps)} candidates, {ms:.0f} ms")
    for (s, e), prob in zip(ps, probs):
        mark = "MASK" if prob >= args.threshold else "keep"
        print(f"    {mark}  {prob:6.3f}  {text[s:e]!r}")


if args.text:
    run(" ".join(args.text))
elif not sys.stdin.isatty():
    for line in sys.stdin:
        if line.strip():
            run(line.rstrip("\n"))
else:
    print(f"checkpoint {args.checkpoint} | threshold {args.threshold} | Ctrl-D to quit")
    for line in sys.stdin:
        if line.strip():
            run(line.rstrip("\n"))
