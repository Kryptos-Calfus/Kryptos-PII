"""Start a training run that outlives the shell that started it.

    python finetune/launch.py --out finetune/runs/nemotron-1k -- \
        --base finetune/laya-pii --extra finetune/nemotron_1k.json --synth 1500

A run started with plain nohup from a tool call died with its session and,
because train.py only wrote a checkpoint at the very end, took two hours of
work with it. This double-forks and calls setsid, so the trainer leaves the
session's process group entirely and a restart cannot reap it. Output goes to a
log beside the checkpoint rather than into a temp directory that gets cleaned.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(prog="launch")
    parser.add_argument("--out", required=True, help="checkpoint directory; the log sits beside it")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("passthrough", nargs=argparse.REMAINDER,
                        help="arguments after -- go to train.py")
    args = parser.parse_args()

    extra = args.passthrough
    if extra and extra[0] == "--":
        extra = extra[1:]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")

    command = [args.python, str(ROOT / "finetune" / "train.py"), "--out", str(out), *extra]

    # First fork: the parent returns to the shell immediately.
    if os.fork() != 0:
        print(f"launched: {' '.join(command)}")
        print(f"log: {log}")
        return 0

    # Child: leave the session's process group, then fork again so the trainer
    # is not a session leader and can never reacquire a controlling terminal.
    os.setsid()
    if os.fork() != 0:
        os._exit(0)

    with open(log, "wb", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    os.chdir(ROOT)
    os.execve(command[0], command, env)
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
