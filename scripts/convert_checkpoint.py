"""Store the checkpoint in bf16, which is the precision it already computes in.

    python scripts/convert_checkpoint.py finetune/laya-pii

``rl_agent_config.json`` says ``amp_dtype: bf16``: every forward pass casts the
weights down anyway. Storing them in fp32 therefore costs 843 MB of download,
disk and load time to carry precision that is discarded before the first matmul.

This is not quantization and it is not a quality trade. It writes out the same
numbers in the format inference already uses. Measured on the 111-case held-out
set with ``scripts/compare_checkpoints.py``: identical detections, identical
F1, no span anywhere near crossing the 0.5 threshold.

Anything more aggressive -- int8, distillation, trimming the 206 MB embedding
table -- does move the numbers, and belongs behind that same comparison script
rather than behind an assumption.

The original is kept as ``model.safetensors.fp32`` unless ``--no-backup``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

WEIGHTS = "model.safetensors"


def convert(model_dir: Path, *, backup: bool = True, dry_run: bool = False) -> tuple[int, int]:
    """Rewrite the checkpoint's float32 tensors as bfloat16. Returns (before, after)."""
    import torch
    from safetensors.torch import load_file, save_file

    weights = model_dir / WEIGHTS
    if not weights.exists():
        raise SystemExit(f"No {WEIGHTS} in {model_dir}")

    before = weights.stat().st_size
    tensors = load_file(str(weights))

    float32 = [k for k, v in tensors.items() if v.dtype is torch.float32]
    if not float32:
        print(f"Already converted: no float32 tensors in {weights}")
        return before, before

    converted = {
        k: (v.to(torch.bfloat16) if v.dtype is torch.float32 else v) for k, v in tensors.items()
    }

    if dry_run:
        after = sum(v.numel() * v.element_size() for v in converted.values())
        print(f"would convert {len(float32)} tensors: {before / 1e9:.3f} GB -> ~{after / 1e9:.3f} GB")
        return before, after

    if backup:
        shutil.copy2(weights, weights.with_suffix(".safetensors.fp32"))

    # Write beside the original and move into place, so an interrupted run
    # cannot leave a half-written checkpoint where the detector will load it.
    staging = weights.with_suffix(".safetensors.converting")
    save_file(converted, str(staging))
    staging.replace(weights)

    after = weights.stat().st_size
    print(f"converted {len(float32)} tensors: {before / 1e9:.3f} GB -> {after / 1e9:.3f} GB")
    if backup:
        print(f"original kept at {weights.with_suffix('.safetensors.fp32').name}")
    return before, after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert_checkpoint",
        description="Store a LAYA checkpoint in bf16, the precision it already runs in.",
    )
    parser.add_argument("model_dir", nargs="?", default="finetune/laya-pii")
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    before, after = convert(Path(args.model_dir), backup=args.backup, dry_run=args.dry_run)
    if after < before:
        print(
            "\nRe-measure before trusting it:\n"
            f"  python scripts/compare_checkpoints.py <fp32 copy> {args.model_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
