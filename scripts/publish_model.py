"""Upload the checkpoint to the Hugging Face Hub.

    huggingface-cli login
    python scripts/publish_model.py --repo sathvik-17/kryptos-pii

Dry run first; it is the default:

    python scripts/publish_model.py --repo sathvik-17/kryptos-pii --dry-run

Two things this refuses to do, because both are hard to undo once pushed:

* Upload fp32 weights. The Hub keeps LFS history, so a 1.6 GB blob stays in the
  repository forever even after you replace it. Convert first.
* Upload without a model card. A detector published with no stated limits is
  one someone will trust further than it deserves.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "finetune" / "laya-pii"

# Everything the detector loads, and nothing else. No optimiser state, no
# training logs, no dataset.
UPLOAD = ["model.safetensors", "rl_agent_config.json", "README.md", "encoder", "tokenizer"]


def dtypes_in(weights: Path) -> set[str]:
    """Read the safetensors header without loading ~1 GB of weights."""
    with weights.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return {spec["dtype"] for spec in header.values()}


def check(model_dir: Path) -> None:
    missing = [name for name in UPLOAD if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(f"{model_dir} is missing: {', '.join(missing)}")

    dtypes = dtypes_in(model_dir / "model.safetensors")
    if "F32" in dtypes:
        raise SystemExit(
            "This checkpoint holds float32 tensors, and the Hub keeps LFS history "
            "forever.\nConvert first:  python scripts/convert_checkpoint.py "
            f"{model_dir.relative_to(ROOT) if model_dir.is_relative_to(ROOT) else model_dir}\n"
            "Then confirm nothing moved:  python scripts/compare_checkpoints.py <fp32 copy> "
            "<converted>"
        )

    card = (model_dir / "README.md").read_text()
    if "---" not in card.split("\n")[0]:
        raise SystemExit("README.md has no YAML front matter; the Hub needs it for the model card.")
    for required in ("## Limits", "license:"):
        if required not in card:
            raise SystemExit(f"README.md is missing {required!r}. Publish the limits with the weights.")

    size = (model_dir / "model.safetensors").stat().st_size
    print(f"checkpoint {size / 1e6:.0f} MB, dtypes {sorted(dtypes)}, model card present")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="publish_model")
    parser.add_argument("--repo", required=True, help="Hub repo id, e.g. sathvik-17/kryptos-pii")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--revision", help="Branch to push to; defaults to main")
    parser.add_argument("--private", action="store_true", help="Create the repo private")
    parser.add_argument(
        "--push", action="store_true", help="Actually upload. Without it this only checks."
    )
    args = parser.parse_args(argv)

    check(args.model_dir)

    if not args.push:
        print(f"\nDry run. Would upload {UPLOAD} to {args.repo}.")
        print("Add --push to do it.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(args.model_dir),
        revision=args.revision,
        allow_patterns=[f"{name}*" for name in UPLOAD] + [f"{name}/**" for name in UPLOAD],
        commit_message="Publish laya-pii detector (bf16)",
    )
    print(f"\nPublished https://huggingface.co/{args.repo}")
    print("Verify from a clean machine:")
    print(f"  KRYPTOS_PII_MODEL_REPO={args.repo} kryptos-pii-model download")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
