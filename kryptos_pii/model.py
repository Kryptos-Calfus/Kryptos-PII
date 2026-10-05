"""Where the checkpoint lives and how it gets there.

The detector is a fine-tuned LAYA classifier, and the weights are ~850 MB. They
are too large to ship inside a wheel, so the local SDK installs in seconds and
the checkpoint arrives in a second, explicit step:

    kryptos-pii-model download

Three places are searched, in order, and the first that holds weights wins:

    1. ``KRYPTOS_PII_MODEL_DIR``   -- an explicit path, which always wins
    2. ``<repo>/finetune/laya-pii`` -- the training output, when running in a checkout
    3. ``~/.cache/kryptos/pii/<repo>`` -- what ``download`` writes

The order matters for the two people who use this. Someone working in the
repository gets the checkpoint they just trained, without setting anything.
Someone who ran ``pip install kryptos-pii-local`` gets the published one. Both
can override with the environment variable and neither has to know the other
case exists.

Nothing here is on the request path. The detector asks for a directory once per
process; this module's only job is to answer where it is and, when asked, to
fetch it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The published checkpoint. Overridable so a deployment can pin its own
# retrained copy without a code change -- the detector's reported F1 belongs to
# a specific checkpoint, and a team that retrains needs somewhere to put theirs.
DEFAULT_REPO = os.environ.get("KRYPTOS_PII_MODEL_REPO", "sathvik-17/kryptos-pii")
DEFAULT_REVISION = os.environ.get("KRYPTOS_PII_MODEL_REVISION") or None

# The file whose presence means "this directory holds a usable checkpoint".
WEIGHTS = "model.safetensors"

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DIR = REPO_ROOT / "finetune" / "laya-pii"


def cache_root() -> Path:
    """Where downloaded checkpoints are kept, honouring XDG on Linux."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "kryptos" / "pii"


def cache_dir_for(repo_id: str = DEFAULT_REPO) -> Path:
    return cache_root() / repo_id.replace("/", "--")


def has_weights(path: Path | str) -> bool:
    return (Path(path) / WEIGHTS).exists()


def model_dir(repo_id: str = DEFAULT_REPO) -> Path:
    """The directory the detector should load from.

    Returns the cache path even when nothing has been downloaded yet, so the
    caller has somewhere to name in an error message. Use :func:`is_installed`
    to ask whether the weights are actually there.
    """
    explicit = os.environ.get("KRYPTOS_PII_MODEL_DIR")
    if explicit:
        return Path(explicit).expanduser()
    if has_weights(BUNDLED_DIR):
        return BUNDLED_DIR
    return cache_dir_for(repo_id)


def is_installed(repo_id: str = DEFAULT_REPO) -> bool:
    return has_weights(model_dir(repo_id))


class DownloadFailed(RuntimeError):
    """The checkpoint could not be fetched."""


def download(
    repo_id: str = DEFAULT_REPO,
    *,
    revision: str | None = DEFAULT_REVISION,
    target: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Fetch the checkpoint from the Hugging Face Hub into the local cache.

    Resumable and content-addressed by the hub client, so an interrupted
    download continues rather than restarting, and a second call on an
    up-to-date cache does no network work at all.
    """
    destination = Path(target).expanduser() if target else cache_dir_for(repo_id)
    if has_weights(destination) and not force:
        return destination

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - import guard
        raise DownloadFailed(
            "The checkpoint downloader needs huggingface_hub. "
            "Install it with: pip install 'kryptos-pii-local[download]'"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            # The detector loads from a plain directory, so materialise real
            # files rather than a tree of symlinks into the hub cache.
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001 - network, auth and 404 are one condition for the caller
        raise DownloadFailed(
            f"Could not download '{repo_id}': {exc}\n"
            "If the repository is private, authenticate first with 'huggingface-cli login'."
        ) from exc

    if not has_weights(destination):
        raise DownloadFailed(
            f"'{repo_id}' downloaded to {destination} but contains no {WEIGHTS}. "
            "That is not a usable checkpoint."
        )
    return destination


def main(argv: list[str] | None = None) -> int:
    """``kryptos-pii-model``: where is the checkpoint, and fetch it."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kryptos-pii-model",
        description="Manage the PII detection checkpoint used by local execution.",
    )
    sub = parser.add_subparsers(dest="command")

    get = sub.add_parser("download", help="Fetch the checkpoint into the local cache")
    get.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face repo id (default: {DEFAULT_REPO})")
    get.add_argument("--revision", default=DEFAULT_REVISION, help="Branch, tag or commit to pin")
    get.add_argument("--target", help="Download somewhere other than the cache")
    get.add_argument("--force", action="store_true", help="Re-download even if the cache is populated")

    sub.add_parser("status", help="Report whether a checkpoint is installed")
    sub.add_parser("path", help="Print the directory the detector will load from")

    args = parser.parse_args(argv)
    command = args.command or "status"

    if command == "path":
        print(model_dir())
        return 0

    if command == "status":
        where = model_dir()
        if has_weights(where):
            size = (where / WEIGHTS).stat().st_size / 1e9
            print(f"installed: {where} ({size:.1f} GB)")
            return 0
        print(f"not installed. The detector would load from: {where}")
        print("Fetch it with: kryptos-pii-model download")
        return 1

    print(f"Downloading {args.repo} -> {args.target or cache_dir_for(args.repo)}")
    try:
        where = download(args.repo, revision=args.revision, target=args.target, force=args.force)
    except DownloadFailed as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Ready: {where}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
