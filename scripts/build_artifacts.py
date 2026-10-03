"""Build everything extension.yaml promises a customer can download.

    python scripts/build_artifacts.py

Reads the manifest, builds each delivery channel's artifact, writes it to the
path the manifest names, and prints the sha256 of each. Publishing then uploads
exactly these files:

    kryptos registry publish --manifest extension.yaml --endpoint URL

The manifest is the input, not a thing to keep in step by hand: add a channel
with an artifact and this builds it; the filename in the manifest is the
filename that ships. A mismatch is an error here rather than a 404 for a
customer.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "extension.yaml"

# Where each artifact comes from.
#
# The hosted client is one self-contained wheel: its only dependency is httpx,
# which is on PyPI, so a customer can install the file on its own.
#
# The local SDK cannot be. It depends on kryptos-pii, and until that is on an
# index, a lone kryptos_pii_local wheel installs to an error. So its artifact is
# a bundle of both wheels that 'pip install --find-links .' resolves offline.
HOSTED_SDK = ROOT / "sdk" / "python-hosted"
LOCAL_SDK = ROOT / "sdk" / "python-local"
EXTENSION = ROOT  # kryptos-pii itself: the detector the local SDK wraps
PLUGIN_DIR = ROOT / "integrations" / "claude-plugin"

# Never ship these into a plugin archive: caches, local virtualenvs and the
# .mcpb-cache Claude Code writes into a plugin root at runtime.
PLUGIN_EXCLUDE = {"__pycache__", ".venv", ".mcpb-cache", ".DS_Store", ".pytest_cache"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_wheel(package_dir: Path, out_dir: Path) -> Path:
    """Build one wheel into ``out_dir`` and return it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("*.whl"))

    for command in (
        ["uv", "build", "--wheel", "--out-dir", str(out_dir), str(package_dir)],
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(package_dir)],
    ):
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            break
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Building {package_dir.name} failed:\n{exc.stderr or exc.stdout}"
            ) from exc
    else:
        raise SystemExit("No build frontend. Install one with: pip install build (or use uv)")

    produced = sorted(set(out_dir.glob("*.whl")) - before)
    if not produced:
        # Rebuilding an unchanged package writes the same filename again.
        produced = sorted(out_dir.glob("*.whl"))
    if not produced:
        raise SystemExit(f"No wheel produced in {out_dir}")
    return produced[-1]


def copy_wheel(package_dir: Path, destination: Path) -> Path:
    """Build a wheel and put it exactly where the manifest says it lives."""
    staging = package_dir / "dist"
    if staging.exists():
        shutil.rmtree(staging)
    wheel = build_wheel(package_dir, staging)
    if wheel.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, destination)
    return destination


def build_offline_bundle(destination: Path) -> Path:
    """The local SDK plus the extension it wraps, as one installable zip.

    Everything needed from Kryptos is in the archive; the handful of ordinary
    dependencies (laya, and torch underneath it) still come from PyPI. The
    install is then one command against the unpacked directory, which is also
    exactly what an air-gapped site wants:

        pip install --find-links . kryptos-pii-local
    """
    staging = ROOT / "dist" / "_bundle"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    wheels = [build_wheel(EXTENSION, staging), build_wheel(LOCAL_SDK, staging)]

    (staging / "INSTALL.txt").write_text(
        "Kryptos PII Protection - local SDK\n"
        "==================================\n\n"
        "Needs Python 3.12 or newer.\n\n"
        "  python3.12 -m venv .venv\n"
        "  source .venv/bin/activate          # Windows: .venv\\Scripts\\activate\n"
        "  pip install --find-links . kryptos-pii-local\n"
        "  kryptos-pii-model download\n\n"
        "Then:\n\n"
        "  from kryptos_pii_local import redact\n"
        '  redact("Call Priya on 9812345678").text\n\n'
        "The wheels in this directory are everything Kryptos ships. The rest of\n"
        "the dependencies come from your package index as usual.\n"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    root = destination.stem
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging.iterdir()):
            archive.write(path, f"{root}/{path.name}")
    shutil.rmtree(staging)
    print(f"  bundled {', '.join(w.name for w in wheels)} + INSTALL.txt")
    return destination


def build_plugin_zip(destination: Path) -> Path:
    """Zip the Claude plugin with its root one level down.

    Claude Code accepts the plugin root at the top of the archive or one
    directory down; one directory down is what a person gets when they unzip it
    by hand, so that is what ships.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    root = destination.stem  # kryptos-pii-claude-plugin-0.3.0
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PLUGIN_DIR.rglob("*")):
            if any(part in PLUGIN_EXCLUDE for part in path.parts):
                continue
            if path.is_file():
                archive.write(path, f"{root}/{path.relative_to(PLUGIN_DIR)}")
    return destination


def main() -> int:
    manifest = yaml.safe_load(MANIFEST.read_text())
    channels = [c for c in manifest.get("delivery", []) if c.get("artifact")]
    if not channels:
        print("No delivery channel declares an artifact. Nothing to build.")
        return 0

    results: list[tuple[str, Path]] = []
    for channel in channels:
        artifact = channel["artifact"]
        artifact_id = artifact["id"]
        destination = ROOT / artifact["source"]

        if destination.name != artifact["filename"]:
            raise SystemExit(
                f"Channel '{channel['id']}': source ends in '{destination.name}' but "
                f"filename says '{artifact['filename']}'. They must agree."
            )

        print(f"building {artifact_id} -> {artifact['source']}")
        if artifact_id == "python-sdk-hosted":
            built = copy_wheel(HOSTED_SDK, destination)
        elif artifact_id == "python-sdk-local":
            built = build_offline_bundle(destination)
        elif artifact_id == "claude-plugin":
            built = build_plugin_zip(destination)
        else:
            raise SystemExit(
                f"No build rule for artifact '{artifact_id}'. Add one in main()."
            )
        results.append((artifact_id, built))

    print()
    for artifact_id, path in results:
        size = path.stat().st_size
        print(f"{artifact_id:22} {size / 1024:8.1f} KiB  sha256:{sha256_of(path)}")
    print("\nPublish with: kryptos registry publish --manifest extension.yaml --endpoint URL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
