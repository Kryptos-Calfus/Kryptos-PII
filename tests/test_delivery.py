"""The things a customer downloads, and where the checkpoint comes from.

These never touch the network and never load the checkpoint. What they protect
is the promise: extension.yaml advertises SDKs, a plugin and documentation, and
every one of those paths has to exist and agree with itself. A manifest that
promises a file nobody shipped is a 404 on the extension page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "sdk" / "python-local")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from kryptos_pii import model
from kryptos_pii.engine import OPERATIONS

MANIFEST = yaml.safe_load((ROOT / "extension.yaml").read_text())
PLUGIN = ROOT / "integrations" / "claude-plugin"

REGEX = {"detector_mode": "regex"}


# --- where the checkpoint lives ------------------------------------------


def test_an_explicit_model_dir_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KRYPTOS_PII_MODEL_DIR", str(tmp_path / "somewhere"))
    assert model.model_dir() == tmp_path / "somewhere"


def test_a_checkout_uses_its_own_trained_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Someone working in the repository gets the checkpoint they just trained,
    without having to set anything."""
    monkeypatch.delenv("KRYPTOS_PII_MODEL_DIR", raising=False)
    if not model.has_weights(model.BUNDLED_DIR):
        pytest.skip("no checkpoint in this checkout")
    assert model.model_dir() == model.BUNDLED_DIR


def test_without_a_checkout_it_falls_back_to_the_download_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KRYPTOS_PII_MODEL_DIR", raising=False)
    monkeypatch.setattr(model, "BUNDLED_DIR", tmp_path / "not-a-checkout")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    where = model.model_dir()

    assert where == tmp_path / "cache" / "kryptos" / "pii" / "kryptos--laya-pii"
    assert not model.is_installed()  # the path is answerable before anything is there


def test_download_refuses_a_directory_without_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 'successful' download that produced no checkpoint is a failure, not a
    silently broken install discovered at the first request."""
    empty = tmp_path / "empty"
    empty.mkdir()

    def fake_snapshot(**kwargs: object) -> str:
        return str(empty)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", type(sys)("huggingface_hub")
    )
    sys.modules["huggingface_hub"].snapshot_download = fake_snapshot  # type: ignore[attr-defined]

    with pytest.raises(model.DownloadFailed, match="not a usable checkpoint"):
        model.download("kryptos/laya-pii", target=empty)


# --- the local SDK --------------------------------------------------------


def test_the_local_sdk_masks_without_a_key_or_a_network() -> None:
    import kryptos_pii_local as local

    r = local.redact("Email priya@acme.com, card 4111111111111111", **REGEX)

    assert r.text == "Email [EMAIL], card [PAYMENT_CARD]"
    assert r.decision == "redact"
    assert r.reason_codes == ["PII_EMAIL", "PII_PAYMENT_CARD"]
    # Findings locate the values without carrying them.
    assert [(f.type, f.start, f.end) for f in r.findings] == [("email", 6, 20), ("payment_card", 27, 43)]
    assert "priya@acme.com" not in json.dumps([f.__dict__ for f in r.findings])


def test_mask_preserves_length_and_redact_does_not() -> None:
    text = "Email priya@acme.com"

    assert local_module().mask(text, **REGEX).text == "Email " + "*" * len("priya@acme.com")
    assert local_module().redact(text, **REGEX).text == "Email [EMAIL]"


def test_tokenize_round_trips() -> None:
    local = local_module()
    r = local.tokenize("Email priya@acme.com about the refund", **REGEX)

    assert "priya@acme.com" not in (r.text or "")
    assert r.detokenize(r.text or "") == "Email priya@acme.com about the refund"


def test_detokenize_refuses_a_result_that_has_no_vault() -> None:
    local = local_module()
    r = local.redact("Email priya@acme.com", **REGEX)

    with pytest.raises(local.PIIError, match="no token mapping"):
        r.detokenize(r.text or "")


def test_protect_returns_the_string_and_leaves_clean_text_alone() -> None:
    local = local_module()

    assert local.protect("Email priya@acme.com", **REGEX) == "Email [EMAIL]"
    assert local.protect("nothing in here", **REGEX) == "nothing in here"


def test_a_missing_checkpoint_is_an_error_not_a_silent_regex_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole value of this extension is that it detects. Degrading quietly
    to a weaker detector is the one failure nobody would notice."""
    local = local_module()
    monkeypatch.setattr(local, "model_installed", lambda: False)
    monkeypatch.setattr(local, "model_path", lambda: "/nowhere")

    with pytest.raises(local.ModelMissing, match="kryptos-pii-model download"):
        local.redact("Email priya@acme.com")  # hybrid: needs the checkpoint

    # Asking for regex explicitly still works -- it is a choice, not a fallback.
    assert local.redact("Email priya@acme.com", **REGEX).text == "Email [EMAIL]"


def local_module():  # noqa: ANN201 - imported late so sys.path is set up first
    import kryptos_pii_local

    return kryptos_pii_local


# --- the package has to be installable ------------------------------------


def test_the_package_declares_a_build_backend_and_what_ships() -> None:
    """Without these, `pip install kryptos-pii` fails at build time with
    'Multiple top-level packages discovered in a flat-layout' -- this directory
    is a workspace, and the build cannot guess which part is the distribution."""
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert config["build-system"]["build-backend"]
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["kryptos_pii"]


def test_nothing_in_the_package_imports_from_outside_it() -> None:
    """The wheel ships ``kryptos_pii`` and nothing else.

    A runtime import of ``finetune`` resolves fine in a checkout and fails on
    every real install, which is how the detector shipped unusable: the span
    proposer lived in ``finetune/common.py``, beside a checkpoint that
    can never go in a wheel. It is ``kryptos_pii.candidates`` now, and this
    keeps it from drifting back.
    """
    import ast

    outside = {"finetune", "evals", "sdk", "integrations", "scripts", "tests"}
    offenders: list[str] = []

    for path in sorted((ROOT / "kryptos_pii").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots = [node.module.split(".")[0]]
            offenders += [f"{path.name} imports {root}" for root in roots if root in outside]

    assert not offenders, offenders


def test_the_local_sdk_depends_on_the_package_that_carries_the_detector() -> None:
    """kryptos-pii-local is a wrapper. If it stopped depending on kryptos-pii,
    it would install cleanly and then fail on first use."""
    import tomllib

    config = tomllib.loads((ROOT / "sdk" / "python-local" / "pyproject.toml").read_text())
    assert any(dep.startswith("kryptos-pii") for dep in config["project"]["dependencies"])


# --- what the manifest promises ------------------------------------------


def test_every_documented_operation_is_one_the_engine_runs() -> None:
    assert set(MANIFEST["operation_docs"]) == set(OPERATIONS)


def test_every_delivery_channel_points_at_documentation_that_exists() -> None:
    for channel in MANIFEST["delivery"]:
        if channel.get("docs"):
            assert (ROOT / channel["docs"]).exists(), f"{channel['id']} documents a missing file"


def test_an_artifacts_filename_matches_the_file_it_is_built_from() -> None:
    """The control plane serves `filename` and the build writes `source`. If
    they disagree, publishing uploads one name and customers download another."""
    for channel in MANIFEST["delivery"]:
        artifact = channel.get("artifact")
        if artifact:
            assert Path(artifact["source"]).name == artifact["filename"], channel["id"]


def test_the_claude_channel_names_the_plugin_the_plugin_calls_itself() -> None:
    """The marketplace entry the control plane generates uses this metadata. If
    it drifts from plugin.json, '/plugin install' names something that is not
    there."""
    channel = next(c for c in MANIFEST["delivery"] if c["integration"] == "claude")
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    marketplace = json.loads((PLUGIN / ".claude-plugin" / "marketplace.json").read_text())

    assert channel["metadata"]["plugin_name"] == plugin["name"]
    assert channel["metadata"]["marketplace_name"] == marketplace["name"]
    assert marketplace["plugins"][0]["name"] == plugin["name"]
    assert plugin["version"] == MANIFEST["version"]


def test_the_plugin_declares_every_option_its_server_config_references() -> None:
    """A ${user_config.x} that no option declares makes the plugin fail to load."""
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    declared = set(plugin["userConfig"])

    referenced = set()
    for server in plugin["mcpServers"].values():
        blob = json.dumps(server)
        for fragment in blob.split("${user_config.")[1:]:
            referenced.add(fragment.split("}")[0])

    assert referenced <= declared, f"undeclared options: {sorted(referenced - declared)}"


def test_the_plugin_server_is_where_the_manifest_says_it_is() -> None:
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    args = plugin["mcpServers"]["kryptos-pii"]["args"]

    for arg in args:
        if arg.startswith("${CLAUDE_PLUGIN_ROOT}"):
            relative = arg.removeprefix("${CLAUDE_PLUGIN_ROOT}/")
            assert (PLUGIN / relative).exists(), f"plugin.json points at a missing {relative}"
