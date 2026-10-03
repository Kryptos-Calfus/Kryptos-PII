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


# --- scoring large documents ---------------------------------------------


def test_candidate_count_grows_with_document_length() -> None:
    """The reason scoring has to be bounded at all.

    Candidates scale with input, so an unbounded batch makes peak memory a
    function of document size. Spelled out as a test because it is the premise
    the batching in ``detector.score`` rests on.
    """
    from kryptos_pii.candidates import pieces

    unit = (
        "Jennifer Lopez-Garcia called from (415) 555-0132 about order #88213. "
        "Her SSN is 536-22-8841 and she lives at 1600 Pennsylvania Avenue. "
    )
    small = len(pieces(unit))
    large = len(pieces(unit * 50))

    assert small > 0
    # Linear, not bounded: 50x the text proposes roughly 50x the candidates.
    assert large > small * 25


def test_batching_is_bounded_and_covers_every_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every candidate is scored exactly once, in batches no larger than the cap.

    Runs against a stub agent rather than the checkpoint: the property under
    test is how spans are grouped, and that should be verifiable without a
    850 MB download or a minute of inference.
    """
    from kryptos_pii import detector

    batches: list[int] = []

    class StubAgent:
        def predict_batch(self, states, question):  # noqa: ANN001, ANN202
            batches.append(len(states))
            return [{"answers": {"pii": {"noul": 0.9}}} for _ in states]

    checkpoint = detector._Checkpoint(ROOT / "nowhere", 120)
    checkpoint._agent = StubAgent()
    monkeypatch.setattr(detector, "MAX_BATCH_SPANS", 10)

    spans = [(i, i + 3) for i in range(0, 250, 5)]
    scores = checkpoint.score("x" * 300, spans)

    assert len(scores) == len(spans), "every span gets exactly one score"
    assert batches, "the agent was actually called"
    assert max(batches) <= 10, f"a batch exceeded the cap: {batches}"
    assert sum(batches) == len(spans), "spans were scored once each, not dropped or repeated"


def test_scoring_no_spans_never_loads_the_model() -> None:
    """Text with no candidates must not pay for a checkpoint it does not need."""
    from kryptos_pii import detector

    checkpoint = detector._Checkpoint(ROOT / "definitely-not-a-checkpoint", 120)

    assert checkpoint.score("nothing here", []) == []


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


# --- the Nemotron label map ----------------------------------------------


def _nemotron():  # noqa: ANN202 - finetune/ is a flat module dir, imported late
    sys.path.insert(0, str(ROOT / "finetune"))
    import nemotron

    return nemotron


def test_every_label_lands_in_exactly_one_bucket() -> None:
    """A label in two buckets trains as both positive and negative."""
    n = _nemotron()
    buckets = [set(n.ADOPT), set(n.NEGATIVE), n.SENSITIVE_LABELS, set(n.SKIP)]

    for i, a in enumerate(buckets):
        for b in buckets[i + 1 :]:
            assert not (a & b), f"label in two buckets: {sorted(a & b)}"


def test_adopted_labels_map_onto_types_the_detector_uses() -> None:
    """A typo here would train a category nothing downstream can act on."""
    n = _nemotron()
    known = {
        "person_name", "email", "phone", "address", "date_of_birth",
        "government_id", "financial", "payment_card", "secret",
        "network_address", "personal_data",
    }

    unknown = {t for t in n.ADOPT.values() if t not in known}
    assert not unknown, f"ADOPT maps to types the taxonomy does not have: {unknown}"


def test_the_categories_our_detector_confuses_are_trained_as_negatives() -> None:
    """The measured false positives are company and place names. If these drift
    to the positive side the retrain makes precision worse, not better."""
    n = _nemotron()

    for label in ("company_name", "city", "state", "country", "occupation"):
        assert label in n.NEGATIVE, f"{label} must stay a hard negative"
        assert label not in n.ADOPT


def test_conversion_keeps_offsets_that_resolve_and_drops_the_rest() -> None:
    """Offsets are trusted only when they still point at the stated value."""
    n = _nemotron()
    text = "Call Priya at ACME Corp on 555-0100."

    good = n.convert_row({
        "text": text,
        "spans": [
            {"start": 5, "end": 10, "text": "Priya", "label": "first_name"},
            {"start": 14, "end": 23, "text": "ACME Corp", "label": "company_name"},
            {"start": 999, "end": 1004, "text": "bogus", "label": "first_name"},
            {"start": 0, "end": 4, "text": "WRONG", "label": "first_name"},
        ],
    })

    assert good["pii_spans"] == [(5, 10, "person_name")], "only resolving positives survive"
    assert good["keep"] == ["ACME Corp"], "the company name becomes a hard negative"


def test_a_row_with_nothing_mappable_is_dropped() -> None:
    n = _nemotron()

    assert n.convert_row({"text": "Posted at 10:30 AM.",
                          "spans": [{"start": 10, "end": 18, "text": "10:30 AM", "label": "time"}]}) is None
    assert n.convert_row({"text": "", "spans": []}) is None
