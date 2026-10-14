"""HuggingFace weight-source resolution in runtime settings (no network, no weights)."""
from types import SimpleNamespace

import pytest

from tcr_workbench import runtime_settings as rs


def _args(**overrides):
    base = dict(config=None, decoder_dir="/decoder", python_executable="/decoder/bin/python",
                model="esmc-300m", device="cpu", precision=None, checkpoint=None, mlx_python=None,
                batch_size=None, token_budget=None, cache_bytes=None, timeout=None,
                weight_source="registry", hf_repo=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # Do not pick up a real ./.tcr-workbench.json from the working directory.
    monkeypatch.chdir(tmp_path)


def test_registry_is_the_default_weight_source():
    options = rs.resolve_settings(_args())
    assert options["weight_source"] == "registry"
    assert options["hf_repo"] is None


def test_huggingface_source_resolves_for_every_model():
    # The one published registry mirror carries every member, so each model resolves.
    for model in ("esmc-300m", "esmc-600m", "esmc-6b"):
        options = rs.resolve_settings(_args(weight_source="huggingface", model=model))
        assert options["weight_source"] == "huggingface"
    # An explicit OWNER/NAME override is preserved verbatim for later spec resolution.
    pinned = rs.resolve_settings(_args(weight_source="huggingface", hf_repo="me/custom"))
    assert pinned["hf_repo"] == "me/custom"


def test_huggingface_source_rejected_on_apple():
    with pytest.raises(ValueError, match="cpu or cuda only"):
        rs.resolve_settings(_args(weight_source="huggingface", device="apple", mlx_python="/mlx/bin/python"))


def test_huggingface_source_rejects_a_malformed_repo():
    with pytest.raises(ValueError, match="OWNER/NAME"):
        rs.resolve_settings(_args(weight_source="huggingface", hf_repo="not-a-repo"))


def test_unknown_weight_source_fails_closed():
    # The RuntimeSettings Literal rejects an unknown source before it is used.
    with pytest.raises(ValueError, match="registry.*huggingface"):
        rs.resolve_settings(_args(weight_source="s3"))
