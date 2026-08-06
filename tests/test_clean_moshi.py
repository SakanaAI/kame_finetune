import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

import tools.clean_moshi as clean_moshi
from models.moshi_for_finetuning import MoshiForFinetuning


class DummyEmbedding(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4, 3), value))


class DummyConvertibleLM(nn.Module):
    materialize_oracle_embedding_for_checkpoint_ = (
        MoshiForFinetuning.materialize_oracle_embedding_for_checkpoint_
    )

    def __init__(self):
        super().__init__()
        self.text_emb = DummyEmbedding(1.0)
        self.oracle_emb = DummyEmbedding(7.0)
        self.moshi_lm_kwargs = {"dim": 3}

    def to_original_moshi_lm(self):
        return self


def _write_training_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")


def _resolve_mode(tmp_path, *, requested_mode=None, training_config_path=None):
    return clean_moshi._resolve_oracle_embedding_mode(
        moshi_ft_dir=str(tmp_path / "step_fp32"),
        training_config_path=training_config_path,
        requested_mode=requested_mode,
    )


def _run_clean(monkeypatch, tmp_path, *, requested_mode=None):
    model = DummyConvertibleLM()
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(
        clean_moshi.MoshiForFinetuning,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )
    args = SimpleNamespace(
        moshi_ft_dir=str(tmp_path / "fp32"),
        save_dir=str(tmp_path / "cleaned"),
        model_dtype="float32",
        remove_modules_for_user_stream=False,
        oracle_embedding_mode=requested_mode,
    )

    clean_moshi.main(args)
    state = load_file(tmp_path / "cleaned" / "model.safetensors")
    return initial_state, state


def test_clean_moshi_materializes_tied_oracle_embedding(monkeypatch, tmp_path):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": "tie"})
    initial_state, state = _run_clean(monkeypatch, tmp_path)

    assert torch.equal(state["text_emb.weight"], initial_state["text_emb.weight"])
    assert torch.equal(state["oracle_emb.weight"], initial_state["text_emb.weight"])


def test_clean_moshi_preserves_separate_oracle_embedding(monkeypatch, tmp_path):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": "separate"})
    initial_state, state = _run_clean(monkeypatch, tmp_path)

    assert torch.equal(state["text_emb.weight"], initial_state["text_emb.weight"])
    assert torch.equal(state["oracle_emb.weight"], initial_state["oracle_emb.weight"])
    assert not torch.equal(state["oracle_emb.weight"], state["text_emb.weight"])


@pytest.mark.parametrize("mode", ["tie", "separate"])
def test_resolver_accepts_matching_cli_mode(tmp_path, mode):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": mode})

    assert _resolve_mode(tmp_path, requested_mode=mode) == mode


@pytest.mark.parametrize(
    ("metadata_mode", "requested_mode"),
    [("tie", "separate"), ("separate", "tie")],
)
def test_resolver_rejects_mode_mismatch(tmp_path, metadata_mode, requested_mode):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": metadata_mode})

    with pytest.raises(
        ValueError,
        match=rf"mismatch.*{metadata_mode!r}.*{requested_mode!r}",
    ):
        _resolve_mode(tmp_path, requested_mode=requested_mode)


@pytest.mark.parametrize("config", [None, {}], ids=["missing-file", "missing-key"])
def test_resolver_uses_cli_mode_for_legacy_checkpoint(tmp_path, config):
    if config is not None:
        _write_training_config(tmp_path / "config.json", config)

    with pytest.warns(UserWarning, match="legacy fallback"):
        mode = _resolve_mode(tmp_path, requested_mode="tie")

    assert mode == "tie"


@pytest.mark.parametrize("config", [None, {}], ids=["missing-file", "missing-key"])
def test_resolver_requires_mode_when_metadata_is_unavailable(tmp_path, config):
    if config is not None:
        _write_training_config(tmp_path / "config.json", config)

    with pytest.raises(ValueError, match="Could not determine oracle_embedding_mode"):
        _resolve_mode(tmp_path)


@pytest.mark.parametrize("invalid_mode", [None, 1, "invalid"])
def test_resolver_rejects_invalid_metadata_even_with_cli_fallback(tmp_path, invalid_mode):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": invalid_mode})

    with pytest.raises(ValueError, match="invalid oracle_embedding_mode"):
        _resolve_mode(tmp_path, requested_mode="tie")


def test_resolver_rejects_malformed_training_config(tmp_path):
    (tmp_path / "config.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        _resolve_mode(tmp_path, requested_mode="tie")


def test_resolver_rejects_non_object_training_config(tmp_path):
    _write_training_config(tmp_path / "config.json", [])

    with pytest.raises(ValueError, match="JSON object"):
        _resolve_mode(tmp_path, requested_mode="tie")


def test_resolver_prefers_explicit_training_config(tmp_path):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": "tie"})
    explicit_config = tmp_path / "run" / "config.json"
    _write_training_config(explicit_config, {"oracle_embedding_mode": "separate"})

    assert _resolve_mode(tmp_path, training_config_path=str(explicit_config)) == "separate"


@pytest.mark.parametrize("explicit_path_type", ["missing", "directory"])
def test_resolver_does_not_fallback_from_invalid_explicit_path(tmp_path, explicit_path_type):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": "tie"})
    explicit_path = tmp_path / "explicit"
    if explicit_path_type == "directory":
        explicit_path.mkdir()
        expected_error = ValueError
    else:
        expected_error = FileNotFoundError

    with pytest.raises(expected_error, match="Training config"):
        _resolve_mode(
            tmp_path,
            requested_mode="tie",
            training_config_path=str(explicit_path),
        )


def test_resolver_rejects_invalid_programmatic_cli_mode(tmp_path):
    with pytest.raises(ValueError, match="invalid oracle_embedding_mode"):
        _resolve_mode(tmp_path, requested_mode="invalid")


def test_mode_mismatch_fails_before_loading_model(monkeypatch, tmp_path):
    _write_training_config(tmp_path / "config.json", {"oracle_embedding_mode": "tie"})
    model_loader = Mock()
    monkeypatch.setattr(
        clean_moshi.MoshiForFinetuning,
        "from_pretrained",
        model_loader,
    )
    args = SimpleNamespace(
        moshi_ft_dir=str(tmp_path / "step_fp32"),
        save_dir=str(tmp_path / "cleaned"),
        model_dtype="float32",
        remove_modules_for_user_stream=False,
        training_config_path=None,
        oracle_embedding_mode="separate",
    )

    with pytest.raises(ValueError, match="mismatch"):
        clean_moshi.main(args)

    model_loader.assert_not_called()
    assert not (tmp_path / "cleaned").exists()


def test_materialization_fails_closed_when_copy_does_not_match(monkeypatch):
    model = DummyConvertibleLM()
    monkeypatch.setattr(
        model,
        "materialize_oracle_embedding_for_checkpoint_",
        lambda mode: None,
    )

    with pytest.raises(RuntimeError, match="was not materialized"):
        clean_moshi.materialize_oracle_embedding_for_inference(model, "tie")


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_materialization_rejects_non_finite_embedding(non_finite):
    model = DummyConvertibleLM()
    with torch.no_grad():
        model.text_emb.weight[0, 0] = non_finite

    with pytest.raises(RuntimeError, match="contains non-finite values.*weight"):
        clean_moshi.materialize_oracle_embedding_for_inference(model, "tie")
