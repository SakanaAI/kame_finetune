from types import SimpleNamespace

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


def _run_clean(monkeypatch, tmp_path, *, mode: str | None, include_mode: bool = True):
    model = DummyConvertibleLM()
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
    )
    if include_mode:
        args.oracle_embedding_mode = mode

    clean_moshi.main(args)
    state = load_file(tmp_path / "cleaned" / "model.safetensors")
    return model, state


def test_clean_moshi_materializes_tied_oracle_embedding(monkeypatch, tmp_path):
    _, state = _run_clean(monkeypatch, tmp_path, mode="tie")

    assert torch.equal(state["oracle_emb.weight"], state["text_emb.weight"])


def test_clean_moshi_preserves_separate_oracle_embedding(monkeypatch, tmp_path):
    model, state = _run_clean(monkeypatch, tmp_path, mode="separate")

    assert torch.equal(state["oracle_emb.weight"], model.oracle_emb.weight)
    assert not torch.equal(state["oracle_emb.weight"], state["text_emb.weight"])


def test_clean_moshi_requires_oracle_embedding_mode(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="oracle_embedding_mode is required"):
        _run_clean(monkeypatch, tmp_path, mode=None, include_mode=False)


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
