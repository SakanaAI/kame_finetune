import json

import pytest
import torch
import torch.nn as nn

import models.moshi_for_finetuning as moshi_for_finetuning_module
from models.moshi_for_finetuning import MoshiForFinetuning
from models.oracle_embedding_utils import (
    backfill_oracle_embedding_from_text,
    expected_missing_oracle_embedding_keys,
    validate_oracle_embedding_checkpoint_load,
)


class MultiKeyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3))
        self.bias = nn.Parameter(torch.randn(3))
        self.register_buffer("scale", torch.randn(1))


class DummyLM:
    def __init__(self):
        self.text_emb = MultiKeyEmbedding()
        self.oracle_emb = MultiKeyEmbedding()


def test_backfill_oracle_embedding_from_text_copies_all_state():
    lm = DummyLM()
    original_text_state = {
        key: value.detach().clone() for key, value in lm.text_emb.state_dict().items()
    }

    backfill_oracle_embedding_from_text(lm)

    for key, value in original_text_state.items():
        assert torch.equal(lm.oracle_emb.state_dict()[key], value)


def test_validate_oracle_embedding_checkpoint_load_allows_full_oracle_gap():
    lm = DummyLM()

    oracle_emb_missing = validate_oracle_embedding_checkpoint_load(
        lm,
        missing_keys=sorted(expected_missing_oracle_embedding_keys(lm)),
        unexpected_keys=[],
        context="loading a legacy checkpoint",
    )

    assert oracle_emb_missing is True


def test_validate_oracle_embedding_checkpoint_load_rejects_partial_oracle_gap():
    lm = DummyLM()

    with pytest.raises(RuntimeError, match="Partially missing oracle_emb weights"):
        validate_oracle_embedding_checkpoint_load(
            lm,
            missing_keys=["oracle_emb.bias"],
            unexpected_keys=[],
            context="loading a partially broken checkpoint",
        )


def test_validate_oracle_embedding_checkpoint_load_rejects_non_oracle_gaps_and_extras():
    lm = DummyLM()

    with pytest.raises(RuntimeError, match="Incompatible state_dict"):
        validate_oracle_embedding_checkpoint_load(
            lm,
            missing_keys=["text_emb.weight"],
            unexpected_keys=["unexpected.weight"],
            context="loading an incompatible checkpoint",
        )


class DummySerializableLM(nn.Module):
    materialize_oracle_embedding_for_checkpoint_ = (
        MoshiForFinetuning.materialize_oracle_embedding_for_checkpoint_
    )
    save_pretrained = MoshiForFinetuning.save_pretrained

    def __init__(self):
        super().__init__()
        self.text_emb = MultiKeyEmbedding()
        self.oracle_emb = MultiKeyEmbedding()
        self.moshi_lm_kwargs = {"dim": 3}


def test_save_pretrained_materializes_oracle_embedding_for_tie(monkeypatch, tmp_path):
    lm = DummySerializableLM()
    with torch.no_grad():
        lm.text_emb.weight.fill_(1.0)
        lm.text_emb.bias.fill_(2.0)
        lm.oracle_emb.weight.fill_(7.0)
        lm.oracle_emb.bias.fill_(8.0)
        lm.text_emb.scale.fill_(3.0)
        lm.oracle_emb.scale.fill_(9.0)

    saved = {}

    def fake_save_file(state_dict, output_path):
        saved["state_dict"] = {key: value.detach().clone() for key, value in state_dict.items()}
        saved["output_path"] = output_path

    monkeypatch.setattr(moshi_for_finetuning_module, "save_file", fake_save_file)

    lm.save_pretrained(tmp_path, oracle_embedding_mode="tie")

    assert torch.equal(saved["state_dict"]["oracle_emb.weight"], lm.text_emb.weight.detach())
    assert torch.equal(saved["state_dict"]["oracle_emb.bias"], lm.text_emb.bias.detach())
    assert torch.equal(saved["state_dict"]["oracle_emb.scale"], lm.text_emb.scale.detach())

    kwargs_path = tmp_path / "moshi_lm_kwargs.json"
    assert kwargs_path.exists()
    assert json.loads(kwargs_path.read_text()) == lm.moshi_lm_kwargs
