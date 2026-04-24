from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import tools.init_moshi_for_ft as init_moshi_for_ft


class DummyLM:
    def __init__(self, *, with_depformer_text_emb: bool = True):
        self.text_emb = nn.Embedding(4, 3)
        self.oracle_emb = nn.Embedding(4, 3)
        self.depformer_text_emb = nn.Embedding(4, 3) if with_depformer_text_emb else None


class MultiKeyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3))
        self.bias = nn.Parameter(torch.randn(3))


class DummyLoadableLM:
    def __init__(self):
        self.text_emb = MultiKeyEmbedding()
        self.oracle_emb = MultiKeyEmbedding()
        self.depformer_text_emb = None

    def load_state_dict(self, state_dict, strict=False, assign=True):
        return SimpleNamespace(missing_keys=["oracle_emb.bias"], unexpected_keys=[])


def test_prepare_text_embedding_modules_resets_oracle_from_reinitialized_text(monkeypatch):
    lm = DummyLM()

    init_calls = []

    def fake_init_embedding_module(emb, retain_token_ids):
        init_calls.append((emb, tuple(retain_token_ids)))
        if emb is lm.text_emb:
            emb.weight.data.fill_(1.0)
        elif emb is lm.depformer_text_emb:
            emb.weight.data.fill_(2.0)
        else:
            raise AssertionError("oracle_emb should be synchronized from text_emb")
        return emb

    monkeypatch.setattr(init_moshi_for_ft, "init_embedding_module", fake_init_embedding_module)

    init_moshi_for_ft._prepare_text_embedding_modules(
        lm,
        [0, 3],
        init_text_embeddings=True,
        oracle_emb_missing=False,
    )

    assert init_calls == [
        (lm.text_emb, (0, 3)),
        (lm.depformer_text_emb, (0, 3)),
    ]
    assert torch.equal(lm.oracle_emb.weight, lm.text_emb.weight)
    assert torch.all(lm.depformer_text_emb.weight == 2.0)


def test_prepare_text_embedding_modules_backfills_missing_oracle_without_reinitializing(
    monkeypatch,
):
    lm = DummyLM(with_depformer_text_emb=False)
    original_text = lm.text_emb.weight.detach().clone()

    init_calls = []

    def fake_init_embedding_module(emb, retain_token_ids):
        init_calls.append((emb, tuple(retain_token_ids)))
        return emb

    monkeypatch.setattr(init_moshi_for_ft, "init_embedding_module", fake_init_embedding_module)

    lm.oracle_emb.weight.data.zero_()
    init_moshi_for_ft._prepare_text_embedding_modules(
        lm,
        [0],
        init_text_embeddings=False,
        oracle_emb_missing=True,
    )

    assert init_calls == []
    assert torch.equal(lm.text_emb.weight, original_text)
    assert torch.equal(lm.oracle_emb.weight, original_text)


def test_load_moshi_lm_lenient_rejects_partially_missing_oracle(monkeypatch):
    lm = DummyLoadableLM()

    monkeypatch.setattr(init_moshi_for_ft.loaders, "get_moshi_lm", lambda **kwargs: lm)
    monkeypatch.setattr(init_moshi_for_ft, "load_file", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="Partially missing oracle_emb weights"):
        init_moshi_for_ft._load_moshi_lm_lenient("dummy.safetensors")
