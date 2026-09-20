import numpy as np
import pytest

from utils.data import DataCollator


def test_events_to_oracle_1d_samples_hint_only_once_per_example(monkeypatch):
    collator = DataCollator(
        zero_token_id=0,
        oracle_pad_id=0,
        oracle_start_id=99,
        oracle_hint_only=True,
        oracle_hint_only_warmup_start=0,
        oracle_hint_only_warmup_end=10,
    )

    calls = {"count": 0}

    def fake_get_effective_hint_only():
        calls["count"] += 1
        return calls["count"] == 1

    monkeypatch.setattr(collator, "_get_effective_hint_only", fake_get_effective_hint_only)

    example = {
        "oracle_event_frame_pos": np.array([0, 3], dtype=np.int32),
        "oracle_event_ratio": np.array([0.0, 0.0], dtype=np.float32),
        "oracle_event_skip_forbid": np.array([1, 1], dtype=np.int8),
        "oracle_pred_values": np.array([21, 22], dtype=np.int32),
        "oracle_pred_offsets": np.array([0, 1, 2], dtype=np.int32),
        "oracle_hint_values": np.array([11, 12], dtype=np.int32),
        "oracle_hint_offsets": np.array([0, 1, 2], dtype=np.int32),
    }

    oracle = collator._events_to_oracle_1d(example, t=6)

    assert calls["count"] == 1
    assert oracle.tolist() == [99, 11, 0, 99, 12, 0]


@pytest.mark.parametrize("hint_only", [False, True])
def test_explicit_hint_mask_overrides_ratio_and_survives_hint_only_mode(hint_only):
    collator = DataCollator(zero_token_id=0, oracle_start_id=99, oracle_hint_only=hint_only)
    example = {
        "oracle_event_frame_pos": np.array([0, 3]),
        "oracle_event_ratio": np.array([1.0, 0.75]),
        "oracle_event_skip_forbid": np.array([1, 1]),
        "oracle_event_use_hint": np.array([0, 1]),
        "oracle_pred_values": np.array([21, 22]),
        "oracle_pred_offsets": np.array([0, 1, 2]),
        "oracle_hint_values": np.array([11, 12]),
        "oracle_hint_offsets": np.array([0, 1, 2]),
    }
    assert collator._events_to_oracle_1d(example, t=6).tolist() == (
        [0, 0, 0, 99, 12, 0] if hint_only else [99, 21, 0, 99, 12, 0]
    )
    del example["oracle_event_use_hint"]
    assert collator._events_to_oracle_1d(example, t=6).tolist() == (
        [99, 11, 0, 99, 12, 0] if hint_only else [99, 11, 0, 99, 22, 0]
    )


@pytest.mark.parametrize("hint_only", [False, True])
def test_skipping_protects_explicit_hints_and_preserves_legacy_behavior(hint_only):
    collator = DataCollator(
        zero_token_id=0,
        oracle_start_id=99,
        oracle_hint_only=hint_only,
        oracle_skip_prob_min=1.0,
        oracle_skip_prob_max=1.0,
    )
    example = {
        "oracle_event_frame_pos": np.array([0, 3, 6]),
        "oracle_event_ratio": np.array([0.75, 1.0, 1.0]),
        "oracle_event_skip_forbid": np.array([0, 0, 1]),
        "oracle_event_use_hint": np.array([1, 0, 0]),
        "oracle_pred_values": np.array([21, 22, 23]),
        "oracle_pred_offsets": np.array([0, 1, 2, 3]),
        "oracle_hint_values": np.array([11, 12, 13]),
        "oracle_hint_offsets": np.array([0, 1, 2, 3]),
    }
    assert collator._events_to_oracle_1d(example, t=9).tolist() == (
        [99, 11, 0, 0, 0, 0, 0, 0, 0] if hint_only else [99, 11, 0, 0, 0, 0, 99, 23, 0]
    )
    assert example["oracle_event_skip_forbid"].tolist() == [0, 0, 1]

    del example["oracle_event_use_hint"]
    assert collator._events_to_oracle_1d(example, t=9).tolist() == [0, 0, 0, 0, 0, 0, 99, 13, 0]
