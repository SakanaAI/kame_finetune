import numpy as np

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
