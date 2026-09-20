"""Storage types shared by dataset preparation and training preprocessing."""

from datasets import Features, Sequence, Value


def _oracle_features(prefix: str) -> dict:
    return {
        f"{prefix}_{name}": Sequence(Value(dtype))
        for name, dtype in {
            "event_frame_pos": "int32",
            "event_ratio": "float32",
            "event_skip_forbid": "int8",
            "pred_values": "int32",
            "pred_offsets": "int32",
            "hint_values": "int32",
            "hint_offsets": "int32",
            # A null list means legacy selection; an empty list is explicit.
            "event_use_hint": "int8",
        }.items()
    }


def prepared_dataset_features(*, use_oracle: bool) -> Features:
    features = Features(
        {
            "dialogue_id": Value("string"),
            "A": Sequence(Sequence(Value("int64"))),
            "B": Sequence(Sequence(Value("int64"))),
        }
    )
    if use_oracle:
        for speaker in ("A", "B"):
            features.update(_oracle_features(f"{speaker}_oracle"))
    return features


def preprocessed_dataset_features(*, use_oracle: bool) -> Features:
    features = Features(
        {
            "streams": Sequence(Sequence(Value("int64"))),
            "labels": Sequence(Sequence(Value("int64"))),
            "num_streams": Value("int64"),
            "num_frames": Value("int64"),
        }
    )
    if use_oracle:
        features.update(_oracle_features("oracle"))
    return features
