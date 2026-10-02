from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import load_dataset

from tools import prepare_dataset
from utils.data import DataCollator, preprocess_function
from utils.dataset_schema import prepared_dataset_features, preprocessed_dataset_features

PREPROCESSING = {
    "speakers": ["A", "B"],
    "max_length": None,
    "min_length": None,
    "delays": [0] * 17,
    "initial_token_ids": [1000] * 17,
    "padding_token_ids": [0] * 17,
    "zero_token_id": 0,
}


def _events(mask):
    empty = mask == []
    events = {
        "event_frame_pos": np.array([] if empty else [2, 6], dtype=np.int32),
        "event_ratio": np.array([] if empty else [0.75, 1.0], dtype=np.float32),
        "event_skip_forbid": np.array([] if empty else [0, 1], dtype=np.int8),
        "pred_values": np.array([] if empty else [21, 22], dtype=np.int32),
        "pred_offsets": np.array([0] if empty else [0, 1, 2], dtype=np.int32),
        "hint_values": np.array([] if empty else [11, 12], dtype=np.int32),
        "hint_offsets": np.array([0] if empty else [0, 1, 2], dtype=np.int32),
    }
    if mask is not None:
        events["event_use_hint"] = np.array(mask, dtype=np.int8)
    return events


@pytest.fixture
def tokenized_dialogues(tmp_path):
    audio, text, oracle = [tmp_path / name for name in ("audio", "text", "oracle")]
    for directory in (audio, text, oracle):
        directory.mkdir()
    for index, mask in enumerate([None, [], [0, 1]]):
        name = str(index)
        np.savez(audio / f"{name}.npz", A=np.ones((8, 12)), B=np.ones((8, 12)))
        np.savez(text / f"{name}.npz", A=np.ones(12), B=np.ones(12))
        np.savez(
            oracle / f"{name}.npz",
            **{f"{sp}_{key}": value for sp in ("A", "B") for key, value in _events(mask).items()},
        )
    return SimpleNamespace(
        tokenized_audio_dir=str(audio),
        tokenized_text_dir=str(text),
        tokenized_oracle_dir=str(oracle),
        output_prefix=str(tmp_path / "train"),
        num_examples_per_parquet=1,
        text_padding_id=3,
    )


@pytest.mark.parametrize("shard_size", [1, 3])
def test_saved_schema_preserves_null_empty_and_nonempty_masks(
    tokenized_dialogues, tmp_path, shard_size
):
    tokenized_dialogues.num_examples_per_parquet = shard_size
    prepare_dataset.main(tokenized_dialogues)
    tables = [pq.read_table(path) for path in sorted(tmp_path.glob("train-*.parquet"))]
    for table in tables:
        for speaker in ("A", "B"):
            field = table.schema.field(f"{speaker}_oracle_event_use_hint")
            assert field.nullable and field.type == pa.list_(pa.int8())
        assert table.schema == tables[0].schema
    combined = pa.concat_tables(tables)
    assert combined["B_oracle_event_use_hint"].to_pylist() == [None, [], [0, 1]]
    processed = preprocess_function(combined.to_pydict(), **PREPROCESSING)
    assert [
        None if mask is None else mask.tolist() for mask in processed["oracle_event_use_hint"]
    ] == ([None, [], [0, 1]] * 2)
    batch = DataCollator(zero_token_id=0, oracle_start_id=99)(
        [{key: values[i] for key, values in processed.items()} for i in range(6)]
    )
    assert batch.oracle_tokens.shape == (6, 1, 13)
    assert int((batch.oracle_tokens == 99).sum()) == 8


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 0, 2)])
@pytest.mark.parametrize("num_proc", [None, 2])
@pytest.mark.parametrize("old_parquet", [False, True])
def test_shard_order_and_map_batches_preserve_mask_meaning(
    tokenized_dialogues, tmp_path, order, num_proc, old_parquet
):
    prepare_dataset.main(tokenized_dialogues)
    paths = sorted(tmp_path.glob("train-*.parquet"))
    if old_parquet:
        # Existing Parquet files may have no mask columns at all.
        table = (
            pq.read_table(paths[0])
            .drop_columns([f"{sp}_oracle_event_use_hint" for sp in ("A", "B")])
            .replace_schema_metadata(None)
        )
        pq.write_table(table, paths[0])
    input_features = prepared_dataset_features(use_oracle=True)
    dataset = load_dataset(
        "parquet",
        data_files=[str(paths[i]) for i in order],
        split="train",
        cache_dir=str(tmp_path / "cache"),
        features=input_features,
        columns=list(input_features),
    )
    expected = [[None, [], [0, 1]][i] for i in order]
    assert dataset["B_oracle_event_use_hint"] == expected
    mapped = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1,
        num_proc=num_proc,
        fn_kwargs=PREPROCESSING,
        features=preprocessed_dataset_features(use_oracle=True),
    )
    assert mapped["oracle_event_use_hint"] == [mask for mask in expected for _ in range(2)]
    assert mapped.features["oracle_event_use_hint"].feature.dtype == "int8"
    batch = DataCollator(zero_token_id=0, oracle_start_id=99)(list(mapped))
    assert int((batch.oracle_tokens == 99).sum()) == 8


@pytest.mark.parametrize("hint_only", [False, True])
@pytest.mark.parametrize("skip_probability", [0.0, 0.1, 0.7, 1.0])
@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_mask_preserves_legacy_selection_and_rng(hint_only, skip_probability, missing):
    row = {"A": [np.ones((9, 12))], "B": [np.ones((9, 12))]}
    row.update({f"B_oracle_{key}": [value] for key, value in _events(None).items()})
    legacy = preprocess_function(row, **(PREPROCESSING | {"speakers": ["B"]}))
    row["B_oracle_event_use_hint"] = [missing]
    nullable = preprocess_function(row, **(PREPROCESSING | {"speakers": ["B"]}))
    assert nullable["oracle_event_use_hint"] == [None]
    # The old collator path receives no mask key. Compare the complete tensor and
    # RNG state, covering ratio selection, hint_only, skip protection and jitter.
    del legacy["oracle_event_use_hint"]
    results = []
    states = []
    for data in (legacy, nullable):
        collator = DataCollator(
            zero_token_id=0,
            oracle_start_id=99,
            oracle_hint_only=hint_only,
            oracle_skip_prob_min=skip_probability,
            oracle_skip_prob_max=skip_probability,
            oracle_max_time_jitter_frames=1,
        )
        collator._rng = np.random.default_rng(42)
        collator._rng_worker_id = -1
        batch = collator([{key: values[0] for key, values in data.items()}])
        results.append(batch.oracle_tokens)
        states.append(collator._rng.bit_generator.state)
    assert results[0].equal(results[1])
    assert states[0] == states[1]


@pytest.mark.parametrize("mask", [[], [1], [1, None], [0.5, 1], [256, 1]])
def test_partial_or_invalid_mask_is_not_treated_as_legacy(mask):
    row = {"A": [np.ones((9, 12))], "B": [np.ones((9, 12))]}
    row.update({f"B_oracle_{key}": [value] for key, value in _events(None).items()})
    row["B_oracle_event_use_hint"] = [mask]
    with pytest.raises(ValueError, match="one 0/1 flag per event"):
        preprocess_function(row, **(PREPROCESSING | {"speakers": ["B"]}))


@pytest.mark.parametrize("per_file_speakers", [None, ["B", "A", "B"]])
def test_training_loader_uses_explicit_schemas(
    tokenized_dialogues, tmp_path, monkeypatch, per_file_speakers
):
    import finetune

    prepare_dataset.main(tokenized_dialogues)
    paths = sorted(tmp_path.glob("train-*.parquet"))
    args = SimpleNamespace(
        use_oracle=True,
        train_data_file_speakers=per_file_speakers,
        train_data_files=[str(path) for path in paths],
        dataset_cache_dir=str(tmp_path / "cache"),
        dataset_processing_workers=None,
        moshi_speakers=["B"],
    )
    monkeypatch.setattr(finetune, "logger", SimpleNamespace(info=lambda *args: None))
    monkeypatch.setattr(finetune, "_build_preprocessing_kwargs", lambda *args: PREPROCESSING)
    dataset = finetune._load_train_dataset(
        args,
        SimpleNamespace(main_process_first=nullcontext),
        moshi_lm=None,
    )
    assert dataset["oracle_event_use_hint"] == [None, [], [0, 1]]
