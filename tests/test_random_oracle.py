import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import sentencepiece as spm
from datasets import load_dataset

from tools import generate_oracle_from_text, prepare_dataset, tokenize_oracle
from tools.oracle_generation import OracleGenerator, Word, words_from_word_transcript
from tools.random_oracle import build_response_pool, generate_random_predictions
from utils.data import DataCollator, preprocess_function
from utils.dataset_schema import preprocessed_dataset_features

SAMPLE = Path(__file__).resolve().parents[1] / "data/random_oracle_sample/text"


@pytest.fixture
def sample(tmp_path):
    dialogues = {
        path.stem: words_from_word_transcript(json.loads(path.read_text()))
        for path in sorted(SAMPLE.glob("*.json"))
    }
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(" ".join(w.text for w in words) for words in dialogues.values()),
        model_writer=model,
        model_type="word",
        vocab_size=128,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(model.getvalue())
    tokenizer = spm.SentencePieceProcessor(model_proto=model.getvalue())
    pool = build_response_pool(dialogues.items(), tokenizer)
    return dialogues, tokenizer, model_path, pool


def test_random_generation_preserves_schedule_and_selects_distinct_cross_dialogue_responses(sample):
    dialogues, tokenizer, _, pool = sample
    for dialogue_id, words in dialogues.items():
        requests = list(OracleGenerator().generate_requests(words))
        predictions = generate_random_predictions(
            words, dialogue_id=dialogue_id, pool=pool, tokenizer=tokenizer
        )
        assert [p.timestamp_ms for p in predictions] == [r.timestamp_ms for r in requests]
        assert [p.channel for p in predictions] == [r.target_channel for r in requests]
        assert [p.use_hint for p in predictions] == [False, False, False, True]
        assert predictions[-1].hint == requests[-1].next_utterance_hint
        random_texts = [p.prediction for p in predictions if not p.use_hint]
        assert len(set(random_texts)) == len(random_texts)
        for text in random_texts:
            candidate = next(c for c in pool if c.text == text)
            assert dialogue_id not in candidate.owners
            assert text != predictions[-1].hint
            ratio = len(candidate.token_ids) / len(tokenizer.encode(predictions[-1].hint))
            assert 0.5 <= ratio <= 2.0


def test_seed_is_reproducible_across_dialogue_order_and_changes_random_text(sample):
    dialogues, tokenizer, _, pool = sample
    reverse_pool = build_response_pool(reversed(list(dialogues.items())), tokenizer)
    assert reverse_pool == pool
    words = dialogues["japan"]
    first = generate_random_predictions(words, dialogue_id="japan", pool=pool, tokenizer=tokenizer)
    repeat = generate_random_predictions(
        words, dialogue_id="japan", pool=reverse_pool, tokenizer=tokenizer
    )
    assert repeat == first
    other = generate_random_predictions(
        words, dialogue_id="japan", pool=pool, tokenizer=tokenizer, seed=777
    )
    assert [p.prediction for p in other] != [p.prediction for p in first]
    assert other[-1] == first[-1]


def test_duplicate_response_owners_are_preserved_and_not_sampled(sample):
    dialogues, tokenizer, _, _ = sample
    pool = build_response_pool([*dialogues.items(), ("copy", dialogues["japan"])], tokenizer)
    duplicates = [candidate for candidate in pool if "copy" in candidate.owners]
    assert len(duplicates) == 1
    assert duplicates[0].owners == frozenset({"japan", "copy"})
    predictions = generate_random_predictions(
        dialogues["japan"], dialogue_id="japan", pool=pool, tokenizer=tokenizer
    )
    assert all(p.prediction != duplicates[0].text for p in predictions)


def test_insufficient_pool_fails_without_self_or_target_fallback(sample):
    dialogues, tokenizer, _, _ = sample
    pool = build_response_pool([("japan", dialogues["japan"])], tokenizer)
    with pytest.raises(ValueError, match="Insufficient random responses"):
        generate_random_predictions(
            dialogues["japan"], dialogue_id="japan", pool=pool, tokenizer=tokenizer
        )


def test_identical_responses_in_separate_turns_keep_separate_final_hints(sample):
    dialogues, tokenizer, _, pool = sample
    original = dialogues["japan"]
    offset = original[-1].end_time + 0.5
    words = original + [
        replace(word, start_time=word.start_time + offset, end_time=word.end_time + offset)
        for word in original
    ]
    requests = list(OracleGenerator().generate_requests(words))
    predictions = generate_random_predictions(
        words, dialogue_id="japan", pool=pool, tokenizer=tokenizer
    )
    target_indices = {request.target_index for request in requests}
    assert len(target_indices) == 3
    for target_index in target_indices:
        indices = [i for i, request in enumerate(requests) if request.target_index == target_index]
        assert [predictions[i].use_hint for i in indices] == [False] * (len(indices) - 1) + [True]
    assert predictions[3].hint == predictions[-1].hint


@pytest.mark.parametrize("target_channel", [None, 1])
def test_multiturn_final_hints_survive_skipping_with_channel_filter(sample, target_channel):
    dialogues, tokenizer, _, pool = sample
    original = dialogues["japan"]
    words = original + [
        replace(word, start_time=word.start_time + 4, end_time=word.end_time + 4)
        for word in original
    ]
    predictions = generate_random_predictions(
        words,
        dialogue_id="japan",
        pool=pool,
        tokenizer=tokenizer,
        target_channel=target_channel,
    )
    records = [generate_oracle_from_text.prediction_to_record(p) for p in predictions]
    events = tokenize_oracle.build_oracle_events_for_channel(records, 1, 100, tokenizer, 12.5)
    hint_indices = events["event_use_hint"].astype(bool)
    assert events["event_frame_pos"][hint_indices].tolist() == [25, 75]
    # With B-only generation, the inherited channel-run heuristic protects only the last hint.
    assert events["event_skip_forbid"][hint_indices].tolist() == (
        [1, 1] if target_channel is None else [0, 1]
    )

    collator = DataCollator(
        zero_token_id=0,
        oracle_start_id=999,
        oracle_skip_prob_min=1.0,
        oracle_skip_prob_max=1.0,
    )
    actual = collator._events_to_oracle_1d(
        {f"oracle_{key}": value for key, value in events.items()}, t=100
    )
    expected = np.zeros(100, dtype=np.int64)
    hint_tokens = tokenizer.encode("Tokyo is the capital of Japan")
    for frame in (25, 75):
        expected[frame] = 999
        expected[frame + 1 : frame + 1 + len(hint_tokens)] = hint_tokens
    assert actual.tolist() == expected.tolist()


def _random_args(model_path, output_dir, **overrides):
    return SimpleNamespace(
        **{
            "strategy": "random",
            "text_dir": str(SAMPLE),
            "pool_text_dir": str(SAMPLE),
            "text_tokenizer_path": str(model_path),
            "output_dir": str(output_dir),
            "seed": 42,
            "time_interval": 0.5,
            "target_channel": None,
            "A_channel": 0,
            "B_channel": 1,
            "min_length_ratio": 0.5,
            "max_length_ratio": 2.0,
            "limit": None,
            "resume": False,
            **overrides,
        }
    )


def test_cli_uses_no_api_and_records_portable_reproducible_outputs(sample, tmp_path, monkeypatch):
    _, _, model_path, _ = sample
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(**kwargs):
        pytest.fail("Random generation must not create an API client")

    monkeypatch.setattr(generate_oracle_from_text, "make_openai_predict_fn", forbidden)
    first, second = tmp_path / "first", tmp_path / "second"
    generate_oracle_from_text.main(_random_args(model_path, first))
    generate_oracle_from_text.main(_random_args(model_path, second))
    for path in first.glob("*.json"):
        assert path.read_bytes() == (second / path.name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert len(manifest["outputs"]) == 6
    assert manifest["hint_policy"] == "final_scheduled_event_must_use_hint"
    with pytest.raises(ValueError, match="empty output directory"):
        generate_oracle_from_text.main(_random_args(model_path, first, seed=1))


def test_random_json_survives_tokenization_parquet_delay_chunking_and_collation(sample, tmp_path):
    dialogues, tokenizer, model_path, _ = sample
    raw, audio, text, oracle = [tmp_path / name for name in ("raw", "audio", "text", "oracle")]
    for directory in (audio, text, oracle):
        directory.mkdir()
    generate_oracle_from_text.main(_random_args(model_path, raw))
    for name in dialogues:
        # Stand-ins for already-tokenized audio/text; no model download or GPU is needed.
        np.savez(audio / f"{name}.npz", A=np.ones((8, 48)), B=np.ones((8, 48)))
        np.savez(text / f"{name}.npz", A=np.ones(48), B=np.ones(48))
    tokenize_oracle.worker(
        0,
        list(dialogues),
        SimpleNamespace(
            tokenized_audio_dir=str(audio),
            oracle_dir=str(raw),
            output_dir=str(oracle),
            text_tokenizer_path=str(model_path),
            oracle_suffix=".json",
            A_channel=0,
            B_channel=1,
            audio_tokenizer_frame_rate=12.5,
        ),
    )
    prepare_dataset.main(
        SimpleNamespace(
            tokenized_audio_dir=str(audio),
            tokenized_text_dir=str(text),
            tokenized_oracle_dir=str(oracle),
            output_prefix=str(tmp_path / "train"),
            num_examples_per_parquet=10,
            text_padding_id=3,
        )
    )
    table = pq.read_table(next(tmp_path.glob("train-*.parquet")))
    assert table.column("B_oracle_event_use_hint").to_pylist() == [[0, 0, 0, 1]] * 6
    assert table.column("A_oracle_event_use_hint").to_pylist() == [[]] * 6
    rows = {key: table.column(key).to_pylist() for key in table.column_names}
    kwargs = {
        "speakers": ["B"],
        "min_length": None,
        "delays": [2] + [0] * 16,
        "initial_token_ids": [1000] * 17,
        "padding_token_ids": [0] * 17,
        "zero_token_id": 0,
    }
    features = preprocess_function(rows, max_length=None, **kwargs)
    collator = DataCollator(zero_token_id=0, oracle_start_id=999)
    examples = [{key: value[i] for key, value in features.items()} for i in range(6)]
    batch = collator(examples)
    assert batch.oracle_tokens.shape == (6, 1, 51)
    for i, row in enumerate(table.to_pylist()):
        name = Path(row["dialogue_id"]).name
        records = json.loads((raw / f"{name}.json").read_text())
        expected = np.zeros(51, dtype=np.int64)
        for record in records:
            start = int(record["timestamp_ms"] / 1000 * 12.5) + 3
            selected = record["hint"] if record["use_hint"] else record["prediction"]
            tokens = tokenizer.encode(selected)
            expected[start] = 999
            end = min(start + 1 + len(tokens), len(expected))
            expected[start + 1 : end] = tokens[: end - start - 1]
        assert batch.oracle_tokens[i, 0].tolist() == expected.tolist()

    chunked = preprocess_function(rows, max_length=20, **kwargs)
    assert [mask.tolist() for mask in chunked["oracle_event_use_hint"]] == [[0, 0], [0, 1], []] * 6
    assert [pos.tolist() for pos in chunked["oracle_event_frame_pos"]] == [
        [9, 15],
        [4, 11],
        [],  # Existing splitting divides 51 frames into three equal chunks.
    ] * 6

    # Exercise the Arrow serialization used by training: A has empty masks,
    # while B has oracle events. Direct preprocessing/collation misses this boundary.
    dataset = load_dataset(
        "parquet",
        data_files=str(next(tmp_path.glob("train-*.parquet"))),
        split="train",
        cache_dir=str(tmp_path / "dataset_cache"),
    )
    skipping_collator = DataCollator(
        zero_token_id=0,
        oracle_start_id=999,
        oracle_skip_prob_min=1.0,
        oracle_skip_prob_max=1.0,
    )
    for max_length in (None, 20):
        mapped = dataset.map(
            preprocess_function,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=2,
            fn_kwargs=kwargs | {"speakers": ["A", "B"], "max_length": max_length},
            features=preprocessed_dataset_features(use_oracle=True),
        )
        expected_b_masks = [[0, 0, 0, 1]] if max_length is None else [[0, 0], [0, 1], []]
        # Each input batch has two A views followed by two B views.
        assert (
            mapped["oracle_event_use_hint"]
            == ([[]] * (2 * len(expected_b_masks)) + expected_b_masks * 2) * 3
        )
        batch = skipping_collator(list(mapped))
        assert int((batch.oracle_tokens == 999).sum()) == 6


@pytest.mark.parametrize("masks", [[False, None], [False, "true"], [False, True]])
def test_tokenizer_rejects_partial_invalid_or_empty_hint_decisions(sample, masks):
    _, tokenizer, _, _ = sample
    records = [
        {"timestamp_ms": 500 * (i + 1), "channel": 1, "prediction": "hello"} for i in range(2)
    ]
    for record, mask in zip(records, masks, strict=True):
        if mask is not None:
            record["use_hint"] = mask
    with pytest.raises(ValueError, match="use_hint"):
        tokenize_oracle.build_oracle_events_for_channel(records, 1, 30, tokenizer, 12.5)


def _paused_question(response_start=1.86):
    texts = "Could you tell me the capital city of Japan please".split()
    ends = [0.25, 0.5] + [1.5 + i * 0.05 for i in range(8)]
    starts = [0, 0.25, 1.4] + ends[2:-1]
    words = [Word(t, s, e, "A") for t, s, e in zip(texts, starts, ends, strict=True)]
    return words + [
        Word(t, response_start + i * 0.1, response_start + (i + 1) * 0.1, "B")
        for i, t in enumerate("Tokyo is the capital of Japan".split())
    ]


@pytest.mark.parametrize(
    "words, reason, target_index",
    [
        (
            [Word("Hello", 0, 0.1, "A"), Word("there", 0.1, 0.2, "A"), Word("Hi", 0.2, 0.3, "B")],
            "no_scheduled_events",
            2,
        ),
        (
            [
                Word(t, i * 0.2, (i + 1) * 0.2, "A")
                for i, t in enumerate("What is the capital".split())
            ]
            + [Word("Tokyo", 0.81, 0.91, "B")],
            "no_eligible_hint",
            4,
        ),
        (_paused_question(), "events_after_last_eligible_hint", 10),
    ],
)
def test_invalid_terminal_hint_fails_before_sampling(sample, words, reason, target_index):
    _, tokenizer, _, _ = sample
    # An empty pool would fail sampling: the response contract must be checked first.
    with pytest.raises(ValueError) as error:
        generate_random_predictions(words, dialogue_id="edge", pool=(), tokenizer=tokenizer)
    message = str(error.value)
    assert f"reason={reason}" in message
    assert f"dialogue='edge' target_index={target_index} speaker=B channel=1" in message
    if reason != "no_scheduled_events":
        assert "last_event_ms=" in message and "last_ratio=" in message


def test_nonmonotonic_ratios_are_allowed_when_final_event_has_hint(sample):
    _, tokenizer, _, pool = sample
    words = _paused_question(response_start=2.06)
    requests = list(OracleGenerator().generate_requests(words))
    assert [r.current_spoken_ratio for r in requests] == [1.0, 1.0, 0.3, 1.0]
    predictions = generate_random_predictions(
        words, dialogue_id="edge", pool=pool, tokenizer=tokenizer
    )
    assert [p.use_hint for p in predictions] == [False, False, False, True]


@pytest.mark.parametrize("mapping", [{"A": 0, "B": 1}, {"A": 1, "B": 0}])
def test_targets_respect_channel_mapping_and_allow_single_hint(sample, mapping):
    _, tokenizer, _, _ = sample
    words = [Word("Hello", 0, 0.2, "A"), Word("there", 0.2, 0.4, "A"), Word("Tokyo", 0.6, 0.7, "B")]
    predictions = generate_random_predictions(
        words,
        dialogue_id="edge",
        pool=(),
        tokenizer=tokenizer,
        speaker_to_channel=mapping,
        target_channel=mapping["B"],
    )
    assert len(predictions) == 1
    assert predictions[0].use_hint is True
    assert predictions[0].channel == mapping["B"]
    assert (
        generate_random_predictions(
            words,
            dialogue_id="edge",
            pool=(),
            tokenizer=tokenizer,
            speaker_to_channel=mapping,
            target_channel=mapping["A"],
        )
        == []
    )
    # The final B response has no scheduled events. It is irrelevant for A-only
    # generation, but must be rejected when B is included, even after valid targets.
    words += [
        Word("Hello", 0.8, 1.1, "A"),
        Word("there", 1.1, 1.2, "A"),
        Word("Tokyo", 1.21, 1.3, "B"),
    ]
    a_predictions = generate_random_predictions(
        words,
        dialogue_id="edge",
        pool=(),
        tokenizer=tokenizer,
        speaker_to_channel=mapping,
        target_channel=mapping["A"],
    )
    assert len(a_predictions) == 1 and a_predictions[0].use_hint is True
    with pytest.raises(ValueError, match="target_index=5 speaker=B.*no_scheduled_events"):
        generate_random_predictions(
            words,
            dialogue_id="edge",
            pool=(),
            tokenizer=tokenizer,
            speaker_to_channel=mapping,
            target_channel=mapping["B"],
        )


def test_empty_json_is_unspecified_but_empty_channel_is_explicit(sample):
    _, tokenizer, _, _ = sample
    empty = tokenize_oracle.build_oracle_events_for_channel([], 0, 30, tokenizer, 12.5)
    assert "event_use_hint" not in empty
    records = [{"timestamp_ms": 500, "channel": 1, "hint": "Tokyo", "use_hint": True}]
    empty_channel = tokenize_oracle.build_oracle_events_for_channel(records, 0, 30, tokenizer, 12.5)
    assert empty_channel["event_use_hint"].tolist() == []
    assert empty_channel["event_use_hint"].dtype == np.int8
