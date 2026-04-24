import json
from dataclasses import replace

from tools.generate_oracle_from_text import (
    build_user_prompt,
    fallback_prediction_for_request,
    generate_oracle_records,
    process_text_file,
)
from tools.oracle_generation import OraclePredictionRequest

# Wrapper-level tests for prompt building, fallback behavior, and JSON/file output.


def test_generate_oracle_records_matches_tokenize_oracle_schema():
    def predict_fn(request):
        return f"pred-for-{request.next_speaker}-{request.target_channel}"

    transcript = [
        {"speaker": "A", "word": "hello", "start": 0.0, "end": 0.4},
        {"speaker": "A", "word": "there", "start": 0.4, "end": 0.8},
        {"speaker": "A", "word": "how", "start": 0.8, "end": 1.2},
        {"speaker": "A", "word": "today", "start": 1.2, "end": 1.6},
        {"speaker": "B", "word": "fine", "start": 1.7, "end": 2.1},
        {"speaker": "B", "word": "thanks", "start": 2.1, "end": 2.5},
    ]

    records = generate_oracle_records(
        transcript,
        predict_fn=predict_fn,
        time_interval=0.5,
        target_channel=None,
        speaker_to_channel={"A": 0, "B": 1},
    )

    assert len(records) == 3
    assert list(records[0].keys()) == [
        "timestamp_ms",
        "conversation_context",
        "prediction",
        "total_word_count",
        "trigger_word",
        "recent_words",
        "current_spoken_ratio",
        "channel",
        "hint",
    ]
    assert records[0]["channel"] == 1
    assert records[0]["hint"] == ""
    assert records[1]["hint"] == "fine thanks"


def test_process_text_file_writes_json_output(tmp_path):
    text_path = tmp_path / "sample.json"
    output_path = tmp_path / "oracle.json"
    text_path.write_text(
        json.dumps(
            [
                {"speaker": "A", "word": "one", "start": 0.0, "end": 0.3},
                {"speaker": "A", "word": "two", "start": 0.3, "end": 0.6},
                {"speaker": "B", "word": "three", "start": 0.7, "end": 1.0},
                {"speaker": "B", "word": "four", "start": 1.0, "end": 1.3},
                {"speaker": "A", "word": "five", "start": 1.4, "end": 1.7},
                {"speaker": "A", "word": "six", "start": 1.7, "end": 2.0},
            ]
        ),
        encoding="utf-8",
    )

    count = process_text_file(
        text_path,
        output_path,
        predict_fn=lambda request: "synthetic prediction",
        time_interval=0.5,
        target_channel=None,
        speaker_to_channel={"A": 0, "B": 1},
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert count == len(written)
    assert written[0]["prediction"] == "synthetic prediction"


def test_build_user_prompt_switches_after_halfway_point():
    base_request = OraclePredictionRequest(
        timestamp_ms=500,
        conversation_context="A: hello there",
        next_utterance_hint="fine thanks",
        total_word_count=2,
        trigger_word="there",
        recent_words="hello there",
        current_spoken_ratio=0.5,
        current_speaker="A",
        next_speaker="B",
        target_channel=1,
    )

    low_prompt = build_user_prompt(base_request)
    high_prompt = build_user_prompt(replace(base_request, current_spoken_ratio=0.75))

    assert "based only on the conversation history" in low_prompt
    assert "Hidden hint" not in low_prompt
    assert "Hidden hint" in high_prompt
    assert 'similar to: "fine thanks"' in high_prompt


def test_fallback_prediction_respects_no_hint_first_half():
    early_request = OraclePredictionRequest(
        timestamp_ms=500,
        conversation_context="A: hello there",
        next_utterance_hint="fine thanks",
        total_word_count=2,
        trigger_word="there",
        recent_words="hello there",
        current_spoken_ratio=0.5,
        current_speaker="A",
        next_speaker="B",
        target_channel=1,
    )
    late_request = replace(early_request, current_spoken_ratio=0.75)

    assert fallback_prediction_for_request(early_request) == ""
    assert fallback_prediction_for_request(late_request) == "fine thanks"
