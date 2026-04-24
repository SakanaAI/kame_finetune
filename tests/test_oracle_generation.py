import pytest

from tools.oracle_generation import (
    OracleGenerator,
    OraclePredictionRequest,
    TurtleAlignment,
    Word,
    merge_word_streams,
    words_from_turtle_alignments,
    words_from_word_transcript,
)

# Core oracle-generation tests: timing, speaker transitions, and canonical A/B behavior.


def test_oracle_generator_switches_hint_after_halfway_point():
    requests: list[OraclePredictionRequest] = []

    def predict_fn(request: OraclePredictionRequest) -> str:
        requests.append(request)
        return f"pred-{request.current_spoken_ratio:.2f}"

    words = words_from_word_transcript(
        [
            {"speaker": "A", "word": "hello", "start": 0.0, "end": 0.4},
            {"speaker": "A", "word": "there", "start": 0.4, "end": 0.8},
            {"speaker": "A", "word": "how", "start": 0.8, "end": 1.2},
            {"speaker": "A", "word": "today", "start": 1.2, "end": 1.6},
            {"speaker": "B", "word": "fine", "start": 1.7, "end": 2.1},
            {"speaker": "B", "word": "thanks", "start": 2.1, "end": 2.5},
        ]
    )

    generator = OracleGenerator(
        predict_fn,
        time_interval=0.5,
    )
    predictions = generator.generate_predictions(words)

    assert len(predictions) == 3
    assert [prediction.channel for prediction in predictions] == [1, 1, 1]

    assert predictions[0].current_spoken_ratio == pytest.approx(0.5)
    assert predictions[0].hint == ""
    assert predictions[1].current_spoken_ratio == pytest.approx(0.75)
    assert predictions[1].hint == "fine thanks"
    assert predictions[2].current_spoken_ratio == pytest.approx(1.0)
    assert predictions[2].hint == "fine thanks"

    assert requests[0].conversation_context == "A: hello there"
    assert requests[1].next_utterance_hint == "fine thanks"
    assert requests[2].target_channel == 1


def test_word_adapters_share_the_same_word_model():
    transcript_words = words_from_word_transcript(
        [{"speaker": "A", "word": "later", "start": 0.3, "end": 0.7}]
    )
    turtle_alignments: list[TurtleAlignment] = [("first", (0.0, 0.2), "SPEAKER_MAIN")]
    turtle_words = words_from_turtle_alignments(turtle_alignments, speaker="B")

    merged = merge_word_streams(transcript_words, turtle_words)

    assert [(word.speaker, word.text) for word in merged] == [
        ("B", "first"),
        ("A", "later"),
    ]


def test_current_speaker_info_uses_the_exact_current_word_instance():
    generator = OracleGenerator(lambda _request: "")
    all_words = [
        Word(text="hello", start_time=0.0, end_time=0.5, speaker="A"),
        Word(text="hello", start_time=0.1, end_time=0.5, speaker="B"),
        Word(text="there", start_time=0.5, end_time=0.8, speaker="B"),
    ]

    speaker_info = generator._get_current_speaker_info(
        words_so_far=all_words[:2],
        current_time=0.5,
        all_words=all_words,
    )

    assert speaker_info["speaker"] == "B"
    assert speaker_info["ratio"] == pytest.approx(0.5)


def test_oracle_generator_requires_canonical_ab_speaker_mapping():
    with pytest.raises(ValueError, match="speaker_to_channel"):
        OracleGenerator(lambda _request: "", speaker_to_channel={"USER": 0, "ASSISTANT": 1})


def test_oracle_generator_rejects_non_ab_transcripts():
    generator = OracleGenerator(lambda _request: "")
    words = [
        Word(text="hello", start_time=0.0, end_time=0.4, speaker="USER"),
        Word(text="there", start_time=0.4, end_time=0.8, speaker="ASSISTANT"),
    ]

    with pytest.raises(ValueError, match="canonical A/B transcripts"):
        generator.generate_predictions(words)
