from types import SimpleNamespace

import pytest

import tools.tokenize_text as tokenize_text


class FakeTokenizer:
    def __init__(self, tokens):
        self._tokens = list(tokens)

    def encode_as_pieces(self, text):
        return list(self._tokens)

    def decode_pieces(self, pieces):
        return "".join(pieces)

    def piece_to_id(self, token):
        return len(token)


def _make_args(tmp_path, *, allow_missing_speakers=False, allow_alignment_warnings=False):
    return SimpleNamespace(
        word_transcript_dir=str(tmp_path / "word_transcripts"),
        output_dir=str(tmp_path / "tokenized"),
        text_tokenizer_repo="unused/repo",
        text_tokenizer_name="unused.model",
        no_whitespace_before_word=True,
        text_padding_id=3,
        end_of_text_padding_id=0,
        audio_tokenizer_frame_rate=12.5,
        num_workers=1,
        resume=False,
        allow_missing_speakers=allow_missing_speakers,
        allow_alignment_warnings=allow_alignment_warnings,
    )


def test_tokenize_and_pad_text_raises_alignment_error_by_default():
    tokenizer = FakeTokenizer(["a", "b"])
    word_transcript = [{"speaker": "A", "start": 0.0, "end": 1.0, "word": "a"}]

    with pytest.raises(ValueError, match="Empty chars for token 'b'"):
        tokenize_text.tokenize_and_pad_text(
            word_transcript=word_transcript,
            no_whitespace_before_word=True,
            text_tokenizer=tokenizer,
            text_padding_id=3,
            end_of_text_padding_id=0,
            audio_tokenizer_frame_rate=12.5,
        )


def test_tokenize_and_pad_text_allows_alignment_warnings_opt_in():
    tokenizer = FakeTokenizer(["a", "b"])
    word_transcript = [{"speaker": "A", "start": 0.0, "end": 1.0, "word": "a"}]

    with pytest.warns(UserWarning, match="Empty chars for token 'b'"):
        token_ids = tokenize_text.tokenize_and_pad_text(
            word_transcript=word_transcript,
            no_whitespace_before_word=True,
            text_tokenizer=tokenizer,
            text_padding_id=3,
            end_of_text_padding_id=0,
            audio_tokenizer_frame_rate=12.5,
            allow_alignment_warnings=True,
        )

    assert token_ids


def test_worker_raises_on_missing_speakers_by_default(tmp_path, monkeypatch):
    args = _make_args(tmp_path)
    word_transcript_dir = tmp_path / "word_transcripts"
    output_dir = tmp_path / "tokenized"
    word_transcript_dir.mkdir()
    output_dir.mkdir()
    (word_transcript_dir / "dialogue.json").write_text(
        '[{"speaker": "A", "start": 0.0, "end": 1.0, "word": "hello"}]'
    )

    monkeypatch.setattr(tokenize_text, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_text, "SentencePieceProcessor", lambda *args, **kwargs: object())

    with pytest.raises(ValueError, match="missing speaker B"):
        tokenize_text.worker(0, ["dialogue"], args)


def test_worker_skips_missing_speakers_when_allowed(tmp_path, monkeypatch):
    args = _make_args(tmp_path, allow_missing_speakers=True)
    word_transcript_dir = tmp_path / "word_transcripts"
    output_dir = tmp_path / "tokenized"
    word_transcript_dir.mkdir()
    output_dir.mkdir()
    (word_transcript_dir / "dialogue.json").write_text(
        '[{"speaker": "A", "start": 0.0, "end": 1.0, "word": "hello"}]'
    )

    monkeypatch.setattr(tokenize_text, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_text, "SentencePieceProcessor", lambda *args, **kwargs: object())

    tokenize_text.worker(0, ["dialogue"], args)

    assert list(output_dir.iterdir()) == []


def test_raise_if_any_worker_failed_rejects_nonzero_exitcodes():
    class DummyProcess:
        def __init__(self, exitcode):
            self.exitcode = exitcode

    with pytest.raises(RuntimeError, match="worker 1 exited with code 1"):
        tokenize_text._raise_if_any_worker_failed([DummyProcess(0), DummyProcess(1)])


def test_tokenize_and_pad_text_raises_on_token_drop_by_default():
    tokenizer = FakeTokenizer(["a"] * 13)
    word_transcript = [
        {"speaker": "A", "start": 0.0, "end": 0.0, "word": "a" * 13},
    ]

    with pytest.raises(ValueError, match="are dropped due to the insufficient number of frames"):
        tokenize_text.tokenize_and_pad_text(
            word_transcript=word_transcript,
            no_whitespace_before_word=True,
            text_tokenizer=tokenizer,
            text_padding_id=3,
            end_of_text_padding_id=0,
            audio_tokenizer_frame_rate=12.5,
        )


def test_tokenize_and_pad_text_allows_token_drop_warning_opt_in():
    tokenizer = FakeTokenizer(["a"] * 13)
    word_transcript = [
        {"speaker": "A", "start": 0.0, "end": 0.0, "word": "a" * 13},
    ]

    with pytest.warns(UserWarning, match="are dropped due to the insufficient number of frames"):
        token_ids = tokenize_text.tokenize_and_pad_text(
            word_transcript=word_transcript,
            no_whitespace_before_word=True,
            text_tokenizer=tokenizer,
            text_padding_id=3,
            end_of_text_padding_id=0,
            audio_tokenizer_frame_rate=12.5,
            allow_alignment_warnings=True,
        )

    assert token_ids == [1] * 12
