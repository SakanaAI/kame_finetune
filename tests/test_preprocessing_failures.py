import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.prepare_dataset as prepare_dataset
import tools.tokenize_audio as tokenize_audio
import tools.tokenize_oracle as tokenize_oracle
import tools.tokenize_text as tokenize_text


def test_tokenize_text_worker_reraises_save_failures(tmp_path, monkeypatch):
    word_transcript_dir = tmp_path / "word_transcripts"
    output_dir = tmp_path / "tokenized"
    word_transcript_dir.mkdir()
    output_dir.mkdir()
    (word_transcript_dir / "dialogue.json").write_text(
        json.dumps(
            [
                {"speaker": "A", "start": 0.0, "end": 0.1, "word": "hello"},
                {"speaker": "B", "start": 0.0, "end": 0.1, "word": "world"},
            ]
        )
    )

    args = SimpleNamespace(
        word_transcript_dir=str(word_transcript_dir),
        output_dir=str(output_dir),
        text_tokenizer_repo="unused/repo",
        text_tokenizer_name="unused.model",
        no_whitespace_before_word=True,
        text_padding_id=3,
        end_of_text_padding_id=0,
        audio_tokenizer_frame_rate=12.5,
        allow_alignment_warnings=False,
    )

    monkeypatch.setattr(tokenize_text, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_text, "SentencePieceProcessor", lambda *args, **kwargs: object())
    monkeypatch.setattr(tokenize_text, "tokenize_and_pad_text", lambda **kwargs: [1, 2])

    def fail_save(path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(tokenize_text.np, "savez_compressed", fail_save)

    with pytest.raises(OSError, match="disk full"):
        tokenize_text.worker(0, ["dialogue"], args)

    assert not (output_dir / "dialogue.npz").exists()


def test_tokenize_audio_worker_reraises_save_failures(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "tokenized"
    audio_dir.mkdir()
    output_dir.mkdir()
    (audio_dir / "dialogue.wav").write_bytes(b"unused")

    args = SimpleNamespace(
        audio_dir=str(audio_dir),
        output_dir=str(output_dir),
        audio_tokenizer_repo="unused/repo",
        audio_tokenizer_name="unused.model",
        audio_chunk_size=1,
    )

    class DummyResampler:
        def to(self, device):
            return self

        def __call__(self, wavs):
            return wavs

    monkeypatch.setattr(tokenize_audio, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(
        tokenize_audio.loaders,
        "get_mimi",
        lambda *args, **kwargs: SimpleNamespace(sample_rate=24000),
    )
    monkeypatch.setattr(
        tokenize_audio.torchaudio, "load", lambda *args, **kwargs: (torch.zeros(2, 4), 24000)
    )
    monkeypatch.setattr(
        tokenize_audio.torchaudio.transforms, "Resample", lambda *args, **kwargs: DummyResampler()
    )
    real_torch_device = torch.device
    monkeypatch.setattr(
        tokenize_audio.torch, "device", lambda *args, **kwargs: real_torch_device("cpu")
    )
    monkeypatch.setattr(
        tokenize_audio,
        "tokenize_audio",
        lambda *args, **kwargs: torch.ones((8, 2), dtype=torch.long),
    )

    def fail_save(path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(tokenize_audio.np, "savez_compressed", fail_save)

    with pytest.raises(OSError, match="disk full"):
        tokenize_audio.worker(0, ["dialogue"], args)

    assert not (output_dir / "dialogue.npz").exists()


def test_tokenize_oracle_worker_reraises_save_failures(tmp_path, monkeypatch):
    tokenized_audio_dir = tmp_path / "tokenized_audio"
    oracle_dir = tmp_path / "oracle_raw"
    output_dir = tmp_path / "tokenized_oracle"
    tokenized_audio_dir.mkdir()
    oracle_dir.mkdir()
    output_dir.mkdir()
    np.savez_compressed(tokenized_audio_dir / "dialogue.npz", A=np.ones((8, 2), dtype=np.int32))
    (oracle_dir / "dialogue.json").write_text("[]")

    args = SimpleNamespace(
        tokenized_audio_dir=str(tokenized_audio_dir),
        oracle_dir=str(oracle_dir),
        output_dir=str(output_dir),
        text_tokenizer_repo="unused/repo",
        text_tokenizer_name="unused.model",
        oracle_suffix=".json",
        A_channel=0,
        B_channel=1,
        audio_tokenizer_frame_rate=12.5,
    )

    monkeypatch.setattr(tokenize_oracle, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_oracle, "SentencePieceProcessor", lambda *args, **kwargs: object())

    def fail_save(path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(tokenize_oracle.np, "savez_compressed", fail_save)

    with pytest.raises(OSError, match="disk full"):
        tokenize_oracle.worker(0, ["dialogue"], args)

    assert not (output_dir / "dialogue.npz").exists()


def test_tokenize_oracle_worker_raises_when_audio_is_missing(tmp_path, monkeypatch):
    args = SimpleNamespace(
        tokenized_audio_dir=str(tmp_path / "tokenized_audio"),
        oracle_dir=str(tmp_path / "oracle_raw"),
        output_dir=str(tmp_path / "tokenized_oracle"),
        text_tokenizer_repo="unused/repo",
        text_tokenizer_name="unused.model",
        oracle_suffix=".json",
        A_channel=0,
        B_channel=1,
        audio_tokenizer_frame_rate=12.5,
    )
    Path(args.tokenized_audio_dir).mkdir()
    Path(args.oracle_dir).mkdir()
    Path(args.output_dir).mkdir()

    monkeypatch.setattr(tokenize_oracle, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_oracle, "SentencePieceProcessor", lambda *args, **kwargs: object())

    with pytest.raises(FileNotFoundError, match="Missing tokenized audio"):
        tokenize_oracle.worker(0, ["dialogue"], args)


def test_tokenize_oracle_worker_raises_when_oracle_is_missing(tmp_path, monkeypatch):
    args = SimpleNamespace(
        tokenized_audio_dir=str(tmp_path / "tokenized_audio"),
        oracle_dir=str(tmp_path / "oracle_raw"),
        output_dir=str(tmp_path / "tokenized_oracle"),
        text_tokenizer_repo="unused/repo",
        text_tokenizer_name="unused.model",
        oracle_suffix=".json",
        A_channel=0,
        B_channel=1,
        audio_tokenizer_frame_rate=12.5,
    )
    tokenized_audio_dir = Path(args.tokenized_audio_dir)
    oracle_dir = Path(args.oracle_dir)
    output_dir = Path(args.output_dir)
    tokenized_audio_dir.mkdir()
    oracle_dir.mkdir()
    output_dir.mkdir()
    np.savez_compressed(tokenized_audio_dir / "dialogue.npz", A=np.ones((8, 2), dtype=np.int32))

    monkeypatch.setattr(tokenize_oracle, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_oracle, "SentencePieceProcessor", lambda *args, **kwargs: object())

    with pytest.raises(FileNotFoundError, match="Missing oracle transcript"):
        tokenize_oracle.worker(0, ["dialogue"], args)


def test_tokenize_oracle_resume_reprocesses_invalid_npz(tmp_path, monkeypatch):
    tokenized_audio_dir = tmp_path / "tokenized_audio"
    oracle_dir = tmp_path / "oracle_raw"
    output_dir = tmp_path / "tokenized_oracle"
    tokenized_audio_dir.mkdir()
    oracle_dir.mkdir()
    output_dir.mkdir()
    np.savez_compressed(tokenized_audio_dir / "dialogue.npz", A=np.ones((8, 2), dtype=np.int32))
    (oracle_dir / "dialogue.json").write_text("[]")
    (output_dir / "dialogue.npz").write_bytes(b"partial")

    args = SimpleNamespace(
        tokenized_audio_dir=str(tokenized_audio_dir),
        oracle_dir=str(oracle_dir),
        output_dir=str(output_dir),
        text_tokenizer_repo="unused/repo",
        text_tokenizer_name="unused.model",
        oracle_suffix=".json",
        A_channel=0,
        B_channel=1,
        audio_tokenizer_frame_rate=12.5,
        resume=True,
        num_workers=1,
    )

    monkeypatch.setattr(tokenize_oracle, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_oracle, "SentencePieceProcessor", lambda *args, **kwargs: object())

    tokenize_oracle.main(args)

    with np.load(output_dir / "dialogue.npz", allow_pickle=False) as npz:
        assert set(tokenize_oracle.TOKENIZED_ORACLE_NPZ_KEYS).issubset(npz.files)


def test_tokenize_oracle_resume_reprocesses_npz_with_missing_keys(tmp_path, monkeypatch):
    tokenized_audio_dir = tmp_path / "tokenized_audio"
    oracle_dir = tmp_path / "oracle_raw"
    output_dir = tmp_path / "tokenized_oracle"
    tokenized_audio_dir.mkdir()
    oracle_dir.mkdir()
    output_dir.mkdir()
    np.savez_compressed(tokenized_audio_dir / "dialogue.npz", A=np.ones((8, 2), dtype=np.int32))
    (oracle_dir / "dialogue.json").write_text("[]")
    np.savez_compressed(output_dir / "dialogue.npz", A_event_frame_pos=np.array([], dtype=np.int32))

    args = SimpleNamespace(
        tokenized_audio_dir=str(tokenized_audio_dir),
        oracle_dir=str(oracle_dir),
        output_dir=str(output_dir),
        text_tokenizer_repo="unused/repo",
        text_tokenizer_name="unused.model",
        oracle_suffix=".json",
        A_channel=0,
        B_channel=1,
        audio_tokenizer_frame_rate=12.5,
        resume=True,
        num_workers=1,
    )

    monkeypatch.setattr(tokenize_oracle, "hf_hub_download", lambda *args, **kwargs: "unused.model")
    monkeypatch.setattr(tokenize_oracle, "SentencePieceProcessor", lambda *args, **kwargs: object())

    tokenize_oracle.main(args)

    with np.load(output_dir / "dialogue.npz", allow_pickle=False) as npz:
        assert set(tokenize_oracle.TOKENIZED_ORACLE_NPZ_KEYS).issubset(npz.files)


def test_prepare_dataset_filters_non_npz_and_allows_output_prefix_without_dir(
    tmp_path, monkeypatch
):
    tokenized_text_dir = tmp_path / "tokenized_text"
    tokenized_audio_dir = tmp_path / "tokenized_audio"
    tokenized_text_dir.mkdir()
    tokenized_audio_dir.mkdir()

    np.savez_compressed(
        tokenized_text_dir / "dialogue.npz",
        A=np.array([1, 2], dtype=np.int32),
        B=np.array([3, 4], dtype=np.int32),
    )
    np.savez_compressed(
        tokenized_audio_dir / "dialogue.npz",
        A=np.array([[10, 11]], dtype=np.int32),
        B=np.array([[12, 13]], dtype=np.int32),
    )
    (tokenized_text_dir / ".DS_Store").write_text("noise")

    monkeypatch.chdir(tmp_path)

    def fake_to_parquet(self, path, index=False):
        Path(path).write_text("ok")

    monkeypatch.setattr(prepare_dataset.pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)

    args = SimpleNamespace(
        tokenized_text_dir=str(tokenized_text_dir),
        tokenized_audio_dir=str(tokenized_audio_dir),
        tokenized_oracle_dir=None,
        output_prefix="train",
        text_padding_id=3,
        num_examples_per_parquet=100_000,
    )

    prepare_dataset.main(args)

    assert (tmp_path / "train-001-of-001.parquet").exists()


def test_prepare_dataset_raises_on_text_audio_mismatch(tmp_path, monkeypatch):
    tokenized_text_dir = tmp_path / "tokenized_text"
    tokenized_audio_dir = tmp_path / "tokenized_audio"
    tokenized_text_dir.mkdir()
    tokenized_audio_dir.mkdir()
    np.savez_compressed(
        tokenized_text_dir / "dialogue.npz",
        A=np.array([1, 2], dtype=np.int32),
        B=np.array([3, 4], dtype=np.int32),
    )

    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        tokenized_text_dir=str(tokenized_text_dir),
        tokenized_audio_dir=str(tokenized_audio_dir),
        tokenized_oracle_dir=None,
        output_prefix="train",
        text_padding_id=3,
        num_examples_per_parquet=100_000,
    )

    with pytest.raises(ValueError, match="Both text and audio tokenized dialogues should match"):
        prepare_dataset.main(args)


def test_prepare_dataset_raises_on_missing_oracle(tmp_path, monkeypatch):
    tokenized_text_dir = tmp_path / "tokenized_text"
    tokenized_audio_dir = tmp_path / "tokenized_audio"
    tokenized_oracle_dir = tmp_path / "tokenized_oracle"
    tokenized_text_dir.mkdir()
    tokenized_audio_dir.mkdir()
    tokenized_oracle_dir.mkdir()
    np.savez_compressed(
        tokenized_text_dir / "dialogue.npz",
        A=np.array([1, 2], dtype=np.int32),
        B=np.array([3, 4], dtype=np.int32),
    )
    np.savez_compressed(
        tokenized_audio_dir / "dialogue.npz",
        A=np.array([[10, 11]], dtype=np.int32),
        B=np.array([[12, 13]], dtype=np.int32),
    )

    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        tokenized_text_dir=str(tokenized_text_dir),
        tokenized_audio_dir=str(tokenized_audio_dir),
        tokenized_oracle_dir=str(tokenized_oracle_dir),
        output_prefix="train",
        text_padding_id=3,
        num_examples_per_parquet=100_000,
    )

    with pytest.raises(ValueError, match="Tokenized oracle dialogues should match"):
        prepare_dataset.main(args)
