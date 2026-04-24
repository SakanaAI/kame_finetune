import argparse
import json
import multiprocessing as mp
import os
import warnings

import numpy as np
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


def _handle_alignment_issue(message: str, *, allow_alignment_warnings: bool) -> None:
    if allow_alignment_warnings:
        warnings.warn(message, stacklevel=2)
        return
    raise ValueError(message)


def _raise_if_any_worker_failed(processes: list[mp.Process]) -> None:
    failed_workers = [
        (index, process.exitcode)
        for index, process in enumerate(processes)
        if process.exitcode not in (0, None)
    ]
    if failed_workers:
        raise RuntimeError(
            "One or more tokenization workers failed: "
            + ", ".join(
                f"worker {index} exited with code {exitcode}" for index, exitcode in failed_workers
            )
        )


def _save_tokenized_text(output_path: str, token_ids_a: list[int], token_ids_b: list[int]) -> None:
    try:
        np.savez_compressed(output_path, A=token_ids_a, B=token_ids_b)
    except Exception as e:
        print(f"Error in saving {output_path}: {e}")
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
        raise


def encode_as_pieces_wo_byte_fallback(sp: SentencePieceProcessor, text: str) -> list[str]:
    """
    Tokenize the text without using byte fallback.
    """
    tokens = sp.encode_as_pieces(text)
    if not tokens:
        return []

    tokens_wo_byte = []
    last_byte_tokens = []
    for token in tokens:
        if not token.startswith("<0x"):
            tokens_wo_byte.append(token)
            text = text[len(token) :]
        else:
            last_byte_tokens.append(token)
            token = sp.decode_pieces(last_byte_tokens)
            if text.startswith(token):  # token is successfully decoded
                tokens_wo_byte.append(token)
                text = text[len(token) :]
                last_byte_tokens = []
    if last_byte_tokens:
        merged = "".join(last_byte_tokens)
        tokens_wo_byte.append(merged)
    return tokens_wo_byte


def tokenize_and_pad_text(
    word_transcript: list[dict[str, str | float]],
    no_whitespace_before_word: bool,
    text_tokenizer: SentencePieceProcessor,
    text_padding_id: int,
    end_of_text_padding_id: int,
    audio_tokenizer_frame_rate: float,
    allow_alignment_warnings: bool = False,
) -> list[int]:
    """
    Tokenize the word transcript of single speaker.
    Fill the appropriate frames with the tokens based on the word-level timestamps,
    and frames without tokens are filled with the padding token.
    """
    # assert single speaker
    speakers = [seg["speaker"] for seg in word_transcript]
    assert len(set(speakers)) == 1

    # sort the word transcript by the start time
    word_transcript = sorted(word_transcript, key=lambda x: x["start"])

    # add whitespace to the beginning of each transcript word
    if not no_whitespace_before_word:
        # ensure that the first word has no whitespace before it
        word_transcript[0]["word"] = word_transcript[0]["word"].strip()
        for seg in word_transcript[1:]:
            seg["word"] = " " + seg["word"].strip()

    # tokenize the text
    text = "".join([seg["word"] for seg in word_transcript])
    tokens = encode_as_pieces_wo_byte_fallback(text_tokenizer, text)

    # word-level transcript to character-level transcript
    char_transcript = []
    for seg in word_transcript:
        num_chars = len(seg["word"])
        if num_chars == 0:
            continue
        start = seg["start"]
        end = seg["end"]
        # split the duration into num_chars
        char_duration = (end - start) / num_chars
        for i, char in enumerate(seg["word"]):
            char_transcript.append(
                {
                    "speaker": seg["speaker"],
                    "start": start + i * char_duration,
                    "end": start + (i + 1) * char_duration,
                    "char": char,
                }
            )

    # make token-level transcript by aligning the timestamps
    token_transcript = []
    for i, token in enumerate(tokens):
        if i == 0 and token == "▁":
            # skip the first underscore of sentencepiece
            continue
        if i == 0 and token.startswith("▁"):
            # don't count the first underscore
            chars = char_transcript[: len(token) - 1]
        else:
            chars = char_transcript[: len(token)]

        if not chars:
            _handle_alignment_issue(
                f"Empty chars for token '{token}', skipping token.",
                allow_alignment_warnings=allow_alignment_warnings,
            )
            continue

        token_transcript.append(
            {
                "speaker": chars[0]["speaker"],
                "start": chars[0]["start"],
                "end": chars[-1]["end"],
                "token": token,
            }
        )
        # remove the characters that are already processed
        char_transcript = char_transcript[len(chars) :]
    if char_transcript:
        _handle_alignment_issue(
            f"Remaining {len(char_transcript)} characters after tokenization "
            f"(likely caused by byte-fallback alignment): {char_transcript}",
            allow_alignment_warnings=allow_alignment_warnings,
        )

    # make tokenized ids with padding
    if token_transcript:
        num_frames = int((token_transcript[-1]["end"] + 1) * audio_tokenizer_frame_rate)
    else:
        num_frames = 0
    seconds_per_frame = 1 / audio_tokenizer_frame_rate
    token_ids = [text_padding_id] * num_frames
    token_count = 0
    for seg in token_transcript:
        frame_index = int(seg["start"] // seconds_per_frame)
        try:
            # find the next padding index to insert the token
            while token_ids[frame_index] != text_padding_id:
                frame_index += 1
        except IndexError:
            _handle_alignment_issue(
                f"Last {len(token_transcript) - token_count} tokens out of {len(token_transcript)} tokens "
                f"are dropped due to the insufficient number of frames ({num_frames}).",
                allow_alignment_warnings=allow_alignment_warnings,
            )
            break
        token_ids[frame_index] = text_tokenizer.piece_to_id(seg["token"])
        token_count += 1
        if frame_index > 0 and token_ids[frame_index - 1] == text_padding_id:
            # insert end_of_text_padding_id
            token_ids[frame_index - 1] = end_of_text_padding_id
    return token_ids


def worker(process_id: int, dialogue_names: list[str], args: argparse.Namespace):
    sp = SentencePieceProcessor(hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name))
    skipped = []
    pbar = tqdm(dialogue_names, desc=f"Worker {process_id}", dynamic_ncols=True)
    for dialogue_name in pbar:
        pbar.set_postfix_str(dialogue_name)

        # load word-level transcript
        with open(os.path.join(args.word_transcript_dir, f"{dialogue_name}.json")) as f:
            word_transcript = json.load(f)

        # tokenize text
        word_transcript_A = [seg for seg in word_transcript if seg["speaker"] == "A"]
        word_transcript_B = [seg for seg in word_transcript if seg["speaker"] == "B"]

        if not word_transcript_A or not word_transcript_B:
            missing = []
            if not word_transcript_A:
                missing.append("A")
            if not word_transcript_B:
                missing.append("B")
            if getattr(args, "allow_missing_speakers", False):
                print(f"[SKIP] {dialogue_name}: missing speaker {', '.join(missing)}")
                skipped.append(dialogue_name)
                continue
            raise ValueError(f"{dialogue_name}: missing speaker {', '.join(missing)}")

        token_ids_A = tokenize_and_pad_text(
            word_transcript=word_transcript_A,
            no_whitespace_before_word=args.no_whitespace_before_word,
            text_tokenizer=sp,
            text_padding_id=args.text_padding_id,
            end_of_text_padding_id=args.end_of_text_padding_id,
            audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
            allow_alignment_warnings=getattr(args, "allow_alignment_warnings", False),
        )
        token_ids_B = tokenize_and_pad_text(
            word_transcript=word_transcript_B,
            no_whitespace_before_word=args.no_whitespace_before_word,
            text_tokenizer=sp,
            text_padding_id=args.text_padding_id,
            end_of_text_padding_id=args.end_of_text_padding_id,
            audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
            allow_alignment_warnings=getattr(args, "allow_alignment_warnings", False),
        )

        # save the tokenized text
        output_path = os.path.join(args.output_dir, f"{dialogue_name}.npz")
        _save_tokenized_text(output_path, token_ids_A, token_ids_B)

    # report skipped files
    if skipped:
        print(f"[Worker {process_id}] Skipped {len(skipped)} files: {skipped}")


def main(args):
    dialogue_names = [
        os.path.splitext(d)[0] for d in os.listdir(args.word_transcript_dir) if d.endswith(".json")
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume:
        tokenized_dialogue_names = [
            os.path.splitext(d)[0] for d in os.listdir(args.output_dir) if d.endswith(".npz")
        ]
        print(f"Skipping {len(tokenized_dialogue_names)} already tokenized dialogues.")
        dialogue_names = list(set(dialogue_names) - set(tokenized_dialogue_names))

    if args.num_workers == 1:
        worker(0, dialogue_names, args)

    else:
        dialogue_names_per_worker = np.array_split(dialogue_names, args.num_workers)
        print(
            f"Each of {args.num_workers} workers processes {len(dialogue_names_per_worker[0])} dialogues."
        )

        processes = []
        for i, dialogue_names in enumerate(dialogue_names_per_worker):
            p = mp.Process(target=worker, args=(i, dialogue_names, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        _raise_if_any_worker_failed(processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize word-level transcripts using a text tokenizer."
    )
    parser.add_argument(
        "--word_transcript_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the transcripts with word level timestamps. "
            "Each file should contain a list of dictionaries "
            "(`{{'speaker': str, 'start': float, 'end': float, 'word': str}}`) "
            "and the filename should be the same as the corresponding audio file."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the tokenized data.",
    )

    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Repository of the text tokenizer.",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="tokenizer_spm_32k_3.model",
        help="Name of the text tokenizer.",
    )

    parser.add_argument(
        "--no_whitespace_before_word",
        action="store_true",
        help=(
            "No whitespace before each word. Set this flag if the language "
            "has no whitespace between words (e.g., Japanese and Chinese)."
        ),
    )

    parser.add_argument("--text_padding_id", type=int, default=3, help="Padding id for text.")
    parser.add_argument(
        "--end_of_text_padding_id", type=int, default=0, help="End of text padding id."
    )
    parser.add_argument(
        "--audio_tokenizer_frame_rate",
        type=float,
        default=12.5,
        help="Frame rate for the audio tokenizer.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )
    parser.add_argument("--resume", action="store_true", help="Resume tokenization.")
    parser.add_argument(
        "--allow_missing_speakers",
        action="store_true",
        help="Skip dialogues with a missing speaker instead of failing immediately.",
    )
    parser.add_argument(
        "--allow_alignment_warnings",
        action="store_true",
        help="Warn and continue on token/character alignment mismatches instead of failing immediately.",
    )

    args = parser.parse_args()

    main(args)
