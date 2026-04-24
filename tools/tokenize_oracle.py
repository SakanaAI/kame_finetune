# tools/tokenize_oracle.py
import argparse
import json
import multiprocessing as mp
import os
from functools import reduce

import numpy as np
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

TOKENIZED_ORACLE_NPZ_KEYS = (
    "A_event_frame_pos",
    "A_event_ratio",
    "A_event_skip_forbid",
    "A_pred_values",
    "A_pred_offsets",
    "A_hint_values",
    "A_hint_offsets",
    "B_event_frame_pos",
    "B_event_ratio",
    "B_event_skip_forbid",
    "B_pred_values",
    "B_pred_offsets",
    "B_hint_values",
    "B_hint_offsets",
)


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


def _save_tokenized_oracle(output_path: str, a_evt: dict, b_evt: dict) -> None:
    payload = {
        "A_event_frame_pos": a_evt["event_frame_pos"],
        "A_event_ratio": a_evt["event_ratio"],
        "A_event_skip_forbid": a_evt["event_skip_forbid"],
        "A_pred_values": a_evt["pred_values"],
        "A_pred_offsets": a_evt["pred_offsets"],
        "A_hint_values": a_evt["hint_values"],
        "A_hint_offsets": a_evt["hint_offsets"],
        "B_event_frame_pos": b_evt["event_frame_pos"],
        "B_event_ratio": b_evt["event_ratio"],
        "B_event_skip_forbid": b_evt["event_skip_forbid"],
        "B_pred_values": b_evt["pred_values"],
        "B_pred_offsets": b_evt["pred_offsets"],
        "B_hint_values": b_evt["hint_values"],
        "B_hint_offsets": b_evt["hint_offsets"],
    }
    try:
        np.savez_compressed(output_path, **payload)
    except Exception as e:
        print(f"Failed to save {output_path}: {e}")
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
        raise


def _is_valid_tokenized_oracle(path: str, *, validate_contents: bool = False) -> bool:
    try:
        with np.load(path, allow_pickle=False) as npz:
            missing_keys = [key for key in TOKENIZED_ORACLE_NPZ_KEYS if key not in npz.files]
            if missing_keys:
                print(f"Invalid tokenized oracle file {path}: missing keys {missing_keys}")
                return False
            if validate_contents:
                for key in TOKENIZED_ORACLE_NPZ_KEYS:
                    _ = npz[key]
    except Exception as e:
        print(f"Invalid tokenized oracle file {path}: {e}")
        return False
    return True


def tokenize(
    tokenizer: SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
) -> list[int]:
    """
    Tokenize a possibly multi-line string in an Interleaver-compatible way.

    Why not `tokenizer.encode(text)` directly?
      - We want explicit line boundaries: encode each line separately and insert the tokenizer's
        newline piece between lines (line1 + [NL] + line2 + ...), matching Interleaver behavior.
      - Some SentencePiece models may return an empty encoding for "\\n"; this function falls back
        to concatenating lines without an NL token in that case.
    """
    # Safely get newline token
    nl_encoded = tokenizer.encode("\n")
    nl_piece: int | None = nl_encoded[-1] if nl_encoded else None

    lines = text.split("\n")
    if alpha is not None:
        encoded_lines = tokenizer.encode(lines, enable_sampling=True, alpha=alpha, nbest_size=-1)
    else:
        encoded_lines = tokenizer.encode(lines)

    if not encoded_lines:
        tokens: list[int] = []
    elif nl_piece is None:
        # If there is no dedicated newline token: concatenate lines directly
        tokens = [t for line in encoded_lines for t in line]
    else:
        # Same behavior as Interleaver: line1 + [nl] + line2 + ...
        tokens = reduce(lambda a, b: [*a, nl_piece, *b], encoded_lines)

    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


def _pack_ragged_tokens(token_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack list[list[int]] into (values, offsets) without pickle.

    values: int32[sum(len(tokens))]
    offsets: int32[len(token_lists)+1], offsets[0]=0
    """
    offsets = np.zeros((len(token_lists) + 1,), dtype=np.int32)
    total = 0
    for i, toks in enumerate(token_lists):
        total += len(toks)
        offsets[i + 1] = total

    values = np.empty((total,), dtype=np.int32)
    for i, toks in enumerate(token_lists):
        start = offsets[i]
        end = offsets[i + 1]
        if end != start:
            values[start:end] = toks
    return values, offsets


def build_oracle_events_for_channel(
    oracle_data: list[dict],
    ch_number: int,
    num_frames: int,
    tokenizer: SentencePieceProcessor,
    frame_rate: float,
) -> dict[str, np.ndarray]:
    """
    Build event-level oracle representation (no randomness).
    This preserves enough information to reconstruct frame-level oracle with skip/jitter/shift at training time.

    Returns a dict of numpy arrays safe to store in npz without pickle.
    """
    event_frame_pos: list[int] = []
    event_ratio: list[float] = []
    event_skip_forbid: list[int] = []  # 1 if skip is forbidden for this event, else 0

    pred_token_lists: list[list[int]] = []
    hint_token_lists: list[list[int]] = []

    # IMPORTANT: iterate with original ordering to compute skip_forbid like Turtle:
    # "If the next entry is the same channel, this is not the last entry."
    for idx, entry in enumerate(oracle_data):
        if int(entry.get("channel", 0)) != ch_number:
            continue

        # Turtle-style "last entry" is "next entry is different channel" (end of a run)
        is_last_entry = (
            idx == len(oracle_data) - 1 or int(oracle_data[idx + 1].get("channel", 0)) != ch_number
        )

        ts_ms = entry.get("timestamp_ms", None)
        if ts_ms is None:
            continue

        timestamp_sec = float(ts_ms) / 1000.0
        frame_pos = int(timestamp_sec * frame_rate)
        if frame_pos < 0 or frame_pos >= num_frames:
            # TODO: Consider migrating to logger for consistency with other modules.
            print(
                f"WARNING: Oracle event is outside audio range and will be ignored. "
                f"channel={ch_number} timestamp_ms={ts_ms} frame_pos={frame_pos} num_frames={num_frames}"
            )
            continue

        ratio = entry.get("current_spoken_ratio", None)
        try:
            ratio_f = float(ratio) if ratio is not None else 0.0
        except (ValueError, TypeError):
            ratio_f = 0.0

        pred_text = (entry.get("prediction", "") or "").strip()
        hint_text = (entry.get("hint", "") or "").strip()

        pred_tokens = tokenize(tokenizer, pred_text, bos=False) if pred_text else []
        hint_tokens = tokenize(tokenizer, hint_text, bos=False) if hint_text else []

        # Drop events with no usable tokens at all
        if not pred_tokens and not hint_tokens:
            continue

        event_frame_pos.append(frame_pos)
        event_ratio.append(ratio_f)
        event_skip_forbid.append(1 if is_last_entry else 0)
        pred_token_lists.append(pred_tokens)
        hint_token_lists.append(hint_tokens)

    pred_values, pred_offsets = _pack_ragged_tokens(pred_token_lists)
    hint_values, hint_offsets = _pack_ragged_tokens(hint_token_lists)

    return {
        "event_frame_pos": np.asarray(event_frame_pos, dtype=np.int32),
        "event_ratio": np.asarray(event_ratio, dtype=np.float32),
        "event_skip_forbid": np.asarray(event_skip_forbid, dtype=np.int8),
        "pred_values": pred_values,
        "pred_offsets": pred_offsets,
        "hint_values": hint_values,
        "hint_offsets": hint_offsets,
    }


def worker(process_id: int, dialogue_names: list[str], args: argparse.Namespace) -> None:
    sp = SentencePieceProcessor(hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name))

    pbar = tqdm(dialogue_names, desc=f"Worker {process_id}", dynamic_ncols=True)
    for dialogue_name in pbar:
        pbar.set_postfix_str(dialogue_name)

        audio_npz_path = os.path.join(args.tokenized_audio_dir, f"{dialogue_name}.npz")
        if not os.path.exists(audio_npz_path):
            raise FileNotFoundError(
                f"Missing tokenized audio for dialogue '{dialogue_name}': {audio_npz_path}"
            )
        audio_ids = np.load(audio_npz_path)
        arrA = audio_ids["A"]
        num_frames = arrA.shape[0] if arrA.ndim == 1 else arrA.shape[-1]

        oracle_path = os.path.join(
            args.oracle_dir,
            f"{dialogue_name}{args.oracle_suffix}",
        )
        if not os.path.exists(oracle_path):
            raise FileNotFoundError(
                f"Missing oracle transcript for dialogue '{dialogue_name}': {oracle_path}"
            )

        with open(oracle_path) as f:
            oracle_data = json.load(f)

        # events (no randomness)
        A_evt = build_oracle_events_for_channel(
            oracle_data=oracle_data,
            ch_number=args.A_channel,
            num_frames=num_frames,
            tokenizer=sp,
            frame_rate=args.audio_tokenizer_frame_rate,
        )
        B_evt = build_oracle_events_for_channel(
            oracle_data=oracle_data,
            ch_number=args.B_channel,
            num_frames=num_frames,
            tokenizer=sp,
            frame_rate=args.audio_tokenizer_frame_rate,
        )

        out_path = os.path.join(args.output_dir, f"{dialogue_name}.npz")
        _save_tokenized_oracle(out_path, A_evt, B_evt)


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    # Use audio npz as the basis to decide dialogue_name
    dialogue_names = [
        os.path.splitext(d)[0] for d in os.listdir(args.tokenized_audio_dir) if d.endswith(".npz")
    ]

    if args.resume:
        # Skip already tokenized dialogues.
        tokenized_dialogue_names = []
        invalid_count = 0
        for filename in os.listdir(args.output_dir):
            if not filename.endswith(".npz"):
                continue
            output_path = os.path.join(args.output_dir, filename)
            if _is_valid_tokenized_oracle(
                output_path,
                validate_contents=getattr(args, "validate_resume_contents", False),
            ):
                tokenized_dialogue_names.append(os.path.splitext(filename)[0])
                continue
            invalid_count += 1
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass
        print(f"Skipping {len(tokenized_dialogue_names)} already tokenized dialogues.")
        if invalid_count:
            print(f"Reprocessing {invalid_count} invalid tokenized oracle files.")
        tokenized_dialogue_names_set = set(tokenized_dialogue_names)
        dialogue_names = [
            name for name in dialogue_names if name not in tokenized_dialogue_names_set
        ]

    if args.num_workers == 1:
        worker(0, dialogue_names, args)
    else:
        dialogue_names_per_worker = np.array_split(dialogue_names, args.num_workers)
        print(
            f"Each of {args.num_workers} workers processes "
            f"{len(dialogue_names_per_worker[0])} dialogues."
        )
        processes = []
        for i, names in enumerate(dialogue_names_per_worker):
            p = mp.Process(target=worker, args=(i, list(names), args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        _raise_if_any_worker_failed(processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize oracle predictions into event-level oracle records."
    )
    parser.add_argument(
        "--oracle_dir",
        type=str,
        required=True,
        help="Directory containing oracle JSON files.",
    )
    parser.add_argument(
        "--tokenized_audio_dir",
        type=str,
        required=True,
        help="Directory containing tokenized audio *.npz.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save oracle token *.npz.",
    )

    parser.add_argument(
        "--oracle_suffix",
        type=str,
        default="_oracle_predictions_curated.json",
    )

    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="tokenizer_spm_32k_3.model",
    )

    parser.add_argument(
        "--audio_tokenizer_frame_rate",
        type=float,
        default=12.5,
    )

    parser.add_argument("--A_channel", type=int, default=0)
    parser.add_argument("--B_channel", type=int, default=1)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate_resume_contents",
        action="store_true",
        help=(
            "When resuming, read every array in existing oracle npz files to catch rare "
            "member-level corruption. By default resume only opens each archive and checks "
            "required member names, which is much faster on large corpora."
        ),
    )

    args = parser.parse_args()
    main(args)
