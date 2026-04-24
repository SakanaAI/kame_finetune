import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def merge_text_audio(
    text_ids: np.ndarray, audio_ids: np.ndarray, text_padding_id: int
) -> np.ndarray:
    """
    Merge the tokenized text and audio stream of a single speaker.
    Args:
        text_ids: Tokenized text stream. Shape: [T_text]
        audio_ids: Tokenized audio stream. Shape: [K=8, T_audio]
        text_padding_id: Padding id for text stream to fill the gap between audio and text streams.
    Returns:
        Merged tokenized text and audio stream. Shape: [K=8+1, T_audio]
    """
    assert text_ids.ndim == 1, f"Expected 1D tensor, got {text_ids.ndim}D tensor."
    assert audio_ids.ndim == 2, f"Expected 2D tensor, got {audio_ids.ndim}D tensor."
    # pad the text stream to match the audio stream
    audio_len = audio_ids.shape[-1]
    if text_ids.shape[0] > audio_len:
        text_ids = text_ids[:audio_len]
    elif text_ids.shape[0] < audio_len:
        text_ids = np.concat(
            [text_ids, np.full(audio_len - text_ids.shape[0], text_padding_id)], axis=0
        )
    return np.concat([text_ids[None], audio_ids], axis=0).astype(np.int32).tolist()


def _list_npz_dialogue_names(directory: str) -> list[str]:
    return [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".npz")]


def main(args):
    text_dialogue_names = _list_npz_dialogue_names(args.tokenized_text_dir)
    audio_dialogue_names = _list_npz_dialogue_names(args.tokenized_audio_dir)
    oracle_dialogue_names = (
        _list_npz_dialogue_names(args.tokenized_oracle_dir)
        if args.tokenized_oracle_dir is not None
        else None
    )
    missing_text_dialogue_names = set(audio_dialogue_names) - set(text_dialogue_names)
    missing_audio_dialogue_names = set(text_dialogue_names) - set(audio_dialogue_names)
    missing_oracle_dialogue_names = (
        set(text_dialogue_names) - set(oracle_dialogue_names)
        if oracle_dialogue_names is not None
        else set()
    )
    if missing_text_dialogue_names:
        print(f"Missing tokenized text for {len(missing_text_dialogue_names)} dialogues.")
        open("missing_text_dialogue_names.txt", "w").write("\n".join(missing_text_dialogue_names))
    if missing_audio_dialogue_names:
        print(f"Missing tokenized audio for {len(missing_audio_dialogue_names)} dialogues.")
        open("missing_audio_dialogue_names.txt", "w").write("\n".join(missing_audio_dialogue_names))
    if missing_oracle_dialogue_names:
        print(f"Missing tokenized oracle for {len(missing_oracle_dialogue_names)} dialogues.")
        open("missing_oracle_dialogue_names.txt", "w").write(
            "\n".join(missing_oracle_dialogue_names)
        )
    if missing_text_dialogue_names or missing_audio_dialogue_names:
        raise ValueError("Both text and audio tokenized dialogues should match.")
    if missing_oracle_dialogue_names:
        raise ValueError("Tokenized oracle dialogues should match tokenized text/audio dialogues.")

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    if oracle_dialogue_names is None:
        dialogue_names = sorted(set(text_dialogue_names) & set(audio_dialogue_names))
    else:
        dialogue_names = sorted(
            set(text_dialogue_names) & set(audio_dialogue_names) & set(oracle_dialogue_names)
        )

    num_dialogues = len(dialogue_names)
    num_parquets = -(-num_dialogues // args.num_examples_per_parquet)

    for i in range(num_parquets):
        dials_per_parquet = dialogue_names[
            i * args.num_examples_per_parquet : (i + 1) * args.num_examples_per_parquet
        ]

        # load the tokenized text and audio data
        data = []
        for dialogue_name in tqdm(
            dials_per_parquet, desc=f"Processing parquet {i + 1}/{num_parquets}"
        ):
            text_path = os.path.join(args.tokenized_text_dir, f"{dialogue_name}.npz")
            text_ids = np.load(text_path)
            audio_path = os.path.join(args.tokenized_audio_dir, f"{dialogue_name}.npz")
            audio_ids = np.load(audio_path)
            oracle_data = None
            if args.tokenized_oracle_dir is not None:
                oracle_path = os.path.join(args.tokenized_oracle_dir, f"{dialogue_name}.npz")
                oracle_data = np.load(oracle_path)
            rec = {
                "dialogue_id": os.path.join(args.output_prefix, dialogue_name),  # unique identifier
                "A": merge_text_audio(text_ids["A"], audio_ids["A"], args.text_padding_id),
                "B": merge_text_audio(text_ids["B"], audio_ids["B"], args.text_padding_id),
            }
            if oracle_data is not None:
                # Events-only oracle format
                required = [
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
                ]
                missing = [k for k in required if k not in oracle_data]
                if missing:
                    raise ValueError(
                        f"Oracle npz must be event-level, but missing keys={missing}. "
                        f"dialogue={dialogue_name}, keys={list(oracle_data.keys())}"
                    )

                rec["A_oracle_event_frame_pos"] = (
                    oracle_data["A_event_frame_pos"].astype(np.int32).tolist()
                )
                rec["A_oracle_event_ratio"] = (
                    oracle_data["A_event_ratio"].astype(np.float32).tolist()
                )
                rec["A_oracle_event_skip_forbid"] = (
                    oracle_data["A_event_skip_forbid"].astype(np.int8).tolist()
                )
                rec["A_oracle_pred_values"] = oracle_data["A_pred_values"].astype(np.int32).tolist()
                rec["A_oracle_pred_offsets"] = (
                    oracle_data["A_pred_offsets"].astype(np.int32).tolist()
                )
                rec["A_oracle_hint_values"] = oracle_data["A_hint_values"].astype(np.int32).tolist()
                rec["A_oracle_hint_offsets"] = (
                    oracle_data["A_hint_offsets"].astype(np.int32).tolist()
                )

                rec["B_oracle_event_frame_pos"] = (
                    oracle_data["B_event_frame_pos"].astype(np.int32).tolist()
                )
                rec["B_oracle_event_ratio"] = (
                    oracle_data["B_event_ratio"].astype(np.float32).tolist()
                )
                rec["B_oracle_event_skip_forbid"] = (
                    oracle_data["B_event_skip_forbid"].astype(np.int8).tolist()
                )
                rec["B_oracle_pred_values"] = oracle_data["B_pred_values"].astype(np.int32).tolist()
                rec["B_oracle_pred_offsets"] = (
                    oracle_data["B_pred_offsets"].astype(np.int32).tolist()
                )
                rec["B_oracle_hint_values"] = oracle_data["B_hint_values"].astype(np.int32).tolist()
                rec["B_oracle_hint_offsets"] = (
                    oracle_data["B_hint_offsets"].astype(np.int32).tolist()
                )
            data.append(rec)

        # save the merged data
        df = pd.DataFrame(data)
        output_path = f"{args.output_prefix}-{i + 1:03d}-of-{num_parquets:03d}.parquet"
        df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge the tokenized text and audio data into a single dataset in parquet format."
    )
    parser.add_argument(
        "--tokenized_text_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized text data.",
    )
    parser.add_argument(
        "--tokenized_audio_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized audio data.",
    )
    parser.add_argument(
        "--tokenized_oracle_dir",
        type=str,
        default=None,
        help="Path to the directory containing the tokenized oracle data. If not provided, oracle data will be ignored.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help=(
            "Prefix for the output dataset. Output files will be named as "
            "`{{output_prefix}}-001-of-002.parquet` etc."
        ),
    )
    parser.add_argument(
        "--text_padding_id",
        type=int,
        default=3,
        help="Padding id for text stream to fill the gap between audio and text streams.",
    )
    parser.add_argument(
        "--num_examples_per_parquet",
        type=int,
        default=100_000,
        help="Number of samples per parquet file.",
    )
    args = parser.parse_args()

    main(args)

# Example usage: this command will move to README in final version.
# uv run -m tools.prepare_dataset \
#     --tokenized_text_dir data/gsm8k_kame/tokenized_text_reversed \
#     --tokenized_audio_dir data/gsm8k_kame/tokenized_audio \
#     --tokenized_oracle_dir data/gsm8k_kame/tokenized_oracle_a0b1_events \
#     --output_prefix processed_data/gsm8k_kame_oracle/train_text_re_oracle_a0b1_events
