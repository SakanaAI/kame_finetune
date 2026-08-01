import argparse
import json
import os
from copy import deepcopy

import torch
from safetensors.torch import save_file

from models import (
    MoshiForFinetuning,
    remove_moshi_modules_for_user_stream,
)


def _validate_tied_oracle_embedding(model, *, context: str) -> None:
    text_state = model.text_emb.state_dict()
    oracle_state = model.oracle_emb.state_dict()
    if text_state.keys() != oracle_state.keys():
        raise RuntimeError(
            f"Tied oracle embedding state mismatch {context}: "
            f"{sorted(text_state.keys())} != {sorted(oracle_state.keys())}"
        )

    non_finite = []
    mismatched = []
    for key, text_value in text_state.items():
        oracle_value = oracle_state[key]
        if text_value.shape != oracle_value.shape or text_value.dtype != oracle_value.dtype:
            mismatched.append(key)
            continue

        if any(
            (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all().item()
            for value in (text_value, oracle_value)
        ):
            non_finite.append(key)
            continue

        if not torch.equal(text_value, oracle_value):
            mismatched.append(key)

    if non_finite:
        raise RuntimeError(
            f"Tied oracle embedding contains non-finite values {context}: {non_finite}"
        )
    if mismatched:
        raise RuntimeError(f"Tied oracle embedding was not materialized {context}: {mismatched}")


def materialize_oracle_embedding_for_inference(
    moshi_lm_for_ft: MoshiForFinetuning,
    oracle_embedding_mode: str,
) -> None:
    """Apply the training-time oracle embedding contract after ZeRO consolidation."""
    print(f"Applying oracle embedding mode for inference export: {oracle_embedding_mode}")
    moshi_lm_for_ft.materialize_oracle_embedding_for_checkpoint_(oracle_embedding_mode)
    if oracle_embedding_mode == "tie":
        _validate_tied_oracle_embedding(
            moshi_lm_for_ft,
            context="after materializing the consolidated finetuning model",
        )


def main(args):
    oracle_embedding_mode = getattr(args, "oracle_embedding_mode", None)
    if oracle_embedding_mode is None:
        raise ValueError(
            "oracle_embedding_mode is required and must match the training configuration"
        )

    moshi_lm_for_ft = MoshiForFinetuning.from_pretrained(
        args.moshi_ft_dir,
        device="cpu",
        dtype=getattr(torch, args.model_dtype),
    )

    materialize_oracle_embedding_for_inference(
        moshi_lm_for_ft,
        oracle_embedding_mode,
    )

    print("Converting MoshiForFinetuning to the original Moshi model...")
    moshi_lm = moshi_lm_for_ft.to_original_moshi_lm()
    moshi_lm_kwargs = deepcopy(moshi_lm_for_ft.moshi_lm_kwargs)

    # Remove the model to save memory
    del moshi_lm_for_ft

    if args.remove_modules_for_user_stream:
        print("Removing the depth transformer's modules for user stream...")
        # check if the model has the modules for user stream
        assert moshi_lm_kwargs["dep_q"] == 16 and moshi_lm_kwargs["depformer_context"] == 16, (
            f"{moshi_lm_kwargs['dep_q']=}, {moshi_lm_kwargs['depformer_context']=}"
        )
        moshi_lm = remove_moshi_modules_for_user_stream(moshi_lm)
        moshi_lm_kwargs.update(
            {
                "dep_q": 8,
                "depformer_context": 8,
            }
        )

    if oracle_embedding_mode == "tie":
        _validate_tied_oracle_embedding(
            moshi_lm,
            context="before saving the cleaned inference model",
        )

    print(f"Saving the cleaned up Moshi model to {args.save_dir}...")
    os.makedirs(args.save_dir, exist_ok=True)
    # Save the model
    save_file(moshi_lm.state_dict(), os.path.join(args.save_dir, "model.safetensors"))
    # Save the kwargs
    with open(os.path.join(args.save_dir, "moshi_lm_kwargs.json"), "w") as f:
        json.dump(moshi_lm_kwargs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moshi_ft_dir",
        type=str,
        required=True,
        help="Directory path to the Moshi model for finetuning",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory path to save the cleaned up Moshi model",
    )
    parser.add_argument(
        "--model_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type of the model",
    )
    parser.add_argument(
        "--remove_modules_for_user_stream",
        action="store_true",
        help="Whether to remove the depth transformer's modules for user stream",
    )
    parser.add_argument(
        "--oracle_embedding_mode",
        choices=["separate", "tie"],
        required=True,
        help=(
            "Oracle embedding mode used during training. Pass 'tie' to copy the learned "
            "text_emb state into oracle_emb after ZeRO-to-fp32 conversion and verify exact "
            "equality. Pass 'separate' to preserve the separately trained oracle_emb."
        ),
    )
    args = parser.parse_args()

    main(args)
