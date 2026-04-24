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


def main(args):
    moshi_lm_for_ft = MoshiForFinetuning.from_pretrained(
        args.moshi_ft_dir,
        device="cpu",
        dtype=getattr(torch, args.model_dtype),
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
    args = parser.parse_args()

    main(args)
