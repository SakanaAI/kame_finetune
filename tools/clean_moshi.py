import argparse
import json
import os
import warnings
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import save_file

from models import (
    MoshiForFinetuning,
    remove_moshi_modules_for_user_stream,
)

_VALID_ORACLE_EMBEDDING_MODES = ("separate", "tie")


def _validate_oracle_embedding_mode(mode: object, *, source: str) -> str:
    if not isinstance(mode, str) or mode not in _VALID_ORACLE_EMBEDDING_MODES:
        raise ValueError(
            f"{source} has invalid oracle_embedding_mode {mode!r}; expected one of "
            f"{_VALID_ORACLE_EMBEDDING_MODES}"
        )
    return mode


def _load_training_config_mode(config_path: Path) -> str | None:
    try:
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training config is not valid JSON: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a JSON object: {config_path}")
    if "oracle_embedding_mode" not in config:
        return None
    return _validate_oracle_embedding_mode(
        config["oracle_embedding_mode"],
        source=f"Training config at {config_path}",
    )


def _resolve_oracle_embedding_mode(
    *,
    moshi_ft_dir: str,
    training_config_path: str | None,
    requested_mode: str | None,
) -> str:
    if training_config_path is not None:
        config_path = Path(training_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Training config does not exist: {config_path}")
    else:
        config_path = Path(moshi_ft_dir).parent / "config.json"

    config_exists = config_path.exists()
    if config_exists and not config_path.is_file():
        raise ValueError(f"Training config path must be a file: {config_path}")

    metadata_mode = _load_training_config_mode(config_path) if config_exists else None
    cli_mode = (
        _validate_oracle_embedding_mode(requested_mode, source="CLI argument")
        if requested_mode is not None
        else None
    )

    if metadata_mode is not None:
        if cli_mode is not None and cli_mode != metadata_mode:
            raise ValueError(
                "oracle_embedding_mode mismatch: training config at "
                f"{config_path} records {metadata_mode!r}, but the CLI requested "
                f"{cli_mode!r}"
            )
        print(f"Resolved oracle embedding mode from training config {config_path}: {metadata_mode}")
        return metadata_mode

    if cli_mode is not None:
        metadata_status = (
            f"Training config at {config_path} does not contain oracle_embedding_mode"
            if config_exists
            else f"Training config was not found at {config_path}"
        )
        warnings.warn(
            f"{metadata_status}; using --oracle_embedding_mode={cli_mode} as a legacy "
            "fallback. The training mode cannot be independently verified.",
            stacklevel=2,
        )
        return cli_mode

    metadata_status = (
        f"training config at {config_path} does not contain oracle_embedding_mode"
        if config_exists
        else f"training config was not found at {config_path}"
    )
    raise ValueError(
        "Could not determine oracle_embedding_mode because "
        f"{metadata_status}. Pass --oracle_embedding_mode for a legacy checkpoint."
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
    oracle_embedding_mode = _resolve_oracle_embedding_mode(
        moshi_ft_dir=args.moshi_ft_dir,
        training_config_path=getattr(args, "training_config_path", None),
        requested_mode=getattr(args, "oracle_embedding_mode", None),
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
        "--training_config_path",
        type=str,
        default=None,
        help=(
            "Path to the training config.json. By default, config.json is read from "
            "the parent directory of --moshi_ft_dir."
        ),
    )
    parser.add_argument(
        "--oracle_embedding_mode",
        choices=_VALID_ORACLE_EMBEDDING_MODES,
        default=None,
        help=(
            "Optional assertion for the mode recorded in the training config, or a "
            "required fallback for legacy checkpoints without this metadata."
        ),
    )
    args = parser.parse_args()

    main(args)
