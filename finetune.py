import argparse
import collections
import itertools
import json
import logging
import math
import os
from datetime import timedelta

import deepspeed
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DummyOptim,
    DummyScheduler,
    InitProcessGroupKwargs,
    set_seed,
)
from datasets import concatenate_datasets, load_dataset
from kame.modules.transformer import set_attention_context
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.moshi_for_finetuning import MoshiForFinetuning
from utils import (
    Batch,
    DataCollator,
    preprocess_function,
    set_mpi_env_vars,
)

logger = get_logger(__name__)


# Parsing input arguments
def setup_argparser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--launcher",
        choices=["accelerate", "mpi"],
        default="accelerate",
        help="Launcher type to use for distributed training",
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Whether to use deepspeed for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        nargs="+",
        required=True,
        help="Patterns to the training data files. Each file should be parquet file.",
    )
    parser.add_argument(
        "--eval_data_files",
        type=str,
        nargs="+",
        default=None,
        help="Patterns to the evaluation data files. Each file should be parquet file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the pre-trained KAME model (`model.safetensors`) "
            "and config for initializing model (`init_moshi_lm_kwargs.json`)."
        ),
    )
    parser.add_argument(
        "--model_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model data type",
    )

    parser.add_argument(
        "--moshi_speakers",
        choices=["A", "B"],
        nargs="+",
        default=["A"],
        help="Speakers to use as the main stream when no per-dataset override is provided.",
    )
    parser.add_argument(
        "--train_data_file_speakers",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional per-train-file speaker setting. Provide one item per train_data_files entry; "
            "each item must be 'A', 'B', or 'A,B'."
        ),
    )
    parser.add_argument(
        "--model_user_stream",
        action="store_true",
        help="Whether to train the user's audio stream",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequence",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=None,
        help="Minimum length of the input sequence",
    )
    parser.add_argument(
        "--dataset_processing_workers",
        type=int,
        default=16,
        help="Number of workers to use for processing the dataset.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=".cache/huggingface/datasets",
        help="Directory to cache the datasets.",
    )
    parser.add_argument(
        "--process_group_timeout",
        type=int,
        default=3600,
        help="Timeout for the process group initialization (in seconds).",
    )

    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default=None,
        help="Path to a DeepSpeed config file. Required if `use_deepspeed` is set.",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Whether to use activation checkpointing in the model.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--tempformer_learning_rate",
        type=float,
        default=3e-5,
        help="The initial learning rate for the Temporal Transformer model.",
    )
    parser.add_argument(
        "--depformer_learning_rate",
        type=float,
        default=3e-5,
        help="The initial learning rate for the Depth Transformer model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay to apply.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Stop training after this number of steps.",
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the learning rate scheduler.",
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run evaluation every X updates steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--report_to",
        choices=["wandb"],
        default=None,
        help='The integration to report the results and logs to. Currently only "wandb" is supported.',
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="kame-finetuning",
        help="The name of the project to which the training run belongs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )

    parser.add_argument(
        "--parameters_to_finetune",
        choices=["all", "tempformer", "depformer", "text_embedding"],
        default="all",
        help="Which parameters to finetune.",
    )
    parser.add_argument(
        "--text_padding_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the padding loss in the text loss.",
    )
    parser.add_argument(
        "--semantic_loss_weight",
        type=float,
        default=100.0,
        help="Weight for the semantic loss in the audio loss.",
    )
    parser.add_argument(
        "--acoustic_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the acoustic loss in the audio loss.",
    )
    parser.add_argument(
        "--use_oracle",
        action="store_true",
        help="Use oracle tokens if *_oracle columns exist in the parquet.",
    )
    parser.add_argument(
        "--oracle_embedding_mode",
        choices=["separate", "tie"],
        default="separate",
        help=(
            "How to embed oracle tokens. "
            "'separate' uses oracle_emb, while 'tie' reuses text_emb during training "
            "and materializes oracle_emb from text_emb before checkpoint save for inference compatibility."
        ),
    )
    # ---- oracle augmentation / randomness (applies when --use_oracle is set) ----
    parser.add_argument(
        "--oracle_start_id",
        type=int,
        default=32000,
        help="Oracle start token id used when reconstructing oracle from events.",
    )
    parser.add_argument(
        "--oracle_shift_prob",
        type=float,
        default=1.0,
        help="Probability of applying Turtle-style shift augmentation to oracle tokens. Range: [0.0, 1.0]. Typical: 1.0.",
    )
    parser.add_argument(
        "--oracle_right_shift_min",
        type=int,
        default=1,
        help="Minimum right shift amount (frames). Must be >= 0. Typical: 1.",
    )
    parser.add_argument(
        "--oracle_right_shift_max",
        type=int,
        default=10,
        help="Maximum right shift amount (frames). Must be >= oracle_right_shift_min. Typical: 10.",
    )
    parser.add_argument(
        "--oracle_left_shift_min",
        type=int,
        default=1,
        help="Minimum left shift amount (frames). Must be >= 0. Typical: 1.",
    )
    parser.add_argument(
        "--oracle_left_shift_max",
        type=int,
        default=15,
        help="Maximum left shift amount (frames). Must be >= oracle_left_shift_min. Typical: 15.",
    )
    parser.add_argument(
        "--oracle_skip_prob_min",
        type=float,
        default=0.1,
        help="Minimum per-sample oracle event skip probability (events only). Range: [0.0, 1.0]. Typical: 0.1.",
    )
    parser.add_argument(
        "--oracle_skip_prob_max",
        type=float,
        default=0.7,
        help="Maximum per-sample oracle event skip probability (events only). Range: [0.0, 1.0]. Must be >= oracle_skip_prob_min. Typical: 0.7.",
    )
    parser.add_argument(
        "--oracle_max_time_jitter_frames",
        type=int,
        default=0,
        help="Max time jitter (in frames) applied to oracle event positions (events only). Must be >= 0.  Typical: 4 (approx 0.3 sec at 12.5 Hz).",
    )
    parser.add_argument(
        "--oracle_hint_only",
        action="store_true",
        help="When set, oracle tokens use only hint (no prediction fallback). "
        "By default, hint is used when ratio >= 1.0 and hint exists, "
        "otherwise prediction tokens are used as fallback.",
    )
    parser.add_argument(
        "--oracle_hint_only_warmup_start",
        type=int,
        default=None,
        help="Step at which to begin transitioning from hint-only to prediction mixing. "
        "Only effective when --oracle_hint_only is set. If None, no curriculum is applied.",
    )
    parser.add_argument(
        "--oracle_hint_only_warmup_end",
        type=int,
        default=None,
        help="Step at which the transition to full prediction mixing is complete. "
        "Only effective when --oracle_hint_only is set. If None, no curriculum is applied.",
    )
    parser.add_argument(
        "--semantic_emb_dropout",
        type=float,
        default=0.0,
        help="Probability of zeroing out the semantic (codebook 0) audio embedding during training. Forces the model to rely more on text/oracle for speech content. Range: [0.0, 1.0].",
    )
    parser.add_argument(
        "--text_emb_dropout",
        type=float,
        default=0.0,
        help="Probability of zeroing out the text embedding for an entire sample during training. "
        "Applied per sample in the batch (all timesteps zeroed). Range: [0.0, 1.0].",
    )
    parser.add_argument(
        "--text_token_dropout",
        type=float,
        default=0.0,
        help="Probability of flipping each non-special text token to the padding token "
        "before embedding. Applied independently per token, preserving the padding "
        "token's embedding (inner-monologue semantics retained). Range: [0.0, 1.0].",
    )
    parser.add_argument(
        "--attention_context",
        type=int,
        default=None,
        help="Sliding window size (in frames) for causal attention in the temporal transformer. "
        "If set, each token attends to at most this many past tokens instead of the full sequence. "
        "This reduces memory usage without truncating the input sequence. "
        "If None, uses the model's default (typically full context).",
    )


def postprocess_args(args: argparse.Namespace):
    # post process
    if args.launcher == "accelerate":
        args.use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED") == "true"
        args.deepspeed_config_file = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE")
        if args.use_deepspeed and "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            # unset the mpirun's variables for deepspeed to work
            # ref: https://github.com/Lightning-AI/pytorch-lightning/issues/13567
            del os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    elif args.launcher == "mpi":
        _ = set_mpi_env_vars()
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"  # set manually here
    else:
        raise ValueError(f"Unknown launcher: {args.launcher}")

    # check the deepspeed settings
    if not args.use_deepspeed:
        raise NotImplementedError("Only DeepSpeed is supported for now.")
    if args.deepspeed_config_file is None:
        raise ValueError(
            "DeepSpeed config file (`--deepspeed_config_file`) is required when "
            "`use_deepspeed` is set."
        )

    # check the dataset files
    if not args.train_data_files:
        raise ValueError("No training data files provided.")
    if args.train_data_file_speakers is not None and len(args.train_data_file_speakers) != len(
        args.train_data_files
    ):
        raise ValueError(
            "train_data_file_speakers must have the same number of entries as train_data_files: "
            f"{len(args.train_data_file_speakers)} != {len(args.train_data_files)}"
        )
    if args.eval_data_files is not None:
        assert isinstance(args.eval_steps, int), (
            "eval_steps is required when eval_data_files is provided."
        )

    warmup_start = args.oracle_hint_only_warmup_start
    warmup_end = args.oracle_hint_only_warmup_end
    if (warmup_start is None) != (warmup_end is None):
        raise ValueError(
            "oracle_hint_only_warmup_start and oracle_hint_only_warmup_end must be set together."
        )
    if warmup_start is not None:
        if not args.oracle_hint_only:
            raise ValueError("oracle_hint_only_warmup_start/end require --oracle_hint_only.")
        if warmup_start < 0 or warmup_end < 0:
            raise ValueError("oracle_hint_only_warmup_start/end must be >= 0.")
        if warmup_end <= warmup_start:
            raise ValueError(
                "oracle_hint_only_warmup_end must be greater than oracle_hint_only_warmup_start."
            )

    args.with_tracking = args.report_to is not None

    if args.resume_from_checkpoint:
        assert os.path.exists(args.resume_from_checkpoint), (
            f"Checkpoint file not found: {args.resume_from_checkpoint}"
        )
        resume_config_path = os.path.join(
            os.path.dirname(args.resume_from_checkpoint), "config.json"
        )
        assert os.path.exists(resume_config_path), f"Config file not found: {resume_config_path}"
        prev_config = json.load(open(resume_config_path))
        # check consistency of the config
        different_keys = ["output_dir", "max_train_steps", "resume_from_checkpoint"]
        mismatch_args = []
        for key, value in vars(args).items():
            if key in different_keys:
                continue
            prev_value = prev_config.get(key)
            if prev_value != value:
                mismatch_args.append(f"{key}: {prev_value} != {value}")
        if mismatch_args:
            raise ValueError(f"Mismatch in the config: {mismatch_args}")

        # Tracking is optional, so non-tracked runs legitimately have no run_id.
        args.run_id_to_resume = prev_config.get("run_id")
    else:
        args.run_id_to_resume = None


def _parse_speaker_spec(spec: str) -> list[str]:
    speakers = [speaker.strip() for speaker in spec.split(",") if speaker.strip()]
    if not speakers:
        raise ValueError(f"Empty speaker spec: {spec!r}")

    invalid = [speaker for speaker in speakers if speaker not in {"A", "B"}]
    if invalid:
        raise ValueError(f"Invalid speaker spec {spec!r}: {invalid}")

    if len(set(speakers)) != len(speakers):
        raise ValueError(f"Duplicate speakers are not allowed in spec {spec!r}")

    return speakers


def _compute_num_update_steps_per_epoch(
    num_local_batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> int:
    if num_local_batches_per_epoch < 0:
        raise ValueError("num_local_batches_per_epoch must be >= 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    return math.ceil(num_local_batches_per_epoch / gradient_accumulation_steps)


def _compute_resume_batch_offset(
    current_steps: int,
    *,
    gradient_accumulation_steps: int,
    num_local_batches_per_epoch: int,
) -> int:
    if current_steps < 0:
        raise ValueError("current_steps must be >= 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if num_local_batches_per_epoch <= 0:
        return 0
    return (current_steps * gradient_accumulation_steps) % num_local_batches_per_epoch


def _build_preprocessing_kwargs(
    args: argparse.Namespace, moshi_lm: MoshiForFinetuning
) -> dict[str, object]:
    return {
        "max_length": args.max_length,
        "min_length": args.min_length,
        "delays": moshi_lm.delays,
        "initial_token_ids": [moshi_lm.text_initial_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [moshi_lm.text_padding_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "zero_token_id": moshi_lm.zero_token_id,
        "oracle_column_suffix": "_oracle" if args.use_oracle else None,
    }


def _get_oracle_embedding_module(
    moshi_lm: MoshiForFinetuning,
    oracle_embedding_mode: str,
):
    if oracle_embedding_mode == "separate":
        return moshi_lm.oracle_emb
    if oracle_embedding_mode == "tie":
        return moshi_lm.text_emb
    raise ValueError(f"Unknown oracle embedding mode: {oracle_embedding_mode}")


def _iter_oracle_embedding_parameters(
    moshi_lm: MoshiForFinetuning,
    oracle_embedding_mode: str,
):
    if oracle_embedding_mode == "tie":
        return ()
    if oracle_embedding_mode == "separate":
        return moshi_lm.oracle_emb.parameters()
    raise ValueError(f"Unknown oracle embedding mode: {oracle_embedding_mode}")


def _load_train_dataset(
    args: argparse.Namespace,
    accelerator: Accelerator,
    moshi_lm: MoshiForFinetuning,
):
    preprocessing_base = _build_preprocessing_kwargs(args, moshi_lm)

    with accelerator.main_process_first():
        if args.train_data_file_speakers is None:
            logger.info(f"Loading train dataset from {args.train_data_files}")
            train_dataset = load_dataset(
                "parquet",
                split="train",
                data_files={"train": args.train_data_files},
                cache_dir=args.dataset_cache_dir,
            )
            train_dataset = train_dataset.map(
                preprocess_function,
                remove_columns=train_dataset.column_names,
                batched=True,
                num_proc=args.dataset_processing_workers,
                fn_kwargs=preprocessing_base | {"speakers": args.moshi_speakers},
                desc="Preprocessing train dataset",
            )
            return train_dataset

        train_datasets = []
        for data_file, speaker_spec in zip(
            args.train_data_files, args.train_data_file_speakers, strict=True
        ):
            speakers = _parse_speaker_spec(speaker_spec)
            logger.info(f"Loading train dataset from {data_file} with speakers={speakers}")
            train_dataset = load_dataset(
                "parquet",
                split="train",
                data_files={"train": [data_file]},
                cache_dir=args.dataset_cache_dir,
            )
            train_dataset = train_dataset.map(
                preprocess_function,
                remove_columns=train_dataset.column_names,
                batched=True,
                num_proc=args.dataset_processing_workers,
                fn_kwargs=preprocessing_base | {"speakers": speakers},
                desc=f"Preprocessing {os.path.basename(data_file)}",
            )
            train_datasets.append(train_dataset)

    return concatenate_datasets(train_datasets)


def get_parameters(
    moshi_lm: MoshiForFinetuning,
    parameter_name: str,
    oracle_embedding_mode: str = "separate",
):
    if parameter_name == "all":
        if oracle_embedding_mode == "separate":
            return moshi_lm.parameters()
        if oracle_embedding_mode == "tie":
            return (
                param
                for name, param in moshi_lm.named_parameters()
                if not name.startswith("oracle_emb.")
            )
        raise ValueError(f"Unknown oracle embedding mode: {oracle_embedding_mode}")
    elif parameter_name == "tempformer":
        return itertools.chain(
            moshi_lm.emb.parameters(),
            moshi_lm.text_emb.parameters(),
            _iter_oracle_embedding_parameters(moshi_lm, oracle_embedding_mode),
            moshi_lm.transformer.parameters(),
            moshi_lm.out_norm.parameters(),
            moshi_lm.text_linear.parameters(),
        )
    elif parameter_name == "depformer":
        return itertools.chain(
            moshi_lm.depformer_in.parameters(),
            moshi_lm.depformer_emb.parameters(),
            moshi_lm.depformer_text_emb.parameters(),
            moshi_lm.depformer.parameters(),
            moshi_lm.linears.parameters(),
        )
    elif parameter_name == "text_embedding":
        return itertools.chain(
            moshi_lm.text_emb.parameters(),
            _iter_oracle_embedding_parameters(moshi_lm, oracle_embedding_mode),
            moshi_lm.depformer_text_emb.parameters(),
        )
    else:
        raise ValueError(f"Unknown parameter name: {parameter_name}")


def tempformer_forward(
    moshi_lm: MoshiForFinetuning,
    batch: Batch,
    use_oracle: bool = False,
    oracle_embedding_mode: str = "separate",
    semantic_emb_dropout: float = 0.0,
    text_emb_dropout: float = 0.0,
    text_token_dropout: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 1 Encode text
    text_input_ids = batch.input_ids[:, 0]  # [B, T]

    # Token-level dropout: randomly flip ordinary text tokens to padding_token_id before
    # embedding while preserving structural sentinels such as the text start token.
    if text_token_dropout > 0.0 and moshi_lm.training:
        protected_tokens = (
            (text_input_ids == moshi_lm.text_padding_token_id)
            | (text_input_ids == moshi_lm.zero_token_id)
            | (text_input_ids == moshi_lm.text_initial_token_id)
        )
        flip_mask = (
            torch.rand_like(text_input_ids, dtype=torch.float) < text_token_dropout
        ) & ~protected_tokens
        text_input_ids = text_input_ids.clone()
        text_input_ids[flip_mask] = moshi_lm.text_padding_token_id

    text_emb = moshi_lm.text_emb(text_input_ids)

    # Sample-level dropout: zero out the entire embedding for randomly selected samples in the batch.
    if text_emb_dropout > 0.0 and moshi_lm.training:
        drop_mask = torch.rand(text_emb.shape[0], 1, 1, device=text_emb.device) < text_emb_dropout
        text_emb = text_emb.masked_fill(drop_mask, 0.0)

    if use_oracle and batch.oracle_tokens is not None:
        oracle_tokens = batch.oracle_tokens[:, 0]
        assert oracle_tokens.shape == batch.input_ids[:, 0].shape, (
            f"{oracle_tokens.shape} != {batch.input_ids[:, 0].shape}"
        )

        oracle_embedding = _get_oracle_embedding_module(moshi_lm, oracle_embedding_mode)
        oracle_emb = oracle_embedding(oracle_tokens)  # [B, T, D]
        text_emb = text_emb + oracle_emb

    # 2 Encode audio
    audio_emb = None
    for acb_index in range(moshi_lm.num_audio_codebooks):
        audio_emb_ = moshi_lm.emb[acb_index](batch.input_ids[:, moshi_lm.audio_offset + acb_index])
        # Dropout semantic embedding (codebook 0) during training
        if acb_index == 0 and semantic_emb_dropout > 0.0 and moshi_lm.training:
            # Per-sample dropout: each sample in the batch independently
            drop_mask = (
                torch.rand(audio_emb_.shape[0], 1, 1, device=audio_emb_.device)
                < semantic_emb_dropout
            )
            audio_emb_ = audio_emb_.masked_fill(drop_mask, 0.0)
        audio_emb = audio_emb_ if audio_emb is None else audio_emb + audio_emb_

    # 3 Feed embeddings to temporal transformer
    tempformer_input = text_emb + audio_emb
    tempformer_out = moshi_lm.transformer(tempformer_input)
    if moshi_lm.out_norm:
        tempformer_out = moshi_lm.out_norm(tempformer_out)
    text_logits = moshi_lm.text_linear(tempformer_out)

    # 4 Compute loss
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    text_logits = text_logits.float()
    # Shift so that tokens < n predict n
    text_logits = text_logits[..., :-1, :].contiguous()
    text_labels = batch.labels[:, 0, 1:].contiguous()

    text_losses = F.cross_entropy(
        input=text_logits.view(-1, moshi_lm.text_card),
        target=text_labels.view(-1),
        ignore_index=moshi_lm.zero_token_id,
        reduction="none",
    ).view(text_labels.size())
    text_accuracy = (text_logits.argmax(-1) == text_labels).float()

    assert text_labels.shape == text_losses.shape, f"{text_labels.shape} != {text_losses.shape}"

    non_pad_indices = (
        (text_labels != moshi_lm.text_padding_token_id)
        & (text_labels != moshi_lm.zero_token_id)  # zero token is ignored
    )
    # we treat moshi_lm.end_of_text_padding_id as non-padding token
    pad_indices = text_labels == moshi_lm.text_padding_token_id

    result = {
        "non_pad_losses": text_losses[non_pad_indices],
        "pad_losses": text_losses[pad_indices],
        "non_pad_accuracy": text_accuracy[non_pad_indices].mean(),
        "pad_accuracy": text_accuracy[pad_indices].mean(),
    }
    return tempformer_out, result


def depformer_forward(
    moshi_lm: MoshiForFinetuning,
    batch: Batch,
    tempformer_out: torch.Tensor,
    model_user_stream: bool,
) -> dict[str, torch.Tensor]:
    # 1 Mapping tempformer's output to depformer's input
    depformer_inputs = []
    for acb_index in range(moshi_lm.dep_q):
        if moshi_lm.depformer_multi_linear:
            # use different linear layers for different audio codebooks
            depformer_input_ = moshi_lm.depformer_in[acb_index](tempformer_out[:, :-1])
        else:
            # use the same linear layer for all audio codebooks
            depformer_input_ = moshi_lm.depformer_in[0](tempformer_out[:, :-1])
        depformer_inputs.append(depformer_input_)
    depformer_input = torch.stack(depformer_inputs, dim=2)

    # 2 Encode last token of each audio streams
    last_token_embs = []
    last_token_emb_ = moshi_lm.depformer_text_emb(batch.input_ids[:, 0, 1:])  # text token
    last_token_embs.append(last_token_emb_)
    for acb_index in range(moshi_lm.dep_q - 1):
        last_token_emb_ = moshi_lm.depformer_emb[acb_index](
            batch.input_ids[:, moshi_lm.audio_offset + acb_index, 1:]  # audio token
        )
        last_token_embs.append(last_token_emb_)
    last_token_emb = torch.stack(last_token_embs, dim=2)

    # 3 Feed embeddings to rq-transformer
    depformer_input = depformer_input + last_token_emb
    # flatten batch_size and num_frames
    depformer_input = torch.flatten(depformer_input, 0, 1)
    depformer_out = moshi_lm.depformer(depformer_input)

    audio_logits = []
    for acb_index in range(moshi_lm.dep_q):
        if moshi_lm.depformer_multi_linear:
            audio_logits_ = moshi_lm.linears[acb_index](depformer_out[:, acb_index])
        else:
            audio_logits_ = moshi_lm.linears[0](depformer_out[:, acb_index])
        audio_logits.append(audio_logits_)
    audio_logits = torch.stack(audio_logits, dim=1)
    audio_logits = audio_logits.float()
    # >>> depformer_logits.shape
    # torch.Size([batch_size * num_frames, dep_q, card])

    # 4 Compute loss
    if model_user_stream:
        audio_labels = batch.labels[:, 1:, 1:].transpose(1, 2).contiguous()
    else:
        audio_labels = batch.labels[:, 1:9, 1:].transpose(1, 2).contiguous()
    # >>> depformer_labels.shape
    # torch.Size([batch_size, num_frames, dep_q])

    audio_losses = F.cross_entropy(
        input=audio_logits.view(-1, moshi_lm.card),
        target=audio_labels.view(-1),
        ignore_index=moshi_lm.zero_token_id,
        reduction="none",
    ).view(audio_labels.size())
    audio_accuracy = (audio_logits.argmax(-1).view(audio_labels.shape) == audio_labels).float()

    assert audio_labels.shape == audio_losses.shape, f"{audio_labels.shape} != {audio_losses.shape}"
    assert audio_accuracy.shape == audio_labels.shape, (
        f"{audio_accuracy.shape} != {audio_labels.shape}"
    )

    result = {
        "semantic_losses": audio_losses[..., 0][audio_labels[..., 0] != moshi_lm.zero_token_id],
        "acoustic_losses": audio_losses[..., 1:8][audio_labels[..., 1:8] != moshi_lm.zero_token_id],
        "semantic_accuracy": audio_accuracy[..., 0][
            audio_labels[..., 0] != moshi_lm.zero_token_id
        ].mean(),
        "acoustic_accuracy": audio_accuracy[..., 1:8][
            audio_labels[..., 1:8] != moshi_lm.zero_token_id
        ].mean(),
    }
    if model_user_stream:
        result.update(
            {
                "semantic_losses_user": audio_losses[..., 8][
                    audio_labels[..., 8] != moshi_lm.zero_token_id
                ],
                "acoustic_losses_user": audio_losses[..., 9:][
                    audio_labels[..., 9:] != moshi_lm.zero_token_id
                ],
                "semantic_accuracy_user": audio_accuracy[..., 8][
                    audio_labels[..., 8] != moshi_lm.zero_token_id
                ].mean(),
                "acoustic_accuracy_user": audio_accuracy[..., 9:][
                    audio_labels[..., 9:] != moshi_lm.zero_token_id
                ].mean(),
            }
        )

    return result


def forward(
    moshi_lm: MoshiForFinetuning,
    batch: Batch,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Forward pass of the Moshi model
    1. Temporal Transformer
    2. Depth Transformer
    3. Compute loss

    Args:
        args (argparse.Namespace): input arguments
        moshi_lm (MoshiForFinetuning): Moshi model
        batch (Batch): input batch
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: final loss and logging values
    """
    log = {}

    # 1 Forward pass of Temporal Transformer
    use_oracle = getattr(args, "use_oracle", False)
    oracle_embedding_mode = getattr(args, "oracle_embedding_mode", "separate")
    semantic_emb_dropout = getattr(args, "semantic_emb_dropout", 0.0)
    text_emb_dropout = getattr(args, "text_emb_dropout", 0.0)
    text_token_dropout = getattr(args, "text_token_dropout", 0.0)
    tempformer_out, temp_result = tempformer_forward(
        moshi_lm,
        batch,
        use_oracle=use_oracle,
        oracle_embedding_mode=oracle_embedding_mode,
        semantic_emb_dropout=semantic_emb_dropout,
        text_emb_dropout=text_emb_dropout,
        text_token_dropout=text_token_dropout,
    )
    text_loss = 0.0
    if temp_result["non_pad_losses"].size(0) > 0:
        text_loss += temp_result["non_pad_losses"].mean()
    if temp_result["pad_losses"].size(0) > 0:
        text_loss += temp_result["pad_losses"].mean() * args.text_padding_loss_weight
    log["loss/text_total"] = text_loss.detach()
    log["loss/text_non_pad"] = temp_result["non_pad_losses"].mean().detach()
    log["loss/text_pad"] = temp_result["pad_losses"].mean().detach()
    log["accuracy/text_non_pad"] = temp_result["non_pad_accuracy"].detach()
    log["accuracy/text_pad"] = temp_result["pad_accuracy"].detach()

    del temp_result

    # 2 Forward pass of Depth Transformer
    dep_result = depformer_forward(moshi_lm, batch, tempformer_out, args.model_user_stream)
    audio_weight = (
        dep_result["semantic_losses"].size(0) * args.semantic_loss_weight
        + dep_result["acoustic_losses"].size(0) * args.acoustic_loss_weight
    )
    if args.model_user_stream:
        audio_weight += (
            dep_result["semantic_losses_user"].size(0) * args.semantic_loss_weight
            + dep_result["acoustic_losses_user"].size(0) * args.acoustic_loss_weight
        )
    if audio_weight > 0:
        semantic_scale = args.semantic_loss_weight / audio_weight
        acoustic_scale = args.acoustic_loss_weight / audio_weight
    else:
        semantic_scale = 0.0
        acoustic_scale = 0.0
    audio_loss = (
        dep_result["semantic_losses"].sum() * semantic_scale
        + dep_result["acoustic_losses"].sum() * acoustic_scale
    )
    if args.model_user_stream:
        audio_loss += (
            dep_result["semantic_losses_user"].sum() * semantic_scale
            + dep_result["acoustic_losses_user"].sum() * acoustic_scale
        )
    log["loss/audio_total"] = audio_loss.detach()
    log["loss/audio_semantic"] = dep_result["semantic_losses"].mean().detach()
    log["loss/audio_acoustic"] = dep_result["acoustic_losses"].mean().detach()
    log["accuracy/audio_semantic"] = dep_result["semantic_accuracy"].detach()
    log["accuracy/audio_acoustic"] = dep_result["acoustic_accuracy"].detach()
    if args.model_user_stream:
        log["loss/audio_semantic_user"] = dep_result["semantic_losses_user"].mean().detach()
        log["loss/audio_acoustic_user"] = dep_result["acoustic_losses_user"].mean().detach()
        log["accuracy/audio_semantic_user"] = dep_result["semantic_accuracy_user"].detach()
        log["accuracy/audio_acoustic_user"] = dep_result["acoustic_accuracy_user"].detach()
    del dep_result

    # 3 Compute final loss
    total_loss = text_loss + audio_loss
    log["loss/total"] = total_loss.detach()

    return total_loss, log


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Finetune Moshi LM on a custom dataset")
    setup_argparser(parser)
    args = parser.parse_args()
    postprocess_args(args)

    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kwargs_handlers": [
            InitProcessGroupKwargs(timeout=timedelta(seconds=args.process_group_timeout)),
        ],
    }

    if args.with_tracking:
        accelerator_kwargs.update(
            {
                "log_with": args.report_to,
                "project_dir": args.output_dir,
            }
        )

    if args.launcher != "accelerate":
        from accelerate import DeepSpeedPlugin

        accelerator_kwargs.update(
            {
                "deepspeed_plugin": DeepSpeedPlugin(
                    hf_ds_config=args.deepspeed_config_file,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
            }
        )

    accelerator = Accelerator(**accelerator_kwargs)
    args.num_processes = accelerator.num_processes

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the model
    logger.info(f"Loading Moshi model from {args.model_dir}")
    moshi_lm = MoshiForFinetuning.from_pretrained(
        save_dir=args.model_dir,
        device="cpu",  # accelerator will move the model to the correct device
        dtype=torch.float32,
    )
    moshi_lm.oracle_embedding_mode = args.oracle_embedding_mode

    # Set activation checkpointing
    if args.activation_checkpointing:
        assert os.environ.get("NO_TORCH_COMPILE", "1") == "1", "Not compatible with torch.compile"

        activation_checkpointing_kwargs = accelerator.deepspeed_plugin.hf_ds_config.config.get(
            "activation_checkpointing", {}
        )
        deepspeed.checkpointing.configure(
            mpu_=None, deepspeed_config=None, **activation_checkpointing_kwargs
        )
        moshi_lm.enable_activation_checkpointing(deepspeed.checkpointing.checkpoint)

    # Set attention context if specified
    if args.attention_context is not None:
        set_attention_context(moshi_lm, context=args.attention_context)

    # Set active/frozen for the finetuning
    for param in moshi_lm.parameters():
        param.requires_grad = False
    for param in get_parameters(
        moshi_lm,
        args.parameters_to_finetune,
        oracle_embedding_mode=args.oracle_embedding_mode,
    ):
        param.requires_grad = True
    for name, param in moshi_lm.named_parameters():
        logger.info(f"{name}: {param.requires_grad}")

    # Load the dataset
    train_dataset = _load_train_dataset(args, accelerator, moshi_lm)
    if args.eval_data_files is not None:
        preprocessing_kwargs = _build_preprocessing_kwargs(args, moshi_lm) | {
            "speakers": args.moshi_speakers
        }
        with accelerator.main_process_first():
            logger.info(f"Loading eval dataset from {args.eval_data_files}")
            eval_dataset = load_dataset(
                "parquet",
                split="validation",
                data_files={"validation": args.eval_data_files},
                cache_dir=args.dataset_cache_dir,
            )
            eval_dataset = eval_dataset.map(
                preprocess_function,
                remove_columns=eval_dataset.column_names,
                batched=True,
                num_proc=args.dataset_processing_workers,
                fn_kwargs=preprocessing_kwargs,
                desc="Preprocessing validation dataset",
            )
    else:
        eval_dataset = None

    train_data_collator_kwargs = {
        "zero_token_id": moshi_lm.zero_token_id,
    }

    if args.use_oracle:
        train_data_collator_kwargs.update(
            {
                "oracle_pad_id": moshi_lm.text_padding_token_id,  # typically 3
                "oracle_shift_prob": float(args.oracle_shift_prob),
                "oracle_right_shift_range": (
                    int(args.oracle_right_shift_min),
                    int(args.oracle_right_shift_max),
                ),
                "oracle_left_shift_range": (
                    int(args.oracle_left_shift_min),
                    int(args.oracle_left_shift_max),
                ),
                "oracle_start_id": int(args.oracle_start_id),
                "oracle_skip_prob_min": float(args.oracle_skip_prob_min),
                "oracle_skip_prob_max": float(args.oracle_skip_prob_max),
                "oracle_max_time_jitter_frames": int(args.oracle_max_time_jitter_frames),
                "oracle_hint_only": bool(args.oracle_hint_only),
                "oracle_hint_only_warmup_start": args.oracle_hint_only_warmup_start,
                "oracle_hint_only_warmup_end": args.oracle_hint_only_warmup_end,
            }
        )

    train_data_collator = DataCollator(**train_data_collator_kwargs)

    eval_data_collator = None
    if eval_dataset is not None:
        eval_data_collator_kwargs = {
            "zero_token_id": moshi_lm.zero_token_id,
        }
        if args.use_oracle:
            eval_data_collator_kwargs.update(
                {
                    "oracle_pad_id": moshi_lm.text_padding_token_id,  # typically 3
                    "oracle_shift_prob": 0.0,
                    "oracle_right_shift_range": (
                        int(args.oracle_right_shift_min),
                        int(args.oracle_right_shift_max),
                    ),
                    "oracle_left_shift_range": (
                        int(args.oracle_left_shift_min),
                        int(args.oracle_left_shift_max),
                    ),
                    "oracle_start_id": int(args.oracle_start_id),
                    "oracle_skip_prob_min": 0.0,
                    "oracle_skip_prob_max": 0.0,
                    "oracle_max_time_jitter_frames": 0,
                    "oracle_hint_only": False,
                    "oracle_hint_only_warmup_start": None,
                    "oracle_hint_only_warmup_end": None,
                }
            )
        eval_data_collator = DataCollator(**eval_data_collator_kwargs)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=train_data_collator,
        shuffle=True,
    )
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=eval_data_collator,
            shuffle=False,
        )
    else:
        eval_dataloader = None

    global_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    scheduler_num_update_steps_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
    scheduler_total_num_update_steps = args.num_train_epochs * scheduler_num_update_steps_per_epoch

    # Prepare optimizer and learning rate scheduler
    param_groups = [
        {  # Temporal Transformer
            "params": get_parameters(
                moshi_lm,
                "tempformer",
                oracle_embedding_mode=args.oracle_embedding_mode,
            ),
            "lr": args.tempformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {  # Depth Transformer
            "params": get_parameters(
                moshi_lm,
                "depformer",
                oracle_embedding_mode=args.oracle_embedding_mode,
            ),
            "lr": args.depformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
    ]

    optimizer = DummyOptim(
        param_groups,
        lr=args.tempformer_learning_rate,  # for accelerator to set deepspeed's lr
        weight_decay=args.weight_decay,  # for accelerator to set deepspeed's weight decay
    )
    # `defaults["lr"]` is used by accelerator to set max_lr of deepspeed's scheduler
    # Ref: Accelerator._prepare_deepspeed()
    optimizer.defaults = {
        "lr": [args.tempformer_learning_rate, args.depformer_learning_rate],
    }

    lr_scheduler_type = None
    if "scheduler" in accelerator.deepspeed_plugin.hf_ds_config.config:
        lr_scheduler_type = accelerator.deepspeed_plugin.hf_ds_config.config["scheduler"]["type"]
    if lr_scheduler_type is None:
        lr_scheduler = None
    elif lr_scheduler_type == "WarmupLR":
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            warmup_num_steps=args.num_warmup_steps,
        )
    elif lr_scheduler_type == "WarmupDecayLR":
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            warmup_num_steps=args.num_warmup_steps,
            total_num_steps=scheduler_total_num_update_steps,
        )
    else:
        raise NotImplementedError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")

    # Prepare everything with our `accelerator`.
    moshi_lm, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        moshi_lm, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    local_num_batches_per_epoch = len(train_dataloader)
    num_update_steps_per_epoch = _compute_num_update_steps_per_epoch(
        local_num_batches_per_epoch,
        args.gradient_accumulation_steps,
    )
    total_num_update_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Resume training
    current_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        current_steps = int(os.path.basename(args.resume_from_checkpoint).split("_")[1])
        starting_epoch = current_steps // num_update_steps_per_epoch

    # Initialize tracker
    config = vars(args)
    config["deepspeed_config"] = json.load(open(args.deepspeed_config_file))
    if args.with_tracking and accelerator.is_main_process:
        wandb_init_kwargs = {
            "name": os.path.basename(args.output_dir),
        }
        if args.run_id_to_resume is not None:
            wandb_init_kwargs.update(
                {
                    "resume": "must",
                    "id": args.run_id_to_resume,
                }
            )
        accelerator.init_trackers(
            project_name=args.project_name,
            config=config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        config["run_id"] = accelerator.get_tracker(name="wandb", unwrap=True).id  # for later resume
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Start training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {global_batch_size}"
    )
    logger.info(f"  Total optimization steps = {total_num_update_steps}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resume from step {current_steps}")

    # Only show the progress bar once on each machine.
    pbar = tqdm(
        range(total_num_update_steps),
        initial=current_steps,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    should_stop = False
    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.resume_from_checkpoint and epoch == starting_epoch:
            num_batches_to_skip = _compute_resume_batch_offset(
                current_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_local_batches_per_epoch=local_num_batches_per_epoch,
            )
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, num_batches_to_skip
            )
        else:
            num_batches_to_skip = 0
            active_dataloader = train_dataloader

        logging_buffer = collections.defaultdict(list)
        active_dataloader_iter = iter(active_dataloader)

        for step in range(num_batches_to_skip, len(train_dataloader)):
            # Update curriculum step before fetching the batch so the collator applies it
            # to the current batch instead of the next one.
            train_data_collator.set_current_step(current_steps)
            try:
                batch = next(active_dataloader_iter)
            except StopIteration:
                break
            moshi_lm.train()
            batch = batch.to(accelerator.device)
            # Forward pass
            total_loss, log = forward(moshi_lm=moshi_lm, batch=batch, args=args)
            for key, value in log.items():
                logging_buffer[f"training_{key}"].append(value)
            # Backward pass
            accelerator.backward(total_loss)
            # The following update steps are handled by deepseed's backward() in accelerator,
            # so we don't need to do them here manually
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
                pbar.update(1)
                current_steps += 1

                # Log metrics
                if current_steps % args.logging_steps == 0:
                    lrs = {
                        "tempformer": f"{optimizer.param_groups[0]['lr']:.3e}",
                        "depformer": f"{optimizer.param_groups[1]['lr']:.3e}",
                    }
                    logger.info(
                        f"Epoch: {epoch}, "
                        f"Steps: {current_steps}, "
                        f"LRs: {lrs}, "
                        f"Loss: {total_loss.item():.5f} "
                        f"(text: {log['loss/text_total'].item():.5f}, "
                        f"audio: {log['loss/audio_total'].item():.5f})"
                    )
                    if args.with_tracking:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in logging_buffer.items()
                            }
                        )

                        accelerator.log(
                            {
                                **{
                                    key: values.nanmean()
                                    for key, values in gathered_metrics.items()
                                },
                                "learning_rate/tempformer": optimizer.param_groups[0]["lr"],
                                "learning_rate/depformer": optimizer.param_groups[1]["lr"],
                            },
                            step=current_steps,
                        )
                    logging_buffer = collections.defaultdict(list)  # reset

                # Evaluate the model
                if args.eval_steps is not None and current_steps % args.eval_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info(f"  Num examples = {len(eval_dataset)}")
                    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
                    logger.info(f"  Num steps = {len(eval_dataloader)}")
                    eval_logging_buffer = collections.defaultdict(list)
                    moshi_lm.eval()
                    for batch in tqdm(
                        eval_dataloader,
                        desc="Evaluating",
                        dynamic_ncols=True,
                        disable=not accelerator.is_main_process,
                    ):
                        batch = batch.to(accelerator.device)
                        with torch.no_grad():
                            _, log = forward(moshi_lm=moshi_lm, batch=batch, args=args)
                            for key, value in log.items():
                                eval_logging_buffer[f"evaluation_{key}"].append(value)
                    if args.with_tracking:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in eval_logging_buffer.items()
                            }
                        )
                        accelerator.log(
                            {key: values.nanmean() for key, values in gathered_metrics.items()},
                            step=current_steps,
                        )

                # Save checkpoint
                if args.save_steps is not None and current_steps % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
                    accelerator.unwrap_model(moshi_lm).materialize_oracle_embedding_for_checkpoint_(
                        args.oracle_embedding_mode
                    )
                    accelerator.save_state(output_dir)

                # Stop training
                if args.max_train_steps is not None and current_steps >= args.max_train_steps:
                    should_stop = True
                    break

        if should_stop:
            break

    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
    accelerator.unwrap_model(moshi_lm).materialize_oracle_embedding_for_checkpoint_(
        args.oracle_embedding_mode
    )
    accelerator.save_state(output_dir)


if __name__ == "__main__":
    main()
