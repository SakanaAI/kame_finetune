#!/usr/bin/env bash
set -euo pipefail

# Reference full-training example.
# The current training stack requires DeepSpeed. This script keeps the default
# finetuning target and is closer to the configuration used for real runs, but
# may require substantial GPU memory and may be better suited to multi-GPU jobs.
#
# Optional environment variables:
#   TRAIN_DATA_GLOB
#   MODEL_DIR
#   OUTPUT_DIR
#   DEEPSPEED_CONFIG
#   NUM_PROCESSES
#   MODEL_DTYPE
#   MAX_LENGTH
#   NUM_EPOCHS
#   PER_DEVICE_BATCH_SIZE
#   GRADIENT_ACCUMULATION_STEPS
#   NUM_WARMUP_STEPS
#   LOGGING_STEPS
#   SAVE_STEPS
#   USE_ORACLE

TRAIN_DATA_GLOB="${TRAIN_DATA_GLOB:-processed_data/spokenwoz_sample/train_text_oracle_a0b1_events-*.parquet}"
MODEL_DIR="${MODEL_DIR:-init_models/moshiko-one_streams-bfloat16}"
OUTPUT_DIR="${OUTPUT_DIR:-output/moshiko-finetuned}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-ds_configs/zero3-bfp16-warmlr-act_ckpt.json}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
USE_ORACLE="${USE_ORACLE:-1}"

EXTRA_ARGS=()
if [ "${USE_ORACLE}" = "1" ]; then
    EXTRA_ARGS+=(--use_oracle)
fi

uv run accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --use_deepspeed \
    --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
    finetune.py \
        --launcher accelerate \
        --output_dir "${OUTPUT_DIR}" \
        --train_data_files "${TRAIN_DATA_GLOB}" \
        --model_dir "${MODEL_DIR}" \
        --model_dtype "${MODEL_DTYPE}" \
        --max_length "${MAX_LENGTH}" \
        --min_length 128 \
        --num_train_epochs "${NUM_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --num_warmup_steps "${NUM_WARMUP_STEPS}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        "${EXTRA_ARGS[@]}"
