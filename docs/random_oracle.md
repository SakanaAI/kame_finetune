# Randomized Training Guidance

KAME training uses text guidance, called an oracle, for upcoming spoken responses.
The random strategy samples guidance from other training dialogues while retaining
the last available ground-truth hint for each response. It preserves the existing
update schedule and requires no LLM calls during guidance preparation.
Inference still uses the KAME back-end LLM.

The default generation strategy remains `llm`. Both strategies use the same
preprocessing and training workflow.

To try generation first, use the [synthetic text-only demo](../data/random_oracle_sample/README.md).
It runs on CPU without an API key or recorded audio and shows the generated oracle JSON.

## Generate Guidance for Your Data

Run these commands from the repository root after [installation](../README.md#installation).

### 1. Prepare the Inputs

- Prepare [canonical word-timed A/B transcripts](../README.md#canonical-dataset-layout).
- Choose a candidate pool containing **training transcripts only**. Do not add validation
  or test transcripts. Keep dialogue filenames stable and unique across splits.
- Use the SentencePiece tokenizer used by your training setup. For the standard English
  setup, follow the demo's [tokenizer preparation](../data/random_oracle_sample/README.md#1-prepare-the-tokenizer).

### 2. Generate Oracle JSON

This example uses `data/my_dataset/text` as both the training input and the candidate pool.
For other splits, keep `--pool_text_dir` set to the training transcripts.
Choose a new or empty output directory.

```bash
uv run -m tools.generate_oracle_from_text \
  --strategy random \
  --text_dir data/my_dataset/text \
  --pool_text_dir data/my_dataset/text \
  --text_tokenizer_path .cache/tokenizers/moshiko/tokenizer_spm_32k_3.model \
  --output_dir data/my_dataset/oracle_raw \
  --seed 42
```

This creates `data/my_dataset/oracle_raw/<dialogue_id>.json` for each transcript
and a `manifest.json` recording the generation settings.

### 3. Continue with Preprocessing and Training

With aligned audio in `data/my_dataset/audio`, follow
[Audio Tokenization](../README.md#2-audio-tokenization) and
[Text Tokenization](../README.md#3-text-tokenization). Then pass the generated
`data/my_dataset/oracle_raw` directory to `--oracle_dir` in
[Oracle Tokenization](../README.md#4-optional-oracle-tokenization) and
[build the Parquet files](../README.md#5-build-parquet-files).

Use the same tokenizer for generation and tokenization; `tools.tokenize_oracle` accepts
`--text_tokenizer_path` for a local model. Keep any custom A/B channel mapping consistent
between both commands.

Follow [Model Initialization](../README.md#model-initialization) and
[Training](../README.md#training), setting `TRAIN_DATA_GLOB` to those Parquet files.
The text-only demo supplies no audio for these steps.

## Generation Rules

- The pool is deduplicated by token sequence. Sampling excludes the current
  dialogue and its turn texts.
- By default, candidate responses have 0.5–2 times as many tokens as the target response.
  Adjust these bounds with `--min_length_ratio` and `--max_length_ratio`.
- Responses are sampled uniformly without replacement for each target response.
- Sampling is reproducible with the same `--seed`, transcripts, tokenizer, and settings.

## Oracle Selection Format

Generated events include a boolean `use_hint`: `true` selects `hint`, and `false` selects
`prediction`. The preprocessing and training tools preserve this choice automatically.
Existing oracle files without this field retain their original behavior.

## Troubleshooting

| Error | What to check |
| --- | --- |
| `Insufficient random responses` | Supply more training dialogues or widen the token-length bounds. The pool must still exclude validation/test data. |
| `requires an empty output directory` | Choose a fresh `--output_dir` for the new run. |

## Known Limitation

The existing training collator can retain trailing tokens from an earlier, longer
guidance update after a shorter update.
