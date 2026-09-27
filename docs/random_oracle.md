# Randomized Training Guidance

KAME training uses text guidance, called an oracle, for upcoming spoken responses.
The random strategy prepares this guidance from transcripts without calling an LLM:
each response group retains its last available ground-truth hint, and all other updates
use responses sampled from other training dialogues. Groups without an available hint
use random responses throughout. The existing update times are preserved.

This strategy uses the existing preprocessing and training workflow. It replaces LLM calls
during guidance preparation; inference still uses the KAME back-end LLM. The default
generation strategy remains `llm`.

To try generation first, use the [synthetic text-only demo](../data/random_oracle_sample/README.md).
It runs on CPU without an API key or recorded audio and shows the generated oracle JSON.

## Generate Guidance for Your Data

Run these commands from the repository root after [installation](../README.md#installation).

### 1. Prepare the Inputs

- Prepare [canonical word-timed A/B transcripts](../README.md#canonical-dataset-layout).
  Overlapping words and short responses are accepted. The shared generator uses the
  supplied word order and timestamps to extract responses; random generation does not
  reorder words or add updates for short turns. Check the extracted hints for your data.
- Choose a candidate pool containing **training transcripts only**. Do not add validation
  or test transcripts. Keep dialogue filenames stable and unique across splits.
- Use the SentencePiece tokenizer used by your training setup. For the standard English
  setup, follow the demo's [tokenizer preparation](../data/random_oracle_sample/README.md#1-prepare-the-tokenizer).

### 2. Generate Oracle JSON

This example treats `data/my_dataset/text` as training data and also uses it as the pool.
When processing another split, keep `--pool_text_dir` pointed at the training transcripts.
Use a new or empty output directory; random generation does not support `--resume`.

```bash
uv run -m tools.generate_oracle_from_text \
  --strategy random \
  --text_dir data/my_dataset/text \
  --pool_text_dir data/my_dataset/text \
  --text_tokenizer_path .cache/tokenizers/moshiko/tokenizer_spm_32k_3.model \
  --output_dir data/my_dataset/oracle_raw \
  --seed 42
```

This creates `data/my_dataset/oracle_raw/<dialogue_id>.json` for each transcript.
After all dialogues succeed, `manifest.json` records the settings, hint policy, hashes
of the inputs, pool transcripts, tokenizer, and outputs, and per-dialogue and total
statistics. The totals are also printed. Missing hints or updates are counted rather
than rejected; see [Generation Rules](#generation-rules).

### 3. Continue with Preprocessing and Training

With aligned audio in `data/my_dataset/audio`, continue with
[Audio Tokenization](../README.md#2-audio-tokenization) and
[Text Tokenization](../README.md#3-text-tokenization). Pass the generated
`data/my_dataset/oracle_raw` directory to `--oracle_dir` in
[Oracle Tokenization](../README.md#4-optional-oracle-tokenization), then build the Parquet files.
Use the same tokenizer for generation and tokenization; `tools.tokenize_oracle` accepts
`--text_tokenizer_path` for a local model. Keep any custom A/B channel mapping consistent
between both commands.

Before training, note that selecting a hint does not guarantee a pure hint in the final
training tensor: the existing collator can retain trailing tokens from an earlier,
longer update after a shorter hint. This implementation does not change that behavior.

Follow [Model Initialization](../README.md#model-initialization) and
[Training](../README.md#training), setting `TRAIN_DATA_GLOB` to those Parquet files.
The text-only demo supplies no audio for these steps.

## Generation Rules

Updates from the shared generator are grouped by their target start index and speaker,
so repeated response text remains separate. By default both channels are included,
with A mapped to 0 and B to 1.
`--target_channel` filters targets using the mapping set by `--A_channel` and `--B_channel`.

Event times and hint availability use the existing generator, with `--time_interval 0.5`
by default. A nonempty hint is available when `current_spoken_ratio > 0.5`. Each group
selects its last available hint, if any; all remaining events receive random responses,
including events after the selected hint. A group with no available hint uses only random
responses. The generator does not add endpoint events or require an event for every turn.
An empty event sequence is saved as `[]`.

The text of the selected hint is the reference for token-length filtering. If no hint
is available, the last update's extracted response text is used. Pool construction uses
the same rule, then deduplicates by token sequence. Sampling excludes the current dialogue
and all its turn texts, keeps candidates within 0.5–2 times the reference's token length,
and samples without replacement within each response group. Adjust the bounds with `--min_length_ratio` and
`--max_length_ratio`. Too few eligible candidates raises an error.

The seed and stable dialogue/response identities make results independent of dialogue
processing order for the same inputs, tokenizer, and settings.

The manifest's `statistics` contains `dialogues` and `totals`, with event, response-group
and selected-hint counts plus:

- `groups_without_hint`: groups that receive only random responses.
- `groups_with_updates_after_hint`: groups with updates after their selected hint.
- `transcript_turns_without_events`: consecutive same-speaker word blocks without a
  targeted update, excluding the opening block and channels not selected for generation.
  This counts blocks in the supplied transcript, not inferred conversational turns.

## Oracle Selection Format

Generated events carry a boolean `use_hint`: `true` selects `hint`, and `false` selects
`prediction`. Tokenization and dataset preparation preserve this choice automatically.
For custom oracle JSON, include the field on every event or omit it on every event in a
dialogue. Partial specifications, invalid flags, and masks with the wrong length are errors.

In NPZ, absent `A_event_use_hint` and `B_event_use_hint` keys mean unspecified selection.
Explicit selection supplies both keys as integer arrays, including an empty array for a
channel with no events. An oracle JSON containing only `[]` has no format information and
remains unspecified.

Parquet and preprocessing use nullable integer lists for the selection masks:

| Value | Meaning |
| --- | --- |
| `null` / Python `None` | Unspecified selection; retain existing selection and skipping behavior. |
| `[]` | Explicit selection with no events in this channel or chunk. |
| `[0, 1, ...]` | One flag per event: 0 selects `prediction`, 1 selects `hint`. |

These distinctions are preserved across shards, speaker views, and chunks. Parquet stores
`A_oracle_event_use_hint` and `B_oracle_event_use_hint` as nullable `list<int8>` columns;
preprocessing produces `oracle_event_use_hint` with the same type.

Events marked `use_hint: true` are protected from training-time event skipping, including
with a target-channel filter. Other events follow the existing skip settings. In hint-only
mode, explicit random events are omitted. Timing jitter and shifts still apply.

## Troubleshooting

| Error | What to check |
| --- | --- |
| `Insufficient random responses` | Supply more training dialogues or widen the token-length bounds. The pool must still exclude validation/test data. |
| `requires an empty output directory` | Choose a fresh `--output_dir` for the new run. |
