# Randomized Training Guidance

KAME training uses text guidance, called an oracle, for upcoming spoken responses.
The random strategy prepares this guidance from transcripts without calling an LLM:
intermediate updates use responses sampled from other training dialogues, and the final
scheduled update for each target response selects its ground-truth transcript as a hint.
Here, "final" means the last guidance update for that response, not the end of its audio.

This strategy uses the existing preprocessing and training workflow. It replaces LLM calls
during guidance preparation; inference still uses the KAME back-end LLM. The default
generation strategy remains `llm`.

To try generation first, use the [synthetic text-only demo](../data/random_oracle_sample/README.md).
It runs on CPU without an API key or recorded audio and shows the generated oracle JSON.

## Generate Guidance for Your Data

Run these commands from the repository root after [installation](../README.md#installation).

### 1. Prepare the Inputs

- Prepare [canonical word-timed A/B transcripts](../README.md#canonical-dataset-layout).
  Words must be ordered and non-overlapping. Real recordings may need preprocessing for
  short backchannels, turn boundaries, and overlapping speech.
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
After all dialogues succeed, `manifest.json` records the settings, hint policy, and hashes
of the inputs, pool transcripts, tokenizer, and outputs. If a target response has no
scheduled events or cannot end with an eligible hint, generation stops with an error;
see [Troubleshooting](#troubleshooting).

### 3. Continue with Preprocessing and Training

With aligned audio in `data/my_dataset/audio`, continue with
[Audio Tokenization](../README.md#2-audio-tokenization) and
[Text Tokenization](../README.md#3-text-tokenization). Pass the generated
`data/my_dataset/oracle_raw` directory to `--oracle_dir` in
[Oracle Tokenization](../README.md#4-optional-oracle-tokenization), then build the Parquet files.
Use the same tokenizer for generation and tokenization; `tools.tokenize_oracle` accepts
`--text_tokenizer_path` for a local model. Keep any custom A/B channel mapping consistent
between both commands.

Before training, note that the final-hint guarantee applies to the generated event sequence.
The existing collator can retain trailing tokens from an earlier, longer update after a
shorter hint. This implementation does not change that behavior.

Follow [Model Initialization](../README.md#model-initialization) and
[Training](../README.md#training), setting `TRAIN_DATA_GLOB` to those Parquet files.
The text-only demo supplies no audio for these steps.

## Generation Rules

Each turn following a speaker change is a target response; the opening turn is excluded.
Targets are identified by their transcript start index and speaker, so repeated response
text remains separate. By default both channels are included, with A mapped to 0 and B to 1.
`--target_channel` filters targets using the mapping set by `--A_channel` and `--B_channel`.

Event times and hint eligibility use the existing generator, with `--time_interval 0.5`
by default. Each target must have at least one event, and its last event must have a
nonempty hint and `current_spoken_ratio > 0.5`. Earlier events receive random responses.
A single hint-only event is valid; the ratio need not increase monotonically. The generator
does not add endpoint events or silently drop targets that fail these conditions.

Candidate responses are extracted from the pool transcripts using the same generator and
deduplicated by token sequence. Sampling excludes the current dialogue and all its turn
texts, keeps candidates within 0.5–2 times the target's token length, and samples without
replacement within each response. Adjust the bounds with `--min_length_ratio` and
`--max_length_ratio`. Too few eligible candidates raises an error.

The seed and stable dialogue/response identities make results independent of dialogue
processing order for the same inputs, tokenizer, and settings.

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

Response validation errors identify the dialogue, target start index, speaker/channel,
and, when available, the last event's time and ratio.

| Error | What to check |
| --- | --- |
| `no_scheduled_events` | The target has no events at the chosen interval. Check timestamps and turn boundaries; a shorter `--time_interval` may help if appropriate for the data. |
| `no_eligible_hint` | No event can select a nonempty hint with ratio greater than 0.5. Inspect the indicated turn and its scheduled updates. |
| `events_after_last_eligible_hint` | An eligible hint is followed by an ineligible update. Review pauses and turn segmentation; the shared ratio calculation can decrease after a pause. |
| `Insufficient random responses` | Supply more training dialogues or widen the token-length bounds. The pool must still exclude validation/test data. |
| `requires an empty output directory` | Choose a fresh `--output_dir` for the new run. |
