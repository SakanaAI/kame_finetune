# Synthetic random-oracle example

This demo generates [randomized training guidance](../../docs/random_oracle.md) from six
fictional English dialogues. It runs on CPU without an API key and produces oracle JSON
files for inspection. It does not train a speech model or provide a paper-reproduction
experiment configuration.

The A/B transcripts have simple, non-overlapping turns. Times are illustrative seconds;
there is no recorded audio. The files are included under the repository's Apache-2.0 license.

Run all commands from the repository root after [installation](../../README.md#installation).

## 1. Prepare the Tokenizer

Download the SentencePiece tokenizer used by the repository's standard English setup:

```bash
uv run python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="kyutai/moshiko-pytorch-bf16",
    filename="tokenizer_spm_32k_3.model",
    local_dir=".cache/tokenizers/moshiko",
)
PY
```

This downloads only the tokenizer; the demo needs no KAME model weights. If you already
have a local tokenizer, use its path in the generation command instead. For your own
training data, generation and tokenization must use the same tokenizer.

## 2. Generate Oracle JSON

The six dialogues serve as both inputs and a small training-only candidate pool. Each
provides one response; generation excludes the current dialogue when sampling candidates.
Use a new or empty output directory. To rerun the demo, choose a new `--output_dir`.

```bash
uv run -m tools.generate_oracle_from_text \
  --strategy random \
  --text_dir data/random_oracle_sample/text \
  --pool_text_dir data/random_oracle_sample/text \
  --text_tokenizer_path .cache/tokenizers/moshiko/tokenizer_spm_32k_3.model \
  --output_dir processed_data/random_oracle_sample/oracle_raw \
  --seed 42
```

This creates six dialogue JSON files and a `manifest.json` in
`processed_data/random_oracle_sample/oracle_raw`. Each dialogue has three random updates
followed by one selected ground-truth hint.

## 3. Check the Output

Inspect the update times, target channel, and selection flags for `japan.json`:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

events = json.loads(Path("processed_data/random_oracle_sample/oracle_raw/japan.json").read_text())
print([(event["timestamp_ms"], event["channel"], event["use_hint"]) for event in events])
print(events[-1]["hint"])
PY
```

Expected output:

```text
[(500, 1, False), (1000, 1, False), (1500, 1, False), (2000, 1, True)]
Tokyo is the capital of Japan
```

Channel 1 is speaker B. `False` selects the sampled text in `prediction`; `True` selects
the ground-truth text in `hint`. Open the JSON file to inspect the sampled responses.

For audio-backed training, follow [Generate Guidance for Your Data](../../docs/random_oracle.md#generate-guidance-for-your-data)
with your own aligned audio/transcript pairs. The demo stops at oracle generation.
