# SpokenWOZ Sample Data

This directory contains a small sample derived from the SpokenWOZ dataset for
demonstration and preprocessing tests in `kame_finetune`.

Included sample files:
- `audio/`: stereo dialogue wav files
- `text/`: word-level transcript JSON files aligned to the sample audio
- `oracle_raw/`: oracle prediction JSON files generated for the bundled sample

Source dataset:
- SpokenWOZ
- Project page: https://spokenwoz.github.io/SpokenWOZ-github.io/
- Original license: CC BY-NC 4.0

License for this sample:
- The sample data in this directory is distributed under CC BY-NC 4.0.
- This data license is separate from the software license used for the code in this repository.

Notes:
- This directory is included only as a lightweight sample for testing the
  preprocessing and training workflow.
- To keep the bundled sample lightweight, the included `oracle_raw/` files use
  a coarse 5-second event interval.
- This differs from the default `tools.generate_oracle_from_text` setting
  (`--time_interval 0.5`).
- As a result, `current_spoken_ratio` values in the bundled sample may look
  more coarsely stepped than in oracle files generated with the default
  settings.
- If you use SpokenWOZ beyond this bundled sample, please review the original
  dataset documentation and license terms.
