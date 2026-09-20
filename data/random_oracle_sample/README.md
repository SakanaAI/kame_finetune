# Synthetic random-oracle example

These six fictional English dialogues contain word-level A/B transcripts
with simple, non-overlapping turns. Times are illustrative seconds, not
alignments to recorded audio. The files are included under the repository's
Apache-2.0 license.

Run the command in the main README's **Random oracle generation without an
LLM** section with the same local SentencePiece tokenizer you use for KAME.
Each file supplies one target response to the candidate pool. The generated
`oracle_raw/<dialogue_id>.json` contains the existing scheduled events, random
responses from other dialogues, and one explicitly selected final hint.
The six-file pool is deliberately large enough for this small example.

Use your own aligned audio/transcript pairs for the remaining preprocessing
and finetuning steps. This text-only example does not supply audio, model
weights, or a paper-reproduction experiment configuration.
