from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.oracle_generation import (
    OracleGenerator,
    OraclePrediction,
    OraclePredictionRequest,
    PredictFn,
    words_from_word_transcript,
)

DEFAULT_MODEL_NAME = "gpt-4.1-nano"


def prediction_to_record(prediction: OraclePrediction) -> dict[str, object]:
    return {
        "timestamp_ms": prediction.timestamp_ms,
        "conversation_context": prediction.conversation_context,
        "prediction": prediction.prediction,
        "total_word_count": prediction.total_word_count,
        "trigger_word": prediction.trigger_word,
        "recent_words": prediction.recent_words,
        "current_spoken_ratio": prediction.current_spoken_ratio,
        "channel": prediction.channel,
        "hint": prediction.hint,
    }


def fallback_prediction_for_request(request: OraclePredictionRequest) -> str:
    """Fallback used when the OpenAI request fails.

    Keep the "no hint in the first half" semantics intact by returning an empty
    prediction when the current speaker is at or before the halfway point.
    """
    if request.current_spoken_ratio <= 0.5:
        return ""
    return request.next_utterance_hint


def build_user_prompt(request: OraclePredictionRequest) -> str:
    if request.current_spoken_ratio <= 0.5:
        prompt = f"""You are predicting what the next speaker will say in a conversation.

Conversation so far:
{request.conversation_context}

The current speaker ({request.current_speaker}) has only spoken {request.current_spoken_ratio:.0%} of their turn and is still talking.
Predict what {request.next_speaker} will say next when it is their turn.

Since the current speaker has just begun, make your prediction based only on the conversation history so far.
Generate a natural, contextually appropriate response."""
    else:
        prompt = f"""You are predicting what the next speaker will say in a conversation.

Conversation so far:
{request.conversation_context}

The current speaker ({request.current_speaker}) has spoken {request.current_spoken_ratio:.0%} of their turn and is still talking.
Predict what {request.next_speaker} will say next when it is their turn.

Hidden hint (do not mention or quote this directly, but let it guide your prediction):
The actual next utterance will be similar to: "{request.next_utterance_hint}"

Guidelines based on current speaker's progress ({request.current_spoken_ratio:.1%}):"""

        if request.current_spoken_ratio <= 0.65:
            prompt += """
- You have only heard about half of the current turn.
- Rely primarily on the conversation flow so far.
- You may use hint keywords, but avoid following the hint too closely."""
        elif request.current_spoken_ratio <= 0.8:
            prompt += """
- You are in the latter half of the current turn.
- Respect the conversation flow while referencing the hint.
- Include some content that differs from the hint."""
        elif request.current_spoken_ratio <= 0.95:
            prompt += """
- Most of the current turn is complete.
- The hint can be an important clue.
- Avoid copying the hint verbatim; prefer a slightly different expression."""
        elif request.current_spoken_ratio <= 0.99:
            prompt += """
- You are approaching the end of the current turn.
- The hint can be used more directly, while still sounding like a natural prediction."""
        else:
            prompt += """
- You have effectively heard the full current turn.
- The output may closely match the hint if that is the most natural continuation."""

    prompt += f"""

Generate a natural response that sounds like a genuine spoken prediction.

Important guidelines:
- Speak confidently on the predicted topic without asking for confirmation
- Use at most 30 words and at least 10 words when possible
- Use conversational spoken language only
- Do not include quotes, bullets, or explanations
- Avoid text formatting and avoid punctuation other than periods and commas

Respond with only the predicted text that {request.next_speaker} will say."""
    return prompt


def make_openai_predict_fn(
    *,
    model_name: str,
    fallback_to_hint_on_error: bool = False,
):
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    def predict(request: OraclePredictionRequest) -> str:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that predicts the next response in a "
                            "spoken conversation."
                        ),
                    },
                    {"role": "user", "content": build_user_prompt(request)},
                ],
                stream=False,
            )
        except Exception:
            if fallback_to_hint_on_error:
                return fallback_prediction_for_request(request)
            raise

        return (response.choices[0].message.content or "").strip()

    return predict


def generate_oracle_records(
    transcript_records: list[dict[str, object]],
    *,
    predict_fn: PredictFn,
    time_interval: float,
    target_channel: int | None,
    speaker_to_channel: dict[str, int],
) -> list[dict[str, object]]:
    words = words_from_word_transcript(transcript_records)
    generator = OracleGenerator(
        predict_fn,
        time_interval=time_interval,
        target_channel=target_channel,
        speaker_to_channel=speaker_to_channel,
    )
    predictions = generator.generate_predictions(words)
    return [prediction_to_record(prediction) for prediction in predictions]


def process_text_file(
    text_path: Path,
    output_path: Path,
    *,
    predict_fn: PredictFn,
    time_interval: float,
    target_channel: int | None,
    speaker_to_channel: dict[str, int],
) -> int:
    with text_path.open(encoding="utf-8") as f:
        transcript_records = json.load(f)

    oracle_records = generate_oracle_records(
        transcript_records,
        predict_fn=predict_fn,
        time_interval=time_interval,
        target_channel=target_channel,
        speaker_to_channel=speaker_to_channel,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(oracle_records, f, ensure_ascii=False, indent=2)

    return len(oracle_records)


def main(args: argparse.Namespace) -> None:
    speaker_to_channel = {"A": args.A_channel, "B": args.B_channel}
    predict_fn = make_openai_predict_fn(
        model_name=args.model,
        fallback_to_hint_on_error=args.fallback_to_hint_on_error,
    )

    text_paths = sorted(Path(args.text_dir).glob("*.json"))
    if args.limit is not None:
        text_paths = text_paths[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for text_path in text_paths:
        output_path = output_dir / text_path.name
        if args.resume and output_path.exists():
            print(f"Skipping {text_path.name}: oracle already exists")
            continue

        num_records = process_text_file(
            text_path,
            output_path,
            predict_fn=predict_fn,
            time_interval=args.time_interval,
            target_channel=args.target_channel,
            speaker_to_channel=speaker_to_channel,
        )
        print(f"Wrote {num_records} oracle events to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate oracle_raw JSON files directly from canonical text transcripts."
    )
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--time_interval", type=float, default=0.5)
    parser.add_argument("--target_channel", type=int, choices=[0, 1], default=None)
    parser.add_argument("--A_channel", type=int, default=0)
    parser.add_argument("--B_channel", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fallback_to_hint_on_error", action="store_true")
    main(parser.parse_args())
