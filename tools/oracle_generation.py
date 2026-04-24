from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Canonical Kame convention:
#   - speaker A -> left / channel 0
#   - speaker B -> right / channel 1
DEFAULT_SPEAKER_TO_CHANNEL = {"A": 0, "B": 1}
SUPPORTED_SPEAKERS = frozenset(DEFAULT_SPEAKER_TO_CHANNEL)
type WordTranscriptRecord = Mapping[str, Any]
type TimeRange = tuple[float, float]
type TurtleAlignment = tuple[str, TimeRange, str]


@dataclass(frozen=True)
class Word:
    text: str
    start_time: float
    end_time: float
    speaker: str


@dataclass(frozen=True)
class OraclePredictionRequest:
    timestamp_ms: int
    conversation_context: str
    next_utterance_hint: str
    total_word_count: int
    trigger_word: str
    recent_words: str
    current_spoken_ratio: float
    current_speaker: str
    next_speaker: str
    target_channel: int


@dataclass(frozen=True)
class OraclePrediction:
    timestamp_ms: int
    conversation_context: str
    prediction: str
    total_word_count: int
    trigger_word: str
    recent_words: str
    current_spoken_ratio: float
    channel: int
    hint: str


PredictFn = Callable[[OraclePredictionRequest], str]


def words_from_word_transcript(records: Iterable[WordTranscriptRecord]) -> list[Word]:
    words: list[Word] = []
    for record in records:
        words.append(
            Word(
                text=str(record["word"]),
                start_time=float(record["start"]),
                end_time=float(record["end"]),
                speaker=str(record["speaker"]),
            )
        )
    return words


def words_from_turtle_alignments(
    alignments: Iterable[TurtleAlignment],
    speaker: str,
) -> list[Word]:
    words: list[Word] = []
    for word_text, time_range, _speaker_tag in alignments:
        words.append(
            Word(
                text=str(word_text),
                start_time=float(time_range[0]),
                end_time=float(time_range[1]),
                speaker=speaker,
            )
        )
    return words


def merge_word_streams(*word_streams: Iterable[Word]) -> list[Word]:
    all_words = [word for stream in word_streams for word in stream]
    all_words.sort(key=lambda word: word.end_time)
    return all_words


class OracleGenerator:
    """Generate oracle predictions for canonical two-speaker A/B transcripts."""

    def __init__(
        self,
        predict_fn: PredictFn,
        *,
        time_interval: float = 0.5,
        target_channel: int | None = None,
        speaker_to_channel: Mapping[str, int] | None = None,
        recent_word_window: int = 5,
    ):
        self.predict_fn = predict_fn
        self.time_interval = time_interval
        self.target_channel = target_channel
        self.speaker_to_channel = dict(
            DEFAULT_SPEAKER_TO_CHANNEL if speaker_to_channel is None else speaker_to_channel
        )
        if set(self.speaker_to_channel) != SUPPORTED_SPEAKERS:
            raise ValueError(
                "OracleGenerator requires speaker_to_channel to define exactly speakers "
                "'A' and 'B'."
            )
        self.recent_word_window = recent_word_window

    def generate_predictions(self, words: Sequence[Word]) -> list[OraclePrediction]:
        predictions: list[OraclePrediction] = []
        words = list(words)
        if not words:
            return predictions
        unexpected_speakers = {word.speaker for word in words} - SUPPORTED_SPEAKERS
        if unexpected_speakers:
            raise ValueError(
                "OracleGenerator only supports canonical A/B transcripts. "
                f"Unexpected speakers: {sorted(unexpected_speakers)}"
            )

        max_time = max(word.end_time for word in words)
        current_time = self.time_interval

        while current_time < max_time:
            words_so_far = [word for word in words if word.end_time <= current_time]
            if len(words_so_far) < 2:
                current_time += self.time_interval
                continue

            conversation_context = self._build_conversation_context(words_so_far)
            recent_words_list = (
                words_so_far[-self.recent_word_window :]
                if len(words_so_far) >= self.recent_word_window
                else words_so_far
            )
            recent_words = " ".join(word.text for word in recent_words_list)

            current_speaker_info = self._get_current_speaker_info(words_so_far, current_time, words)
            current_speaker = current_speaker_info["speaker"]
            next_speaker = "B" if current_speaker == "A" else "A"
            target_channel = self.speaker_to_channel[next_speaker]

            if self.target_channel is not None and target_channel != self.target_channel:
                current_time += self.time_interval
                continue

            next_utterance_hint = self._get_next_utterance_hint(
                words, words_so_far, current_speaker
            )
            if not next_utterance_hint:
                break

            request = OraclePredictionRequest(
                timestamp_ms=int(current_time * 1000),
                conversation_context=conversation_context,
                next_utterance_hint=next_utterance_hint,
                total_word_count=len(words_so_far),
                trigger_word=words_so_far[-1].text,
                recent_words=recent_words,
                current_spoken_ratio=current_speaker_info["ratio"],
                current_speaker=current_speaker,
                next_speaker=next_speaker,
                target_channel=target_channel,
            )
            prediction = self.predict_fn(request).strip()
            if prediction:
                predictions.append(
                    OraclePrediction(
                        timestamp_ms=request.timestamp_ms,
                        conversation_context=request.conversation_context,
                        prediction=prediction,
                        total_word_count=request.total_word_count,
                        trigger_word=request.trigger_word,
                        recent_words=request.recent_words,
                        current_spoken_ratio=request.current_spoken_ratio,
                        channel=request.target_channel,
                        hint=(
                            request.next_utterance_hint
                            if request.current_spoken_ratio > 0.5
                            else ""
                        ),
                    )
                )

            current_time += self.time_interval

        return predictions

    def _build_conversation_context(self, words: Sequence[Word]) -> str:
        if not words:
            return ""

        context_lines: list[str] = []
        current_speaker: str | None = None
        current_text: list[str] = []

        for word in words:
            if current_speaker != word.speaker:
                if current_text:
                    context_lines.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = word.speaker
                current_text = [word.text]
            else:
                current_text.append(word.text)

        if current_text:
            context_lines.append(f"{current_speaker}: {' '.join(current_text)}")

        return "\n".join(context_lines)

    def _get_current_speaker_info(
        self,
        words_so_far: Sequence[Word],
        current_time: float,
        all_words: Sequence[Word],
    ) -> dict[str, str | float]:
        del current_time
        if not words_so_far:
            return {"speaker": "A", "ratio": 0.0}

        current_speaker = words_so_far[-1].speaker
        current_utterance: list[Word] = []
        for i in range(len(words_so_far) - 1, -1, -1):
            if words_so_far[i].speaker == current_speaker:
                current_utterance.insert(0, words_so_far[i])
            else:
                break

        complete_utterance = list(current_utterance)
        current_word = words_so_far[-1]
        start_index = None
        for i, word in enumerate(all_words):
            if word is current_word:
                start_index = i
                break
        if start_index is None:
            for i, word in enumerate(all_words):
                if (
                    word.start_time == current_word.start_time
                    and word.end_time == current_word.end_time
                    and word.text == current_word.text
                    and word.speaker == current_word.speaker
                ):
                    start_index = i
                    break

        if start_index is None:
            for i, word in enumerate(all_words):
                if word.end_time == current_word.end_time and word.text == current_word.text:
                    start_index = i
                    break

        if start_index is not None:
            for i in range(start_index + 1, len(all_words)):
                if all_words[i].speaker != current_speaker:
                    break
                if i > 0 and all_words[i].end_time - all_words[i - 1].end_time >= 1.0:
                    break
                complete_utterance.append(all_words[i])

        if complete_utterance:
            words_spoken = len(current_utterance)
            total_words = len(complete_utterance)
            ratio = words_spoken / total_words if total_words > 0 else 0.0
        else:
            ratio = 1.0

        return {"speaker": current_speaker, "ratio": ratio}

    def _get_next_utterance_hint(
        self,
        all_words: Sequence[Word],
        words_so_far: Sequence[Word],
        current_speaker: str,
    ) -> str:
        if not words_so_far:
            return ""

        last_word_time = words_so_far[-1].end_time
        next_speaker = "B" if current_speaker == "A" else "A"
        next_utterance: list[str] = []
        found_next_speaker = False

        for word in all_words:
            if word.end_time <= last_word_time:
                continue
            if word.speaker == next_speaker:
                found_next_speaker = True
                next_utterance.append(word.text)
            elif found_next_speaker:
                break

        return " ".join(next_utterance)
