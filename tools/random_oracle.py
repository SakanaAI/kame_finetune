"""CPU-only random oracle construction for turn-aligned A/B transcripts."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby

from sentencepiece import SentencePieceProcessor

from tools.oracle_generation import (
    OracleGenerator,
    OraclePrediction,
    Word,
    prediction_from_request,
)


@dataclass(frozen=True)
class ResponseCandidate:
    text: str
    token_ids: tuple[int, ...]
    owners: frozenset[str]


def validate_transcript(words: Sequence[Word]) -> None:
    """Validate the non-overlapping, word-timed input supported by this example."""
    previous_end = 0.0
    for word in words:
        if word.speaker not in {"A", "B"} or not word.text.strip():
            raise ValueError("Random oracle input requires nonempty words and A/B speakers")
        if not (
            math.isfinite(word.start_time)
            and math.isfinite(word.end_time)
            and 0 <= word.start_time <= word.end_time
        ):
            raise ValueError("Word times must be finite with 0 <= start <= end")
        if word.start_time < previous_end:
            raise ValueError("Random oracle input requires ordered, non-overlapping words")
        previous_end = word.end_time


def build_response_pool(
    dialogues: Iterable[tuple[str, Sequence[Word]]],
    tokenizer: SentencePieceProcessor,
    *,
    time_interval: float = 0.5,
) -> tuple[ResponseCandidate, ...]:
    """Collect training-owned responses using the same target extraction as generation."""
    texts: dict[tuple[int, ...], str] = {}
    owners: dict[tuple[int, ...], set[str]] = defaultdict(set)
    generator = OracleGenerator(time_interval=time_interval)
    for dialogue_id, words in dialogues:
        validate_transcript(words)
        for request in generator.generate_requests(words):
            text = request.next_utterance_hint
            tokens = tuple(tokenizer.encode(text, out_type=int))
            if not tokens:
                continue
            texts[tokens] = min(texts.get(tokens, text), text)
            owners[tokens].add(dialogue_id)
    if not texts:
        raise ValueError("No response candidates found in the training transcripts")
    return tuple(
        ResponseCandidate(texts[tokens], tokens, frozenset(owners[tokens]))
        for tokens in sorted(texts)
    )


def _sample_responses(
    pool: Sequence[ResponseCandidate],
    *,
    count: int,
    dialogue_id: str,
    excluded_tokens: set[tuple[int, ...]],
    target_length: int,
    min_length_ratio: float,
    max_length_ratio: float,
    rng: random.Random,
) -> list[str]:
    def eligible(index: int) -> bool:
        candidate = pool[index]
        return (
            dialogue_id not in candidate.owners
            and candidate.token_ids not in excluded_tokens
            and min_length_ratio * target_length
            <= len(candidate.token_ids)
            <= max_length_ratio * target_length
        )

    # Rejection draws avoid scanning a large training pool for every response.
    selected: list[int] = []
    seen: set[int] = set()
    for _ in range(max(1000, count * 20)):
        if len(selected) == count or not pool:
            break
        index = rng.randrange(len(pool))
        if index not in seen and eligible(index):
            selected.append(index)
            seen.add(index)
    if len(selected) < count:
        remaining = [i for i in range(len(pool)) if i not in seen and eligible(i)]
        if len(selected) + len(remaining) < count:
            raise ValueError(
                f"Insufficient random responses for dialogue {dialogue_id!r}: "
                f"need {count}, found {len(selected) + len(remaining)}. "
                "Supply more training dialogues or widen the token-length bounds."
            )
        selected.extend(rng.sample(remaining, count - len(selected)))
    return [pool[index].text for index in selected]


def generate_random_predictions(
    words: Sequence[Word],
    *,
    dialogue_id: str,
    pool: Sequence[ResponseCandidate],
    tokenizer: SentencePieceProcessor,
    seed: int = 42,
    time_interval: float = 0.5,
    target_channel: int | None = None,
    speaker_to_channel: Mapping[str, int] | None = None,
    min_length_ratio: float = 0.5,
    max_length_ratio: float = 2.0,
) -> list[OraclePrediction]:
    """Keep the last nonempty eligible hint per response; randomize other events.

    Event times, response extraction and hint eligibility (> half the input words)
    follow OracleGenerator. No event is added to force an endpoint hint.
    """
    validate_transcript(words)
    if not (
        math.isfinite(min_length_ratio)
        and math.isfinite(max_length_ratio)
        and 0 < min_length_ratio <= max_length_ratio
    ):
        raise ValueError("Token-length bounds must be finite with 0 < min <= max")
    generator = OracleGenerator(
        time_interval=time_interval,
        target_channel=target_channel,
        speaker_to_channel=speaker_to_channel,
    )
    requests = list(generator.generate_requests(words))
    # Exclude text in any turn of this dialogue, including its opening turn.
    excluded_tokens = {
        tuple(tokenizer.encode(" ".join(w.text for w in turn), out_type=int))
        for _speaker, turn in groupby(words, key=lambda w: w.speaker)
    }
    excluded_tokens.update(
        tuple(tokenizer.encode(request.next_utterance_hint, out_type=int)) for request in requests
    )
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, request in enumerate(requests):
        groups[request.target_index, request.next_speaker].append(index)

    predictions: dict[int, OraclePrediction] = {}
    for (target_index, speaker), indices in groups.items():
        eligible = [i for i in indices if requests[i].current_spoken_ratio > 0.5]
        hint_index = eligible[-1] if eligible else None
        random_indices = [i for i in indices if i != hint_index]
        target = requests[indices[0]].next_utterance_hint
        key = f"{seed}\0{dialogue_id}\0{target_index}\0{speaker}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
        texts = _sample_responses(
            pool,
            count=len(random_indices),
            dialogue_id=dialogue_id,
            excluded_tokens=excluded_tokens,
            target_length=len(tokenizer.encode(target, out_type=int)),
            min_length_ratio=min_length_ratio,
            max_length_ratio=max_length_ratio,
            rng=rng,
        )
        for index, text in zip(random_indices, texts, strict=True):
            predictions[index] = prediction_from_request(requests[index], text, use_hint=False)
        if hint_index is not None:
            predictions[hint_index] = prediction_from_request(
                requests[hint_index], "", use_hint=True
            )
    return [predictions[i] for i in range(len(requests))]
