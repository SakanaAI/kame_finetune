"""CPU-only random oracle construction from word-timed A/B transcripts."""

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
    OraclePredictionRequest,
    Word,
    prediction_from_request,
)


@dataclass(frozen=True)
class ResponseCandidate:
    text: str
    token_ids: tuple[int, ...]
    owners: frozenset[str]


@dataclass(frozen=True)
class RandomOracleResult:
    predictions: list[OraclePrediction]
    statistics: dict[str, int]


@dataclass(frozen=True)
class _ResponseGroup:
    indices: list[int]
    hint_index: int | None
    reference_text: str


def _response_groups(
    requests: Sequence[OraclePredictionRequest],
) -> dict[tuple[int, str], _ResponseGroup]:
    indices_by_target: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, request in enumerate(requests):
        indices_by_target[request.target_index, request.next_speaker].append(index)
    groups = {}
    for key, indices in indices_by_target.items():
        hint_index = next(
            (i for i in reversed(indices) if prediction_from_request(requests[i], "").hint.strip()),
            None,
        )
        reference_index = hint_index if hint_index is not None else indices[-1]
        groups[key] = _ResponseGroup(
            indices, hint_index, requests[reference_index].next_utterance_hint
        )
    return groups


def validate_transcript(words: Sequence[Word]) -> None:
    """Validate individual words, allowing overlap and retaining the supplied order."""
    for word in words:
        if word.speaker not in {"A", "B"} or not word.text.strip():
            raise ValueError("Random oracle input requires nonempty words and A/B speakers")
        if not (
            math.isfinite(word.start_time)
            and math.isfinite(word.end_time)
            and 0 <= word.start_time <= word.end_time
        ):
            raise ValueError("Word times must be finite with 0 <= start <= end")


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
        requests = list(generator.generate_requests(words))
        for group in _response_groups(requests).values():
            text = group.reference_text
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
) -> RandomOracleResult:
    """Retain the last available hint per response and randomize all other events.

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
    groups = _response_groups(requests)
    statistics = {
        "events": len(requests),
        "response_groups": len(groups),
        "selected_hints": sum(group.hint_index is not None for group in groups.values()),
        "groups_without_hint": sum(group.hint_index is None for group in groups.values()),
        "groups_with_updates_after_hint": sum(
            group.hint_index is not None and group.hint_index != group.indices[-1]
            for group in groups.values()
        ),
        "transcript_turns_without_events": 0,
    }

    # Transcript blocks serve text exclusion and diagnostics, not event generation.
    covered_indices = {request.target_index for request in requests}
    excluded_tokens = set()
    for speaker, block in groupby(enumerate(words), key=lambda item: item[1].speaker):
        turn = list(block)
        excluded_tokens.add(
            tuple(tokenizer.encode(" ".join(word.text for _, word in turn), out_type=int))
        )
        if (
            turn[0][0] > 0
            and (target_channel is None or generator.speaker_to_channel[speaker] == target_channel)
            and not any(index in covered_indices for index, _word in turn)
        ):
            statistics["transcript_turns_without_events"] += 1
    excluded_tokens.update(
        tuple(tokenizer.encode(text, out_type=int))
        for text in {request.next_utterance_hint for request in requests}
    )
    predictions: dict[int, OraclePrediction] = {}
    for (target_index, speaker), group in groups.items():
        random_indices = [i for i in group.indices if i != group.hint_index]
        key = f"{seed}\0{dialogue_id}\0{target_index}\0{speaker}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
        texts = _sample_responses(
            pool,
            count=len(random_indices),
            dialogue_id=dialogue_id,
            excluded_tokens=excluded_tokens,
            target_length=len(tokenizer.encode(group.reference_text, out_type=int)),
            min_length_ratio=min_length_ratio,
            max_length_ratio=max_length_ratio,
            rng=rng,
        )
        for index, text in zip(random_indices, texts, strict=True):
            predictions[index] = prediction_from_request(requests[index], text, use_hint=False)
        if group.hint_index is not None:
            predictions[group.hint_index] = prediction_from_request(
                requests[group.hint_index], "", use_hint=True
            )
    return RandomOracleResult([predictions[i] for i in range(len(requests))], statistics)
