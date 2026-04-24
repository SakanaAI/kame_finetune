from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import get_worker_info


def main_speaker_streams(
    batched_examples: dict[str, list[Any]],
    speakers: list[str],
) -> list[np.ndarray]:
    """
    Pick the main speaker and create the streams.
    """
    list_of_streams: list[np.ndarray] = []
    for speaker in speakers:
        other = {"A": "B", "B": "A"}[speaker]
        for main_example, other_example in zip(
            batched_examples[speaker], batched_examples[other], strict=True
        ):
            streams = np.concat(
                [
                    np.array(main_example),  # 1 text + 8 audio
                    np.array(other_example)[1:],  # 8 audio
                ],
                axis=0,
            )
            list_of_streams.append(streams)
    return list_of_streams


def delay_and_pad_streams(
    *,
    list_of_streams: list[np.ndarray],
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
) -> list[np.ndarray]:
    """
    Apply delays and padding to the streams.
    """
    max_delay = max(delays)

    list_of_delayed_streams: list[np.ndarray] = []

    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        delayed = np.zeros((num_streams, num_frames + max_delay), dtype=streams.dtype)
        for i, delay in enumerate(delays):
            delayed[i, :delay] = initial_token_ids[i]
            delayed[i, delay : delay + num_frames] = streams[i]
            delayed[i, delay + num_frames :] = padding_token_ids[i]
        delayed = np.concat(
            [
                np.array(initial_token_ids, dtype=streams.dtype)[
                    :, None
                ],  # add initial token at the beginning
                delayed,
            ],
            axis=1,
        )

        list_of_delayed_streams.append(delayed)

    return list_of_delayed_streams


@dataclass(frozen=True)
class OracleEvents:
    """
    Event-level oracle representation.

    `*_values` + `*_offsets` represent ragged list[list[int]] without pickle:
      tokens_i = values[offsets[i]:offsets[i+1]]
    """

    frame_pos: np.ndarray  # (E,) int32, pre-delay positions; aligned later in preprocess
    ratio: np.ndarray  # (E,) float32
    skip_forbid: np.ndarray  # (E,) int8
    pred_values: np.ndarray  # (sum_pred,) int32
    pred_offsets: np.ndarray  # (E+1,) int32
    hint_values: np.ndarray  # (sum_hint,) int32
    hint_offsets: np.ndarray  # (E+1,) int32


def _subset_packed_ragged(
    values: np.ndarray, offsets: np.ndarray, keep_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Subset packed ragged tokens by event indices, returning new (values, offsets)."""
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    new_offsets = np.zeros((keep_indices.size + 1,), dtype=np.int32)

    pieces: list[np.ndarray] = []
    total = 0
    for j, i in enumerate(keep_indices.tolist()):
        start_idx = int(offsets[i])
        end_idx = int(offsets[i + 1])
        seg = values[start_idx:end_idx]
        pieces.append(seg)
        total += int(seg.size)
        new_offsets[j + 1] = total

    if total == 0:
        new_values = np.empty((0,), dtype=np.int32)
    else:
        new_values = np.concatenate(pieces).astype(np.int32, copy=False)

    return new_values, new_offsets


def _subset_oracle_events(
    events: OracleEvents,
    keep_indices: np.ndarray,
    *,
    pos_shift: int = 0,
) -> OracleEvents:
    """
    Create a new OracleEvents by selecting events and optionally shifting positions.

    This function is used in two contexts:
    1. `split_streams_with_oracle_events`: Split events by chunk boundaries and convert
       absolute frame positions to chunk-local positions (pos_shift = chunk_start).
    2. `align_oracle_events_to_delayed_text_timeline`: Filter out-of-range events after
       applying delay offset (pos_shift = 0, positions already adjusted).

    Args:
        events: Source OracleEvents object.
        keep_indices: 1-D array of integer indices specifying which events to keep.
            Each element must satisfy 0 <= i < len(events.frame_pos).
            The j-th event in the result corresponds to events' original event at
            index keep_indices[j].
        pos_shift: Integer offset subtracted from each selected event's frame_pos.
            Used to convert absolute frame indices to chunk-local coordinates when
            splitting streams (e.g., pos_shift = chunk_start_frame).
            Set to 0 when only filtering is needed.

    Returns:
        A new OracleEvents instance with fields restricted according to keep_indices,
        and frame_pos adjusted by pos_shift.
    """
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    frame_pos = events.frame_pos[keep_indices] - int(pos_shift)
    ratio = events.ratio[keep_indices]
    skip_forbid = events.skip_forbid[keep_indices]

    pred_values, pred_offsets = _subset_packed_ragged(
        events.pred_values, events.pred_offsets, keep_indices
    )
    hint_values, hint_offsets = _subset_packed_ragged(
        events.hint_values, events.hint_offsets, keep_indices
    )
    return OracleEvents(
        frame_pos=frame_pos.astype(np.int32, copy=False),
        ratio=ratio.astype(np.float32, copy=False),
        skip_forbid=skip_forbid.astype(np.int8, copy=False),
        pred_values=pred_values,
        pred_offsets=pred_offsets,
        hint_values=hint_values,
        hint_offsets=hint_offsets,
    )


def align_oracle_events_to_delayed_text_timeline(
    *,
    list_of_streams: list[np.ndarray],
    list_of_oracle_events: list[OracleEvents],
    text_delay: int,
) -> list[OracleEvents]:
    """
    Align oracle events (pre-delay positions) to the delayed/padded streams timeline.

    offset = 1 (prepended initial token) + text_delay (delay for text stream)

    Note:
      The oracle start token (oracle_start_id) is part of the oracle content stream and is
      unrelated to the initial token prepended by `delay_and_pad_streams()`. The +1 offset is
      required to keep oracle positions aligned with the model input timeline.
    """
    offset = 1 + int(text_delay)
    out: list[OracleEvents] = []

    for streams, ev in zip(list_of_streams, list_of_oracle_events, strict=True):
        T_new = int(streams.shape[1])

        pos = ev.frame_pos.astype(np.int64) + offset
        keep = np.nonzero((0 <= pos) & (pos < T_new))[0]
        ev2 = _subset_oracle_events(
            OracleEvents(
                frame_pos=pos.astype(np.int32, copy=False),
                ratio=ev.ratio,
                skip_forbid=ev.skip_forbid,
                pred_values=ev.pred_values,
                pred_offsets=ev.pred_offsets,
                hint_values=ev.hint_values,
                hint_offsets=ev.hint_offsets,
            ),
            keep_indices=keep,
            pos_shift=0,
        )
        out.append(ev2)

    return out


def split_streams(
    *,
    list_of_streams: list[np.ndarray],
    max_length: int,
) -> list[np.ndarray]:
    """
    Split the streams into chunks of max_length.
    """
    list_of_chunked_streams: list[np.ndarray] = []

    for _idx, streams in enumerate(list_of_streams):
        num_streams, num_frames = streams.shape
        num_splits = -(-num_frames // max_length)
        list_of_chunked_streams.extend(np.array_split(streams, num_splits, axis=1))

    return list_of_chunked_streams


def split_streams_with_oracle_events(
    *,
    list_of_streams: list[np.ndarray],
    max_length: int,
    list_of_oracle_events: list[OracleEvents],
) -> tuple[list[np.ndarray], list[OracleEvents]]:
    """
    Split streams into chunks and split oracle events accordingly.

    Uses the same chunking as `np.array_split` used for streams, and then assigns
    events into each chunk by [start, end) boundaries and shifts positions to chunk-local.
    """
    chunked_streams: list[np.ndarray] = []
    chunked_events: list[OracleEvents] = []

    for streams, ev in zip(list_of_streams, list_of_oracle_events, strict=True):
        num_frames = int(streams.shape[1])
        num_splits = -(-num_frames // int(max_length))

        stream_chunks = np.array_split(streams, num_splits, axis=1)
        # derive boundaries from produced chunk sizes (matches np.array_split behavior)
        sizes = [int(c.shape[1]) for c in stream_chunks]
        starts = np.cumsum([0] + sizes[:-1]).astype(np.int64)
        ends = starts + np.asarray(sizes, dtype=np.int64)

        # sort events by position once (optional but makes behavior stable)
        order = np.argsort(ev.frame_pos.astype(np.int64), kind="stable")
        ev_sorted = _subset_oracle_events(ev, order, pos_shift=0)

        for c, s, e in zip(stream_chunks, starts.tolist(), ends.tolist(), strict=True):
            pos = ev_sorted.frame_pos.astype(np.int64)
            keep = np.nonzero((s <= pos) & (pos < e))[0]
            ev_chunk = _subset_oracle_events(ev_sorted, keep_indices=keep, pos_shift=int(s))
            chunked_streams.append(c)
            chunked_events.append(ev_chunk)

    return chunked_streams, chunked_events


def filter_out_short_streams(
    *, list_of_streams: list[np.ndarray], min_length: int
) -> list[np.ndarray]:
    """
    Filter out streams that are shorter than min_length.
    """
    return [s for s in list_of_streams if int(s.shape[1]) >= int(min_length)]


def filter_out_short_streams_with_oracle_events(
    *,
    list_of_streams: list[np.ndarray],
    min_length: int,
    list_of_oracle_events: list[OracleEvents],
) -> tuple[list[np.ndarray], list[OracleEvents]]:
    filtered_streams: list[np.ndarray] = []
    filtered_events: list[OracleEvents] = []
    for s, ev in zip(list_of_streams, list_of_oracle_events, strict=True):
        if int(s.shape[1]) >= int(min_length):
            filtered_streams.append(s)
            filtered_events.append(ev)
    return filtered_streams, filtered_events


def make_streams_labels(
    *,
    list_of_streams: list[np.ndarray],
    initial_token_ids: list[int],
    zero_token_id: int,
) -> list[np.ndarray]:
    """
    Make the labels for the streams.
    """
    list_of_labels = []
    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        label = streams.copy()
        for i in range(num_streams):
            # zero out initial tokens
            label[i] = np.where(
                label[i] < initial_token_ids[i],
                label[i],
                zero_token_id,
            )
        list_of_labels.append(label)
    return list_of_labels


def preprocess_function(
    batched_examples: dict[str, list[Any]],
    *,
    speakers: list[str],
    max_length: int | None,
    min_length: int | None,
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
    zero_token_id: int,
    oracle_column_suffix: str | None = "_oracle",
) -> dict[str, list[Any]]:
    # 0. make oracle tokens
    list_of_oracle_events: list[OracleEvents] | None = None

    if oracle_column_suffix is not None:

        def _need_keys_for(sp: str) -> list[str]:
            base = f"{sp}{oracle_column_suffix}"
            return [
                f"{base}_event_frame_pos",
                f"{base}_event_ratio",
                f"{base}_event_skip_forbid",
                f"{base}_pred_values",
                f"{base}_pred_offsets",
                f"{base}_hint_values",
                f"{base}_hint_offsets",
            ]

        missing: list[str] = []
        for sp in speakers:
            for k in _need_keys_for(sp):
                if k not in batched_examples:
                    missing.append(k)

        if missing:
            raise KeyError(
                "Oracle is enabled but event-level oracle columns are missing. "
                f"Missing keys (sample): {missing[:8]}"
            )

        # Prefer events if both exist
        list_of_oracle_events = []
        for sp in speakers:
            base = f"{sp}{oracle_column_suffix}"
            for pos, ratio, skip_forbid, pv, po, hv, ho in zip(
                batched_examples[f"{base}_event_frame_pos"],
                batched_examples[f"{base}_event_ratio"],
                batched_examples[f"{base}_event_skip_forbid"],
                batched_examples[f"{base}_pred_values"],
                batched_examples[f"{base}_pred_offsets"],
                batched_examples[f"{base}_hint_values"],
                batched_examples[f"{base}_hint_offsets"],
                strict=True,
            ):
                ev = OracleEvents(
                    frame_pos=np.asarray(pos, dtype=np.int32),
                    ratio=np.asarray(ratio, dtype=np.float32),
                    skip_forbid=np.asarray(skip_forbid, dtype=np.int8),
                    pred_values=np.asarray(pv, dtype=np.int32),
                    pred_offsets=np.asarray(po, dtype=np.int32),
                    hint_values=np.asarray(hv, dtype=np.int32),
                    hint_offsets=np.asarray(ho, dtype=np.int32),
                )
                list_of_oracle_events.append(ev)

    # 1. make main speaker streams
    list_of_streams = main_speaker_streams(
        batched_examples=batched_examples,
        speakers=speakers,
    )

    if list_of_oracle_events is not None:
        assert len(list_of_streams) == len(list_of_oracle_events), (
            f"Expected {len(list_of_streams)} == {len(list_of_oracle_events)}"
        )

    # 2. delay and pad streams
    list_of_streams = delay_and_pad_streams(
        list_of_streams=list_of_streams,
        delays=delays,
        initial_token_ids=initial_token_ids,
        padding_token_ids=padding_token_ids,
    )

    # 2.2 align oracle events to delayed text timeline
    if list_of_oracle_events is not None:
        list_of_oracle_events = align_oracle_events_to_delayed_text_timeline(
            list_of_streams=list_of_streams,
            list_of_oracle_events=list_of_oracle_events,
            text_delay=delays[0],
        )

    # 3. split streams by max length
    if max_length is not None:
        if list_of_oracle_events is not None:
            list_of_streams, list_of_oracle_events = split_streams_with_oracle_events(
                list_of_streams=list_of_streams,
                max_length=max_length,
                list_of_oracle_events=list_of_oracle_events,
            )
        else:
            list_of_streams = split_streams(list_of_streams=list_of_streams, max_length=max_length)

    # 4. filter out short streams
    if min_length is not None:
        if list_of_oracle_events is not None:
            list_of_streams, list_of_oracle_events = filter_out_short_streams_with_oracle_events(
                list_of_streams=list_of_streams,
                min_length=min_length,
                list_of_oracle_events=list_of_oracle_events,
            )
        else:
            list_of_streams = filter_out_short_streams(
                list_of_streams=list_of_streams,
                min_length=min_length,
            )

    # 5. make labels
    list_of_labels = make_streams_labels(
        list_of_streams=list_of_streams,
        initial_token_ids=initial_token_ids,
        zero_token_id=zero_token_id,
    )

    list_of_num_streams = [streams.shape[0] for streams in list_of_streams]
    list_of_num_frames = [streams.shape[1] for streams in list_of_streams]

    features = {
        "streams": list_of_streams,
        "labels": list_of_labels,
        "num_streams": list_of_num_streams,
        "num_frames": list_of_num_frames,
    }

    if list_of_oracle_events is not None:
        features["oracle_event_frame_pos"] = [e.frame_pos for e in list_of_oracle_events]
        features["oracle_event_ratio"] = [e.ratio for e in list_of_oracle_events]
        features["oracle_event_skip_forbid"] = [e.skip_forbid for e in list_of_oracle_events]
        features["oracle_pred_values"] = [e.pred_values for e in list_of_oracle_events]
        features["oracle_pred_offsets"] = [e.pred_offsets for e in list_of_oracle_events]
        features["oracle_hint_values"] = [e.hint_values for e in list_of_oracle_events]
        features["oracle_hint_offsets"] = [e.hint_offsets for e in list_of_oracle_events]

    return features


def undelay_tokens(
    tokens: np.ndarray | torch.LongTensor, delays: list[int]
) -> np.ndarray | torch.LongTensor | None:
    """
    Restore the undelayed tokens from the delayed tokens

    Args:
        tokens (np.ndarray | torch.LongTensor): shape is (B, K, Td).
        delays (list[int]): delays for each of K codebooks

    Returns:
        undelayed_tokens (np.ndarray | torch.LongTensor): shape is (B, K, T).
            T is not necessarily equal to Td because of the delays.
    """
    max_delay = max(delays)
    B, K, Td = tokens.shape

    if Td < max_delay + 1:  # too short
        return None

    assert K == len(delays), f"Expected K == {len(delays)}, but got {K}"

    T = Td - max_delay
    if isinstance(tokens, np.ndarray):
        undelayed_tokens = np.zeros((B, K, T), dtype=tokens.dtype)
    else:
        undelayed_tokens = torch.zeros((B, K, T), dtype=tokens.dtype, device=tokens.device)
    for cb_index, delay in enumerate(delays):
        undelayed_tokens[:, cb_index] = tokens[:, cb_index, delay : delay + T]

    return undelayed_tokens


@dataclass
class Batch:
    example_ids: list[int] | None
    input_ids: torch.LongTensor
    text_attention_mask: torch.LongTensor
    labels: torch.LongTensor
    oracle_tokens: torch.LongTensor | None = None

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            example_ids=self.example_ids,
            input_ids=self.input_ids.to(device),
            text_attention_mask=self.text_attention_mask.to(device),
            labels=self.labels.to(device),
            oracle_tokens=self.oracle_tokens.to(device) if self.oracle_tokens is not None else None,
        )


class DataCollator:
    def __init__(
        self,
        zero_token_id: int,
        *,
        oracle_pad_id: int | None = None,
        oracle_shift_prob: float = 0.0,
        oracle_right_shift_range: tuple[int, int] = (1, 10),
        oracle_left_shift_range: tuple[int, int] = (1, 15),
        # events -> frame reconstruction params (skip/jitter can be enabled later)
        oracle_start_id: int = 32000,
        oracle_skip_prob_min: float = 0.0,
        oracle_skip_prob_max: float = 0.0,
        oracle_max_time_jitter_frames: int = 0,
        oracle_hint_only: bool = False,
        # --- hint -> prediction curriculum schedule ---
        oracle_hint_only_warmup_start: int | None = None,
        oracle_hint_only_warmup_end: int | None = None,
    ):
        self.zero_token_id = zero_token_id

        # Padding used for oracle tokens when batching. For Turtle-compatibility, use text padding (usually 3).
        self.oracle_pad_id = zero_token_id if oracle_pad_id is None else oracle_pad_id

        # Oracle shift augmentation parameters
        self.oracle_shift_prob = oracle_shift_prob
        self.oracle_right_shift_range = oracle_right_shift_range
        self.oracle_left_shift_range = oracle_left_shift_range

        # events params
        self.oracle_start_id = int(oracle_start_id)
        self.oracle_skip_prob_min = float(oracle_skip_prob_min)
        self.oracle_skip_prob_max = float(oracle_skip_prob_max)
        self.oracle_max_time_jitter_frames = int(oracle_max_time_jitter_frames)
        self.oracle_hint_only = bool(oracle_hint_only)

        # Curriculum schedule: transition from hint_only=True to hint_only=False
        # between [warmup_start, warmup_end] steps.
        # At step <= warmup_start: hint_only probability = 1.0 (always hint only)
        # At step >= warmup_end:   hint_only probability = 0.0 (normal hint/pred selection)
        # In between: linear interpolation
        self.oracle_hint_only_warmup_start = oracle_hint_only_warmup_start
        self.oracle_hint_only_warmup_end = oracle_hint_only_warmup_end

        # Current training step, updated externally from training loop
        self._current_step: int = 0

        # Per-worker RNG (initialized lazily). This avoids identical randomness across DataLoader workers.
        self._rng: np.random.Generator | None = None
        self._rng_worker_id: int | None = None

    def _get_rng(self) -> np.random.Generator:
        worker_info = get_worker_info()
        worker_id = -1 if worker_info is None else worker_info.id
        if self._rng is None or self._rng_worker_id != worker_id:
            # torch.initial_seed() is different per DataLoader worker, so this gives worker-safe randomness.
            seed = int(torch.initial_seed() % (2**32))
            self._rng = np.random.default_rng(seed)
            self._rng_worker_id = worker_id
        return self._rng

    def set_current_step(self, step: int) -> None:
        """Called from the training loop to update the current step for curriculum scheduling."""
        self._current_step = step

    def _get_effective_hint_only(self) -> bool:
        """
        Determine whether to use hint-only mode for the current example,
        based on curriculum schedule.

        If oracle_hint_only is False (from args), always returns False.
        If warmup_start/end are None, returns the static oracle_hint_only value.
        Otherwise, linearly transitions from hint_only=True to hint_only=False
        between [warmup_start, warmup_end] steps.
        """
        if not self.oracle_hint_only:
            return False

        if self.oracle_hint_only_warmup_start is None or self.oracle_hint_only_warmup_end is None:
            return self.oracle_hint_only

        step = self._current_step
        start = self.oracle_hint_only_warmup_start
        end = self.oracle_hint_only_warmup_end

        if step <= start:
            return True  # pure hint only
        if step >= end:
            return False  # fully transitioned to normal mode

        # Linear interpolation: probability of using hint_only decreases
        hint_only_prob = 1.0 - (step - start) / (end - start)

        rng = self._get_rng()
        return bool(rng.random() < hint_only_prob)

    def _maybe_shift_oracle_1d(self, o_1d: torch.LongTensor) -> torch.LongTensor:
        """
        Apply Turtle-style left/right shift augmentation to a 1D oracle token stream.

        Args:
            o_1d: shape (T,), oracle token ids already aligned to the model timeline.

        Returns:
            Shifted oracle stream of the same shape, padded with `oracle_pad_id`.
        """
        if self.oracle_shift_prob <= 0.0:
            return o_1d

        rng = self._get_rng()
        if rng.random() >= self.oracle_shift_prob:
            return o_1d

        T = int(o_1d.numel())
        if T <= 1:
            return o_1d

        do_right = bool(rng.integers(0, 2))  # choose left/right randomly
        if do_right:
            lo, hi = self.oracle_right_shift_range
        else:
            lo, hi = self.oracle_left_shift_range

        # Clamp the shift amount to [1, T-1]
        s = int(rng.integers(lo, hi + 1))
        s = max(1, min(s, T - 1))

        pad = int(self.oracle_pad_id)
        shifted = torch.full((T,), pad, dtype=o_1d.dtype, device=o_1d.device)

        if do_right:
            # [pad]*s + original[:-s]
            shifted[s:] = o_1d[:-s]
        else:
            # original[s:] + [pad]*s
            shifted[: T - s] = o_1d[s:]

        return shifted

    def _events_to_oracle_1d(self, e: dict[str, Any], t: int) -> torch.LongTensor:
        """
        Build a frame-level oracle stream from event-level fields for a single example.
        (Randomness: optional skip/jitter; shift is applied outside.)
        """
        rng = self._get_rng()

        pos = np.asarray(e["oracle_event_frame_pos"], dtype=np.int64)
        ratio = np.asarray(e["oracle_event_ratio"], dtype=np.float32)
        skip_forbid = np.asarray(e["oracle_event_skip_forbid"], dtype=np.int8)

        pred_values = np.asarray(e["oracle_pred_values"], dtype=np.int32)
        pred_offsets = np.asarray(e["oracle_pred_offsets"], dtype=np.int32)
        hint_values = np.asarray(e["oracle_hint_values"], dtype=np.int32)
        hint_offsets = np.asarray(e["oracle_hint_offsets"], dtype=np.int32)

        out = torch.full((t,), int(self.oracle_pad_id), dtype=torch.long)

        # sample per-example skip probability (Turtle-style)
        skip_prob = 0.0
        if self.oracle_skip_prob_max > 0.0 or self.oracle_skip_prob_min > 0.0:
            lo = min(self.oracle_skip_prob_min, self.oracle_skip_prob_max)
            hi = max(self.oracle_skip_prob_min, self.oracle_skip_prob_max)
            skip_prob = float(rng.uniform(lo, hi))

        # stable order
        order = np.argsort(pos, kind="stable")
        # Sample once per example so warmup does not vary across events within the same oracle stream.
        effective_hint_only = self._get_effective_hint_only()

        for i in order.tolist():
            frame_pos = int(pos[i])
            if frame_pos < 0 or frame_pos >= t:
                continue

            if int(skip_forbid[i]) == 0 and skip_prob > 0.0 and float(rng.random()) < skip_prob:
                continue

            # jitter in frames (approx, since we store integer frame positions)
            if self.oracle_max_time_jitter_frames > 0:
                frame_pos = int(
                    frame_pos + int(rng.integers(0, self.oracle_max_time_jitter_frames + 1))
                )
                if frame_pos < 0 or frame_pos >= t:
                    continue

            hint_tok = self._get_event_tokens(hint_values, hint_offsets, i)
            if effective_hint_only:
                # Use hint only, with no fallback to prediction
                if hint_tok.size == 0:
                    continue
                seq = hint_tok
            else:
                # Use hint when it is complete (ratio >= 1.0); otherwise fall back to prediction
                pred_tok = self._get_event_tokens(pred_values, pred_offsets, i)
                use_hint = (float(ratio[i]) >= 1.0) and (hint_tok.size > 0)
                seq = hint_tok if use_hint else pred_tok
                if seq.size == 0:
                    continue

            out[frame_pos] = int(self.oracle_start_id)
            write_pos = frame_pos + 1
            for tok in seq.tolist():
                if write_pos >= t:
                    break
                out[write_pos] = int(tok)
                write_pos += 1

        return out

    def _get_event_tokens(self, values: np.ndarray, offsets: np.ndarray, i: int) -> np.ndarray:
        s = int(offsets[i])
        e = int(offsets[i + 1])
        if e <= s:
            return np.empty((0,), dtype=np.int32)
        return values[s:e].astype(np.int32, copy=False)

    def __call__(self, examples: list[dict[str, Any]]) -> Batch:
        """
        Collate the examples into a batch.
        Args:
            examples (list[dict[str, Any]]): list of examples
                ```
                [
                    {
                        "streams": np.ndarray,  # shape: (num_streams, num_frames)
                        "labels": np.ndarray,
                        "num_streams": int,
                        "num_frames": int,
                        # Optional oracle event fields (present if use_oracle=True):
                        "oracle_event_frame_pos": np.ndarray,  # (E,) int32
                        "oracle_event_ratio": np.ndarray,      # (E,) float32
                        "oracle_event_skip_forbid": np.ndarray,  # (E,) int8
                        "oracle_pred_values": np.ndarray,      # (sum_pred,) int32
                        "oracle_pred_offsets": np.ndarray,     # (E+1,) int32
                        "oracle_hint_values": np.ndarray,      # (sum_hint,) int32
                        "oracle_hint_offsets": np.ndarray,     # (E+1,) int32
                    },
                    ...
                ]
                ```
        Returns:
            batch (Batch): batch of examples
        """
        # pad with zero tokens
        batch_size = len(examples)
        num_streams = examples[0]["num_streams"]
        max_frames = max([e["num_frames"] for e in examples])
        zero_ids = torch.full(
            (batch_size, num_streams, max_frames), fill_value=self.zero_token_id, dtype=torch.long
        )

        input_ids = zero_ids.clone()
        for i, e in enumerate(examples):
            input_ids[i, :, : e["num_frames"]] = torch.tensor(e["streams"], dtype=torch.long)

        text_attention_mask = zero_ids[:, 0, :].clone()  # (batch_size, max_frames)
        for i, e in enumerate(examples):
            text_attention_mask[i, : e["num_frames"]] = 1

        labels = zero_ids.clone()
        for i, e in enumerate(examples):
            labels[i, :, : e["num_frames"]] = torch.tensor(e["labels"], dtype=torch.long)

        # --- oracle_tokens (optional) ---
        has_oracle_events = "oracle_event_frame_pos" in examples[0]

        oracle_ids = None
        if has_oracle_events:
            oracle_ids = torch.full(
                (batch_size, 1, max_frames),
                fill_value=int(self.oracle_pad_id),
                dtype=torch.long,
            )

            for i, e in enumerate(examples):
                t = int(e["num_frames"])
                # events -> frame
                o0 = self._events_to_oracle_1d(e, t=t)
                o0 = self._maybe_shift_oracle_1d(o0)
                oracle_ids[i, 0, : o0.numel()] = o0

        example_ids = None
        if "example_id" in examples[0]:
            example_ids = [e["example_id"] for e in examples]

        batch = Batch(
            example_ids=example_ids,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
            oracle_tokens=oracle_ids,
        )
        return batch
