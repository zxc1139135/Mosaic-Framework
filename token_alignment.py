"""Utilities for tokenizer-independent alignment via character spans."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Span = Tuple[int, int]


def _valid_spans(offsets: Sequence[Sequence[int]]) -> List[Span]:
    spans: List[Span] = []
    for item in offsets:
        if item is None or len(item) != 2:
            continue
        start, end = int(item[0]), int(item[1])
        if end > start:
            spans.append((start, end))
    return spans


def build_char_signal(text_length: int, offsets: Sequence[Sequence[int]], values: Sequence[float]):
    text_length = max(int(text_length), 0)
    signal_sum = np.zeros(text_length, dtype=np.float32)
    signal_count = np.zeros(text_length, dtype=np.float32)
    for raw_span, value in zip(offsets, values):
        if raw_span is None or len(raw_span) != 2:
            continue
        start, end = int(raw_span[0]), int(raw_span[1])
        start = max(0, min(start, text_length))
        end = max(0, min(end, text_length))
        if end <= start:
            continue
        signal_sum[start:end] += float(value)
        signal_count[start:end] += 1.0
    return signal_sum, signal_count


def token_aligned_values_from_char_spans(
    text_length: int,
    source_offsets: Sequence[Sequence[int]],
    source_values: Sequence[float],
    target_offsets: Sequence[Sequence[int]],
    default_value: float = 0.0,
) -> np.ndarray:
    target_spans = _valid_spans(target_offsets)
    aligned = np.full(len(target_spans), float(default_value), dtype=np.float32)
    if text_length <= 0 or len(target_spans) == 0:
        return aligned

    signal_sum, signal_count = build_char_signal(text_length, source_offsets, source_values)
    avg_signal = np.divide(
        signal_sum,
        np.maximum(signal_count, 1.0),
        out=np.zeros_like(signal_sum),
        where=(signal_count > 0),
    )
    for idx, (start, end) in enumerate(target_spans):
        window_count = signal_count[start:end]
        if window_count.size == 0:
            continue
        covered = window_count > 0
        if covered.any():
            aligned[idx] = float(avg_signal[start:end][covered].mean())
    return aligned


def mean_abs_discrepancy_over_text(
    text_length: int,
    offsets_a: Sequence[Sequence[int]],
    values_a: Sequence[float],
    offsets_b: Sequence[Sequence[int]],
    values_b: Sequence[float],
) -> float:
    if text_length <= 0:
        return 0.0
    sum_a, cnt_a = build_char_signal(text_length, offsets_a, values_a)
    sum_b, cnt_b = build_char_signal(text_length, offsets_b, values_b)
    avg_a = np.divide(sum_a, np.maximum(cnt_a, 1.0), out=np.zeros_like(sum_a), where=(cnt_a > 0))
    avg_b = np.divide(sum_b, np.maximum(cnt_b, 1.0), out=np.zeros_like(sum_b), where=(cnt_b > 0))
    mask = (cnt_a > 0) & (cnt_b > 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(avg_a[mask] - avg_b[mask])))
