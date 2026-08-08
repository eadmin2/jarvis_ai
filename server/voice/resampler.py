"""Deterministic streaming PCM resampling utilities."""

from __future__ import annotations

import math

import numpy as np

try:
    from scipy import signal as scipy_signal
except ImportError:  # NumPy interpolation remains a deterministic local fallback.
    scipy_signal = None


class StreamingPCMResampler:
    """Stateful high-quality PCM16 resampler with exact stream continuity.

    When SciPy is installed (the supported server environment), polyphase FIR
    resampling filters frequencies above the target Nyquist limit. A short
    unsettled tail is retained between HTTP chunks, which makes streamed output
    byte-identical to one-shot output without filter-edge clicks. NumPy linear
    interpolation remains a deterministic fallback for lightweight clients.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._samples = np.empty(0, dtype=np.int16)
        self._pending_byte = b""
        self._output_index = 0
        self._flushed = False
        common = math.gcd(source_rate, target_rate)
        self._up = target_rate // common
        self._down = source_rate // common

    def process(self, pcm: bytes) -> bytes:
        if self._flushed:
            raise RuntimeError("cannot process PCM after flush")
        if not pcm:
            return b""
        data = self._pending_byte + pcm
        even_length = len(data) - (len(data) % 2)
        self._pending_byte = data[even_length:]
        if even_length:
            incoming = np.frombuffer(data[:even_length], dtype="<i2")
            self._samples = np.concatenate((self._samples, incoming))
        return self._produce(final=False)

    def flush(self) -> bytes:
        if self._flushed:
            return b""
        self._flushed = True
        # An incomplete int16 sample is invalid provider output; dropping it is
        # safer than inventing a high-order byte that can create a loud spike.
        self._pending_byte = b""
        return self._produce(final=True)

    def _produce(self, final: bool) -> bytes:
        sample_count = len(self._samples)
        if sample_count == 0:
            return b""

        if scipy_signal is not None and self.source_rate != self.target_rate:
            filtered = scipy_signal.resample_poly(
                self._samples.astype(np.float32), self._up, self._down,
            )
            # SciPy's default FIR uses roughly 20 * max(up, down) taps. Keeping
            # 64 output samples un-emitted is comfortably beyond its trailing
            # boundary dependency for the supported voice sample-rate ratios.
            available = len(filtered) if final else max(0, len(filtered) - 64)
            if available <= self._output_index:
                return b""
            output = filtered[self._output_index : available]
            self._output_index = available
            return np.clip(output, -32768, 32767).astype("<i2").tobytes()

        if final:
            target_count = round(sample_count * self.target_rate / self.source_rate)
        else:
            target_count = None

        output: list[int] = []
        while target_count is None or self._output_index < target_count:
            numerator = self._output_index * self.source_rate
            left = numerator // self.target_rate
            fraction = numerator % self.target_rate
            if left >= sample_count:
                break
            if fraction and left + 1 >= sample_count and not final:
                break

            left_value = int(self._samples[left])
            if fraction:
                # At end-of-stream, hold the final input sample for the last
                # fractional output position. This gives an exact rounded
                # duration when a caller requests upsampling.
                right_value = (
                    int(self._samples[left + 1])
                    if left + 1 < sample_count
                    else left_value
                )
                value = (
                    left_value * (self.target_rate - fraction)
                    + right_value * fraction
                ) / self.target_rate
                output.append(int(round(value)))
            else:
                output.append(left_value)
            self._output_index += 1

        if not output:
            return b""
        return np.asarray(output, dtype="<i2").tobytes()


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """One-shot helper using the same deterministic streaming implementation."""
    if source_rate == target_rate:
        return pcm[: len(pcm) - (len(pcm) % 2)]
    resampler = StreamingPCMResampler(source_rate, target_rate)
    return resampler.process(pcm) + resampler.flush()
