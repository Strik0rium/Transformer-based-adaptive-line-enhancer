"""Visualization helpers for generated signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .signals import NoisySignal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_noisy_signal_components(
    sample: NoisySignal,
    *,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Plot the clean, noise, and noisy signals in separate subplots.

    The function does not call ``show`` or save a file. It returns the figure
    and axes so callers can further customize, display, or save the plot.
    """
    import matplotlib.pyplot as plt

    figure, axes_array = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=figsize,
        constrained_layout=True,
    )
    axes = (axes_array[0], axes_array[1], axes_array[2])
    series = (
        (sample.clean_signal, "Clean signal", "C0"),
        (sample.noise, "Noise", "C1"),
        (sample.noisy_signal, "Noisy signal", "C2"),
    )

    for axis, (values, title, color) in zip(axes, series, strict=True):
        axis.plot(sample.time_s, values, color=color, linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("Amplitude")
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    return figure, axes


def plot_line_enhancement_spectrogram(
    sample: NoisySignal,
    enhanced_signal: ArrayLike,
    *,
    sample_rate_hz: float,
    start_sample: int = 0,
    n_fft: int = 256,
    hop_length: int | None = None,
    max_frequency_hz: float | None = None,
    dynamic_range_db: float = 80.0,
    figsize: tuple[float, float] = (10.0, 9.0),
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Compare clean, noisy, and enhanced signals using shared-scale spectrograms.

    ``start_sample`` can exclude an adaptive filter's warm-up region. All three
    panels use the same absolute dB color limits so visual differences remain
    comparable rather than being hidden by per-panel normalization.
    """
    import matplotlib.pyplot as plt

    enhanced = np.asarray(enhanced_signal, dtype=np.float64)
    if enhanced.ndim != 1 or enhanced.shape != sample.noisy_signal.shape:
        raise ValueError("enhanced_signal must be one-dimensional and match the sample")
    if not np.all(np.isfinite(enhanced)):
        raise ValueError("enhanced_signal must contain only finite values")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive finite number")
    _validate_nonnegative_integer(start_sample, "start_sample")
    _validate_positive_integer(n_fft, "n_fft")
    if n_fft < 4:
        raise ValueError("n_fft must be at least 4")
    if start_sample >= sample.noisy_signal.size:
        raise ValueError("start_sample must be smaller than the signal length")

    if hop_length is None:
        hop_length = n_fft // 4
    _validate_positive_integer(hop_length, "hop_length")
    if hop_length > n_fft:
        raise ValueError("hop_length must not exceed n_fft")
    if sample.noisy_signal.size - start_sample < n_fft:
        raise ValueError("the selected signal region must contain at least n_fft samples")
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0:
        raise ValueError("dynamic_range_db must be a positive finite number")

    nyquist_hz = sample_rate_hz / 2.0
    if max_frequency_hz is None:
        max_frequency_hz = nyquist_hz
    if (
        not np.isfinite(max_frequency_hz)
        or max_frequency_hz <= 0
        or max_frequency_hz > nyquist_hz
    ):
        raise ValueError("max_frequency_hz must lie between 0 and Nyquist")

    offset_s = start_sample / sample_rate_hz
    signals = (
        sample.clean_signal[start_sample:],
        sample.noisy_signal[start_sample:],
        enhanced[start_sample:],
    )
    spectrograms = tuple(
        _spectrogram_db(
            values,
            sample_rate_hz=sample_rate_hz,
            n_fft=n_fft,
            hop_length=hop_length,
            offset_s=offset_s,
        )
        for values in signals
    )

    common_max_db = max(float(np.max(power_db)) for _, _, power_db in spectrograms)
    common_min_db = common_max_db - dynamic_range_db
    figure, axes_array = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=figsize,
        constrained_layout=True,
    )
    axes = (axes_array[0], axes_array[1], axes_array[2])
    titles = ("Clean reference", "Noisy input", "Enhanced output")
    image = None

    for axis, title, (times_s, frequencies_hz, power_db) in zip(
        axes, titles, spectrograms, strict=True
    ):
        frequency_mask = frequencies_hz <= max_frequency_hz
        image = axis.pcolormesh(
            times_s,
            frequencies_hz[frequency_mask],
            power_db[frequency_mask],
            shading="nearest",
            cmap="magma",
            vmin=common_min_db,
            vmax=common_max_db,
        )
        axis.set_title(title)
        axis.set_ylabel("Frequency (Hz)")
        axis.set_ylim(0.0, max_frequency_hz)

    axes[-1].set_xlabel("Time (s)")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_label("Power (dB)")
    return figure, axes


def _spectrogram_db(
    signal: NDArray[np.float64],
    *,
    sample_rate_hz: float,
    n_fft: int,
    hop_length: int,
    offset_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    window = np.hanning(n_fft)
    window_energy = float(window @ window)
    frame_starts = np.arange(0, signal.size - n_fft + 1, hop_length)
    power = np.empty((n_fft // 2 + 1, frame_starts.size), dtype=np.float64)

    for frame_index, frame_start in enumerate(frame_starts):
        frame = signal[frame_start : frame_start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        power[:, frame_index] = np.square(np.abs(spectrum)) / window_energy

    tiny = np.finfo(np.float64).tiny
    power_db = 10.0 * np.log10(np.maximum(power, tiny))
    frequencies_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    times_s = (frame_starts + n_fft / 2.0) / sample_rate_hz + offset_s
    return times_s, frequencies_hz, power_db


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
