import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from transformer_based_adaptive_line_enhancer import (
    adaptive_line_enhance,
    generate_noisy_signal,
    plot_line_enhancement_spectrogram,
    plot_noisy_signal_components,
)


def test_plot_noisy_signal_components_draws_three_separate_signals() -> None:
    sample = generate_noisy_signal(num_samples=100, seed=42)
    figure, axes = plot_noisy_signal_components(sample)

    try:
        assert len(axes) == 3
        assert [axis.get_title() for axis in axes] == [
            "Clean signal",
            "Noise",
            "Noisy signal",
        ]
        expected_values = (
            sample.clean_signal,
            sample.noise,
            sample.noisy_signal,
        )
        for axis, expected in zip(axes, expected_values, strict=True):
            assert len(axis.lines) == 1
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), sample.time_s)
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), expected)
        assert axes[-1].get_xlabel() == "Time (s)"
        assert figure.get_figwidth() == 10.0
        assert figure.get_figheight() == 8.0
    finally:
        plt.close(figure)


def test_plot_line_enhancement_spectrogram_uses_shared_color_scale() -> None:
    sample = generate_noisy_signal(
        num_samples=2_000,
        sample_rate_hz=1_000.0,
        snr_db=-10.0,
        seed=42,
        scaling="power-normalized",
    )
    result = adaptive_line_enhance(sample.noisy_signal)
    figure, axes = plot_line_enhancement_spectrogram(
        sample,
        result.enhanced_signal,
        sample_rate_hz=1_000.0,
        start_sample=result.adaptation_start,
        n_fft=128,
        hop_length=32,
        max_frequency_hz=250.0,
    )

    try:
        assert [axis.get_title() for axis in axes] == [
            "Clean reference",
            "Noisy input",
            "Enhanced output",
        ]
        color_limits = [axis.collections[0].get_clim() for axis in axes]
        assert color_limits[0] == color_limits[1] == color_limits[2]
        assert all(axis.get_ylim()[1] <= 250.0 for axis in axes)
        assert axes[-1].get_xlabel() == "Time (s)"
        assert figure.axes[-1].get_ylabel() == "Power (dB)"
    finally:
        plt.close(figure)


def test_spectrogram_rejects_incompatible_enhanced_signal() -> None:
    sample = generate_noisy_signal(num_samples=512, seed=1)

    with np.testing.assert_raises(ValueError):
        plot_line_enhancement_spectrogram(
            sample,
            np.zeros(511),
            sample_rate_hz=1_000.0,
        )
