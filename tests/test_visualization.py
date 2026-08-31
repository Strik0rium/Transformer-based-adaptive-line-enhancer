import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from transformer_based_adaptive_line_enhancer import (
    generate_noisy_signal,
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
