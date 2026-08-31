"""Visualization helpers for generated signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
