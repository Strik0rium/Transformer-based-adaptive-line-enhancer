"""Transformer-based adaptive line enhancer."""

from .ale import ALEResult, adaptive_line_enhance
from .signals import NoisySignal, generate_noisy_signal
from .visualization import plot_noisy_signal_components

__all__ = [
    "ALEResult",
    "NoisySignal",
    "adaptive_line_enhance",
    "generate_noisy_signal",
    "plot_noisy_signal_components",
]
