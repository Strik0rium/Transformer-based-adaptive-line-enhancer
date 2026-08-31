"""Delayed adaptive line enhancement with LMS or normalized LMS updates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ALEMode = Literal["nlms", "lms"]


@dataclass(frozen=True)
class ALEResult:
    """Signals and final state produced by an adaptive line enhancer."""

    enhanced_signal: NDArray[np.float64]
    residual: NDArray[np.float64]
    final_weights: NDArray[np.float64]
    adaptation_start: int

    @property
    def valid_slice(self) -> slice:
        """Slice selecting samples for which the adaptive filter ran."""
        return slice(self.adaptation_start, None)


def adaptive_line_enhance(
    signal: ArrayLike,
    *,
    delay: int = 10,
    filter_length: int = 32,
    step_size: float = 0.05,
    mode: ALEMode = "nlms",
    epsilon: float = 1e-8,
) -> ALEResult:
    """Enhance predictable narrowband components in a noisy 1-D signal.

    At sample ``n``, the adaptive FIR filter receives
    ``[x[n-delay], ..., x[n-delay-filter_length+1]]`` and predicts ``x[n]``.
    The prediction is the enhanced output; the prediction error is returned as
    the residual. Samples before the first complete reference vector have zero
    enhanced output and an unchanged residual.

    ``nlms`` is the default because its update size is normalized by reference
    energy. ``lms`` performs the classic unnormalized LMS update.
    """
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal must contain only finite values")
    _validate_positive_integer(delay, "delay")
    _validate_positive_integer(filter_length, "filter_length")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be a positive finite number")
    if mode not in ("nlms", "lms"):
        raise ValueError("mode must be 'nlms' or 'lms'")
    if mode == "nlms" and step_size >= 2:
        raise ValueError("step_size must be less than 2 for NLMS")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")

    adaptation_start = delay + filter_length - 1
    if samples.size <= adaptation_start:
        raise ValueError(
            "signal is too short for the requested delay and filter_length"
        )

    weights = np.zeros(filter_length, dtype=np.float64)
    enhanced = np.zeros_like(samples)
    residual = samples.copy()

    for sample_index in range(adaptation_start, samples.size):
        reference = samples[
            sample_index - delay - filter_length + 1 : sample_index - delay + 1
        ][::-1]
        prediction = float(weights @ reference)
        error = float(samples[sample_index] - prediction)

        enhanced[sample_index] = prediction
        residual[sample_index] = error

        if mode == "nlms":
            effective_step = step_size / (epsilon + float(reference @ reference))
        else:
            effective_step = step_size
        weights += effective_step * error * reference

    return ALEResult(
        enhanced_signal=enhanced,
        residual=residual,
        final_weights=weights.copy(),
        adaptation_start=adaptation_start,
    )


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
