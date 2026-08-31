"""Synthetic noisy tonal signals ported from the reference MATLAB scripts."""

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

ScalingMode = Literal["legacy", "power-normalized"]


@dataclass(frozen=True)
class NoisySignal:
    """All components of one generated noisy signal."""

    time_s: NDArray[np.float64]
    base_signal: NDArray[np.float64]
    clean_signal: NDArray[np.float64]
    noise: NDArray[np.float64]
    noisy_signal: NDArray[np.float64]
    scale: float
    requested_snr_db: float
    measured_snr_db: float


def generate_noisy_signal(
    *,
    frequencies_hz: float | Sequence[float] = 100.0,
    amplitudes: float | Sequence[float] = 1.0,
    phases_rad: float | Sequence[float] = 0.0,
    sample_rate_hz: float = 1_000.0,
    num_samples: int = 16_000,
    snr_db: float = -20.0,
    noise_std: float = 1.0,
    seed: int | None = None,
    scaling: ScalingMode = "legacy",
) -> NoisySignal:
    """Generate Gaussian noise plus one or more sinusoidal spectral lines.

    ``legacy`` reproduces the active scaling in ``data_p.m``,
    ``data_zczs.m`` and ``ha_test_transformer.m``::

        En = mean(n**2)
        a = sqrt(En * 10**(SNR/10))
        r = a * s + n

    Those scripts omit division by signal power, so the measured SNR of a
    unit sine is about 3.01 dB below ``snr_db``. ``power-normalized`` mirrors
    ``ale_signal.m`` and divides by ``mean(s**2)``, making the realized signal
    and noise powers have exactly the requested ratio for the generated draw.

    NumPy's random-number stream is reproducible for a given ``seed`` but is
    not sample-for-sample identical to MATLAB's ``randn`` stream.
    """
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive finite number")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be a positive finite number")
    if scaling not in ("legacy", "power-normalized"):
        raise ValueError("scaling must be 'legacy' or 'power-normalized'")

    frequencies = _as_1d_float_array(frequencies_hz, "frequencies_hz")
    tone_amplitudes = _broadcast_parameter(amplitudes, frequencies.size, "amplitudes")
    phases = _broadcast_parameter(phases_rad, frequencies.size, "phases_rad")

    if np.any(frequencies <= 0) or np.any(frequencies >= sample_rate_hz / 2):
        raise ValueError("frequencies_hz must lie strictly between 0 and Nyquist")

    # MATLAB: t = (0:N1-1) / Fs; s = sum(A*sin(2*pi*f*t + phase))
    time_s = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
    angles = 2.0 * np.pi * frequencies[:, None] * time_s + phases[:, None]
    base_signal = np.sum(tone_amplitudes[:, None] * np.sin(angles), axis=0)
    signal_power = float(np.mean(np.square(base_signal)))
    if signal_power == 0.0:
        raise ValueError("the requested tones produce a zero-power signal")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, num_samples)
    noise_power = float(np.mean(np.square(noise)))
    linear_snr = 10.0 ** (snr_db / 10.0)

    if scaling == "legacy":
        scale = float(np.sqrt(noise_power * linear_snr))
    else:
        scale = float(np.sqrt(noise_power * linear_snr / signal_power))

    clean_signal = scale * base_signal
    noisy_signal = clean_signal + noise
    measured_snr_db = float(
        10.0 * np.log10(np.mean(np.square(clean_signal)) / noise_power)
    )

    return NoisySignal(
        time_s=time_s,
        base_signal=base_signal,
        clean_signal=clean_signal,
        noise=noise,
        noisy_signal=noisy_signal,
        scale=scale,
        requested_snr_db=float(snr_db),
        measured_snr_db=measured_snr_db,
    )


def _as_1d_float_array(value: float | Sequence[float], name: str) -> NDArray[np.float64]:
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain one or more finite numbers")
    return array


def _broadcast_parameter(
    value: float | Sequence[float], size: int, name: str
) -> NDArray[np.float64]:
    array = _as_1d_float_array(value, name)
    if array.size == 1:
        return np.full(size, array.item(), dtype=np.float64)
    if array.size != size:
        raise ValueError(f"{name} must be scalar or match frequencies_hz in length")
    return array
