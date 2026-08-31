import numpy as np
import pytest

from transformer_based_adaptive_line_enhancer import generate_noisy_signal


def test_generation_is_reproducible_and_components_sum() -> None:
    first = generate_noisy_signal(seed=7)
    second = generate_noisy_signal(seed=7)

    np.testing.assert_array_equal(first.noisy_signal, second.noisy_signal)
    np.testing.assert_allclose(first.noisy_signal, first.clean_signal + first.noise)
    assert first.time_s.shape == (16_000,)


def test_time_and_tone_follow_matlab_formula() -> None:
    result = generate_noisy_signal(
        frequencies_hz=100.0,
        sample_rate_hz=1_000.0,
        num_samples=10,
        seed=1,
    )

    expected_time = np.arange(10) / 1_000.0
    np.testing.assert_allclose(result.time_s, expected_time)
    np.testing.assert_allclose(result.base_signal, np.sin(2 * np.pi * 100 * expected_time))


def test_power_normalized_mode_realizes_requested_snr() -> None:
    result = generate_noisy_signal(snr_db=-17.0, seed=42, scaling="power-normalized")

    assert result.measured_snr_db == pytest.approx(-17.0, abs=1e-12)


def test_legacy_mode_matches_active_matlab_scaling() -> None:
    result = generate_noisy_signal(snr_db=-20.0, seed=42, scaling="legacy")
    noise_power = np.mean(result.noise**2)

    assert result.scale == pytest.approx(np.sqrt(noise_power * 10 ** (-20.0 / 10.0)))
    assert result.measured_snr_db == pytest.approx(-23.0102999566, abs=1e-9)


def test_multiple_tones_are_summed() -> None:
    result = generate_noisy_signal(
        frequencies_hz=[100.0, 180.0],
        amplitudes=[1.0, 0.25],
        phases_rad=[0.0, np.pi / 2],
        num_samples=20,
        seed=3,
    )

    expected = np.sin(2 * np.pi * 100 * result.time_s) + 0.25 * np.sin(
        2 * np.pi * 180 * result.time_s + np.pi / 2
    )
    np.testing.assert_allclose(result.base_signal, expected)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("sample_rate_hz", 0),
        ("num_samples", 0),
        ("noise_std", 0),
        ("frequencies_hz", 500),
        ("frequencies_hz", []),
    ],
)
def test_invalid_parameters_are_rejected(keyword: str, value: object) -> None:
    with pytest.raises(ValueError):
        generate_noisy_signal(**{keyword: value})
