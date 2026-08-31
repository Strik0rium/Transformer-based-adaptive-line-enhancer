import numpy as np
import pytest

from transformer_based_adaptive_line_enhancer import (
    adaptive_line_enhance,
    generate_noisy_signal,
)


def _snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = estimate - reference
    return float(10 * np.log10(np.mean(reference**2) / np.mean(error**2)))


@pytest.mark.parametrize("snr_db", [-20.0, -10.0])
def test_nlms_ale_enhances_a_tone_in_white_noise(snr_db: float) -> None:
    sample = generate_noisy_signal(
        frequencies_hz=100.0,
        sample_rate_hz=1_000.0,
        num_samples=16_000,
        snr_db=snr_db,
        seed=42,
        scaling="power-normalized",
    )
    result = adaptive_line_enhance(
        sample.noisy_signal,
        delay=10,
        filter_length=32,
        step_size=0.05,
    )
    evaluation = slice(sample.noisy_signal.size // 2, None)
    input_snr = _snr_db(
        sample.clean_signal[evaluation], sample.noisy_signal[evaluation]
    )
    output_snr = _snr_db(
        sample.clean_signal[evaluation], result.enhanced_signal[evaluation]
    )

    assert input_snr == pytest.approx(snr_db, abs=0.15)
    assert output_snr > input_snr + 8.0


def test_lms_update_matches_a_small_hand_calculation() -> None:
    result = adaptive_line_enhance(
        [1.0, 2.0, 3.0, 4.0],
        delay=1,
        filter_length=1,
        step_size=0.1,
        mode="lms",
    )

    np.testing.assert_allclose(result.enhanced_signal, [0.0, 0.0, 0.4, 2.16])
    np.testing.assert_allclose(result.residual, [1.0, 2.0, 2.6, 1.84])
    np.testing.assert_allclose(result.final_weights, [1.272])
    assert result.adaptation_start == 1
    assert result.valid_slice == slice(1, None)


def test_warmup_region_is_explicit() -> None:
    signal = np.arange(10.0)
    result = adaptive_line_enhance(signal, delay=2, filter_length=3)

    assert result.adaptation_start == 4
    np.testing.assert_array_equal(result.enhanced_signal[:4], 0.0)
    np.testing.assert_array_equal(result.residual[:4], signal[:4])


@pytest.mark.parametrize(
    ("signal", "kwargs"),
    [
        ([], {}),
        ([[1.0, 2.0]], {}),
        ([1.0, np.nan, 2.0], {}),
        ([1.0, 2.0], {"delay": 0}),
        ([1.0, 2.0], {"filter_length": 0}),
        ([1.0, 2.0], {"step_size": 0}),
        ([1.0, 2.0, 3.0], {"step_size": 2.0}),
        ([1.0, 2.0], {"epsilon": 0}),
        ([1.0, 2.0], {"mode": "other"}),
        ([1.0, 2.0], {"delay": 1, "filter_length": 2}),
    ],
)
def test_invalid_inputs_are_rejected(signal: object, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        adaptive_line_enhance(signal, **kwargs)
