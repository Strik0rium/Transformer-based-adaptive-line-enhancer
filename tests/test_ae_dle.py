import numpy as np
import pytest
import torch

from transformer_based_adaptive_line_enhancer import (
    AEDLEConfig,
    AutoencoderDLE,
    enhance_with_ae_dle,
    generate_noisy_signal,
    train_ae_dle,
)


def _small_config(**overrides: object) -> AEDLEConfig:
    values: dict[str, object] = {
        "frame_length": 64,
        "frame_hop": 16,
        "prediction_delay": 10,
        "hidden_dim": 32,
        "bottleneck_dim": 8,
        "learning_rate": 2e-3,
        "batch_size": 16,
        "epochs": 12,
        "seed": 7,
    }
    values.update(overrides)
    return AEDLEConfig(**values)


def test_autoencoder_preserves_frame_shape_and_output_range() -> None:
    model = AutoencoderDLE(frame_length=32, hidden_dim=16, bottleneck_dim=4)
    output = model(torch.zeros((3, 32)))

    assert output.shape == (3, 32)
    assert torch.all((output >= 0) & (output <= 1))


def test_autoencoder_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        AutoencoderDLE(frame_length=32, hidden_dim=8, bottleneck_dim=8)


def test_training_is_self_supervised_and_loss_decreases() -> None:
    sample = generate_noisy_signal(
        frequencies_hz=50.0,
        sample_rate_hz=500.0,
        num_samples=2_000,
        snr_db=-5.0,
        seed=10,
        scaling="power-normalized",
    )
    training = train_ae_dle(sample.noisy_signal, config=_small_config())

    assert len(training.loss_history) == 12
    assert training.loss_history[-1] < training.loss_history[0]


def test_ae_dle_improves_single_tone_snr_in_a_locked_scene() -> None:
    sample = generate_noisy_signal(
        frequencies_hz=50.0,
        sample_rate_hz=500.0,
        num_samples=4_000,
        snr_db=-10.0,
        seed=10,
        scaling="power-normalized",
    )
    training = train_ae_dle(sample.noisy_signal, config=_small_config())
    result = enhance_with_ae_dle(sample.noisy_signal, training)
    clean = sample.clean_signal[result.valid_slice]
    noisy = sample.noisy_signal[result.valid_slice]
    enhanced = result.enhanced_signal[result.valid_slice]

    input_snr = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    output_snr = 10 * np.log10(np.mean(clean**2) / np.mean((enhanced - clean) ** 2))

    assert output_snr > input_snr + 8.0


def test_enhancement_output_is_aligned_and_finite() -> None:
    sample = generate_noisy_signal(num_samples=1_000, seed=2)
    config = _small_config(epochs=2)
    training = train_ae_dle(sample.noisy_signal, config=config)
    result = enhance_with_ae_dle(sample.noisy_signal, training)

    assert result.enhanced_signal.shape == sample.noisy_signal.shape
    assert result.residual.shape == sample.noisy_signal.shape
    assert result.valid_start == config.prediction_delay
    assert result.valid_end <= sample.noisy_signal.size
    assert np.all(np.isfinite(result.enhanced_signal))
    np.testing.assert_allclose(
        result.residual[result.valid_slice],
        sample.noisy_signal[result.valid_slice] - result.enhanced_signal[result.valid_slice],
    )


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    signal = np.sin(2 * np.pi * 0.05 * np.arange(512))
    config = _small_config(epochs=2)

    first = train_ae_dle(signal, config=config)
    second = train_ae_dle(signal, config=config)

    assert first.loss_history == second.loss_history
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_length": 0},
        {"frame_hop": 65},
        {"prediction_delay": 0},
        {"hidden_dim": 8, "bottleneck_dim": 8},
        {"learning_rate": 0},
        {"batch_size": 0},
        {"epochs": 0},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _small_config(**overrides)


def test_too_short_signal_is_rejected() -> None:
    with pytest.raises(ValueError):
        train_ae_dle(np.arange(32.0), config=_small_config())
