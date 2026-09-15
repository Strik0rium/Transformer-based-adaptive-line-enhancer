import numpy as np
import pytest
import torch

from transformer_based_adaptive_line_enhancer import (
    AETDLEConfig,
    AutoencoderTransformerDLE,
    enhance_with_aet_dle,
    generate_noisy_signal,
    train_aet_dle,
)


def _small_config(**overrides: object) -> AETDLEConfig:
    values: dict[str, object] = {
        "frame_length": 64,
        "frame_hop": 16,
        "prediction_delay": 64,
        "sequence_length": 4,
        "sequence_stride": 2,
        "embedding_dim": 16,
        "attention_heads": 4,
        "transformer_layers": 1,
        "feedforward_dim": 32,
        "dropout": 0.0,
        "learning_rate": 2e-3,
        "batch_size": 16,
        "epochs": 8,
        "seed": 7,
    }
    values.update(overrides)
    return AETDLEConfig(**values)


def test_model_preserves_sequence_shape_and_output_range() -> None:
    config = _small_config()
    model = AutoencoderTransformerDLE(config)
    output = model(torch.zeros((3, config.sequence_length, config.frame_length)))

    assert output.shape == (3, config.sequence_length, config.frame_length)
    assert torch.all((output >= 0) & (output <= 1))


def test_model_rejects_non_sequence_input() -> None:
    model = AutoencoderTransformerDLE(_small_config())

    with pytest.raises(ValueError):
        model(torch.zeros((3, 64)))


def test_attention_uses_context_from_other_frames() -> None:
    config = _small_config()
    torch.manual_seed(config.seed)
    model = AutoencoderTransformerDLE(config).eval()
    baseline = torch.zeros((1, config.sequence_length, config.frame_length))
    changed_context = baseline.clone()
    changed_context[:, 0, :] = 1.0

    with torch.inference_mode():
        baseline_output = model(baseline)
        changed_output = model(changed_context)

    assert not torch.allclose(baseline_output[:, -1], changed_output[:, -1])


def test_training_loss_decreases() -> None:
    sample = generate_noisy_signal(
        frequencies_hz=50.0,
        sample_rate_hz=500.0,
        num_samples=2_000,
        snr_db=-5.0,
        seed=10,
        scaling="power-normalized",
    )
    training = train_aet_dle(sample.noisy_signal, config=_small_config())

    assert len(training.loss_history) == 8
    assert training.loss_history[-1] < training.loss_history[0]


def test_aet_dle_improves_single_tone_snr_in_a_locked_scene() -> None:
    sample = generate_noisy_signal(
        frequencies_hz=50.0,
        sample_rate_hz=500.0,
        num_samples=4_000,
        snr_db=-10.0,
        seed=10,
        scaling="power-normalized",
    )
    training = train_aet_dle(sample.noisy_signal, config=_small_config())
    result = enhance_with_aet_dle(sample.noisy_signal, training)
    clean = sample.clean_signal[result.valid_slice]
    noisy = sample.noisy_signal[result.valid_slice]
    enhanced = result.enhanced_signal[result.valid_slice]

    input_snr = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    output_snr = 10 * np.log10(np.mean(clean**2) / np.mean((enhanced - clean) ** 2))

    assert output_snr > input_snr + 8.0


def test_enhancement_is_time_aligned_and_finite() -> None:
    sample = generate_noisy_signal(num_samples=1_000, seed=2)
    training = train_aet_dle(
        sample.noisy_signal, config=_small_config(epochs=2)
    )
    result = enhance_with_aet_dle(sample.noisy_signal, training)

    assert result.enhanced_signal.shape == sample.noisy_signal.shape
    assert result.residual.shape == sample.noisy_signal.shape
    assert result.valid_start == 0
    assert result.valid_end < sample.noisy_signal.size
    assert np.all(np.isfinite(result.enhanced_signal))
    np.testing.assert_allclose(
        result.residual[result.valid_slice],
        sample.noisy_signal[result.valid_slice] - result.enhanced_signal[result.valid_slice],
    )


def test_training_is_deterministic_for_fixed_seed() -> None:
    signal = np.sin(2 * np.pi * 0.05 * np.arange(800))
    config = _small_config(epochs=2)
    first = train_aet_dle(signal, config=config)
    second = train_aet_dle(signal, config=config)

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
        {"prediction_delay": 63},
        {"sequence_length": 1},
        {"sequence_stride": 0},
        {"embedding_dim": 15, "attention_heads": 4},
        {"feedforward_dim": 8},
        {"dropout": 1.0},
        {"learning_rate": 0},
        {"epochs": 0},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _small_config(**overrides)


def test_too_short_signal_is_rejected() -> None:
    with pytest.raises(ValueError):
        train_aet_dle(np.arange(64.0), config=_small_config())
