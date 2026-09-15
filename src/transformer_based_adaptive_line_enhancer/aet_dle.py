"""Autoencoder-Transformer deep line enhancement from delayed self-supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class AETDLEConfig:
    """Framing, architecture, and optimization settings for AET-DLE."""

    frame_length: int = 320
    frame_hop: int = 64
    prediction_delay: int = 1_000
    sequence_length: int = 8
    sequence_stride: int = 1
    embedding_dim: int = 64
    attention_heads: int = 8
    transformer_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 30
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "frame_length",
            "frame_hop",
            "prediction_delay",
            "sequence_length",
            "sequence_stride",
            "embedding_dim",
            "attention_heads",
            "transformer_layers",
            "feedforward_dim",
            "batch_size",
            "epochs",
        ):
            _validate_positive_integer(getattr(self, name), name)
        if self.frame_hop > self.frame_length:
            raise ValueError("frame_hop must not exceed frame_length")
        if self.prediction_delay < self.frame_length:
            raise ValueError("prediction_delay must not be smaller than frame_length")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.embedding_dim % self.attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if self.feedforward_dim < self.embedding_dim:
            raise ValueError("feedforward_dim must not be smaller than embedding_dim")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class AETMinMaxScaler:
    """Min-max state learned only from the noisy training observation."""

    minimum: float
    maximum: float

    @classmethod
    def fit(cls, signal: NDArray[np.float64]) -> AETMinMaxScaler:
        minimum = float(np.min(signal))
        maximum = float(np.max(signal))
        if maximum <= minimum:
            raise ValueError("signal must have nonzero range")
        return cls(minimum=minimum, maximum=maximum)

    def transform(
        self, signal: NDArray[np.float64], *, clip: bool = False
    ) -> NDArray[np.float64]:
        normalized = (signal - self.minimum) / (self.maximum - self.minimum)
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def inverse_transform(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        return signal * (self.maximum - self.minimum) + self.minimum


class SinusoidalPositionEncoding(nn.Module):
    """Deterministic frame-position encoding for a batch-first sequence."""

    def __init__(self, embedding_dim: int, maximum_length: int) -> None:
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        frequency_scale = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-np.log(10_000.0) / embedding_dim)
        )
        encoding = torch.zeros((maximum_length, embedding_dim), dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequency_scale)
        encoding[:, 1::2] = torch.cos(
            positions * frequency_scale[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, sequence: Tensor) -> Tensor:
        return sequence + self.encoding[: sequence.shape[1]]


class AutoencoderTransformerDLE(nn.Module):
    """Encode frames, attend across frames, and decode every frame."""

    def __init__(self, config: AETDLEConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = nn.Sequential(
            nn.Linear(config.frame_length, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
        )
        self.position_encoding = SinusoidalPositionEncoding(
            config.embedding_dim, config.sequence_length
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.frame_decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.frame_length),
            nn.Sigmoid(),
        )

    def forward(self, frame_sequences: Tensor) -> Tensor:
        if frame_sequences.ndim != 3:
            raise ValueError("frame_sequences must have shape (batch, sequence, frame)")
        if frame_sequences.shape[1:] != (
            self.config.sequence_length,
            self.config.frame_length,
        ):
            raise ValueError("frame_sequences shape does not match the model config")
        encoded = self.frame_encoder(frame_sequences)
        contextual = self.transformer(self.position_encoding(encoded))
        return self.frame_decoder(contextual)


@dataclass(frozen=True)
class AETDLETrainingResult:
    """A trained AET-DLE and its preprocessing state."""

    model: AutoencoderTransformerDLE
    scaler: AETMinMaxScaler
    config: AETDLEConfig
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class AETDLEEnhancementResult:
    """Offline time-aligned AET-DLE reconstruction."""

    enhanced_signal: NDArray[np.float64]
    residual: NDArray[np.float64]
    valid_start: int
    valid_end: int

    @property
    def valid_slice(self) -> slice:
        return slice(self.valid_start, self.valid_end)


def train_aet_dle(
    signal: ArrayLike,
    *,
    config: AETDLEConfig | None = None,
) -> AETDLETrainingResult:
    """Train AET-DLE from current noisy sequences and delayed noisy targets."""
    settings = config or AETDLEConfig()
    samples = _as_signal(signal)
    scaler = AETMinMaxScaler.fit(samples)
    normalized = scaler.transform(samples)
    current_sequences, delayed_sequences, _ = _sequence_pairs(normalized, settings)

    torch.manual_seed(settings.seed)
    model = AutoencoderTransformerDLE(settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(current_sequences).to(torch.float32),
        torch.from_numpy(delayed_sequences).to(torch.float32),
    )
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
    )
    loss_history: list[float] = []

    model.train()
    for _ in range(settings.epochs):
        total_squared_error = 0.0
        total_values = 0
        for current_batch, delayed_batch in loader:
            optimizer.zero_grad()
            prediction = model(current_batch)
            loss = loss_function(prediction, delayed_batch)
            loss.backward()
            optimizer.step()

            total_squared_error += float(loss.detach()) * delayed_batch.numel()
            total_values += delayed_batch.numel()
        loss_history.append(total_squared_error / total_values)

    model.eval()
    return AETDLETrainingResult(
        model=model,
        scaler=scaler,
        config=settings,
        loss_history=tuple(loss_history),
    )


def enhance_with_aet_dle(
    signal: ArrayLike,
    training: AETDLETrainingResult,
) -> AETDLEEnhancementResult:
    """Enhance a signal and place predictions at their delayed target times."""
    samples = _as_signal(signal)
    normalized = training.scaler.transform(samples, clip=True)
    current_sequences, _, target_starts = _sequence_pairs(
        normalized, training.config
    )

    with torch.inference_mode():
        predicted_sequences = (
            training.model(torch.from_numpy(current_sequences).to(torch.float32))
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    reconstructed = np.zeros(samples.size, dtype=np.float64)
    overlap_count = np.zeros(samples.size, dtype=np.float64)
    for sequence_starts, predicted_sequence in zip(
        target_starts, predicted_sequences, strict=True
    ):
        for frame_start, predicted_frame in zip(
            sequence_starts, predicted_sequence, strict=True
        ):
            frame_end = frame_start + training.config.frame_length
            reconstructed[frame_start:frame_end] += predicted_frame
            overlap_count[frame_start:frame_end] += 1.0

    valid = overlap_count > 0
    reconstructed[valid] /= overlap_count[valid]
    enhanced = np.zeros_like(samples)
    enhanced[valid] = training.scaler.inverse_transform(reconstructed[valid])
    residual = samples.copy()
    residual[valid] = samples[valid] - enhanced[valid]
    valid_indices = np.flatnonzero(valid)

    return AETDLEEnhancementResult(
        enhanced_signal=enhanced,
        residual=residual,
        valid_start=int(valid_indices[0]),
        valid_end=int(valid_indices[-1] + 1),
    )


def _sequence_pairs(
    normalized_signal: NDArray[np.float64],
    config: AETDLEConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    last_target_start = (
        normalized_signal.size - config.prediction_delay - config.frame_length
    )
    if last_target_start < 0:
        raise ValueError(
            "signal is too short for prediction_delay and frame_length"
        )
    frame_starts = np.arange(
        0, last_target_start + 1, config.frame_hop, dtype=np.int64
    )
    if frame_starts.size < config.sequence_length:
        raise ValueError("signal does not contain enough frames for one sequence")

    sequence_offsets = np.arange(config.sequence_length) * config.frame_hop
    first_frame_indices = np.arange(
        0,
        frame_starts.size - config.sequence_length + 1,
        config.sequence_stride,
    )
    target_starts = np.stack(
        [frame_starts[index] + sequence_offsets for index in first_frame_indices]
    )
    delayed_sequences = np.stack(
        [
            np.stack(
                [
                    normalized_signal[start : start + config.frame_length]
                    for start in starts
                ]
            )
            for starts in target_starts
        ]
    )
    current_sequences = np.stack(
        [
            np.stack(
                [
                    normalized_signal[
                        start + config.prediction_delay :
                        start + config.prediction_delay + config.frame_length
                    ]
                    for start in starts
                ]
            )
            for starts in target_starts
        ]
    )
    return current_sequences, delayed_sequences, target_starts


def _as_signal(signal: ArrayLike) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal must contain only finite values")
    return samples


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
