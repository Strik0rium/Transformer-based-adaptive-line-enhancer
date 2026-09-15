"""Self-supervised autoencoder deep line enhancement inspired by delayed ALE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class AEDLEConfig:
    """Architecture, framing, and optimization settings for AE-DLE."""

    frame_length: int = 256
    frame_hop: int = 64
    prediction_delay: int = 1_000
    hidden_dim: int = 128
    bottleneck_dim: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "frame_length",
            "frame_hop",
            "prediction_delay",
            "hidden_dim",
            "bottleneck_dim",
            "batch_size",
            "epochs",
        ):
            _validate_positive_integer(getattr(self, name), name)
        if self.frame_hop > self.frame_length:
            raise ValueError("frame_hop must not exceed frame_length")
        if self.bottleneck_dim >= self.hidden_dim:
            raise ValueError("bottleneck_dim must be smaller than hidden_dim")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class MinMaxScaler:
    """Training-signal min-max normalization state."""

    minimum: float
    maximum: float

    @classmethod
    def fit(cls, signal: NDArray[np.float64]) -> MinMaxScaler:
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


class AutoencoderDLE(nn.Module):
    """Symmetric bottleneck MLP mapping delayed frames to current frames."""

    def __init__(self, *, frame_length: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        _validate_positive_integer(frame_length, "frame_length")
        _validate_positive_integer(hidden_dim, "hidden_dim")
        _validate_positive_integer(bottleneck_dim, "bottleneck_dim")
        if bottleneck_dim >= hidden_dim:
            raise ValueError("bottleneck_dim must be smaller than hidden_dim")
        self.encoder = nn.Sequential(
            nn.Linear(frame_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, frame_length),
            nn.Sigmoid(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.decoder(self.encoder(frames))


@dataclass(frozen=True)
class AEDLETrainingResult:
    """A trained model and all preprocessing state needed for enhancement."""

    model: AutoencoderDLE
    scaler: MinMaxScaler
    config: AEDLEConfig
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class AEDLEEnhancementResult:
    """Enhanced waveform and its aligned valid interval."""

    enhanced_signal: NDArray[np.float64]
    residual: NDArray[np.float64]
    valid_start: int
    valid_end: int

    @property
    def valid_slice(self) -> slice:
        return slice(self.valid_start, self.valid_end)


def train_ae_dle(
    signal: ArrayLike,
    *,
    config: AEDLEConfig | None = None,
) -> AEDLETrainingResult:
    """Train AE-DLE from delayed/current frame pairs of one noisy observation.

    No clean reference is accepted by this API. The delayed noisy frame is the
    network input and the corresponding current noisy frame is the target.
    """
    settings = config or AEDLEConfig()
    samples = _as_signal(signal)
    scaler = MinMaxScaler.fit(samples)
    normalized = scaler.transform(samples)
    delayed_frames, current_frames = _delayed_frame_pairs(normalized, settings)

    torch.manual_seed(settings.seed)
    model = AutoencoderDLE(
        frame_length=settings.frame_length,
        hidden_dim=settings.hidden_dim,
        bottleneck_dim=settings.bottleneck_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(delayed_frames).to(torch.float32),
        torch.from_numpy(current_frames).to(torch.float32),
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
        for delayed_batch, current_batch in loader:
            optimizer.zero_grad()
            prediction = model(delayed_batch)
            loss = loss_function(prediction, current_batch)
            loss.backward()
            optimizer.step()

            total_squared_error += float(loss.detach()) * current_batch.numel()
            total_values += current_batch.numel()
        loss_history.append(total_squared_error / total_values)

    model.eval()
    return AEDLETrainingResult(
        model=model,
        scaler=scaler,
        config=settings,
        loss_history=tuple(loss_history),
    )


def enhance_with_ae_dle(
    signal: ArrayLike,
    training: AEDLETrainingResult,
) -> AEDLEEnhancementResult:
    """Predict and overlap-average aligned future frames with a trained AE-DLE."""
    samples = _as_signal(signal)
    settings = training.config
    normalized = training.scaler.transform(samples, clip=True)
    frame_starts = _frame_starts(samples.size, settings)
    input_frames = np.stack(
        [normalized[start : start + settings.frame_length] for start in frame_starts]
    )

    with torch.inference_mode():
        predicted_frames = (
            training.model(torch.from_numpy(input_frames).to(torch.float32))
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    reconstructed = np.zeros(samples.size, dtype=np.float64)
    overlap_count = np.zeros(samples.size, dtype=np.float64)
    for frame_start, predicted_frame in zip(frame_starts, predicted_frames, strict=True):
        output_start = frame_start + settings.prediction_delay
        output_end = output_start + settings.frame_length
        reconstructed[output_start:output_end] += predicted_frame
        overlap_count[output_start:output_end] += 1.0

    valid = overlap_count > 0
    reconstructed[valid] /= overlap_count[valid]
    enhanced = np.zeros_like(samples)
    enhanced[valid] = training.scaler.inverse_transform(reconstructed[valid])
    residual = samples.copy()
    residual[valid] = samples[valid] - enhanced[valid]
    valid_indices = np.flatnonzero(valid)

    return AEDLEEnhancementResult(
        enhanced_signal=enhanced,
        residual=residual,
        valid_start=int(valid_indices[0]),
        valid_end=int(valid_indices[-1] + 1),
    )


def _delayed_frame_pairs(
    normalized_signal: NDArray[np.float64],
    config: AEDLEConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    frame_starts = _frame_starts(normalized_signal.size, config)
    delayed_frames = np.stack(
        [normalized_signal[start : start + config.frame_length] for start in frame_starts]
    )
    current_frames = np.stack(
        [
            normalized_signal[
                start + config.prediction_delay :
                start + config.prediction_delay + config.frame_length
            ]
            for start in frame_starts
        ]
    )
    return delayed_frames, current_frames


def _frame_starts(signal_length: int, config: AEDLEConfig) -> NDArray[np.int64]:
    last_start = signal_length - config.prediction_delay - config.frame_length
    if last_start < 0:
        raise ValueError(
            "signal is too short for prediction_delay and frame_length"
        )
    return np.arange(0, last_start + 1, config.frame_hop, dtype=np.int64)


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
