"""Transformer-based adaptive line enhancer."""

from .ale import ALEResult, adaptive_line_enhance
from .ae_dle import (
    AEDLEConfig,
    AEDLEEnhancementResult,
    AEDLETrainingResult,
    AutoencoderDLE,
    enhance_with_ae_dle,
    train_ae_dle,
)
from .aet_dle import (
    AETDLEConfig,
    AETDLEEnhancementResult,
    AETDLETrainingResult,
    AutoencoderTransformerDLE,
    enhance_with_aet_dle,
    train_aet_dle,
)
from .signals import NoisySignal, generate_noisy_signal
from .visualization import (
    plot_line_enhancement_spectrogram,
    plot_noisy_signal_components,
)

__all__ = [
    "ALEResult",
    "AEDLEConfig",
    "AEDLEEnhancementResult",
    "AEDLETrainingResult",
    "AutoencoderDLE",
    "AETDLEConfig",
    "AETDLEEnhancementResult",
    "AETDLETrainingResult",
    "AutoencoderTransformerDLE",
    "NoisySignal",
    "adaptive_line_enhance",
    "enhance_with_ae_dle",
    "enhance_with_aet_dle",
    "generate_noisy_signal",
    "plot_line_enhancement_spectrogram",
    "plot_noisy_signal_components",
    "train_ae_dle",
    "train_aet_dle",
]
