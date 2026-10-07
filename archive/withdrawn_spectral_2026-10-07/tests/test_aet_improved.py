"""Behavioral checks for the experimental RMS decoder and spectral head."""
import numpy as np
import pytest
import torch

from transformer_based_adaptive_line_enhancer.aet_dle import AETDLEConfig
from transformer_based_adaptive_line_enhancer.aet_improved import (
    RMSScaler,RegularizedAET,SpectralHeadConfig,spectral_residual_head,
)


def test_spectral_head_detects_unlisted_off_grid_lines_and_restores_them():
    t=np.arange(4096)/1000
    clean=np.sin(2*np.pi*83.173*t+.7)+.4*np.sin(2*np.pi*211.431*t-1.2)
    noisy=clean+np.random.default_rng(12).normal(0,.1,len(t))
    result=spectral_residual_head(noisy,np.zeros_like(noisy))
    np.testing.assert_allclose(result['frequencies_hz'],[83.173,211.431],atol=.02,rtol=0)
    assert np.mean((result['enhanced_signal']-clean)**2)<.02*np.mean(clean**2)


def test_zero_neural_weight_matches_ablation_and_zero_input_has_no_lines():
    t=np.arange(2048)/1000
    noisy=np.sin(2*np.pi*143.7*t)+np.random.default_rng(2).normal(0,.1,len(t))
    result=spectral_residual_head(noisy,np.zeros_like(noisy),config=SpectralHeadConfig(neural_weight=0))
    np.testing.assert_array_equal(result['enhanced_signal'],result['spectral_only'])
    zero=spectral_residual_head(np.zeros(2048),np.zeros(2048))
    assert zero['no_line_detected'] and zero['frequencies_hz']==[]
    np.testing.assert_array_equal(zero['enhanced_signal'],np.zeros(2048))


def test_rms_inference_preserves_values_outside_training_range():
    scaler=RMSScaler.fit(np.array([-1.,0.,1.]))
    unseen=np.array([-20.,20.])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(unseen,clip=True)),unseen)


def test_regularized_decoder_can_represent_negative_rms_targets():
    model=RegularizedAET(AETDLEConfig(embedding_dim=8,attention_heads=2,transformer_layers=1,feedforward_dim=16))
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.frame_decoder[-2].bias.fill_(-2)
    with torch.inference_mode():prediction=model(torch.zeros(2,8,320))
    torch.testing.assert_close(prediction,torch.full((2,8,320),-2.))


def test_invalid_head_configuration_and_mismatched_observation_rejected():
    with pytest.raises(ValueError):SpectralHeadConfig(neural_weight=1.1)
    with pytest.raises(ValueError):spectral_residual_head(np.ones(64),np.ones(63))
