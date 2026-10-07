from dataclasses import asdict

import numpy as np
import pytest
import torch

from transformer_based_adaptive_line_enhancer.aet_attention import (
    AttentionConfig, LongRangeLagAttention, frame_pairs, train_attention_aet,
    enhance_attention_aet, restore_attention_aet,
)


@pytest.mark.parametrize('mode', ['lag', 'uniform', 'frozen'])
def test_attention_matches_finite_masked_weighted_sum(mode):
    """Check FFT orientation, non-circular boundaries and target-value masking."""
    torch.manual_seed(7)
    config=AttentionConfig(patch_length=4,heads=2,prediction_delay=16,top_k=8,attention=mode)
    layer=LongRangeLagAttention(config).double()
    x=torch.randn(2,20,4,dtype=torch.float64)
    actual=layer(x,x)
    lags=layer.last_lags;weights=layer.last_weights
    assert torch.all(lags!=0)
    assert torch.all((config.prediction_delay+lags*config.patch_length).abs()>=config.patch_length)
    values=layer.v(x).reshape(2,20,2,2).permute(0,2,1,3)
    expected=torch.zeros_like(values)
    for batch in range(2):
        for head in range(2):
            for query in range(20):
                index=query+lags[batch,head]
                valid=(index>=0)&(index<20)
                mass=weights[batch,head,valid].sum()
                if mass>1e-6:
                    expected[batch,head,query]=(values[batch,head,index[valid]]*weights[batch,head,valid,None]).sum(0)/mass
    expected=layer.out(expected.permute(0,2,1,3).reshape(2,20,4))
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-9)
    if mode=='lag':
        grad_actual=torch.autograd.grad(actual.square().sum(),layer.v.weight,retain_graph=True)[0]
        grad_expected=torch.autograd.grad(expected.square().sum(),layer.v.weight)[0]
        torch.testing.assert_close(grad_actual,grad_expected,rtol=1e-8,atol=1e-8)


def test_frame_pairs_have_correct_physical_delay():
    config=AttentionConfig(patch_length=4,prediction_delay=16)
    x,y,end=frame_pairs(np.arange(101),config)
    assert end==84
    np.testing.assert_array_equal(x-y,16)
    assert x[-1,-1]==99


def test_training_selects_noisy_validation_and_checkpoint_replays():
    torch.set_num_threads(2)
    rng=np.random.default_rng(12)
    clean=np.sin(np.arange(256)*2*np.pi/16)
    noisy=clean+rng.normal(size=256)
    validation=clean[None]+rng.normal(size=(3,256))
    config=AttentionConfig(patch_length=4,prediction_delay=32,updates=12,validate_every=1,patience=4,top_k=16)
    trained=train_attention_aet(noisy,validation,config)
    losses=trained['history']
    assert any(r['update']==trained['selected_update'] for r in losses)
    selected=next(r for r in losses if r['update']==trained['selected_update'])
    assert selected['validation_loss']==trained['best_validation_loss']
    assert selected['validation_loss']<=min(r['validation_loss'] for r in losses)+1e-6
    prediction,end=enhance_attention_aet(noisy,trained)
    assert np.isfinite(prediction).all()
    restored=restore_attention_aet({'state_dict':trained['model'].state_dict(),'config':asdict(config),
        'mean':trained['mean'],'scale':trained['scale'],'selected_update':trained['selected_update']})
    replay,replay_end=enhance_attention_aet(noisy,restored)
    assert replay_end==end
    np.testing.assert_array_equal(replay,prediction)
    with pytest.raises(ValueError,match='independent validation'):
        train_attention_aet(noisy,validation[:1],config)
    with pytest.raises(ValueError,match='zero variance'):
        train_attention_aet(np.ones(256),validation,config)


@pytest.mark.parametrize('kwargs', [dict(heads=3),dict(top_k=0),dict(prediction_delay=0),dict(temperature=0),dict(attention='invalid')])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):AttentionConfig(**kwargs)
