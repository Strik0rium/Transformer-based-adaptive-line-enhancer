from dataclasses import asdict

import numpy as np
import pytest
import torch

from transformer_based_adaptive_line_enhancer import aet_aligned_attention as a


@pytest.mark.parametrize('mode',['lag','frozen'])
def test_aligned_queries_match_direct_cross_correlation(mode):
    """Q uses target time, K/V delayed time, with exactly the physical mask."""
    torch.manual_seed(17)
    config=a.AttentionConfig(patch_length=4,prediction_delay=16,top_k=8,attention=mode)
    layer=a.LongRangeLagAttention(config).double()
    source=torch.randn(1,32,4,dtype=torch.float64)
    query=torch.randn(1,32,4,dtype=torch.float64)
    actual=layer(source,source,query,query)
    q=(query if mode=='frozen' else layer.q(query))[0]
    k=(source if mode=='frozen' else layer.k(source))[0]
    allowed=[lag for lag in range(-16,17) if lag and abs(16+lag*4)>=4]
    scores=[]
    for lag in allowed:
        index=torch.arange(32);valid=(index+lag>=0)&(index+lag<32)
        scores.append((q[index[valid]]*k[index[valid]+lag]).mean())
    scores=torch.stack(scores)/(q.square().mean()*k.square().mean()).sqrt()
    top_scores,index=scores.topk(8)
    chosen=torch.tensor(allowed)[index]
    temperature=config.temperature if mode=='frozen' else layer.log_temperature.exp().clamp(.002,.5)
    weights=(top_scores/temperature).softmax(-1)
    torch.testing.assert_close(layer.last_lags[0,0],chosen)
    torch.testing.assert_close(layer.last_weights[0,0],weights)
    value=layer.v(source)[0];expected=torch.zeros_like(value)
    for t in range(32):
        indexes=t+chosen;valid=(indexes>=0)&(indexes<32)
        mass=weights[valid].sum()
        if mass>1e-6:expected[t]=(value[indexes[valid]]*weights[valid,None]).sum(0)/mass
    expected=layer.out(expected)[None]
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)
    if mode=='lag':
        actual.square().sum().backward()
        assert torch.isfinite(layer.q.weight.grad).all()
        assert torch.linalg.vector_norm(layer.q.weight.grad)>0


def test_checkpoint_and_independent_noisy_validation_replay():
    torch.set_num_threads(2)
    rng=np.random.default_rng(29)
    clean=np.sin(np.arange(256)*.73)
    noisy=clean+rng.normal(size=256)
    validation=clean[None]+rng.normal(size=(3,256))
    config=a.AttentionConfig(patch_length=4,prediction_delay=32,top_k=16,updates=15,validate_every=1,patience=5)
    trained=a.train_attention_aet(noisy,validation,config)
    vx=[];vq=[]
    for observation in validation:
        x,q,_=a.frame_pairs((observation-trained['mean'])/trained['scale'],config)
        vx.append(x);vq.append(q)
    inputs=torch.from_numpy(np.stack(vx)).float()
    queries=torch.from_numpy(np.stack(vq)).float()
    targets=torch.from_numpy(np.roll(np.stack(vq),-1,axis=0).copy()).float()
    with torch.inference_mode():loss=float(torch.nn.functional.mse_loss(trained['model'](inputs,queries),targets))
    assert loss==trained['best_validation_loss']
    checkpoint={'state_dict':trained['model'].state_dict(),'config':asdict(config),'mean':trained['mean'],
        'scale':trained['scale'],'selected_update':trained['selected_update']}
    restored=a.restore_attention_aet(checkpoint)
    prediction,end=a.enhance_attention_aet(noisy,trained)
    replay,replay_end=a.enhance_attention_aet(noisy,restored)
    assert end==replay_end
    np.testing.assert_array_equal(prediction,replay)
