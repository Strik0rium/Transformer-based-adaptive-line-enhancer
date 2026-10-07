"""Experimental AET with trainable long-range lag attention.

The FFT below evaluates Q/K correlations inside attention. There is no spectral
peak detector, harmonic fit, filtered-input branch, or output postprocessor.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AttentionConfig:
    patch_length: int = 32
    prediction_delay: int = 1000
    heads: int = 4
    top_k: int = 64
    temperature: float = .05
    learning_rate: float = .002
    weight_decay: float = .01
    updates: int = 400
    validate_every: int = 5
    patience: int = 80
    seed: int = 0
    attention: str = 'lag'
    ffn: bool = True


class LongRangeLagAttention(nn.Module):
    """Shared relative-lag attention, trained Q/K/V, finite non-circular context."""
    def __init__(self, config):
        super().__init__()
        self.config=config
        p=config.patch_length
        self.q=nn.Linear(p,p,bias=False)
        self.k=nn.Linear(p,p,bias=False)
        self.v=nn.Linear(p,p,bias=False)
        self.out=nn.Linear(p,p,bias=False)
        for layer in [self.q,self.k,self.v,self.out]:
            nn.init.eye_(layer.weight)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(config.temperature)))
        self.last_lags=None
        self.last_weights=None

    def forward(self, encoded, raw=None):
        cfg=self.config
        b,t,p=encoded.shape;h=cfg.heads;d=p//h
        if cfg.attention=='none':return encoded
        values=self.v(encoded).reshape(b,t,h,d).permute(0,2,1,3)
        cap=t//2
        lags=torch.arange(-cap,cap+1,device=encoded.device)
        # Input patch k covers delay + k*p; target q covers q*p.
        # Exclude all value patches that overlap the target's noise samples.
        allowed=(lags!=0)&((cfg.prediction_delay+lags*p).abs()>=p)
        lags=lags[allowed]
        if cfg.attention=='uniform':
            chosen=lags[None,None].expand(b,h,-1)
            weights=torch.ones_like(chosen,dtype=encoded.dtype)/len(lags)
        else:
            if cfg.attention=='frozen':
                queries=raw.reshape(b,t,h,d).permute(0,2,3,1)
                keys=queries
            else:
                queries=self.q(encoded).reshape(b,t,h,d).permute(0,2,3,1)
                keys=self.k(encoded).reshape(b,t,h,d).permute(0,2,3,1)
            size=2*t
            qfft=torch.fft.rfft(queries,n=size)
            kfft=torch.fft.rfft(keys,n=size)
            correlation=torch.fft.irfft(qfft.conj()*kfft,n=size).mean(dim=2)
            scores=correlation[:,:,lags.remainder(size)]/(t-lags.abs())
            norm=(queries.square().mean(dim=(-1,-2))*keys.square().mean(dim=(-1,-2))).sqrt().clamp_min(1e-8)
            scores=scores/norm[:,:,None]
            k=min(cfg.top_k,len(lags))
            scores,indices=scores.topk(k,dim=-1)
            chosen=lags[indices]
            temperature=(cfg.temperature if cfg.attention=='frozen' else self.log_temperature.exp().clamp(.002,.5))
            weights=(scores/temperature).softmax(dim=-1)
        self.last_lags=chosen.detach()
        self.last_weights=weights.detach()
        positions=torch.arange(t,device=encoded.device)[None,None,:,None]+chosen[:,:,None,:]
        valid=(positions>=0)&(positions<t)
        indexes=positions.clamp(0,t-1)+torch.arange(b*h,device=encoded.device).reshape(b,h,1,1)*t
        selected=nn.functional.embedding(indexes,values.reshape(b*h*t,d))
        local_weights=weights[:,:,None,:]*valid
        local_weights=local_weights/local_weights.sum(dim=-1,keepdim=True).clamp_min(1e-8)
        attended=(selected*local_weights[...,None]).sum(dim=-2)
        attended=attended.permute(0,2,1,3).reshape(b,t,p)
        return self.out(attended)


class AttentionAET(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        p=config.patch_length
        if p%config.heads:raise ValueError('patch length must divide into attention heads')
        self.encoder=nn.Linear(p,p,bias=False)
        self.attention=LongRangeLagAttention(config)
        self.norm=nn.LayerNorm(p)
        self.ffn=nn.Sequential(nn.Linear(p,2*p),nn.GELU(),nn.Linear(2*p,p))
        self.decoder=nn.Linear(p,p,bias=False)
        nn.init.eye_(self.encoder.weight);nn.init.eye_(self.decoder.weight)
        nn.init.zeros_(self.ffn[-1].weight);nn.init.zeros_(self.ffn[-1].bias)
        with torch.no_grad():
            self.encoder.weight.add_(.005*torch.randn_like(self.encoder.weight))

    def forward(self,frames):
        encoded=self.encoder(frames)
        attended=self.attention(encoded,frames)
        if self.config.ffn:attended=attended+.1*self.ffn(self.norm(attended))
        return self.decoder(attended)


def frame_pairs(signal,config):
    count=(len(signal)-config.prediction_delay)//config.patch_length
    end=count*config.patch_length
    if count<8:raise ValueError('observation is too short')
    target=np.asarray(signal[:end]).reshape(count,config.patch_length)
    source=np.asarray(signal[config.prediction_delay:config.prediction_delay+end]).reshape(count,config.patch_length)
    return source,target,end


def train_attention_aet(noisy,validation_observations,config):
    """Select with independent input/target noise; accepts no clean reference."""
    samples=np.asarray(noisy,dtype=np.float64)
    mean=float(np.mean(samples));scale=float(np.std(samples))
    x,y,end=frame_pairs((samples-mean)/scale,config)
    x=torch.from_numpy(x).float()[None];y=torch.from_numpy(y).float()[None]
    vx,vy=[],[]
    for observation in validation_observations:
        a,z,_=frame_pairs((np.asarray(observation)-mean)/scale,config)
        vx.append(a);vy.append(z)
    vx=torch.from_numpy(np.stack(vx)).float()
    # Independent noisy realization at matching physical target times.
    vy=torch.from_numpy(np.roll(np.stack(vy),-1,axis=0).copy()).float()
    torch.manual_seed(config.seed)
    model=AttentionAET(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    best=float('inf');best_update=0;best_state=None;history=[]
    for update in range(1,config.updates+1):
        model.train();optimizer.zero_grad()
        loss=nn.functional.mse_loss(model(x),y)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if update==1 or update%config.validate_every==0:
            model.eval()
            with torch.inference_mode():val_loss=float(nn.functional.mse_loss(model(vx),vy))
            history.append({'update':update,'training_loss':float(loss.detach()),'validation_loss':val_loss})
            if val_loss<best-1e-6:
                best=val_loss;best_update=update;best_state=copy.deepcopy(model.state_dict())
            if update-best_update>=config.patience:break
    model.load_state_dict(best_state);model.eval()
    return {'model':model,'mean':mean,'scale':scale,'config':config,'history':history,
            'selected_update':best_update,'stopped_update':update,'best_validation_loss':best}


def enhance_attention_aet(noisy,training):
    samples=np.asarray(noisy,dtype=np.float64)
    x,_,end=frame_pairs((samples-training['mean'])/training['scale'],training['config'])
    with torch.inference_mode():values=training['model'](torch.from_numpy(x).float()[None])[0].numpy().reshape(-1)
    output=np.zeros_like(samples)
    output[:end]=values*training['scale']+training['mean']
    return output,end
