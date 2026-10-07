"""Experimental AET changes, with explicit stationary-line assumptions.

No clean waveform, scene label, true line frequency or requested SNR enters
either enhancement method. The spectral head is an offline, global operation.
"""
from dataclasses import dataclass, replace
import copy

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .aet_dle import (
    AETDLEConfig, AETDLETrainingResult, AutoencoderTransformerDLE,
    _as_signal, _sequence_pairs,
)


@dataclass(frozen=True)
class RMSScaler:
    mean: float
    scale: float

    @classmethod
    def fit(cls, signal):
        mean = float(np.mean(signal))
        scale = float(np.std(signal))
        if scale <= 0:
            raise ValueError('signal must have nonzero variance')
        return cls(mean, scale)

    def transform(self, signal, *, clip=False):
        # A linear output has no [0,1] bound; inference must not clip RMS values.
        return (signal-self.mean)/self.scale

    def inverse_transform(self, signal):
        return signal*self.scale+self.mean


class RegularizedAET(AutoencoderTransformerDLE):
    def __init__(self, config):
        super().__init__(config)
        self.frame_decoder[-1] = nn.Identity()


def train_regularized_aet(noisy: ArrayLike, validation_noisy, *, seed=0,
                          maximum_epochs=100, minimum_epochs=40, patience=20):
    """RMS + linear decoder + AdamW/dropout; independent noisy validation."""
    samples = _as_signal(noisy)
    config = AETDLEConfig(seed=seed, epochs=maximum_epochs, dropout=.05, learning_rate=3e-4)
    scaler = RMSScaler.fit(samples)
    x,y,_ = _sequence_pairs(scaler.transform(samples),config)
    dataset = TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
    loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    vx,vy = [],[]
    for observation in validation_noisy:
        a,b,_ = _sequence_pairs(scaler.transform(_as_signal(observation)),config)
        vx.append(a);vy.append(b)
    vx = torch.from_numpy(np.concatenate(vx)).float()
    vy = torch.from_numpy(np.concatenate(vy)).float()
    torch.manual_seed(seed)
    model = RegularizedAET(config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.01)
    best_loss,best_epoch,best_state,stale = float('inf'),0,None,0
    history = []
    for epoch in range(1,maximum_epochs+1):
        model.train();total,count = 0.,0
        for inputs,targets in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(inputs),targets)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total += float(loss.detach())*targets.numel();count += targets.numel()
        model.eval();val_total = 0.
        with torch.inference_mode():
            for start in range(0,len(vx),64):
                error = model(vx[start:start+64])-vy[start:start+64]
                val_total += float(torch.sum(error.double()**2))
        val_loss = val_total/vy.numel()
        if epoch >= minimum_epochs:
            if val_loss < best_loss*(1-1e-4):
                best_loss,best_epoch,stale = val_loss,epoch,0
                best_state = copy.deepcopy(model.state_dict())
            else: stale += 1
        history.append({'epoch':epoch,'training_mse':total/count,'validation_mse':val_loss})
        if stale >= patience: break
    if best_state is None:
        raise ValueError('maximum_epochs must be at least minimum_epochs')
    model.load_state_dict(best_state);model.eval()
    selected_config = replace(config,epochs=best_epoch)
    trained = AETDLETrainingResult(model,scaler,selected_config,
                                   tuple(r['training_mse'] for r in history[:best_epoch]))
    return trained,history,{'selected_epoch':best_epoch,'stopped_epoch':len(history),'best_validation_mse':best_loss}


@dataclass(frozen=True)
class SpectralHeadConfig:
    sample_rate_hz: float = 1000.
    family_false_alarm: float = .01
    maximum_lines: int = 24
    padding_factor: int = 8
    neural_weight: float = .05

    def __post_init__(self):
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError('sample rate must be positive')
        if not 0 < self.family_false_alarm < 1:
            raise ValueError('false-alarm setting must lie in (0,1)')
        if not 0 <= self.neural_weight <= 1:
            raise ValueError('neural weight must lie in [0,1]')
        if self.maximum_lines < 1 or self.padding_factor < 1:
            raise ValueError('line limit and padding factor must be positive')


def spectral_residual_head(noisy: ArrayLike, neural_prediction: ArrayLike,
                           *, config: SpectralHeadConfig | None = None,
                           neural_valid_slice: slice | None = None):
    """Detect lines from the noisy observation; anchor AET in their subspace.

    y = (1-w) P(noisy) + w P(AET). Peak locations are inferred automatically.
    Frequency detection uses a Hann periodogram with a conservative threshold;
    its false-alarm setting is nominal, not a proven detection guarantee.
    """
    settings = config or SpectralHeadConfig()
    samples = _as_signal(noisy)
    neural = _as_signal(neural_prediction)
    if neural.shape != samples.shape:
        raise ValueError('prediction and observation must have identical lengths')
    count = samples.size
    if count < 32:
        raise ValueError('spectral head needs at least 32 samples')
    n_fft = count*settings.padding_factor
    centered = samples-np.mean(samples)
    power = np.abs(np.fft.rfft(centered*np.hanning(count),n=n_fft))**2
    noise_floor = float(np.median(power[1:-1])/np.log(2))
    threshold = noise_floor*np.log((power.size-2)/settings.family_false_alarm)
    times = np.arange(count)/settings.sample_rate_hz
    frequencies = []
    residual = centered.copy()
    # Remove each fitted line before looking for another one, so the sidelobes
    # of a strong off-grid line are not mistaken for additional physical lines.
    for _ in range(settings.maximum_lines):
        power = np.abs(np.fft.rfft(residual*np.hanning(count),n=n_fft))**2
        peaks = np.flatnonzero((power[1:-1]>power[:-2]) & (power[1:-1]>power[2:]) &
                               (power[1:-1]>threshold))+1
        ranked = sorted(peaks,key=lambda i:power[i],reverse=True)
        ranked = [i for i in ranked if all(abs(i*settings.sample_rate_hz/n_fft-f)>2*settings.sample_rate_hz/count
                                           for f in frequencies)]
        if not ranked: break
        index = ranked[0]
        # Sub-bin log-power parabola. No true frequency list is supplied.
        a,b,c = np.log(np.maximum(power[index-1:index+2],np.finfo(float).tiny))
        offset = float(np.clip(.5*(a-c)/(a-2*b+c),-.5,.5))
        frequencies.append((index+offset)*settings.sample_rate_hz/n_fft)
        basis = np.column_stack([fn(2*np.pi*f*times) for f in frequencies for fn in (np.sin,np.cos)])
        residual = centered-basis@np.linalg.lstsq(basis,centered,rcond=None)[0]
    frequencies.sort()
    if not frequencies:
        zero = np.zeros_like(samples)
        return {'enhanced_signal':zero,'spectral_only':zero.copy(),'frequencies_hz':[],
                'noise_floor':noise_floor,'detection_threshold':threshold,
                'neural_weight':settings.neural_weight,'no_line_detected':True}
    basis = np.column_stack([fn(2*np.pi*f*times) for f in frequencies for fn in (np.sin,np.cos)])
    coefficients = np.linalg.lstsq(basis,samples,rcond=None)[0]
    interval = neural_valid_slice or slice(None)
    neural_coefficients = np.linalg.lstsq(basis[interval],neural[interval],rcond=None)[0]
    spectral = basis@coefficients
    enhanced = basis@((1-settings.neural_weight)*coefficients+settings.neural_weight*neural_coefficients)
    return {'enhanced_signal':enhanced,'spectral_only':spectral,'frequencies_hz':frequencies,
            'noise_floor':noise_floor,'detection_threshold':threshold,
            'neural_weight':settings.neural_weight,'no_line_detected':False}
