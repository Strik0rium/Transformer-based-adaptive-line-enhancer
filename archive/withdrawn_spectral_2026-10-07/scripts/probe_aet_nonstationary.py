"""Frozen-method scope probe: chirp from 80 to 160 Hz, rather than stable tones."""
from pathlib import Path
import hashlib
import json

import evaluate_aet_improvements as e
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer import (
    adaptive_line_enhance,train_ae_dle,enhance_with_ae_dle,train_aet_dle,
    enhance_with_aet_dle,AEDLEConfig,AETDLEConfig,
)
from transformer_based_adaptive_line_enhancer.aet_improved import spectral_residual_head


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    output=e.OUT/'nonstationary_probe';output.mkdir(exist_ok=True)
    times=np.arange(16000)/1000
    latent=np.sin(2*np.pi*(80*times+.5*5*times**2))
    rows=[]
    for snr in [-10,0]:
        for seed in [300,301]:
            noise=np.random.default_rng(seed).normal(size=len(times))
            clean=latent*np.sqrt(np.mean(noise**2)*10**(snr/10)/np.mean(latent**2))
            noisy=clean+noise
            ae_train=train_ae_dle(noisy,config=AEDLEConfig(epochs=60,seed=1000+seed))
            aet_train=train_aet_dle(noisy,config=AETDLEConfig(epochs=60,seed=1000+seed))
            signals={'ALE':adaptive_line_enhance(noisy).enhanced_signal,
                'AE':enhance_with_ae_dle(noisy,ae_train).enhanced_signal,
                'AET':enhance_with_aet_dle(noisy,aet_train).enhanced_signal}
            head=spectral_residual_head(noisy,signals['AET'],neural_valid_slice=slice(0,14976))
            signals['AET_spectral']=head['enhanced_signal'];signals['spectral_only']=head['spectral_only']
            sl=slice(8000,14976)
            measured={name:e.b.snr_db(clean[sl],values[sl]) for name,values in signals.items()}
            best=max(measured[name] for name in ['ALE','AE','AET'])
            for name,value in measured.items():
                rows.append({'requested_snr_db':snr,'noise_seed':seed,'algorithm':name,'output_snr_db':value,
                    'best_original_output_snr_db':best,'passes_db_10_percent':value>best and value-best>=.1*abs(best)})
            np.savez_compressed(output/f'chirp_snr_{snr}_seed_{seed}.npz',clean=clean,noisy=noisy,noise=noise,time_s=times,**signals)
            print(f'Chirp {snr} seed {seed}: '+str({name:round(value,3) for name,value in measured.items()}),flush=True)
    e.b.save_csv(output/'metrics.csv',rows)
    e.b.save_json(output/'manifest.json',{'scope':'Nonstationary counterexample probe, not used for tuning',
        'frequency_hz':'80 + 5 * time_seconds','noise_seeds':[300,301],'snrs':[-10,0],
        'evaluation_interval':[8000,14976],'head_config':e.asdict(e.SpectralHeadConfig()),
        'source_sha256':hashlib.sha256((e.ROOT/'src/transformer_based_adaptive_line_enhancer/aet_improved.py').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=='__main__':main()
