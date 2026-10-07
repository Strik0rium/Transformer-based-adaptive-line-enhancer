"""Diagnose saved AET results: error projection, new noise, epoch sensitivity.

Known frequencies and clean signals are used only for external diagnostics.
The algorithm modules remain unchanged.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_based_adaptive_line_enhancer.ae_dle import (
    AEDLEConfig,AEDLETrainingResult,AutoencoderDLE,MinMaxScaler,enhance_with_ae_dle,
)
from transformer_based_adaptive_line_enhancer.aet_dle import (
    AETDLEConfig,AETDLETrainingResult,AutoencoderTransformerDLE,AETMinMaxScaler,
    enhance_with_aet_dle,train_aet_dle,
)

OUTPUT=ROOT / "results/reproduction_2026-10-07"
DIAGNOSTICS=OUTPUT / "diagnostics"


def write_csv(name: str,rows: list[dict]) -> None:
    with (DIAGNOSTICS/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def measure(data,name: str,values: np.ndarray) -> dict:
    sl=slice(int(data['evaluation_start']),int(data['evaluation_end']))
    t=data['time_s'][sl]; clean=data['clean'][sl]; y=values[sl]; noise=data['noise'][sl]
    frequencies=[100.] if name=='single' else [50.,100.,180.]
    basis=np.column_stack([fn(2*np.pi*f*t) for f in frequencies for fn in (np.sin,np.cos)])
    q,_=np.linalg.qr(basis)
    constant=np.ones(len(t)); constant-=q@(q.T@constant); constant/=np.linalg.norm(constant)
    error=y-clean; tonal=q@(q.T@error); dc=constant*(constant@error); other=error-tonal-dc
    power=float(np.mean(clean**2)); mse=float(np.mean(error**2))
    noise_other=noise-q@(q.T@noise)-constant*(constant@noise)
    np.testing.assert_allclose(np.mean(tonal**2)+np.mean(dc**2)+np.mean(other**2),mse,atol=1e-12)
    return {'output_snr_db':float(10*np.log10(power/mse)),
        'tone_error_over_signal':float(np.mean(tonal**2)/power),
        'dc_error_over_signal':float(np.mean(dc**2)/power),
        'other_error_over_signal':float(np.mean(other**2)/power),
        'other_error_fraction':float(np.mean(other**2)/mse),
        'non_tone_target_noise_correlation':float(np.corrcoef(other,noise_other)[0,1])}


def load_model(case: Path,name: str):
    saved=torch.load(case/f'{name}_checkpoint.pt',map_location='cpu',weights_only=True)
    if name=='AE':
        config=AEDLEConfig(**saved['config'])
        model=AutoencoderDLE(frame_length=config.frame_length,hidden_dim=config.hidden_dim,
                            bottleneck_dim=config.bottleneck_dim)
        model.load_state_dict(saved['state_dict'])
        return AEDLETrainingResult(model.eval(),MinMaxScaler(**saved['scaler']),config,tuple(saved['loss_history']))
    config=AETDLEConfig(**saved['config'])
    model=AutoencoderTransformerDLE(config); model.load_state_dict(saved['state_dict'])
    return AETDLETrainingResult(model.eval(),AETMinMaxScaler(**saved['scaler']),config,tuple(saved['loss_history']))


def main() -> None:
    DIAGNOSTICS.mkdir(exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    decompositions=[]
    for marker in sorted((OUTPUT/'cases').glob('*/metrics.json')):
        payload=json.loads(marker.read_text())
        history=json.loads((marker.parent/'loss_history.json').read_text())
        with np.load(marker.parent/'signals.npz') as data:
            for row in payload['metrics']:
                loss=history.get(row['algorithm'])
                decompositions.append({k:row[k] for k in ['scene','requested_snr_db','noise_seed','model_seed','algorithm']}
                    |measure(data,row['scene'],data[row['algorithm']])
                    |{'final_training_loss':loss[-1] if loss else '',
                      'last10_relative_loss_drop':(loss[-10]-loss[-1])/loss[-10] if loss else ''})
    write_csv('error_decomposition.csv',decompositions)
    summaries=[]
    for scene in ['single','multiline']:
        for snr in [-20,-15,-10,-5,0]:
            for algorithm in ['ALE','AE','AET']:
                selected=[r for r in decompositions if r['scene']==scene and r['requested_snr_db']==snr and r['algorithm']==algorithm]
                summary={'scene':scene,'requested_snr_db':snr,'algorithm':algorithm}
                for key in ['tone_error_over_signal','dc_error_over_signal','other_error_over_signal','other_error_fraction','non_tone_target_noise_correlation']:
                    summary[key]=float(np.mean([r[key] for r in selected]))
                summaries.append(summary)
    write_csv('error_decomposition_summary.csv',summaries)
    independent=[]
    for scene in ['single','multiline']:
        for snr in [-20,-10,0]:
            case=OUTPUT/'cases'/f'{scene}_snr_{snr}_seed_42'
            with np.load(case/'signals.npz') as archive:
                data={key:archive[key].copy() for key in archive.files}
            for name in ['AE','AET']:
                trained=load_model(case,name)
                enhancer=enhance_with_ae_dle if name=='AE' else enhance_with_aet_dle
                independent.append({'scene':scene,'requested_snr_db':snr,'algorithm':name,'noise_seed':42,
                    'observation':'training_observation','clipped_fraction':0.0}|measure(data,scene,data[name]))
                for seed in [1042,1043,1044]:
                    noise=np.random.default_rng(seed).normal(size=len(data['clean']))
                    noise*=np.sqrt(np.mean(data['noise']**2)/np.mean(noise**2))
                    noisy=data['clean']+noise
                    result=enhancer(noisy,trained)
                    independent.append({'scene':scene,'requested_snr_db':snr,'algorithm':name,'noise_seed':seed,
                        'observation':'independent_noise','clipped_fraction':float(np.mean((noisy<trained.scaler.minimum)|(noisy>trained.scaler.maximum)))}
                        |measure(data|{'noise':noise},scene,result.enhanced_signal))
    write_csv('independent_noise.csv',independent)
    epoch_rows=[]
    for scene,snr in [('single',-20),('single',0),('multiline',-20),('multiline',0)]:
        case=OUTPUT/'cases'/f'{scene}_snr_{snr}_seed_42'
        with np.load(case/'signals.npz') as archive:
            data={key:archive[key].copy() for key in archive.files}
        original=load_model(case,'AET')
        for epochs in [10,30,60,120]:
            if epochs==60:
                trained=original; enhanced=data['AET']
            else:
                config=AETDLEConfig(**(asdict(original.config)|{'epochs':epochs}))
                trained=train_aet_dle(data['noisy'],config=config)
                enhanced=enhance_with_aet_dle(data['noisy'],trained).enhanced_signal
            row={'scene':scene,'requested_snr_db':snr,'noise_seed':42,'model_seed':0,'epochs':epochs,
                 'final_training_loss':trained.loss_history[-1]}|measure(data,scene,enhanced)
            epoch_rows.append(row)
            write_csv('epoch_sensitivity.csv',epoch_rows)
            print(f"EPOCH {scene} {snr} {epochs}: output={row['output_snr_db']:.3f} loss={row['final_training_loss']:.6f}",flush=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for axis,(scene,snr) in zip(axes.flat,[('single',-20),('single',0),('multiline',-20),('multiline',0)]):
        selected=[r for r in epoch_rows if r['scene']==scene and r['requested_snr_db']==snr]
        axis.plot([r['epochs'] for r in selected],[r['output_snr_db'] for r in selected],'-o',color='#3d9d64',label='Output SNR')
        twin=axis.twinx()
        twin.plot([r['epochs'] for r in selected],[r['final_training_loss'] for r in selected],'--s',color='#9753a4',label='Noisy-target loss')
        axis.set(title=f'{scene}, input {snr} dB, seed 42',xlabel='Training epochs',ylabel='Output SNR (dB)')
        twin.set_ylabel('Normalized noisy-target MSE',color='#9753a4')
        axis.grid(alpha=0.2)
    fig.suptitle('AET epoch sensitivity: lower training loss need not mean better enhancement')
    fig.savefig(DIAGNOSTICS/'epoch_sensitivity.png',dpi=160); plt.close(fig)
    result={'error_projection_rows':len(decompositions),'independent_observation_rows':len(independent),
        'epoch_sensitivity_rows':len(epoch_rows),'note':'Epoch and independent-noise probes use only original noise_seed 42/model_seed 0; causal architecture claims require further matched ablations.'}
    (DIAGNOSTICS/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
