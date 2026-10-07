"""Locked benchmark of native AET lag attention against unchanged algorithms.

All algorithms adapt to the same observation. No clean labels/frequencies enter
training. Extra independent noisy observations select checkpoints only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import reproduce_comparison as b
from compare_aet_early_stopping import train_with_early_stopping
from aet_epoch_sweep import compute_metrics
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer import (
    generate_noisy_signal,adaptive_line_enhance,train_ae_dle,enhance_with_ae_dle,
    train_aet_dle,enhance_with_aet_dle,AEDLEConfig,AETDLEConfig,
)
from transformer_based_adaptive_line_enhancer.aet_attention import (
    AttentionConfig,train_attention_aet,enhance_attention_aet,restore_attention_aet,
)

OUT=b.ROOT/'results/aet_transformer_2026-10-07'
BASELINES=('ALE','AE','AET_60','AET_ES')
CONTROLS={
    'AET_frozen_qk':dict(attention='frozen'),
    'AET_no_attention':dict(attention='none'),
    'AET_uniform_attention':dict(attention='uniform'),
    'AET_short_attention':dict(max_lag_tokens=48),
    'AET_noisy_residual':dict(noisy_residual=True),
}


def checkpoint(trained):
    return {'state_dict':trained['model'].state_dict(),'config':asdict(trained['config']),
        'mean':trained['mean'],'scale':trained['scale'],'selected_update':trained['selected_update']}


def metrics(values,data,scene_config):
    sl=slice(int(data['evaluation_start']),int(data['evaluation_end']))
    clean=data['clean'][sl];estimate=values[sl]
    frequencies=scene_config['frequencies_hz']
    truth=b.tone_fit(clean,data['time_s'][sl],frequencies)
    fit=b.tone_fit(estimate,data['time_s'][sl],frequencies)
    ratios=[float(np.hypot(*fit[1+2*i:3+2*i])/np.hypot(*truth[1+2*i:3+2*i])) for i in range(len(frequencies))]
    output_snr=b.snr_db(clean,estimate);input_snr=b.snr_db(clean,data['noisy'][sl])
    return {'output_snr_db':output_snr,'input_snr_db':input_snr,'snr_gain_db':output_snr-input_snr,
        'mse':float(np.mean((estimate-clean)**2)),'strongest_tone_retention':ratios[0],
        'weakest_tone_retention':ratios[-1]}


def run_case(scene,snr,seed,args,manifest):
    label=f'{scene}_snr_{snr}_seed_{seed}';case=args.output/'cases'/label
    case.mkdir(parents=True,exist_ok=True)
    marker=case/'metrics.json'
    if marker.exists():
        saved=json.loads(marker.read_text(encoding='utf-8'))
        assert saved['protocol_hash']==manifest['protocol_hash']
        return saved
    started=time.perf_counter();model_seed=seed+2000
    scene_config=dict(b.SCENES[scene])
    if args.phase=='offgrid':
        rng=np.random.default_rng(800000+seed)
        ranges=[(70,130)] if scene=='single' else [(35,65),(85,120),(160,205)]
        scene_config['frequencies_hz']=[float(rng.uniform(*r)) for r in ranges]
        scene_config['phases_rad']=rng.uniform(-np.pi,np.pi,len(ranges)).tolist()
    sample=generate_noisy_signal(**scene_config,num_samples=16000,sample_rate_hz=1000.,
        snr_db=snr,seed=seed,scaling='power-normalized')
    data={'time_s':sample.time_s,'noisy':sample.noisy_signal,'clean':sample.clean_signal,
        'noise':sample.noise,'evaluation_start':8000,'evaluation_end':14976}
    validation_seeds=[seed+10000*(i+1) for i in range(3)]
    sigma=float(np.sqrt(np.mean(sample.noise**2)))
    validation=np.stack([sample.clean_signal+np.random.default_rng(s).normal(0,sigma,16000) for s in validation_seeds])
    # Baselines are recomputed, without any new preprocessing or attention module.
    ale=adaptive_line_enhance(sample.noisy_signal)
    ae_train=train_ae_dle(sample.noisy_signal,config=AEDLEConfig(epochs=60,seed=model_seed))
    ae=enhance_with_ae_dle(sample.noisy_signal,ae_train)
    aet_train=train_aet_dle(sample.noisy_signal,config=AETDLEConfig(epochs=60,seed=model_seed))
    aet=enhance_with_aet_dle(sample.noisy_signal,aet_train)
    es_train,es_history,es_selection=train_with_early_stopping(sample.noisy_signal,validation,
        AETDLEConfig(epochs=120,seed=model_seed),10,1e-4)
    es=enhance_with_aet_dle(sample.noisy_signal,es_train)
    signals={'ALE':ale.enhanced_signal,'AE':ae.enhanced_signal,'AET_60':aet.enhanced_signal,'AET_ES':es.enhanced_signal}
    rows=[];selections={'AET_ES':es_selection}
    for name,values in signals.items():
        rows.append({'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'model_seed':model_seed,
            'algorithm':name,**metrics(values,data,scene_config),'selected_update':0,
            'seconds':0.,'parameters':(32 if name=='ALE' else sum(p.numel() for p in (ae_train.model if name=='AE' else aet_train.model).parameters()))})
    b.save_csv(case/'original_aet_es_history.csv',es_history)
    variants={'AET_LA':{}}
    if args.ablations:variants.update(CONTROLS)
    else:variants['AET_frozen_qk']=CONTROLS['AET_frozen_qk']
    for name,changes in variants.items():
        config=replace(AttentionConfig(),seed=model_seed,**changes)
        start=time.perf_counter();trained=train_attention_aet(sample.noisy_signal,validation,config)
        values,end=enhance_attention_aet(sample.noisy_signal,trained)
        assert end>=data['evaluation_end']
        signals[name]=values
        rows.append({'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'model_seed':model_seed,
            'algorithm':name,**metrics(values,data,scene_config),'selected_update':trained['selected_update'],
            'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in trained['model'].parameters())})
        selections[name]={'selected_update':trained['selected_update'],'stopped_update':trained['stopped_update'],
            'best_validation_loss':trained['best_validation_loss']}
        b.save_csv(case/f'{name}_history.csv',trained['history'])
        torch.save(checkpoint(trained),case/f'{name}_checkpoint.pt')
        replay,_=enhance_attention_aet(sample.noisy_signal,restore_attention_aet(torch.load(case/f'{name}_checkpoint.pt',weights_only=True)))
        np.testing.assert_array_equal(values,replay)
    baseline=max((r for r in rows if r['algorithm'] in BASELINES),key=lambda r:r['output_snr_db'])
    comparisons=[]
    for row in rows:
        if row['algorithm'] in BASELINES:continue
        difference=row['output_snr_db']-baseline['output_snr_db'];denominator=abs(baseline['output_snr_db'])
        comparisons.append({'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'algorithm':row['algorithm'],
            'best_baseline':baseline['algorithm'],'baseline_snr_db':baseline['output_snr_db'],
            'difference_db':difference,'relative_db_percent':100*difference/denominator if denominator>1e-9 else None,
            'margin_db':difference-.1*denominator,'passes_10_percent':difference>0 and difference>=.1*denominator})
    np.savez_compressed(case/'signals.npz',**data,**signals)
    np.savez_compressed(case/'validation_observations.npz',noisy=validation,noise_seeds=validation_seeds)
    saved={'protocol_hash':manifest['protocol_hash'],'scene_config':scene_config,'metrics':rows,
        'comparisons':comparisons,'selections':selections,'validation_noise_seeds':validation_seeds,
        'wall_seconds':time.perf_counter()-started}
    b.save_json(marker,saved)
    primary=comparisons[0];value=next(r['output_snr_db'] for r in rows if r['algorithm']=='AET_LA')
    print(f'{label}: LA={value:.3f}, best={baseline["algorithm"]} {baseline["output_snr_db"]:.3f}, '
        f'margin={primary["margin_db"]:.3f}, pass={primary["passes_10_percent"]}, {saved["wall_seconds"]:.1f}s',flush=True)
    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['holdout','offgrid'],default='holdout')
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(400,410)))
    p.add_argument('--snrs',nargs='+',type=int,default=[-20,-15,-10,-5,0])
    p.add_argument('--scenes',nargs='+',default=['single','multiline'])
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output is None:args.output=OUT/args.phase
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    paths=list((b.ROOT/'src/transformer_based_adaptive_line_enhancer').glob('*.py'))+[Path(__file__),b.ROOT/'scripts/compare_aet_early_stopping.py']
    protocol={'phase':args.phase,'noise_seeds':args.seeds,'model_seed_offset':2000,'snrs':args.snrs,
        'scenes':args.scenes,'ablations':args.ablations,'attention_config':asdict(AttentionConfig()),
        'baseline_epochs':60,'early_stopping':{'max_epochs':120,'patience':10,'relative_delta':1e-4},
        'evaluation_interval':[8000,14976],'criterion':'output difference >= 0.1*abs(best baseline dB) and > 0',
        'source_hashes':{str(path.relative_to(b.ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'selection':'Architecture fixed before this evaluation; no clean labels in training or early stopping',
        'torch':torch.__version__,'numpy':np.__version__}
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    protocol['protocol_hash']=digest
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['protocol_hash']!=digest:raise ValueError('Protocol/source changed; choose a fresh output directory')
    else:
        manifest={**protocol,'locked_at_utc':datetime.now(timezone.utc).isoformat()}
        b.save_json(manifest_path,manifest)
    rows=[];comparisons=[]
    for scene in args.scenes:
        for snr in args.snrs:
            for seed in args.seeds:
                saved=run_case(scene,snr,seed,args,manifest)
                rows.extend(saved['metrics']);comparisons.extend(saved['comparisons'])
                b.save_csv(args.output/'metrics.csv',rows);b.save_csv(args.output/'comparisons.csv',comparisons)
    counts={}
    for name in sorted({r['algorithm'] for r in comparisons}):
        selected=[r for r in comparisons if r['algorithm']==name]
        counts[name]={'cases':len(selected),'passed':sum(r['passes_10_percent'] for r in selected),
            'min_margin_db':min(r['margin_db'] for r in selected),
            'min_relative_db_percent':min(r['relative_db_percent'] for r in selected if r['relative_db_percent'] is not None)}
    b.save_json(args.output/'summary.json',counts)
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
