"""Tuning/held-out comparison of regularized AET and a global spectral head."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import reproduce_comparison as b
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer import (
    generate_noisy_signal,adaptive_line_enhance,train_ae_dle,enhance_with_ae_dle,
    train_aet_dle,enhance_with_aet_dle,AEDLEConfig,AETDLEConfig,
)
from transformer_based_adaptive_line_enhancer.aet_improved import (
    SpectralHeadConfig,spectral_residual_head,train_regularized_aet,
)

ROOT=b.ROOT
BASE=ROOT/'results/reproduction_2026-10-07'
OUT=ROOT/'results/aet_improvements_2026-10-07'
ALGORITHMS=['ALE','AE','AET','AET_spectral','spectral_only']


def measure(estimate,data,frequencies):
    sl=slice(int(data['evaluation_start']),int(data['evaluation_end']))
    clean=data['clean'][sl];prediction=estimate[sl]
    truth=b.tone_fit(clean,data['time_s'][sl],frequencies)
    fit=b.tone_fit(prediction,data['time_s'][sl],frequencies)
    ratios=[float(np.hypot(*fit[1+2*i:3+2*i])/np.hypot(*truth[1+2*i:3+2*i])) for i in range(len(frequencies))]
    return {'output_snr_db':b.snr_db(clean,prediction),'mse':float(np.mean((prediction-clean)**2)),
        'input_snr_db':b.snr_db(clean,data['noisy'][sl]),
        'strongest_tone_retention':ratios[0],'weakest_tone_retention':ratios[-1]}


def run_case(scene,snr,seed,model_seed,args,protocol_hash):
    name=f'{scene}_snr_{snr:g}_seed_{seed}'
    case=args.output/args.phase/'cases'/name;case.mkdir(parents=True,exist_ok=True)
    if args.resume and (case/'metrics.json').exists():
        payload=json.loads((case/'metrics.json').read_text(encoding='utf-8'))
        if payload['protocol_hash']!=protocol_hash:raise ValueError(f'Protocol mismatch {name}')
        print('REUSE '+name,flush=True);return payload
    source=BASE/'cases'/name
    if args.phase=='pilot':
        frequencies=b.SCENES[scene]['frequencies_hz']
        with np.load(source/'signals.npz') as archive:
            data={key:archive[key].copy() for key in archive.files}
        signals={key:data[key] for key in ['ALE','AE','AET']}
        with np.load(ROOT/'results/aet_early_stopping_2026-10-07/cases'/name/'signals.npz') as archive:
            signals['AET_early_stop']=archive['AET_ES'].copy()
        metadata={'scene_config':b.SCENES[scene],'baseline':'Reused original 60-epoch outputs',
                  'source_sha256':hashlib.sha256((source/'signals.npz').read_bytes()).hexdigest()}
    else:
        scene_config=dict(b.SCENES[scene])
        if args.phase=='offgrid':
            # Different latent frequencies/phases/amplitudes, selected before outputs.
            rng=np.random.default_rng(700000+seed)
            if scene=='single':scene_config['frequencies_hz']=[float(rng.uniform(70,130))]
            else:scene_config['frequencies_hz']=[float(rng.uniform(a,c)) for a,c in [(35,65),(85,120),(160,205)]]
            scene_config['amplitudes']=[float(a*rng.uniform(.8,1.2)) for a in b.SCENES[scene]['amplitudes']]
            scene_config['phases_rad']=rng.uniform(-np.pi,np.pi,len(scene_config['amplitudes'])).tolist()
        frequencies=scene_config['frequencies_hz']
        sample=generate_noisy_signal(**scene_config,num_samples=16000,sample_rate_hz=1000.,snr_db=snr,seed=seed,scaling='power-normalized')
        ale=adaptive_line_enhance(sample.noisy_signal)
        ae_train=train_ae_dle(sample.noisy_signal,config=AEDLEConfig(epochs=60,seed=model_seed))
        ae=enhance_with_ae_dle(sample.noisy_signal,ae_train)
        aet_train=train_aet_dle(sample.noisy_signal,config=AETDLEConfig(epochs=60,seed=model_seed))
        aet=enhance_with_aet_dle(sample.noisy_signal,aet_train)
        data={'time_s':sample.time_s,'clean':sample.clean_signal,'noisy':sample.noisy_signal,'noise':sample.noise,
              'evaluation_start':8000,'evaluation_end':min(ae.valid_end,aet.valid_end)}
        signals={'ALE':ale.enhanced_signal,'AE':ae.enhanced_signal,'AET':aet.enhanced_signal}
        metadata={'scene_config':scene_config,'baseline':'Fresh original 60-epoch AE/AET, same observation',
                  'original_aet_valid_end':aet.valid_end}
        # Save deployment weights for one held-out representative at every condition.
        if seed==args.seeds[0]:
            torch.save({'state_dict':aet_train.model.state_dict(),'config':asdict(aet_train.config),
                'scaler':asdict(aet_train.scaler),'loss_history':aet_train.loss_history,
                'spectral_head_config':asdict(SpectralHeadConfig()),'protocol_hash':protocol_hash},case/'AET_spectral_checkpoint.pt')
    started=time.perf_counter()
    head=spectral_residual_head(data['noisy'],signals['AET'],neural_valid_slice=slice(0,int(data['evaluation_end'])))
    head_seconds=time.perf_counter()-started
    signals['AET_spectral']=head['enhanced_signal'];signals['spectral_only']=head['spectral_only']
    regularized_selection={}
    if args.regularized:
        with np.load(ROOT/'results/aet_early_stopping_2026-10-07/cases'/name/'validation_observations.npz') as archive:
            validation=archive['noisy'].copy()
        trained,history,regularized_selection=train_regularized_aet(data['noisy'],validation,seed=model_seed)
        signals['AET_regularized']=enhance_with_aet_dle(data['noisy'],trained).enhanced_signal
        b.save_csv(case/'regularized_history.csv',history)
        torch.save({'state_dict':trained.model.state_dict(),'config':asdict(trained.config),
            'scaler':asdict(trained.scaler),'selection':regularized_selection},case/'AET_regularized_checkpoint.pt')
    rows=[]
    for name,values in signals.items():
        metrics=measure(values,data,frequencies)
        rows.append({'phase':args.phase,'scene':scene,'requested_snr_db':snr,'noise_seed':seed,
                     'model_seed':model_seed,'algorithm':name,**metrics,
                     'evaluation_start':int(data['evaluation_start']),'evaluation_end':int(data['evaluation_end'])})
    # Comparator includes original AET as an additional, stronger requirement.
    baseline=max((r for r in rows if r['algorithm'] in ['ALE','AE','AET','AET_early_stop']),key=lambda r:r['output_snr_db'])
    comparisons=[]
    for row in rows:
        if row['algorithm'] in ['ALE','AE','AET','AET_early_stop']:continue
        difference=row['output_snr_db']-baseline['output_snr_db']
        denominator=abs(baseline['output_snr_db'])
        comparisons.append({'phase':args.phase,'scene':scene,'requested_snr_db':snr,'noise_seed':seed,
            'algorithm':row['algorithm'],'best_baseline':baseline['algorithm'],
            'baseline_output_snr_db':baseline['output_snr_db'],'output_snr_db':row['output_snr_db'],
            'snr_difference_db':difference,'required_db_improvement':.1*denominator,
            'relative_db_improvement_percent':100*difference/denominator if denominator>1e-10 else None,
            'passes_db_10_percent':difference>0 and difference>=.1*denominator,
            'mse_reduction_percent':100*(1-row['mse']/baseline['mse'])})
    np.savez_compressed(case/'signals.npz',**(data | signals))
    metadata.update({'head_frequencies_hz':head['frequencies_hz'],'head_noise_floor':head['noise_floor'],
        'head_threshold':head['detection_threshold'],'head_seconds':head_seconds,
        'regularized_selection':regularized_selection})
    payload={'protocol_hash':protocol_hash,'metrics':rows,'comparisons':comparisons,'metadata':metadata}
    b.save_json(case/'metrics.json',payload)
    selected=next(r for r in comparisons if r['algorithm']=='AET_spectral')
    print(f'{scene} {snr:g} seed {seed}: AET+head={selected["output_snr_db"]:.2f} '
          f'baseline={selected["baseline_output_snr_db"]:.2f}, margin={selected["snr_difference_db"]-selected["required_db_improvement"]:.2f}, '
          f'lines={[round(f,4) for f in head["frequencies_hz"]]}',flush=True)
    return payload


def summarize(rows,scenes,snrs,names):
    summary=[]
    for scene in scenes:
        for snr in snrs:
            for name in names:
                chosen=[r for r in rows if r['scene']==scene and r['requested_snr_db']==snr and r['algorithm']==name]
                if not chosen:continue
                row={'scene':scene,'requested_snr_db':snr,'algorithm':name,'samples':len(chosen)}
                for field in ['output_snr_db','mse','weakest_tone_retention','strongest_tone_retention']:
                    values=[r[field] for r in chosen]
                    row[field+'_mean']=float(np.mean(values));row[field+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
                summary.append(row)
    return summary


def plots(summary,comparisons,args):
    names=ALGORITHMS+(['AET_regularized'] if args.regularized else [])
    colors={'ALE':'#2878b5','AE':'#e58b27','AET':'#888888','AET_spectral':'#26985d','spectral_only':'#9851a5','AET_regularized':'#cb343e'}
    fig,axes=plt.subplots(2,len(args.scenes),figsize=(7*len(args.scenes),8),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        for name in names:
            records=[r for r in summary if r['scene']==scene and r['algorithm']==name]
            for i,field in enumerate(['output_snr_db','weakest_tone_retention']):
                factor=100 if i else 1
                axes[i,j].errorbar(args.snrs,[factor*r[field+'_mean'] for r in records],
                    yerr=[factor*r[field+'_std'] for r in records],label=name,color=colors[name],
                    marker='o',capsize=3,linestyle='--' if name=='spectral_only' else '-')
        axes[1,j].axhline(100,linestyle=':',color='gray')
        for i,ylabel in enumerate(['Output SNR (dB)','Weakest tone retention (%)']):
            axes[i,j].set(title=scene,xlabel='Requested input SNR (dB)',ylabel=ylabel)
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    fig.suptitle(f'{args.phase} | mean +/- sample SD ({len(args.seeds)} independent noise/model seeds)')
    for extension in ['png','svg']:fig.savefig(args.output/args.phase/f'comparison.{extension}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,len(args.scenes),figsize=(7*len(args.scenes),4.5),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        for name in ['AET_spectral']+(['AET_regularized'] if args.regularized else []):
            chosen=[r for r in comparisons if r['scene']==scene and r['algorithm']==name]
            for snr in args.snrs:
                subset=[r for r in chosen if r['requested_snr_db']==snr]
                margins=[r['snr_difference_db']-r['required_db_improvement'] for r in subset]
                axes[0,j].scatter([snr]*len(margins),margins,color=colors[name],alpha=.5,s=22)
            axes[0,j].plot(args.snrs,[min(r['snr_difference_db']-r['required_db_improvement'] for r in chosen if r['requested_snr_db']==snr) for snr in args.snrs],
                           '-o',color=colors[name],label=name+' minimum')
        axes[0,j].axhline(0,color='black',linestyle='--',label='Pass boundary')
        axes[0,j].set(title=scene,xlabel='Requested input SNR (dB)',ylabel='SNR improvement - required 10% dB improvement (dB)')
        axes[0,j].legend(fontsize=8);axes[0,j].grid(alpha=.2)
    fig.suptitle('Every test case must lie above zero to pass the requested criterion')
    fig.savefig(args.output/args.phase/'acceptance_margin.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['pilot','holdout','offgrid'],required=True)
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--snrs',nargs='+',type=float,default=[-20,-15,-10,-5,0])
    parser.add_argument('--scenes',nargs='+',default=list(b.SCENES),choices=list(b.SCENES))
    parser.add_argument('--regularized',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.seeds=args.seeds or ({'pilot':[42,43,44],'holdout':list(range(100,110)),'offgrid':[200,201,202]}[args.phase])
    if args.regularized and args.phase!='pilot':parser.error('Regularized candidate is tested in pilot only.')
    if args.phase!='pilot' and set(args.seeds)&{42,43,44}:parser.error('Held-out seed overlap with tuning.')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (args.output/args.phase).mkdir(parents=True,exist_ok=True)
    model_seeds=list(range(len(args.seeds))) if args.phase=='pilot' else [1000+s for s in args.seeds]
    manifest={'phase':args.phase,'noise_seeds':args.seeds,'model_seeds':model_seeds,
        'scenes':args.scenes,'snrs':args.snrs,'head_config':asdict(SpectralHeadConfig()),
        'regularized':args.regularized,'criterion':'new - best >= 0.10 * abs(best output SNR in dB), with positive improvement',
        'baseline':['ALE','AE_60','AET_60']+(['AET_early_stop'] if args.phase=='pilot' else []),
        'evaluation_interval':[8000,14976],'torch_version':torch.__version__,'numpy_version':np.__version__,
        'threads':2,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'src').rglob('*.py')},
        'latent_signal_randomization':args.phase=='offgrid',
        'all_test_configurations_frozen_before_evaluation':True}
    manifest['protocol_hash']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    b.save_json(args.output/args.phase/'manifest.json',manifest)
    rows,comparisons=[],[]
    for scene in args.scenes:
        for snr in args.snrs:
            for i,seed in enumerate(args.seeds):
                payload=run_case(scene,snr,seed,model_seeds[i],args,manifest['protocol_hash'])
                rows.extend(payload['metrics']);comparisons.extend(payload['comparisons'])
    names=ALGORITHMS+(['AET_regularized','AET_early_stop'] if args.regularized else [])
    summary=summarize(rows,args.scenes,args.snrs,names)
    b.save_csv(args.output/args.phase/'metrics.csv',rows)
    b.save_csv(args.output/args.phase/'comparisons.csv',comparisons)
    b.save_csv(args.output/args.phase/'summary.csv',summary)
    plots(summary,comparisons,args)
    acceptance={name:{'case_count':sum(r['algorithm']==name for r in comparisons),
        'passed_cases':sum(r['algorithm']==name and r['passes_db_10_percent'] for r in comparisons),
        'minimum_margin_db':min(r['snr_difference_db']-r['required_db_improvement'] for r in comparisons if r['algorithm']==name),
        'minimum_relative_db_improvement_percent':min(r['relative_db_improvement_percent'] for r in comparisons if r['algorithm']==name and r['relative_db_improvement_percent'] is not None)}
        for name in sorted({r['algorithm'] for r in comparisons})}
    b.save_json(args.output/args.phase/'acceptance.json',acceptance)
    print(json.dumps(acceptance,indent=2),flush=True)


if __name__=='__main__':main()
