"""Final locked validation of physically aligned Transformer queries.

Cases are independent processes (no agents); each receives two CPU threads.
Fixed frequencies use seeds 800..804; random frequencies use seeds 900..904.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import evaluate_aet_attention as e
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer import aet_aligned_attention as aligned
from transformer_based_adaptive_line_enhancer import aet_attention as unaligned

OUT=e.OUT/'final_validation'
CONTROLS={'AET_frozen_qk':{'attention':'frozen'},'AET_short_attention':{'max_lag_tokens':48},'AET_no_attention':{'attention':'none'}}


def initialize_worker():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)


def run_case(spec):
    domain,scene,snr,seed,protocol_hash=spec
    model_seed=seed+2000
    case=OUT/'cases'/f'{domain}_{scene}_snr_{snr}_seed_{seed}';case.mkdir(parents=True,exist_ok=True)
    marker=case/'metrics.json'
    if marker.exists():
        saved=json.loads(marker.read_text(encoding='utf-8'));assert saved['protocol_hash']==protocol_hash;return saved
    started=time.perf_counter()
    scene_config=dict(e.b.SCENES[scene])
    if domain=='random':
        rng=np.random.default_rng(900000+seed)
        ranges=[(70,130)] if scene=='single' else [(35,65),(85,120),(160,205)]
        scene_config['frequencies_hz']=[float(rng.uniform(*r)) for r in ranges]
        scene_config['phases_rad']=rng.uniform(-np.pi,np.pi,len(ranges)).tolist()
    sample=e.generate_noisy_signal(**scene_config,num_samples=16000,sample_rate_hz=1000.,snr_db=snr,seed=seed,scaling='power-normalized')
    data={'time_s':sample.time_s,'noisy':sample.noisy_signal,'noise':sample.noise,'clean':sample.clean_signal,
        'evaluation_start':8000,'evaluation_end':14976}
    validation_seeds=[seed+10000*(i+1) for i in range(3)]
    validation=np.stack([sample.clean_signal+np.random.default_rng(s).normal(0,np.sqrt(np.mean(sample.noise**2)),16000) for s in validation_seeds])
    signals={'ALE':e.adaptive_line_enhance(sample.noisy_signal).enhanced_signal}
    ae=e.train_ae_dle(sample.noisy_signal,config=e.AEDLEConfig(epochs=60,seed=model_seed))
    signals['AE']=e.enhance_with_ae_dle(sample.noisy_signal,ae).enhanced_signal
    aet=e.train_aet_dle(sample.noisy_signal,config=e.AETDLEConfig(epochs=60,seed=model_seed))
    signals['AET_60']=e.enhance_with_aet_dle(sample.noisy_signal,aet).enhanced_signal
    es,history,selection=e.train_with_early_stopping(sample.noisy_signal,validation,e.AETDLEConfig(epochs=120,seed=model_seed),10,1e-4)
    signals['AET_ES']=e.enhance_with_aet_dle(sample.noisy_signal,es).enhanced_signal
    e.b.save_csv(case/'original_aet_es_history.csv',history)
    selections={'AET_ES':selection}
    for name,changes in {'AET_LA':{},**CONTROLS,'AET_old_queries':{}}.items():
        module=unaligned if name=='AET_old_queries' else aligned
        config=replace(module.AttentionConfig(updates=1200),seed=model_seed,**changes)
        trained=module.train_attention_aet(sample.noisy_signal,validation,config)
        signals[name],end=module.enhance_attention_aet(sample.noisy_signal,trained)
        assert end>=14976
        selections[name]={'selected_update':trained['selected_update'],'stopped_update':trained['stopped_update'],
            'best_validation_loss':trained['best_validation_loss']}
        e.b.save_csv(case/f'{name}_history.csv',trained['history'])
        state=e.checkpoint(trained);state['implementation']='unaligned' if name=='AET_old_queries' else 'aligned'
        torch.save(state,case/f'{name}_checkpoint.pt')
        replay,_=module.enhance_attention_aet(sample.noisy_signal,module.restore_attention_aet(torch.load(case/f'{name}_checkpoint.pt',weights_only=True)))
        np.testing.assert_array_equal(replay,signals[name])
    rows=[{'domain':domain,'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'model_seed':model_seed,'algorithm':name,
        **e.metrics(values,data,scene_config),'selected_update':selections.get(name,{}).get('selected_update',0)} for name,values in signals.items()]
    baseline=max((r for r in rows if r['algorithm'] in e.BASELINES),key=lambda r:r['output_snr_db'])
    comparisons=[]
    for row in rows:
        if row['algorithm'] in e.BASELINES:continue
        difference=row['output_snr_db']-baseline['output_snr_db'];denominator=abs(baseline['output_snr_db'])
        comparisons.append({'domain':domain,'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'algorithm':row['algorithm'],
            'best_baseline':baseline['algorithm'],'baseline_snr_db':baseline['output_snr_db'],'difference_db':difference,
            'relative_db_percent':100*difference/denominator,'margin_db':difference-.1*denominator,
            'passes_10_percent':difference>0 and difference>=.1*denominator})
    np.savez_compressed(case/'signals.npz',**data,**signals)
    np.savez_compressed(case/'validation_observations.npz',noisy=validation,noise_seeds=validation_seeds)
    saved={'protocol_hash':protocol_hash,'scene_config':scene_config,'metrics':rows,'comparisons':comparisons,
        'selections':selections,'validation_noise_seeds':validation_seeds,'wall_seconds':time.perf_counter()-started}
    e.b.save_json(marker,saved)
    return saved


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=list((e.b.ROOT/'src/transformer_based_adaptive_line_enhancer').glob('*.py'))+[Path(__file__),Path(e.__file__),e.b.ROOT/'scripts/compare_aet_early_stopping.py']
    protocol={'domain_seeds':{'fixed':list(range(800,805)),'random':list(range(900,905))},
        'scenes':['single','multiline'],'snrs':[-20,-15,-10,-5,0],'attention_config':asdict(aligned.AttentionConfig()),
        'baseline_epochs':60,'baseline_es':{'max_epochs':120,'patience':10,'relative_delta':1e-4},
        'evaluation_interval':[8000,14976],'model_seed_offset':2000,'validation_noise_seed_offsets':[10000,20000,30000],
        'controls':CONTROLS,'additional_control':'same architecture and budget with old misaligned queries',
        'criterion':'output difference > 0 and >= 0.1*abs(best of ALE,AE,AET_60,AET_ES dB)',
        'role':'Final held-out test; architecture developed on seeds 42-44,600-601,700-704. No tuning on these seeds.',
        'source_hashes':{str(p.relative_to(e.b.ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    protocol['protocol_hash']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (OUT/'manifest.json').exists():
        manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['protocol_hash']==protocol['protocol_hash']
    else:
        manifest={**protocol,'locked_at_utc':datetime.now(timezone.utc).isoformat()};e.b.save_json(OUT/'manifest.json',manifest)
    specs=[(domain,scene,snr,seed,manifest['protocol_hash']) for domain,seeds in manifest['domain_seeds'].items()
        for scene in manifest['scenes'] for snr in manifest['snrs'] for seed in seeds]
    rows=[];comparisons=[];done=0
    with ProcessPoolExecutor(max_workers=4,initializer=initialize_worker) as pool:
        futures=[pool.submit(run_case,spec) for spec in specs]
        for future in as_completed(futures):
            saved=future.result();rows.extend(saved['metrics']);comparisons.extend(saved['comparisons']);done+=1
            primary=saved['comparisons'][0];result=next(r for r in saved['metrics'] if r['algorithm']=='AET_LA')
            print(f'{done}/100 {result["domain"]} {result["scene"]} {result["requested_snr_db"]} seed={result["noise_seed"]}: '
                f'LA={result["output_snr_db"]:.3f}, margin={primary["margin_db"]:.3f}, pass={primary["passes_10_percent"]}',flush=True)
            key=lambda r:(r['domain'],r['scene'],r['requested_snr_db'],r['noise_seed'],r['algorithm'])
            e.b.save_csv(OUT/'metrics.csv',sorted(rows,key=key));e.b.save_csv(OUT/'comparisons.csv',sorted(comparisons,key=key))
    summary={}
    for domain in manifest['domain_seeds']:
        summary[domain]={}
        for name in sorted({r['algorithm'] for r in comparisons}):
            group=[r for r in comparisons if r['domain']==domain and r['algorithm']==name]
            summary[domain][name]={'cases':len(group),'passed':sum(r['passes_10_percent'] for r in group),
                'min_margin_db':min(r['margin_db'] for r in group),'min_relative_db_percent':min(r['relative_db_percent'] for r in group)}
    e.b.save_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
