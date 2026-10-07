"""Fresh random-frequency tests after locking a larger training budget.

Seeds 600/601 diagnosed the 400-update cap; only seeds 700..704 are tested here.
The model/optimizer/attention are unchanged. The validation patience is unchanged.
"""
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import evaluate_aet_attention as e
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer.aet_attention import AttentionConfig,train_attention_aet,enhance_attention_aet


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=e.OUT/'transfer';out.mkdir(exist_ok=True)
    paths=list((e.b.ROOT/'src/transformer_based_adaptive_line_enhancer').glob('*.py'))+[Path(__file__),Path(e.__file__),e.b.ROOT/'scripts/compare_aet_early_stopping.py']
    manifest={'phase':'transfer','noise_seeds':list(range(700,705)),'snrs':[-20,-10,0],
        'scenes':['single','multiline'],'attention_config':asdict(AttentionConfig(updates=1200)),
        'source_hashes':{str(p.relative_to(e.b.ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'selection':'Only cap increased to 1200 based on seeds 600/601; architecture unchanged; fresh frequencies, noise and model seeds',
        'evaluation_interval':[8000,14976],'baseline_epochs':60,'validation_noise_seed_offsets':[10000,20000,30000]}
    manifest['protocol_hash']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():
        saved=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        assert saved['protocol_hash']==manifest['protocol_hash'];manifest=saved
    else:
        manifest['locked_at_utc']=datetime.now(timezone.utc).isoformat();e.b.save_json(out/'manifest.json',manifest)
    rows=[];comparisons=[]
    for scene in manifest['scenes']:
        for snr in manifest['snrs']:
            for seed in manifest['noise_seeds']:
                case=out/'cases'/f'{scene}_snr_{snr}_seed_{seed}';case.mkdir(parents=True,exist_ok=True)
                marker=case/'metrics.json'
                if marker.exists():
                    saved=json.loads(marker.read_text(encoding='utf-8'));assert saved['protocol_hash']==manifest['protocol_hash']
                    rows.extend(saved['metrics']);comparisons.extend(saved['comparisons']);continue
                started=time.perf_counter();model_seed=seed+2000
                rng=np.random.default_rng(800000+seed)
                ranges=[(70,130)] if scene=='single' else [(35,65),(85,120),(160,205)]
                scene_config={'frequencies_hz':[float(rng.uniform(*r)) for r in ranges],
                    'amplitudes':e.b.SCENES[scene]['amplitudes'],'phases_rad':rng.uniform(-np.pi,np.pi,len(ranges)).tolist()}
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
                trained=train_attention_aet(sample.noisy_signal,validation,AttentionConfig(seed=model_seed,updates=1200))
                signals['AET_LA'],_=enhance_attention_aet(sample.noisy_signal,trained)
                e.b.save_csv(case/'AET_LA_history.csv',trained['history']);torch.save(e.checkpoint(trained),case/'AET_LA_checkpoint.pt')
                local=[{'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'model_seed':model_seed,'algorithm':name,
                    **e.metrics(values,data,scene_config),'selected_update':trained['selected_update'] if name=='AET_LA' else 0} for name,values in signals.items()]
                baseline=max((r for r in local if r['algorithm'] in e.BASELINES),key=lambda r:r['output_snr_db'])
                difference=local[-1]['output_snr_db']-baseline['output_snr_db'];denominator=abs(baseline['output_snr_db'])
                comparison={'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'algorithm':'AET_LA',
                    'best_baseline':baseline['algorithm'],'baseline_snr_db':baseline['output_snr_db'],'difference_db':difference,
                    'relative_db_percent':100*difference/denominator,'margin_db':difference-.1*denominator,
                    'passes_10_percent':difference>0 and difference>=.1*denominator}
                saved={'protocol_hash':manifest['protocol_hash'],'scene_config':scene_config,'metrics':local,'comparisons':[comparison],
                    'selections':{'AET_ES':selection,'AET_LA':{'selected_update':trained['selected_update'],'stopped_update':trained['stopped_update'],'best_validation_loss':trained['best_validation_loss']}},
                    'validation_noise_seeds':validation_seeds,'wall_seconds':time.perf_counter()-started}
                np.savez_compressed(case/'signals.npz',**data,**signals);np.savez_compressed(case/'validation_observations.npz',noisy=validation,noise_seeds=validation_seeds)
                e.b.save_json(marker,saved);rows.extend(local);comparisons.append(comparison)
                print(f'{case.name}: LA={local[-1]["output_snr_db"]:.3f}, margin={comparison["margin_db"]:.3f}, pass={comparison["passes_10_percent"]}, step={trained["selected_update"]}',flush=True)
                e.b.save_csv(out/'metrics.csv',rows);e.b.save_csv(out/'comparisons.csv',comparisons)
    e.b.save_csv(out/'metrics.csv',rows);e.b.save_csv(out/'comparisons.csv',comparisons)
    summary={'cases':len(comparisons),'passed':sum(r['passes_10_percent'] for r in comparisons),
        'minimum_margin_db':min(r['margin_db'] for r in comparisons),'minimum_relative_db_percent':min(r['relative_db_percent'] for r in comparisons)}
    e.b.save_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
