"""Development-only sweep; clean references score candidates after training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import reproduce_comparison as b
from aet_epoch_sweep import compute_metrics
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer.aet_attention import (
    AttentionConfig,train_attention_aet,enhance_attention_aet,
)


VARIANTS={
    'lag8_wide':dict(patch_length=8,heads=1,top_k=512),
    'lag8_soft':dict(patch_length=8,heads=1,top_k=512,temperature=.1),
    'lag8_frozen':dict(patch_length=8,heads=1,top_k=512,attention='frozen'),
    'lag8_none':dict(patch_length=8,heads=1,attention='none'),
    'lag8_uniform':dict(patch_length=8,heads=1,attention='uniform'),
    'lag8_short':dict(patch_length=8,heads=1,top_k=512,max_lag_tokens=48),
    'lag8_residual':dict(patch_length=8,heads=1,top_k=512,noisy_residual=True),
    'lag8':dict(patch_length=8,heads=1,top_k=256),
    'lag4':dict(patch_length=4,heads=1,top_k=512),
    'lag16_soft':dict(patch_length=16,heads=1,top_k=256,temperature=.1),
    'lag16_sharp':dict(patch_length=16,heads=1,top_k=64,temperature=.01),
    'lag16_all':dict(patch_length=16,heads=1,top_k=512),
    'lag16_linear':dict(patch_length=16,heads=1,top_k=128,ffn=False),
    'lag16_none':dict(patch_length=16,heads=1,attention='none'),
    'lag16_uniform':dict(patch_length=16,heads=1,attention='uniform'),
    'lag16_frozen':dict(patch_length=16,heads=1,top_k=128,attention='frozen'),
    'lag32_h1':dict(patch_length=32,heads=1,top_k=64),
    'lag32_h4':dict(patch_length=32,heads=4,top_k=64),
    'lag16_h1':dict(patch_length=16,heads=1,top_k=128),
    'lag64_h1':dict(patch_length=64,heads=1,top_k=64),
    'lag32_linear':dict(patch_length=32,heads=1,top_k=64,ffn=False),
    'lag32_sharp':dict(patch_length=32,heads=1,top_k=32,temperature=.01),
    'no_attention':dict(patch_length=32,heads=1,attention='none'),
    'uniform':dict(patch_length=32,heads=1,attention='uniform'),
    'frozen':dict(patch_length=32,heads=1,attention='frozen'),
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS)[:4])
    parser.add_argument('--snrs',nargs='+',type=int,default=[-20,0])
    parser.add_argument('--scenes',nargs='+',default=['single','multiline'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--updates',type=int,default=400)
    parser.add_argument('--output',type=Path,default=b.ROOT/'results/aet_transformer_2026-10-07/exploration_v1')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    source=b.ROOT/'src/transformer_based_adaptive_line_enhancer/aet_attention.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output/'source_snapshot.py').write_bytes(source.read_bytes())
    b.save_json(args.output/'manifest.json',{'source_sha256':digest,'variants':args.variants,
        'snrs':args.snrs,'seeds':args.seeds,'scenes':args.scenes,'updates':args.updates,
        'role':'development data; not a locked test'})
    rows=[]
    for scene in args.scenes:
        for snr in args.snrs:
            for seed in args.seeds:
                label=f'{scene}_snr_{snr}_seed_{seed}'
                with np.load(b.ROOT/'results/reproduction_2026-10-07/cases'/label/'signals.npz') as archive:
                    data={key:archive[key].copy() for key in archive.files}
                with np.load(b.ROOT/'results/aet_early_stopping_2026-10-07/cases'/label/'validation_observations.npz') as archive:
                    validation=archive['noisy'].copy()
                baseline=max(compute_metrics(data[name],data,scene)['output_snr_db'] for name in ['ALE','AE','AET'])
                for variant in args.variants:
                    case=args.output/f'{label}_{variant}';case.mkdir(exist_ok=True)
                    marker=case/'metrics.json'
                    if marker.exists():
                        saved=json.loads(marker.read_text())
                        if saved['source_sha256']!=digest:raise ValueError('Use a fresh output directory after source changes')
                        rows.append(saved['metrics']);continue
                    config=AttentionConfig(seed=seed-42,updates=args.updates,**VARIANTS[variant])
                    started=time.perf_counter()
                    trained=train_attention_aet(data['noisy'],validation,config)
                    prediction,end=enhance_attention_aet(data['noisy'],trained)
                    assert end>=int(data['evaluation_end'])
                    measured=compute_metrics(prediction,data,scene)
                    row={'scene':scene,'requested_snr_db':snr,'seed':seed,'variant':variant,
                        'selected_update':trained['selected_update'],'stopped_update':trained['stopped_update'],
                        'best_baseline_snr_db':baseline,**measured,
                        'margin_db':measured['output_snr_db']-baseline-.1*abs(baseline),
                        'seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in trained['model'].parameters())}
                    rows.append(row)
                    b.save_json(marker,{'source_sha256':digest,'config':asdict(config),'metrics':row})
                    b.save_csv(case/'history.csv',trained['history'])
                    np.savez_compressed(case/'prediction.npz',enhanced=prediction)
                    torch.save({'state_dict':trained['model'].state_dict(),'config':asdict(config),
                        'mean':trained['mean'],'scale':trained['scale'],'selected_update':trained['selected_update']},case/'checkpoint.pt')
                    print(f'{label} {variant}: out={measured["output_snr_db"]:.3f} baseline={baseline:.3f} '
                          f'margin={row["margin_db"]:.3f} amp={measured["strongest_tone_retention"]:.3f}/{measured["weakest_tone_retention"]:.3f} '
                          f'update={trained["selected_update"]}/{trained["stopped_update"]} {row["seconds"]:.1f}s',flush=True)
    b.save_csv(args.output/'metrics.csv',rows)


if __name__=='__main__':main()
