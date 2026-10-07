"""Development study of Q/K physical-time alignment after transfer failures."""
import json
from dataclasses import asdict
import hashlib

import evaluate_aet_attention as e
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer.aet_aligned_attention import AttentionConfig,train_attention_aet,enhance_attention_aet


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=e.OUT/'aligned_development';out.mkdir(exist_ok=True)
    source=e.b.ROOT/'src/transformer_based_adaptive_line_enhancer/aet_aligned_attention.py'
    e.b.save_json(out/'manifest.json',{'role':'development only; transfer seeds reused after failure','config':asdict(AttentionConfig()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    rows=[]
    for phase in ('offgrid','transfer'):
        for marker in sorted((e.OUT/phase/'cases').glob('*/metrics.json')):
            saved=json.loads(marker.read_text(encoding='utf-8'))
            with np.load(marker.parent/'signals.npz') as a:data={k:a[k].copy() for k in a.files}
            with np.load(marker.parent/'validation_observations.npz') as a:validation=a['noisy'].copy()
            before=next(r for r in saved['metrics'] if r['algorithm']=='AET_LA')
            trained=train_attention_aet(data['noisy'],validation,AttentionConfig(seed=before['model_seed']))
            enhanced,_=enhance_attention_aet(data['noisy'],trained)
            measured=e.metrics(enhanced,data,saved['scene_config'])
            base=saved['comparisons'][0]['baseline_snr_db'];margin=measured['output_snr_db']-base-.1*abs(base)
            row={k:before[k] for k in ['scene','requested_snr_db','noise_seed','model_seed']}
            row.update({'before_snr_db':before['output_snr_db'],**measured,'baseline_snr_db':base,'margin_db':margin,
                'selected_update':trained['selected_update'],'stopped_update':trained['stopped_update']})
            rows.append(row);case=out/marker.parent.name;case.mkdir(exist_ok=True)
            e.b.save_json(case/'metrics.json',row);e.b.save_csv(case/'history.csv',trained['history'])
            torch.save(e.checkpoint(trained),case/'checkpoint.pt');np.savez_compressed(case/'prediction.npz',enhanced=enhanced)
            print(f'{marker.parent.name}: {before["output_snr_db"]:.2f} -> {measured["output_snr_db"]:.2f}, margin {margin:.2f}',flush=True)
    e.b.save_csv(out/'metrics.csv',rows)


if __name__=='__main__':main()
