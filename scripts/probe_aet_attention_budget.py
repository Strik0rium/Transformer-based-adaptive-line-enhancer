"""Post-hoc diagnosis of an update cap; this is not an independent test."""
from dataclasses import asdict
import json

import evaluate_aet_attention as e
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer.aet_attention import AttentionConfig,train_attention_aet,enhance_attention_aet


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=e.OUT/'offgrid_budget_diagnostic';out.mkdir(exist_ok=True)
    rows=[]
    for marker in sorted((e.OUT/'offgrid/cases').glob('*/metrics.json')):
        saved=json.loads(marker.read_text(encoding='utf-8'))
        with np.load(marker.parent/'signals.npz') as archive:data={k:archive[k].copy() for k in archive.files}
        with np.load(marker.parent/'validation_observations.npz') as archive:validation=archive['noisy'].copy()
        row=next(r for r in saved['metrics'] if r['algorithm']=='AET_LA')
        trained=train_attention_aet(data['noisy'],validation,AttentionConfig(seed=row['model_seed'],updates=1200))
        prediction,_=enhance_attention_aet(data['noisy'],trained)
        measured=e.metrics(prediction,data,saved['scene_config'])
        baseline=saved['comparisons'][0]['baseline_snr_db'];difference=measured['output_snr_db']-baseline
        result={k:row[k] for k in ('scene','requested_snr_db','noise_seed','model_seed')}
        result.update({'before_output_snr_db':row['output_snr_db'],**measured,'selected_update':trained['selected_update'],
            'stopped_update':trained['stopped_update'],'baseline_snr_db':baseline,
            'margin_db':difference-.1*abs(baseline),'passes_10_percent':difference>0 and difference>=.1*abs(baseline)})
        rows.append(result)
        case=out/marker.parent.name;case.mkdir(exist_ok=True)
        e.b.save_csv(case/'history.csv',trained['history'])
        e.b.save_json(case/'metrics.json',{'role':'post-hoc budget diagnosis, not held-out evidence','config':asdict(trained['config']),'metrics':result})
        torch.save(e.checkpoint(trained),case/'checkpoint.pt');np.savez_compressed(case/'prediction.npz',enhanced=prediction)
        print(f'{marker.parent.name}: {row["output_snr_db"]:.3f} -> {measured["output_snr_db"]:.3f}, margin {result["margin_db"]:.3f}, step {trained["selected_update"]}',flush=True)
    e.b.save_csv(out/'metrics.csv',rows)


if __name__=='__main__':main()
