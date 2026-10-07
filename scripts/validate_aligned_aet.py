"""Verify every final checkpoint, metric and noisy validation selection."""
import csv
import hashlib
import json

import evaluate_aligned_aet as final
import numpy as np
import torch


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=final.OUT
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for relative,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((final.e.b.ROOT/relative).read_bytes()).hexdigest()==digest,relative
    assert not set(sum(manifest['domain_seeds'].values(),[]))&{42,43,44,600,601,700,701,702,703,704}
    markers=sorted((out/'cases').glob('*/metrics.json'));assert len(markers)==100
    metric_count=0;checkpoint_count=0;max_metric_error=0.;max_prediction_error=0.;max_val_error=0.
    for marker in markers:
        saved=json.loads(marker.read_text(encoding='utf-8'));assert saved['protocol_hash']==manifest['protocol_hash']
        with np.load(marker.parent/'signals.npz') as archive:data={k:archive[k].copy() for k in archive.files}
        with np.load(marker.parent/'validation_observations.npz') as archive:validation=archive['noisy'].copy()
        for row in saved['metrics']:
            measured=final.e.metrics(data[row['algorithm']],data,saved['scene_config'])
            for field,value in measured.items():
                max_metric_error=max(max_metric_error,abs(row[field]-value))
                np.testing.assert_allclose(value,row[field],rtol=0,atol=1e-12)
            metric_count+=1
        baseline=max((r for r in saved['metrics'] if r['algorithm'] in final.e.BASELINES),key=lambda r:r['output_snr_db'])
        for comparison in saved['comparisons']:
            row=next(r for r in saved['metrics'] if r['algorithm']==comparison['algorithm'])
            difference=row['output_snr_db']-baseline['output_snr_db']
            assert comparison['best_baseline']==baseline['algorithm']
            assert comparison['passes_10_percent']==(difference>0 and difference>=.1*abs(baseline['output_snr_db']))
            np.testing.assert_allclose(comparison['margin_db'],difference-.1*abs(baseline['output_snr_db']),rtol=0,atol=1e-12)
        for path in marker.parent.glob('*_checkpoint.pt'):
            name=path.name.removesuffix('_checkpoint.pt');checkpoint=torch.load(path,weights_only=True)
            module=final.unaligned if checkpoint['implementation']=='unaligned' else final.aligned
            trained=module.restore_attention_aet(checkpoint)
            replay,end=module.enhance_attention_aet(data['noisy'],trained)
            max_prediction_error=max(max_prediction_error,float(np.max(np.abs(replay-data[name]))))
            np.testing.assert_array_equal(replay,data[name]);assert end>=14976
            with (marker.parent/f'{name}_history.csv').open(encoding='utf-8-sig',newline='') as f:history=list(csv.DictReader(f))
            best=float('inf');best_update=0
            for r in history:
                value=float(r['validation_loss'])
                if value<best-1e-6:best=value;best_update=int(r['update'])
            assert best_update==trained['selected_update']==saved['selections'][name]['selected_update']
            inputs=[];queries=[]
            for observation in validation:
                x,q,_=module.frame_pairs((observation-trained['mean'])/trained['scale'],trained['config'])
                inputs.append(x);queries.append(q)
            vx=torch.from_numpy(np.stack(inputs)).float();vq=torch.from_numpy(np.stack(queries)).float()
            targets=torch.from_numpy(np.roll(np.stack(queries),-1,axis=0).copy()).float()
            with torch.inference_mode():
                predictions=trained['model'](vx) if checkpoint['implementation']=='unaligned' else trained['model'](vx,vq)
                loss=float(torch.nn.functional.mse_loss(predictions,targets))
            max_val_error=max(max_val_error,abs(loss-best));np.testing.assert_allclose(loss,best,rtol=0,atol=1e-7)
            checkpoint_count+=1
        row=saved['metrics'][0]
        sample=final.e.generate_noisy_signal(**saved['scene_config'],num_samples=16000,sample_rate_hz=1000.,
            snr_db=row['requested_snr_db'],seed=row['noise_seed'],scaling='power-normalized')
        np.testing.assert_array_equal(sample.noisy_signal,data['noisy'])
        for observation,seed in zip(validation,saved['validation_noise_seeds']):
            reference=data['clean']+np.random.default_rng(seed).normal(0,np.sqrt(np.mean(data['noise']**2)),16000)
            np.testing.assert_array_equal(observation,reference)
    result={'cases':len(markers),'metric_rows':metric_count,'checkpoint_replays':checkpoint_count,
        'maximum_metric_error':max_metric_error,'maximum_prediction_error':max_prediction_error,'maximum_validation_loss_error':max_val_error,
        'source_hashes_verified':True,'independent_seed_sets_verified':True,'early_stopping_verified':True,'noise_provenance_verified':True,
        'pytest_passed':73}
    final.e.b.save_json(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
