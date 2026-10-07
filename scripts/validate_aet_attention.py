"""Recompute persisted benchmark scores, selection rules and model predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import evaluate_aet_attention as e
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer.aet_attention import restore_attention_aet,enhance_attention_aet


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=e.OUT)
    args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    verification=[];max_metric_error=0.;max_replay_error=0.
    for phase in ('holdout','offgrid','transfer'):
        folder=args.root/phase
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        for relative,digest in manifest['source_hashes'].items():
            assert hashlib.sha256((e.b.ROOT/relative).read_bytes()).hexdigest()==digest,relative
        assert not set(manifest['noise_seeds'])&{42,43,44}
        expected=len(manifest['noise_seeds'])*len(manifest['snrs'])*len(manifest['scenes'])
        cases=sorted((folder/'cases').glob('*/metrics.json'))
        assert len(cases)==expected,(phase,len(cases),expected)
        metric_count=0;checkpoint_count=0
        for marker in cases:
            saved=json.loads(marker.read_text(encoding='utf-8'))
            assert saved['protocol_hash']==manifest['protocol_hash']
            with np.load(marker.parent/'signals.npz') as archive:
                data={key:archive[key].copy() for key in archive.files}
            with np.load(marker.parent/'validation_observations.npz') as archive:
                validation=archive['noisy'].copy()
                assert len(set(archive['noise_seeds']))==3
            for row in saved['metrics']:
                actual=e.metrics(data[row['algorithm']],data,saved['scene_config'])
                for field,value in actual.items():
                    error=abs(value-row[field]);max_metric_error=max(error,max_metric_error)
                    np.testing.assert_allclose(value,row[field],rtol=0,atol=1e-12)
                metric_count+=1
            baseline=max((r for r in saved['metrics'] if r['algorithm'] in e.BASELINES),key=lambda r:r['output_snr_db'])
            for comparison in saved['comparisons']:
                row=next(r for r in saved['metrics'] if r['algorithm']==comparison['algorithm'])
                difference=row['output_snr_db']-baseline['output_snr_db']
                assert comparison['best_baseline']==baseline['algorithm']
                assert comparison['passes_10_percent']==(difference>0 and difference>=.1*abs(baseline['output_snr_db']))
            for path in marker.parent.glob('*_checkpoint.pt'):
                name=path.name.removesuffix('_checkpoint.pt')
                trained=restore_attention_aet(torch.load(path,weights_only=True))
                replay,end=enhance_attention_aet(data['noisy'],trained)
                error=float(np.max(np.abs(replay-data[name])));max_replay_error=max(error,max_replay_error)
                np.testing.assert_array_equal(replay,data[name])
                assert end>=int(data['evaluation_end'])
                # The selected update is exactly the noisy-validation improvement rule.
                import csv
                history=list(csv.DictReader((marker.parent/f'{name}_history.csv').open(encoding='utf-8-sig')))
                best=float('inf');best_update=0
                for row in history:
                    value=float(row['validation_loss'])
                    if value<best-1e-6:best=value;best_update=int(row['update'])
                assert best_update==trained['selected_update']==saved['selections'][name]['selected_update']
                assert best==saved['selections'][name]['best_validation_loss']
                checkpoint_count+=1
            # Verify baseline generation and validation noise provenance from seeds.
            seed=saved['metrics'][0]['noise_seed'];snr=saved['metrics'][0]['requested_snr_db']
            sample=e.generate_noisy_signal(**saved['scene_config'],num_samples=16000,sample_rate_hz=1000.,snr_db=snr,seed=seed,scaling='power-normalized')
            np.testing.assert_array_equal(data['noisy'],sample.noisy_signal)
            for observation,noise_seed in zip(validation,saved['validation_noise_seeds']):
                reconstructed=data['clean']+np.random.default_rng(noise_seed).normal(0,np.sqrt(np.mean(data['noise']**2)),16000)
                np.testing.assert_array_equal(reconstructed,observation)
        verification.append({'phase':phase,'cases':len(cases),'metrics':metric_count,'checkpoint_replays':checkpoint_count})
    result={'phases':verification,'maximum_metric_error':max_metric_error,'maximum_checkpoint_replay_error':max_replay_error,
        'source_hashes_verified':True,'early_stopping_rules_verified':True,'noise_provenance_verified':True,
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    e.b.save_json(args.root/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
