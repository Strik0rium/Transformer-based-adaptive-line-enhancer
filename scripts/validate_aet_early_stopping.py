"""Recompute persisted early-stop metrics, selection decisions and validation loss."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import compare_aet_early_stopping as experiment
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer.aet_dle import (
    AETDLEConfig, AETMinMaxScaler, AutoencoderTransformerDLE, _sequence_pairs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=experiment.ROOT/'results/aet_early_stopping_2026-10-07')
    args=parser.parse_args()
    output=args.output
    torch.set_num_threads(2)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for path,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((experiment.ROOT/path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest()==manifest['script_sha256']
    for name,digest in manifest['helper_hashes'].items():
        assert hashlib.sha256((experiment.ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    rows=[];max_metric_difference=0.;max_validation_difference=0.;cases=0
    for scene in manifest['scenes']:
        for snr in manifest['snrs']:
            for seed in manifest['seeds']:
                label=f'{scene}_snr_{snr:g}_seed_{seed}'
                case=output/'cases'/label
                original=experiment.BASE/'cases'/label/'signals.npz'
                assert hashlib.sha256(original.read_bytes()).hexdigest()==manifest['baseline_input_hashes'][label]
                payload=json.loads((case/'metrics.json').read_text(encoding='utf-8'))
                assert payload['protocol_hash']==manifest['protocol_hash']
                with np.load(case/'signals.npz') as archive:
                    data={key:archive[key].copy() for key in archive.files}
                with np.load(original) as archive:
                    for name in ('clean','noisy','noise','ALE','AE'):
                        np.testing.assert_array_equal(data[name],archive[name])
                    np.testing.assert_array_equal(data['AET_fixed60'],archive['AET'])
                for row in payload['metrics']:
                    metrics=experiment.compute_metrics(data[row['algorithm']],data,scene)
                    for field,value in metrics.items():
                        difference=abs(value-row[field]);max_metric_difference=max(max_metric_difference,difference)
                        np.testing.assert_allclose(value,row[field],atol=1e-12,rtol=0)
                # Replay the predeclared rule independently from persisted loss values.
                best_loss=float('inf');best_epoch=0;stale=0
                for row in payload['history']:
                    improved=row['validation_mse']<best_loss*(1-manifest['relative_min_delta'])
                    if improved:
                        best_loss=row['validation_mse'];best_epoch=row['epoch'];stale=0
                    else: stale+=1
                    assert row['significant_improvement']==improved
                    assert row['best_epoch_so_far']==best_epoch and row['stale_epochs']==stale
                    assert row['epoch']==len(payload['history']) or stale<manifest['patience']
                selection=payload['selection']
                assert best_epoch==selection['selected_epoch']
                assert len(payload['history'])==selection['stopped_epoch']
                assert stale>=manifest['patience'] or len(payload['history'])==manifest['max_epochs']
                checkpoint=torch.load(case/'AET_ES_checkpoint.pt',weights_only=True)
                config=AETDLEConfig(**checkpoint['config'])
                assert config.epochs==best_epoch
                model=AutoencoderTransformerDLE(config);model.load_state_dict(checkpoint['state_dict']);model.eval()
                scaler=AETMinMaxScaler(**checkpoint['scaler'])
                val_inputs=[];val_targets=[]
                with np.load(case/'validation_observations.npz') as archive:
                    for observation,val_seed in zip(archive['noisy'],archive['noise_seeds'],strict=True):
                        assert val_seed not in manifest['seeds']
                        generated=data['clean']+np.random.default_rng(int(val_seed)).normal(
                            0,selection['validation_noise_sigma'],data['clean'].size)
                        np.testing.assert_array_equal(observation,generated)
                        inputs,_,_=_sequence_pairs(scaler.transform(observation,clip=True),config)
                        _,targets,_=_sequence_pairs(scaler.transform(observation),config)
                        val_inputs.append(inputs);val_targets.append(targets)
                actual=experiment.validation_mse(model,torch.from_numpy(np.concatenate(val_inputs)).float(),
                                                 torch.from_numpy(np.concatenate(val_targets)).float())
                max_validation_difference=max(max_validation_difference,abs(actual-best_loss))
                np.testing.assert_allclose(actual,best_loss,atol=1e-12,rtol=0)
                rows.extend(payload['metrics']);cases+=1
    summary=experiment.summarize(rows,SimpleNamespace(scenes=manifest['scenes'],snrs=manifest['snrs']))
    persisted=list(csv.DictReader((output/'summary.csv').open(encoding='utf-8-sig')))
    assert len(summary)==len(persisted)
    for expected,saved in zip(summary,persisted,strict=True):
        for field,value in expected.items():
            if isinstance(value,str):assert value==saved[field]
            else:np.testing.assert_allclose(value,float(saved[field]),rtol=0,atol=1e-12)
    figures=list(output.glob('*.png'))
    for figure in figures:
        pixels=experiment.plt.imread(figure)
        assert pixels.ndim==3 and pixels.shape[0]>100 and np.all(np.isfinite(pixels))
    result={'cases_verified':cases,'metric_rows_verified':len(rows),'summary_rows_verified':len(summary),
        'pngs_verified':len(figures),'max_saved_signal_metric_difference':max_metric_difference,
        'max_saved_checkpoint_validation_loss_difference':max_validation_difference,
        'independent_validation_observations_reproduced':True,'early_stop_decisions_replayed':True,
        'baseline_signals_unchanged':True,'source_and_input_hashes_match':True,
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    experiment.bench.save_json(output/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
