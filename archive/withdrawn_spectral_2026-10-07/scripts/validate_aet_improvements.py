"""Verify saved signals/criteria and test whether the head also benefits AE."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import evaluate_aet_improvements as exp
import numpy as np
import torch
from transformer_based_adaptive_line_enhancer.aet_dle import (
    AETDLEConfig,AETDLETrainingResult,AETMinMaxScaler,AutoencoderTransformerDLE,enhance_with_aet_dle,
)
from transformer_based_adaptive_line_enhancer.aet_improved import SpectralHeadConfig,spectral_residual_head


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=exp.OUT)
    args=parser.parse_args()
    torch.set_num_threads(2)
    controls=[];verified=[];max_difference=0.;checkpoint_difference=0.
    for phase in ['pilot','holdout','offgrid']:
        folder=args.output/phase
        if not (folder/'manifest.json').exists():continue
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(Path(exp.__file__).read_bytes()).hexdigest()==manifest['script_sha256']
        for path,digest in manifest['source_hashes'].items():
            assert hashlib.sha256((exp.ROOT/path).read_bytes()).hexdigest()==digest
        assert manifest['head_config']==asdict(SpectralHeadConfig())
        if phase!='pilot':
            assert not set(manifest['noise_seeds'])&{42,43,44}
            assert not set(manifest['model_seeds'])&{0,1,2}
        expected=len(manifest['noise_seeds'])*len(manifest['scenes'])*len(manifest['snrs'])
        case_count=0;row_count=0
        for marker in sorted((folder/'cases').glob('*/metrics.json')):
            payload=json.loads(marker.read_text(encoding='utf-8'))
            assert payload['protocol_hash']==manifest['protocol_hash']
            with np.load(marker.parent/'signals.npz') as archive:
                data={key:archive[key].copy() for key in archive.files}
            frequencies=payload['metadata']['scene_config']['frequencies_hz']
            for row in payload['metrics']:
                actual=exp.measure(data[row['algorithm']],data,frequencies)
                for field,value in actual.items():
                    max_difference=max(max_difference,abs(value-row[field]))
                    np.testing.assert_allclose(value,row[field],atol=1e-12,rtol=0)
                row_count+=1
            baseline=max((r for r in payload['metrics'] if r['algorithm'] in ['ALE','AE','AET','AET_early_stop']),key=lambda r:r['output_snr_db'])
            for row in payload['comparisons']:
                assert row['best_baseline']==baseline['algorithm']
                assert row['passes_db_10_percent']==(row['snr_difference_db']>0 and row['snr_difference_db']>=.1*abs(baseline['output_snr_db']))
            replay=spectral_residual_head(data['noisy'],data['AET'],neural_valid_slice=slice(0,int(data['evaluation_end'])))
            np.testing.assert_array_equal(replay['enhanced_signal'],data['AET_spectral'])
            np.testing.assert_array_equal(replay['spectral_only'],data['spectral_only'])
            control=spectral_residual_head(data['noisy'],data['AE'],neural_valid_slice=slice(1000,15976))
            measured=exp.measure(control['enhanced_signal'],data,frequencies)
            aet=next(r for r in payload['metrics'] if r['algorithm']=='AET_spectral')
            controls.append({'phase':phase,'scene':aet['scene'],'requested_snr_db':aet['requested_snr_db'],
                'noise_seed':aet['noise_seed'],'AE_shared_head_output_snr_db':measured['output_snr_db'],
                'AET_shared_head_output_snr_db':aet['output_snr_db'],
                'AET_vs_AE_with_same_head_db':aet['output_snr_db']-measured['output_snr_db']})
            checkpoint=marker.parent/'AET_spectral_checkpoint.pt'
            if checkpoint.exists():
                saved=torch.load(checkpoint,weights_only=True)
                config=AETDLEConfig(**saved['config'])
                model=AutoencoderTransformerDLE(config);model.load_state_dict(saved['state_dict']);model.eval()
                trained=AETDLETrainingResult(model,AETMinMaxScaler(**saved['scaler']),config,tuple(saved['loss_history']))
                raw=enhance_with_aet_dle(data['noisy'],trained)
                difference=float(np.max(np.abs(raw.enhanced_signal-data['AET'])))
                checkpoint_difference=max(checkpoint_difference,difference)
                np.testing.assert_array_equal(raw.enhanced_signal,data['AET'])
            case_count+=1
        assert case_count==expected
        verified.append({'phase':phase,'case_count':case_count,'metric_rows':row_count,
                         'all_head_outputs_replayed':True,'all_10_percent_decisions_recomputed':True})
    exp.b.save_csv(args.output/'shared_head_control.csv',controls)
    result={'phases':verified,'maximum_metric_difference':max_difference,
        'maximum_original_checkpoint_replay_difference':checkpoint_difference,
        'independent_noise_and_model_seeds_verified':True,'shared_head_controls':len(controls),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    exp.b.save_json(args.output/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
