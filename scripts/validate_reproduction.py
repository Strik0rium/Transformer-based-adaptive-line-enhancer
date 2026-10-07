"""Check saved benchmark data and replay two primary-seed checkpoints."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from transformer_based_adaptive_line_enhancer.ae_dle import (
    AEDLEConfig, AEDLETrainingResult, AutoencoderDLE, MinMaxScaler, enhance_with_ae_dle,
)
from transformer_based_adaptive_line_enhancer.aet_dle import (
    AETDLEConfig, AETDLETrainingResult, AutoencoderTransformerDLE, AETMinMaxScaler,
    enhance_with_aet_dle,
)


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT / "results/reproduction_2026-10-07"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected_cases = len(manifest["scenes"])*len(manifest["snrs"])*len(manifest["seeds"])
    metrics = list(csv.DictReader((output / "metrics.csv").open(encoding="utf-8-sig")))
    tone_metrics = list(csv.DictReader((output / "tone_metrics.csv").open(encoding="utf-8-sig")))
    summary = list(csv.DictReader((output / "summary.csv").open(encoding="utf-8-sig")))
    cases = sorted((output / "cases").glob("*/metrics.json"))
    assert len(cases)==expected_cases
    assert len(metrics)==expected_cases*3
    keys = {(r['scene'],r['requested_snr_db'],r['noise_seed'],r['algorithm']) for r in metrics}
    assert len(keys)==len(metrics)
    for relative, expected in manifest["source_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((ROOT / "scripts/reproduce_comparison.py").read_bytes()).hexdigest()==manifest["script_sha256"]
    checked_rows = 0
    for marker in cases:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload['protocol_hash']==manifest['protocol_hash']
        with np.load(marker.parent / "signals.npz") as signals:
            np.testing.assert_allclose(signals['noisy'],signals['clean']+signals['noise'],rtol=0,atol=1e-14)
            for name in ['time_s','clean','noise','noisy','ALE','AE','AET']:
                assert signals[name].shape==(manifest['num_samples'],)
                assert np.all(np.isfinite(signals[name]))
            for row in payload['metrics']:
                start, end = row['evaluation_start'], row['evaluation_end']
                assert (start,end)==(int(signals['evaluation_start']),int(signals['evaluation_end']))
                clean = signals['clean'][start:end]
                estimate = signals[row['algorithm']][start:end]
                calculated = 10*np.log10(np.mean(clean**2)/np.mean((estimate-clean)**2))
                np.testing.assert_allclose(calculated,row['output_snr_db'],rtol=0,atol=1e-12)
                measured = 10*np.log10(np.mean(signals['clean']**2)/np.mean(signals['noise']**2))
                np.testing.assert_allclose(measured,row['requested_snr_db'],rtol=0,atol=1e-10)
                checked_rows+=1
    for row in summary:
        selected = [r for r in metrics if all(r[k]==row[k] for k in ['scene','requested_snr_db','algorithm'])]
        assert len(selected)==len(manifest['seeds'])
        values = np.array([float(r['output_snr_db']) for r in selected])
        np.testing.assert_allclose(values.mean(),float(row['output_snr_db_mean']),atol=1e-12)
        if values.size>1:
            np.testing.assert_allclose(values.std(ddof=1),float(row['output_snr_db_std']),atol=1e-12)
    torch.set_num_threads(manifest['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    replay_case = output / 'cases' / f"multiline_snr_-10_seed_{manifest['seeds'][0]}"
    replay_results = {}
    with np.load(replay_case / 'signals.npz') as signals:
        for name in ['AE','AET']:
            saved = torch.load(replay_case / f'{name}_checkpoint.pt',map_location='cpu',weights_only=True)
            if name=='AE':
                config = AEDLEConfig(**saved['config'])
                model = AutoencoderDLE(frame_length=config.frame_length,hidden_dim=config.hidden_dim,
                    bottleneck_dim=config.bottleneck_dim)
                model.load_state_dict(saved['state_dict'])
                trained = AEDLETrainingResult(model.eval(),MinMaxScaler(**saved['scaler']),config,tuple(saved['loss_history']))
                result = enhance_with_ae_dle(signals['noisy'],trained)
            else:
                config = AETDLEConfig(**saved['config'])
                model = AutoencoderTransformerDLE(config)
                model.load_state_dict(saved['state_dict'])
                trained = AETDLETrainingResult(model.eval(),AETMinMaxScaler(**saved['scaler']),config,tuple(saved['loss_history']))
                result = enhance_with_aet_dle(signals['noisy'],trained)
            difference = float(np.max(np.abs(result.enhanced_signal-signals[name])))
            np.testing.assert_allclose(result.enhanced_signal,signals[name],rtol=0,atol=1e-7)
            replay_results[name] = {'maximum_absolute_difference':difference,'passed':True}
    pngs = sorted(output.glob('*.png'))
    assert len(pngs)==9
    result = {'passed':True,'case_count':len(cases),'metrics_recomputed':checked_rows,
        'tone_metric_rows':len(tone_metrics),'summary_rows':len(summary),'png_count':len(pngs),
        'source_hashes_verified':True,'checkpoint_replay':replay_results}
    (output / 'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
