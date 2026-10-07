"""Compare original ALE/AE with AET selected using independent noisy validation.

Clean references are used to construct synthetic repeated observations and to
evaluate AFTER training. train_with_early_stopping receives no clean reference.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import reproduce_comparison as bench
from aet_epoch_sweep import compute_metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_based_adaptive_line_enhancer.aet_dle import (
    AETDLEConfig, AETDLETrainingResult, AETMinMaxScaler,
    AutoencoderTransformerDLE, _sequence_pairs, enhance_with_aet_dle,
)

ROOT = bench.ROOT
BASE = ROOT / 'results/reproduction_2026-10-07'
NAMES = ('ALE', 'AE', 'AET_ES')
COLORS = {'ALE':'#2878b5', 'AE':'#e58b27', 'AET_ES':'#3d9d64'}
LABELS = {'ALE':'ALE', 'AE':'AE (60 epochs)', 'AET_ES':'AET (early stop)'}


def validation_mse(model, inputs, targets, batch_size=64):
    """Unclipped noisy targets; validation input clipping matches inference."""
    model.eval()
    squared_error, count = 0., 0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            target = targets[start:start+batch_size]
            error = model(inputs[start:start+batch_size]) - target
            squared_error += float(torch.sum(error.double()**2))
            count += target.numel()
    return squared_error / count


def train_with_early_stopping(noisy, validation_noisy, config, patience, relative_delta):
    """Selection depends solely on independent noisy-validation frame MSE."""
    scaler = AETMinMaxScaler.fit(noisy)
    current, target, _ = _sequence_pairs(scaler.transform(noisy), config)
    train_data = TensorDataset(torch.from_numpy(current).float(), torch.from_numpy(target).float())
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_inputs, val_targets = [], []
    for observation in validation_noisy:
        # Delay exceeds sequence coverage: each noisy target is independent of
        # the noise in its own input; validation realizations never train model.
        inputs, _, _ = _sequence_pairs(scaler.transform(observation, clip=True), config)
        _, targets, _ = _sequence_pairs(scaler.transform(observation), config)
        val_inputs.append(inputs); val_targets.append(targets)
    inputs = torch.from_numpy(np.concatenate(val_inputs)).float()
    targets = torch.from_numpy(np.concatenate(val_targets)).float()
    torch.manual_seed(config.seed)
    model = AutoencoderTransformerDLE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_loss, best_epoch, stale, best_state = float('inf'), 0, 0, None
    history = []
    for epoch in range(1, config.epochs+1):
        model.train()
        total, count = 0., 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * y.numel(); count += y.numel()
        val_loss = validation_mse(model, inputs, targets)
        improved = val_loss < best_loss * (1-relative_delta)
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({'epoch':epoch, 'training_mse':total/count,
                        'validation_mse':val_loss, 'significant_improvement':improved,
                        'best_epoch_so_far':best_epoch, 'stale_epochs':stale})
        if stale >= patience:
            break
    model.load_state_dict(best_state); model.eval()
    replay_loss = validation_mse(model, inputs, targets)
    np.testing.assert_allclose(replay_loss, best_loss, rtol=0, atol=1e-12)
    selected_config = replace(config, epochs=best_epoch)
    training = AETDLETrainingResult(model, scaler, selected_config,
                                   tuple(r['training_mse'] for r in history[:best_epoch]))
    return training, history, {'selected_epoch':best_epoch, 'stopped_epoch':len(history),
        'best_validation_mse':best_loss, 'raw_minimum_validation_epoch':min(history,key=lambda r:r['validation_mse'])['epoch'],
        'stop_reason':'patience' if stale>=patience else 'maximum_epochs',
        'checkpoint_validation_replay_difference':abs(replay_loss-best_loss)}


def run_case(scene, snr, seed, model_seed, args, output, protocol_hash):
    label = f'{scene}_snr_{snr:g}_seed_{seed}'
    source = BASE/'cases'/label
    case = output/'cases'/label; case.mkdir(parents=True, exist_ok=True)
    if args.resume and (case/'metrics.json').exists():
        saved = json.loads((case/'metrics.json').read_text(encoding='utf-8'))
        if saved['protocol_hash'] != protocol_hash:
            raise ValueError(f'Protocol mismatch: {label}')
        print(f'REUSE {label}', flush=True)
        return saved
    with np.load(source/'signals.npz') as archive:
        data = {key:archive[key].copy() for key in archive.files}
    validation_seeds = [seed+10000*(i+1) for i in range(args.validation_repeats)]
    sigma = np.sqrt(np.mean(data['noise']**2))
    # IID samples use population sigma; no normalization by realized noise power.
    # The synthetic latent signal is shared, whereas all noise realizations differ.
    validation = np.stack([data['clean'] + np.random.default_rng(s).normal(0,sigma,data['clean'].size)
                           for s in validation_seeds])
    config = AETDLEConfig(epochs=args.max_epochs, seed=model_seed)
    assert config.prediction_delay > config.frame_length+(config.sequence_length-1)*config.frame_hop
    started = time.perf_counter()
    trained, history, selection = train_with_early_stopping(data['noisy'], validation, config,
                                                            args.patience, args.relative_delta)
    seconds = time.perf_counter()-started
    enhanced = enhance_with_aet_dle(data['noisy'], trained)
    assert enhanced.valid_start<=int(data['evaluation_start']) and enhanced.valid_end>=int(data['evaluation_end'])
    signals = {name:data[name] for name in ('ALE','AE')}
    signals['AET_ES'] = enhanced.enhanced_signal
    metrics = []
    original = json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    for name, values in signals.items():
        measured = compute_metrics(values, data, scene)
        if name != 'AET_ES':
            baseline = next(r for r in original['metrics'] if r['algorithm']==name)
            np.testing.assert_allclose(measured['output_snr_db'],baseline['output_snr_db'],atol=1e-10,rtol=0)
        metrics.append({'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'model_seed':model_seed,
            'algorithm':name, **measured, 'selected_epoch':selection['selected_epoch'] if name=='AET_ES' else (60 if name=='AE' else 0),
            'stopped_epoch':selection['stopped_epoch'] if name=='AET_ES' else (60 if name=='AE' else 0),
            'evaluation_start':int(data['evaluation_start']),'evaluation_end':int(data['evaluation_end'])})
    fixed = compute_metrics(data['AET'],data,scene)
    selection.update({'scene':scene,'requested_snr_db':snr,'noise_seed':seed,'model_seed':model_seed,
        'validation_noise_seeds':validation_seeds,'validation_noise_sigma':float(sigma),
        'training_and_validation_seconds':seconds,'fixed_60_output_snr_db':fixed['output_snr_db'],
        'early_stop_output_snr_db':metrics[-1]['output_snr_db'],
        'improvement_vs_fixed_60_db':metrics[-1]['output_snr_db']-fixed['output_snr_db']})
    bench.save_csv(case/'epoch_history.csv',history)
    bench.save_json(case/'selection.json',selection)
    np.savez_compressed(case/'signals.npz',**signals, clean=data['clean'],noisy=data['noisy'],
        time_s=data['time_s'],noise=data['noise'],AET_fixed60=data['AET'],
        evaluation_start=data['evaluation_start'],evaluation_end=data['evaluation_end'])
    np.savez_compressed(case/'validation_observations.npz',noisy=validation,noise_seeds=validation_seeds)
    torch.save({'state_dict':trained.model.state_dict(),'config':asdict(trained.config),
        'scaler':asdict(trained.scaler),'loss_history':trained.loss_history,
        'selection':selection,'protocol_hash':protocol_hash},case/'AET_ES_checkpoint.pt')
    # Check persisted model reconstructs the exact saved prediction.
    checkpoint = torch.load(case/'AET_ES_checkpoint.pt',weights_only=True)
    restored = AutoencoderTransformerDLE(AETDLEConfig(**checkpoint['config']))
    restored.load_state_dict(checkpoint['state_dict']); restored.eval()
    replay = enhance_with_aet_dle(data['noisy'],AETDLETrainingResult(restored,trained.scaler,
        restored.config,checkpoint['loss_history']))
    np.testing.assert_array_equal(replay.enhanced_signal,enhanced.enhanced_signal)
    selection['checkpoint_signal_replay_max_difference'] = float(np.max(np.abs(replay.enhanced_signal-enhanced.enhanced_signal)))
    payload = {'protocol_hash':protocol_hash,'metrics':metrics,'selection':selection,'history':history}
    bench.save_json(case/'metrics.json',payload)
    print(f'{label}: selected={selection["selected_epoch"]}, stopped={selection["stopped_epoch"]}, '
          f'AET_ES={metrics[-1]["output_snr_db"]:.3f}, AET60={fixed["output_snr_db"]:.3f} dB, {seconds:.1f}s',flush=True)
    return payload


def summarize(rows, args):
    output=[]
    for scene in args.scenes:
        for snr in args.snrs:
            for name in NAMES:
                selected=[r for r in rows if r['scene']==scene and r['requested_snr_db']==snr and r['algorithm']==name]
                row={'scene':scene,'requested_snr_db':snr,'algorithm':name,'seeds':len(selected)}
                for field in ('output_snr_db','snr_gain_db','input_snr_db','weakest_tone_retention','selected_epoch','stopped_epoch'):
                    row[field+'_mean']=float(np.mean([r[field] for r in selected]))
                    row[field+'_std']=float(np.std([r[field] for r in selected],ddof=1)) if len(selected)>1 else 0.
                output.append(row)
    return output


def plot(summary, rows, selections, args, output):
    fig,axes=plt.subplots(2,len(args.scenes),figsize=(7*len(args.scenes),8),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        for name in NAMES:
            selected=[r for r in summary if r['scene']==scene and r['algorithm']==name]
            for i,metric in enumerate(('output_snr_db','snr_gain_db')):
                axes[i,j].errorbar(args.snrs,[r[metric+'_mean'] for r in selected],
                    yerr=[r[metric+'_std'] for r in selected],marker='o',capsize=4,color=COLORS[name],label=LABELS[name])
        axes[0,j].axhline(0,color='gray',linestyle=':',label='Zero-output baseline (0 dB)')
        axes[1,j].plot(args.snrs,[-np.mean([r['input_snr_db'] for r in rows if r['scene']==scene and r['requested_snr_db']==s]) for s in args.snrs],
                       color='gray',linestyle=':',label='Zero-output gain')
        for i,label in enumerate(('Output SNR (dB)','Output - measured input SNR (dB)')):
            axes[i,j].set(title=scene,xlabel='Requested input SNR (dB)',ylabel=label)
            axes[i,j].grid(alpha=.25);axes[i,j].legend(fontsize=9)
    fig.suptitle(f'ALE / AE / AET early stopping | mean +/- sample SD ({len(args.seeds)} seeds)')
    for extension in ('png','svg'): fig.savefig(output/f'snr_comparison.{extension}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,len(args.scenes),figsize=(7*len(args.scenes),4.5),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        for name in NAMES:
            selected=[r for r in summary if r['scene']==scene and r['algorithm']==name]
            axes[0,j].errorbar(args.snrs,[100*r['weakest_tone_retention_mean'] for r in selected],
                yerr=[100*r['weakest_tone_retention_std'] for r in selected],marker='o',capsize=4,color=COLORS[name],label=LABELS[name])
        axes[0,j].axhline(100,color='gray',linestyle='--',label='Correct amplitude')
        axes[0,j].set(title=scene,xlabel='Requested input SNR (dB)',ylabel='Weakest tone amplitude retention (%)')
        axes[0,j].legend(fontsize=8);axes[0,j].grid(alpha=.25)
    fig.savefig(output/'weakest_tone_retention.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,len(args.scenes),figsize=(7*len(args.scenes),8),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        selected=[r for r in selections if r['scene']==scene]
        for seed in args.seeds:
            subset=[r for r in selected if r['noise_seed']==seed]
            axes[0,j].plot(args.snrs,[r['selected_epoch'] for r in subset],'-o',label=f'noise seed {seed}')
        for field,label,color in [('fixed_60_output_snr_db','AET fixed 60','#9753a4'),('early_stop_output_snr_db','AET early stop',COLORS['AET_ES'])]:
            axes[1,j].errorbar(args.snrs,[np.mean([r[field] for r in selected if r['requested_snr_db']==s]) for s in args.snrs],
                yerr=[np.std([r[field] for r in selected if r['requested_snr_db']==s],ddof=1) if len(args.seeds)>1 else 0 for s in args.snrs],
                marker='o',capsize=4,color=color,label=label)
        for i,ylabel in enumerate(('Restored checkpoint epoch','Output SNR (dB)')):
            axes[i,j].set(title=scene,xlabel='Requested input SNR (dB)',ylabel=ylabel)
            axes[i,j].grid(alpha=.25);axes[i,j].legend(fontsize=8)
    fig.suptitle('Validation-selected epochs and effect of early stopping')
    fig.savefig(output/'early_stopping_effect.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,len(args.scenes),figsize=(7*len(args.scenes),4.5),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        for name in NAMES:
            subset=[r for r in rows if r['scene']==scene and r['noise_seed']==args.seeds[0] and r['algorithm']==name]
            axes[0,j].plot(args.snrs,[r['output_snr_db'] for r in subset],'-o',color=COLORS[name],label=LABELS[name])
        axes[0,j].axhline(0,color='gray',linestyle=':')
        axes[0,j].set(title=scene,xlabel='Requested input SNR (dB)',ylabel='Output SNR (dB)')
        axes[0,j].legend(fontsize=8);axes[0,j].grid(alpha=.25)
    fig.suptitle(f'Primary observation: noise seed {args.seeds[0]}, model seed 0')
    fig.savefig(output/'seed42_snr_comparison.png',dpi=180);plt.close(fig)


def report(summary, rows, selections, args, output):
    text=['# AET 独立带噪验证早停：ALE / AE / AET 对比','',
        '日期：2026-10-07。以下数值来自实际运行；早停规则在计算最终干净参考指标之前确定。','',
        '## 实验与早停机制','',
        '- 原项目没有内置早停。本次在实验脚本中实现验证、停止和最佳权重恢复，网络、Adam、洗牌顺序与原 AET 保持一致。',
        f'- 输入 SNR={args.snrs} dB，噪声种子={args.seeds}、模型种子依次为 0/1/2；单谱线 100 Hz，多谱线 50/100/180 Hz、幅度 1/0.6/0.3。',
        '- 原观测 16000 点、1 kHz，沿用之前原始数据。ALE/AE 直接复用并重算之前输出，AE 固定 60 轮；只对 AET 更换停止机制。',
        f'- AET 最多 {args.max_epochs} 轮，每轮完成后验证。验证 MSE 必须相对已保存最佳值下降超过 {args.relative_delta*100:g}% 才算改善；连续 {args.patience} 轮无此改善则停止，恢复上一次明显改善的检查点。没有达到 patience 时用预算内最佳检查点。',
        f'- 每段观测另生成 {args.validation_repeats} 个独立噪声重复观测：seed = 原 noise_seed + 10000 × (1..{args.validation_repeats})。共享同一合成潜在信号，噪声为独立 IID 高斯，标准差取原噪声 RMS；不按样本功率二次缩放。',
        '- 验证集完全不参与优化，归一化只在原训练观测上拟合。验证输入按推理规则裁剪到 [0,1]，带噪目标不裁剪；归一化帧 MSE 跨全部验证对求平均。',
        '- 延迟 1000 点大于序列覆盖 768 点，因此单个验证输入/目标的噪声采样不重叠。早停函数只接收训练带噪观测、验证带噪观测和配置，不接收干净标签、频率或最终 SNR。',
        '- 干净信号用于合成重复观测与最终外部评价，不用于早停损失或选轮次；这是具有重复观测的合成实验条件。现实只有一条观测时，不能凭空得到这组验证数据。',
        '- 统一评价区间 [8000,14976)，输出 SNR=10log10(mean(clean²)/mean((output-clean)²))；增益相对于该区间实测输入 SNR。误差条为 3 次重复的样本标准差，不是置信区间。',
        '', '## 三算法输出 SNR（均值 ± 标准差，dB）','',
        '| 场景 | 输入 SNR | ALE | AE（60 轮） | AET（验证早停） |','|---|---:|---:|---:|---:|']
    for scene in args.scenes:
        for snr in args.snrs:
            subset=[r for r in summary if r['scene']==scene and r['requested_snr_db']==snr]
            values=[next(r for r in subset if r['algorithm']==n) for n in NAMES]
            text.append(f'| {scene} | {snr:g} | '+ ' | '.join(f'{r["output_snr_db_mean"]:.3f} ± {r["output_snr_db_std"]:.3f}' for r in values)+' |')
    text += ['', '![三算法 SNR 对比](snr_comparison.png)','', '## seed=42、输入 −20 dB','',
        '| 场景 | 算法 | 选中轮次 | 实际停止轮次 | 输出 SNR | SNR 增益 | 最强 / 最弱谱线幅值保留 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['noise_seed']==42 and r['requested_snr_db']==-20:
            text.append(f'| {r["scene"]} | {r["algorithm"]} | {r["selected_epoch"]} | {r["stopped_epoch"]} | {r["output_snr_db"]:.3f} | {r["snr_gain_db"]:.3f} | {100*r["strongest_tone_retention"]:.3f}% / {100*r["weakest_tone_retention"]:.3f}% |')
    text += ['', '## AET：固定 60 轮与早停','',
        '| 场景 | 输入 SNR | 早停相对 60 轮变化（均值 dB） | 三个种子选中轮次 | 三个种子停止轮次 |',
        '|---|---:|---:|---|---|']
    for scene in args.scenes:
        for snr in args.snrs:
            subset=[r for r in selections if r['scene']==scene and r['requested_snr_db']==snr]
            text.append(f'| {scene} | {snr:g} | {np.mean([r["improvement_vs_fixed_60_db"] for r in subset]):+.3f} | {[r["selected_epoch"] for r in subset]} | {[r["stopped_epoch"] for r in subset]} |')
    text += ['', '![早停轮次和变化](early_stopping_effect.png)','', '## 如何解读','',
        '早停限制了模型继续降低训练误差时对特定噪声的拟合，但不保证学到有效谱线。必须同时看输出 SNR、谱线幅值保留与波形；尤其在 −20 dB，输出全为零就有 0 dB 输出 SNR、约 20 dB 增益，不能仅凭增益宣称成功恢复。图中加入全零基线用于识别这个现象。',
        '', '此前逐轮实验的单谱线 16 轮、多谱线 5 轮，是用干净参考 SNR 事后选出的峰值。本次独立带噪验证早停可选择不同轮次，不能把之前峰值直接作为所有 SNR 的固定停止规则。',
        '', '验证 MSE 与实际同段增强的 SNR 不完全相同：验证包含额外噪声、帧重叠加权，最终输出还做重叠平均；有限样本与耐心参数也影响选轮次。因此早停并非每个条件都提高 SNR。本次没有用最终测试指标反复调 patience 或阈值。',
        '', '本对比回答“把现有 AET 改为验证早停后效果如何”。AET 有额外验证观测，AE 固定 60 轮；模型容量、训练预算和可用验证信息不同，不能解释为三种架构最优调参后的公平排名。三个种子不足以推广成普遍结论。',
        '', '![谱线保留](weakest_tone_retention.png)','', '## 交付与复跑','',
        '- `snr_comparison.png/.svg`：两场景三算法输出 SNR 与增益。',
        '- `seed42_snr_comparison.png`：seed=42 的三算法扫描对比。',
        '- `early_stopping_effect.png`：停止轮次和 AET 固定 60 轮前后效果。',
        '- `weakest_tone_retention.png`：弱谱线幅值保留。另有 seed=42 波形、频谱和共享色标时频图。',
        '- `metrics.csv` / `summary.csv` / `early_stopping_selections.csv`：逐观测、汇总指标和轮次。',
        '- 每个 `cases/` 子目录：完整输出信号、独立验证观测、逐轮验证日志、早停检查点和选择依据。',
        '- `manifest.json` 保存协议与来源哈希，`validation.json` 保存重算和检查点一致性检查。','',
        '```powershell', '.\\.venv\\Scripts\\python.exe scripts/compare_aet_early_stopping.py --resume', '```','']
    (output/'EARLY_STOPPING_REPORT.md').write_text('\n'.join(text),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-epochs',type=int,default=120)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--relative-delta',type=float,default=1e-4)
    parser.add_argument('--validation-repeats',type=int,default=3)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--snrs',nargs='+',type=float,default=[-20,-15,-10,-5,0])
    parser.add_argument('--scenes',nargs='+',choices=list(bench.SCENES),default=list(bench.SCENES))
    parser.add_argument('--output',type=Path,default=ROOT/'results/aet_early_stopping_2026-10-07')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.max_epochs,args.patience,args.validation_repeats,args.threads)<1 or not 0<=args.relative_delta<1:
        parser.error('Epochs/patience/repeats/threads must be positive; relative delta must be in [0,1).')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    for path,digest in baseline['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, f'Baseline source changed: {path}'
    manifest={'max_epochs':args.max_epochs,'patience':args.patience,'relative_min_delta':args.relative_delta,
        'validation_repeats':args.validation_repeats,'validation':'Independent IID noisy realizations; delayed-pair MSE',
        'baseline_protocol_hash':baseline['protocol_hash'],'seeds':args.seeds,'snrs':args.snrs,'scenes':args.scenes,
        'source_hashes':baseline['source_hashes'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_hashes':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['reproduce_comparison.py','aet_epoch_sweep.py']},
        'threads':args.threads,'torch_version':torch.__version__,'numpy_version':np.__version__,
        'baseline_input_hashes':{f'{scene}_snr_{s:g}_seed_{seed}':hashlib.sha256((BASE/'cases'/f'{scene}_snr_{s:g}_seed_{seed}'/'signals.npz').read_bytes()).hexdigest()
            for scene in args.scenes for s in args.snrs for seed in args.seeds}}
    manifest['protocol_hash']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    bench.save_json(args.output/'manifest.json',manifest)
    rows,selections=[],[]
    for scene in args.scenes:
        for snr in args.snrs:
            for i,seed in enumerate(args.seeds):
                payload=run_case(scene,snr,seed,i,args,args.output,manifest['protocol_hash'])
                rows.extend(payload['metrics']);selections.append(payload['selection'])
    summary=summarize(rows,args)
    bench.save_csv(args.output/'metrics.csv',rows);bench.save_csv(args.output/'summary.csv',summary)
    bench.save_csv(args.output/'early_stopping_selections.csv',selections)
    plot(summary,rows,selections,args,args.output)
    bench.ALGORITHMS=NAMES;bench.COLORS=COLORS
    bench.plot_cases(args,args.output)
    report(summary,rows,selections,args,args.output)
    bench.save_json(args.output/'validation.json',{'case_count':len(selections),'metric_rows':len(rows),
        'baseline_ALE_AE_metrics_recomputed':True,'validation_best_checkpoint_replayed':True,
        'max_signal_checkpoint_replay_difference':max(r['checkpoint_signal_replay_max_difference'] for r in selections),
        'max_validation_checkpoint_replay_difference':max(r['checkpoint_validation_replay_difference'] for r in selections),
        'source_hashes_match_original':True,'selection_function_has_no_clean_reference':True})
    print(f'DONE: {args.output}',flush=True)


if __name__=='__main__': main()
