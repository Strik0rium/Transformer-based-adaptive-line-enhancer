"""Evaluate AET after every epoch without changing its original optimization.

Uses the saved seed-42 / -20 dB observations and model seed 0 from the previous
experiment. Instruments the original training loop with deterministic inference
after each epoch; matches previous epoch-10/30/60/120 reference results.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transformer_based_adaptive_line_enhancer.aet_dle import (
    AETDLEConfig, AETDLETrainingResult, AETMinMaxScaler, AutoencoderTransformerDLE,
    _sequence_pairs, enhance_with_aet_dle,
)

BASE = ROOT / 'results/reproduction_2026-10-07'
COLORS = {'single':'#2878b5','multiline':'#e58b27'}


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def compute_metrics(values: np.ndarray, data: dict, scene: str) -> dict:
    start, end = int(data['evaluation_start']), int(data['evaluation_end'])
    interval = slice(start, end)
    clean = data['clean'][interval]
    error = values[interval] - clean
    times = data['time_s'][interval]
    frequencies = [100.] if scene=='single' else [50.,100.,180.]
    basis = np.column_stack([function(2*np.pi*frequency*times)
        for frequency in frequencies for function in [np.sin, np.cos]])
    q, _ = np.linalg.qr(basis)
    constant = np.ones(times.size); constant -= q@(q.T@constant)
    constant /= np.linalg.norm(constant)
    tone_error = q@(q.T@error)
    dc_error = constant*(constant@error)
    other_error = error-tone_error-dc_error
    signal_power = float(np.mean(clean**2))
    mse = float(np.mean(error**2))
    input_snr = float(10*np.log10(signal_power/np.mean((data['noisy'][interval]-clean)**2)))
    output_snr = float(10*np.log10(signal_power/mse))
    coefficients = np.linalg.lstsq(np.column_stack([basis,np.ones(times.size)]), values[interval], rcond=None)[0]
    clean_coefficients = np.linalg.lstsq(np.column_stack([basis,np.ones(times.size)]), clean, rcond=None)[0]
    ratios = [float(np.hypot(*coefficients[2*i:2*i+2])/np.hypot(*clean_coefficients[2*i:2*i+2]))
              for i in range(len(frequencies))]
    noise = data['noise'][interval]
    noise_other = noise-q@(q.T@noise)-constant*(constant@noise)
    np.testing.assert_allclose(np.mean(tone_error**2)+np.mean(dc_error**2)+np.mean(other_error**2),mse,atol=1e-12)
    return {'input_snr_db':input_snr,'output_snr_db':output_snr,'snr_gain_db':output_snr-input_snr,
        'mse':mse,'output_power_over_clean':float(np.mean(values[interval]**2)/signal_power),
        'tone_error_over_clean':float(np.mean(tone_error**2)/signal_power),
        'dc_error_over_clean':float(np.mean(dc_error**2)/signal_power),
        'other_error_over_clean':float(np.mean(other_error**2)/signal_power),
        'noise_error_correlation':float(np.corrcoef(other_error,noise_other)[0,1]),
        'strongest_tone_retention':ratios[0],'weakest_tone_retention':ratios[-1]}


def run_scene(scene: str, args, output: Path) -> tuple[list[dict], dict]:
    case = BASE/'cases'/f'{scene}_snr_-20_seed_42'
    with np.load(case/'signals.npz') as archive:
        data = {key:archive[key].copy() for key in archive.files}
    settings = AETDLEConfig(epochs=args.epochs, seed=0)
    scaler = AETMinMaxScaler.fit(data['noisy'])
    normalized = scaler.transform(data['noisy'])
    current, delayed, _ = _sequence_pairs(normalized,settings)
    torch.manual_seed(settings.seed)
    model = AutoencoderTransformerDLE(settings)
    optimizer = torch.optim.Adam(model.parameters(),lr=settings.learning_rate)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(current).to(torch.float32),torch.from_numpy(delayed).to(torch.float32))
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(dataset,batch_size=settings.batch_size,shuffle=True,generator=generator)
    history, rows, waveforms = [], [], []
    best_snr = -float('inf')
    best_state, best_epoch, best_waveform = None, None, None
    for epoch in range(1,args.epochs+1):
        model.train()
        total_squared_error,total_values = 0.,0
        for current_batch, delayed_batch in loader:
            optimizer.zero_grad()
            prediction = model(current_batch)
            loss = loss_function(prediction,delayed_batch)
            loss.backward(); optimizer.step()
            total_squared_error += float(loss.detach())*delayed_batch.numel()
            total_values += delayed_batch.numel()
        history.append(total_squared_error/total_values)
        model.eval()
        training = AETDLETrainingResult(model,scaler,settings,tuple(history))
        result = enhance_with_aet_dle(data['noisy'],training)
        assert result.valid_start <= int(data['evaluation_start'])
        assert result.valid_end >= int(data['evaluation_end'])
        metrics = compute_metrics(result.enhanced_signal,data,scene)
        row = {'scene':scene,'noise_seed':42,'model_seed':0,'requested_snr_db':-20,
            'epoch':epoch,'training_loss':history[-1],**metrics,
            'gain_change_from_previous_epoch_db':metrics['snr_gain_db']-rows[-1]['snr_gain_db'] if rows else '',
            'evaluation_start':int(data['evaluation_start']),'evaluation_end':int(data['evaluation_end'])}
        rows.append(row); waveforms.append(result.enhanced_signal.copy())
        if metrics['output_snr_db'] > best_snr:
            best_snr = metrics['output_snr_db']; best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_waveform = result.enhanced_signal.copy()
        csv_write(output/f'{scene}_per_epoch.csv',rows)
        if epoch==1 or epoch%10==0:
            print(f'{scene} epoch={epoch} output={metrics["output_snr_db"]:.5f} gain={metrics["snr_gain_db"]:.5f} best={best_epoch} loss={history[-1]:.7f}',flush=True)
    saved_config = replace(settings,epochs=best_epoch)
    torch.save({'state_dict':best_state,'config':asdict(saved_config),'scaler':asdict(scaler),
        'loss_history':tuple(history[:best_epoch]),'selected_epoch':best_epoch,
        'selection':'Maximum clean-reference SNR on this same observation, not independent validation'},
        output/f'{scene}_best_checkpoint.pt')
    np.savez_compressed(output/f'{scene}_per_epoch_signals.npz',time_s=data['time_s'],
        clean=data['clean'],noisy=data['noisy'],epoch=np.arange(1,args.epochs+1),
        predictions=np.stack(waveforms),best_prediction=best_waveform,
        evaluation_start=data['evaluation_start'],evaluation_end=data['evaluation_end'])
    reference_rows = list(csv.DictReader((BASE/'diagnostics/epoch_sensitivity.csv').open(encoding='utf-8-sig')))
    validated = []
    for reference in reference_rows:
        epoch = int(reference['epochs'])
        if reference['scene']==scene and float(reference['requested_snr_db'])==-20 and epoch<=args.epochs:
            current_row = rows[epoch-1]
            np.testing.assert_allclose(current_row['output_snr_db'],float(reference['output_snr_db']),rtol=0,atol=1e-10)
            np.testing.assert_allclose(current_row['training_loss'],float(reference['final_training_loss']),rtol=0,atol=1e-12)
            validated.append(epoch)
    # Reload the selected model and verify the persisted best waveform.
    restored = AutoencoderTransformerDLE(saved_config)
    restored.load_state_dict(best_state); restored.eval()
    replay = enhance_with_aet_dle(data['noisy'],AETDLETrainingResult(restored,scaler,saved_config,tuple(history[:best_epoch])))
    np.testing.assert_allclose(replay.enhanced_signal,best_waveform,rtol=0,atol=1e-12)
    best_row = rows[best_epoch-1]
    summary = {'best':best_row,'reference_epochs_matched':validated,
        'best_checkpoint_replay_max_difference':float(np.max(np.abs(replay.enhanced_signal-best_waveform))),
        'zero_output_snr_db':0.0,'zero_output_gain_db':-best_row['input_snr_db'],
        'near_best_epochs_within_0_1_db':[r['epoch'] for r in rows if r['output_snr_db']>=best_snr-0.1],
        'best_is_range_boundary':best_epoch in [1,args.epochs],
        'training_sequences':len(dataset),'updates_per_epoch':len(loader)}
    print(f'BEST {scene} epoch={best_epoch} output={best_snr:.5f} gain={best_row["snr_gain_db"]:.5f}',flush=True)
    return rows,summary


def plot(rows: list[dict], summaries: dict,args,output: Path) -> None:
    fig,axes = plt.subplots(2,len(args.scenes),figsize=(7*len(args.scenes),8),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        selected = [r for r in rows if r['scene']==scene]; best = summaries[scene]['best']
        for i in [0,1]:
            displayed = selected if i==0 else [r for r in selected if r['epoch']<=min(40,args.epochs)]
            axes[i,j].plot([r['epoch'] for r in displayed],[r['snr_gain_db'] for r in displayed],
                '-o',markersize=2.5,linewidth=1.3,color=COLORS[scene],label='Measured gain; one point per epoch')
            axes[i,j].axhline(summaries[scene]['zero_output_gain_db'],linestyle='--',color='gray',label='Zero-output baseline')
            if i==0 or best['epoch']<=40:
                axes[i,j].scatter(best['epoch'],best['snr_gain_db'],marker='*',s=200,color='#b31d28',zorder=5,
                    label=f'Best: epoch {best["epoch"]}, {best["snr_gain_db"]:.3f} dB')
            axes[i,j].set(title=f'{scene}'+(' | early-epoch detail' if i else ''),
                xlabel='Completed training epochs',ylabel='Output SNR - measured input SNR (dB)')
            axes[i,j].grid(alpha=0.25); axes[i,j].legend(fontsize=8)
    fig.suptitle('AET | input -20 dB | noise seed 42, model seed 0 | evaluation after EVERY epoch')
    for extension in ['png','svg']:
        fig.savefig(output/f'snr_gain_per_epoch.{extension}',dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(3,len(args.scenes),figsize=(7*len(args.scenes),10),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(args.scenes):
        selected = [r for r in rows if r['scene']==scene]; epochs=[r['epoch'] for r in selected]
        axes[0,j].plot(epochs,[r['training_loss'] for r in selected],color='#9753a4')
        axes[0,j].set(title=scene,ylabel='Noisy-target training MSE')
        for field,label,color in [('tone_error_over_clean','Target-tone distortion','#2878b5'),
                                  ('other_error_over_clean','Non-target-tone error','#e58b27')]:
            axes[1,j].plot(epochs,[r[field] for r in selected],label=label,color=color)
        axes[1,j].set(ylabel='Error energy / clean signal power'); axes[1,j].legend(fontsize=8)
        for field,label,color in [('strongest_tone_retention','Strongest tone','#2878b5'),
                                  ('weakest_tone_retention','Weakest tone','#e58b27')]:
            if scene=='single' and field=='weakest_tone_retention': continue
            axes[2,j].plot(epochs,[100*r[field] for r in selected],label=label,color=color)
        axes[2,j].axhline(100,color='gray',linestyle='--'); axes[2,j].set(ylabel='Tone amplitude retention (%)',xlabel='Completed training epochs')
        axes[2,j].legend(fontsize=8)
        for i in range(3): axes[i,j].grid(alpha=0.25)
    fig.suptitle('Training objective, reconstruction error, and spectral-line preservation')
    fig.savefig(output/'training_and_signal_diagnostics.png',dpi=180); plt.close(fig)


def report(rows: list[dict],summaries: dict,args,output: Path) -> None:
    text = ['# AET 逐轮 SNR 增益实验与最佳停止轮次','',
        '实验日期：2026-10-07（Asia/Shanghai）。','',
        f'噪声 seed=42、模型初始化 seed=0，沿用此前观测和默认网络结构；输入请求 SNR=-20 dB。完整训练范围为第 1 至 {args.epochs} 轮，每轮评估一次，没有插值或平滑。',
        '', '## 定义与可比性','',
        '- 主图中的 SNR 差值定义为：ΔSNR(epoch) = 输出 SNR(epoch) − 同一评价区间的带噪输入 SNR。CSV 另存相邻轮次增益变化。',
        '- 信号长度 16,000 点、采样率 1 kHz，power-normalized。单谱线 100 Hz；多谱线 50/100/180 Hz、幅度 1/0.6/0.3。',
        '- 评价区间与原三算法对比相同，即 [8000,14976)。整段输入 SNR 精确为 -20 dB；评价子区间实测 SNR 有轻微差异。',
        '- 一次连续训练，在各轮结束后切换 eval 模式计算指标，再恢复 train 模式。优化器、数据洗牌、种子和网络与原训练函数一致。',
        '- 已在 10/30/60/120 轮与上一轮独立训练的结果核对（若在本次范围内）；损失与输出 SNR 一致。',
        '- CPU 线程数 2；Adam 学习率 0.001、batch=16、dropout=0；223 个高度重叠序列，每轮 14 次参数更新。',
        '', '## 最佳轮次（按输出 SNR 最大）','',
        '| 场景 | 最佳轮次 | 实测输入 SNR | 输出 SNR | SNR 增益 | 最强/最弱线幅值保留 |',
        '|---|---:|---:|---:|---:|---:|']
    for scene in args.scenes:
        best=summaries[scene]['best']
        text.append(f'| {scene} | {best["epoch"]} | {best["input_snr_db"]:.4f} dB | {best["output_snr_db"]:.4f} dB | {best["snr_gain_db"]:.4f} dB | {100*best["strongest_tone_retention"]:.2f}% / {100*best["weakest_tone_retention"]:.2f}% |')
    text += ['', '![逐轮 SNR 增益](snr_gain_per_epoch.png)','', '## 原因与选择局限','']
    for scene in args.scenes:
        best=summaries[scene]['best']; near=summaries[scene]['near_best_epochs_within_0_1_db']
        text += [f'- {scene}：最大值出现在第 {best["epoch"]} 轮；距离最佳值不超过 0.1 dB 的轮次为 {near}。该峰值是本观测、初始化、模型和搜索范围内的实测最大值。',
            f'- {scene}：最佳时刻，谱线子空间误差/干净功率={best["tone_error_over_clean"]:.5f}，谱线外误差/干净功率={best["other_error_over_clean"]:.5f}，输出功率/干净功率={best["output_power_over_clean"]:.5f}。']
        if best['output_snr_db'] < 0:
            text += [f'- {scene}：最佳输出 SNR 仍低于“输出全为零”的 0 dB 基线。这说明最大增益并不等价于足够好的波形恢复；弱信号场景下接近零的输出本身就能获得约 20 dB 的输入到输出 SNR 增益。']
        if best['strongest_tone_retention']<0.1:
            text += [f'- {scene}：最佳轮次最强谱线幅值保留低于 10%，因此这个 SNR 峰值主要体现抑制输出误差，不能称为成功恢复谱线的最佳模型。']
    text += ['',
        '训练目标是带噪序列。随着训练推进，网络可以同时学到谱线和目标中的特定噪声；当谱线外误差的增加超过谱线恢复的收益时，输出 SNR 下降，即使训练 MSE 继续下降。误差分解与谱线保留曲线用于检查这个过程，不将所有谱线外误差都武断归为噪声复制。',
        '', '![诊断](training_and_signal_diagnostics.png)','',
        '这里使用已知干净参考在同一观测上选择峰值，属于合成数据的事后诊断，不是独立验证集上确定的早停策略。若要为真实数据选择轮次，应结合独立噪声/隔离验证和谱线保留要求；不能直接把一个种子的最大值当作所有数据的固定停止轮次。',
        '', '## 交付与复跑','',
        '- `snr_gain_per_epoch.png/.svg`：完整曲线和前 40 轮细节，每轮一个数据点。',
        '- `per_epoch_metrics.csv`、`single_per_epoch.csv`、`multiline_per_epoch.csv`：逐轮输出 SNR、增益、损失、误差分解及谱线保留。',
        '- `*_per_epoch_signals.npz`：每轮完整预测和干净/带噪观测，支持重算。',
        '- `*_best_checkpoint.pt`：按上述 SNR 标准选出的检查点，附完整配置和归一化参数。',
        '- `summary.json`、`manifest.json`：峰值、来源文件哈希、配置和运行环境。',
        '', '```powershell', '.\\.venv\\Scripts\\python.exe scripts/aet_epoch_sweep.py --epochs 120', '```']
    (output/'EPOCH_SWEEP_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--scenes',nargs='+',choices=['single','multiline'],default=['single','multiline'])
    parser.add_argument('--output',type=Path,default=BASE/'aet_epoch_sweep_seed42_snr-20')
    args=parser.parse_args()
    if args.epochs<1: parser.error('epochs must be positive')
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    manifest={'noise_seed':42,'model_seed':0,'requested_snr_db':-20,'max_epochs':args.epochs,
        'epoch_step':1,'scenes':args.scenes,'config':asdict(AETDLEConfig(epochs=args.epochs,seed=0)),
        'source_sha256':hashlib.sha256((ROOT/'src/transformer_based_adaptive_line_enhancer/aet_dle.py').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_sha256':{scene:hashlib.sha256((BASE/'cases'/f'{scene}_snr_-20_seed_42'/'signals.npz').read_bytes()).hexdigest() for scene in args.scenes},
        'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','matplotlib']}}
    json_write(output/'manifest.json',manifest)
    all_rows,summaries=[],{}
    for scene in args.scenes:
        rows,summary=run_scene(scene,args,output); all_rows.extend(rows); summaries[scene]=summary
    csv_write(output/'per_epoch_metrics.csv',all_rows)
    json_write(output/'summary.json',summaries)
    plot(all_rows,summaries,args,output); report(all_rows,summaries,args,output)
    print(f'COMPLETE {len(all_rows)} per-epoch observations saved to {output}',flush=True)


if __name__=='__main__':
    main()
