"""Reproduce the repository algorithms on shared observations and SNR sweeps.

Run from the repository root with the project environment. The original model
implementations are imported unchanged. Every case saves data before plotting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_based_adaptive_line_enhancer import (
    AEDLEConfig, AETDLEConfig, adaptive_line_enhance, generate_noisy_signal,
    train_ae_dle, train_aet_dle, enhance_with_ae_dle, enhance_with_aet_dle,
)
from transformer_based_adaptive_line_enhancer.visualization import _spectrogram_db

ALGORITHMS = ("ALE", "AE", "AET")
COLORS = {"ALE": "#2878b5", "AE": "#e58b27", "AET": "#3d9d64"}
SCENES = {
    "single": {"frequencies_hz": [100.0], "amplitudes": [1.0]},
    "multiline": {"frequencies_hz": [50.0, 100.0, 180.0], "amplitudes": [1.0, 0.6, 0.3]},
}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(reference**2) / np.mean((estimate-reference)**2)))


def tone_fit(values: np.ndarray, times: np.ndarray, frequencies: list[float]) -> np.ndarray:
    columns = [np.ones(times.size)]
    for frequency in frequencies:
        columns.extend([np.sin(2*np.pi*frequency*times), np.cos(2*np.pi*frequency*times)])
    coefficients = np.linalg.lstsq(np.column_stack(columns), values, rcond=None)[0]
    return coefficients


def timed_call(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter()-start


def run_case(scene: str, requested: float, seed: int, model_seed: int,
             args, output: Path, manifest: dict) -> dict:
    label = f"{scene}_snr_{requested:g}_seed_{seed}"
    case_dir = output / "cases" / label
    case_dir.mkdir(parents=True, exist_ok=True)
    marker = case_dir / "metrics.json"
    if args.resume and marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["protocol_hash"] != manifest["protocol_hash"]:
            raise ValueError(f"Resume protocol mismatch: {label}")
        print(f"REUSE {label}", flush=True)
        return saved
    sample = generate_noisy_signal(**SCENES[scene], sample_rate_hz=1000.0,
        num_samples=16000, snr_db=requested, seed=seed, scaling="power-normalized")
    ae_config = AEDLEConfig(epochs=args.epochs, seed=model_seed)
    aet_config = AETDLEConfig(epochs=args.epochs, seed=model_seed)
    assert ae_config.prediction_delay >= ae_config.frame_length
    assert aet_config.prediction_delay >= (aet_config.frame_length
        + (aet_config.sequence_length-1)*aet_config.frame_hop)
    assert aet_config.sequence_stride == 1
    print(f"START {label}", flush=True)
    ale, ale_seconds = timed_call(adaptive_line_enhance, sample.noisy_signal)
    print(f"  ALE {ale_seconds:.2f}s", flush=True)
    ae_training, ae_train_seconds = timed_call(train_ae_dle, sample.noisy_signal, config=ae_config)
    ae, ae_inference_seconds = timed_call(enhance_with_ae_dle, sample.noisy_signal, ae_training)
    print(f"  AE train={ae_train_seconds:.2f}s inference={ae_inference_seconds:.2f}s", flush=True)
    aet_training, aet_train_seconds = timed_call(train_aet_dle, sample.noisy_signal, config=aet_config)
    aet, aet_inference_seconds = timed_call(enhance_with_aet_dle, sample.noisy_signal, aet_training)
    print(f"  AET train={aet_train_seconds:.2f}s inference={aet_inference_seconds:.2f}s", flush=True)
    start = max(8000, ale.adaptation_start, ae.valid_start, aet.valid_start)
    end = min(16000, ae.valid_end, aet.valid_end)
    assert end > start
    valid = slice(start, end)
    reference = sample.clean_signal[valid]
    input_snr = snr_db(reference, sample.noisy_signal[valid])
    times = sample.time_s[valid]
    frequencies = SCENES[scene]["frequencies_hz"]
    clean_fit = tone_fit(reference, times, frequencies)
    metrics, tones = [], []
    results = {"ALE": ale, "AE": ae, "AET": aet}
    timing = {"ALE": (0.0, ale_seconds), "AE": (ae_train_seconds, ae_inference_seconds),
              "AET": (aet_train_seconds, aet_inference_seconds)}
    parameter_counts = {"ALE": 32, "AE": sum(p.numel() for p in ae_training.model.parameters()),
                        "AET": sum(p.numel() for p in aet_training.model.parameters())}
    for name, result in results.items():
        estimate = result.enhanced_signal[valid]
        assert np.all(np.isfinite(estimate))
        np.testing.assert_allclose(result.residual[valid], sample.noisy_signal[valid]-estimate)
        output_snr = snr_db(reference, estimate)
        fit = tone_fit(estimate, times, frequencies)
        ratios = []
        for i, frequency in enumerate(frequencies):
            index = 1+2*i
            clean_complex = complex(clean_fit[index], clean_fit[index+1])
            output_complex = complex(fit[index], fit[index+1])
            ratio = abs(output_complex)/abs(clean_complex)
            phase_error = float(np.angle(output_complex/clean_complex, deg=True))
            ratios.append(ratio)
            tones.append({"scene": scene, "requested_snr_db": requested, "noise_seed": seed,
                "model_seed": model_seed, "algorithm": name, "frequency_hz": frequency,
                "clean_amplitude": abs(clean_complex), "output_amplitude": abs(output_complex),
                "amplitude_retention": ratio, "amplitude_error_percent": 100*(ratio-1),
                "phase_error_deg": phase_error})
        metrics.append({"scene": scene, "requested_snr_db": requested, "noise_seed": seed,
            "model_seed": model_seed, "algorithm": name, "full_input_snr_db": sample.measured_snr_db,
            "evaluation_input_snr_db": input_snr, "output_snr_db": output_snr,
            "snr_gain_db": output_snr-input_snr, "rmse": float(np.sqrt(np.mean((estimate-reference)**2))),
            "weakest_tone_retention": ratios[-1], "evaluation_start": start, "evaluation_end": end,
            "training_seconds": timing[name][0], "inference_seconds": timing[name][1],
            "total_seconds": sum(timing[name]), "parameter_count": parameter_counts[name]})
    np.savez_compressed(case_dir / "signals.npz", time_s=sample.time_s, clean=sample.clean_signal,
        noisy=sample.noisy_signal, noise=sample.noise, ALE=ale.enhanced_signal,
        AE=ae.enhanced_signal, AET=aet.enhanced_signal, evaluation_start=start, evaluation_end=end)
    save_json(case_dir / "loss_history.json", {"AE": ae_training.loss_history, "AET": aet_training.loss_history})
    if seed == args.seeds[0]:
        for name, trained in [("AE", ae_training), ("AET", aet_training)]:
            torch.save({"state_dict": trained.model.state_dict(), "config": asdict(trained.config),
                "scaler": asdict(trained.scaler), "loss_history": trained.loss_history,
                "protocol_hash": manifest["protocol_hash"]}, case_dir / f"{name}_checkpoint.pt")
        np.save(case_dir / "ALE_final_weights.npy", ale.final_weights)
    payload = {"protocol_hash": manifest["protocol_hash"], "metrics": metrics, "tones": tones}
    save_json(marker, payload)
    print("DONE " + label + " " + " ".join(f"{m['algorithm']}={m['output_snr_db']:.2f}dB" for m in metrics), flush=True)
    return payload


def aggregate(rows: list[dict], snrs: list[float], scenes: list[str]) -> list[dict]:
    output = []
    columns = ["evaluation_input_snr_db", "output_snr_db", "snr_gain_db", "rmse",
               "weakest_tone_retention", "training_seconds", "inference_seconds", "total_seconds"]
    for scene in scenes:
        for requested in snrs:
            for name in ALGORITHMS:
                selected = [r for r in rows if r["scene"] == scene
                    and r["requested_snr_db"] == requested and r["algorithm"] == name]
                row = {"scene": scene, "requested_snr_db": requested, "algorithm": name,
                       "seeds": len(selected), "parameter_count": selected[0]["parameter_count"]}
                for column in columns:
                    values = [r[column] for r in selected]
                    row[column+"_mean"] = float(np.mean(values))
                    row[column+"_std"] = float(np.std(values, ddof=1)) if len(values)>1 else 0.0
                output.append(row)
    return output


def plot_curves(summary: list[dict], args, output: Path) -> None:
    fig, axes = plt.subplots(2, len(args.scenes), figsize=(7*len(args.scenes), 8), squeeze=False,
                             constrained_layout=True)
    for j, scene in enumerate(args.scenes):
        for name in ALGORITHMS:
            selected = [r for r in summary if r["scene"] == scene and r["algorithm"] == name]
            x = [r["requested_snr_db"] for r in selected]
            for i, metric in enumerate(["output_snr_db", "snr_gain_db"]):
                axes[i,j].errorbar(x, [r[metric+"_mean"] for r in selected],
                    yerr=[r[metric+"_std"] for r in selected], marker="o", capsize=4,
                    color=COLORS[name], label=name)
        axes[0,j].plot(args.snrs, args.snrs, "--", color="gray", label="Requested input SNR")
        for i, ylabel in enumerate(["Output SNR (dB)", "SNR gain (dB)"]):
            axes[i,j].set(title=scene, xlabel="Requested input SNR (dB)", ylabel=ylabel)
            axes[i,j].grid(alpha=0.25)
            axes[i,j].legend()
    fig.suptitle(f"Original algorithms | {args.epochs} epochs | mean +/- sample SD ({len(args.seeds)} seeds)")
    for extension in ["png", "svg"]:
        fig.savefig(output / f"snr_comparison.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(args.scenes), figsize=(7*len(args.scenes),4.5), squeeze=False,
                             constrained_layout=True)
    for j, scene in enumerate(args.scenes):
        for name in ALGORITHMS:
            selected = [r for r in summary if r["scene"]==scene and r["algorithm"]==name]
            axes[0,j].errorbar([r["requested_snr_db"] for r in selected],
                [100*r["weakest_tone_retention_mean"] for r in selected],
                yerr=[100*r["weakest_tone_retention_std"] for r in selected],
                marker="o", capsize=4, color=COLORS[name], label=name)
        axes[0,j].axhline(100, linestyle="--", color="gray", label="Correct amplitude")
        axes[0,j].set(title=scene, xlabel="Requested input SNR (dB)", ylabel="Weakest tone amplitude retention (%)")
        axes[0,j].grid(alpha=0.25)
        axes[0,j].legend()
    fig.savefig(output / "weakest_tone_retention.png", dpi=180)
    plt.close(fig)


def plot_cases(args, output: Path) -> None:
    names = ("clean", "noisy", *ALGORITHMS)
    for scene in args.scenes:
        stored = []
        for requested in args.snrs:
            path = output / "cases" / f"{scene}_snr_{requested:g}_seed_{args.seeds[0]}"
            with np.load(path / "signals.npz") as data:
                signals = {key: data[key].copy() for key in names}
                start, end = int(data["evaluation_start"]), int(data["evaluation_end"])
                spectra = [_spectrogram_db(signals[key][start:end], sample_rate_hz=1000.0,
                    n_fft=256, hop_length=64, offset_s=start/1000) for key in names]
                stored.append((requested, signals, start, end, spectra))
        maximum = max(float(np.max(spectrum[2])) for record in stored for spectrum in record[4])
        fig, axes = plt.subplots(len(args.snrs), 5, figsize=(19, 3*len(args.snrs)),
            sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        for i, (requested, signals, start, end, spectra) in enumerate(stored):
            for j, (name, (times, frequencies, power)) in enumerate(zip(names, spectra)):
                mask = frequencies <= 250
                artist = axes[i,j].pcolormesh(times, frequencies[mask], power[mask],
                    shading="nearest", cmap="magma", vmin=maximum-70, vmax=maximum)
                if i==0: axes[i,j].set_title(name)
                if j==0: axes[i,j].set_ylabel(f"SNR {requested:g} dB\nFrequency (Hz)")
                if i==len(args.snrs)-1: axes[i,j].set_xlabel("Time (s)")
        fig.colorbar(artist, ax=axes, label="Power (dB); one shared scale for this figure", shrink=0.7)
        fig.suptitle(f"{scene} | identical observations, seed {args.seeds[0]} | common evaluation interval")
        fig.savefig(output / f"{scene}_spectrogram_comparison.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(len(args.snrs), 1, figsize=(12, 2.6*len(args.snrs)),
            squeeze=False, constrained_layout=True)
        for i, (requested, signals, start, end, spectra) in enumerate(stored):
            clip = slice(end-200, end)
            times = np.arange(end-200, end)/1000
            axes[i,0].plot(times, signals["clean"][clip], color="black", linewidth=1.6, label="Clean")
            for name in ALGORITHMS:
                axes[i,0].plot(times, signals[name][clip], color=COLORS[name], alpha=0.8, label=name)
            axes[i,0].set(title=f"Input SNR {requested:g} dB", ylabel="Amplitude")
            axes[i,0].grid(alpha=0.2)
            if i==0: axes[i,0].legend(ncol=4)
        axes[-1,0].set_xlabel("Time (s)")
        fig.savefig(output / f"{scene}_waveform_comparison.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(len(args.snrs), 1, figsize=(12, 2.6*len(args.snrs)),
            squeeze=False, constrained_layout=True)
        for i, (requested, signals, start, end, spectra) in enumerate(stored):
            for name, (times, frequencies, power) in zip(names, spectra):
                mean_db = 10*np.log10(np.mean(10**(power/10), axis=1))
                axes[i,0].plot(frequencies, mean_db, label=name,
                    color=COLORS.get(name, "black" if name=="clean" else "gray"), alpha=0.8)
            axes[i,0].set(xlim=(0,250), title=f"Input SNR {requested:g} dB", ylabel="Mean power (dB)")
            axes[i,0].grid(alpha=0.2)
            if i==0: axes[i,0].legend(ncol=5)
        axes[-1,0].set_xlabel("Frequency (Hz)")
        fig.savefig(output / f"{scene}_spectrum_comparison.png", dpi=160)
        plt.close(fig)


def write_report(summary: list[dict], args, output: Path, manifest: dict) -> None:
    lines = ["# ALE / AE / AET 随 SNR 变化的复现对比", "",
        "日期：2026-10-07（Asia/Shanghai）。本报告由实测数据自动生成。", "",
        "## 实验协议", "",
        f"- Git 基线：`{manifest['git_commit']}`；直接调用原项目三种算法，未修改模型实现。",
        "- 单谱线：100 Hz，幅度 1。多谱线：50/100/180 Hz，幅度 1/0.6/0.3。",
        f"- 输入 SNR：{args.snrs} dB；噪声种子：{args.seeds}，模型种子依次为 0/1/2（按种子列表索引）。",
        "- 采样率 1 kHz、长度 16,000 点、高斯白噪声、power-normalized 缩放。",
        f"- AE/AET 保留原网络默认结构；每段观测从随机初始化训练 {args.epochs} 轮，然后增强同一段观测。",
        "- 所有算法使用完全相同的观测；评价从第 8,000 点开始，取三种算法有效区间交集。",
        "- 延迟 1,000 点，AE 帧长 256，AET 序列覆盖 768 点；单个训练对输入与目标不共享采样点。",
        f"- CPU 运行，PyTorch 线程数 {args.threads}。运行环境见 environment.json。",
        "- SNR = 10 log10(mean(clean²) / mean((output-clean)²))；增益相对于同一评价区间的带噪输入。",
        "- 幅值保留：在已知频率处同时拟合正弦/余弦及常数项，再用输出幅值除以干净参考幅值；频率信息仅用于外部评价。",
        "- 误差棒为种子之间的样本标准差，不是置信区间；3 个种子只反映有限场景波动。", "",
        "## 输出 SNR（均值 ± 样本标准差，dB）", "",
        "| 场景 | 请求输入 SNR | ALE | AE | AET |", "|---|---:|---:|---:|---:|"]
    for scene in args.scenes:
        for requested in args.snrs:
            chosen = {r["algorithm"]:r for r in summary if r["scene"]==scene and r["requested_snr_db"]==requested}
            cells = [f"{chosen[name]['output_snr_db_mean']:.2f} ± {chosen[name]['output_snr_db_std']:.2f}" for name in ALGORITHMS]
            lines.append(f"| {scene} | {requested:g} | " + " | ".join(cells) + " |")
    lines += ["", "## 最弱谱线幅值保留（均值，%）", "",
        "单谱线为 100 Hz；多谱线为幅度最弱的 180 Hz。100% 表示幅值正确，超过 100% 表示放大，不能一律视为更好。", "",
        "| 场景 | 请求输入 SNR | ALE | AE | AET |", "|---|---:|---:|---:|---:|"]
    for scene in args.scenes:
        for requested in args.snrs:
            chosen = {r["algorithm"]:r for r in summary if r["scene"]==scene and r["requested_snr_db"]==requested}
            cells = [f"{100*chosen[name]['weakest_tone_retention_mean']:.1f}" for name in ALGORITHMS]
            lines.append(f"| {scene} | {requested:g} | " + " | ".join(cells) + " |")
    lines += ["", "## 结果解读", ""]
    for scene in args.scenes:
        selected = [r for r in summary if r["scene"]==scene]
        wins = {name:0 for name in ALGORITHMS}
        for requested in args.snrs:
            winner = max((r for r in selected if r['requested_snr_db']==requested), key=lambda r:r['output_snr_db_mean'])
            wins[winner['algorithm']] += 1
        lines.append(f"- {scene} 场景中，按平均输出 SNR 比较，最高值次数为 " + "、".join(f"{name} {wins[name]}/{len(args.snrs)}" for name in ALGORITHMS) + "。这只是本协议下的结果。")
    for name in ALGORITHMS:
        selected = [r for r in summary if r['algorithm']==name]
        lines.append(f"- {name}：参数数 {selected[0]['parameter_count']:,}；平均单次总耗时 {np.mean([r['total_seconds_mean'] for r in selected]):.2f} 秒（训练与增强合计）。")
    lines += ["", "## 图表", "", "![SNR 对比](snr_comparison.png)", "",
        "![最弱谱线保留](weakest_tone_retention.png)", ""]
    for scene in args.scenes:
        lines += [f"### {scene}", "", f"![时频图]({scene}_spectrogram_comparison.png)", "",
                  f"![频谱]({scene}_spectrum_comparison.png)", "", f"![波形]({scene}_waveform_comparison.png)", ""]
    lines += ["## 适用范围与局限", "",
        "- 这是原模型的统一观测对比，不是参数量或算力预算匹配的架构消融；AE 和 AET 的参数数、训练批量与优化步数不同。",
        "- 本实验是同段自适应增强，未验证跨观测泛化、真实数据或有色噪声；同段训练不代表独立测试集效果。",
        "- 当前频率为整数 Hz，1 秒延迟恰好跨越整数周期；不能推断非整数频率、频漂或非平稳信号效果。",
        "- AET 使用未来观测且没有因果注意力掩码，结果不代表实时因果处理能力。",
        "- 逐谱线幅值指标用于暴露弱线抑制；总 SNR 提升不自动证明检测率或弱线保留提升。",
        "- 当前 pytest 中仍含输入/目标重叠的小配置；测试通过不消除此前接手报告指出的研究风险。", "",
        "## 产物与复跑", "",
        "- `metrics.csv`：每个算法、SNR、种子的 SNR/RMSE/耗时等原始指标。",
        "- `tone_metrics.csv`：逐谱线幅值、相位和保留率。",
        "- `summary.csv`：跨种子的均值与样本标准差。",
        "- `manifest.json`、`environment.json`：协议、源文件哈希与环境。",
        "- `cases/`：30 组场景的完整信号 NPZ、损失曲线 JSON；首个种子另存模型权重和 ALE 最终权重。",
        "- `loss_comparison.png`：首个种子各场景/SNR 的训练损失。",
        "- `pytest.log`：环境建立后执行的原项目测试记录。", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe scripts/reproduce_comparison.py --epochs 60 --seeds 42 43 44 --resume", "```", "",
        "各 case 在训练完成后立即保存；中断后相同协议可恢复。运行不同协议请使用另一个 --output 目录。"]
    (output / "REPRODUCTION_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    parser.add_argument("--snrs", type=float, nargs="+", default=[-20,-15,-10,-5,0])
    parser.add_argument("--scenes", nargs="+", choices=list(SCENES), default=list(SCENES))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduction_2026-10-07")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs<1 or args.threads<1 or len(set(args.seeds))!=len(args.seeds):
        parser.error("Positive epochs/threads and distinct seeds are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").rglob("*.py"))}
    protocol = {"epochs": args.epochs, "seeds": args.seeds, "snrs": args.snrs,
        "scenes": args.scenes, "scene_configs": SCENES, "threads": args.threads,
        "sample_rate_hz":1000, "num_samples":16000, "evaluation_from":8000,
        "source_hashes": sources, "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest = {**protocol, "protocol_hash":protocol_hash,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "ae_config":asdict(AEDLEConfig(epochs=args.epochs)),
        "aet_config":asdict(AETDLEConfig(epochs=args.epochs))}
    previous = output / "manifest.json"
    if previous.exists() and json.loads(previous.read_text(encoding="utf-8"))["protocol_hash"]!=protocol_hash:
        raise ValueError("Output directory contains a different experiment protocol")
    save_json(previous, manifest)
    save_json(output / "environment.json", {"python":sys.version, "platform":platform.platform(),
        "processor":platform.processor(), "logical_cpus":os.cpu_count(), "torch_threads":args.threads,
        "cuda_available":torch.cuda.is_available(), "execution_device":"cpu",
        "packages":{name:importlib.metadata.version(name) for name in
            ["numpy","torch","matplotlib","pytest","jupyter","ipykernel"]}})
    all_metrics, all_tones = [], []
    for scene in args.scenes:
        for requested in args.snrs:
            for model_seed, seed in enumerate(args.seeds):
                case = run_case(scene, requested, seed, model_seed, args, output, manifest)
                all_metrics.extend(case["metrics"])
                all_tones.extend(case["tones"])
                save_csv(output / "metrics.csv", all_metrics)
                save_csv(output / "tone_metrics.csv", all_tones)
    summary = aggregate(all_metrics, args.snrs, args.scenes)
    save_csv(output / "summary.csv", summary)
    plot_curves(summary, args, output)
    plot_cases(args, output)
    fig, axes = plt.subplots(len(args.scenes), len(args.snrs), figsize=(4*len(args.snrs),4*len(args.scenes)),
        squeeze=False, constrained_layout=True)
    for i, scene in enumerate(args.scenes):
        for j, requested in enumerate(args.snrs):
            history = json.loads((output / "cases" / f"{scene}_snr_{requested:g}_seed_{args.seeds[0]}" / "loss_history.json").read_text())
            for name, values in history.items():
                axes[i,j].plot(np.arange(1,len(values)+1),values,color=COLORS[name],label=name)
            axes[i,j].set(title=f"{scene}, {requested:g} dB",xlabel="Epoch",ylabel="Normalized noisy-target MSE")
            axes[i,j].grid(alpha=0.2)
            axes[i,j].legend()
    fig.savefig(output / "loss_comparison.png", dpi=160)
    plt.close(fig)
    write_report(summary,args,output,manifest)
    print(f"COMPLETE {len(all_metrics)} metric rows; {len(all_tones)} tone rows; {output}",flush=True)


if __name__ == "__main__":
    main()
