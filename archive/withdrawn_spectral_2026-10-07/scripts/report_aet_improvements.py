"""Write a Chinese report from completed, frozen AET improvement experiments."""
import csv
import json
from pathlib import Path

import evaluate_aet_improvements as exp
import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def main():
    output=exp.OUT
    phases=['pilot','holdout','offgrid']
    acceptance={phase:json.loads((output/phase/'acceptance.json').read_text(encoding='utf-8')) for phase in phases}
    controls=read_csv(output/'shared_head_control.csv')
    holdout=read_csv(output/'holdout/summary.csv')
    comparison=read_csv(output/'holdout/comparisons.csv')
    pilot=read_csv(output/'pilot/summary.csv')
    validation=json.loads((output/'validation.json').read_text(encoding='utf-8'))
    regularized=acceptance['pilot']['AET_regularized']
    tested=acceptance['holdout']['AET_spectral']
    stress=acceptance['offgrid']['AET_spectral']
    text=['# AET 提升方案：按 SNR 的 dB 数值提高 10% 的实测验收','',
        '日期：2026-10-07（Asia/Shanghai）。先用原种子试验，再冻结方法与参数，最后运行新的噪声/模型种子和非整数频率测试。','',
        '## 结论','',
        f'可用方案是给原 AET 加上“全局自动谱线检测 + 频域约束输出 + 谱线残差旁路”。在当前平稳正弦谱线加白噪声的实验范围内，独立种子测试 {tested["passed_cases"]}/{tested["case_count"]} 个案例达到你要求的 10% 标准；非整数频率、随机相位/幅度的额外测试 {stress["passed_cases"]}/{stress["case_count"]} 个案例达标。',
        '',
        '**不能将这些有限测试称为对所有信号的保证。** 改善的主要证据指向全局谱线先验与更长观测，尚未证明 Transformer 本身稳定领先于获得相同频域处理的其他方法。纯频域方法和 AE 加相同输出头也必须作为对照。',
        '', '## 你指定的 10% 口径','',
        '- 对每一条观测，先取原 ALE、原 AE（60 轮）、原 AET（60 轮）中输出 SNR 最高者作为基线；试验阶段还额外纳入上一轮 AET 验证早停结果。这比只要求领先 ALE/AE 更严格。',
        '- 按 dB 数值计算：`改进输出 SNR − 最强基线输出 SNR ≥ 0.10 × |最强基线输出 SNR|`，并要求变化为正。',
        '- 例如基线 16 dB，需至少达到 17.6 dB；基线 −3 dB，需至少达到 −2.7 dB。基线为 0 dB 时百分比没有定义，只能单列绝对变化。本批观测没有恰为 0 的最强基线。',
        '- 这是你选定的 dB 数值比例，并不等于线性信噪比提高 10%。同时保留 MSE 变化供复核，没有用 MSE 口径替代本次验收标准。',
        '- 主要验收逐案例进行，不能用平均值掩盖某个种子失败。图中 margin = 实际 SNR 增量 − 要求的 10% 增量，必须逐点大于等于零。',
        '', '## 已测试的改动','',
        '| 方法 | 改动 | 原种子试验达标案例 | 判断 |','|---|---|---:|---|',
        f'| AET_regularized | RMS 归一化、线性输出、dropout=0.05、AdamW 衰减 0.01、梯度裁剪、最少 40 轮后验证早停 | {regularized["passed_cases"]}/{regularized["case_count"]} | 本次组合未稳定达标，不推荐这组参数 |',
        f'| AET_spectral | 原 AET 不重新调参，增加自动全局谱线检测、投影和残差输出头 | {acceptance["pilot"]["AET_spectral"]["passed_cases"]}/30 | 进入独立测试 |',
        f'| spectral_only | 移除 AET，仅使用同一自动谱线重建 | {acceptance["pilot"]["spectral_only"]["passed_cases"]}/30 | 关键消融对照 |',
        '', '第一组改动只代表上述参数组合的实测结果，不能据此推断所有正则化、线性输出或归一化方案均无效。其网络仍能在高度重叠训练帧上拟合目标噪声；增加最低训练轮次并没有自动解决弱谱线学习与过拟合的矛盾。',
        '', '## 推荐方案的具体实现','',
        '1. 从整段带噪观测计算 Hann 窗、8 倍零填充的频谱，以中位数估计噪声底，按名义全族误警水平 0.01 设峰值门限。每检出一条线先拟合并移除，再找下一条，防止强线旁瓣被重复当作谱线；最多允许 24 条，不预先指定真实条数。',
        '2. 用峰附近的对数谱插值估计频率，再对带噪观测做多谱线联合正弦/余弦最小二乘重建。算法不读取 50/100/180 Hz 频率、真实幅度、干净参考或请求的 SNR。',
        '3. 把原 AET 输出投影到同一自动检测子空间，利用有效输出区间拟合幅相。最终 `y = 0.95 × P(noisy) + 0.05 × P(AET)`。这个全局谱线残差旁路避免弱线必须先被原网络学到；5% 神经分支权重在独立测试前固定。',
        '4. 原 AET 网络、训练数据、60 轮训练与推理规则保持不变。这个改动属于增加频域输出头和残差旁路，没有将测试干净标签用于训练或选择混合权重。无需新增预训练数据。',
        '',
        '实现位于 `src/transformer_based_adaptive_line_enhancer/aet_improved.py` 的 `spectral_residual_head`。它接收带噪观测及 AET 原输出，可直接接到现有推理后。返回自动检测频率、门限、最终输出和纯频域消融输出；无谱线达到门限时显式标记 `no_line_detected=True`。',
        '', '## 为什么能改善当前数据','',
        '- 原 AET 的单个输入序列覆盖 768 点。本方案利用整段 16000 点进行谱线检测和幅相估计；观测更长，可对相干谱线积累证据，同时削弱非相干噪声。没有新增观测，但有效感受范围大幅增加。',
        '- 原解码器输出自由度高，能把训练观测中的随机噪声写入波形。新输出限制在自动检测到的少数谱线子空间，减少谱线外误差。',
        '- 旁路保留输入中长期相干的谱线，缓解之前早停模型将弱信号压低的问题；这同时引入了“信号是全局平稳、稀疏谱线”的强先验。',
        '- 已知且正确的 K 条频率、IID 白噪声、正交长观测条件下，2K 维正弦/余弦最小二乘投影的期望噪声 MSE 约为 `2K × noise_variance / N`。本实现的频率是从噪声中估计的，且会漏检/误检，因此这项条件性计算不是本方法无条件的误差保证。',
        '', '## 冻结后的独立测试','',
        '- 原种子试验：noise_seed=42/43/44，model_seed=0/1/2，共 30 个案例。原结果用于分析方案，不作为最后的泛化证据。',
        '- 独立主测试：noise_seed=100..109，model_seed=1100..1109；单/多谱线 × −20/−15/−10/−5/0 dB，共 100 个案例。每个案例重新训练原 AE/AET，所有算法输入相同。',
        '- 额外频率测试：noise_seed=200/201/202，model_seed=1200/1201/1202，共 30 个案例；频率不局限于整数 FFT 栅格，单线从 70..130 Hz 随机取值，多线从 35..65/85..120/160..205 Hz 取值，幅度按原幅度乘 0.8..1.2、相位在 −π..π 随机生成。',
        '- 1 kHz、16000 点、power-normalized 高斯白噪声；公共评价区间 [8000,14976)。SNR 定义和原实验完全一致。',
        '- 这是每段观测训练再增强该观测的实验，测试种子与调参种子分离；并不是固定预训练模型对完全未参与适配的观测做一次推理的测试。',
        '- 所有测试在运行前锁定配置，锁定记录见 `selection_lock.json`。没有看到测试结果后调整谱线门限或神经分支权重。',
        '', '## 主测试结果：输出 SNR 均值（dB）','',
        '| 场景 | 输入 SNR | ALE | AE | 原 AET | AET + 频域输出头 | 纯频域重建 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for scene in ['single','multiline']:
        for snr in [-20,-15,-10,-5,0]:
            values=[next(r for r in holdout if r['scene']==scene and float(r['requested_snr_db'])==snr and r['algorithm']==name) for name in exp.ALGORITHMS]
            text.append(f'| {scene} | {snr} | '+' | '.join(f'{float(r["output_snr_db_mean"]):.3f}' for r in values)+' |')
    text += ['', '![独立种子主测试](holdout/comparison.png)','',
        '| 测试范围 | 逐案例达标 | 最小验收余量 | 最小 dB 数值相对提升 |',
        '|---|---:|---:|---:|',
        f'| 主测试 | {tested["passed_cases"]}/{tested["case_count"]} | {tested["minimum_margin_db"]:.3f} dB | {tested["minimum_relative_db_improvement_percent"]:.2f}% |',
        f'| 非整数频率/随机相位 | {stress["passed_cases"]}/{stress["case_count"]} | {stress["minimum_margin_db"]:.3f} dB | {stress["minimum_relative_db_improvement_percent"]:.2f}% |',
        '', '![逐案例验收余量](holdout/acceptance_margin.png)','', '## 消融揭示的限制','']
    deltas=[float(r['AET_vs_AE_with_same_head_db']) for r in controls if r['phase']=='holdout']
    aet_beats=sum(value>0 for value in deltas)
    pure_deltas=[]
    for row in comparison:
        if row['algorithm']=='AET_spectral':
            pure=next(r for r in comparison if r['algorithm']=='spectral_only' and r['scene']==row['scene'] and r['requested_snr_db']==row['requested_snr_db'] and r['noise_seed']==row['noise_seed'])
            pure_deltas.append(float(row['output_snr_db'])-float(pure['output_snr_db']))
    text += [f'在主测试中，给 AE 加相同输出头后，AET+输出头只在 {aet_beats}/{len(deltas)} 个案例 SNR 更高；两者平均差为 {np.mean(deltas):+.4f} dB，范围 [{min(deltas):+.4f}, {max(deltas):+.4f}] dB。',
        '', f'相对纯频域消融，AET+输出头平均差为 {np.mean(pure_deltas):+.4f} dB，范围 [{min(pure_deltas):+.4f}, {max(pure_deltas):+.4f}] dB。',
        '',
        '**所以，已验证的是“改动后的完整 AET 流程领先原 ALE/AE/AET”，而不是“Transformer 架构稳定领先其他同样改动过的方法”。** 如果项目目标只有当前平稳谱线 SNR，纯频域对照也是值得采用的候选；如果目标是证明 Transformer 的独立贡献，本次证据不够。',
        '', '## 能否确保以后仍有 10% 提升','',
        '只能给出上述冻结测试中的通过记录，不能给出无限样本或所有实际工况的绝对保证。白噪声随机误差没有有限幅值上界；基线在某条观测上已接近最佳估计时，任意固定方法都未必还能提高指定比例。',
        '',
        '同一个噪声种子跨 SNR/场景的结果相关，100 个案例不能被当成 100 次完全独立的成功试验。若按 10 个独立种子分组、每组都成功来做二项事件近似，10/10 成功对应的单侧 95% 精确成功概率下界仅约 74.1%，不足以证明几乎必然成功。这一数字只在分组独立同分布假设下成立。',
        '',
        '稳定频率突然变化、啁啾、短瞬态、宽带目标、密集/近邻谱线、非白或含干扰谱线的噪声，均可能破坏全局稀疏谱线先验。低 SNR 下漏检弱线也仍存在：达到整体 SNR 指标不意味着全部谱线都被恢复。近实时应用还需接受使用完整 16 秒观测的等待和计算方式。',
        '', '## 下一步建议','',
        '先将频域输出头作为可开关的实验功能，用真实任务中的同一评价口径验收；保留纯频域和 AE+相同输出头对照。若需要非平稳信号能力，下一步测试分段谱线跟踪、频率连续性约束，以及真实谱线/噪声分布的预训练；这些尚未验证，不能承诺达到 10%。不要仅为了保留 AET 名称而把频域对照的收益解释成 Transformer 收益。',
        '', '## 验证与交付','',
        f'- 已核验 {sum(r["case_count"] for r in validation["phases"])} 个案例，逐条重算 SNR/MSE/幅值保留和 10% 判定；保存信号与指标最大差异为 {validation["maximum_metric_difference"]}。',
        f'- 自动谱线头已从保存输入重放；代表性原 AET 检查点与保存预测的最大差为 {validation["maximum_original_checkpoint_replay_difference"]}。',
        '- `pilot/`：方案试验；`holdout/`：独立主测试；`offgrid/`：频率/相位测试。各目录包含原始信号、指标、基线、门限、自动检测频率、协议哈希和图。',
        '- `shared_head_control.csv`：AE 使用同样频域输出头的对照。`validation.json`：核验结果。',
        '- `selection_lock.json`：测试前锁定的方法、参数与种子。`tests/test_aet_improved.py`：未提供频率的自动检测、旁瓣去除、输出边界与输入验证检查。',
        '', '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/evaluate_aet_improvements.py --phase pilot --regularized --resume',
        '.\\.venv\\Scripts\\python.exe scripts/evaluate_aet_improvements.py --phase holdout --resume',
        '.\\.venv\\Scripts\\python.exe scripts/evaluate_aet_improvements.py --phase offgrid --resume',
        '.\\.venv\\Scripts\\python.exe scripts/validate_aet_improvements.py',
        '.\\.venv\\Scripts\\python.exe scripts/report_aet_improvements.py', '```','',
        '## 文献与本实验的关系','',
        '频域稀疏表示与分解后建模的方向参考 [FEDformer 原论文](https://proceedings.mlr.press/v162/zhou22g.html)；它研究时间序列预测，不为本项目的去噪性能提供保证，本实现也不是 FEDformer 的复现。',
        '',
        '独立噪声重复观测下使用带噪目标的动机参考 [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html) 和 [Noise2Self](https://proceedings.mlr.press/v97/batson19a.html)。本次最终提升结论来自本项目保存的实测数据，不从论文迁移百分比。','']
    (output/'AET_IMPROVEMENT_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print(output/'AET_IMPROVEMENT_REPORT.md')


if __name__=='__main__':main()
