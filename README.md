# Transformer-based Adaptive Line Enhancer

从零实现谱线增强方法的 Python 项目。当前阶段完成了与参考 MATLAB 脚本一致的带噪窄带信号生成。

## 安装

```bash
uv sync --dev
```

## 生成带噪信号

```python
from transformer_based_adaptive_line_enhancer import generate_noisy_signal

sample = generate_noisy_signal(
    frequencies_hz=100.0,
    sample_rate_hz=1_000.0,
    num_samples=16_000,
    snr_db=-20.0,
    seed=42,
    scaling="legacy",
)

time_s = sample.time_s
clean = sample.clean_signal
noise = sample.noise
noisy = sample.noisy_signal
```

`scaling="legacy"` 对应 `data_p.m`、`data_zczs.m` 和
`ha_test_transformer.m` 的活跃代码：先从本次高斯噪声计算功率，再令
`a = sqrt(En * 10^(SNR/10))`，最后计算 `r = a*s + n`。由于原脚本没有除以
正弦信号功率，单位正弦的实测 SNR 会比传入值低约 3.01 dB。

`scaling="power-normalized"` 对应 `ale_signal.m`：比例中额外除以信号功率，
因此实测功率比等于传入的 `snr_db`。

NumPy 和 MATLAB 的随机数流不同，所以 Python 端在公式和计算顺序上保持一致，
但不会在相同 seed 下生成逐样点相同的 `randn` 序列。

## 可视化带噪信号

```python
import matplotlib.pyplot as plt

from transformer_based_adaptive_line_enhancer import (
    generate_noisy_signal,
    plot_noisy_signal_components,
)

sample = generate_noisy_signal(seed=42)
figure, axes = plot_noisy_signal_components(sample)

figure.savefig("noisy-signal-components.png", dpi=150)
plt.show()
```

该方法在三个独立子图中依次绘制 `clean_signal`、`noise` 和 `noisy_signal`。
函数本身不会显示窗口或写入文件，而是返回 Matplotlib 的 `Figure` 和三个 `Axes`，
由调用者决定继续调整、保存或显示。

## Delayed Adaptive Line Enhancement

从 ALE 阶段开始，本项目采用独立设计，不再复刻 MATLAB 实现。ALE 将当前带噪样本
$x[n]$ 作为期望响应，并以延迟后的输入构造参考向量：
$$
\mathbf{x}_{\Delta}[n]
= \begin{bmatrix}
x[n - \Delta] & x[n - \Delta - 1] & \cdots & x[n - \Delta - M + 1]
\end{bmatrix}^{\mathrm{T}}.
$$
滤波器预测值与残差分别为
$$
\hat{s}[n] = \mathbf{w}^{\mathrm{T}}[n]\mathbf{x}_{\Delta}[n],
\qquad
e[n] = x[n] - \hat{s}[n].
$$
延迟量 $\Delta$ 应超过噪声的主要相关长度，同时让窄带信号仍保持可预测性。默认采用
NLMS 更新，以降低输入幅度对步长的影响：
$$
\mathbf{w}[n + 1]
= \mathbf{w}[n]
+ \frac{\mu}{\varepsilon + \lVert\mathbf{x}_{\Delta}[n]\rVert_2^2}
e[n]\mathbf{x}_{\Delta}[n].
$$
实现要求 NLMS 步长满足 $0 < \mu < 2$；默认值 $\mu = 0.05$ 偏保守，以降低稳态失调。

```python
from transformer_based_adaptive_line_enhancer import adaptive_line_enhance

result = adaptive_line_enhance(
    noisy_signal,
    delay=10,
    filter_length=32,
    step_size=0.05,
    mode="nlms",
)

enhanced = result.enhanced_signal
residual = result.residual
weights = result.final_weights

# ALE 启动前没有完整的延迟参考向量；评价时使用有效区间。
enhanced_valid = enhanced[result.valid_slice]
```

`enhanced_signal` 是对可预测窄带成分的估计，`residual` 是当前输入减去预测值。
谱线更清晰并不自动等价于目标波形、检测概率或方位信息均得到改善，这些指标需要在后续
实验中分别验证。

## Autoencoder Deep Line Enhancement

AE-DLE 采用论文第 3.1 节的核心思想，但代码完全独立实现：从同一段带噪观测构造
“延迟帧 → 当前帧”训练对，通过瓶颈自编码器学习跨延迟仍可预测的窄带成分。训练 API
不接收干净标签；`clean_signal` 只用于受控实验中的外部评价。

```python
from transformer_based_adaptive_line_enhancer import (
    AEDLEConfig,
    enhance_with_ae_dle,
    train_ae_dle,
)

config = AEDLEConfig(
    frame_length=256,
    frame_hop=64,
    prediction_delay=1_000,  # 采样率为 1 kHz 时对应论文实验中的 1 s 延迟
    hidden_dim=128,
    bottleneck_dim=32,
    learning_rate=1e-3,
    batch_size=32,
    epochs=30,
    seed=0,
)

training = train_ae_dle(sample.noisy_signal, config=config)
result = enhance_with_ae_dle(sample.noisy_signal, training)

enhanced = result.enhanced_signal
residual = result.residual
enhanced_valid = enhanced[result.valid_slice]
loss_history = training.loss_history
```

实现使用两层 ReLU 编码器、对称解码器、MSE 损失和 Adam。论文报告学习率 0.01，
但没有完整给出优化器与软件细节；本实现独立选择更保守的 Adam 学习率 0.001。
推理时，每个输入帧预测其 `prediction_delay` 样本后的帧，再按目标时间位置重叠平均，
因此输出与原始时间轴对齐。超出训练最小-最大范围的推理输入会裁剪至训练归一化区间。

固定种子的单谱线白噪声试验验证了明显 SNR 增益，但这不是普遍性能保证。多谱线中，
瓶颈自编码器可能保留强谱线而压制弱谱线；必须逐谱线报告结果，不能只看总输出 SNR。

## Autoencoder-Transformer Deep Line Enhancement

AET-DLE 按论文第 3.2 节重新独立实现，没有读取或复用其他 AET-DLE 代码。模型先对
每个重叠矩形窗帧进行两层 ReLU 编码，再让 Transformer Encoder 在连续帧序列之间
执行注意力，最后逐帧解码。训练目标是用当前观测帧序列预测同一观测的延迟版本，
不使用干净信号或目标频率标签。

```python
from transformer_based_adaptive_line_enhancer import (
    AETDLEConfig,
    enhance_with_aet_dle,
    train_aet_dle,
)

config = AETDLEConfig(
    frame_length=320,
    frame_hop=64,
    prediction_delay=1_000,
    sequence_length=8,
    embedding_dim=64,
    attention_heads=8,
    transformer_layers=2,
    feedforward_dim=128,
    learning_rate=1e-3,
    epochs=30,
    seed=0,
)

training = train_aet_dle(sample.noisy_signal, config=config)
result = enhance_with_aet_dle(sample.noisy_signal, training)

enhanced = result.enhanced_signal
enhanced_valid = enhanced[result.valid_slice]
```

默认 `frame_length=320`，使 64 维嵌入恰好是输入维度的五分之一；Transformer 使用
论文所述的 8 个注意力头和 2 个编码层。论文没有规定可执行的序列长度、位置编码和
FFN 宽度，本实现分别选择 8、正弦位置编码和 128。解码输出使用 Sigmoid 与 `[0,1]`
最小-最大归一化匹配。

实现要求 `prediction_delay >= frame_length`，避免输入帧和延迟目标帧共享相同噪声
样本而形成直接复制捷径。实际延迟还应超过目标噪声的主要相关长度。

论文损失将网络当前输入的输出与延迟目标比较。本实现把预测结果放回延迟目标对应的
时间位置，以便与原波形对齐。这需要未来 `prediction_delay` 个样本，因此当前 API 是
离线增强器，`valid_slice` 不包含末尾无法预测的区域；不能据此宣称实时因果处理。

固定种子单谱线场景获得了明显 SNR 增益，但多谱线试验中最弱谱线仍可能被压制。
Transformer 的加入本身不证明弱谱线一定改善，后续比较必须报告逐谱线指标并采用
参数量和训练预算匹配的 AE-DLE 对照。

## 可视化谱线增强效果

```python
import matplotlib.pyplot as plt

from transformer_based_adaptive_line_enhancer import (
    adaptive_line_enhance,
    generate_noisy_signal,
    plot_line_enhancement_spectrogram,
)

sample = generate_noisy_signal(
    sample_rate_hz=1_000.0,
    snr_db=-10.0,
    seed=42,
    scaling="power-normalized",
)
result = adaptive_line_enhance(sample.noisy_signal)

figure, axes = plot_line_enhancement_spectrogram(
    sample,
    result.enhanced_signal,
    sample_rate_hz=1_000.0,
    start_sample=result.adaptation_start,
    n_fft=256,
    hop_length=64,
    max_frequency_hz=250.0,
    dynamic_range_db=80.0,
)

figure.savefig("ale-spectrogram.png", dpi=150)
plt.show()
```

三幅时频图分别显示干净参考、带噪输入和增强输出。它们共享同一个绝对 dB 色标，
因此不会因各自归一化而夸大增强效果。`start_sample=result.adaptation_start` 会排除
ALE 尚未取得完整延迟输入向量的启动区间。

## 测试

```bash
uv run pytest
```
