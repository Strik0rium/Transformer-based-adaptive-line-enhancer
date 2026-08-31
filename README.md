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

## 测试

```bash
uv run pytest
```
