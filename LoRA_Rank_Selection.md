# 给定基础权重，LoRA 的 r 怎么定

2026-09-14 · 机器学习 · 参数高效微调


## 秩与奇异值谱

一个 d×k 的矩阵有 min(d,k) 个奇异值。秩是一个整数，等于非零奇异值的个数，只回答有几个非零方向的问题。

以 d=k=5120 的层为例，ΔW 有 5120 个奇异值。全参微调之后通常是满秩的：5120 个奇异值全部非零，只是绝大多数极小。所以在这个尺度上问“秩是多少”得不到任何信息，有信息的是衰减速度。

$$
\begin{aligned}
\Delta W &= W_{\mathrm{ft}}-W_0 = U\Sigma V^{\top}, \\
\Sigma &= \operatorname{diag}(\sigma_1,\dots,\sigma_{\min(d,k)}), \\
&\quad \sigma_1\ge\sigma_2\ge\cdots\ge\sigma_{\min(d,k)}\ge 0, \\
\operatorname{rank}(\Delta W) &= \bigl|\{\,i:\sigma_i>0\,\}\bigr|.
\end{aligned}
$$

这里 U 和 V 的列分别是左右奇异向量，σᵢ 是奇异值。秩只看 σᵢ 是否为零，完全不管它们有多小，这正是它在微调场景里失效的原因。

## LoRA 的秩上界

LoRA 把这层改动限制在一个低秩乘积上，见式 2。

$$
W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\quad A\in\mathbb{R}^{r\times k},\quad r\ll\min(d,k).
$$

初始化通常取 A ~ N(0, σ²)、B = 0，于是 ΔW = 0，训练从原模型出发，起点没有扰动。约束来自一行代数，见式 3。

$$
\operatorname{rank}(\Delta W)=\operatorname{rank}(BA)\le\min(\operatorname{rank}A,\operatorname{rank}B)\le r.
$$

这是硬上限，不是正则化：r 设小了，最优解落在可达集合之外，怎么训都够不到。但“够不到”并不等于“效果差”。只要被丢掉的那些方向对 loss 贡献极小，就无所谓。

## 截断 SVD 与能量占比

由 Eckart–Young 定理，截断 SVD 是最优的秩 r 近似，无论用 Frobenius 范数还是谱范数衡量，见式 4。

$$
\begin{aligned}
\|\Delta W-\Delta W_r\|_F &= \min_{\operatorname{rank}(M)\le r}\|\Delta W-M\|_F = \Bigl(\sum_{i>r}\sigma_i^2\Bigr)^{1/2}, \\
\|\Delta W-\Delta W_r\|_2 &= \sigma_{r+1}.
\end{aligned}
$$

于是前 r 项的能量占比，见式 5。

$$
E(r)=\frac{\sum_{i\le r}\sigma_i^2}{\sum_{i=1}^{\min(d,k)}\sigma_i^2}=1-\frac{\|\Delta W-\Delta W_r\|_F^2}{\|\Delta W\|_F^2}
$$

它恰好衡量“秩 r 能保留多少改动”。判据因此非常具体。

## 有全参 ckpt 时的直接测量

逐层算 ΔW = W_ft − W₀，做 SVD，看 E(48)。48 只是一个取样点：先在一个有代表性的 r 上看曲线，再决定往哪边走。

- E(48) > 0.90：r = 48 足够，甚至可以试 16；
- 0.70 ≤ E(48) ≤ 0.90：提到 128 至 256；
- E(48) < 0.70：放弃 LoRA，改用全参。

E(r) 应当逐层画，而不是只看全模型平均：同一句判据在不同层上会给出不同答案。

## 内在维度与早期谱代理

一种是内在维度法。把更新限制在一个随机 d 维子空间里训至收敛，画 loss–d 曲线找拐点，见式 6。

$$
\theta=\theta_0+P\theta',\qquad P\in\mathbb{R}^{D\times d},\quad d\ll D,\qquad d^{*}=\min\bigl\{\,d:\ R(d)\ge 0.9\,R_{\mathrm{full}}\,\bigr\},
$$

其中 R 是验证集指标，R_full 是全参微调的结果。对应到 LoRA，有一个简单得多的做法：固定随机初始化的 A、只训 B，等价于只在一个随机 r 维子空间里更新。成本中等。

另一种是早期谱代理，成本最低，推荐先跑。全参先跑 200 到 500 步，对累计更新做 SVD，见式 7。

$$
S_T=\sum_{t=1}^{T}\Delta W_t,\qquad \Delta W_t=W_t-W_{t-1},
$$

再对 S_T 套用式 5 计算 E(r)。以 SGD 为例 ΔW_t = −η_t g_t，用 Adam 一类优化器时换成对应的更新量即可。改动的主方向在早期即已成形，此时 E(r) 曲线基本稳定，这条路成本最低。

“主方向早期成形”是经验观察，不是定理。它值得先跑一次的原因只有一个：比起全参训完再回头测，200 到 500 步的代价几乎可以忽略。

## 模块选择、α 缩放与逐层 r

改哪些模块：同等参数量下，“全部线性层配小 r”通常优于“只用 q、v 配大 r”，优先覆盖 q、k、v、o 与 FFN。参数量相同不代表表达能力相同：前者可选的更新方向更多，只是每个方向更窄。

α 的缩放。真正生效的是 α/r，见式 8。

$$
s=\frac{\alpha}{r},\qquad \Delta W=s\,BA.
$$

常取 α = 2r，使有效尺度 s 恒为 2。改 r 必须同步改 α，否则等于偷偷改了学习率——把 r 从 r₁ 换到 r₂ 而 α 不变，更新整体乘上 r₁/r₂。若 r 很大，α/r 会偏小，可考虑 α/√r 一类的稳定化缩放。

逐层 r。E(r) 要按层分别算。通常越靠后的层衰减越慢、需要更大的 r，但逐层配置带来的收益往往不抵复杂度。

## 训后的参与比校验

对训好的 BA 做 SVD，算参与比，见式 9。

$$
\mathrm{PR}=\frac{\bigl(\sum_{i=1}^{r}\sigma_i\bigr)^2}{\sum_{i=1}^{r}\sigma_i^2}\in[1,r].
$$

奇异值分布均匀时它约等于 r，集中在单一方向时趋近 1。参与比远小于 r，说明 r 给多了，可以缩；奇异值尾部仍然很大，说明 r 卡住了该加。这一步能把冗余直接回收成推理开销：权重合并进 W 后推理没有额外成本，但在多套 adapter 并存、不合并权重时，开销与 r 成正比。

## 边界

E(r) > 0.9 只说明改动方向被低秩捕获。式 4 逼近的是 ΔW 的 Frobenius 误差，而训练优化的是任务 loss，两者只有在 loss 对 ΔW 各方向的敏感度相同时才一致。所以这是“与全参差距小”的必要条件代理，而非充分条件：截断 SVD 是最优近似，不代表训练找得到它，loss 相当也不代表解相同。最终仍要靠 held-out 验证。

此外，所需的秩取决于数据量。数据越多，最优 ΔW 越可能带上高秩分量：同一任务样本从两万增到两百万，所需 r 会变大。内在维度的工作也显示，任务离预训练分布越远，需要的维度越高。r 是一次测量，不是一次定终身。

核心一句：先测 ΔW 的秩需求，再定 r。

## 参考来源

- [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021](https://arxiv.org/abs/2106.09685)
- [Eckart & Young, *The Approximation of One Matrix by Another of Lower Rank*, Psychometrika, 1936](https://doi.org/10.1007/BF02288367)
- [Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020](https://arxiv.org/abs/2012.13255)
- [Li et al., *Measuring the Intrinsic Dimension of Objective Landscapes*, 2018](https://arxiv.org/abs/1804.08838)
- [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023](https://arxiv.org/abs/2305.14314)
- [Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*, 2023](https://arxiv.org/abs/2312.03732)
- [Biderman et al., *LoRA Learns Less and Forgets Less*, 2024](https://arxiv.org/abs/2405.09673)
- [Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*, 2024](https://arxiv.org/abs/2410.21228)
