# 给定基础权重，LoRA 的 r 怎么定

2026-09-14 · 机器学习 · 参数高效微调

LoRA 的 r 不用一个个试。需要多大的秩是可以测的：ΔW 的谱说明这份权重上 LoRA 够不够用，内在维度给出量级，最后用两三个点跑 held-out 确认。


## 秩与奇异值谱

一个 d×k 的矩阵有 min(d,k) 个奇异值。秩等于非零奇异值的个数，只回答有几个非零方向的问题。

以 d=k=5120 的层为例，ΔW 有 5120 个奇异值。全参微调之后通常是满秩的：5120 个奇异值全部非零，只是大多数极小。所以在这个尺度上问秩是多少得不到任何信息，有信息的是衰减速度。

$$
\begin{aligned}
\Delta W &= W_{\mathrm{ft}}-W_0 = U\Sigma V^{\top}, \\
\Sigma &= \operatorname{diag}(\sigma_1,\dots,\sigma_{\min(d,k)}), \\
&\quad \sigma_1\ge\sigma_2\ge\cdots\ge\sigma_{\min(d,k)}\ge 0, \\
\operatorname{rank}(\Delta W) &= \bigl|\{\,i:\sigma_i>0\,\}\bigr|.
\end{aligned}
$$

这里 U 和 V 的列分别是左右奇异向量，σᵢ 是奇异值。秩只看 σᵢ 是否为零，这正是它在微调场景里失效的原因。

## LoRA 的秩上界

LoRA 把这层改动限制在一个低秩乘积上，见式 2。

$$
W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\quad A\in\mathbb{R}^{r\times k},\quad r\ll\min(d,k).
$$

初始化通常取 A ~ N(0, σ²)、B = 0，于是 ΔW = 0，训练从原模型出发，起点没有扰动。约束来自一行代数，见式 3。

$$
\operatorname{rank}(\Delta W)=\operatorname{rank}(BA)\le\min(\operatorname{rank}A,\operatorname{rank}B)\le r.
$$

r 设小了，最优解落在可达集合之外，怎么训都训不到。但训不到并不等于差，只要被丢掉的那些方向对 loss 贡献极小就无所谓。

## 截断 SVD 与能量占比

由 Eckart–Young 定理，截断 SVD 是最优的秩不超过 r 的近似，无论用 Frobenius 范数还是谱范数衡量，见式 4。

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

它衡量“秩 r 能保留多少改动”，判据因此具体。

## 有全参 ckpt 时的直接测量

逐层算 ΔW = W_ft − W₀，做 SVD，得到奇异值谱 σ₁ ≥ … ≥ σ_min(d,k) ≥ 0，画 E(r) 的整条曲线，不要只看一个 r 点。

读谱得到的答案通常很大。根据 Biderman et al. 2024 在代码、数学任务上的测量，4096×4096 的 W_q 要 1500 以上的秩才解释 ΔW 90% 的方差，比常用 LoRA 秩大一两个数量级；数据越多秩越高，MLP 比 attention 高，首尾层比中间层低。Shuttleworth et al. 2024 测到的全参更新有效秩也在 400 以上。所以谱上一般没有能取 r 的拐点：ΔW 本身不小秩，LoRA 也不是靠复现它起作用的——Biderman et al. 2024 报告，代码、数学这类任务上 LoRA 追不上全参，提高 r 也补不平。

如果非要用“低秩逼近 ΔW”这个标准，就按可接受的相对误差反推 r：Eckart–Young 给出 ‖ΔW − ΔW_r‖_F / ‖ΔW‖_F = √(1 − E(r))（式 4、式 5），要 5% 以内就取满足它的最小 r。这么算出来的 r 通常大得没法用，这本身就是结论。

E(r) 高只说明全参解可以被低秩逼近，不说明 LoRA 找得到它：根据 Shuttleworth et al. 2024，低秩下 LoRA 学出的更新与全参谱不同，会出现与预训练谱近似正交的新方向，有效秩不到全参的一半。E(r) 也只数能量、不数重要性——谱上大的方向对任务可以并不重要，真实的 loss 增量由曲率决定，二阶近似下是 ½ Σ_{i>r} h_i σ_i²。所以最后仍要扫几个 r 跑 held-out，逐层分别定，不要只看全模型平均。

## 内在维度与早期谱代理
无全参 ckpt 时

一种是内在维度法。把更新限制在一个随机 d 维子空间里训至收敛，画 loss–d 曲线找拐点，见式 6。

$$
\theta=\theta_0+P\theta',\qquad P\in\mathbb{R}^{D\times d},\quad d\ll D,\qquad d^{*}=\min\bigl\{\,d:\ R(d)\ge 0.9\,R_{\mathrm{full}}\,\bigr\},
$$

其中 R 是验证集指标，R_full 是全参微调的结果。对应到 LoRA，有一个简单得多的做法：固定随机初始化的 A、只训 B。ΔW = BA 的行空间被限制在 A 的 r 维行空间内，可训练参数量 d·r。根据 Zhu et al. 2024，只训 B 比只训 A 有效，随机不训练的 A 和训过的 A 差不多；FLoRA（Hao et al. 2024）把这个更新近似成对梯度做一次随机投影。成本中等。

另一种是早期谱代理，成本低。全参先跑 200 到 500 步，对累计更新做 SVD，见式 7。

$$
S_T=\sum_{t=1}^{T}\Delta W_t,\qquad \Delta W_t=W_t-W_{t-1},
$$

再对 S_T 套用式 5 计算 E(r)。以 SGD 为例 ΔW_t = −η_t g_t，用 Adam 一类优化器时换成对应的更新量即可。这条路的先例都是拿这个谱去做初始化，不是用它定 r——SLoRA（Babakniya et al. 2023）用一段全参后的 ΔW 初始化 LoRA，EVA（Paischer et al. 2024）用激活的增量 SVD。「主方向早期就成形、之后不再变」这个前提没有直接证据，反证倒有：GaLore（Zhao et al. 2024）默认每 200 步重取一次子空间。

跑这几百步比起跑完，代价🉑以忽略；但拿到的谱只能当粗估。

## 模块选择、α 缩放与逐层 r

同等参数量下，“全部线性层配小 r”通常优于“只用 q、v 配大 r”，优先覆盖 q、k、v、o 与 FFN。参数量相同不代表表达能力相同：前者可选的更新方向更多，只是每个方向更窄。

α 的缩放。真正生效的是 α/r，见式 8。

$$
s=\frac{\alpha}{r},\qquad \Delta W=s\,BA.
$$

常取 α = 2r，使有效尺度 s 恒为 2。改 r 必须同步改 α，否则等于偷偷改了学习率——把 r 从 r₁ 换到 r₂ 而 α 不变，更新整体乘上 r₁/r₂。若 r 很大，α/r 会偏小，可考虑 α/√r 一类的稳定化缩放。

逐层 r。E(r) 要按层分别算。根据 Biderman et al. 2024，首尾层的秩偏低、中间层偏高，MLP 比 attention 高；但逐层配置带来的收益往往不抵复杂度。

## 训后的参与比校验

对训好的 BA 做 SVD，算参与比，见式 9。

$$
\mathrm{PR}=\frac{\bigl(\sum_{i=1}^{r}\sigma_i\bigr)^2}{\sum_{i=1}^{r}\sigma_i^2}\in[1,r].
$$

奇异值分布均匀时它约等于 r，集中在单一方向时趋近 1。参与比远小于 r，说明 r 给多了，可以缩；奇异值尾部仍然很大，说明 r 卡住了该加。这一步能把冗余直接换成推理上的节省：权重合并进 W 后推理没有额外成本，但在多套 adapter 并存、不合并权重时，开销与 r 成正比。

## 边界

E(r) ≥ 0.9975（即 5% 相对误差，由式 4、式 5 反推）说明改动方向被低秩捕获。式 4 逼近的是 ΔW 的 Frobenius 误差，训练优化的是 loss，两者只有在 loss 对 ΔW 各方向的敏感度相同时才一致。所以这是与全参差距小的必要条件代理，而非充分条件：截断 SVD 是最优近似，不代表训练找得到它，loss 相当也不代表解相同。最终仍要靠 held-out 验证。

此外，所需的秩取决于数据量。数据越多，最优 ΔW 越可能带上高秩分量，同一任务样本从两万增到两百万，所需 r 会变大；任务离预训练分布越远，需要的维度越高。


## 参考

- [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021](https://arxiv.org/abs/2106.09685)
- [Eckart & Young, *The Approximation of One Matrix by Another of Lower Rank*, Psychometrika, 1936](https://doi.org/10.1007/BF02288367)
- [Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020](https://arxiv.org/abs/2012.13255)
- [Li et al., *Measuring the Intrinsic Dimension of Objective Landscapes*, 2018](https://arxiv.org/abs/1804.08838)
- [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023](https://arxiv.org/abs/2305.14314)
- [Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*, 2023](https://arxiv.org/abs/2312.03732)
- [Paischer et al., *Parameter Efficient Fine-tuning via Explained Variance Adaptation*, 2024](https://arxiv.org/abs/2410.07170)
- [Biderman et al., *LoRA Learns Less and Forgets Less*, 2024](https://arxiv.org/abs/2405.09673)
- [Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*, 2024](https://arxiv.org/abs/2410.21228)
- [Zhu et al., *Asymmetry in Low-Rank Adapters of Foundation Models*, 2024](https://arxiv.org/abs/2402.16842)
- [Hao et al., *Low-Rank Adapters Are Secretly Gradient Compressors*, 2024](https://arxiv.org/abs/2402.03293)
- [Babakniya et al., *SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models*, 2023](https://arxiv.org/abs/2308.06522)
- [Zhao et al., *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*, 2024](https://arxiv.org/abs/2403.03507)
