# LoRA 的 r 怎么定

2026-09-14 · 机器学习 · 参数高效微调

结论是没有准确公式可算，只能大致扫/估计

## 秩与奇异值谱

一个 d×k 的矩阵有 min(d,k) 个奇异值。秩等于非零奇异值的个数，回答有几个非零方向。

以 d=k=5120 的层为例，ΔW 有 5120 个奇异值。全参微调之后通常是满秩的：5120 个奇异值全部非零，只是大多数极小。Biderman et al. 2024 在代码、数学任务上测到的 ΔW 谱就是这种样子，衰减很慢。所以在这个尺度上问秩是多少得不到任何信息，有信息的是衰减速度。

$$
\begin{aligned}
&\Delta W = W_{\mathrm{ft}}-W_0 = U\Sigma V^{\top}, \\
&\Sigma = \operatorname{diag}(\sigma_1,\dots,\sigma_{\min(d,k)}), \\
&\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_{\min(d,k)}\ge 0, \\
&\operatorname{rank}(\Delta W) = \bigl|\{\,i:\sigma_i>0\,\}\bigr|.
\end{aligned}
$$

这里 U 和 V 的列分别是左右奇异向量，σᵢ 是奇异值。秩只看 σᵢ 是否为零，这正是它在微调场景里失效的原因。

## LoRA

LoRA 把这层改动限制在低秩乘积上，见式 2。

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

它衡量秩 r 能保留多少改动。

## 有全参 ckpt 时的直接测量

逐层算 ΔW = W_ft − W₀，做 SVD，得到奇异值谱 σ₁ ≥ … ≥ σ_min(d,k) ≥ 0，画 E(r) 整条曲线，不要只看一个 r 点。

这个谱用于判断要不要LoRA。根据 Biderman et al. 2024 在代码、数学任务上的测量，4096×4096 的 W_q 要 1500 以上的秩才解释 ΔW 90% 的方差，比常用 LoRA 秩大一两个数量级；同一批实验里，他们给出的实用建议是按显存约束挑 r，16 配全部模块是个好起点。Shuttleworth et al. 2024 测到的全参更新有效秩也在 400 以上。所以谱衰减慢不是「r 该取大」的信号，而是这个任务上 LoRA 追不上全参。

E(r) 数的是能量，谱上大的方向对任务可以并不重要，真实的 loss 增量由曲率决定，二阶近似下是 ½ Σ_{i>r} h_i σ_i²，h_i 是 loss 在方向 i 上的曲率。所以拿 E(r) 反推 r 得不到🉑用的数；这个谱的用处是判断该不该 LoRA。

E(r) 高只说明全参解可以被低秩逼近：根据 Shuttleworth et al. 2024，低秩下 LoRA 学出的更新与全参谱不同，会出现与预训练谱近似正交的新方向，有效秩不到全参的一半。所以最后仍要在 held-out 上确认。

## 甜点在哪

r 往上加表现并非一直变好。欠秩时丢掉被截掉的那些方向，截断误差降不下来；过了某个点，它已经接近零，而每多一维秩要多付一份方差，量级是 r·d/n。

Arunan 2026 把这个写成了界：目标更新有一个固有的秩，r 小于它时误差卡在截断项上，r 大于它时按 r·d/n 线性涨——曲线是 U 形，过了甜点不只是不涨，是会掉。他们跑了 168 组 DistilBERT、RoBERTa 的微调，验证 loss 都是这个形状，其中两个 SST-2 配置在大秩上掉得统计显著。这条只对固定秩的经验风险最小化解成立；换成核范数一类会自动收敛到那个秩的估计器，过秩就无害了。所以实测看到的平台，有一部分是训练过程在替你兜底。

准确位置算不出来。U 形的底很大程度由任务的谱定，不由模型结构定。实测也是这么回事：GPT-3 175B 上 WikiSQL 的准确率从 r=1 到 r=64 基本不动（Hu et al. 2021）；同一个 Llama-2-7B，math 上指令微调到 r=256 才追平全参，持续预训练连 r=256 都追不上（Biderman et al. 2024）。结构一样时任务了，甜点差两个数量级。


## 内在维度与早期谱代理

内在维度法。把更新限制在一个随机 d 维子空间里训至收敛，画 loss–d 曲线找拐点，见式 6。

$$
\theta=\theta_0+P\theta',\qquad P\in\mathbb{R}^{D\times d},\quad d\ll D,\qquad d^{*}=\min\bigl\{\,d:\ R(d)\ge 0.9\,R_{\mathrm{full}}\,\bigr\},
$$

其中 R 是验证集指标，R_full 是全参微调的结果。对应到 LoRA，有一个简单得多的做法：固定随机初始化的 A、只训 B，取最小的 r 使 held-out 达到 0.9 R_full（判据同式 6）。ΔW = BA 的行空间被限制在 A 的 r 维行空间内，可训练参数量 d·r。根据 Zhu et al. 2024，只训 B 比只训 A 有效，随机不训练的 A 和训过的 A 差不多；FLoRA（Hao et al. 2024）把这个更新近似成对梯度做一次随机投影。成本中等。

早期谱代理。全参先跑 200 到 500 步，对累计更新做 SVD，见式 7。

$$
S_T=\sum_{t=1}^{T}\Delta W_t,\qquad \Delta W_t=W_t-W_{t-1},
$$

再对 S_T 套用式 5 计算 E(r)。以 SGD 为例 ΔW_t = −η_t g_t，用 Adam 一类优化器时换成对应的更新量即可。这条路的先例都是拿谱做初始化，不是拿它定 r——SLoRA（Babakniya et al. 2023）用一段全参后的 ΔW 初始化 LoRA；EVA（Paischer et al. 2024）用激活的增量 SVD 初始化 A，并按解释方差把给定的秩预算分到各层（它的谱来自初始化时的激活，不是训了几百步的 ΔW）。

跑这几百步比起跑完，代价🉑以忽略；但拿到的谱只能当粗估。

## 模块选择、α 缩放与逐层 r

模块选择。QLoRA（Dettmers et al. 2023）消融出来的结论是把 adapter 加到全部线性层（含 FFN），只配 q、v 会明显掉质量——覆盖面比秩更关键。所以同等参数量下，“全部线性层配小 r”优于“只用 q、v 配大 r”：参数量相同不代表表达能力相同，前者可选的更新方向更多，只是每个方向更窄。

α 的缩放。真正生效的是 α/r，见式 8。

$$
s=\frac{\alpha}{r},\qquad \Delta W=s\,BA.
$$

Hu et al. 2021 用的是 α/r 这个缩放，常取 α = 2r，使有效尺度 s 恒为 2。改 r 必须同步改 α，否则等于偷偷改了学习率——把 r 从 r₁ 换到 r₂ 而 α 不变，更新整体乘上 r₁/r₂。r 很大时 α/r 会偏小：Kalajdzievski 2023 把缩放换成 α/√r；Shuttleworth et al. 2024 测到固定 α 配高秩会引入更多与预训练谱正交的新方向，换成 α/√r 后高秩 LoRA 才更像全参。

逐层 r。E(r) 要按层分别算。根据 Biderman et al. 2024，首尾层的秩偏低、中间层偏高，MLP 比 attention 高；不过手调的逐层配置收益往往不抵复杂度——按解释方差在预算内自动分配（EVA）是同预算下权衡更好的一条路。

## 训后参与比校验

对训好的 BA 做 SVD，算参与比，见式 9。

$$
\mathrm{PR}=\frac{\bigl(\sum_{i=1}^{r}\sigma_i\bigr)^2}{\sum_{i=1}^{r}\sigma_i^2}\in[1,r].
$$

奇异值分布均匀时它约等于 r，集中在单一方向时趋近 1。训完还能砍秩这件事有先例：AdaLoRA（Zhang et al. 2023）在训练中按重要度给各层分配并裁剪秩，低预算下优于固定秩。所以参与比远小于 r 就可以缩，奇异值尾部仍然很大就该加。这一步能把冗余直接换成推理上的节省：权重合并进 W 后推理没有额外成本，但在多套 adapter 并存、不合并权重时，开销与 r 成正比。

## 结论

到头来很大程度还是看硬件资源

先算 r 的成本上界。适配器的参数量是 r·Σ_l(d_l + k_l)，反过来就是 r ≤ 预算 / Σ_l(d_l + k_l)。Llama-3-8B 的规模（32 层、隐藏 4096、FFN 14336、7 个线性层）Σ(d+k) 约 2.8M：预算给到全参的 0.5%（约 45M 参数），除下来 r ≈ 16。Biderman et al. 2024 说的“按显存约束挑 r”就是这个意思——在这个范围里，r 是成本调节，16 配全部模块就是他们试过的配置里给的推荐。

r 的下界没有式子，只能按任务给量级：数据越多、离预训练分布越远，需要的秩越高。所以可以默认全模块 r = 16，α 同步取 2r。要更准，就在这个范围里把 r 定在 held-out 达到全参九成的那一点（式 6）：有全参 ckpt 就冻结随机 A、只训 B；没有就把更新限制在随机 d 维子空间里扫。

训完用参与比收尾，远小于 r 就缩，奇异值尾部还大就加。手调的逐层配置通常不抵复杂度。所有数字最后都要在 held-out 上过一遍。


## 参考

- [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021](https://arxiv.org/abs/2106.09685)
- [Eckart & Young, *The Approximation of One Matrix by Another of Lower Rank*, Psychometrika, 1936](https://doi.org/10.1007/BF02288367)
- [Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020](https://arxiv.org/abs/2012.13255)
- [Li et al., *Measuring the Intrinsic Dimension of Objective Landscapes*, 2018](https://arxiv.org/abs/1804.08838)
- [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023](https://arxiv.org/abs/2305.14314)
- [Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*, 2023](https://arxiv.org/abs/2312.03732)
- [Zhang et al., *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, ICLR 2023](https://arxiv.org/abs/2303.10512)
- [Paischer et al., *Parameter Efficient Fine-tuning via Explained Variance Adaptation*, 2024](https://arxiv.org/abs/2410.07170)
- [Biderman et al., *LoRA Learns Less and Forgets Less*, 2024](https://arxiv.org/abs/2405.09673)
- [Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*, 2024](https://arxiv.org/abs/2410.21228)
- [Zhu et al., *Asymmetry in Low-Rank Adapters of Foundation Models*, 2024](https://arxiv.org/abs/2402.16842)
- [Hao et al., *Low-Rank Adapters Are Secretly Gradient Compressors*, 2024](https://arxiv.org/abs/2402.03293)
- [Babakniya et al., *SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models*, 2023](https://arxiv.org/abs/2308.06522)
- [Arunan, *Tight Sample Complexity for Low-Rank Adaptation: Matching Bounds and Rank Selection*, 2026](https://arxiv.org/abs/2607.27680)
