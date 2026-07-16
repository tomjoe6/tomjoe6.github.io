# 为什么 Drifting 更接近生成建模的第一性原理

2026-07-16 · 机器学习 · 生成模型

*Generative Modeling via Drifting* 改变的是分布运输发生的时机。扩散模型通常在推理时把噪声逐步带向数据；这篇工作把同类修正交给训练中的参数更新，最后保留一个直接的生成器。

令 ε ~ pε，x = fθ(ε)。网络输出的分布是严格的 pushforward，见式 1。

生成模型要达到的条件是生成分布与数据分布相同。预测噪声、判别器博弈和 ELBO 都是逼近这个条件的具体方案。

扩散与 flow 模型通常先指定一条分布路径，并在推理时解动力系统，见式 2。

式 2 描述概率质量守恒。该方案的代价是运输在推理时执行，因此需要多次评估网络。Drifting 把这段迭代放进训练：SGD 改变生成器时，也改变了它诱导的分布。

## 训练中完成运输

论文为目标分布 p 与当前生成分布 q 定义漂移场 Vp,q(x)，规定当前样本应被怎样修正，见式 3。

场需要满足的平衡条件见式 4。

分布匹配后，样本不再移动。论文用反对称性保证这一点：交换数据与生成样本，修正方向随之反转。

它的核化实现来自 mean shift。数据样本提供吸引，生成样本提供排斥，具体定义见式 5。

这里比较的是两个局部重心：数据云把样本拉向自己，生成云把样本推出过密区域。它不能从理论上消除所有 mode collapse，但场中确实包含了反向的密度反馈。

训练目标是让网络输出追上漂移后的、冻结的自己，见式 6。

sg 是 stop-gradient。反向传播时，右侧是固定目标；优化器只把当前输出推向该目标。每一次 SGD 更新近似执行一次由场给出的修正，长期累积后，生成器本身承担了运输。推理只需执行一次生成器，不再积分采样轨迹。

> 数值上，残差对应漂移；但这里不是对漂移大小直接做全梯度下降。stop-gradient 固定了“估计场，再沿场更新”的顺序；后续理论工作正是在分析这个顺序对应什么样的动力学。

## 这改变了什么

**GAN** 也直接匹配分布，但“往哪走”由同时训练的判别器给出，因而是一个 min-max 博弈。Drifting 不训练 critic，而用显式的样本间场；代价是距离度量被放进核、带宽和特征空间。

**VAE** 以概率图模型和 ELBO 为中心，面对的是重建、先验、近似后验之间的变分折中。Drifting 不需要显式似然或后验编码器，而是直接修正生成分布与数据分布的几何不匹配。

**Diffusion、score matching 与 flow matching** 的强项是把高维分布匹配改为稳定的局部回归；但通常仍把运输路径保留在推理时。Drifting 没有预先规定每个样本在推理时必须经过的噪声路径，而是在当前分布不匹配时估计局部修正，并让训练更新积累这些修正。

## 边界

“更第一性”在这里指建模组织方式，不是数学上的绝对优越。扩散的连续性方程同样基本，flow matching 同样在做分布运输。Drifting 的直接性在于，它以生成分布等于数据分布为固定点，并把训练迭代作为分布变化的时间尺度。

核、特征空间和 mini-batch 估计会决定场的几何。原论文也说明：对任意场，零漂移并不自动意味着两个分布相同；它只为特定核化构造给出充分条件。后续工作把 Gaussian-kernel 情形联系到平滑分布的 score difference 与 Wasserstein gradient flow，也讨论了实际算法与理想流之间的差异。

这篇工作的价值在于给出另一种解释：一步生成可以来自训练期完成的分布运输，而不一定来自对多步采样器的蒸馏。视频生成、世界模型和控制会更在意这一点，因为推理时延会直接限制系统的反应速度。

## 参考来源

- [Deng et al., *Generative Modeling via Drifting*, 2026](https://arxiv.org/abs/2602.04770)
- [Generative Modeling via Drifting: project page and code](https://lambertae.github.io/projects/drifting/)
- [Lipman et al., *Flow Matching for Generative Modeling*, 2022](https://arxiv.org/abs/2210.02747)
- [Song et al., *Score-Based Generative Modeling through SDEs*, 2020](https://arxiv.org/abs/2006.11239)
- [Gretton et al., *On the Wasserstein Gradient Flow Interpretation of Drifting Models*, 2026](https://arxiv.org/abs/2605.05118)
- [Turan et al., *Generative Drifting is Secretly Score Matching*, 2026](https://arxiv.org/abs/2603.09936)
