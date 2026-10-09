---
title: "画面连贯不等于懂物理：SOTA 视频模型的物理一致性仍是短板"
date: 2026-10-09
tags:
  - 视频生成
  - 世界模型
  - 物理一致性
  - Benchmark
---

如果视频生成模型要作为 world model 用于预测、规划或具身交互，评价标准就不能只看视觉保真度和时序连续性，还要看生成序列能否在受控条件下重现可测的物理关系。

Einsia AI 团队最近发布了预印本 [World Models' Last Exam in Physics](https://arxiv.org/abs/2610.08791)。从这篇论文出发，我们能看见什么？这篇工作设计了 40 个受控任务，覆盖 9 类物理现象。8 个视频模型在每个任务上生成 4 个样本，共得到 1,280 段视频。

![Benchmark 范围、生成协议、测量流程和总体结果](assets/blog/world-models-physics/figure-1-benchmark.png)

评测把证据是否足够与物理关系是否成立拆开。设 $C(V)$ 为一致性与可观测性分数，$P(V)$ 为物理测量分数：

\[
S(V)=0.15C(V)+0.85\widetilde P(V),\qquad \widetilde P(V)=
\begin{cases}
P(V),&C(V)\geq80,\\
0,&C(V)<80.
\end{cases}
\]

最高综合分只有 57.76/100。含石头冰块融化任务平均分为 3.45，8 个模型的物理分数全部为零。自由落体任务中，所有可用视频都通过一致性门控，但跨模型平均分只有 26.61，说明时序连续性并不等于动力学正确性。

45° 抛体任务检验的不是“看起来像抛物线”，而是：

\[
r_{\mathrm{spatial}}=\left|\frac{H}{R}-\frac14\right|.
\]

![不同难度任务上的模型表现](assets/blog/world-models-physics/figure-2-difficulty.png)

数据和训练目标可能是原因之一。被动互联网视频缺少对质量、初速度、外力、边界条件和材料属性的系统干预，也缺少成对的反事实轨迹。更可扩展的方向可能包括干预式数据、测量驱动奖励、物理约束后训练和跨任务迁移的世界模型表示。

单次横截面 benchmark 还不足以证明存在稳定的物理 scaling law。要判断扩大模型、数据和算力是否有效，需要控制架构与训练预算，并报告模型规模、token、计算量和物理得分之间的关系。

![光反射与带电小球平衡的定性比较](assets/blog/world-models-physics/figure-3-qualitative.png)

## 参考资料

- [World Models' Last Exam in Physics](https://arxiv.org/abs/2610.08791)
- [VideoPhy](https://arxiv.org/abs/2406.03520)
- [PhyWorldBench](https://research.nvidia.com/labs/cosmos-lab/phyworldbench/)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
