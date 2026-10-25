# 为什么大厂不用 VLM 洗数据

2026-10-26 · 机器学习 · 视频生成 · 数据

我们洗数据全程 VLM，账单也好看——直到精读六份前沿技术报告才发现：过滤段没有一家用 VLM。这不是"贵"一个字能解释完的。

## 我们的账单

| 工序 | 模型 / 输入 | 吞吐（实测） | 14.8 万源池耗时 |
| --- | --- | --- | --- |
| 宽定义初筛 | Qwen2.5-VL-7B，宽帧 | ~35-40 条/分/卡 | 2 卡 × 3 天左右（历史账） |
| 收紧初筛（进档判定） | Qwen3-VL-30B，18帧/条 | ~120 条/分（4卡） | 10.4 万 KEEP 约 14.5h |
| 强 prompt 重筛（4fps 逐段） | Qwen3-VL-30B，12-16段蒙太奇 | ~55 条/分（4卡） | 19,816 候选 ≈ 6h |
| 8fps 富矿挖掘 | Qwen3-VL-30B，5视频×3chunk/请求 | ~113 条/分（2卡） | 28,771 候选 ≈ 4.2h |
| 事件窗定位（v3 补扫） | Qwen3-VL-30B，8fps | ~45 条/分（4卡） | 5,950 条 ≈ 2.2h |

A800 实测。合计约 243 A800·小时，14.8 万条源，平均约 5.9 卡秒/条。

## 六份报告的过滤段

| 漏斗层 | 用什么 | 出处 |
| --- | --- | --- |
| 镜头切分 | PySceneDetect / TransNetV2 / FFmpeg 场景分数 | 全部 |
| 技术质量 | Laplacian 方差、VMAF、专训质量 operator | Open-Sora 2.0 / HunyuanVideo 1.5 |
| 水印字幕 | PaddleOCR + 专用检测器，安全区裁剪 | MovieGen / Seedance / Tiger200K |
| 美学 | CLIP+MLP 小回归器 | MovieGen / Open-Sora 2.0 / HunyuanVideo 1.5 |
| 运动 | 静态检测器、VMAF motion、光流 | Seedance / MovieGen / Open-Sora 2.0 |
| 去重配比 | embedding 聚类 + 反比采样 | Seedance / MovieGen |
| Caption | 专训/微调 VLM（VLM 在此进场） | 全部 |
| 后训练 | 人工标注 + VLM 奖励模型 | HunyuanVideo 1.5 / Seedance |

## 三层答案

1. **贵两个数量级以上。** 8 亿 clip × 5.9 卡秒 ≈ 131 万 A800·小时 ≈ 150 卡年，1000 卡 55 天筛一遍。
2. **就算免费也不选。** 过滤器是量具，第一美德是稳定不是聪明；美学回归器与 VLM 本是人类判断的同源蒸馏，选的是容器不是知识。
3. **定义权。** 管线是按成本排序的代理塔，"好数据"的定义顺序被经济学塑造。

## 预测

7B→30B 吞吐基本持平；成本再降两个数量级后过滤段将向 VLM judge 迁移，美学回归器会像像素阈值一样原始。

## 参考资料

- Seedance 1.0: https://arxiv.org/abs/2506.09113
- MovieGen: https://arxiv.org/abs/2410.13720
- Open-Sora 2.0: https://arxiv.org/abs/2503.09642
- HunyuanVideo 1.5: https://arxiv.org/abs/2511.18870
- Koala-36M: https://arxiv.org/abs/2410.08260
- Tiger200K: https://arxiv.org/abs/2504.15182
- AVVA: https://arxiv.org/abs/2503.09205
