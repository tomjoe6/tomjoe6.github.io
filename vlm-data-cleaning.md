# 技术报告漫读：工业界怎么洗数据

2026-10-26 · 机器学习 · 视频生成 · 数据

在学界论文中，洗数据除了规格筛选外常常全程 VLM：对于 1B~14B 的小模型与十秒内的 480P 视频，只需要十几张卡较短时间内就能跑出一个差强人意的数据集。

一道洗数据 pipe，去重、初筛、重筛、富矿挖掘、事件窗定位，到 caption 重写与泄漏校验，多道工序除了第一道 ffprobe，其余全部由 VLM 严格出判决，做 SFT ablation 的效果仍然很不好。数据清洗本质上是个处理数据算力、数据丰富程度/质量等因素的 Tradeoff，工业界是如何洗出一份好的数据集的？

翻阅技术报告：MovieGen、Seedance 1.0、HunyuanVideo 1.5、Open-Sora 2.0，加上专门讲数据集的 Koala-36M 和 Tiger200K，还有最新的闭源 Seedance 2.0，精读一遍 Data 部分。首先管线里的确有 VLM——caption 和奖励模型。而过滤段从头到尾几乎没有用 VLM。

## 一份账单

在展开之前，先看看我们小实验的消耗，后面所有的外推都要以其为参考：

| # | 工序 | 模型/口径 | 进 → 出 | 留存率 | 耗时（4×A800 基准） |
| --- | --- | --- | --- | --- | --- |
| 0 | 去重/探针过滤 | ffprobe | 152,300 → 100000 | 68% | ~0（CPU） |
| 1 | 宽定义初筛 | Qwen2.5-VL-7B 宽帧 | 100000 → 20000 | 19% | 2卡×~3天 ≈ 145 GPU·h |
| 2 | 收紧初筛 | Qwen3-VL 18帧 | 20000 → ~12000 | 61% | ~14.5h ≈ 58 GPU·h |
| 3 | 4fps 强 prompt 重筛 | Qwen3-VL 12-16段 | 12000 → 3000 | 23% | ~6h ≈ 24 GPU·h |
| 4 | 富矿再挖掘 | Qwen3-VL 15帧 | 30,000 候选 → 26,000 pass → 12,000 入档 | 93% pass | 9 GPU·h |
| 5 | 事件窗定位 | Qwen3-VL 8fps | 5,950 个缺窗 | 80.8% | ~9 GPU·h |
| 6 | caption 重写 | Qwen3-VL | 14,826 | — | 12 GPU·h |
| 7 | 泄漏校验+修复 | Qwen3-VL | 519 嫌疑 → 473 判泄漏 → 447 重写 | 91% | 2 GPU·h |

其中最便宜的一道工序毫无疑问是 ffprobe：在 CPU 上把 15.2 万条砍到 10.4 万，成本约等于零。业界 pipe 最开始也是进行规格粗糙筛选。

## 技术报告

把报告的数据管线按漏斗层归并，大致如下：

| 漏斗层 | 用什么 | 出处 |
| --- | --- | --- |
| 镜头切分 | PySceneDetect / TransNetV2 / FFmpeg 场景分数 | all |
| 技术质量 | Laplacian 方差、VMAF、专训质量 operator | Open-Sora 2.0 / HunyuanVideo 1.5 |
| 水印与字幕 | PaddleOCR + 专用检测器，安全区裁剪 | MovieGen / Seedance / Tiger200K |
| 美学 | CLIP+MLP 小回归器 | MovieGen / Open-Sora 2.0 / HunyuanVideo 1.5 |
| 运动 | 静态检测器、VMAF motion、光流 | Seedance / MovieGen / Open-Sora 2.0 |
| 去重与配比 | embedding 聚类 + 反比采样 | Seedance / MovieGen |
| Caption | 专训 / 微调 VLM | all |
| 后训练 | 人工标注 + VLM 奖励模型 | HunyuanVideo 1.5 / Seedance |

VLM 只出现在 caption、后训练的小规模判断，以及 RL 阶段的奖励模型。captioner 力求贴近人类写描述的方式，奖励模型放大人类的偏好判断。而在过滤段，也即镜头切分、质量过滤、运动过滤、去重这一段，清一色是 PySceneDetect、PaddleOCR、Laplacian 方差、VMAF、光流、CLIP+MLP 这些工具和小模型。

关于开源状态，Seedance 2.0 无权重，MiniMax H3 给权重不给报告，HunyuanVideo 报告权重都给了——但把数据处理管线这一真正的核心科技开源出来的一家大厂都没有。开源世界能找到的完整管线全部来自开源阵营：Open-Sora 把报告里的管线原样放进 repo，NeMo Curator 和 Data-Juicer 提供现成的过滤去重框架，再配上开源的 captioner、美学回归器和 Koala-36M、Tiger200K 这样的数据集，社区是拿开源工具复现工业 pipe。

第二层是一批便宜的量具。水印和字幕交给 PaddleOCR 和专用检测器，裁出安全区，裁完保留面积不足就整条丢弃；清晰度用 Laplacian 方差和 VMAF；美学是 CLIP+MLP 小回归器；运动用光流和 VMAF motion，太静和太抖都删。全是冻结的小模型，毫秒级一条。2026 年的 Seedance 2.0 报告里，这一层还是同样的配置。

第三层是去重与配比，操作对象从单条视频换成了分布。clip 用视频表征模型编码，聚类之后每个近重复簇只留质量分最高的那条；再按主体、场景、动作、风格数分布，头部下采样、尾部定向补采（Seedance），MovieGen 则用聚类反比采样加一套 600 个动词的词表做概念平衡。

第四层 caption，VLM 在这里进场，但没有一家裸用通用 VLM：Seedance 在 Tarsier2 上专训了双语 captioner，MovieGen 微调 LLaMA3-Video，HunyuanVideo 用 OPA-DPO 后训练来平衡描述密度与幻觉，输出的都是 80 到 200 词的密集结构化描述；Seedance 甚至单独训了一个 prompt 改写器，保证用户输入和 caption 同分布。

第五层是数据金字塔。同一批语料按质量分出梯度：Open-Sora 2.0 是 7000 万的低分辨率池喂出 500 万的高分辨率精选，HunyuanVideo 从 1000 万小时原始视频筛出 8 亿 clip，逐级收紧，贵的高分辨率训练只吃塔尖那一层。

后训练和 RLHF 不再是过滤大池子，而是人工构造小数据集。prompt 池人工验证，候选由不同 checkpoint 对同一 prompt 滚出，让人来做成对标注（HunyuanVideo 标了约 1 万对 DPO），最后把这 1 万对人类偏好蒸馏成 VLM 奖励模型供在线 RL 批量打分。

既然 VLM 可以靠近人类感官，为什么不在全流程尽量多用？

## 拼漏斗

每一层都有开源下位替代物。切分用 PySceneDetect 一行，转场敏感的场景换学习式的 TransNetV2；清晰度是 Laplacian 方差，运动是相邻帧光流的平均幅值，如：

```python
from scenedetect import detect, ContentDetector
import cv2, numpy as np

scenes = detect("raw.mp4", ContentDetector(threshold=27))

def motion_sharpness(path):
    cap, prev, mags, laps = cv2.VideoCapture(path), None, [], []
    while True:
        ok, frame = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        laps.append(cv2.Laplacian(g, cv2.CV_64F).var())
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None,
                    0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(np.linalg.norm(flow, axis=-1).mean())
        prev = g
    return np.median(laps), np.mean(mags)   # 清晰度、运动幅度
```

按六要素写 prompt——工业漏斗的每一层都有开源件，区别只在阈值要自己调、专训的 operator 没有现成的，这恰恰是最花时间的部分。

既然 VLM 可以靠近人类感官，为什么不在全流程尽量多用？


最直接是贵。六份报告没有一个披露数据处理本身的算力开销：HunyuanVideo 只给了「1000 万小时原始视频 → 8 亿 clip」的比例，Open-Sora 只给了 20 万美元的模型训练成本。

若乐观设每条 6 卡秒，8 亿 clip，约 135 万 A800 小时，折合 150 卡年，即约一千张 A800 跑五十多天才筛一遍；而数据引擎常常每个模型版本迭代或表现不达预期都需要重筛。另一边，CLIP+MLP 美学回归器是毫秒级一帧。



进一步地，过滤器和 captioner 不一样：captioner 的输出直接进训练，好坏由下游模型消化；过滤器定义的是整个语料的分布，所以它首先得是一件量具，量具首先需要稳定。

CLIP+MLP 美学回归器是完全冻结的：同一个模型同一个分数语义，跑十遍结果一致，分数分布可以画出来审计。VLM 则是生成器：prompt 换个措辞、甚至同一批数据重跑一遍，分数的含义都可能漂移。




对于一个模型，什么是好数据无法量化，唯一的测法是拿去训练到一定程度再 evaluate，烧掉无法计量的卡时，最后得到数据配比 recipe。所以在工业界，加组件、加约束提升几个点的算法也许根本没那么重要。

**Methods to get good data is all we need.**








## 参考资料

- Seedance 1.0: https://arxiv.org/abs/2506.09113
- Seedance 2.0: https://arxiv.org/abs/2604.14148
- MovieGen: https://arxiv.org/abs/2410.13720
- Open-Sora 2.0: https://arxiv.org/abs/2503.09642
- HunyuanVideo 1.5: https://arxiv.org/abs/2511.18870
- Koala-36M: https://arxiv.org/abs/2410.08260
- Tiger200K: https://arxiv.org/abs/2504.15182
- AVVA: https://arxiv.org/abs/2503.09205
