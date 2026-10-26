# 技术报告漫读：工业界怎么洗数据

2026-10-26 · 机器学习 · 视频生成 · 数据

在学界论文中，洗数据除了规格筛选外常常全程 VLM：对于 5B~14B 的小模型与十秒内的 480P 视频，只需要十几张卡较短时间内就能跑出一个差强人意的数据集。

最近尝试一遍洗数据 pipe，去重、初筛、重筛、富矿挖掘、事件窗定位，到 caption 重写与泄漏校验，多道工序除了第一道 ffprobe，其余全部由 VLM 严格出判决，做 SFT ablation 的效果仍然很不好，于是自然想到：数据清洗本质上是个处理数据算力、数据丰富程度/质量等因素的 Tradeoff。工业界是如何洗出一份好的数据集的？

直接翻阅技术报告：MovieGen、Seedance 1.0、HunyuanVideo 1.5、Open-Sora 2.0，加上专门讲数据集的 Koala-36M 和 Tiger200K，新一点的还对照了 Seedance 2.0 精读一遍 Data 部分。首先这些管线里的确有 VLM——caption 和奖励模型。而过滤段从头到尾几乎没有用 VLM 的。

## 一份账单

在展开之前，先看看我们小实验的消耗，后面所有的外推都要以其为参考：

| # | 工序 | 模型/口径 | 进 → 出 | 留存率 | 耗时（4×A800 基准） |
| --- | --- | --- | --- | --- | --- |
| 0 | 去重/探针过滤 | ffprobe | 152,300 → 100000 | 68% | ~0（CPU） |
| 1 | 宽定义初筛 | Qwen2.5-VL-7B 宽帧 | 100000 → 20000 | 19% | 2卡×~3天 ≈ 145 GPU·h |
| 2 | 收紧初筛 | Qwen3-VL 18帧 | 20000 → ~12000 | 61% | ~14.5h ≈ 58 GPU·h |
| 3 | 4fps 强 prompt 重筛 | Qwen3-VL 12-16段 | 12000 → 3000 | 23% | ~6h ≈ 24 GPU·h |
| 4 | 富矿再挖掘 | Qwen3-VL 15帧 | 30,000 候选 → 26,000 pass → 12,000 入档 | 93% pass | 9 GPU·h |
| 5 | 事件窗定位 | Qwen3-VL 8fps | 5,950 个缺窗 | 80.8% | ~2.2h ≈ 8.8 GPU·h |
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

可以看到，VLM 只出现在 caption、后训练的小规模判断，以及 RL 阶段的奖励模型。这三个位置其实是同一身份：captioner 力求贴近人类写描述的方式，奖励模型放大人类的偏好判断。而在过滤段，也即镜头切分、质量过滤、运动过滤、去重这一段，清一色是 PySceneDetect、PaddleOCR、Laplacian 方差、VMAF、光流、CLIP+MLP 这些工具和小模型。

关于开源状态，Seedance 2.0 无权重，MiniMax H3 给权重不给报告，HunyuanVideo 报告权重都给了——但把数据处理管线这个真正的核心科技开源出来的，工业界一家大厂都没有。开源世界能找到的完整管线全部来自开源阵营：Open-Sora 把报告里的管线原样放进 repo，NeMo Curator 和 Data-Juicer 提供现成的过滤去重框架，再配上开源的 captioner、美学回归器和 Koala-36M、Tiger200K 这样的数据集，社区是拿开源工具复现工业 pipe。

## 六层漏斗

把六份报告叠在一起，工业界洗数据的流水线大致是六层。第一层是切分：PySceneDetect、TransNetV2、FFmpeg 的场景分数把原始视频切成 2 到 16 秒的单镜头 clip，转场和特效由专门的小分类器清掉。Tiger200K 给过一个细节：阈值式的 PySceneDetect 会漏叠化转场，所以他们换了学习式的 TransNetV2。这一层没有理解，只有逐帧定位。

第二层是一批便宜的量具。水印和字幕交给 PaddleOCR 和专用检测器，裁出安全区，裁完保留面积不足就整条丢弃；清晰度用 Laplacian 方差和 VMAF；美学是 CLIP+MLP 小回归器；运动用光流和 VMAF motion，太静和太抖都删。全是冻结的小模型，毫秒级一条。2026 年的 Seedance 2.0 报告里，这一层还是同样的配置。

第三层是去重与配比，操作对象从单条视频换成了分布。clip 用视频表征模型编码，聚类之后每个近重复簇只留质量分最高的那条；再按主体、场景、动作、风格数分布，头部下采样、尾部定向补采（Seedance），MovieGen 则用聚类反比采样加一套 600 个动词的词表做概念平衡。

第四层 caption，VLM 在这里进场，但没有一家裸用通用 VLM：Seedance 在 Tarsier2 上专训了双语 captioner，MovieGen 微调 LLaMA3-Video，HunyuanVideo 用 OPA-DPO 后训练来平衡描述密度与幻觉，输出的都是 80 到 200 词的密集结构化描述；Seedance 甚至单独训了一个 prompt 改写器，保证用户输入和 caption 同分布。

第五层是数据金字塔。同一批语料按质量分出梯度：Open-Sora 2.0 是 7000 万的低分辨率池喂出 500 万的高分辨率精选，HunyuanVideo 从 1000 万小时原始视频筛出 8 亿 clip，逐级收紧，贵的高分辨率训练只吃塔尖那一层。

第六层是后训练和 RLHF，筛法在这里彻底变了：不再是过滤大池子，而是人工构造一个小数据集。prompt 池人工验证（HunyuanVideo 的 I2V RL 用了 100 多个类目），候选由不同 checkpoint 对同一 prompt 滚出，人来成对标注（Meta 用 GSB，Seedance 按维度选 best/worst，HunyuanVideo 标了约 1 万对 DPO），最后把这 1 万对人类偏好蒸馏成 VLM 奖励模型，供在线 RL 批量打分。8 亿 clip 对 1 万对标注，差五个数量级，越靠近塔尖，数据越小、越贵、也越真。顺带一提，HunyuanVideo 自己承认现有奖励模型分不清细粒度运动的好坏，所以才先做离线 DPO 再上在线 RL：VLM 裁判也是蒸馏来的，人分不清的它同样分不清。

既然我们用 VLM 洗得好好的，为什么大厂不在过滤段也用它？

## 自己拼一个漏斗

好在管线里的每一层都有开源对应物，方法是真的可以自己拼的。切分用 PySceneDetect 一行，转场敏感的场景换学习式的 TransNetV2；清晰度是 Laplacian 方差，运动是相邻帧光流的平均幅值，都是十几行 OpenCV 的事：

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

美学打分用 LAION 在人类评分上蒸馏的 CLIP+MLP，权重和结构都是现成的：

```python
import torch, torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
head = nn.Sequential(nn.Linear(768, 256), nn.Dropout(0.4),
                     nn.ReLU(), nn.Linear(256, 1))
head.load_state_dict(torch.load("sa_0_4_vit_l_14_linear.pth"))

def aesthetic(pil_img):
    x = CLIPProcessor.from_pretrained(clip.name_or_path)(
        images=pil_img, return_tensors="pt")
    with torch.no_grad():
        feat = clip.get_image_features(**x)
    return head(feat).item()                # 1 到 10 分
```

去重和配比在 embedding 空间里做，FAISS 几行：

```python
import faiss

emb = encode_with_siglip(clips)             # N x D，归一化
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
D, _ = index.search(emb, 2)                 # 最近邻含自身
keep = D[:, 1] < 0.95                       # 近重复只留一条
```

caption 这一层的开源选择最多，Qwen2.5-VL、Tarsier2 都能用，prompt 按六要素（主体、动作、环境、光照、运镜、风格）写就行。拼到这里你会发现：工业漏斗的每一层都有开源件，区别只在阈值要自己调，而专训的 operator 没有现成的——这恰恰是最花时间的部分。

既然我们用 VLM 洗得好好的，为什么大厂不用？

## 贵多少

最直接的解释当然是贵。这个说法大差不差，但它属于"万能模版"式的回答，什么问题都可以拿"贵"来搪塞，所以我们不妨把账认真算一遍。有意思的是，六份报告没有一个披露数据处理本身的算力开销：HunyuanVideo 只给了"1000 万小时原始视频 → 8 亿 clip"的比例，Open-Sora 只给了 20 万美元的模型训练成本。所以这笔账只能拿自己的实测去外推。

把表 1 的耗时加起来约 257 GPU·小时，摊到每条源视频约 6 卡秒；乘以 8 亿 clip，约等于 135 万 A800·小时，折合 150 卡年。什么概念呢？一千张 A800 不吃不喝跑五十多天，才筛完一遍；而数据引擎的常态是每个模型版本迭代都重筛。另一边，CLIP+MLP 美学回归器是毫秒级一帧，前后差了三个数量级。所以在预训练的规模上，全程 VLM 不是贵一点，是根本跑不动。

可能有读者觉得：那不就是纯粹的贵而已，等 VLM 推理降价了自然就换过去了。咋看之下有道理，但笔者认为不成立，问题出在过滤器这个角色的特殊性上。

## 量具与裁判

过滤器和 captioner 不一样：captioner 的输出直接进训练，好坏由下游模型消化；过滤器定义的是整个语料的分布，所以它首先得是一件量具。而量具的第一美德是稳定，不是聪明。

CLIP+MLP 美学回归器是冻结的：同一个模型、同一个分数语义，跑十遍结果一致，失败模式已知，分数分布可以画出来审计。VLM 则是生成器：prompt 换个措辞、底座升个版本、甚至同一批数据重跑一遍，分数的含义都可能漂移。同一条视频这个月 4.2 分、下个月 4.8 分，对 caption 来说无所谓，对过滤器来说是灾难，因为整个语料的筛选标准跟着变了。

这里还有一层更微妙的关系。美学回归器是什么？是在人类美学评分数据上蒸馏出来的 CLIP+MLP；VLM 是什么？同样是人类判断的蒸馏。两者本是同源，所以大厂在过滤段做的选择，其实不是要哪份知识，而是要哪种容器：冻结的标量，还是会漂移的语言。

写到这里，笔者回头看了眼自己的管线，发现一个有点好笑的事实。我们为了把 VLM 用便宜，把 12–16 段拼成一张蒙太奇、一个请求塞 5 条视频 × 3 个 chunk，这些技巧的本质，就是逼 VLM 一次前向吐出尽可能多的标量判决，把它往编码器的用法上推。换句话说，在成本压力下，VLM 重管线会自发长出小模型管线的行为模式。我们和大厂，其实是在从两端做同一件事。

## 代理塔

再往深一层想，就碰到"什么是好数据"本身了。理想的定义其实很清楚：U(x) 等于用 x 训练之后模型的期望提升。麻烦的是这个量没法计算，唯一的测法是真的拿去训练再评测。所以整条数据管线本质上是一座代理塔：塔尖是真训真测，塔下面的一切，元数据、质量分、美学分、embedding 几何、VLM 判断、甚至人工标注，都是对塔尖的代理，越往上越准也越贵。漏斗的规则是便宜的看全部、贵的只看幸存者，而这个排序由单条成本决定，并不由哪个信号最接近 U 决定。

用这个视角回看，一些原来散落的现象就都能解释了。Tiger200K 为什么敢不做任何训练实验就发数据集？因为大家都把代理当成了本体，默认过滤了就是好的。学界论文为什么清一色 VLM 洗数据？算力有限，级联退化成一层，那就只留最全能的那层。HunyuanVideo 为什么花力气用 RL 专训 captioner，而不是直接用 Qwen？因为 caption 是离塔尖最近、又能低成本对齐的环节。各家为什么都把数据飞轮当终局？因为飞轮把塔尖变成了日常操作。

这也意味着"好数据"并没有属性式的定义，只有阶段式的：同一条 clip，在 256p 预训练是好数据，到 768p 精调就是坏数据。"好"不是视频自己的属性，而是视频、训练阶段和目标三者之间的关系。

## 拓展思考

最后说一个可能被未来打脸的预测。当时实测的稳定吞吐，7B 大约 35–40 条/分/卡，换成大四倍的 30B 大约 30 条/分/卡，模型大了四倍，单卡吞吐基本没掉，可见 serving 效率的进步是实打实的。如果 VLM 推理成本再降两个数量级，量具论会松动，过滤段逐步换成 VLM judge 是大概率事件，AVVA 这类工作和大厂 RL 里的 VLM 奖励模型已经是先声。到那时候回头看，CLIP+MLP 美学回归器大概就像我们今天看像素阈值过滤一样原始。

两种管线正在相向而行：我们在学着把 VLM 用成编码器，大厂在把手工过滤器换成 learned judge。至于成本真正崩塌之后，"好数据"会等来一个认识论上的定义，还是只换一个更贵的共识，笔者暂时还没有想清楚。

## 文章小结

这篇文章用我们自己的一份账单和六份技术报告，把工业界的洗数据管线拆成了六层，并回答了一个问题：为什么过滤段没有 VLM。直接原因是贵；更本质的原因是身份，VLM 在这条管线里始终以人类判断放大器的身份出现，而过滤器要的是量具的稳定，两者要的从来不是同一种东西。

## 参考资料

- Seedance 1.0: https://arxiv.org/abs/2506.09113
- Seedance 2.0: https://arxiv.org/abs/2604.14148
- MovieGen: https://arxiv.org/abs/2410.13720
- Open-Sora 2.0: https://arxiv.org/abs/2503.09642
- HunyuanVideo 1.5: https://arxiv.org/abs/2511.18870
- Koala-36M: https://arxiv.org/abs/2410.08260
- Tiger200K: https://arxiv.org/abs/2504.15182
- AVVA: https://arxiv.org/abs/2503.09205
