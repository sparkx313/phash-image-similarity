# phash-image-similarity

只用 Python 标准库实现的图像感知哈希：`phash` / `dhash` / `ahash` 把一张图压成 64 位整数，
汉明距离衡量两张图差多少，分带索引（BKW）让几十万件图库不必逐张比对就能找到近重复。

附带 7 组实测基准。这个库的主要价值不是「又一个 pHash 移植」，而是把这类哈希
**到底对什么免疫、对什么不免疫**用数据说清楚：重新编码免疫，几何变换不免疫，
中间那一档（裁切）在三家哈希之间排名正好相反。

- 零依赖：`numpy`、`Pillow`、`opencv` 都不需要，读写图像是自己实现的 PNM/BMP 解码器。
- 49 个单元测试，一条命令跑完。
- `benchmarks/last_run.txt` 是真实的一次运行输出，下面所有数字都从那里抄来，不是文档里的装饰。

## 它解决什么，不解决什么

**解决**：同一张图的多个副本 —— 缩放、亮度/对比度/gamma 微调、加噪、模糊、量化到 8 bit、
存成不同格式再读回来。这类「会破坏 MD5、不该破坏感知哈希」的编辑。

**不解决**：裁掉画面内容、旋转、镜像、反相。这不是实现得不好，是这类哈希的定义决定的：
频谱按位置索引，内容一动，哈希就动。

实测这三档的差别（24 张场景图 × 15 种编辑，5 bit 阈值内的比例，原文见下文第 1 节）：

| 编辑类别 | phash | dhash | ahash |
|---|---|---|---|
| 画质/色调 | 99.1% | 98.1% | 100.0% |
| 裁切与位移 | 11.1% | 56.9% | 58.3% |
| 反相/镜像/旋转 | 0.0% | 0.0% | 1.4% |

所以 README 剩下的部分在做两件事：把「画质这一档为什么可靠」写成可复现的表，
把「裁切这一档你该选哪个哈希、代价是什么」写成同样可复现的表。

## 安装

```bash
git clone <this repo> && cd phash-image-similarity
pip install -e .        # 或者直接 sys.path.insert(0, ".")，仓库本身可运行
```

要求 Python 3.9+，没有第三方依赖。

## 快速上手

下面每一段都在 CPython 3.11.9 / Windows 上跑过，输出原样粘贴。

```python
from phashlib import scene, phash, dhash, ahash, to_hex

a = scene(3)                     # 合成场景图：渐变底色 + 几个实心图形，种子决定布局
print(to_hex(phash(a)))          # 3c95cae213bcec43
print(to_hex(dhash(a)))          # 9faf9d3533bf3fff
print(to_hex(ahash(a)))          # ffffff3f06000000
```

距离：同一张图重新缩放并加噪 → 2 bit；另一张图 → 30 bit。

```python
from phashlib import hamming, noise, phash, resize, scene

a = scene(3)
b = noise(resize(a, 96, 96), 12.0, seed=2)
print(hamming(phash(a), phash(b)))         # 2   —— 同一个东西，只是被折腾过
print(hamming(phash(a), phash(scene(4))))  # 30  —— 两个不同的东西
```

把一张图存成 PGM 再读回来，哈希一位不差：

```python
import os, tempfile
from phashlib import hamming, load, phash, save_pgm, scene

a = scene(3)
path = os.path.join(tempfile.mkdtemp(), "a.pgm")
save_pgm(path, a)
print(hamming(phash(a), phash(load(path))))  # 0
```

裁切是另一回事 —— 中心裁掉 20% 边距后，128×128 变成 114×114，哈希动了 16 bit：

```python
from phashlib import centre_crop, hamming, phash, scene

a = scene(3)
c = centre_crop(a, 0.8)
print(c.width, c.height)                   # 114 114
print(hamming(phash(a), phash(c)))         # 16
```

几何变换同样不在「近重复」里：

```python
from phashlib import flip_horizontal, hamming, invert, phash, rotate_left, scene

a = scene(3)
print(hamming(phash(a), phash(invert(a))))           # 62
print(hamming(phash(a), phash(flip_horizontal(a))))  # 30
print(hamming(phash(a), phash(rotate_left(a))))      # 28
```

索引：分带 BKW，`coverage` 直接告诉你保证覆盖到几个 bit。

```python
from phashlib import HammingIndex, noise, phash, resize, scene

a = scene(3)
b = noise(resize(a, 96, 96), 12.0, seed=2)
index = HammingIndex([("a", phash(a)), ("b", phash(b)), ("c", phash(scene(4)))], bands=8)
print(index.search(phash(a), 5))     # [('a', 0), ('b', 2)]
print(index.coverage)                # 7  —— max_distance <= 7 时召回与全扫完全一致
print(index.last_examined)           # 2  —— 只比了 2 个候选，没有扫全库
```

聚类去重（并查集，所以分组是传递的）：

```python
from phashlib import near_duplicates, noise, resize, scene

a = scene(3)
b = noise(resize(a, 96, 96), 12.0, seed=2)
print(near_duplicates([("a", a), ("b", b), ("c", scene(4))], max_distance=5))
# [['a', 'b'], ['c']]
```

## 实测结果

跑 `python benchmarks/run_benchmarks.py`，完整原始输出 322 行存在
[`benchmarks/last_run.txt`](benchmarks/last_run.txt)。语料是 24 张种子场景图，另加
`stripes` / `checker` / `flat-grey` / `flat-black` 四张刻意放进去的病态图。
下面的表从那次运行原文摘录，解释文字是我写的。

### 1. 每种编辑把哈希推多远

对 24 张场景图各做一遍，距离是相对未编辑原图差了多少 bit，`med min max` 是在场景之间取的：

```
  edit                phash          dhash          ahash
                      med min max    med min max    med min max
  quality
  rescale down 0.75     0.0    0    0   0.0    0    3   0.0    0    0
  rescale up 1.4        0.0    0    2   0.0    0    2   0.0    0    1
  brightness +30        0.0    0    2   0.0    0    2   0.0    0    2
  exposure x1.3         0.0    0    6   0.0    0    2   1.0    0    4
  gamma 0.8             0.0    0    4   0.0    0    2   1.0    0    4
  contrast 1.25         0.0    0    0   0.0    0    2   0.0    0    1
  noise +-12            0.0    0    6   2.0    0   14   0.0    0    2
  blur r=1              0.0    0    2   0.5    0    2   0.0    0    1
  quantise to 8 bit     0.0    0    2   0.0    0    1   0.0    0    2
  framing
  centre crop 85%       6.0    4   14   3.5    1    8   2.0    1    8
  edge crop 10%        10.0    4   16   3.0    2   10   3.0    1    8
  shift 6px            14.0   10   22  12.0    7   20   7.0    0   14
  other picture
  invert               62.0   62   62  63.0   59   64  64.0   62   64
  flip horizontal      32.0   28   34  42.0   29   59  22.0    4   52
  rotate 90            30.0   20   38  32.0   15   51  30.0   18   42
```

读法：**色调那一档三家都一样好** —— 最差的画质通过率是 98.1%（`dhash`，被 `noise +-12`
那行拖下去的，它的中位数是 2.0、最大值 14：差分哈希读的是相邻像素的边，噪声正好只动边）。
真正按构造区分开三家的是整图缩放那两行（`exposure x1.3`、`gamma 0.8`）：`ahash` 的中位数
走到 1.0，而 `phash` 和 `dhash` 停在 0.0 —— 因为 `ahash` 的阈值就是图像均值，均值被抬，
阈值跟着被抬。纯粹加一个亮度偏移时三家都还是 0.0。

**裁切那一档排名反过来了**。`phash` 读整帧的低频谱，把内容从帧里剪掉对它不是小改动：
11.1% 的裁切对还在 5 bit 内，而 `dhash` 是 56.9%、`ahash` 是 58.3%。差分哈希只看见它取差
的那条边，均值哈希只看见缩略图的明暗配比 —— 所以它们更不在乎你剪掉了什么。

反相那行 62 bit 值得单独说一句：对一个读「高于/低于」的哈希来说，底片是**相反的画面**，
不是同一画面的小扰动。

病态图（`phash` 到自己各编辑的距离）：

```
  frame          as-is   gamma   noise  invert
  stripes            0       0       8      62
  checker            0       4      16      46
  flat-grey          0      19      32      25
  flat-black         0       0      32      32
```

`as-is` 是自己跟自己对，大于 0 就是 bug。纯色图上每个像素都**正好压在阈值上**，
64 个数各落在阈值哪一边是浮点舍入决定的，不是画面决定的 —— 空白图的哈希是一枚没被碰过
就一直朝同一面的硬币。一次 ±12 噪声让它在 32 bit 上跳动。这几张图放进语料，就是为了让表格
说出来，而不是被平均值抹平。

### 2. 重复阈值能放在哪儿

`related` = 一张场景图和它 12 种编辑之一（288 对，不含反相/镜像/旋转，它们不是重复）；
`unrelated` = 276 对不同的场景。误并 = 不相关的一对落进阈值内，漏检 = 相关的一对掉出去。

`phash`：

```
    threshold   FP rate   FN rate   quality FN  framing FN
            0     0.000     0.361       0.148      1.000
            2     0.000     0.274       0.032      1.000
            4     0.000     0.229       0.009      0.889
            5     0.000     0.229       0.009      0.889
            6     0.000     0.174       0.000      0.694
            8     0.000     0.156       0.000      0.625
           10     0.000     0.108       0.000      0.431
           12     0.000     0.069       0.000      0.278
           14     0.000     0.035       0.000      0.139
    closest unrelated pair: scene-00 / scene-14 at 20 bits apart; related median
    0.0, unrelated median 32.0.
```

`dhash` 在 5 bit 时 framing 漏检 43.1%，`ahash` 41.7%，且 `ahash` 在 2 bit 就出现误并
——**它最近的两个不同场景只有 2 bit 之差**（`scene-03` / `scene-04`）。这是一次真实碰撞，
不是理论风险。

文献常用的 5 bit 处：

```
  hash       wrong merges quality misses framing misses closest strangers
  phash              0.0%           0.9%          88.9%              20 bits
  dhash              0.0%           1.9%          43.1%              12 bits
  ahash              0.4%           0.0%          41.7%               2 bits
```

5 bit 处只有 `ahash` 会把不同场景并到一起（0.4%），`phash` 与 `dhash` 不会；但三家都
**丢掉大部分裁切编辑**。「裁掉画面内容的图算不算同一张图」是一个比「找近重复」更小的承诺，
这个 README 只保留后者。

FP 那一列也是合成语料的下限：平滑生成场景彼此相距很远（中位数 30 bit 以上），
真实照片集会挤得多。把这里读成**取舍的形状**，选定数字前在你自己的图上重新标定。

### 3. phash 用中位数切还是均值切

HashLib 原版拿 63 个交流系数的**中位数**当阈值，多数 Python 移植版用**均值**。同一批场景，
两种切法：

```
  edit                median cut   mean cut
  noise +-12                 0.0        0.0
  blur r=1                   0.0        0.0
  centre crop 85%            6.0        7.0
  edge crop 10%             10.0        8.0
  shift 6px                 14.0       14.0
  invert                    62.0       63.0
  flip horizontal           32.0       32.0
  rotate 90                 30.0       29.0
```

两种切法在 24 张场景里有 3 张给出完全相同的哈希，不同的那些相差 1 到 13 bit。均值会被
最响的那个系数拽走，中位数不会 —— 所以能把它们分开的画面，就是块里只有一个突出系数的画面，
而纯色图恰好长这样：

```
  frame            cuts apart  noise moves
  scene-00                  3            0
  scene-01                  1            0
  stripes                  22            8
  checker                  15           16
  flat-grey                25           32
  flat-black                0           32
```

第一列是**算法选择**，第二列是**画面选择**。相关对落在 5 bit 内的数量：中位数切 222/288，
均值切 225/288 —— 平滑合成场景上谁都不是大胜。默认选**中位数切**，理由是空白帧的谱就是
一个响系数，均值会把整条切线推走，中位数不会。

### 4. 64 个 bit 里到底有几个带信息

28 张图的语料统计：

```
  bit 0 set on 27/28 frames; never set anywhere: none; always set: none
  bits that vary somewhere in the corpus: 64 of 64.
  the frames that leave bit 0 clear: flat-black.
```

第 0 位是直流系数，跟一个永不包含直流的阈值比 —— 所以只要图不是纯黑它就置位。
全语料 378 对：最小 20、中位 32.0、最大 44、均值 31.2（独立 bit 应期望 32）。
最近的两个画面是 `scene-00` 和 `scene-14`，20 bit：**低于 20 就判重复的哈希，在这对普通
场景上就失手了**，而且这对不是病态图，所以「只有空白图有风险」这个假设在本语料上是错的。

两张空白图（灰场与黑场）彼此相距 32 bit —— 它们不是近重复，这正是第 1 节那枚硬币的预测。

### 5. 索引跳过扫描之后还准不准

查询 = 库里某个哈希翻转固定数量 bit，所以精确答案已知，召回是**测量**而不是判断：

```
    store  d max  bands  guarantee    recall    cands    index s    scan s vs scan
     1000      4      8       d<=7     50/50     3.1%       0.00      0.03 21.4x exact
     1000      6      8       d<=7     50/50     3.2%       0.00      0.03 20.8x exact
     5000      6      8       d<=7     50/50     3.0%       0.01      0.16 23.3x exact
    20000      6      8       d<=7     50/50     3.1%       0.03      0.63 21.2x exact
    20000      8     16      d<=15     50/50    64.5%       0.82      0.66 0.8x exact
    20000     10      8       d<=7     49/50     3.1%       0.03      0.69 20.9x OVER THE GUARANTEE
```

`max_distance <= bands - 1` 处召回 100%，这是鸽笼原理的必然结果而不是调出来的成绩：相距 `d`
的两个哈希在 `d + 1` 个带里至少有一个带完全相同，于是必然同桶。最后那行拿 8 带索引去查距离
10（覆盖 7），**丢了一个匹配**（49/50）—— 召回、候选比例、内存三列由固定种子决定，
每次跑都一样；`vs scan` 那两个耗时列会漂，多跑几次在 0.7x～1.0x 之间来回。
边界就在表里，`coverage` 字段把它写出来，而不是留给线上事故。

`cands` 是候选占全库的比例，它是全部意义所在：全扫的时间随库大小增长，它不涨。
16 带那行付的是另一半代价 —— 带数必须整除 64，覆盖距离 8 就只能是 16 个 4 bit 的桶，
一个 4 bit 桶装 1/16 的库，取 16 个桶之后索引要看掉 **64.5%** 的语料，加速比落到 1x 以下，
这一档它什么都不买。差几个 bit 以上，老老实实用 64 位整数线性扫是诚实的答案。

建索引 20000 条哈希耗时 0.09 s、2048 个桶；构造期峰值内存：朴素列表 1153 KB，索引 10188 KB
—— 每条记录按带被引用一次，所以 8 带索引约等于它所替代的列表的 8 倍指针，平均每带 1273 KB。

### 6. 一张图多少钱

解码阶段（三种格式，同一帧画面）：

```
  source              decode    bytes  resize 8x8  resize 32   DCT 8x8    phash
  PGM P5 binary       0.0118    16399      0.0020     0.0039    0.0009   0.0049
  BMP 24 bit          0.0123    49206      0.0018     0.0036    0.0008   0.0052
  PGM P2 ascii        0.0180    57806      0.0018     0.0059    0.0009   0.0044
```

三个文件都哈希成与内存里那帧**完全相同的整数**（断言过，不是假设），所以格式的差异只存在于
解码这一步：ASCII PNM 读得最慢、文件最大，之后的成本一模一样。

128×128 单帧各阶段（各 40 次取中位数）：

```
  stage                         ms    share
  resize to 32x32             4.09      80%
  DCT to 8x8                  0.88      17%
    of which factors          0.04       1%
  threshold, pack, residual     0.11       2%
  phash, end to end           5.08     100%
```

最后一行是整次调用，上面几行是分开计时，所以 `threshold, pack, residual` 是减出来的残差，
包含两次计时之间的噪声 —— 某些运行里它是负的（-3% 之类），那只说明两件被测量的事已经
占满了整次调用，不说明阈值和打包是免费的。诚实的结论是第一行：**面积平均重采样就是这个哈希**。
它读源图的每一个像素，而 DCT 每轴只碰 8 个频率。余弦表每次调用都重建、只占 0.8%，
缓存它不会改变这个数。

全语料（28 帧，三遍取最好）：

```
  hash        total s   ms/image    peak KB
  phash          0.14        5.0        184
  dhash          0.06        2.2         66
  ahash          0.05        2.0         54
```

`ahash` 最便宜（读 8×8 缩略图）、`dhash` 读 9×8、`phash` 读 32×32 再加一次变换。
但第 1 节质量那几行三家只差几个 bit，**所以选型看的是裁切行为和误判代价，不是速度。**

### 7. 这些哈希聚出来的组

100 张图 = 24 个场景各存 4 份（原图、缩小、加噪、中心裁切）加 4 张病态图，用
`near_duplicates()` 按 `phash` 分组：

```
  at max_distance 5: 52 clusters, 24 groups whose three quality copies
  merged completely, 0 crops that joined their own group, 24 crops left on
  their own, 0 clusters that span two different pictures.

    scene-00/noisy, scene-00/orig, scene-00/small
    scene-01/noisy, scene-01/orig, scene-01/small
    ...
    left on its own: scene-00/cropped, scene-01/cropped, scene-02/cropped
```

阈值扫描：

```
  max_distance  clusters clean groups  crops in crops alone over-merges
             3        53           23         0          24           0
             5        52           24         0          24           0
             7        48           24         4          20           0
            12        33           24        19           5           0
            20        17           11        14           0           4
```

第一张裁切图在 `max_distance` 6 加入自己的组，24 张全都在 16 以内加入；而本语料最近的两个
不同场景相距 20 bit。所以**阈值 16 换来裁切，代价是花掉 80% 的距离预算**，还没换来第一次误并
—— 且这个预算是在平滑合成场景上量的，陌生人之间离得很远；真实照片集里它们挤得多。

`near_duplicates` 是并查集，所以分组传递：A~B 且 B~C 时三者同组，哪怕 A~C 超线。去重要的正是
这个，但这也是「一串渐进编辑塌成一大堆」的原因 —— 某个组比预期大时，原因在这里。

## 它是怎么决定的

- **重采样**：可分离的面积平均（INTER_AREA / PIL BOX 的做法），每个目标像素按覆盖权重累加。
  缩小是加权平均而不是抽样，所以抗锯齿噪声进不了哈希 —— 代价是第 6 节里它占了八成时间。
- **phash**：32×32 块 → 8×8 低频 DCT-II → 与 63 个交流系数（不含直流）的中位数比大小。
  直流系数字典序位置 0，但它被排除在阈值之外；否则一张亮图会把切线抬到自己内容之上。
- **dhash**：9×8 重采样，每个像素与右邻比符号。对整体亮度免疫（差值不动）。
- **ahash**：8×8 缩略图与自己的均值比。最便宜，阈值随亮度漂移。
- **索引**：64 bit 切成 `bands` 段，每段一个哈希表；查询只取自己各段的桶并集去重。
  带数必须整除 64，所以 `_bands_for` 向上取到 2 的幂（要 3 带会用 4 带 —— 桶更多，保证不变）。
- **图像 IO**：手写 PNM（P1–P6，含二进制 maxval<256 与 16 bit）和未压缩 BMP
  （1/4/8/24/32 bpp、调色板、自上而下与自下而上、4 字节行对齐）。彩色按 BT.601 亮度折成灰度。

## 局限

- 不处理几何：裁切、位移、旋转、镜像都会明显移动哈希，第 1、2、7 节量了具体幅度。
- 不保证跨空白图：纯色帧的哈希由浮点舍入决定，±12 噪声能让它动 32 bit。
- 只读 PNM 和 BMP。**没有 PNG/JPEG 解码** —— 这不是遗漏，是为了保持零依赖；接你的图片库时
  把 `load()` 换成你的解码器，后面全是纯 Python。
- `phash` 实际携带 63 位可用信息（第 4 节），不是 64 位。
- 数字来自单机 CPython 3.11.9 上的合成语料；召回与内存的结论可以外推，耗时不能。
- 索引精确性只在 `max_distance <= bands - 1` 内成立，超出后是概率性的。
- 纯 Python，逐像素循环。它是可读、可测、可复现的参考实现，不是最快的实现。

## API

| 名字 | 作用 |
|---|---|
| `phash(image, size=32, keep=8, threshold="median")` | 64 位感知哈希，`threshold` 可选 `"mean"` |
| `dhash(image, size=8)` / `ahash(image, size=8)` | 差分哈希 / 均值哈希 |
| `hamming(a, b)` / `popcount(v)` | 距离与置位数 |
| `to_hex(v)` / `from_hex(s)` | 与 16 位十六进制字符串互转 |
| `load(path)` / `loads(data)` | PNM/BMP → `Image`（灰度浮点像素） |
| `save_pgm` / `save_ppm` / `save_bmp` | 写回三种格式 |
| `HammingIndex(pairs, bands=8)` | `add` / `search` / `coverage` / `last_examined` / `buckets` |
| `linear_search(pairs, query, max_distance)` | 基线全扫，与索引在覆盖内答案一致 |
| `near_duplicates(images, max_distance=5, hash_function=None)` | 并查集聚类 |
| `resize` `brightness` `gamma` `contrast` `noise` `blur` `crop` `centre_crop` `shift` `flip_horizontal` `rotate_left` | 造编辑用的图像操作（`phashlib.pixels`，另有 `clamp8`） |
| `scene(seed)` / `stripes(seed)` / `corpus(count)` | 可复现的合成语料 |

## 目录

```
phashlib/
  hash.py      三个哈希与距离
  dct.py       部分 DCT-II（只算用到的低频）
  pixels.py    重采样与编辑算子（面积平均、模糊、噪声、裁切…）
  imagio.py    PNM / BMP 读写
  index.py     分带索引与并查集聚类
  scenes.py    种子合成图
benchmarks/
  run_benchmarks.py   7 节实测
  last_run.txt        真实一次运行的原文
tests/
  test_phash.py       49 个测试
```

## 复现

```bash
python -m unittest discover -s tests -p 'test_*.py' -t tests
# Ran 49 tests
# OK

python benchmarks/run_benchmarks.py        # 约 75 秒，末尾打印它写入的文件路径
```

## License

MIT，见 `LICENSE`。
