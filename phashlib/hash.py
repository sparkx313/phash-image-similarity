"""三个 64 位感知哈希，以及它们之间的距离。

它们都把一张图归约成「64 个数各自是否超过某个阈值」，所以整个哈希就是一个 Python
整数，汉明距离只需一次 `bin()` 调用。区别只在那 64 个数是什么：

* `ahash` —— 8x8 重采样后的像素与它自己的均值比较。最便宜，但亮度一变均值跟着变，
  哈希就跟着变。
* `dhash` —— 8x8 次比较，每个像素与它右边那个比。对任何整体亮度变化免疫（差值不动），
  但它本质是边缘检测器，裁掉一条边缘就会让它移动。
* `phash` —— 先重采样到 32x32，取 8x8 低频 DCT 块，与自身 63 个交流系数的中位数比较。
  这是主力：它丢掉的正好是重新编码会改动的高频细节。
"""

from .dct import low_frequency
from .pixels import resize


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bits(values, threshold):
    packed = 0
    for index, value in enumerate(values):
        if value > threshold:
            packed |= 1 << index
    return packed


def ahash(image, size=8):
    """均值哈希：`size` x `size` 缩略图与它自己的均值比较。"""
    small = resize(image, size, size)
    values = [value for row in small.pixels for value in row]
    return _bits(values, sum(values) / len(values))


def dhash(image, size=8):
    """差分哈希：每个像素与它右侧相邻像素比较。"""
    small = resize(image, size + 1, size)
    values = [right - left
              for row in small.pixels
              for left, right in zip(row, row[1:])]
    return _bits(values, 0.0)


def phash(image, size=32, keep=8, threshold="median"):
    """低频谱的感知哈希。

    `threshold` 取 "median"（HashLib 的原始行为，对单个过强的系数稳健）或 "mean"
    （多数 Python 移植版的做法）。基准测试实测两者能扛住多少次编辑、代价是多少误判，
    所以这不是风格选择。

    `[0][0]` 处的直流系数是块均值，因此被排除在阈值之外 —— 否则一张亮图会把阈值抬到
    自己的内容之上。但它仍然占据布局里的第 0 位，也就是说第 0 位对几乎所有图像都相同，
    于是哈希实际携带 63 位可用信息，而不是 64 位。
    """
    if threshold not in ("median", "mean"):
        raise ValueError(f"unknown threshold {threshold!r}")
    if keep > size:
        raise ValueError(f"cannot keep {keep} frequencies from a {size}x{size} block")
    small = resize(image, size, size)
    coefficients = low_frequency(small.pixels, keep)
    values = [value for row in coefficients for value in row]
    rest = values[1:]
    cut = _median(rest) if threshold == "median" else sum(rest) / len(rest)
    return _bits(values, cut)


def popcount(value):
    return bin(value).count("1")


def hamming(first, second):
    """相差的位数 —— 下面每一个阈值都是对着它写的。"""
    return popcount(first ^ second)


def to_hex(value, width=16):
    return "%0*x" % (width, value)


def from_hex(text):
    return int(text, 16)


HASHES = {"phash": phash, "dhash": dhash, "ahash": ahash}
