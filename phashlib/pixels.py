"""像素操作：每个哈希开头都要做的重采样，外加那些替代「两张拷贝之间发生过什么」的
编辑。

这里没有任何函数会修改输入，所以基准测试可以留着原图随时比较。
"""

import math
import random

from .imagio import Image, _clamp


def _weights(scale, index, limit):
    """一个输出像素的 `(源起始下标, 归一化覆盖权重)`。

    输出像素 `index` 覆盖源图的 `[index*scale, (index+1)*scale)` 区间，所以它的值应该
    是该区间下所有源像素的面积加权均值。放大时权重只有一个且为 1.0，等价于重复像素；
    缩小时会对多个求均值。这就是 INTER_AREA / PIL BOX，而不是最近邻：最近邻缩小会丢掉
    哈希本应依赖的细节，还会让哈希对尺寸的一像素变化过度敏感。
    """
    left = index * scale
    right = left + scale
    start = min(int(math.floor(left)), limit - 1)
    stop = min(int(math.ceil(right)), limit)
    weights = [max(0.0, min(right, position + 1) - max(left, position))
               for position in range(start, stop)]
    total = sum(weights)
    if total <= 0:
        return start, [1.0]
    return start, [weight / total for weight in weights]


def resize(image, width, height):
    """面积均值重采样，拆成两次可分离的遍历完成。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"cannot resize to {width}x{height}")
    rows = image.pixels
    if width != image.width:
        columns = [_weights(image.width / width, x, image.width)
                   for x in range(width)]
        rows = [[sum(value * weight
                     for value, weight in
                     zip(row[start:start + len(weights)], weights))
                 for start, weights in columns]
                for row in rows]
    line_weights = [_weights(image.height / height, y, len(rows))
                    for y in range(height)]
    out = []
    for start, weights in line_weights:
        block = rows[start:start + len(weights)]
        totals = [0.0] * width
        for source_row, weight in zip(block, weights):
            for x, value in enumerate(source_row):
                totals[x] += value * weight
        out.append(totals)
    return Image(width, height, out)


def _map_pixels(image, function):
    return Image(image.width, image.height,
                 [[function(value) for value in row] for row in image.pixels])


def brightness(image, factor=1.0, offset=0.0):
    """先缩放再平移，并夹在 0..255 —— 经典的曝光滑动。"""
    return _map_pixels(image, lambda v: _clamp(v * factor + offset))


def gamma(image, exponent):
    """非线性重映射：同一张图换了另一条色调曲线。"""
    return _map_pixels(image,
                       lambda v: _clamp(255.0 * ((v / 255.0) ** exponent)))


def invert(image):
    return _map_pixels(image, lambda v: _clamp(255.0 - v))


def contrast(image, amount):
    """把每个像素按 `amount` 推离均值（1.0 表示原样）。"""
    flat = [value for row in image.pixels for value in row]
    mean = sum(flat) / len(flat)
    return _map_pixels(image, lambda v: _clamp(mean + (v - mean) * amount))


def noise(image, amount=12.0, seed=0):
    """每个像素加均匀分布的 +-`amount` 抖动，同一 seed 结果确定。"""
    rng = random.Random(seed)
    return _map_pixels(image, lambda v: _clamp(v + rng.uniform(-amount, amount)))


def blur(image, radius=1):
    """方形箱式模糊，一遍 —— 足够模仿一次柔和的重新编码。"""
    rows = []
    for y in range(image.height):
        row = []
        for x in range(image.width):
            total = 0.0
            count = 0
            for sy in range(max(0, y - radius), min(image.height, y + radius + 1)):
                source = image.pixels[sy]
                window = source[max(0, x - radius):min(image.width, x + radius + 1)]
                total += sum(window)
                count += len(window)
            row.append(total / count)
        rows.append(row)
    return Image(image.width, image.height, rows)


def crop(image, top=0, left=0, right=0, bottom=0):
    """从四边整行整列地裁掉，剩下的内容位置不动。"""
    if top + bottom >= image.height or left + right >= image.width:
        raise ValueError(f"crop {top}/{left}/{right}/{bottom} eats the image")
    rows = [row[left:image.width - right]
            for row in image.pixels[top:image.height - bottom]]
    return Image(len(rows[0]), len(rows), rows)


def centre_crop(image, fraction):
    """保留画面中间 `fraction` 面积的内容，然后不补回尺寸。"""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction {fraction} is not in (0, 1]")
    keep_w = max(1, int(round(image.width * math.sqrt(fraction))))
    keep_h = max(1, int(round(image.height * math.sqrt(fraction))))
    left = (image.width - keep_w) // 2
    top = (image.height - keep_h) // 2
    return crop(image, top, left, image.width - left - keep_w,
                image.height - top - keep_h)


def shift(image, dx=0, dy=0):
    """整像素移动内容，越界部分按平铺方式绕回。"""
    width, height = image.width, image.height
    rows = []
    for y in range(height):
        source = image.pixels[(y - dy) % height]
        offset = (-dx) % width
        rows.append(source[offset:] + source[:offset])
    return Image(width, height, rows)


def flip_horizontal(image):
    return Image(image.width, image.height, [list(reversed(row))
                                             for row in image.pixels])


def rotate_left(image):
    """逆时针 90 度：先转置，再颠倒行序。"""
    transposed = [list(column) for column in zip(*image.pixels)]
    transposed.reverse()
    return Image(len(transposed[0]), len(transposed), transposed)


def clamp8(image):
    """四舍五入到 8 位整数，也就是一次重新编码会做的事。"""
    return _map_pixels(image, _clamp)
