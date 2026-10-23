"""合成测试图。

感知哈希总得有个「感知」的对象：平滑的明暗变化、几道硬边、位置不同的图形。逐像素随机
噪声只会让每张图和每张图等距，从而让这里每个基准都失去意义，所以场景图是「渐变背景 +
实心图形」搭出来的。种子决定布局，布局在每台机器上完全一致 —— 这才让 README 里的数字
是可复现的，而不是装饰。
"""

import math
import random

from .imagio import Image


def scene(seed=0, width=128, height=128, shapes=4):
    """一张合成图：渐变底色上画几个实心图形。"""
    rng = random.Random(seed)
    angle = rng.uniform(0.0, 2 * math.pi)
    base = rng.uniform(40.0, 120.0)
    swing = rng.uniform(40.0, 90.0)
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            position = x * math.cos(angle) + y * math.sin(angle)
            value = base + swing * (position / (width + height))
            row.append(_wrap(value))
        pixels.append(row)

    for _ in range(shapes):
        kind = rng.choice(("disc", "rect", "ring"))
        shade = rng.uniform(20.0, 235.0)
        radius = rng.uniform(0.06, 0.22) * min(width, height)
        centre_x = rng.uniform(radius, width - radius)
        centre_y = rng.uniform(radius, height - radius)
        _draw(pixels, kind, centre_x, centre_y, radius, shade)
    return Image(width, height, pixels)


def _wrap(value):
    return max(0.0, min(255.0, value))


def _draw(pixels, kind, centre_x, centre_y, radius, shade):
    height = len(pixels)
    width = len(pixels[0])
    top = max(0, int(centre_y - radius - 1))
    bottom = min(height, int(centre_y + radius + 2))
    left = max(0, int(centre_x - radius - 1))
    right = min(width, int(centre_x + radius + 2))
    for y in range(top, bottom):
        row = pixels[y]
        for x in range(left, right):
            distance = math.hypot(x - centre_x, y - centre_y)
            if kind == "disc" and distance <= radius:
                row[x] = shade
            elif kind == "ring" and radius * 0.55 <= distance <= radius:
                row[x] = shade
            elif kind == "rect" and abs(x - centre_x) <= radius * 0.8 \
                    and abs(y - centre_y) <= radius * 0.55:
                row[x] = shade


def stripes(seed=0, width=128, height=128, count=6):
    """一张明暗硬带交替的图 —— 低频哈希理论上处理得最差的情况，因为它就是刻意丢掉
    高频的。
    """
    rng = random.Random(seed)
    low, high = rng.uniform(10, 60), rng.uniform(180, 245)
    band = width / count
    rows = []
    for y in range(height):
        tilt = y * 0.15
        rows.append([high if int((x + tilt) / band) % 2 == 0 else low
                     for x in range(width)])
    return Image(width, height, rows)


def corpus(count=24, seed=0, width=128, height=128):
    """`[(name, Image)]`：若干场景图，再加几张病态图。

    刻意放进来纯色图。空白图的哈希就是在阈值边上抛硬币，任何悄悄把它们排除掉的基准
    都会高估哈希在天空、雪景这类画面上的表现。
    """
    images = [("scene-%02d" % index, scene(seed + index, width, height))
              for index in range(count)]
    images.append(("stripes", stripes(seed, width, height)))
    images.append(("checker", _checker(width, height)))
    images.append(("flat-grey", Image.grey(width, height, 128.0)))
    images.append(("flat-black", Image.grey(width, height, 0.0)))
    return images


def _checker(width, height, squares=8):
    size = max(1, width // squares)
    rows = []
    for y in range(height):
        rows.append([235.0 if (x // size + y // size) % 2 == 0 else 20.0
                     for x in range(width)])
    return Image(width, height, rows)
