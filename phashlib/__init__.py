"""只用标准库实现的图像感知哈希。

`phash`、`dhash`、`ahash` 把一张图变成一个 64 位整数；`hamming` 数出相差的位数；
`HammingIndex` 不用把所有图比一遍就能找到近重复。用 `load` 读 PNM 和 BMP，用
`scenes.scene` 生成测试图，用 `pixels` 模拟那些「会破坏校验和、但不该破坏感知哈希」
的编辑。
"""

from .hash import (HASHES, ahash, dhash, from_hex, hamming, phash, popcount,
                   to_hex)
from .imagio import Image, load, loads, save_bmp, save_pgm, save_ppm
from .index import HammingIndex, linear_search, near_duplicates, split_bands
from .pixels import (blur, brightness, centre_crop, contrast, crop, flip_horizontal,
                     gamma, invert, noise, resize, rotate_left, shift)
from .scenes import corpus, scene, stripes

__version__ = "0.1.0"

__all__ = ["HASHES", "HammingIndex", "Image", "ahash", "blur", "brightness",
           "centre_crop", "contrast", "corpus", "crop", "dhash", "flip_horizontal",
           "from_hex", "gamma", "hamming", "invert", "linear_search", "load",
           "loads", "near_duplicates", "noise", "phash", "popcount",
           "resize", "rotate_left", "save_bmp", "save_pgm", "save_ppm", "scene", "shift", "split_bands",
           "stripes", "to_hex", "__version__"]
