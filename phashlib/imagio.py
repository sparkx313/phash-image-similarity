"""在没有任何第三方库的情况下，读回 Python 能应付的那几种图像格式。

这个包里的所有东西都工作在 `Image` 上：宽、高，以及每个像素一个 0..255 的浮点灰度值。
彩色图在读入时就按 ITU-R BT.601 权重塌成灰度，因为这里每一个哈希算的都是亮度。

支持范围：PNM（P1-P6，ASCII 与二进制，netpbm 家族）和未压缩 BMP（8 位调色板、24 位、
32 位）。选它们的理由是这两份规范一页纸写得下，而且不装依赖就能在这里生成出来。PNG 和
JPEG 分别需要 DEFLATE 和 DCT 解码器 —— 那属于「解码」，超出了这个关于「哈希」的库的
范围。
"""

import struct

# BT.601 亮度权重 —— JPEG 和 MPEG-1 用的就是这组，也是 2006 年之前每一段「转灰度」
# 代码默认的那组。
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


class Image:
    """一张灰度位图：`pixels[row][col]`，取值为 0..255 的浮点数。"""

    __slots__ = ("width", "height", "pixels")

    def __init__(self, width, height, pixels):
        expected = len(pixels)
        if expected != height:
            raise ValueError(f"image is {height} rows tall, got {expected}")
        for row in pixels:
            if len(row) != width:
                raise ValueError(f"every row must be {width} wide")
        self.width = width
        self.height = height
        self.pixels = pixels

    @classmethod
    def grey(cls, width, height, value=0.0):
        return cls(width, height, [[float(value)] * width for _ in range(height)])

    @classmethod
    def from_rgb(cls, width, height, rgb_rows):
        """`rgb_rows` 是若干行 `(r, g, b)` 三元组；按亮度权重塌成灰度。"""
        kr, kg, kb = LUMA_WEIGHTS
        pixels = [[kr * r + kg * g + kb * b for r, g, b in row] for row in rgb_rows]
        return cls(width, height, pixels)

    def get(self, x, y):
        return self.pixels[y][x]

    def copy(self):
        return Image(self.width, self.height, [list(row) for row in self.pixels])

    def __repr__(self):
        return f"Image({self.width}x{self.height})"


def _next_token(handle):
    """读一个 PNM 头部 token，并把分隔符留在流里。

    PNM 二进制主体是从 maxval 之后那一个空白字节开始的，所以停在 maxval 上的读取器不
    能把它吃掉。回退一格正是为此存在的。
    """
    while True:
        char = handle.read(1)
        if char == b"":
            raise ValueError("truncated PNM header")
        if char in b" \t\r\n":
            continue
        if char == b"#":
            handle.readline()  # comments run to the end of the line
            continue
        token = char
        while True:
            char = handle.read(1)
            if char == b"":
                return token
            if char in b" \t\r\n":
                handle.seek(-1, 1)
                return token
            token += char


class _TokenReader:
    """从 PNM 流里取出头部的各个整数。"""

    def __init__(self, handle):
        self.handle = handle

    def int(self):
        return int(_next_token(self.handle))


def _ascii_body(handle, count):
    """余下以空白分隔的 `count` 个 ASCII 采样值。"""
    values = [int(token) for token in handle.read().split()]
    if len(values) < count:
        raise ValueError("truncated ASCII PNM body")
    return values


def _read_pnm(handle):
    magic = handle.read(2)
    if magic not in (b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"):
        raise ValueError(f"unknown PNM magic {magic!r}")
    kind = magic
    numbers = _TokenReader(handle)
    width = numbers.int()
    height = numbers.int()
    max_value = 1 if kind in (b"P1", b"P4") else numbers.int()
    if width <= 0 or height <= 0:
        raise ValueError(f"empty image {width}x{height}")
    if max_value <= 0 or max_value > 65535:
        raise ValueError(f"unsupported PNM maxval {max_value}")

    if kind in (b"P1", b"P2", b"P3"):
        channels = 1 if kind in (b"P1", b"P2") else 3
        values = _ascii_body(handle, width * height * channels)
        step = width * channels
        pixels = []
        for y in range(height):
            row = values[y * step:(y + 1) * step]
            if kind == b"P1":
                pixels.append([255.0 if value else 0.0 for value in row])
            elif kind == b"P2":
                pixels.append([value * 255.0 / max_value for value in row])
            else:
                triples = list(zip(row[0::3], row[1::3], row[2::3]))
                pixels.append([grey * 255.0 / max_value
                               for grey in _grey_triple(triples)])
        return Image(width, height, pixels)

    byte_width = 1 if max_value < 256 else 2
    handle.read(1)  # 二进制主体之前的那一个空白字节
    stride = (width + 7) // 8 if kind == b"P4" else width * (
        3 if kind == b"P6" else 1) * byte_width
    rows = []
    for _ in range(height):
        block = handle.read(stride)
        if len(block) < stride:
            raise ValueError("truncated PNM pixel data")
        rows.append(_binary_row(kind, block, width, max_value, byte_width))
    return Image(width, height, rows)


def _scale(value, max_value):
    return value * 255.0 / max_value


def _binary_row(kind, block, width, max_value, byte_width):
    def sample(index):
        if byte_width == 1:
            return block[index]
        return struct.unpack_from(">H", block, index * 2)[0]

    if kind == b"P4":
        return [255.0 if block[x // 8] & (1 << (7 - x % 8)) else 0.0
                for x in range(width)]
    if kind == b"P5":
        return [_scale(sample(x), max_value) for x in range(width)]
    triples = [(sample(x * 3), sample(x * 3 + 1), sample(x * 3 + 2))
               for x in range(width)]
    return [_grey_triple([triple])[0] for triple in triples]


def _grey_triple(triples):
    kr, kg, kb = LUMA_WEIGHTS
    return [kr * r + kg * g + kb * b for r, g, b in triples]


def _read_bmp(handle):
    if handle.read(2) != b"BM":
        raise ValueError("not a BMP")
    handle.read(8)
    data_offset = struct.unpack("<I", handle.read(4))[0]
    header = handle.read(4)
    header_size = struct.unpack("<I", header)[0]
    if header_size < 12:
        raise ValueError(f"unsupported BMP DIB size {header_size}")
    if header_size >= 40:
        width, height = struct.unpack("<ii", handle.read(8))
        planes, bpp = struct.unpack("<HH", handle.read(4))
        compression = struct.unpack("<I", handle.read(4))[0]
        handle.read(4)  # biSizeImage：这里的行算式不会和它冲突
        used, _important = struct.unpack("<II", handle.read(8))
    elif header_size == 12:  # BITMAPCOREHEADER：16 位字段、无压缩
        width, height = struct.unpack("<hh", handle.read(4))
        planes, bpp = struct.unpack("<HH", handle.read(4))
        compression = used = 0
    else:
        raise ValueError(f"unsupported BMP DIB header size {header_size}")
    if planes != 1:
        raise ValueError(f"BMP plane count {planes} is not supported")
    if compression not in (0, 3):
        raise ValueError(f"BMP compression type {compression} is not supported")
    if bpp not in (1, 4, 8, 24, 32):
        raise ValueError(f"unsupported BMP bit depth {bpp}")
    top_down = height < 0
    height = abs(height)
    palette = []
    if bpp <= 8:
        entries = used or 2 ** bpp
        handle.seek(data_offset - entries * 4)
        for _ in range(entries):
            blue, green, red, _pad = handle.read(4)
            palette.append((red, green, blue))
    handle.seek(data_offset)
    row_bytes = ((width * bpp + 31) // 32) * 4
    rows = []
    for _ in range(height):
        block = handle.read(row_bytes)
        if len(block) < row_bytes:
            raise ValueError("truncated BMP pixel data")
        rows.append(_bmp_row(block, width, bpp, palette))
    if not top_down:
        rows.reverse()  # 高度为正时，磁盘上的第一行其实是底行
    return Image(width, height, rows)


def _bmp_row(block, width, bpp, palette):
    if bpp == 24:
        triples = [(block[x * 3 + 2], block[x * 3 + 1], block[x * 3])
                   for x in range(width)]
        return _grey_triple(triples)
    if bpp == 32:
        triples = [(block[x * 4 + 2], block[x * 4 + 1], block[x * 4])
                   for x in range(width)]
        return _grey_triple(triples)
    if bpp == 8:
        return [_grey_triple([palette[block[x]]])[0] for x in range(width)]
    if bpp == 4:
        values = [(block[x // 2] >> (4 if x % 2 == 0 else 0)) & 0x0F
                  for x in range(width)]
        return [_grey_triple([palette[v]])[0] for v in values]
    values = [(block[x // 8] >> (7 - x % 8)) & 1 for x in range(width)]
    return [_grey_triple([palette[v]])[0] for v in values]


def load(path):
    """把一个 PNM 或 BMP 文件读成 `Image`。"""
    with open(path, "rb") as handle:
        magic = handle.read(2)
        handle.seek(0)
        if magic[:1] == b"P":
            return _read_pnm(handle)
        if magic == b"BM":
            return _read_bmp(handle)
    raise ValueError(f"{path}: not a PNM or BMP file (starts with {magic!r})")


def loads(data):
    """从字节里读 PNM 或 BMP，供在内存里造图的测试使用。"""
    import io
    return _read_bmp(io.BytesIO(data)) if data[:2] == b"BM" else _read_pnm(
        io.BytesIO(data))


def _clamp(value):
    return max(0, min(255, int(round(value))))


def save_pgm(path, image):
    """写 8 位二进制 PGM —— 把测试图落到磁盘上最省事的方式。"""
    with open(path, "wb") as handle:
        handle.write(b"P5\n%d %d\n255\n" % (image.width, image.height))
        for row in image.pixels:
            handle.write(bytes(_clamp(value) for value in row))


def save_ppm(path, image):
    """把灰度按 P6 写回去，每个像素三个相同通道。"""
    with open(path, "wb") as handle:
        handle.write(b"P6\n%d %d\n255\n" % (image.width, image.height))
        for row in image.pixels:
            handle.write(bytes(_clamp(value) for value in row for _ in range(3)))


def save_bmp(path, image):
    """写未压缩的 24 位 BMP、自底向上，用来喂给另一个读取分支。"""
    row_bytes = ((image.width * 24 + 31) // 32) * 4
    pad = b"\x00" * (row_bytes - image.width * 3)
    pixels = bytearray()
    for row in reversed(image.pixels):
        for value in row:
            grey = _clamp(value)
            pixels += bytes((grey, grey, grey))
        pixels += pad
    size = 54 + len(pixels)
    with open(path, "wb") as handle:
        handle.write(b"BM" + struct.pack("<IHHI", size, 0, 0, 54))
        handle.write(struct.pack("<IiiHHIIiiII", 40, image.width, image.height,
                                 1, 24, 0, 0, 0, 0, 0, 0))
        handle.write(bytes(pixels))
