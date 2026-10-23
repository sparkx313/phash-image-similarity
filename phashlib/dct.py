"""二维 DCT-II，但只算感知哈希需要的低频那一角。

对 32x32 的块做完整正变换，每个轴要 32^3 次乘加；而实际只读左上角 8x8，所以这里
每个轴只把 `keep` 个频率对全部 `n` 个采样点做投影 —— 大约便宜八倍，结果与完整变换
左上角那块逐位相同。

不依赖矩阵库、不依赖 numpy：两层循环配一张预先算好的余弦表。
"""

import math


def cos_factors(n, keep):
    """`factors[u][x] = cos(pi * (2x + 1) * u / (2n))`，其中 `u < keep`。"""
    denominator = 2.0 * n
    return [[math.cos(math.pi * (2 * x + 1) * u / denominator)
             for x in range(n)] for u in range(keep)]


def low_frequency(grid, keep):
    """方阵 `grid` 的 `keep` x `keep` 低频 DCT-II 块。

    `grid` 是浮点数行表，且 `len(grid) == len(grid[0]) == n`。结果按
    `[frequency_y][frequency_x]` 索引，`[0][0]` 是直流项 —— 也就是块的均值，
    这正是感知哈希要把它丢掉的原因。
    """
    n = len(grid)
    if n != len(grid[0]):
        raise ValueError("low_frequency needs a square block")
    if keep > n:
        raise ValueError(f"cannot keep {keep} frequencies from {n} samples")
    factors = cos_factors(n, keep)
    # 横向一遍：每一行源数据被压缩到 `keep` 个频率上。
    stage = [[sum(value * factor
                   for value, factor in zip(row, row_factors))
              for row_factors in factors]
             for row in grid]
    # 纵向一遍：此时手里仍有 n 个采样点，但只有 `keep` 个会留下。
    return [[sum(stage[y][x] * column_factors[y] for y in range(n))
             for x in range(keep)]
            for column_factors in factors]
