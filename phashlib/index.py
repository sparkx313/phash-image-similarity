"""在 64 位哈希上做近重复查找。

把查询和库里每个哈希都比一遍是 O(n)；在几十万件图库上，这正是「哈希很便宜」的意义
所在。下面的分带索引拿内存换掉大部分扫描，并且对它声称覆盖的距离返回与全扫完全相同
的答案。
"""

from .hash import hamming

HASH_BITS = 64


def split_bands(value, bands):
    """把 `value` 切成 `bands` 段，低位那段在前。"""
    width = HASH_BITS // bands
    mask = (1 << width) - 1
    return [(value >> (index * width)) & mask for index in range(bands)]


def linear_search(pairs, query, max_distance):
    """返回库里距离不超过 `max_distance` 的 `[(key, distance)]`。基线：全部看一遍。"""
    return [(key, hamming(query, value))
            for key, value in pairs
            if hamming(query, value) <= max_distance]


class HammingIndex:
    """分带划分索引：把每个哈希切成 `bands` 片并按片建桶，查询只读与自己同片的桶。

    两个哈希距离为 `d` 时，至多有 `d` 个带不同，所以只要 `bands = d + 1`，就至少有一
    个带完全相同，这一对必然落进同一个桶。因此 `max_distance <= bands - 1` 时召回是
    精确的，超过之后变成概率性的 —— `coverage` 把这个保证的边界写出来，而不是留成
    隐含信息。
    """

    def __init__(self, pairs=(), bands=8):
        if HASH_BITS % bands:
            raise ValueError(f"{bands} does not divide the hash into whole bands")
        self.bands = bands
        self.width = HASH_BITS // bands
        self.size = 0
        self.last_examined = 0
        self.tables = [{} for _ in range(bands)]
        for key, value in pairs:
            self.add(key, value)

    @property
    def coverage(self):
        """本索引能保证精确召回的最大 max_distance。"""
        return self.bands - 1

    def add(self, key, value):
        if not 0 <= value < 1 << HASH_BITS:
            raise ValueError(f"{value} does not fit in {HASH_BITS} bits")
        for band, piece in enumerate(split_bands(value, self.bands)):
            self.tables[band].setdefault(piece, []).append((key, value))
        self.size += 1

    def search(self, query, max_distance):
        """返回距离不超过 `max_distance` 的 `[(key, distance)]`；在 coverage 内是精确的。"""
        seen = {}
        for band, piece in enumerate(split_bands(query, self.bands)):
            for key, value in self.tables[band].get(piece, ()):
                if key not in seen:
                    seen[key] = value
        self.last_examined = len(seen)
        found = [(key, hamming(query, value)) for key, value in seen.items()]
        return sorted([pair for pair in found if pair[1] <= max_distance],
                      key=lambda pair: (pair[1], str(pair[0])))

    def buckets(self):
        return [len(table) for table in self.tables]


def _bands_for(max_distance):
    """能仍然保证该距离、且可以整除 64 的最小带数。

    `bands` 必须是能整除 64 的 2 的幂，所以请求距离 2（需要 3 带）会用 4 带来满足 ——
    桶更多，保证不变。
    """
    needed = max_distance + 1
    for bands in (1, 2, 4, 8, 16, 32, 64):
        if bands >= needed:
            return bands
    raise ValueError(f"distance {max_distance} needs more bands than a 64-bit hash "
                     f"has")


def near_duplicates(images, max_distance=5, hash_function=None):
    """把 `[(name, Image)]` 聚成距离不超过 `max_distance` 的簇。

    对索引报出的每一对做并查集，所以分组具有传递性：A~B 且 B~C 时，即使 A~C 超线，
    三者也在同一簇。对去重流程这正是想要的效果，但它也是「一连串渐进编辑会塌成一堆」
    的原因 —— 当某个簇比预期更大时，这就是原因。
    """
    if hash_function is None:
        from .hash import phash as hash_function
    entries = [(name, hash_function(image)) for name, image in images]
    parent = {name: name for name, _ in entries}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    index = HammingIndex(bands=_bands_for(max_distance))
    for name, value in entries:
        index.add(name, value)
    for name, value in entries:
        for other, distance in index.search(value, max_distance):
            if other == name:
                continue
            first, second = find(name), find(other)
            if first != second:
                parent[first] = second
    clusters = {}
    for name, _ in entries:
        clusters.setdefault(find(name), []).append(name)
    return sorted(clusters.values(), key=lambda group: (-len(group), group[0]))
