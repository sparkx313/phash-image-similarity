"""测量 phashlib 的哈希、索引与解码器行为。

运行方式：`python benchmarks/run_benchmarks.py`。打印出来的每个数字都来自本次运行，
并保存到 `benchmarks/last_run.txt`，README 引用的就是这份文件。语料由固定随机种子
生成，因此距离表在别的机器上可以完全复现；耗时和内存则不能。

各节依次回答：每张编辑图会让哈希偏移多少、去重阈值能落在哪里、中位数阈值是否优于
均值阈值、64 个比特里哪些真正携带信息、分带索引在跳过扫描的同时是否仍然精确、单张
图像的开销分布，以及按哈希聚类会得到什么。
"""

import os
import random
import statistics
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phashlib import (HammingIndex, ahash, blur, brightness, centre_crop,
                      contrast, corpus, crop, dhash, flip_horizontal, gamma,
                      hamming, invert, near_duplicates, noise, phash, popcount,
                      resize, rotate_left, save_bmp, save_pgm, shift)
from phashlib.dct import cos_factors, low_frequency
from phashlib.imagio import load
from phashlib.pixels import clamp8

HASH_NAMES = ("phash", "dhash", "ahash")
HASHES_BY_NAME = {"phash": phash, "dhash": dhash, "ahash": ahash}

# 同一张图片的第二份拷贝可能经历过的编辑，按“哈希声称能扛住”和“声称不能”分组。
QUALITY = {
    "rescale down 0.75": lambda im: resize(im, int(im.width * 0.75), int(im.height * 0.75)),
    "rescale up 1.4": lambda im: resize(im, int(im.width * 1.4), int(im.height * 1.4)),
    "brightness +30": lambda im: brightness(im, 1.0, 30.0),
    "exposure x1.3": lambda im: brightness(im, 1.3),
    "gamma 0.8": lambda im: gamma(im, 0.8),
    "contrast 1.25": lambda im: contrast(im, 1.25),
    "noise +-12": lambda im: noise(im, 12.0, seed=1),
    "blur r=1": lambda im: blur(im, 1),
    "quantise to 8 bit": lambda im: clamp8(im),
}
FRAMING = {
    "centre crop 85%": lambda im: centre_crop(im, 0.85),
    "edge crop 10%": lambda im: crop(im, 6, 6, 6, 6),
    "shift 6px": lambda im: shift(im, 6, 3),
}
# 属于去重范围：拷贝还是同一张图，只是重新编码过。
DUPLICATE_EDITS = dict(QUALITY, **FRAMING)
# 不属于去重范围。这三类仍然列在表里，因为感知哈希经常被说成对它们不变，
# 而实测数字并不是这样。
NOT_DUPLICATES = {"invert": invert, "flip horizontal": flip_horizontal,
                  "rotate 90": rotate_left}

DUPLICATE_THRESHOLD = 5
QUERIES = 50


def corpus_split(count=24):
    """全部帧、普通场景，以及那几张专门用来添乱的帧。"""
    frames = corpus(count)
    scenes = [(name, image) for name, image in frames if name.startswith("scene")]
    awkward = [(name, image) for name, image in frames if not name.startswith("scene")]
    return frames, scenes, awkward


def _cell(values):
    return "%5.1f %4d %4d" % (statistics.median(values), min(values), max(values))


def _distances(scenes, edits, hash_name):
    function = HASHES_BY_NAME[hash_name]
    base = [function(image) for _, image in scenes]
    return {label: [hamming(value, function(edit(image)))
                    for value, (_, image) in zip(base, scenes)]
            for label, edit in edits.items()}


def _inside(values, limit=DUPLICATE_THRESHOLD):
    return 100.0 * sum(1 for distance in values if distance <= limit) / len(values)


def section_transforms(scenes, awkward):
    _line("\n1. how far does the hash move under each edit?")
    print("  one row per edit, applied to each of %d scenes. The distance is bits" % len(scenes))
    print("  that differ from the unedited original; med/min/max are over scenes.\n")
    rows = {hash_name: _distances(scenes, dict(QUALITY, **FRAMING, **NOT_DUPLICATES),
                                  hash_name) for hash_name in HASH_NAMES}
    print("  %-19s %-14s %-14s %-14s" % ("edit", "phash", "dhash", "ahash"))
    print("  %-19s %-14s %-14s %-14s" % ("", "med min max", "med min max", "med min max"))
    for label, group in [("quality", QUALITY), ("framing", FRAMING),
                         ("other picture", NOT_DUPLICATES)]:
        print("  %s" % label)
        for name in group:
            print("  %-19s %-14s %-14s %-14s" %
                  (name, _cell(rows["phash"][name]), _cell(rows["dhash"][name]),
                   _cell(rows["ahash"][name])))
    print("\n  share of edit pairs that stay inside %d bits:" % DUPLICATE_THRESHOLD)
    print("  %-14s %8s %8s %8s" % ("class", "phash", "dhash", "ahash"))
    for label, edits in (("quality", QUALITY), ("framing", FRAMING),
                         ("quality+framing", DUPLICATE_EDITS),
                         ("invert/flip/rotate", NOT_DUPLICATES)):
        print("  %-14s %8.1f %8.1f %8.1f" %
              (label, *(_inside([d for name in edits for d in rows[hash_name][name]])
                        for hash_name in HASH_NAMES)))
    quality_shares = {name: _inside([d for label in QUALITY for d in rows[name][label]])
                      for name in HASH_NAMES}
    framing_shares = {name: _inside([d for label in FRAMING for d in rows[name][label]])
                      for name in HASH_NAMES}
    tone = {name: statistics.median([d for label in ("brightness +30",)
                                     for d in rows[name][label]]) for name in HASH_NAMES}
    # 这两行整图缩放，所以只有「阈值就是均值」的那家会动。
    scaled = {name: statistics.median([d for label in ("exposure x1.3", "gamma 0.8")
                                       for d in rows[name][label]]) for name in HASH_NAMES}
    weakest = min(quality_shares, key=lambda name: quality_shares[name])
    print(f"  on tone all three are good: the weakest quality share is "
          f"{quality_shares[weakest]:.1f}% ({weakest}), and the row that separates the")
    print(f"  three by construction is ahash, whose threshold is the image mean - where")
    print(f"  the edit scales the whole image it moves a median of {scaled['ahash']:.1f}")
    print(f"  bits against {scaled['phash']:.1f} for phash and {scaled['dhash']:.1f} for")
    print(f"  dhash; a flat brightness offset leaves all three at {tone['ahash']:.1f}.")
    print(f"  On framing the ranking flips: phash reads the")
    print(f"  low-frequency spectrum of the whole frame, so cutting content out of it is")
    print(f"  not a small change - {framing_shares['phash']:.1f}% of framing pairs stay")
    print(f"  inside the threshold against {framing_shares['dhash']:.1f}% for dhash and")
    print(f"  {framing_shares['ahash']:.1f}% for ahash. A difference hash only sees the")
    print("  edge it was taken across and an average hash only sees the balance of the")
    print("  thumbnail, which is why they shrug the crops off better.")
    print(f"  Nothing here is invariant to geometry: invert sits "
          f"{statistics.median(rows['phash']['invert']):.0f} bits away at the median,")
    print("  because a negative is the opposite picture to a hash that reads")
    print("  above/below. Flip and rotate are the same story - the spectrum is indexed")
    print("  by position, so moving the content moves the hash.")
    print("\n  awkward frames, phash distance from the frame to its own edits:")
    print("  %-12s %7s %7s %7s %7s" % ("frame", "as-is", "gamma", "noise", "invert"))
    for name, image in awkward:
        values = [hamming(phash(image), phash(edit(image)))
                  for edit in (lambda im: im, QUALITY["gamma 0.8"],
                               QUALITY["noise +-12"], invert)]
        print("  %-12s %7d %7d %7d %7d" % (name, *values))
    print("  'as-is' is a frame against itself, so anything above 0 there would be a")
    print("  bug. On a flat field every pixel sits exactly on the threshold, so which")
    print("  side of it each of the 64 numbers falls on is decided by float rounding,")
    print("  not by the picture: 32 bits of motion out of a +-12 noise pass on a blank")
    print("  frame means the hash of a blank image is a coin flip that happens to be")
    print("  stable until you touch it. These frames are in the corpus so the tables")
    print("  above say so instead of averaging them away.")


def _pairs(frames, hash_name):
    function = HASHES_BY_NAME[hash_name]
    return [(name, function(image)) for name, image in frames]


def section_threshold(scenes):
    _line("\n2. where can the duplicate threshold sit?")
    print("  related   = a scene and one of its %d re-encode edits (%d pairs);" %
          (len(DUPLICATE_EDITS), len(scenes) * len(DUPLICATE_EDITS)))
    print("            invert, flip and rotate are left out - they are not")
    print("            duplicates and section 1 shows why")
    print("  unrelated = %d pairs of two different scenes" %
          (len(scenes) * (len(scenes) - 1) // 2))
    print("  false positive  = an unrelated pair inside the threshold (a wrong merge)")
    print("  false negative  = a related pair outside it (a duplicate that was missed)\n")
    summary = {}
    for hash_name in HASH_NAMES:
        hashes = dict(_pairs(scenes, hash_name))
        related = [hamming(hashes[name], HASHES_BY_NAME[hash_name](edit(image)))
                   for name, image in scenes for edit in DUPLICATE_EDITS.values()]
        quality = [hamming(hashes[name], HASHES_BY_NAME[hash_name](edit(image)))
                   for name, image in scenes for edit in QUALITY.values()]
        framing = [hamming(hashes[name], HASHES_BY_NAME[hash_name](edit(image)))
                   for name, image in scenes for edit in FRAMING.values()]
        items = list(hashes.items())
        unrelated = [(first, second, hamming(a, b))
                     for index, (first, a) in enumerate(items)
                     for second, b in items[index + 1:]]
        closest = min(unrelated, key=lambda row: row[2])
        print("  %s" % hash_name)
        print("    threshold   FP rate   FN rate   quality FN  framing FN")
        for limit in (0, 2, 4, 5, 6, 8, 10, 12, 14):
            fp = sum(1 for _, _, d in unrelated if d <= limit) / len(unrelated)
            print("    %9d   %7.3f   %7.3f   %9.3f %10.3f" %
                  (limit, fp,
                   sum(1 for d in related if d > limit) / len(related),
                   sum(1 for d in quality if d > limit) / len(quality),
                   sum(1 for d in framing if d > limit) / len(framing)))
        print("    closest unrelated pair: %s / %s at %d bits apart; related median"
              % (closest[0], closest[1], closest[2]))
        print("    %.1f, unrelated median %.1f.\n"
              % (statistics.median(related),
                 statistics.median([d for _, _, d in unrelated])))
        summary[hash_name] = (
            sum(1 for _, _, d in unrelated if d <= DUPLICATE_THRESHOLD) / len(unrelated),
            sum(1 for d in quality if d > DUPLICATE_THRESHOLD) / len(quality),
            sum(1 for d in framing if d > DUPLICATE_THRESHOLD) / len(framing),
            closest[2])
    print("  at the threshold the literature uses (%d bits):" % DUPLICATE_THRESHOLD)
    print("  %-8s %14s %14s %14s %16s" %
          ("hash", "wrong merges", "quality misses", "framing misses", "closest strangers"))
    for hash_name, (fp, quality_fn, framing_fn, gap) in summary.items():
        print("  %-8s %13.1f%% %13.1f%% %13.1f%% %15d bits" %
              (hash_name, 100 * fp, 100 * quality_fn, 100 * framing_fn, gap))
    merging = [("%s at %.1f%%" % (name, 100 * stats[0]))
               for name, stats in summary.items() if stats[0]]
    print("  at %d bits %s merge different scenes together; %s do not. Every one of them"
          % (DUPLICATE_THRESHOLD,
             " and ".join(merging) if merging else "no hash",
             " / ".join(name for name, stats in summary.items() if not stats[0])))
    print("  loses most of the framing edits - a crop that changes what is in the")
    print("  frame is a different picture as far as these hashes are concerned, which")
    print("  is a smaller promise than 'find me the near-duplicates' and the one this")
    print("  README keeps. The FP column is also the floor of a synthetic corpus:")
    print("  smooth generated scenes are far apart (medians above 30 bits) where real")
    print("  photo sets sit much closer, so treat these as the shape of the trade-off")
    print("  and recalibrate on your own pictures before picking a number.")


def section_cut(scenes, awkward):
    _line("\n3. median cut versus mean cut for phash")
    print("  HashLib thresholds against the median of the 63 non-DC coefficients;")
    print("  most Python ports use the mean. Same scenes, both cuts:\n")
    medians = [phash(image, threshold="median") for _, image in scenes]
    means = [phash(image, threshold="mean") for _, image in scenes]
    print("  %-19s %10s %10s" % ("edit", "median cut", "mean cut"))
    for label, edits in (("quality", QUALITY), ("framing", FRAMING),
                         ("other", NOT_DUPLICATES)):
        for name, edit in edits.items():
            median_rows = [hamming(value, phash(edit(image), threshold="median"))
                           for value, (_, image) in zip(medians, scenes)]
            mean_rows = [hamming(value, phash(edit(image), threshold="mean"))
                         for value, (_, image) in zip(means, scenes)]
            print("  %-19s %10.1f %10.1f" % (name, statistics.median(median_rows),
                                             statistics.median(mean_rows)))
    flips = sorted(popcount(a ^ b) for a, b in zip(medians, means))
    identical = sum(1 for a, b in zip(medians, means) if a == b)
    apart = [value for value in flips if value]
    print(f"\n  the cuts give the identical hash on {identical} of {len(medians)} scenes;")
    print(f"  where they differ the gap runs from {min(apart)} to {max(apart)} of the 64 bits.")
    print("  A mean is pulled by the loudest coefficient where a median is not, so the")
    print("  frames that should split them are the ones with one loud term in the")
    print("  block - a flat field is exactly that, and the rows below check it.\n")
    print("  %-14s %12s %12s" % ("frame", "cuts apart", "noise moves"))
    for label, image in scenes[:2] + awkward:
        gap = popcount(phash(image, threshold="median") ^ phash(image, threshold="mean"))
        moved = hamming(phash(image, threshold="median"),
                        phash(QUALITY["noise +-12"](image), threshold="median"))
        print("  %-14s %12d %12d" % (label, gap, moved))
    print("  'cuts apart' is the two hashes of the same frame; 'noise moves' is one")
    print("  cut against itself after a +-12 noise pass, so the first column is a")
    print("  choice of algorithm and the second is a choice of picture.\n")
    inside = lambda values: sum(1 for d in values if d <= DUPLICATE_THRESHOLD)
    med_related = [hamming(value, phash(edit(image), threshold="median"))
                   for value, (_, image) in zip(medians, scenes)
                   for edit in DUPLICATE_EDITS.values()]
    mean_related = [hamming(value, phash(edit(image), threshold="mean"))
                    for value, (_, image) in zip(means, scenes)
                    for edit in DUPLICATE_EDITS.values()]
    print("  related pairs inside %d bits: median cut %d/%d, mean cut %d/%d."
          % (DUPLICATE_THRESHOLD, inside(med_related), len(med_related),
             inside(mean_related), len(mean_related)))
    print("  The cuts are not interchangeable - they disagree on the borderline pairs")
    print("  and swap a few of them each way - but on smooth synthetic scenes neither")
    print("  is a large win. Pick the median cut because a blank or near-blank frame")
    print("  has a spectrum with one loud term, where a mean moves the whole cut and a")
    print("  median does not: the 'cuts apart' column above is that effect measured,")
    print("  0-3 bits on the scenes and 15-25 on the frames with no content in them.")


def section_bits(frames):
    _line("\n4. which of the 64 bits carry anything?")
    hashes = [phash(image) for _, image in frames]
    per_bit = [sum(1 for value in hashes if (value >> index) & 1) for index in range(64)]
    set_zero = [index for index, count in enumerate(per_bit) if count == 0]
    set_all = [index for index, count in enumerate(per_bit) if count == len(hashes)]
    print("  %d hashes from the corpus." % len(hashes))
    print("  bit 0 is the DC coefficient compared against a cut that never includes")
    print("  DC, so it is set for any picture that is not pure black:")
    print("  bit 0 set on %d/%d frames; never set anywhere: %s; always set: %s"
          % (per_bit[0], len(hashes), set_zero or "none", set_all or "none"))
    varying = 64 - len(set_zero) - len(set_all)
    print("  bits that vary somewhere in the corpus: %d of 64." % varying)
    dark = [label for (label, _), value in zip(frames, hashes) if not value & 1]
    print("  the frames that leave bit 0 clear: %s." % (", ".join(dark) or "none"))
    print("  A lopsided bit is not free but it is cheap: it adds 0 to almost every")
    print("  distance and 1 to the pair that straddles it. The library's '63 usable")
    print("  bits' should be read the same way the numbers above do: bit 0 is set on")
    print("  %d of %d frames, so it carries a fraction of a bit rather than none and"
          % (per_bit[0], len(hashes)))
    print("  not quite a whole one.")
    print("  fraction of frames with each bit set, 16 bits per row:")
    for band in range(4):
        row = per_bit[band * 16:band * 16 + 16]
        print("  %2d-%2d  %s" % (band * 16, band * 16 + 15,
                                 " ".join("%4.2f" % (count / len(hashes)) for count in row)))
    balanced = sum(1 for count in per_bit if 0.2 <= count / len(hashes) <= 0.8)
    print("  %d of 64 bits land in the 0.2-0.8 band that actually separates images;" % balanced)
    print("  the rest are almost always or almost never set and only ever move a")
    print("  distance by a bit or two at the edges of the corpus.\n")
    labels = [label for label, _ in frames]
    distances = [(hamming(first, second), label_a, label_b)
                 for index, (label_a, first) in enumerate(zip(labels, hashes))
                 for label_b, second in zip(labels[index + 1:], hashes[index + 1:])]
    ordered = sorted(distances)
    values = [row[0] for row in distances]
    print("  distance over all %d pairs of the whole corpus: min %d, median %.1f,"
          % (len(distances), ordered[0][0], statistics.median(values)))
    print("  max %d, mean %.1f of a 64-bit hash; independent bits would average 32."
          % (ordered[-1][0], statistics.mean(values)))
    print("  the closest two frames are %s and %s at %d bits: below that number the"
          % (ordered[0][1], ordered[0][2], ordered[0][0]))
    print("  hash calls two different pictures duplicates. It is two ordinary scenes,")
    print("  not a blank one, so the assumption that only the pathological frames are")
    print("  risky is wrong on this corpus.")
    blanks = [value for (label, _), value in zip(frames, hashes) if label.startswith("flat")]
    if len(blanks) > 1:
        print("  the two blank frames are %d bits apart from each other - a grey and a"
              % hamming(blanks[0], blanks[1]))
        print("  black field are not close duplicates, which is what section 1's coin")
        print("  flip predicts.")


def random_store(size, seed=7):
    """`size` 个各比特独立为 1 的哈希。

    索引只看 64 位整数、不看像素，所以吞吐表的载入数据是生成的而不是哈希出来的。
    第 4 节说明真实 phash 的比特并非完全独立，这让这里的载入比生产环境略难一些，
    而不是更容易。
    """
    rng = random.Random(seed)
    return [("h%05d" % index, rng.getrandbits(64)) for index in range(size)]


def perturb(value, distance, rng):
    """把 `value` 恰好翻掉 `distance` 个比特 —— 距离已知的查询。"""
    out = value
    for bit in rng.sample(range(64), distance):
        out ^= 1 << bit
    return out


def sweep(pairs, query, max_distance):
    """诚实的基线：每个入库哈希只算一次汉明距离，不重复计算。"""
    found = []
    for key, value in pairs:
        distance = hamming(query, value)
        if distance <= max_distance:
            found.append((key, distance))
    return found


def section_index():
    _line("\n5. does the band index stay exact while skipping the scan?")
    print("  queries are stored hashes with a known number of bits flipped, so the")
    print("  exact answer is known and recall is a measurement, not a judgement.\n")
    print("  %7s %6s %6s %10s %9s %8s %10s %9s %s" %
          ("store", "d max", "bands", "guarantee", "recall", "cands", "index s", "scan s", "vs scan"))
    measured = {}
    for size, max_distance, bands in ((1000, 4, 8), (1000, 6, 8), (5000, 6, 8),
                                      (20000, 6, 8), (20000, 8, 16), (20000, 10, 8)):
        store = random_store(size)
        index = HammingIndex(store, bands=bands)
        rng = random.Random(size * 10 + max_distance)
        queries = [perturb(store[rng.randrange(size)][1], max_distance, rng)
                   for _ in range(QUERIES)]
        found = missing = 0
        for query in queries:
            got = {key for key, _ in index.search(query, max_distance)}
            exact = {key for key, _ in sweep(store, query, max_distance)}
            found += len(got & exact)
            missing += len(exact - got)
        examined = 0
        started = time.perf_counter()
        for query in queries:
            index.search(query, max_distance)
            examined += index.last_examined
        index_time = time.perf_counter() - started
        started = time.perf_counter()
        for query in queries:
            sweep(store, query, max_distance)
        scan_time = time.perf_counter() - started
        print("  %7d %6d %6d %10s %9s %7.1f%% %10.2f %9.2f %.1fx %s" %
              (size, max_distance, bands, "d<=%d" % index.coverage,
               "%d/%d" % (found, found + missing),
               100.0 * examined / (QUERIES * size), index_time, scan_time,
               scan_time / index_time,
               "exact" if max_distance <= index.coverage else "OVER THE GUARANTEE"))
        measured[(size, max_distance)] = (
            scan_time / index_time, 100.0 * examined / (QUERIES * size), missing)
    wide_ratio, wide_cands, _ = measured[(20000, 8)]
    edge_missing = measured[(20000, 10)][2]
    print("\n  recall is 100% wherever max_distance <= bands - 1, which is the")
    print("  pigeonhole guarantee rather than a tuned result: two hashes d bits apart")
    print("  agree in at least one of d + 1 bands, so they share a bucket and the")
    print("  index has to look at it. The row that asks for distance 10 from an")
    print("  8-band index (coverage 7) loses %d match%s - that is the edge, and"
          % (edge_missing, "" if edge_missing == 1 else "es"))
    print("  `coverage` is the field that names it instead of leaving it to a")
    print("  production incident.\n")
    print("  'cands' is the share of the store whose hashes get compared at all, and")
    print("  it is the whole point: it stays flat while the scan it replaces grows.")
    print("  The 16-band row pays the other half of the price - bands are forced to")
    print("  divide 64, so covering distance 8 means 16 bands of 4 bits each, and a")
    print("  4-bit bucket is 1/16 of the store. 16 buckets later the index is comparing")
    print("  %.1f%% of the store and lands at %.1fx the scan, so at that distance the"
          % (wide_cands, wide_ratio))
    print("  index buys nothing - which is what the row says. Past a handful of bits")
    print("  apart, a linear sweep of 64-bit integers is the honest answer.\n")
    store = random_store(20000)
    started = time.perf_counter()
    index = HammingIndex(store, bands=8)
    built = time.perf_counter() - started
    list_peak = _peak_kb(lambda: [(key, value) for key, value in store])
    index_peak = _peak_kb(lambda: HammingIndex(store, bands=8))
    print("  building the 20000-hash index took %.2f s across %d buckets; peak memory" %
          (built, sum(index.buckets())))
    print("  during construction was %.0f KB for a plain list of the pairs and %.0f KB" %
          (list_peak, index_peak))
    print("  for the index: every entry is referenced once per band, so an 8-band")
    print("  index costs about 8x the pointers of the list it replaces, and the")
    print("  buckets themselves are %.0f KB per band on average." % (index_peak / 8))


def _peak_kb(function):
    tracemalloc.start()
    try:
        function()
        return tracemalloc.get_traced_memory()[1] / 1024
    finally:
        tracemalloc.stop()


def write_ascii_pgm(text, path):
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("P2\n%d %d\n255\n" % (text.width, text.height))
        handle.write("\n".join(" ".join("%d" % int(round(v)) for v in row)
                               for row in text.pixels))
        handle.write("\n")


def _timed(function, *args):
    started = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter() - started


def section_cost(frames):
    _line("\n6. what does one image cost?")
    name, image = frames[0]
    with tempfile.TemporaryDirectory() as folder:
        paths = {}
        save_pgm(os.path.join(folder, "frame.pgm"), image)
        paths["PGM P5 binary"] = os.path.join(folder, "frame.pgm")
        save_bmp(os.path.join(folder, "frame.bmp"), image)
        paths["BMP 24 bit"] = os.path.join(folder, "frame.bmp")
        write_ascii_pgm(image, os.path.join(folder, "frame-ascii.pgm"))
        paths["PGM P2 ascii"] = os.path.join(folder, "frame-ascii.pgm")

        print("  the stages of phash, one row per file the decoder has to read.\n")
        print("  %-16s %9s %8s %11s %10s %9s %8s" %
              ("source", "decode", "bytes", "resize 8x8", "resize 32", "DCT 8x8", "phash"))
        for label, path in paths.items():
            decoded, decode_time = _timed(load, path)
            _, small_time = _timed(resize, decoded, 8, 8)
            block, block_time = _timed(resize, decoded, 32, 32)
            _, dct_time = _timed(low_frequency, block.pixels, 8)
            _, phash_time = _timed(phash, decoded)
            print("  %-16s %9.4f %8d %11.4f %10.4f %9.4f %8.4f" %
                  (label, decode_time, os.path.getsize(path), small_time, block_time,
                   dct_time, phash_time))
            assert hamming(phash(decoded), phash(image)) == 0, label
        print("\n  all three files hash to the same integer as the in-memory frame")
        print("  (asserted, not assumed), so the decode step is the only place the")
        print("  formats differ: ASCII PNM is the slowest read and the largest file,")
        print("  and the cost after it is identical.\n")

    print("  per-image stages on a 128x128 frame (median of 40 runs each):")
    print("  %-23s %8s %8s" % ("stage", "ms", "share"))
    block = resize(image, 32, 32)
    factors = _median_time(lambda: cos_factors(32, 8))
    resized = _median_time(lambda: resize(image, 32, 32))
    dct = _median_time(lambda: low_frequency(block.pixels, 8))
    whole = _median_time(lambda: phash(image))
    residual = whole - resized - dct
    rows = [("resize to 32x32", resized),
            ("DCT to 8x8", dct),
            ("  of which factors", factors),
            ("threshold, pack, residual", residual),
            ("phash, end to end", whole)]
    for label, taken in rows:
        print("  %-23s %8.2f %7.0f%%" % (label, 1000 * taken, 100.0 * taken / whole))
    print("  the last row is the whole call and the rows above it are timed apart, so")
    print("  'threshold, pack, residual' is what is left once the two measured stages are")
    print("  subtracted. A negative there is timing noise between separately-timed stages")
    print("  and it means the two measured stages already account for the whole call -")
    print("  thresholding and packing 64 numbers is not where the time goes.")
    print("  The honest statement is the first row: the area-average resample is the hash.")
    print("  It reads every")
    print("  pixel of the source, while the DCT only ever touches 8 frequencies per")
    print("  axis. The cosine table is rebuilt on every call and costs %.1f%%, so"
          % (100.0 * factors / whole))
    print("  caching it would not move this number.\n")

    print("  whole corpus (%d frames), best of three passes:" % len(frames))
    print("  %-8s %10s %10s %10s" % ("hash", "total s", "ms/image", "peak KB"))
    per_image = {}
    for hash_name in HASH_NAMES:
        function = HASHES_BY_NAME[hash_name]
        elapsed = min(_once(function, frames) for _ in range(3))
        per_image[hash_name] = 1000 * elapsed / len(frames)
        print("  %-8s %10.2f %10.1f %10.0f" %
              (hash_name, elapsed, per_image[hash_name],
               _peak_kb(lambda: [function(f) for _, f in frames])))
    fastest = min(per_image, key=per_image.get)
    slowest = max(per_image, key=per_image.get)
    print("  %s is the cheapest at %.1f ms a frame, %s the dearest at %.1f: ahash"
          % (fastest, per_image[fastest], slowest, per_image[slowest]))
    print("  reads an 8x8 thumbnail, dhash a 9x8 one, and phash a 32x32 block plus a")
    print("  transform. On the quality rows of section 1 all three are within a")
    print("  couple of bits of each other, so the choice between them is about")
    print("  framing behaviour and about what a false positive costs, not about speed.")


def _median_time(function, runs=40):
    return statistics.median([_timed(function)[1] for _ in range(runs)])


def _once(function, frames):
    started = time.perf_counter()
    for _, frame in frames:
        function(frame)
    return time.perf_counter() - started


COPIES = {"orig": lambda image, index: image,
          "small": lambda image, index: resize(image, 96, 96),
          "noisy": lambda image, index: noise(image, 12.0, seed=index),
          "cropped": lambda image, index: centre_crop(image, 0.8)}
QUALITY_COPIES = ("orig", "small", "noisy")


def _cluster_report(images, limit):
    clusters = near_duplicates(images, max_distance=limit)
    clean = cross = crops_in = crops_alone = 0
    for cluster in clusters:
        groups = {name.split("/")[0] for name in cluster}
        members = {name.split("/")[1] for name in cluster}
        if len(groups) > 1:
            cross += 1
        elif members >= set(QUALITY_COPIES):
            clean += 1
        if "cropped" in members:
            if len(cluster) > 1:
                crops_in += 1
            else:
                crops_alone += 1
    return clusters, clean, cross, crops_in, crops_alone


def section_clusters(scenes, awkward):
    _line("\n7. what grouping the hashes produces")
    images = []
    for index, (name, frame) in enumerate(scenes):
        for label, copy in COPIES.items():
            images.append(("%s/%s" % (name, label), copy(frame, index)))
    for name, frame in awkward:
        images.append(("%s/orig" % name, frame))
    print("  %d images: %d scenes each kept, rescaled, noised and centre-cropped, plus"
          % (len(images), len(scenes)))
    print("  the %d awkward frames. Grouped by phash with near_duplicates().\n" % len(awkward))
    clusters, clean, cross, crops, crops_alone = _cluster_report(images, DUPLICATE_THRESHOLD)
    print("  at max_distance %d: %d clusters, %d groups whose three quality copies"
          % (DUPLICATE_THRESHOLD, len(clusters), clean))
    print("  merged completely, %d crops that joined their own group, %d crops left on"
          % (crops, crops_alone))
    print("  their own, %d clusters that span two different pictures.\n" % cross)
    for cluster in clusters[:3]:
        print("    %s" % ", ".join(sorted(cluster)))
    stray = [c for c in clusters if len(c) == 1 and c[0].endswith("/cropped")]
    print("    ...")
    if stray:
        print("    left on its own: %s" % ", ".join(name for c in stray[:3] for name in c))
    print("\n  `near_duplicates` is union-find, so grouping is transitive: A~B and")
    print("  B~C put all three together even when A~C is over the line. That is what")
    print("  a de-duplication pass wants, and the reason a chain of gradual edits")
    print("  reads as one group when the pairwise distances do not suggest it.\n")
    print("  %12s %9s %12s %9s %11s %11s" %
          ("max_distance", "clusters", "clean groups", "crops in", "crops alone",
           "over-merges"))
    for limit in (3, DUPLICATE_THRESHOLD, 7, 12, 20):
        other, clean, cross, crops, crops_alone = _cluster_report(images, limit)
        print("  %12d %9d %12d %9d %11d %11d" %
              (limit, len(other), clean, crops, crops_alone, cross))
    first_crop = first_all = None
    for limit in range(1, 17):
        _, _, _, joined, _ = _cluster_report(images, limit)
        if joined and first_crop is None:
            first_crop = limit
        if joined == len(scenes) and first_all is None:
            first_all = limit
    strangers = min(hamming(phash(a), phash(b))
                    for index, (_, a) in enumerate(scenes)
                    for _, b in scenes[index + 1:])
    print("  the crop columns are the ones to read against section 2: the first crop")
    print("  joins its own group at max_distance %d and all %d of them do by %d, while"
          % (first_crop, len(scenes), first_all))
    print("  the two closest different scenes in this corpus are %d bits apart. So a" % strangers)
    print("  threshold of %d buys crops at the cost of %d%% of the distance budget"
          % (first_all, round(100.0 * first_all / strangers)))
    print("  before the first wrong merge - and that budget is measured on smooth")
    print("  synthetic scenes, where strangers are far apart. On a real photo set they")
    print("  sit closer, so the crop either stays a separate picture or you accept")
    print("  merges you cannot bound. That is the trade the tables above are for.")


def _line(title, char="-"):
    print(title)
    print(char * len(title))


class _Tee:
    """stdout 照常打印，同时攒一份写到文件里。

    这样 `last_run.txt` 永远是刚刚那一次运行的原文，README 里的数字不可能跟文件对不上。
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.path = path
        self.parts = []

    def write(self, text):
        self.stream.write(text)
        self.parts.append(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def save(self):
        with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(self.parts))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_run.txt")
    console = sys.stdout
    sys.stdout = _Tee(console, report)
    try:
        print("phashlib benchmarks - python %s" % sys.version.split()[0])
        frames, scenes, awkward = corpus_split()
        section_transforms(scenes, awkward)
        section_threshold(scenes)
        section_cut(scenes, awkward)
        section_bits(frames)
        section_index()
        section_cost(frames)
        section_clusters(scenes, awkward)
    finally:
        sys.stdout.save()
        sys.stdout = console
        print("wrote %s" % report)


if __name__ == "__main__":
    main()
