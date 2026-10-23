"""测试哈希、索引、像素运算以及两个图像读取器。

这里每一个数值断言要么对着一条不变量（同一张图、同一个种子），要么对着一份独立写出的
参照实现（`test_dct_matches_a_full_transform` 里的暴力 DCT），而不是对着从库里抄出来的
黄金值。
"""

import math
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phashlib import (HammingIndex, Image, ahash, blur, brightness, centre_crop,
                      contrast, corpus, crop, dhash, flip_horizontal, from_hex,
                      gamma, hamming, linear_search, load, near_duplicates, noise,
                      phash, popcount, resize, rotate_left, save_bmp, save_pgm,
                      save_ppm, scene, shift, split_bands, stripes, to_hex)
from phashlib.dct import cos_factors, low_frequency
from phashlib.imagio import _clamp, loads
from phashlib.index import _bands_for
from phashlib.pixels import _weights


def gradient(width=32, height=32):
    return Image(width, height,
                 [[255.0 * x / (width - 1) for x in range(width)]
                  for y in range(height)])


class TestResize(unittest.TestCase):
    def test_identity_resize_is_exact(self):
        image = scene(0, 40, 24)
        same = resize(image, 40, 24)
        self.assertEqual(same.width, 40)
        for row, source in zip(same.pixels, image.pixels):
            for value, reference in zip(row, source):
                self.assertAlmostEqual(value, reference, places=9)

    def test_downscale_averages_the_area(self):
        image = Image(4, 1, [[0.0, 0.0, 200.0, 200.0]])
        small = resize(image, 2, 1)
        self.assertEqual(small.pixels[0], [0.0, 200.0])
        half = resize(image, 1, 1)
        self.assertAlmostEqual(half.pixels[0][0], 100.0, places=9)

    def test_upscale_repeats_rather_than_blends(self):
        image = Image(2, 2, [[10.0, 200.0], [10.0, 200.0]])
        big = resize(image, 4, 2)
        self.assertEqual(big.pixels[0], [10.0, 10.0, 200.0, 200.0])

    def test_weights_are_normalised_and_in_range(self):
        for limit, target in ((100, 7), (31, 32), (64, 32), (97, 8)):
            scale = limit / target
            for index in range(target):
                start, weights = _weights(scale, index, limit)
                self.assertAlmostEqual(sum(weights), 1.0, places=9)
                self.assertTrue(all(0.0 <= weight <= 1.0 for weight in weights))
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start + len(weights), limit + 1)
                if start + len(weights) > limit:
                    self.fail("weights reach past the last source pixel")

    def test_resize_rejects_empty_targets(self):
        with self.assertRaises(ValueError):
            resize(gradient(), 0, 4)


class TestPixels(unittest.TestCase):
    def test_brightness_clamps(self):
        image = Image(2, 1, [[100.0, 200.0]])
        lifted = brightness(image, 2.0, 10.0)
        self.assertEqual(lifted.pixels[0], [210.0, 255.0])
        self.assertEqual(brightness(image, 0.0).pixels[0], [0.0, 0.0])

    def test_noise_is_deterministic_per_seed(self):
        image = gradient(24, 24)
        self.assertEqual(noise(image, 9.0, 4).pixels, noise(image, 9.0, 4).pixels)
        self.assertNotEqual(noise(image, 9.0, 4).pixels, noise(image, 9.0, 5).pixels)

    def test_blur_flattens_a_spike(self):
        image = Image.grey(9, 9, 0.0)
        image.pixels[4][4] = 255.0
        soft = blur(image, 1)
        self.assertAlmostEqual(soft.pixels[4][4], 255.0 / 9.0, places=9)
        self.assertEqual(soft.pixels[0][0], 0.0)

    def test_crop_and_centre_crop(self):
        image = gradient(20, 10)
        cut = crop(image, 2, 3, 2, 3)
        self.assertEqual((cut.width, cut.height), (15, 5))
        self.assertEqual(cut.pixels[0][0], image.pixels[2][3])
        kept = centre_crop(image, 0.25)
        self.assertEqual((kept.width, kept.height), (10, 5))

    def test_crop_that_eats_the_image_raises(self):
        with self.assertRaises(ValueError):
            crop(gradient(10, 10), 5, 0, 0, 6)
        with self.assertRaises(ValueError):
            crop(gradient(10, 10), 0, 6, 5, 0)

    def test_geometric_maps_are_reversible(self):
        image = scene(3, 24, 16)
        self.assertEqual(flip_horizontal(flip_horizontal(image)).pixels, image.pixels)
        turned = rotate_left(rotate_left(rotate_left(rotate_left(image))))
        self.assertEqual(turned.pixels, image.pixels)
        moved = shift(image, 5, -3)
        self.assertEqual(shift(moved, -5, 3).pixels, image.pixels)

    def test_contrast_pulls_away_from_the_mean(self):
        image = Image(2, 2, [[100.0, 120.0], [80.0, 100.0]])
        pushed = contrast(image, 2.0)
        self.assertEqual(pushed.pixels[0], [100.0, 140.0])
        self.assertEqual(pushed.pixels[1], [60.0, 100.0])

    def test_gamma_endpoints_stay_fixed(self):
        image = Image(2, 1, [[0.0, 255.0]])
        bent = gamma(image, 2.2)
        self.assertEqual(bent.pixels[0], [0.0, 255.0])


class TestDct(unittest.TestCase):
    def test_dc_term_is_the_block_sum(self):
        grid = [[float(y * 4 + x + 1) for x in range(8)] for y in range(8)]
        block = low_frequency(grid, 2)
        self.assertAlmostEqual(block[0][0], sum(sum(row) for row in grid), places=6)

    def test_dct_matches_a_full_transform(self):
        n = 8
        rng = [math.sin(3.1 * i + 0.7 * j) for i in range(n) for j in range(n)]
        grid = [rng[i * n:(i + 1) * n] for i in range(n)]
        factors = cos_factors(n, n)
        reference = [[sum(grid[y][x] * factors[u][x] * factors[v][y]
                          for y in range(n) for x in range(n))
                      for u in range(n)] for v in range(n)]
        fast = low_frequency(grid, n)
        for row, want in zip(fast, reference):
            for value, expected in zip(row, want):
                self.assertAlmostEqual(value, expected, places=6)

    def test_constant_block_has_no_ac_energy(self):
        grid = [[7.0] * 16 for _ in range(16)]
        block = low_frequency(grid, 4)
        self.assertAlmostEqual(block[0][0], 7.0 * 16 * 16, places=6)
        for y in range(4):
            for x in range(4):
                if (y, x) != (0, 0):
                    self.assertAlmostEqual(block[y][x], 0.0, places=6)

    def test_non_square_block_rejected(self):
        with self.assertRaises(ValueError):
            low_frequency([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 2)

    def test_cannot_ask_for_more_frequencies_than_samples(self):
        with self.assertRaises(ValueError):
            low_frequency([[1.0, 2.0], [3.0, 4.0]], 4)


class TestHashes(unittest.TestCase):
    def test_identical_images_hash_identically(self):
        image = scene(11, 64, 64)
        self.assertEqual(phash(image), phash(image.copy()))
        self.assertEqual(dhash(image), dhash(image.copy()))
        self.assertEqual(ahash(image), ahash(image.copy()))

    def test_hamming_is_a_metric_on_samples(self):
        images = [scene(seed, 48, 48) for seed in range(6)]
        values = [phash(image) for image in images]
        for first in values:
            self.assertEqual(hamming(first, first), 0)
            for second in values:
                self.assertEqual(hamming(first, second), hamming(second, first))
        for a in range(len(values)):
            for b in range(len(values)):
                for c in range(len(values)):
                    self.assertLessEqual(hamming(values[a], values[b]),
                                         hamming(values[a], values[c])
                                         + hamming(values[c], values[b]))

    def test_scale_up_leaves_the_phash_alone(self):
        image = scene(2, 80, 80)
        self.assertLessEqual(hamming(phash(image), phash(resize(image, 160, 160))), 2)

    def test_pixel_noise_leaves_the_phash_alone(self):
        image = scene(4, 80, 80)
        self.assertLessEqual(hamming(phash(image), phash(noise(image, 8.0, 1))), 3)

    def test_dhash_ignores_a_uniform_brightness_shift(self):
        image = scene(5, 64, 64)
        moved = brightness(image, 1.4, 25.0)
        self.assertLessEqual(hamming(dhash(image), dhash(moved)), 3)

    def test_unrelated_scenes_are_far_apart(self):
        distances = [hamming(phash(scene(0, 64, 64)), phash(scene(seed, 64, 64)))
                     for seed in range(1, 9)]
        self.assertTrue(all(distance > 12 for distance in distances), distances)
        self.assertGreater(sum(distances) / len(distances), 20)

    def test_a_flat_field_still_hashes(self):
        value = phash(Image.grey(32, 32, 128.0))
        self.assertEqual(popcount(value), 32)  # every bit sits on the median tie
        self.assertEqual(ahash(Image.grey(8, 8, 0.0)), 0)

    def test_threshold_option_is_a_real_choice(self):
        image = scene(6, 48, 48)
        self.assertEqual(phash(image, threshold="mean"), phash(image))
        with self.assertRaises(ValueError):
            phash(image, threshold="middle")
        with self.assertRaises(ValueError):
            phash(image, size=8, keep=16)
        # 两种阈值只在某个系数明显离群时才会分道扬镳，所以诚实的说法是「有时不同」，
        # 而不是「总是不同」。
        differences = [hamming(phash(scene(seed, 48, 48)),
                               phash(scene(seed, 48, 48), threshold="mean"))
                       for seed in range(12)]
        self.assertGreater(sum(1 for value in differences if value), 0)

    def test_hex_round_trip(self):
        value = phash(scene(9, 48, 48))
        self.assertEqual(from_hex(to_hex(value)), value)
        self.assertEqual(len(to_hex(value)), 16)
        self.assertEqual(to_hex(255, 4), "00ff")

    def test_popcount(self):
        self.assertEqual(popcount(0), 0)
        self.assertEqual(popcount((1 << 64) - 1), 64)
        self.assertEqual(popcount(0b1011), 3)


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.pairs = [("img-%02d" % index, phash(scene(seed, 64, 64)))
                      for index, seed in enumerate(range(12))]

    def test_index_matches_linear_search_within_coverage(self):
        index = HammingIndex(self.pairs, bands=8)
        self.assertEqual(index.coverage, 7)
        for _name, value in self.pairs:
            for distance in range(0, 8):
                got = sorted(index.search(value, distance))
                want = sorted(linear_search(self.pairs, value, distance))
                self.assertEqual(got, want, "distance %d" % distance)

    def test_index_examines_fewer_entries_than_exist(self):
        index = HammingIndex(self.pairs, bands=8)
        index.search(self.pairs[0][1], 4)
        self.assertLessEqual(index.last_examined, len(self.pairs))

    def test_band_count_divides_the_hash(self):
        for bands in (1, 2, 4, 8, 16, 32, 64):
            HammingIndex(bands=bands)
        with self.assertRaises(ValueError):
            HammingIndex(bands=5)

    def test_bands_for_distance_rounds_up(self):
        self.assertEqual(_bands_for(2), 4)
        self.assertEqual(_bands_for(5), 8)
        self.assertEqual(_bands_for(7), 8)
        with self.assertRaises(ValueError):
            _bands_for(64)

    def test_out_of_range_hash_rejected(self):
        index = HammingIndex()
        with self.assertRaises(ValueError):
            index.add("x", 1 << 64)

    def test_split_and_rejoin_bands(self):
        value = 0x0123456789abcdef
        pieces = split_bands(value, 8)
        rebuilt = 0
        for band, piece in enumerate(pieces):
            rebuilt |= piece << (band * 8)
        self.assertEqual(rebuilt, value)

    def test_near_duplicates_groups_edits_not_scenes(self):
        image = scene(1, 64, 64)
        clusters = near_duplicates([("a", image), ("b", noise(image, 6.0, 2)),
                                    ("c", scene(31, 64, 64))], max_distance=5)
        self.assertEqual(clusters, [["a", "b"], ["c"]])

    def test_near_duplicates_with_nothing_close(self):
        clusters = near_duplicates([("a", scene(1, 64, 64)), ("b", scene(2, 64, 64))],
                                   max_distance=2)
        self.assertEqual([len(group) for group in clusters], [1, 1])


class TestImageIO(unittest.TestCase):
    def test_pgm_round_trip(self):
        image = scene(0, 40, 28)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "a.pgm")
            save_pgm(path, image)
            back = load(path)
        self.assertEqual((back.width, back.height), (40, 28))
        worst = max(abs(a - b) for ra, rb in zip(image.pixels, back.pixels)
                    for a, b in zip(ra, rb))
        self.assertLessEqual(worst, 0.5)
        self.assertEqual(hamming(phash(image), phash(back)), 0)

    def test_ppm_and_bmp_round_trip(self):
        image = scene(7, 33, 19)
        with tempfile.TemporaryDirectory() as folder:
            ppm = os.path.join(folder, "a.ppm")
            bmp = os.path.join(folder, "a.bmp")
            save_ppm(ppm, image)
            save_bmp(bmp, image)
            self.assertEqual(hamming(phash(image), phash(load(ppm))), 0)
            back = load(bmp)
        worst = max(abs(a - b) for ra, rb in zip(image.pixels, back.pixels)
                    for a, b in zip(ra, rb))
        self.assertLessEqual(worst, 0.5)

    def test_odd_width_bmp_rows_are_padded(self):
        image = scene(8, 5, 3)  # 5 px * 3 bytes = 15, padded up to 16 per row
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "odd.bmp")
            save_bmp(path, image)
            with open(path, "rb") as handle:
                handle.seek(10)
                offset = struct.unpack("<I", handle.read(4))[0]
            self.assertEqual(offset, 54)
            self.assertEqual(os.path.getsize(path) - offset, 3 * 16)
            self.assertEqual(load(path).width, 5)

    def test_ascii_pnm_with_comment(self):
        image = gradient(9, 5)
        text = "P2\n# written by hand\n%d %d\n255\n" % (image.width, image.height)
        text += "".join(" ".join(str(_clamp(v)) for v in row) + "\n"
                        for row in image.pixels)
        back = loads(text.encode("ascii"))
        self.assertEqual(back.width, 9)
        self.assertAlmostEqual(back.pixels[2][3], image.pixels[2][3], places=0)

    def test_ascii_pixmap(self):
        text = "P3\n2 2\n255\n10 20 30 40 50 60\n70 80 90 100 110 120\n"
        back = loads(text.encode("ascii"))
        self.assertEqual(len(back.pixels), 2)
        self.assertAlmostEqual(back.pixels[0][0], 0.299 * 10 + 0.587 * 20 + 0.114 * 30)

    def test_sixteen_bit_pgm_scales(self):
        raw = b"P5\n2 2\n65535\n" + struct.pack(">HHHH", 0, 32768, 65535, 1000)
        back = loads(raw)
        self.assertAlmostEqual(back.pixels[0][0], 0.0, places=3)
        self.assertAlmostEqual(back.pixels[0][1], 127.5, places=1)
        self.assertAlmostEqual(back.pixels[1][0], 255.0, places=3)

    def test_truncated_pixel_data_raises(self):
        raw = b"P5\n4 4\n255\n" + bytes(9)
        with self.assertRaises(ValueError):
            loads(raw)

    def test_unknown_magic_raises(self):
        with self.assertRaises(ValueError):
            loads(b"\x89PNG\r\n\x1a\n" + bytes(64))

    def test_image_validates_its_rows(self):
        with self.assertRaises(ValueError):
            Image(3, 2, [[1.0, 2.0, 3.0], [1.0, 2.0]])


class TestScenes(unittest.TestCase):
    def test_seed_is_reproducible(self):
        self.assertEqual(scene(4, 32, 32).pixels, scene(4, 32, 32).pixels)
        self.assertNotEqual(scene(4, 32, 32).pixels, scene(5, 32, 32).pixels)

    def test_corpus_includes_the_pathological_frames(self):
        images = corpus(count=4, seed=0, width=48, height=48)
        names = [name for name, _ in images]
        self.assertEqual(len(images), 8)
        for expected in ("stripes", "checker", "flat-grey", "flat-black"):
            self.assertIn(expected, names)

    def test_scenes_have_real_contrast(self):
        spans = []
        for seed in range(12):
            flat = [value for row in scene(seed, 64, 64).pixels for value in row]
            spans.append(max(flat) - min(flat))
        self.assertGreater(min(spans), 40.0)
        self.assertGreater(max(spans), 150.0)

    def test_stripes_alternate(self):
        image = stripes(0, 64, 8, count=4)
        row = image.pixels[0]
        self.assertNotEqual(row[0], row[16])


if __name__ == "__main__":
    unittest.main()
