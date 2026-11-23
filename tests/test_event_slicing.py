"""Binary-search window selection must pick exactly the events of the original boolean mask."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import event_slicing as S  # noqa: E402


class WindowSelection(unittest.TestCase):
    def test_identical_to_mask_on_sorted_input_with_duplicates(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(0, 3000))
            t = np.sort(rng.integers(0, 1500, n)).astype(np.float64)
            x, p = rng.integers(0, 240, n), rng.integers(0, 2, n)
            for _ in range(15):
                t0 = float(rng.integers(-10, 1510))
                t1 = t0 + float(rng.integers(0, 200))
                m = (t >= t0) & (t < t1)
                sl = S.window_index(t, t0, t1, S.is_time_ordered(t))
                self.assertIsInstance(sl, slice)
                for u, v in zip([a[m] for a in (x, t, p)], S.take((x, t, p), sl)):
                    self.assertTrue(np.array_equal(u, v))
                    self.assertEqual(u.dtype, v.dtype)

    def test_unordered_or_nan_keeps_mask(self):
        for t in (np.array([3.0, 1.0, 2.0]), np.array([1.0, np.nan, 2.0])):
            self.assertFalse(S.is_time_ordered(t))
            idx = S.window_index(t, 1.0, 3.0, S.is_time_ordered(t))
            self.assertEqual(idx.dtype, bool)
        self.assertTrue(S.is_time_ordered(np.array([])))

    def test_take_returns_copies(self):
        a = np.arange(10)
        b, = S.take((a,), slice(2, 5))
        b[0] = 99
        self.assertEqual(a[2], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
