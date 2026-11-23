from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tools"))
from stress_operators.event_stress import thin_events, quantize_timestamps
from stress_operators.event_utils import to_voxel_grid
from h5_to_e2vid_txt import convert_events


class OperatorRegressionTests(unittest.TestCase):
    def voxel(self, timestamps, remapping=None):
        n = len(timestamps)
        with patch.dict("os.environ", {"RELEIO_VOXEL_DEVICE": "cpu"}):
            return to_voxel_grid(np.zeros(n), np.zeros(n), np.array(timestamps),
                                 np.ones(n), H=2, W=2, remapping_maps=remapping)

    def test_empty_voxel_is_zero(self):
        voxel = self.voxel([])
        self.assertEqual(tuple(voxel.shape), (5, 2, 2))
        self.assertEqual(float(voxel.sum()), 0)

    def test_collapsed_timestamps_keep_event_mass(self):
        times = quantize_timestamps(np.array([100., 200.]), 1000)
        voxel = self.voxel(times)
        self.assertEqual(float(voxel.sum()), 2)
        self.assertEqual(float(voxel[0].sum()), 2)

    def test_single_event_and_identity_remapping(self):
        yy, xx = np.mgrid[:2, :2]
        self.assertEqual(float(self.voxel([10], np.array([xx, yy])).sum()), 1)

    def test_nonzero_duration_preserves_signed_sum(self):
        self.assertAlmostEqual(float(self.voxel([100, 150, 200]).sum()), 3)

    def test_unsigned_descending_time_rejected(self):
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            self.voxel(np.array([100, 50], dtype=np.uint64))

    def test_explicit_nanoseconds_and_retained_precision(self):
        t = np.array([0, 123, 246, 369, 10_000_000_000], dtype=np.int64)
        x = np.arange(len(t)); y = np.zeros(len(t)); p = np.ones(len(t))
        seconds, *_ = convert_events(t, x, y, p, time_unit="ns", time_origin="zero", keep=1, window_us=None)
        self.assertEqual(seconds[-1], 10)
        t = np.arange(100, dtype=np.int64)*123
        seconds, kept_x, *_ = convert_events(t, np.arange(100), np.zeros(100), np.ones(100),
            time_unit="ns", time_origin="zero", keep=.5, window_us=1000)
        np.testing.assert_array_equal(seconds, t[kept_x]*1e-9)

    def test_stress_parameters_are_not_silently_coerced(self):
        values = np.ones(2)
        with self.assertRaises(ValueError):
            thin_events(values, values, values, values, 1.1, 0)
        with self.assertRaises(ValueError):
            quantize_timestamps(values, -1)


if __name__ == "__main__":
    unittest.main()
