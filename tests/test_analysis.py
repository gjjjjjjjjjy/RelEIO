import csv
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from analysis import Run, direction_counts, read_records, success_curve, summarize, wilson


def run(value, repeat="0"):
    if value is None:
        return Run("data", "scene", "0", repeat, "crash", None)
    return Run("data", "scene", "0", repeat, "ok", Decimal(str(value)))


class StatisticalRegressionTests(unittest.TestCase):
    def test_reference_tie_has_no_direction(self):
        result = direction_counts([run(1), run(3)], [run(2), run(2)])
        self.assertEqual(result["ties"], 4)
        self.assertEqual(result["opposite"], 0)

    def test_single_tie_is_separate_and_swap_symmetric(self):
        a, b = [run(1), run(2)], [run(2), run(4)]
        forward = direction_counts(a, b)
        backward = direction_counts(b, a)
        self.assertEqual(forward["ties"], 1)
        self.assertEqual(forward, backward)

    def test_extreme_excluded_from_conditional_worst(self):
        result = summarize([run("0.46"), run("0.78"), run("504.92"), run(None)])
        self.assertEqual(result["worst_mpe"], .78)
        self.assertEqual(result["extreme"], 1)
        self.assertEqual(result["crash"], 1)
        self.assertEqual(result["failure_rate"], .5)

    def test_empty_success_distribution_is_explicit(self):
        result = summarize([run(None), run(200)])
        self.assertIsNone(result["median_mpe"])
        self.assertIsNone(result["worst_mpe"])

    def test_success_curve_includes_failed_runs_in_denominator(self):
        curve = success_curve([run(.1), run(2), run(None)])
        self.assertEqual(curve[-1]["successes"], 2)
        self.assertEqual(curve[-1]["n"], 3)

    def test_zero_failure_interval_is_not_zero_width(self):
        low, high = wilson(0, 5)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, .43448246478317476)

    def test_malformed_success_and_duplicate_rows_raise(self):
        with tempfile.TemporaryDirectory(prefix="releio-test-") as directory:
            path = Path(directory)/"runs.csv"
            header = "dataset,scene,keep,seed,status,mpe\n"
            path.write_text(header+"d,s,1,0,ok,\n")
            with self.assertRaisesRegex(ValueError, "missing mpe"):
                read_records(path, "keep")
            path.write_text(header+"d,s,1,0,ok,1\n"*2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_records(path, "keep")


if __name__ == "__main__":
    unittest.main()
