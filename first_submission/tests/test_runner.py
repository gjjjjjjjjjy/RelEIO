import csv
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_matrix import execute


class RunnerRegressionTests(unittest.TestCase):
    def test_failed_process_is_retained_and_all_axes_continue(self):
        with tempfile.TemporaryDirectory(prefix="releio-test-") as directory:
            root = Path(directory)
            stub = root/"stub.py"
            stub.write_text("import sys\nraise SystemExit(17 if sys.argv[2] == '1' else 0)\n")
            command = f"{shlex.quote(sys.executable)} {shlex.quote(str(stub))} {{scene}} {{seed}} {{keep}} {{tq_us}} {{patch}}"
            result = execute(template=command, system="test", scenes=["scene"], seeds=[1, 2],
                             axes=["keep", "time", "patch"], output=root/"runs")
            with (root/"runs/execution.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(result, 1)
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(r["exit_code"] == "17" for r in rows), 12)
            self.assertEqual({r["tq_us"] for r in rows}, {"0", "1000", "5000", "10000"})
            self.assertEqual({r["patch"] for r in rows}, {"96", "64", "32", "16"})
            receipt = json.loads((root/"runs/00000/command.json").read_text())
            self.assertEqual(receipt["argv"][2:4], ["scene", "1"])

    def test_missing_stress_argument_raises(self):
        with tempfile.TemporaryDirectory(prefix="releio-test-") as directory:
            with self.assertRaisesRegex(ValueError, "consume"):
                execute(template="python eval.py {scene} {seed}", system="test",
                        scenes=["scene"], seeds=[1], axes=["keep"], output=Path(directory)/"runs")


if __name__ == "__main__":
    unittest.main()
