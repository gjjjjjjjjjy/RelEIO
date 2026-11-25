"""Tests for script/diag/run_ledger.py and the control checks of script/diag/precision_pilot.py.

Standard library only. Run from the repository root:  python3 tests/test_run_ledger.py
"""
import collections
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "script", "diag"))

import run_ledger as L  # noqa: E402
import precision_pilot as P  # noqa: E402
import precision_matrix as M  # noqa: E402

PY = sys.executable


def _json(path):
    with open(path) as f:
        return json.load(f)


def _first_line(path):
    with open(path) as f:
        return f.readline()
TB_ZERO_NORM = '''Traceback (most recent call last):
  File "/x/devo/dba.py", line 1090, in VisualIMUAlignment
    R0 = trans.FromTwoVectors(ng1,ng2)
  File "/x/devo/geoFunc/trans.py", line 245, in FromTwoVectors
    return Rotation.from_quat(q).as_matrix()
  File "_rotation.pyx", line 865, in scipy.spatial.transform._rotation.Rotation.__init__
ValueError: Found zero norm quaternions in `quat`.
'''


class OutcomeRules(unittest.TestCase):
    def hint(self, text, exit_code=None):
        return L.outcome_hint(L.parse_facts(text), exit_code=exit_code)

    def test_completed_requires_finite_mpe_and_clean_exit(self):
        self.assertEqual(self.hint("MPE is 0.42\n", 0), "completed")
        self.assertEqual(self.hint("MPE is 0.42\n", None), "completed")          # historical log
        self.assertEqual(self.hint("MPE is nan\n", 0), "undetermined:mpe_nan")
        self.assertEqual(self.hint("MPE is inf\n", 0), "undetermined:mpe_inf")
        self.assertEqual(self.hint("MPE is 0.42\n", 1), "undetermined:nonzero_exit_after_mpe")
        self.assertEqual(self.hint("nothing\n", 0), "undetermined:no_mpe_no_exception")
        self.assertEqual(self.hint("nothing\n", -11), "undetermined:nonzero_exit_no_exception")

    def test_failure_kinds(self):
        self.assertEqual(self.hint(TB_ZERO_NORM, 1), "numerical_failure:vi_alignment_nan_gravity")
        infra = ('Traceback (most recent call last):\n  File "/x/utils/load_utils.py", line 334, in it\n'
                 'RuntimeError: PytorchStreamReader failed reading zip archive\n')
        self.assertEqual(self.hint(infra, 1), "infrastructure:RuntimeError")
        other = ('Traceback (most recent call last):\n  File "/x/devo/dba.py", line 9, in f\n'
                 'IndexError: index 5 is out of bounds\n')
        self.assertTrue(self.hint(other, 1).startswith("undetermined:exception:IndexError"))
        self.assertEqual(self.hint("MPE is 0.4\n[runner-timeout] killed\n", None), "timeout")

    def test_categories(self):
        self.assertEqual(L.outcome_category("numerical_failure:vi_alignment_nan_gravity"), "numerical_failure")
        self.assertEqual(L.outcome_category("undetermined:mpe_nan"), "undetermined")
        self.assertEqual(L.outcome_category("not_run:gpu_not_exclusive"), "not_run")
        self.assertEqual(L.outcome_category("something new"), "undetermined")

    def test_cholinfo_banner_and_flags(self):
        f = L.parse_facts('[cholinfo] {"kind":"enabled","version":"cholinfo-1","where":"cuda_ba:dense"}\n'
                          '[cholinfo] {"kind":"flag","where":"cuda_ba:dense","info":3}\nMPE is 1\n')
        self.assertTrue(f["cholinfo"]["enabled_banner"])
        self.assertEqual(f["cholinfo"]["flagged"], 1)
        self.assertFalse(L.parse_facts("MPE is 1\n")["cholinfo"]["enabled_banner"])


class RunAndEnvironment(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = {k: os.environ.get(k) for k in ("FREQVIO_FORCE_FP32", "FREQVIO_STRICT_VI_INIT")}
        os.environ["FREQVIO_FORCE_FP32"] = "1"       # left over in the operator's shell
        os.environ["FREQVIO_STRICT_VI_INIT"] = "1"

    def tearDown(self):
        shutil.rmtree(self.dir)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_inherited_switches_do_not_leak(self):
        cmd = [PY, "-c", "import os,json; print(json.dumps({k: v for k, v in os.environ.items() "
                         "if k.startswith('FREQVIO_')})); print('MPE is 0.5')"]
        rec = L.run(cmd, {"FREQVIO_FORCE_FP32": "0"}, 0, self.dir, {"t": 1}, timeout=30, python=PY,
                    require_exclusive=False)
        rdir = os.path.join(self.dir, "runs", rec["run_id"])
        child = json.loads(_first_line(os.path.join(rdir, "run.log")))
        self.assertEqual(child["FREQVIO_FORCE_FP32"], "0")
        self.assertNotIn("FREQVIO_STRICT_VI_INIT", child)
        man = _json(os.path.join(rdir, "manifest.json"))
        self.assertEqual(man["effective_env"]["FREQVIO_FORCE_FP32"], "0")
        self.assertIn("FREQVIO_STRICT_VI_INIT", man["scrubbed_inherited_vars"])
        self.assertEqual(rec["outcome_category"], "completed")

    def test_none_override_unsets_inherited_variable(self):
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
        try:
            cmd = [PY, "-c", "import os; print('OVR=%s' % os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')); print('MPE is 0.5')"]
            rec = L.run(cmd, {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None}, 0, self.dir, {}, timeout=30, python=PY,
                        require_exclusive=False)
            rdir = os.path.join(self.dir, "runs", rec["run_id"])
            self.assertEqual(_first_line(os.path.join(rdir, "run.log")).strip(), "OVR=None")
            self.assertNotIn("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", _json(os.path.join(rdir, "manifest.json"))["effective_env"])
            rec = L.run(cmd, {}, 0, self.dir, {}, timeout=30, python=PY, require_exclusive=False)   # inherited otherwise
            rdir = os.path.join(self.dir, "runs", rec["run_id"])
            self.assertEqual(_first_line(os.path.join(rdir, "run.log")).strip(), "OVR=1")
        finally:
            os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)

    def test_nonzero_exit_after_mpe_is_not_completed(self):
        cmd = [PY, "-c", "print('MPE is 0.5'); raise SystemExit(3)"]
        rec = L.run(cmd, {}, 0, self.dir, {"t": 2}, timeout=30, python=PY, require_exclusive=False)
        self.assertEqual(rec["outcome_hint"], "undetermined:nonzero_exit_after_mpe")
        self.assertEqual(rec["outcome_category"], "undetermined")

    def test_runs_are_never_overwritten(self):
        cmd = [PY, "-c", "print('MPE is 0.5')"]
        a = L.run(cmd, {}, 0, self.dir, {}, timeout=30, python=PY, require_exclusive=False)
        b = L.run(cmd, {}, 0, self.dir, {}, timeout=30, python=PY, require_exclusive=False)
        self.assertNotEqual(a["run_id"], b["run_id"])
        self.assertEqual(len(L.load_ledger(self.dir)), 2)


GOOD_WATCH = {"n_samples": 3, "max_pids_seen": 1, "child_pid_seen": True, "other_pids_seen": [],
              "pids_at_end": [], "exclusive_at_end": True}


def _facts(arm, policy=None, dtype=None, chol=False, errors=0, numtrace=True,
           installed=P.REQUIRED_HOOKS, watch=GOOD_WATCH, stress=None):
    base = {"cholinfo": {"enabled_banner": chol, "flagged": 0}, "gpu_watch": watch}
    if stress is not None:
        base["stress"] = stress
    if not numtrace:
        return dict(base, numtrace=None)
    return dict(base, numtrace={"trace_installed": {"installed": list(installed), "install_errors": {}},
                                "precision_policy": policy, "dtype_probe": dtype,
                                "run_summary": {"tracer_errors": errors, "counts": {}}})


DEF_POL = {"arm": "default", "cfg_MIXED_PRECISION": True, "update_autocast": True, "matmul_allow_tf32": False,
           "cudnn_allow_tf32": True}
FP_POL = {"arm": "fp32_strict", "cfg_MIXED_PRECISION": False, "update_autocast": False, "matmul_allow_tf32": False,
          "cudnn_allow_tf32": False}
AMP_POL = {"arm": "amp_off", "cfg_MIXED_PRECISION": False, "update_autocast": False, "matmul_allow_tf32": False,
           "cudnn_allow_tf32": True}
TF_POL = {"arm": "tf32_off", "cfg_MIXED_PRECISION": True, "update_autocast": True, "matmul_allow_tf32": False,
          "cudnn_allow_tf32": False}
HALF = {"net": "torch.float16", "delta": "torch.float16", "weight": "torch.float16", "corr_input": "torch.float16",
        "imap_": "torch.float16", "fmap1_": "torch.float16", "update_params": "torch.float32",
        "autocast_enabled_now": True}
FULL = {"net": "torch.float32", "delta": "torch.float32", "weight": "torch.float32", "corr_input": "torch.float32",
        "imap_": "torch.float32", "fmap1_": "torch.float32", "update_params": "torch.float32",
        "autocast_enabled_now": False}


class PilotControls(unittest.TestCase):
    def test_good_runs_pass(self):
        self.assertEqual(P.verify_controls(_facts("default", DEF_POL, HALF, chol=True), "default", True), [])
        self.assertEqual(P.verify_controls(_facts("fp32_strict", FP_POL, FULL, chol=True), "fp32_strict", True), [])
        self.assertEqual(P.verify_controls(_facts("u", numtrace=False), "default_untraced", False), [])

    def test_violations_stop_the_pilot(self):
        self.assertTrue(P.verify_controls(_facts("fp32_strict", DEF_POL, HALF, chol=True), "fp32_strict", True))
        self.assertTrue(P.verify_controls(_facts("fp32_strict", FP_POL, HALF, chol=True), "fp32_strict", True))
        self.assertTrue(P.verify_controls(_facts("default", DEF_POL, None, chol=True), "default", True))
        self.assertTrue(P.verify_controls(_facts("default", DEF_POL, HALF, chol=False), "default", True))
        self.assertTrue(P.verify_controls(_facts("default", DEF_POL, HALF, errors=2), "default", False))
        self.assertTrue(P.verify_controls(_facts("default", None, HALF), "default", False))
        self.assertTrue(P.verify_controls(_facts("u", DEF_POL, HALF), "default_untraced", False))
        self.assertTrue(P.verify_controls({"numtrace": None, "cholinfo": {}, "gpu_watch": GOOD_WATCH}, "default", False))

    def test_incomplete_checkpoints_and_dtypes_are_rejected(self):
        # the case found in review: nothing installed, a single float32 field, rest missing
        probs = P.verify_controls(_facts("fp32_strict", FP_POL, {"delta": "torch.float32"}, chol=True,
                                         installed=[]), "fp32_strict", True)
        self.assertTrue(any("checkpoints not installed" in x for x in probs))
        self.assertTrue(any("dtype_probe fields missing" in x for x in probs))
        for hook in P.REQUIRED_HOOKS:                       # each checkpoint is individually required
            five = [h for h in P.REQUIRED_HOOKS if h != hook]
            probs = P.verify_controls(_facts("default", DEF_POL, HALF, installed=five), "default", False)
            self.assertTrue(any(hook in x for x in probs), hook)
        for field in P.DTYPE_FIELDS:                        # each dtype field is individually required
            for arm, pol, dt in (("default", DEF_POL, HALF), ("fp32_strict", FP_POL, FULL)):
                d = {k: v for k, v in dt.items() if k != field}
                self.assertTrue(P.verify_controls(_facts(arm, pol, d), arm, False), (arm, field))
        for field in ("net", "delta", "weight", "corr_input", "imap_", "fmap1_"):   # strict arm: all float32
            self.assertTrue(P.verify_controls(_facts("fp32_strict", FP_POL, dict(FULL, **{field: "torch.float16"})),
                                              "fp32_strict", False), field)
        self.assertTrue(P.verify_controls(_facts("fp32_strict", FP_POL, dict(FULL, autocast_enabled_now=True)),
                                          "fp32_strict", False))
        self.assertTrue(P.verify_controls(_facts("default", DEF_POL, dict(HALF, imap_="torch.float32")),
                                          "default", False))
        self.assertTrue(P.verify_controls(_facts("default", DEF_POL, dict(HALF, autocast_enabled_now=False)),
                                          "default", False))

    def test_gpu_sharing_invalidates(self):
        for arm, pol, dt in (("default", DEF_POL, HALF), ("default_untraced", None, None)):
            nt = arm != "default_untraced"
            self.assertTrue(P.verify_controls(_facts(arm, pol, dt, numtrace=nt, watch=None), arm, False))
            self.assertTrue(P.verify_controls(_facts(arm, pol, dt, numtrace=nt,
                                                     watch=dict(GOOD_WATCH, max_pids_seen=2)), arm, False))
            self.assertTrue(P.verify_controls(_facts(arm, pol, dt, numtrace=nt,
                                                     watch=dict(GOOD_WATCH, exclusive_at_end=False,
                                                                pids_at_end=["9"])), arm, False))
        # nvidia-smi unavailable at the end: unknown, not a violation by itself
        self.assertEqual(P.verify_controls(_facts("default", DEF_POL, HALF,
                                                  watch=dict(GOOD_WATCH, exclusive_at_end=None)), "default", False), [])

    def test_split_arms(self):
        self.assertEqual(P.verify_controls(_facts("amp_off", AMP_POL, FULL), "amp_off", False), [])
        self.assertEqual(P.verify_controls(_facts("tf32_off", TF_POL, HALF), "tf32_off", False), [])
        # wrong dtypes or wrong recorded policy for the arm are rejected
        self.assertTrue(P.verify_controls(_facts("amp_off", AMP_POL, HALF), "amp_off", False))
        self.assertTrue(P.verify_controls(_facts("tf32_off", TF_POL, FULL), "tf32_off", False))
        self.assertTrue(P.verify_controls(_facts("tf32_off", dict(TF_POL, cudnn_allow_tf32=True), HALF), "tf32_off", False))
        self.assertTrue(P.verify_controls(_facts("amp_off", DEF_POL, FULL), "amp_off", False))   # arm mismatch
        self.assertTrue(P.verify_controls(_facts("default", AMP_POL, HALF), "default", False))
        for arm in ("amp_off", "tf32_off", "default", "fp32_strict"):
            self.assertEqual(P.arm_env(arm, False)["FREQVIO_PRECISION"], arm)
        self.assertEqual(P.arm_env("default_untraced", True)["FREQVIO_BA_CHOLINFO"], "0")

    def test_tf32_split_arms(self):
        MM = {"arm": "matmul_tf32_off", "cfg_MIXED_PRECISION": True, "update_autocast": True,
              "matmul_allow_tf32": False, "cudnn_allow_tf32": True}
        CD = {"arm": "cudnn_tf32_off", "cfg_MIXED_PRECISION": True, "update_autocast": True,
              "matmul_allow_tf32": True, "cudnn_allow_tf32": False}
        NO = {"arm": "default", "cfg_MIXED_PRECISION": True, "update_autocast": True,
              "matmul_allow_tf32": False, "cudnn_allow_tf32": True}
        self.assertEqual(P.verify_controls(_facts("m", MM, HALF), "matmul_tf32_off", False), [])
        self.assertEqual(P.verify_controls(_facts("c", CD, HALF), "cudnn_tf32_off", False), [])
        self.assertEqual(P.verify_controls(_facts("n", NO, HALF), "no_tf32_override", False), [])
        # the env-unset arm must really have matmul TF32 off; a container default slipping through is rejected
        self.assertTrue(P.verify_controls(_facts("n", dict(NO, matmul_allow_tf32=True), HALF), "no_tf32_override", False))
        self.assertTrue(P.verify_controls(_facts("m", dict(MM, cudnn_allow_tf32=False), HALF), "matmul_tf32_off", False))
        self.assertIsNone(P.arm_env("no_tf32_override", True)["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"])
        BA = dict(DEF_POL, arm="ba_matmul_fp32", ba_local_matmul_fp32=True)
        self.assertEqual(P.verify_controls(_facts("b", BA, HALF), "ba_matmul_fp32", False), [])
        self.assertTrue(P.verify_controls(_facts("b", dict(BA, ba_local_matmul_fp32=False), HALF), "ba_matmul_fp32", False))

    def test_rotated_schedule(self):
        order = P.schedule(4, 0, arms=("default", "amp_off", "tf32_off", "fp32_strict"))
        self.assertEqual(order[:8], ["default", "amp_off", "tf32_off", "fp32_strict",
                                     "amp_off", "tf32_off", "fp32_strict", "default"])
        for a in ("default", "amp_off", "tf32_off", "fp32_strict"):
            self.assertEqual(order.count(a), 4)

    def test_every_arm_pins_cpu_voxelisation(self):
        # GPU voxelisation became the default after the frozen runs; the pilot must not inherit it
        for arm in P.ARM_ENV:
            self.assertEqual(P.arm_env(arm, True)["FREQVIO_VOXEL_DEVICE"], "cpu", arm)

    def test_schedule(self):
        order = P.schedule(20, 10)
        self.assertEqual(order.count("default"), 20)
        self.assertEqual(order.count("fp32_strict"), 20)
        self.assertEqual(order.count("default_untraced"), 10)
        self.assertEqual(order[:5], ["default", "fp32_strict", "fp32_strict", "default", "default_untraced"])


class PilotSummary(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _add(self, run_id, arm, hint, facts, cholinfo=False, mpe=None):
        os.makedirs(os.path.join(self.dir, "runs", run_id))
        with open(os.path.join(self.dir, "runs", run_id, "facts.json"), "w") as f:
            json.dump(facts, f)
        rec = {"run_id": run_id, "labels": {"arm": arm, "cholinfo": cholinfo}, "gpu_name": "g", "gpu_uuid": "u",
               "git_commit": "c", "git_diff_sha256": "d", "outcome_hint": hint, "mpe": mpe}
        with open(os.path.join(self.dir, "ledger.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def test_control_invalid_runs_are_excluded_from_counts(self):
        # one valid completion + one failure whose controls are invalid -> 0/1, not 1/2
        self._add("r1", "default", "completed", _facts("default", DEF_POL, HALF), mpe=0.4)
        self._add("r2", "default", "numerical_failure:vi_alignment_nan_gravity",
                  _facts("default", DEF_POL, HALF, installed=[]))
        self._add("r3", "default", "not_run:gpu_not_exclusive", {"outcome_hint": "not_run:gpu_not_exclusive"})
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            res = P.summarize(self.dir)
        d = res["default"]
        self.assertEqual((d["failures"], d["determinate"]), (0, 1))
        self.assertEqual((d["n_valid"], d["n_invalid"], d["n_not_run"]), (1, 1, 1))
        self.assertEqual(d["upper_if_undetermined_failed"], (0, 1))

    def test_summary_reverifies_even_without_control_problems_file(self):
        self._add("r1", "fp32_strict", "numerical_failure:vi_alignment_nan_gravity",
                  _facts("fp32_strict", FP_POL, HALF))                     # wrong dtypes, no problems file
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            res = P.summarize(self.dir)
        self.assertEqual(res["fp32_strict"]["n_invalid"], 1)
        self.assertEqual(res["fp32_strict"]["determinate"], 0)


class GpuWatch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (L.gpu_info, L.gpu_pids, L.gpu_mem_used)

    def tearDown(self):
        L.gpu_info, L.gpu_pids, L.gpu_mem_used = self.saved
        shutil.rmtree(self.dir)

    def test_second_process_during_run_is_recorded(self):
        L.gpu_info = lambda gpu: {"uuid": "GPU-fake", "name": "fake", "exclusive_at_start": True}
        calls = {"n": 0}

        def fake_pids(uuid):
            calls["n"] += 1
            return ["111", "222"] if calls["n"] == 1 else []      # someone joins, then everything is gone
        L.gpu_pids = fake_pids
        cmd = [PY, "-c", "import time; time.sleep(0.6); print('MPE is 0.5')"]
        rec = L.run(cmd, {}, 0, self.dir, {"arm": "default_untraced"}, timeout=30, python=PY,
                    require_exclusive=True, gpu_poll_s=0.1)
        facts = _json(os.path.join(self.dir, "runs", rec["run_id"], "facts.json"))
        w = facts["gpu_watch"]
        self.assertGreaterEqual(w["n_samples"], 1)
        self.assertEqual(w["max_pids_seen"], 2)
        self.assertTrue(w["exclusive_at_end"])
        self.assertEqual(rec["outcome_category"], "completed")
        self.assertTrue(any("more than one process" in x for x in P.verify_controls(facts, "default_untraced", False)))

    def test_child_gets_its_run_folder(self):
        L.gpu_info = lambda gpu: {"uuid": "GPU-fake", "name": "fake", "exclusive_at_start": True}
        L.gpu_pids = lambda uuid: []
        cmd = [PY, "-c", "import os; open(os.path.join(os.environ['FREQVIO_RUN_DIR'], 'traj_est_tum.txt'), 'w').write('x')"]
        rec = L.run(cmd, {}, 0, self.dir, {"arm": "default_untraced"}, timeout=30, python=PY,
                    require_exclusive=True, gpu_poll_s=0.1)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "runs", rec["run_id"], "traj_est_tum.txt")))

    def test_device_memory_is_sampled(self):
        L.gpu_info = lambda gpu: {"uuid": "GPU-fake", "name": "fake", "exclusive_at_start": True}
        L.gpu_pids = lambda uuid: []
        mem = iter([900, 2400, 2300] + [5] * 100)       # grows, dips, then the idle baseline after exit
        L.gpu_mem_used = lambda uuid: next(mem)
        cmd = [PY, "-c", "import time; time.sleep(0.5); print('MPE is 0.5')"]
        rec = L.run(cmd, {}, 0, self.dir, {"arm": "default_untraced"}, timeout=30, python=PY,
                    require_exclusive=True, gpu_poll_s=0.1)
        w = _json(os.path.join(self.dir, "runs", rec["run_id"], "facts.json"))["gpu_watch"]
        self.assertGreaterEqual(w["mem_samples"], 3)
        self.assertEqual(w["mem_used_mib_max"], 2400)
        self.assertEqual(w["mem_used_mib_at_end"], 5)

    def test_timeout_still_kills_the_child(self):
        L.gpu_info = lambda gpu: {"uuid": "GPU-fake", "name": "fake", "exclusive_at_start": True}
        L.gpu_pids = lambda uuid: []
        cmd = [PY, "-c", "import time; time.sleep(30)"]
        rec = L.run(cmd, {}, 0, self.dir, {}, timeout=0.5, python=PY, require_exclusive=True, gpu_poll_s=0.1)
        self.assertEqual(rec["outcome_hint"], "timeout")
        self.assertIsNone(rec["exit_code"])


CLEAN_STRESS = {"config": [{"version": "v2", "keep": "1.0", "tq_us": "0", "no_voxcache": "1"}], "evidence": None}


def _stress(keep="1.0", tq="0", ev=None):
    return {"config": [{"version": "v2", "keep": keep, "tq_us": tq, "no_voxcache": "1"}], "evidence": ev}


class StressCells(unittest.TestCase):
    def test_parse_cell(self):
        self.assertEqual(P.parse_cell("keep:0.3")["env"], {"FREQVIO_EVENT_KEEP": "0.3"})
        self.assertEqual(P.parse_cell("tq:5000")["env"], {"FREQVIO_TIME_QUANT_US": "5000"})
        c = P.parse_cell("patch:32")
        self.assertEqual((c["env"], c["opts"], c["patches"]), ({}, ["PATCHES_PER_FRAME", "32"], 32))
        self.assertEqual(P.parse_cell(None)["cell"], "base")
        self.assertEqual(len(P.FROZEN_CELLS), 12)
        for bad in ("keep:0", "keep:1.5", "tq:-1", "patch:0", "patch:x", "blur:3", "keep"):
            with self.assertRaises(ValueError):
                P.parse_cell(bad)

    def test_env_and_cmd(self):
        env = P.arm_env("no_tf32_override", True, cell="keep:0.1")
        self.assertEqual((env["FREQVIO_EVENT_KEEP"], env["FREQVIO_TIME_QUANT_US"]), ("0.1", "0"))
        self.assertIsNone(env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"])
        self.assertEqual(P.arm_env("default", True, cell="tq:1000")["FREQVIO_TIME_QUANT_US"], "1000")
        self.assertEqual(P.arm_env("default", True)["FREQVIO_EVENT_KEEP"], "1.0")
        cmd, _ = P.build_cmd("rpgdavis", "boxes_6dof", 1000, "/tmp/x", opts=["PATCHES_PER_FRAME", "16"])
        self.assertEqual(cmd[-3:], ["--opts", "PATCHES_PER_FRAME", "16"])
        self.assertIn("--t_imu_cam_identity", cmd)
        self.assertNotIn("--opts", P.build_cmd("vector", "desk_normal1", 1000, "/tmp/x")[0])

    def test_stress_checks(self):
        pol = dict(DEF_POL, cfg_PATCHES_PER_FRAME=96, matmul_allow_tf32=True)     # container: TF32 on
        ok = lambda st, cell, p=pol: P.verify_controls(_facts("default", p, HALF, stress=st), "default", False, cell=cell)
        self.assertEqual(ok(_stress(), "base"), [])
        self.assertEqual(ok(_stress(), "patch:96"), [])
        self.assertEqual(ok(_stress("0.3"), "keep:0.3"), [])
        ev = {"keep": "0.3", "tq_us": "0", "n_in": "1000", "actual_keep": "0.301", "tq_changed": "0/1000"}
        self.assertEqual(ok(_stress("0.3", ev=ev), "keep:0.3"), [])
        self.assertTrue(ok(_stress("0.3", ev=dict(ev, actual_keep="0.95")), "keep:0.3"))   # thinning not applied
        self.assertTrue(ok(_stress("1.0"), "keep:0.3"))                                  # banner disagrees
        self.assertTrue(ok({"config": [], "evidence": None}, "keep:0.3"))                # no banner
        self.assertTrue(ok(_stress(), "patch:32"))                                       # patches 96 recorded
        self.assertEqual(ok(_stress(), "patch:32", dict(pol, cfg_PATCHES_PER_FRAME=32)), [])
        tq = {"keep": "1.0", "tq_us": "5000", "n_in": "1000", "actual_keep": "1.0", "tq_changed": "0/1000"}
        self.assertTrue(ok(_stress(tq="5000", ev=tq), "tq:5000"))                        # nothing quantised
        self.assertEqual(ok(_stress(tq="5000", ev=dict(tq, tq_changed="400/1000")), "tq:5000"), [])
        # the default arm must really have TF32 on, otherwise it equals no_tf32_override
        self.assertTrue(ok(_stress(), "keep:1.0", dict(pol, matmul_allow_tf32=False)))
        self.assertTrue(ok(_stress(), "keep:1.0", dict(pol, NVIDIA_TF32_OVERRIDE="0")))       # TF32 off globally
        self.assertEqual(ok(_stress(), "keep:1.0", dict(pol, NVIDIA_TF32_OVERRIDE=None)), [])
        # ledgers before round 3 carry no cell label: stress not checked
        self.assertEqual(P.verify_controls(_facts("default", DEF_POL, HALF), "default", False), [])
        # untraced runs: stress banner still checked, patch count cannot be
        self.assertEqual(P.verify_controls(_facts("u", numtrace=False, stress=_stress()), "default_untraced", False,
                                           cell="patch:32"), [])
        self.assertTrue(P.verify_controls(_facts("u", numtrace=False, stress=_stress("0.5")), "default_untraced", False,
                                          cell="keep:0.3"))

    def test_parse_facts_stress_lines(self):
        text = ("[event-stress] version=v2 keep=0.3 tq_us=0 no_voxcache=1 mvsec_converted\n"
                "[stress-evidence] version=v2 where=run_DEIO2 keep=0.3 tq_us=0 frames=10 n_in=100 n_out=30 "
                "actual_keep=0.3 tq_changed=0/100 tq_changed_ratio=0.0 first_window_checksum=n1 full_stream_hash=None\n"
                "MPE is 0.4\n")
        st = L.parse_facts(text)["stress"]
        self.assertEqual(st["config"][0]["keep"], "0.3")
        self.assertEqual(st["evidence"]["actual_keep"], "0.3")
        self.assertEqual(L.parse_facts("MPE is 1\n")["stress"], {"config": [], "evidence": None})
        # VECtor: tqdm's first bar is on the line before the lazily started iterator prints its banner
        bar = "\r0it [00:00, ?it/s][event-stress] version=v2 keep=0.1 tq_us=0 no_voxcache=1 _cached_voxel_list/left\n"
        st = L.parse_facts(bar + "\r12it [00:01, 9.1it/s][stress-evidence] version=v2 where=run_DEIO2_crash keep=0.1 "
                           "tq_us=0 frames=12 n_in=10 n_out=1 actual_keep=0.1 tq_changed=0/10\n")["stress"]
        self.assertEqual((st["config"][0]["keep"], st["config"][0]["tq_us"]), ("0.1", "0"))
        self.assertEqual(st["evidence"]["where"], "run_DEIO2_crash")

    def test_jobs_completeness_voxcache(self):
        jobs = M.build_jobs([("vector", "desk_normal1"), ("rpgdavis", "boxes_6dof")], ["default", "no_tf32_override"],
                            5, [1000, 1001, 1002, 1003, 1004], cells=list(P.FROZEN_CELLS))
        self.assertEqual(len(jobs), 2 * 12 * 2 * 5)
        c = collections.Counter((j[1], j[4], j[2], j[3]) for j in jobs)
        self.assertEqual(set(c.values()), {1})                   # each seed once per (scene, cell, arm)
        self.assertNotEqual(jobs[0][4], jobs[2][4])              # units interleaved
        mk = lambda j: {"labels": {"dataset": j[0], "scene": j[1], "arm": j[2], "seed": j[3], "cell": j[4]}}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(M.completeness(jobs, [mk(j) for j in jobs], [], False), 0)
            self.assertEqual(M.completeness(jobs, [mk(j) for j in jobs[1:]], [], False), 3)
        ns = types.SimpleNamespace
        self.assertFalse(M.voxcache_for(ns(voxcache=None), "rpgdavis"))
        self.assertTrue(M.voxcache_for(ns(voxcache=[]), "vector"))
        self.assertTrue(M.voxcache_for(ns(voxcache=["rpgdavis"]), "rpgdavis"))
        self.assertFalse(M.voxcache_for(ns(voxcache=["rpgdavis"]), "vector"))
        self.assertFalse(M.voxcache_for(ns(), "vector"))
        self.assertEqual(M.scene_label({"dataset": "a", "scene": "b", "cell": "keep:0.3"}), "a/b keep:0.3")
        self.assertEqual(M.scene_label({"dataset": "a", "scene": "b", "cell": "base"}), "a/b")
        self.assertEqual(M.scene_label({"dataset": "a", "scene": "b"}), "a/b")

    def test_front_load_inputs(self):
        jobs = M.build_jobs([("vector", "desk_normal1"), ("rpgdavis", "boxes_6dof")], ["default", "no_tf32_override"],
                            5, [1000, 1001, 1002, 1003, 1004], cells=list(P.FROZEN_CELLS))
        out = M.front_load_inputs(jobs, lambda ds: ds == "rpgdavis")
        self.assertEqual(collections.Counter(out), collections.Counter(jobs))
        head = out[:7]                                           # 7 input conditions: clean, 3 keep, 3 tq
        conds = {(P.parse_cell(j[4])["keep"], P.parse_cell(j[4])["tq_us"]) for j in head}
        self.assertEqual(len(conds), 7)
        self.assertTrue(all(j[0] == "rpgdavis" for j in head))
        self.assertEqual(collections.Counter(j[2] for j in head), {"default": 4, "no_tf32_override": 3})
        self.assertEqual(M.front_load_inputs(jobs, lambda ds: False), jobs)

    def test_select_part(self):
        from script.run_degradation_sweep import DATASETS
        # the 8 sequences of the main matrix; rpgdavis_up is DEIO's own DAVIS240C pipeline, run only in the
        # comparison of data pipelines (script/diag/run_davis_pipelines.sh) with explicit units
        scenes = [(d, sc) for d in DATASETS if d != "rpgdavis_up" for sc in DATASETS[d][3]]
        self.assertEqual(len(scenes), 8)
        self.assertEqual(DATASETS["rpgdavis_up"][3], ["boxes_6dof", "poster_6dof"])
        jobs = M.build_jobs(scenes, ["default", "no_tf32_override"], 5, [1000, 1001, 1002, 1003, 1004],
                            cells=list(P.FROZEN_CELLS))
        self.assertEqual(len(jobs), 960)
        for n in (1, 2, 3, 4):
            parts = [M.select_part(jobs, i, n) for i in range(1, n + 1)]
            self.assertEqual(sorted(j for p in parts for j in p), sorted(jobs))        # disjoint, complete
            seq_part = {}
            for i, p in enumerate(parts):
                self.assertTrue(p)
                for j in p:
                    self.assertEqual(seq_part.setdefault((j[0], j[1]), i), i)       # a sequence stays on one instance
                for sc in {(j[0], j[1]) for j in p}:                               # with all its cells and arms
                    self.assertEqual(len([j for j in p if (j[0], j[1]) == sc]), 120)

    def test_explicit_units(self):
        units = [("vector", "corner_slow1", "keep:0.1"), ("vector", "corner_slow1", "keep:1.0"),
                 ("fpv", "indoor_forward_3", "patch:64")]
        jobs = M.build_jobs([], ["default", "no_tf32_override"], 10, [1000, 1001, 1002, 1003, 1004], units=units)
        self.assertEqual(len(jobs), 3 * 2 * 10)
        self.assertEqual({(j[0], j[1], j[4]) for j in jobs}, set(units))
        c = collections.Counter((j[1], j[4], j[2], j[3]) for j in jobs)
        self.assertEqual(set(c.values()), {2})                   # each seed twice per (unit, arm)

    def test_paired_mpe_pairs_by_repeat_before_dropping_failures(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "runs"))
            # seed 1000 twice per arm; default: repeat 1 fails, repeat 2 completes (MPE 2.0)
            # stock: repeat 1 completes (MPE 1.0), repeat 2 completes (MPE 2.1)  -> one pair (2.0, 2.1)
            plan = [("default", 1, None), ("no_tf32_override", 2, 1.0), ("default", 3, 2.0), ("no_tf32_override", 4, 2.1)]
            for arm, idx, m in plan:
                rid = "r%03d" % idx
                os.makedirs(os.path.join(d, "runs", rid))
                pol = dict(DEF_POL, cfg_PATCHES_PER_FRAME=96, matmul_allow_tf32=(arm == "default"))
                with open(os.path.join(d, "runs", rid, "facts.json"), "w") as f:
                    json.dump(_facts(arm, pol, HALF, chol=True, stress=_stress("0.1")), f)
                rec = {"run_id": rid, "gpu_uuid": "u", "gpu_name": "g", "git_commit": "c", "git_diff_sha256": "d",
                       "cuda_ba_sha256": "x", "wall_s": 1.0, "mpe": m,
                       "outcome_hint": "completed" if m is not None else "numerical_failure:vi_alignment_nan_gravity",
                       "labels": {"dataset": "vector", "scene": "corner_slow1", "arm": arm, "seed": 1000,
                                  "cell": "keep:0.1", "cholinfo": True, "job_index": idx}}
                with open(os.path.join(d, "ledger.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                res = M.summarize(d)
            pm = res["aggregate"]["paired_mpe"]["no_tf32_override"]
            self.assertEqual(pm["pairs"], 1)
            self.assertAlmostEqual(pm["median_diff"], 0.1)
        finally:
            shutil.rmtree(d)

    def test_clean_cell_and_frozen_plans(self):
        c = P.parse_cell("clean")
        self.assertEqual((c["keep"], c["tq_us"], c["patches"]), (1.0, 0, 96))
        self.assertEqual(c["opts"], ["PATCHES_PER_FRAME", "96"])
        self.assertEqual(P.arm_env("no_tf32_override", True, cell="clean")["FREQVIO_EVENT_KEEP"], "1.0")
        self.assertEqual(len(P.MAIN_CELLS), 10)
        pol = dict(DEF_POL, cfg_PATCHES_PER_FRAME=96, arm="default")
        self.assertEqual(P.verify_controls(_facts("no_tf32_override", pol, HALF, stress=_stress()),
                                           "no_tf32_override", False, cell="clean"), [])
        # the frozen round-4 plans in script/diag/plans/ are what build_jobs produces for the launcher's arguments
        from script.run_degradation_sweep import DATASETS  # noqa: F401
        scenes = [tuple(x.split("/")) for x in ("vector/desk_normal1 vector/corner_slow1 hku/vicon_dark1 "
                  "tumvie/mocap-3d-trans fpv/indoor_forward_3 rpgdavis/boxes_6dof rpgdavis/poster_6dof "
                  "rpgdavis/shapes_6dof").split()]
        main = M.build_jobs(scenes, ["no_tf32_override"], 10, list(range(2000, 2010)), cells=list(P.MAIN_CELLS))
        with open(os.path.join(ROOT, "script", "diag", "plans", "round4_main_plan.json")) as f:
            self.assertEqual(json.load(f), [list(j) for j in main])
        self.assertEqual(len(main), 800)
        self.assertEqual({len({j[3] for j in main if (j[1], j[4]) == (sc, c)}) for sc, c in {(j[1], j[4]) for j in main}}, {10})
        # the dual-arm main block (default and no_tf32_override, the same seeds) proposed 2026-11-20
        dual = M.build_jobs(scenes, ["default", "no_tf32_override"], 10, list(range(2000, 2010)), cells=list(P.MAIN_CELLS))
        with open(os.path.join(ROOT, "script", "diag", "plans", "round4_main_dual_plan.json")) as f:
            self.assertEqual(json.load(f), [list(j) for j in dual])
        self.assertEqual(len(dual), 1600)
        self.assertEqual(set(collections.Counter((j[1], j[4], j[3]) for j in dual).values()), {2})   # both arms per seed
        # the reduced dual-arm design (DESIGN=core): clean and the strongest level of each axis
        core = M.build_jobs(scenes, ["default", "no_tf32_override"], 10, list(range(2000, 2010)),
                            cells=["clean", "keep:0.1", "tq:10000", "patch:16"])
        with open(os.path.join(ROOT, "script", "diag", "plans", "round4_core_dual_plan.json")) as f:
            self.assertEqual(json.load(f), [list(j) for j in core])
        self.assertEqual(len(core), 640)

    def test_compare_inputs_across_ledgers(self):
        sys.path.insert(0, os.path.join(ROOT, "script", "diag"))
        import compare_inputs as CI
        def ledger(*summaries, cell="keep:0.1"):
            d = tempfile.mkdtemp()
            os.makedirs(os.path.join(d, "runs"))
            for i, x in enumerate(summaries):
                rid = "r%d" % i
                os.makedirs(os.path.join(d, "runs", rid))
                with open(os.path.join(d, "runs", rid, "facts.json"), "w") as f:
                    json.dump({"numtrace": {"run_summary": x}}, f)
                rec = {"run_id": rid, "labels": {"dataset": "vector", "scene": "corner_slow1", "cell": cell},
                       "outcome_hint": "completed" if x.get("input_frames") == 300 else "numerical_failure:x"}
                with open(os.path.join(d, "ledger.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            return d
        full = {"input_frames": 300, "input_fingerprint": "F", "input_checkpoints": {"1": "a", "100": "b", "300": "F"}}
        crash = {"input_frames": 120, "input_fingerprint": "c", "input_checkpoints": {"1": "a", "100": "b"}}
        bad = {"input_frames": 120, "input_fingerprint": "c", "input_checkpoints": {"1": "a", "100": "Z"}}
        ref, same, diff = ledger(full, full, crash), ledger(crash), ledger(bad)
        other_cell = ledger(full, cell="keep:1.0")
        try:
            self.assertEqual(CI.compare(same, ref, "vector/corner_slow1", "keep:0.1")[0], 0)
            self.assertEqual(CI.compare(diff, ref, "vector/corner_slow1", "keep:0.1")[0], 1)
            self.assertEqual(CI.compare(same, "/nonexistent", "vector/corner_slow1", "keep:0.1")[0], 2)
            self.assertEqual(CI.compare(other_cell, ref, "vector/corner_slow1", "keep:0.1")[0], 2)    # nothing to compare
            self.assertEqual(CI.compare(other_cell, ledger(full, cell="clean"), "vector/corner_slow1", "clean")[0], 0)
            # completed runs with no whole-input hash are not evidence; completed runs must have the same frames
            nohash = dict(full, input_fingerprint=None)
            a, b = ledger(nohash), ledger(nohash)
            self.assertEqual(CI.compare(a, b, "vector/corner_slow1", "keep:0.1")[0], 2)
            self.assertFalse(M.input_consistency([nohash, nohash])["ok"])
            short = {"input_frames": 300, "input_fingerprint": "G", "input_checkpoints": {"1": "a", "100": "b"}}
            c2 = ledger(short)
            self.assertEqual(CI.compare(c2, ref, "vector/corner_slow1", "keep:0.1")[0], 1)
            code, text = CI.compare(same, ref, "vector/corner_slow1", "keep:0.1")
            self.assertEqual(code, 0)
            self.assertIn("only the common prefix up to frame 100", text)      # the crashed run proves a prefix only
            for d in (a, b, c2):
                shutil.rmtree(d)
        finally:
            for d in (ref, same, diff, other_cell):
                shutil.rmtree(d)

    def test_input_identity_pools_clean_cells(self):
        full = {"input_frames": 300, "input_fingerprint": "f", "input_checkpoints": {"100": "a", "300": "f"}}
        thin = {"input_frames": 300, "input_fingerprint": "t", "input_checkpoints": {"100": "q", "300": "t"}}
        g = lambda cell, seed, x: {"valid": [({"labels": {"seed": seed, "cell": cell}}, {"numtrace": {"run_summary": x}})]}
        ident = M.input_identity({("s keep:1.0", "default"): g("keep:1.0", 1000, full),
                                  ("s patch:32", "default"): g("patch:32", 1001, full),
                                  ("s tq:0", "no_tf32_override"): g("tq:0", 1002, full),
                                  ("s keep:0.3", "default"): g("keep:0.3", 1000, thin)})
        self.assertIn("3 runs, identical", ident["s"])
        self.assertIn("1 runs, identical", ident["s keep=0.3 tq_us=0"])
        self.assertNotIn("INPUT DIFFERS", " ".join(ident.values()))

    def test_summary_aggregates(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "runs"))
            n = 0
            for cell in ("keep:1.0", "keep:0.1"):
                for arm in ("default", "no_tf32_override"):
                    for seed in (1000, 1001):
                        n += 1
                        rid = "r%03d" % n
                        os.makedirs(os.path.join(d, "runs", rid))
                        c = P.parse_cell(cell)
                        pol = dict(DEF_POL, cfg_PATCHES_PER_FRAME=96, matmul_allow_tf32=(arm == "default"))
                        if arm == "no_tf32_override":
                            pol["arm"] = "default"
                        with open(os.path.join(d, "runs", rid, "facts.json"), "w") as f:
                            json.dump(_facts(arm, pol, HALF, chol=True, stress=_stress(c["level"])), f)
                        fail = arm == "default" and cell == "keep:0.1" and seed == 1000
                        rec = {"run_id": rid, "gpu_uuid": "u", "gpu_name": "g", "git_commit": "c", "git_diff_sha256": "d",
                               "cuda_ba_sha256": "x", "wall_s": 1.0, "mpe": None if fail else 0.4 + 0.01 * n,
                               "outcome_hint": "numerical_failure:vi_alignment_nan_gravity" if fail else "completed",
                               "labels": {"dataset": "vector", "scene": "desk_normal1", "arm": arm, "seed": seed,
                                          "cell": cell, "cholinfo": True}}
                        with open(os.path.join(d, "ledger.jsonl"), "a") as f:
                            f.write(json.dumps(rec) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                res = M.summarize(d)
            agg = res["aggregate"]
            self.assertEqual(agg["total"], {"default": (1, 4), "no_tf32_override": (0, 4)})
            self.assertEqual(agg["cell"]["keep:0.1"], {"default": (1, 2), "no_tf32_override": (0, 2)})
            self.assertEqual(agg["clean_vs_degraded"]["default"], ((0, 2), (1, 2)))
            self.assertEqual(agg["paired_mpe"]["no_tf32_override"]["pairs"], 3)
            # a seed repeated in both arms gives one pair per repeat, not one per seed
            with open(os.path.join(d, "ledger.jsonl")) as f:
                recs = [json.loads(x) for x in f]
            extra = []
            for r in recs:
                if r["outcome_hint"] == "completed" and r["labels"]["cell"] == "keep:1.0":
                    n += 1
                    rid = "r%03d" % n
                    shutil.copytree(os.path.join(d, "runs", r["run_id"]), os.path.join(d, "runs", rid))
                    extra.append(dict(r, run_id=rid, mpe=r["mpe"] + 0.1, labels=dict(r["labels"], job_index=n)))
            with open(os.path.join(d, "ledger.jsonl"), "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in extra)
            with contextlib.redirect_stdout(io.StringIO()):
                rep = M.summarize(d)
            self.assertEqual(rep["aggregate"]["paired_mpe"]["no_tf32_override"]["pairs"], 3 + 2)
            with open(os.path.join(d, "ledger.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
            self.assertEqual(res[("vector/desk_normal1 keep:0.1", "default")]["n_valid"], 2)
            # two part ledgers summarised together give the same totals
            d2 = tempfile.mkdtemp()
            try:
                recs = [json.loads(x) for x in open(os.path.join(d, "ledger.jsonl"))]
                os.makedirs(os.path.join(d2, "runs"))
                for r in recs[4:]:
                    shutil.move(os.path.join(d, "runs", r["run_id"]), os.path.join(d2, "runs", r["run_id"]))
                with open(os.path.join(d, "ledger.jsonl"), "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in recs[:4])
                with open(os.path.join(d2, "ledger.jsonl"), "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in recs[4:])
                with contextlib.redirect_stdout(io.StringIO()):
                    res2 = M.summarize([d, d2])
                self.assertEqual(res2["aggregate"]["total"], agg["total"])
            finally:
                shutil.rmtree(d2)
        finally:
            shutil.rmtree(d)


class Matrix(unittest.TestCase):
    def test_build_jobs_balanced(self):
        scenes = [("vector", "desk_normal1"), ("hku", "vicon_dark1")]
        jobs = M.build_jobs(scenes, ["default", "fp32_strict"], 10, [1000, 1001, 1002, 1003, 1004], untraced_k=2)
        self.assertEqual(len(jobs), 2 * (2 * 10 + 2))
        c = collections.Counter((j[1], j[2]) for j in jobs)
        for sc in ("desk_normal1", "vicon_dark1"):
            self.assertEqual(c[(sc, "default")], 10)
            self.assertEqual(c[(sc, "fp32_strict")], 10)
            self.assertEqual(c[(sc, "default_untraced")], 2)
        self.assertNotEqual(jobs[0][1], jobs[1][1])            # scenes interleaved
        seeds = collections.Counter((j[1], j[2], j[3]) for j in jobs if j[2] != "default_untraced")
        self.assertEqual(set(seeds.values()), {2})               # each seed twice per (scene, arm)
        four = M.build_jobs([("vector", "desk_normal1")], ["default", "amp_off", "tf32_off", "fp32_strict"], 4, [1000])
        self.assertEqual([j[2] for j in four[:5]], ["default", "amp_off", "tf32_off", "fp32_strict", "amp_off"])

    def test_worker_exception_is_not_success(self):
        q = M.queue.Queue()
        q.put((0, ("vector", "desk_normal1", "default", 1000)))
        saved = P.build_cmd
        try:
            def boom(*a, **k):
                raise OSError("disk full")
            P.build_cmd = boom
            stop, lock, done, errors = M.threading.Event(), M.threading.Lock(), [], []
            args = types.SimpleNamespace(ledger="/nonexistent", cholinfo=False, timeout=1, gpu_poll=1, total=1)
            with contextlib.redirect_stdout(io.StringIO()):
                M._worker(0, q, args, stop, lock, done, errors)
                status = M.completeness([("vector", "desk_normal1", "default", 1000)], done, errors, stop.is_set())
            self.assertTrue(stop.is_set())
            self.assertEqual(len(errors), 1)
            self.assertEqual(status, 3)
        finally:
            P.build_cmd = saved

    def test_completeness_counts_every_planned_run(self):
        jobs = M.build_jobs([("vector", "desk_normal1")], ["default", "fp32_strict"], 2, [1000])
        mk = lambda j: {"labels": {"dataset": j[0], "scene": j[1], "arm": j[2], "seed": j[3]}}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(M.completeness(jobs, [mk(j) for j in jobs], [], False), 0)
            self.assertEqual(M.completeness(jobs, [mk(j) for j in jobs[:-1]], [], False), 3)

    def test_input_identity(self):
        def g(*summaries):
            return {"valid": [({"labels": {"seed": 1000}}, {"numtrace": {"run_summary": x}}) for x in summaries]}
        full = {"input_frames": 300, "input_fingerprint": "f", "input_checkpoints": {"100": "a", "200": "b", "300": "f"}}
        crash = {"input_frames": 220, "input_fingerprint": "c", "input_checkpoints": {"100": "a", "200": "b"}}
        bad = {"input_frames": 300, "input_fingerprint": "x", "input_checkpoints": {"100": "a", "200": "z", "300": "x"}}
        ok = M.input_identity({("s", "default"): g(full, crash), ("s", "fp32_strict"): g(full)})
        self.assertIn("identical", ok["s"])
        self.assertIn("deepest frame 300", ok["s"])
        self.assertNotIn("INPUT DIFFERS", ok["s"])
        diff = M.input_identity({("s", "default"): g(full, bad)})
        self.assertIn("INPUT DIFFERS", diff["s"])
        # an early stop must not hide a later difference between two other runs
        early = {"input_frames": 10, "input_fingerprint": "e", "input_checkpoints": {"1": "p", "10": "q"}}
        a = {"input_frames": 100, "input_fingerprint": "A", "input_checkpoints": {"1": "p", "10": "q", "100": "r"}}
        b = {"input_frames": 100, "input_fingerprint": "A", "input_checkpoints": {"1": "p", "10": "q", "100": "X"}}
        c = M.input_consistency([early, a, b])
        self.assertFalse(c["ok"])
        self.assertEqual(c["differs_at"], [100])

    def test_voxcache_env(self):
        if not all(os.path.exists(os.path.join(ROOT, rel)) for rel in P.VOXEL_SOURCES):
            self.skipTest("needs the estimator checkout: the cache namespace hashes its voxelisation sources")
        env = P.arm_env("default", True, voxcache=True)
        self.assertEqual(env["FREQVIO_NO_VOXCACHE"], "0")
        self.assertTrue(env["FREQVIO_VOXCACHE_NS"].startswith("cpu"))
        self.assertEqual(P.arm_env("default", True)["FREQVIO_NO_VOXCACHE"], "1")
        self.assertNotIn("FREQVIO_VOXCACHE_NS", P.arm_env("default", True))
        sys.path.insert(0, ROOT)
        from utils import event_stress as ES
        os.environ.pop("FREQVIO_VOXCACHE_NS", None)
        self.assertEqual(ES.cache_namespace(), "")                    # default file names unchanged
        os.environ["FREQVIO_VOXCACHE_NS"] = env["FREQVIO_VOXCACHE_NS"]
        try:
            self.assertEqual(ES.cache_namespace(), "_ns" + env["FREQVIO_VOXCACHE_NS"])
            os.environ["FREQVIO_VOXCACHE_NS"] = "../evil"
            with self.assertRaises(ValueError):
                ES.cache_namespace()
        finally:
            os.environ.pop("FREQVIO_VOXCACHE_NS", None)

    def test_fisher(self):
        self.assertAlmostEqual(M.fisher_two_sided(10, 24, 0, 24), 2 * 1961256 / 6540715896, places=9)  # 2*C(24,10)/C(48,10)
        self.assertAlmostEqual(M.fisher_two_sided(0, 24, 10, 24), M.fisher_two_sided(10, 24, 0, 24))
        self.assertEqual(M.fisher_two_sided(3, 10, 3, 10), 1.0)

    def test_workers_stop_on_control_failure_and_summary(self):
        d = tempfile.mkdtemp()
        saved = (P.build_cmd, L.run)
        calls = {"n": 0}

        def fake_run(cmd, env, gpu, ledger_dir, labels, **kw):
            calls["n"] += 1
            rid = "r%03d" % calls["n"]
            os.makedirs(os.path.join(ledger_dir, "runs", rid))
            bad = calls["n"] == 3                               # third run has broken controls
            arm = labels["arm"]
            pol, dt = {"default": (dict(DEF_POL, matmul_allow_tf32=True), HALF), "fp32_strict": (FP_POL, FULL)}[arm]
            facts = _facts(arm, dict(pol, cfg_PATCHES_PER_FRAME=96), dt, chol=labels["cholinfo"],
                           installed=[] if bad else P.REQUIRED_HOOKS, stress=CLEAN_STRESS)
            with open(os.path.join(ledger_dir, "runs", rid, "facts.json"), "w") as f:
                json.dump(facts, f)
            hint = "numerical_failure:vi_alignment_nan_gravity" if (arm == "default" and calls["n"] % 2) else "completed"
            rec = {"run_id": rid, "labels": labels, "gpu_uuid": "u%d" % gpu, "gpu_name": "g", "git_commit": "c",
                   "git_diff_sha256": "d", "cuda_ba_sha256": "x", "outcome_hint": hint, "mpe": 0.4, "wall_s": 1.0}
            with open(os.path.join(ledger_dir, "ledger.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec
        try:
            P.build_cmd = lambda ds, sc, seed, split, opts=(): (["true"], [])
            L.run = fake_run
            jobs = M.build_jobs([("vector", "desk_normal1")], ["default", "fp32_strict"], 5, [1000])
            q = M.queue.Queue()
            for i, j in enumerate(jobs):
                q.put((i, j))
            args = types.SimpleNamespace(ledger=d, cholinfo=True, timeout=10, gpu_poll=1, total=len(jobs))
            stop, lock, done = M.threading.Event(), M.threading.Lock(), []
            with contextlib.redirect_stdout(io.StringIO()):
                M._worker(0, q, args, stop, lock, done)
            self.assertTrue(stop.is_set())
            self.assertEqual(len(done), 3)                      # stopped right after the bad run
            self.assertTrue(os.path.exists(os.path.join(d, "runs", "r003", "control_problems.json")))
            with contextlib.redirect_stdout(io.StringIO()):
                res = M.summarize(d)
            # order D F | F D ...: r001 default, r002 fp32_strict, r003 fp32_strict (bad controls)
            self.assertEqual((res[("vector/desk_normal1", "default")]["n_valid"],
                              res[("vector/desk_normal1", "default")]["n_invalid"]), (1, 0))
            self.assertEqual((res[("vector/desk_normal1", "fp32_strict")]["n_valid"],
                              res[("vector/desk_normal1", "fp32_strict")]["n_invalid"]), (1, 1))   # r003 excluded
        finally:
            P.build_cmd, L.run = saved
            shutil.rmtree(d)


class AttributionTable(unittest.TestCase):
    def test_counts_each_run_once_and_only_traced_valid_runs(self):
        import attribution_table as A
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            def add(d, rid, arm, failed, flag):
                os.makedirs(os.path.join(d, "runs", rid))
                pol = dict(DEF_POL, cfg_PATCHES_PER_FRAME=96, matmul_allow_tf32=(arm == "default"))
                f = _facts(arm, pol, HALF, chol=True, stress=_stress("0.1"),
                           watch=dict(GOOD_WATCH, mem_used_mib_max=2400))
                if arm == P.UNTRACED:
                    f = _facts(arm, numtrace=False, stress=_stress("0.1"))
                elif failed:
                    f["numtrace"]["first_detected_nonfinite"] = {"where": "fastba.BA(out)", "tensor": "poses",
                                                                 "frame": 121, "imu_enabled": False}
                with open(os.path.join(d, "runs", rid, "facts.json"), "w") as fh:
                    json.dump(f, fh)
                with open(os.path.join(d, "runs", rid, "run.log"), "w") as fh:
                    if flag:
                        fh.write('[cholinfo] {"kind":"flag","where":"cuda_ba:dense","info":1}\n')
                rec = {"run_id": rid, "gpu_uuid": "u", "gpu_name": "g", "git_commit": "c", "git_diff_sha256": "d",
                       "cuda_ba_sha256": "x", "wall_s": 60.0, "mpe": None if failed else 0.5,
                       "outcome_hint": "numerical_failure:vi_alignment_nan_gravity" if failed else "completed",
                       "labels": {"dataset": "vector", "scene": "corner_slow1", "arm": arm, "seed": 1000,
                                  "cell": "keep:0.1", "cholinfo": True}}
                with open(os.path.join(d, "ledger.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            add(a, "r1", "default", True, True)
            add(a, "r2", "default", False, False)
            add(a, "r3", P.UNTRACED, True, False)
            add(b, "r1", "default", True, True)            # the same run listed again in another ledger
            add(b, "r4", "no_tf32_override", False, False)
            rows, excluded = A.collect([a, b])
            self.assertEqual(sorted(x["run_id"] for x in rows), ["r1", "r2", "r4"])
            self.assertEqual(sum(n for k, n in excluded.items() if "duplicate" in k), 1)
            self.assertEqual(sum(n for k, n in excluded.items() if "untraced" in k), 1)
            fail = [x for x in rows if x["category"] == "numerical_failure"]
            self.assertEqual([(x["chol_flags"], x["first_where"], x["ba_tf32"]) for x in fail],
                             [(1, "fastba.BA(out)", True)])
            self.assertEqual({x["device_mem_mib"] for x in rows}, {2400})
            self.assertFalse(next(x for x in rows if x["run_id"] == "r4")["ba_tf32"])
            out, argv = io.StringIO(), sys.argv
            try:
                sys.argv = ["attribution_table.py", "--ledgers", a, b]
                with contextlib.redirect_stdout(out):
                    A.main()
            finally:
                sys.argv = argv
            total = out.getvalue().split("TOTAL:")[1]
            self.assertRegex(total, r"BA TF32 on\s+2\s+1\s+1\s+0")       # runs, failures, flagged, completed+flag
            self.assertRegex(total, r"BA TF32 off\s+1\s+0\s+0\s+0")
        finally:
            shutil.rmtree(a)
            shutil.rmtree(b)


class Round4Analysis(unittest.TestCase):
    def test_newcombe_matches_the_published_example_and_mover(self):
        import round4_analysis as R
        d, lo, hi = R.newcombe(56, 70, 48, 80)       # Newcombe (1998) example, method 10: 0.0524 to 0.3339
        self.assertAlmostEqual(d, 0.2, places=6)
        self.assertAlmostEqual(lo, 0.0524, places=4)
        self.assertAlmostEqual(hi, 0.3339, places=4)
        self.assertEqual(R.newcombe_paired(0, 8, 0, 2), R.newcombe(8, 10, 0, 10))   # phi = 0: same as unpaired
        self.assertEqual((R.q([1, 2, 3, 4], 0.5), R.q([1, 2, 3], 0.5)), (2.5, 2))
        i, lo, hi = R.mover((0.5, 0.2, 0.7), (0.1, -0.1, 0.4))
        self.assertAlmostEqual(i, 0.4)
        self.assertAlmostEqual(lo, 0.4 - (0.09 + 0.09) ** 0.5)
        self.assertAlmostEqual(hi, 0.4 + (0.04 + 0.04) ** 0.5)

    def test_table_from_a_ledger(self):
        import round4_analysis as R
        d = tempfile.mkdtemp()
        try:
            n = [0]

            def add(scene, cell, arm, seed, failed, mpe=0.5):
                n[0] += 1
                rid = "r%03d" % n[0]
                c = P.parse_cell(cell)
                os.makedirs(os.path.join(d, "runs", rid))
                pol = dict(DEF_POL, cfg_PATCHES_PER_FRAME=c["patches"], matmul_allow_tf32=(arm == "default"))
                f = _facts(arm, pol, HALF, chol=True, stress=_stress(str(c["keep"]), str(c["tq_us"])))
                with open(os.path.join(d, "runs", rid, "facts.json"), "w") as fh:
                    json.dump(f, fh)
                rec = {"run_id": rid, "gpu_uuid": "u", "gpu_name": "g", "git_commit": "c", "git_diff_sha256": "d",
                       "cuda_ba_sha256": "x", "wall_s": 60.0, "mpe": None if failed else mpe,
                       "outcome_hint": "numerical_failure:x" if failed else "completed",
                       "labels": {"dataset": "vector", "scene": scene, "arm": arm, "seed": seed, "cell": cell,
                                  "cholinfo": True}}
                with open(os.path.join(d, "ledger.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            for seed in range(4):
                add("corner", "clean", "default", seed, seed == 0)
                add("corner", "clean", "no_tf32_override", seed, False)
                add("corner", "keep:0.1", "default", seed, seed < 3)
                add("corner", "keep:0.1", "no_tf32_override", seed, False, mpe=150.0 if seed == 1 else 0.5)
            rows, excluded = R.collect(d)
            self.assertEqual(len(rows), 16)
            self.assertFalse(excluded)
            out, argv = io.StringIO(), sys.argv
            try:
                sys.argv = ["round4_analysis.py", "--ledger", d]
                with contextlib.redirect_stdout(out):
                    R.main()
            finally:
                sys.argv = argv
            o = out.getvalue()
            self.assertRegex(o, r"vector/corner keep:0.1\s+3/4 .*0/4 \(not observed; upper 49%\)")
            self.assertIn("MPE > 100%: default 0, no_tf32_override 1", o)
            self.assertRegex(o, r"vector/corner\s+keep\s+\+75% .*\+50% ")     # interaction = 75% - 25%
        finally:
            shutil.rmtree(d)


class GyroBiasCell(unittest.TestCase):
    def test_parse_and_ledger_fact(self):
        c = P.parse_cell("gbias:5")
        d = math.radians(5)
        self.assertEqual((c["axis"], c["keep"], c["tq_us"], c["patches"]), ("gbias", 1.0, 0, P.CLEAN_PATCHES))
        for x, y in zip(c["gyro_bias_rad_s"], (d, -d, d)):
            self.assertAlmostEqual(x, y, places=12)
        self.assertEqual(c["env"]["FREQVIO_IMU_GYRO_BIAS_RAD_S"], "0.0872664626,-0.0872664626,0.0872664626")
        for bad in ("gbias:0", "gbias:-1", "gbias:x", "gbias"):
            with self.assertRaises(ValueError):
                P.parse_cell(bad)
        f = L.parse_facts("[imu-stress] gyro_bias_rad_s=0.0872664626,-0.0872664626,0.0872664626 "
                                   "delta_mean_rad_s=0.08726646,-0.08726646,0.08726646 n=12000 changed=1\n")
        self.assertEqual(f["imu_stress"]["changed"], "1")
        self.assertIsNone(L.parse_facts("nothing here\n")["imu_stress"])

    def test_control_check(self):
        good = {"gyro_bias_rad_s": "0.0872664626,-0.0872664626,0.0872664626", "changed": "1"}
        st = _stress("1.0", "0")

        def probs(cell, ims):
            f = {"stress": st, "imu_stress": ims}
            return [p for p in P.stress_problems(f, cell, traced=False)]
        self.assertEqual(probs("gbias:5", good), [])
        self.assertTrue(any("no [imu-stress]" in p for p in probs("gbias:5", None)))
        self.assertTrue(any("expected" in p for p in probs("gbias:5", dict(good, gyro_bias_rad_s="0.1,0.1,0.1"))))
        self.assertTrue(any("did not change" in p for p in probs("gbias:5", dict(good, changed="0"))))
        self.assertTrue(any("without IMU stress" in p for p in probs("clean", good)))
        self.assertEqual(probs("clean", {"gyro_bias_rad_s": "none"}), [])
        self.assertEqual(probs("clean", None), [])          # logs from before the [imu-stress] line


class CrossSystemEval(unittest.TestCase):
    """Real label format (dataset and scene separate), every outcome class, no KeyError."""
    def test_outcomes_and_window(self):
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location(
            "cse", os.path.join(os.path.dirname(__file__), "..", "script", "diag", "cross_system_eval.py"))
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        T = tempfile.mkdtemp()
        try:
            M.ROOT = T
            rng = np.random.default_rng(0)
            led = os.path.join(T, "ledger")
            os.makedirs(os.path.join(led, "runs"))
            cfg = ("%YAML:1.0\nextrinsicRotation: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n"
                   "   data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\nextrinsicTranslation: !!opencv-matrix\n   rows: 3\n"
                   "   cols: 1\n   dt: d\n   data: [0.0, 0.0, 0.0]\n")
            lst, T0 = [], 1.468e15
            good_bias = {"gyro_bias_rad_s": "0.0872664626,-0.0872664626,0.0872664626", "changed": "1"}
            for s in ("boxes_6dof", "poster_6dof"):
                t = T0 + np.arange(0, 60e6, 5000.0)
                p = np.stack([np.sin(t / 1e6), np.cos(t / 1.3e6), 0.3 * np.sin(t / 0.7e6)], 1)
                q = np.tile([0, 0, 0, 1.0], (len(t), 1))
                os.makedirs(os.path.join(T, "outputs", "supereio", "gt"), exist_ok=True)
                np.savetxt(os.path.join(T, "outputs", "supereio", "gt", "%s_optitrack_davis_us.txt" % s),
                           np.column_stack([t, p, q]))
                os.makedirs(os.path.join(T, "data", "RPGDAVIS_pp", s))
                np.savetxt(os.path.join(T, "data", "RPGDAVIS_pp", s, "mocap_data.txt"),
                           np.column_stack([t - T0 + 50.0, p, q]))
                for cond in ("clean", "gbias"):
                    for i in range(1, 4):
                        rid = "%s_%s_%d" % (s, cond, i)
                        os.makedirs(os.path.join(led, "runs", rid))
                        cell = "clean" if cond == "clean" else "gbias:5"
                        c = P.parse_cell(cell)
                        pol = dict(DEF_POL, cfg_PATCHES_PER_FRAME=c["patches"], matmul_allow_tf32=False)
                        f = _facts("no_tf32_override", pol, HALF, chol=True, stress=_stress("1.0", "0"))
                        f["imu_stress"] = good_bias if cond == "gbias" else {"gyro_bias_rad_s": "none"}
                        if s == "poster_6dof" and cond == "gbias" and i == 3:
                            f["imu_stress"] = dict(good_bias, gyro_bias_rad_s="0.1,0.1,0.1")   # wrong bias
                        failed = s == "boxes_6dof" and cond == "gbias" and i == 2
                        json.dump(f, open(os.path.join(led, "runs", rid, "facts.json"), "w"))
                        k = np.arange(100, len(t) - 50, 3)
                        np.savetxt(os.path.join(led, "runs", rid, "traj_est_tum.txt"),
                                   np.column_stack([t[k] - T0 + 50.0, p[k] + 0.01 * rng.standard_normal((len(k), 3)), q[k]]))
                        rec = {"run_id": rid, "outcome_hint": "numerical_failure:x" if failed else "completed",
                               "mpe": None if failed else 0.5, "gpu_uuid": "u", "gpu_name": "g", "git_commit": "c",
                               "labels": {"dataset": "rpgdavis", "scene": s, "cell": cell, "seed": 3000 + i,
                                          "arm": "no_tf32_override", "cholinfo": True}}
                        with open(os.path.join(led, "ledger.jsonl"), "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        rd = os.path.join("se", rid)
                        os.makedirs(os.path.join(T, rd))
                        open(os.path.join(T, rd, "config.yaml"), "w").write(cfg)
                        reboot = 1 if (s == "poster_6dof" and cond == "clean" and i == 2) else 0
                        died = 1 if (s == "boxes_6dof" and cond == "gbias" and i == 1) else 0
                        rc = 1 if died else 0          # exit code not 0, a process died, a trajectory is left
                        if not (s == "boxes_6dof" and cond == "clean" and i == 3):      # this attempt: no summary
                            open(os.path.join(T, rd, "summary.txt"), "w").write(
                                "rosbag play exit code: 0\nevent_detector restarts: 0\n"
                                "failure detections (system reboot!): %d\nprocess deaths: %d\n" % (reboot, died))
                        k2 = np.arange(400, len(t) - 20, 15)
                        with open(os.path.join(T, rd, "evio_result_no_loop.csv"), "w") as fh:
                            for tt, pp in zip(t[k2] * 1e3, p[k2] + 0.02 * rng.standard_normal((len(k2), 3))):
                                fh.write("%.0f,%.5f,%.5f,%.5f,1,0,0,0,0,0,0,\n" % (tt, *pp))
                        lst.append("%s %s %d %s rc=%d" % (s, cond, i, rd, rc))
            open(os.path.join(T, "list.txt"), "w").write("\n".join(lst) + "\n")
            out, argv = io.StringIO(), sys.argv
            try:
                sys.argv = ["x", "--deio-ledger", led, "--supereio-list", os.path.join(T, "list.txt")]
                with contextlib.redirect_stdout(out):
                    M.main()
            finally:
                sys.argv = argv
            o = out.getvalue()
            self.assertRegex(o, r"DEIO\s+boxes_6dof\s+gbias\s+3\s+2/3\s+1\s+0\s+0")           # one numerical failure
            self.assertRegex(o, r"DEIO\s+poster_6dof\s+gbias\s+3\s+2/3\s+0\s+0\s+1")          # one wrong bias: excluded
            self.assertRegex(o, r"SuperEIO\s+boxes_6dof\s+clean\s+3\s+2/3\s+0\s+1\s+0")       # no summary: undetermined
            self.assertRegex(o, r"SuperEIO\s+poster_6dof\s+clean\s+3\s+2/3\s+1\s+0\s+0")      # reboot: failed
            self.assertRegex(o, r"SuperEIO\s+boxes_6dof\s+gbias\s+3\s+2/3\s+0\s+1\s+0")       # died, rc 1: undetermined
            self.assertIn("DEIO time + 1467999999999950 us", o)
        finally:
            shutil.rmtree(T)


class ImuUnits(unittest.TestCase):
    def test_ratio_is_one_with_deg_s_and_57_with_rad_s(self):
        import numpy as np
        import check_imu_units as U
        t_gt = np.arange(0, 30e6, 1e6 / 120)                      # 120 Hz ground truth, 30 s
        rate = 0.8 + 0.5 * np.sin(t_gt / 1e6)                        # rad/s about z, always > 0.3
        ang = np.concatenate([[0.0], np.cumsum(rate[:-1] * np.diff(t_gt) / 1e6)])
        q = np.stack([np.zeros_like(ang), np.zeros_like(ang), np.sin(ang / 2), np.cos(ang / 2)], axis=1)
        t_imu = np.arange(0, 30e6, 1e6 / 200)                       # 200 Hz IMU
        w = np.interp(t_imu, t_gt, rate)
        gyro_deg = np.stack([0 * w, 0 * w, w * 180 / np.pi], axis=1)
        acc = np.tile([0.0, 0.0, 9.81], (len(t_imu), 1))
        good = U.physics(t_imu, gyro_deg, acc, t_gt, q)
        self.assertAlmostEqual(good["window_ratio_median"], 1.0, delta=0.05)
        self.assertTrue(U.verdict(good))
        wrong = U.physics(t_imu, gyro_deg * np.pi / 180, acc, t_gt, q)   # file actually in rad/s
        self.assertLess(wrong["window_ratio_median"], 0.05)
        self.assertFalse(U.verdict(wrong))
        raw = np.column_stack([t_imu / 1e6, gyro_deg * np.pi / 180, acc])
        proc = np.column_stack([t_imu, gyro_deg, acc])
        self.assertTrue(U.conversion(proc, raw)["ok"])
        self.assertFalse(U.conversion(np.column_stack([t_imu, gyro_deg * 2, acc]), raw)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
