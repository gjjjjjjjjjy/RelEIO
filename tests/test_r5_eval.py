#!/usr/bin/env python3
"""Tests of script/diag/r5_eval.py on synthetic runs (no ledger, no bag): the common window, the scores under C and D
on a scale-collapsed trajectory, and the counting of readouts 1-4.

  python3 tests/test_r5_eval.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("r5_eval", os.path.join(ROOT, "script", "diag", "r5_eval.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def _gt(span_s=20.0, hz=100.0, t0=1_000_000_000.0):
    t = t0 + np.arange(0, span_s, 1 / hz) * 1e6
    a = np.linspace(0, 4 * np.pi, len(t))
    p = np.stack([np.cos(a), np.sin(a), 0.2 * np.sin(3 * a)], axis=1)
    return np.column_stack([t, p])


def _run(reader, cond, status="completed", t=None, p=None, rep=1, seq="boxes_6dof", **kw):
    x = {"reader": reader, "seq": seq, "cond": cond, "rep": rep, "run": "%s_%s_%s" % (reader, cond, rep),
         "status": status, "t": t, "p": p, "A": None, "D_full": None}
    x.update(kw)
    return x


def test_common_window_and_scores():
    gt = _gt()
    t, p = gt[:, 0], gt[:, 1:4]
    runs = [_run("deio_own", "clean", t=t[100:], p=p[100:]),             # starts 1 s late
            _run("supereio", "clean", t=t[:-300], p=p[:-300] * 0.3),      # ends 3 s early, scale collapsed
            _run("deio_ours", "keep0.1", t=t, p=p),                       # not clean: does not set the window
            _run("supereio", "clean", status="failed: no trajectory")]
    win, ends = E.common_window(runs, seqs=("boxes_6dof",))
    w0, w1 = win["boxes_6dof"]
    assert w0 == t[100] and w1 == t[-301], (w0, w1)
    assert ends["boxes_6dof"]["start_set_by"][0] == "deio_own" and ends["boxes_6dof"]["end_set_by"][0] == "supereio"
    E.score(runs, {"boxes_6dof": gt}, win)
    exact, shrunk = runs[0], runs[1]
    assert exact["D"] < 1e-6 and exact["C"] < 1e-6
    # same poses, only the scale differs: C (Sim(3)) close to 0, D (SE(3)) large; scale of the fit about 1/0.3
    assert shrunk["C"] < 1e-6 and shrunk["D"] > 5 and abs(shrunk["scale"] - 1 / 0.3) < 1e-6, (shrunk["C"], shrunk["D"])
    assert runs[3]["D"] is None


def test_readouts_counts():
    rows = []
    vals = [(0.5, 0.4, 0.3), (5, 0.4, 0.3), (12, 0.5, 0.4), (150, 0.6, 0.5), (300, 20, 15), (8, 12, 0.2),
            (None, None, 0.3)]                                             # (D, C, A); last: no D in the window
    for i, (d, c, a) in enumerate(vals):
        rows.append(_run("deio_own", "keep0.1", rep=i, D=d, C=c, A=a, scale=1.0))
    rows += [_run("deio_own", "keep0.1", status="failed: numerical failure", rep=10, D=None, C=None, A=None),
             _run("deio_own", "keep0.1", status="undetermined: timeout", rep=11, D=0.5, C=0.5, A=0.5),   # scored, not counted
             _run("deio_own", "keep0.1", status="excluded: seed printed none", rep=12, D=0.1, C=0.1, A=0.1),
             _run("supereio", "keep0.1", rep=1, D=50, C=0.5, scale=0.2)]
    ro = E.readouts(rows)
    o = ro[("deio_own", "boxes_6dof", "keep0.1")]
    assert (o["runs"], o["completed"], o["failed"], o["undetermined"], o["excluded"]) == (10, 7, 1, 1, 1), o
    assert o["denominator"] == 9                       # excluded runs are not in the denominator
    assert o["within_10_D"] == 3                       # 0.5, 5, 8 (the excluded 0.1 does not count)
    assert o["above_100_D"] == 2                       # 150, 300
    assert o["CD_scored"] == 6 and o["C_within_D_not"] == 2 and o["D_within_C_not"] == 1, o   # 12, 150 | 8
    assert o["AD_scored"] == 6 and o["A_within_D_not"] == 2 and o["D_within_A_not"] == 0, o   # 12, 150 (300: A 15)
    s = ro[("supereio", "boxes_6dof", "keep0.1")]
    assert s["C_within_D_not"] == 1 and "AD_scored" not in s


def test_supereio_identity_checked_without_record_field():
    """A record written before r5_accept.py stored as_planned: the identity is checked again from summary.txt."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        def attempt(name, bag):
            d = os.path.join(tmp, name)
            os.makedirs(d)
            open(os.path.join(d, "summary.txt"), "w").write(
                "sequence poster_6dof  tag r5noise5hz3  run %s\nvariant orig\nbag %s\n" % (d, bag))
            return {"seq": "poster_6dof", "cond": "noise5hz", "index": 3, "run": d, "exit_code": 0, "summary": True}
        good = attempt("a", E.RA.bag_of("poster_6dof", "noise5hz"))
        wrong = attempt("b", E.RA.bag_of("poster_6dof", "clean"))
        assert E.planned(good) and not E.planned(wrong)
        assert E.planned(dict(wrong, as_planned=True))          # a stored field is used as recorded
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print("%d tests passed" % len(names))
