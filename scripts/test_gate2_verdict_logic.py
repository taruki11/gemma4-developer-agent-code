"""Exercise the finalise logic against simulated partial runs.

The verdict code is the part of the notebook most likely to be silently wrong,
because it only runs at the very end of a GPU run. This test lifts the real
`HARD_GROUPS` and the real summarising code out of the generated notebook and runs
them against hand-built report states, so "skipped is not failed but also not a
pass" is checked here rather than discovered after a 2-hour run.
"""

import ast
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads(io.open(ROOT / "artifacts" / "training_gate2" / "smoke.ipynb",
                        encoding="utf-8").read())
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
final = cells[-1]

# Lift the real code: everything from HARD_GROUPS to the verdict assignment.
start = final.index("HARD_GROUPS = {")
end = final.index('report["finished_epoch"]')
verdict_src = final[start:end]
assert "HARD_GROUPS" in verdict_src and "report[\"verdict\"]" in verdict_src

# `rec`-style entries as the notebook writes them.
def ok(cid):
    return {cid: {"ok": True, "value": 1, "detail": "pass"}}


def bad(cid, detail="measured the wrong thing"):
    return {cid: {"ok": False, "value": None, "detail": detail}}


def skipped(cid, why="skipped: load stage failed"):
    return {cid: {"ok": False, "value": None, "detail": why}}


ALL_A = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
ALL_B = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B7b"]
ALL_C = ["C1", "C2", "C3"]
ALL_D = ["D2", "D3", "D4"]
ALL_E = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]
ALL = ALL_A + ALL_B + ALL_C + ALL_D + ALL_E


def run(report):
    ns = {"report": report}
    exec(verdict_src, ns)  # noqa: S102 -- testing the notebook's own code
    return ns["summary"], ns["report"]["verdict"]


cases = []

# ---------------------------------------------------------------- all pass
checks = {}
for cid in ALL:
    checks.update(ok(cid))
cases.append(("everything passes", {"checks": checks,
                                    "skipped_stages": [],
                                    "blocked_by": None},
              True, []))

# ------------------------------------------------- nothing ran (load failed)
checks = {"A1": {"ok": False, "value": None,
                 "detail": "OutOfMemoryError: CUDA out of memory."}}
for cid in ALL[1:]:
    if cid in ("A2", "A3", "A4"):
        checks.update(skipped(cid, "skipped: load stage failed"))
    else:
        checks.update(skipped(cid, "skipped: %s not reached because load failed" % cid))
cases.append(("load failed, everything downstream skipped",
              {"checks": checks,
               "skipped_stages": ["lora", "train", "save_reload", "vllm_start",
                                  "e_checks"],
               "blocked_by": "load stage failed (see A1-A4 and errors[where=load])"},
              False, [], ALL_A[1:] + ["B3", "C1", "E3"]))  # A9 included via ALL_A

# ------------------------------------------------- training measured and wrong
checks = {}
for cid in ALL_A + ["B1", "B2"]:
    checks.update(ok(cid))
checks.update(bad("B3", "loss was NaN"))
for cid in ALL_B[3:] + ALL_C + ALL_D + ALL_E:
    checks.update(skipped(cid, "skipped: training stage failed"))
cases.append(("training ran and failed",
              {"checks": checks, "skipped_stages": ["save_reload", "vllm_start",
                                                    "e_checks"],
               "blocked_by": "training stage failed"}, False, ["B3"],
              ["C1", "E3"]))

# ------------------------------------ E3 passes but E4 control also differs
checks = {}
for cid in ALL:
    checks.update(ok(cid))
checks.update(bad("E4", "trained 0.0, control 0.31"))
cases.append(("adapter consumed but control also moved",
              {"checks": checks, "skipped_stages": [], "blocked_by": None},
              False, ["E4"]))

# ------------------------------------------- a check missing without a skip
checks = {}
for cid in ALL:
    checks.update(ok(cid))
del checks["D3"]
cases.append(("one check silently absent",
              {"checks": checks, "skipped_stages": [], "blocked_by": None},
              False, []))

failures = []
for case in cases:
    name, payload, want_pass, want_flag = case[:4]
    want_skipped = case[4] if len(case) > 4 else []
    report = dict(payload)
    report.setdefault("checks", {})
    summary, verdict = run(report)
    flagged = sorted({i for g in summary.values() for i in (g["failed"] + g["missing"])})
    skipped_ids = sorted({i for g in summary.values() for i in g["skipped"]})
    got_pass = verdict["pass"]
    ok_case = got_pass == want_pass
    if want_flag:
        ok_case = ok_case and set(want_flag).issubset(set(flagged))
    if want_skipped:
        ok_case = ok_case and set(want_skipped).issubset(set(skipped_ids))
    # A skipped check must never be counted as a failure: that conflation is the
    # specific misreading this whole mechanism exists to prevent.
    overlap = set(flagged) & set(skipped_ids)
    ok_case = ok_case and not overlap
    print("%-46s pass=%-5s failed/missing=%s skipped=%d"
          % (name, got_pass, flagged or "[]", len(skipped_ids)))
    if not ok_case:
        failures.append((name, want_pass, got_pass, flagged, want_flag,
                         want_skipped, sorted(overlap)))

# Explicitly: a skipped check must make the verdict FAIL.
report = {"checks": {}, "skipped_stages": ["e_checks"], "blocked_by": "x"}
for cid in ALL:
    report["checks"].update(ok(cid))
report["checks"].update(skipped("E3", "skipped: serving did not start"))
summary, verdict = run(report)
if verdict["pass"]:
    failures.append(("a skipped check still yields PASS", False, True, [], ["E3"]))
print("%-46s pass=%-5s (a skip must not pass)"
      % ("skipped check forces FAIL", verdict["pass"]))
if "E3" not in summary["E"]["skipped"]:
    failures.append(("skipped ids not reported apart", None, None, [], ["E3"]))

print()
if failures:
    print("VERDICT-LOGIC TEST: FAIL")
    for f in failures:
        print("  - %r" % (f,))
    sys.exit(1)
print("VERDICT-LOGIC TEST: PASS (%d scenarios)" % (len(cases) + 1))
