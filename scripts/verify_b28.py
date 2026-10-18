"""Verify the raw artifacts of the b28 batch (read-only on run outputs).

Checks, in the order the user asked for them:
  1. config consistency: arm eval_config.json budget vs submission_config/eval_config.yaml
     vs the pre-registered 28 / 6.5 / 50 / 300 (a mismatch makes the batch invalid, per prereg).
  2. completeness: 20 rows, one patch / test_output / result / trace per task.
  3. official test evidence: test_output.txt present and non-empty for every task.
  4. wall clock: kernel push epoch -> remote_status completion, and the 14400 s kernel cap.

Writes artifacts/b28_state_verification.json. Prints a short summary.
Exits non-zero only on a hard inconsistency (missing batch, wrong task count).
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")
OUT = ART / "tool_control_v2_b28_outputs"
FOLDER = ART / "tool_control_v2_b28"
ARM = "tools_28"
EXPECTED_BUDGET = {"tool_calls": 28, "time_minutes": 6.5, "turns": 50, "timeout_seconds": 300}
KERNEL_CAP_SECONDS = 14400
TASKS_EXPECTED = 20


def read_json(path, default=None):
    if not Path(path).is_file():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_submission_yaml(path):
    """Minimal read of the flat evaluation section (no yaml dependency)."""
    values = {}
    if not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for key, cast in (("max_tool_calls", int), ("max_time_minutes", float),
                          ("max_turns", int), ("timeout_seconds", int)):
            if stripped.startswith(f"{key}:"):
                try:
                    values[key] = cast(stripped.split(":", 1)[1].strip())
                except ValueError:
                    values[key] = stripped.split(":", 1)[1].strip()
    return values


def main():
    if not (OUT / "comparison_results.json").is_file():
        print("b28 outputs not collected yet; run verify_b28.py after the batch finishes.")
        return 1

    rows = read_json(OUT / "comparison_results.json", [])
    summary = read_json(OUT / "comparison_summary.json", {})
    remote = read_json(OUT / "remote_status.json", {})
    arm_cfg = read_json(OUT / ARM / "eval_config.json", {})
    budget = (arm_cfg or {}).get("budget", {}) or {}
    yaml_cfg = parse_submission_yaml(OUT / ARM / "submission_config" / "eval_config.yaml")

    # 1. config consistency
    arm_view = {
        "max_tool_calls": budget.get("tool_calls"),
        "max_time_minutes": budget.get("time_minutes"),
        "max_turns": budget.get("turns"),
        "timeout_seconds": (arm_cfg or {}).get("timeout_seconds"),
    }
    mismatches = []
    for key, expected in (("max_tool_calls", 28), ("max_time_minutes", 6.5),
                          ("max_turns", 50), ("timeout_seconds", 300)):
        if arm_view.get(key) != expected:
            mismatches.append(f"arm {key}={arm_view.get(key)!r} != {expected}")
        if yaml_cfg.get(key) != expected:
            mismatches.append(f"file {key}={yaml_cfg.get(key)!r} != {expected}")
        if yaml_cfg.get(key) is not None and arm_view.get(key) != yaml_cfg.get(key):
            mismatches.append(f"arm/file differ on {key}: {arm_view.get(key)!r} vs {yaml_cfg.get(key)!r}")
    config_ok = not mismatches

    # 2/3. completeness of per-task evidence
    # 官方测试只在有补丁时才跑得起来:空补丁 + Agent 侧终止 => 不存在 test_output 是预期结果,
    # 不是"证据缺失";只有"有补丁却没有测试输出"才算真正的缺口。
    per_task = []
    for row in sorted(rows, key=lambda r: r["task_id"]):
        base = OUT / ARM / row["task_id"]
        patch = base / "agent.patch"
        test = base / "test_output.txt"
        trace = OUT / ARM / "results" / "traces" / f"trace_{row['task_id']}.json"
        patch_chars = row["patch_chars"]
        test_ok = test.is_file() and test.stat().st_size > 0
        unresolved_reason = None
        if not test_ok:
            result = read_json(base / "result.json", {}) or {}
            if patch_chars == 0:
                unresolved_reason = ("no patch produced -> official test never ran: "
                                     + str(result.get("error_message", ""))[:120])
            else:
                unresolved_reason = "PATCH BUT NO TEST OUTPUT (real evidence gap)"
        per_task.append({
            "task": row["task_id"],
            "resolved": bool(row["resolved"]),
            "test_exit_code": row["test_exit_code"],
            "tool_calls": row["tool_calls"],
            "duration_s": round(row["duration_seconds"], 1),
            "patch_chars": patch_chars,
            "has_result": (base / "result.json").is_file(),
            "has_patch": patch.is_file(),
            "has_test_output": test_ok,
            "test_not_applicable": patch_chars == 0 and not test_ok,
            "official_test_note": unresolved_reason,
            "has_trace": trace.is_file(),
            "max_tool_calls_field": row.get("max_tool_calls"),
        })
    hard_gaps = [t["task"] for t in per_task
                 if not (t["has_result"] and t["has_patch"] and t["has_trace"])]
    test_gaps = [t["task"] for t in per_task
                 if not t["has_test_output"] and not t["test_not_applicable"]]
    evidence_complete = not hard_gaps and not test_gaps
    missing = {"missing_result_patch_or_trace": hard_gaps,
               "patch_without_test_output": test_gaps,
               "no_patch_so_no_official_test": [t["task"] for t in per_task if t["test_not_applicable"]]}

    # 4. wall clock vs kernel cap
    push = read_json(FOLDER / "push.json", {}) or {}
    pushed_epoch = push.get("epoch")
    finished_epoch = (remote or {}).get("checked_epoch")
    wall_seconds = (finished_epoch - pushed_epoch) if (pushed_epoch and finished_epoch) else None
    qb = read_json(FOLDER / "quota_before.json", {}) or {}
    qa = read_json(FOLDER / "quota_after.json", {}) or {}

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "verified_from": "local raw artifacts only",
        "batch": "b28",
        "remote_status": (remote or {}).get("status"),
        "failure_message": (remote or {}).get("failure_message"),
        "kernel_cap_seconds": KERNEL_CAP_SECONDS,
        "config": {
            "expected": EXPECTED_BUDGET,
            "arm_eval_config": arm_view,
            "submission_eval_config_yaml": yaml_cfg,
            "consistent": config_ok,
            "mismatches": mismatches,
        },
        "counts": {
            "rows": len(rows),
            "resolved": sum(1 for r in rows if r["resolved"]),
            "tasks_with_28_calls_field": sum(1 for r in rows if r.get("max_tool_calls") == 28),
            "expected_tasks": TASKS_EXPECTED,
        },
        "evidence_complete": evidence_complete,
        "missing_evidence": {k: v for k, v in missing.items() if v},
        "wall_clock": {
            "push_epoch": pushed_epoch,
            "finished_epoch": finished_epoch,
            "wall_seconds": round(wall_seconds, 1) if wall_seconds else None,
            "wall_hours": round(wall_seconds / 3600, 3) if wall_seconds else None,
            "within_kernel_cap": (wall_seconds < KERNEL_CAP_SECONDS) if wall_seconds else None,
        },
        "quota": {
            "before_remaining_h": qb.get("remaining_hours"),
            "after_remaining_h": qa.get("remaining_hours"),
            "consumed_h": round(qb.get("remaining_hours", 0) - qa.get("remaining_hours", 0), 4)
            if qb.get("remaining_hours") is not None and qa.get("remaining_hours") is not None else None,
        },
        "summary": summary,
        "per_task": per_task,
    }
    (ART / "b28_state_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"status        : {report['remote_status']} failure={report['failure_message']!r}")
    print(f"config        : arm={arm_view} file={yaml_cfg}")
    print(f"config一致    : {config_ok}" + ("" if config_ok else f" -> {mismatches}"))
    print(f"rows/resolved : {report['counts']['rows']} / {report['counts']['resolved']}")
    print(f"证据齐全      : {evidence_complete}" + ("" if evidence_complete else f" -> {report['missing_evidence']}"))
    print(f"墙钟/上限     : {report['wall_clock']['wall_hours']}h / {KERNEL_CAP_SECONDS/3600:.1f}h")
    print(f"配额消耗      : {report['quota']['consumed_h']}h (剩 {report['quota']['after_remaining_h']}h)")
    return 0 if (config_ok and len(rows) == TASKS_EXPECTED) else 2


if __name__ == "__main__":
    sys.exit(main())
