"""Verify the raw artifacts of val20 / twostage20 / ts28 and the frozen baseline package.

Reads only local artifacts. Writes artifacts/state_verification_2026-10-14.json.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")
report = {"date": "2026-10-14", "verified_from": "local raw artifacts only"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def suite_state(name, budget_expect):
    out = ART / f"tool_control_v2_{name}_outputs"
    folder = ART / f"tool_control_v2_{name}"
    rows = json.loads((out / "comparison_results.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "comparison_summary.json").read_text(encoding="utf-8"))
    qb = json.loads((folder / "quota_before.json").read_text(encoding="utf-8"))
    qa = json.loads((folder / "quota_after.json").read_text(encoding="utf-8"))
    push = json.loads((folder / "push.json").read_text(encoding="utf-8"))
    status = json.loads((out / "remote_status.json").read_text(encoding="utf-8"))
    arm_name = arm_of(budget_expect)
    arm_cfg = json.loads((out / arm_name / "eval_config.json").read_text(encoding="utf-8"))
    cfg = {k: arm_cfg.get(k) for k in ("max_tool_calls", "max_time_minutes", "timeout_seconds")}
    patches = []
    for r in sorted(rows, key=lambda x: x["task_id"]):
        p = out / arm_name / r["task_id"] / "agent.patch"
        patches.append({
            "task": r["task_id"], "resolved": r["resolved"], "exit": r["test_exit_code"],
            "tool_calls": r["tool_calls"], "duration_s": round(r["duration_seconds"]),
            "patch_chars": r["patch_chars"],
            "patch_sha256": sha256(p) if p.is_file() else None,
        })
    state = {
        "kernel": push["stdout"].strip().split(" ")[0:6],
        "remote_status": status["status"],
        "arm": arm_name,
        "config_verified": cfg,
        "config_matches_expected": cfg == budget_expect,
        "quota_before_remaining_h": qb["remaining_hours"],
        "quota_after_remaining_h": qa["remaining_hours"],
        "summary": summary,
        "resolved": sum(1 for r in rows if r["resolved"]),
        "patches": patches,
    }
    return state


def arm_of(budget):
    for key in ("max_tool_calls",):
        return f"tools_{budget[key]}"
    return None


report["val20"] = suite_state("val20", {"max_tool_calls": 20, "max_time_minutes": 5, "timeout_seconds": 300})
report["twostage20"] = suite_state("twostage20", {"max_tool_calls": 20, "max_time_minutes": 5, "timeout_seconds": 300})
report["ts28"] = suite_state("ts28", {"max_tool_calls": 28, "max_time_minutes": 6.5, "timeout_seconds": 300})

# structural evidence
t_layout = json.loads((ART / "tool_control_v2_twostage20_outputs" / "two_stage_layout.json").read_text(encoding="utf-8"))
ts_layout = json.loads((ART / "tool_control_v2_ts28_outputs" / "two_stage_layout.json").read_text(encoding="utf-8"))
arm_root = (ART / "tool_control_v2_twostage20_outputs" / "tools_20" / "submission_config" / "agent.yaml").read_text(encoding="utf-8")
arm_root28 = (ART / "tool_control_v2_ts28_outputs" / "tools_28" / "submission_config" / "agent.yaml").read_text(encoding="utf-8")
report["structure_evidence"] = {
    "twostage20_root_is_sequential": "agent_class: SequentialAgent" in arm_root,
    "ts28_root_is_sequential": "agent_class: SequentialAgent" in arm_root28,
    "twostage20_layout_sha": t_layout["root_agent_sha256_after"],
    "ts28_layout_sha": ts_layout["root_agent_sha256_after"],
    "twostage20_sub_eval_config": (ART / "tool_control_v2_twostage20_outputs" / "tools_20" / "submission_config" / "eval_config.yaml").read_text(encoding="utf-8").strip().splitlines()[0],
    "ts28_sub_eval_config": (ART / "tool_control_v2_ts28_outputs" / "tools_28" / "submission_config" / "eval_config.yaml").read_text(encoding="utf-8").strip().splitlines()[0],
}

# frozen baseline package
zip_path = ART / "submission_candidate.zip"
report["baseline_package"] = {
    "zip": str(zip_path),
    "bytes": zip_path.stat().st_size,
    "sha256": sha256(zip_path),
    "recorded_sha256": "854cd413bcd79c6ba312ad6d4837fc7ee04326f9febb4a3c918dec77e95cc605",
}
report["baseline_package"]["matches_record"] = (
    report["baseline_package"]["sha256"] == report["baseline_package"]["recorded_sha256"])

(ART / "state_verification_2026-10-14.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

for name in ("val20", "twostage20", "ts28"):
    s = report[name]
    print(f"== {name}: resolved={s['resolved']}/{len(s['patches'])} config={s['config_verified']} matches={s['config_matches_expected']} status={s['remote_status']}")
print("structure:", json.dumps(report["structure_evidence"], ensure_ascii=False))
print("baseline:", report["baseline_package"]["sha256"][:16], "matches_record:", report["baseline_package"]["matches_record"])