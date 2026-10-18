"""Compare the val20 baseline and the think1k candidate: outcomes, per-repo, mechanism metrics.

Applies the decision rule pre-registered in artifacts/val20_analysis.md.
Read-only; writes artifacts/think1k_vs_val20.md.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")


def load_suite(name):
    out = ART / f"tool_control_v2_{name}_outputs"
    rows = json.loads((out / "comparison_results.json").read_text(encoding="utf-8"))
    traces = out / "tools_20" / "results" / "traces"
    profile = {}
    for f in traces.glob("trace_*.json"):
        tid = f.stem.replace("trace_", "")
        data = json.loads(f.read_text(encoding="utf-8"))
        prev = 0.0
        turn_walls = []
        for s in data["steps"]:
            extra = s.get("extra") or {}
            elapsed = extra.get("elapsed_s") or prev
            d = max(elapsed - prev, 0)
            prev = elapsed
            if s["source"] == "agent":
                turn_walls.append(d)
        profile[tid] = {
            "turns": len(turn_walls),
            "turn_wall_sum": sum(turn_walls),
            "max_turn": max(turn_walls) if turn_walls else 0,
            "completion": data["final_metrics"]["total_completion_tokens"],
            "prompt": data["final_metrics"]["total_prompt_tokens"],
        }
    return rows, profile


base_rows, base_prof = load_suite("val20")
cand_rows, cand_prof = load_suite("think1k")
base_res = {r["task_id"]: bool(r["resolved"]) for r in base_rows}
cand_res = {r["task_id"]: bool(r["resolved"]) for r in cand_rows}
tasks = sorted(base_res)

base_n = sum(base_res.values())
cand_n = sum(cand_res.values())
gained = [t for t in tasks if cand_res[t] and not base_res[t]]
lost = [t for t in tasks if base_res[t] and not cand_res[t]]


def group_stats(prof, res, value):
    vals = [prof[t] for t in prof if res.get(t) == value]
    if not vals:
        return None
    return {
        "n": len(vals),
        "turns": statistics.mean(v["turns"] for v in vals),
        "turn_wall": statistics.mean(v["turn_wall_sum"] for v in vals),
        "max_turn": statistics.mean(v["max_turn"] for v in vals),
        "completion": statistics.mean(v["completion"] for v in vals),
    }


lines = ["# think1k vs val20 对照（同 20 题、同 20 工具/5 分钟预算）", ""]
lines.append(f"- 基线 val20：**{base_n}/20**；候选 think1k：**{cand_n}/20**（+{cand_n - base_n}）")
lines.append(f"- 获得：{gained or '无'}；丢失：{lost or '无'}")
lines.append("")
lines.append("| 任务 | val20 | think1k | | 任务 | val20 | think1k |")
lines.append("|---|---|---|---|---|---|---|")
half = (len(tasks) + 1) // 2
for i in range(half):
    a = tasks[i]
    b = tasks[i + half] if i + half < len(tasks) else None
    cell_b = f"| {b} | {'✓' if base_res[b] else '✗'} | {'✓' if cand_res[b] else '✗'} |" if b else "| | | |"
    lines.append(f"| {a} | {'✓' if base_res[a] else '✗'} | {'✓' if cand_res[a] else '✗'} | {cell_b}")
lines.append("")
lines.append("## 机制指标（组内平均）")
lines.append("")
lines.append("| 组 | 回合数 | 回合墙钟(s) | 单回合最大(s) | completion tokens |")
lines.append("|---|---|---|---|---|")
for label, prof, res, value in [("val20 通过", base_prof, base_res, True), ("val20 未过", base_prof, base_res, False),
                                 ("think1k 通过", cand_prof, cand_res, True), ("think1k 未过", cand_prof, cand_res, False)]:
    g = group_stats(prof, res, value)
    if g:
        lines.append(f"| {label} | {g['turns']:.1f} | {g['turn_wall']:.0f} | {g['max_turn']:.0f} | {g['completion']:.0f} |")

all_base_max = statistics.median(v["max_turn"] for v in base_prof.values())
all_cand_max = statistics.median(v["max_turn"] for v in cand_prof.values())
all_base_comp = statistics.mean(v["completion"] for v in base_prof.values())
all_cand_comp = statistics.mean(v["completion"] for v in cand_prof.values())
mech_ok = all_cand_max < 50 and all_cand_comp <= 0.75 * all_base_comp
lines.append("")
lines.append(f"- 全体单回合最大中位数：val20 {all_base_max:.0f}s → think1k {all_cand_max:.0f}s（目标 < 50s）")
lines.append(f"- 全体 completion 均值：val20 {all_base_comp:.0f} → think1k {all_cand_comp:.0f}（目标降幅 ≥ 25%）")
lines.append(f"- 机制成立：**{mech_ok}**")
verdict = ("机制与成绩双达标 → 采纳" if mech_ok and cand_n >= 13 else
           "成绩无提升 → 否决" if cand_n <= 11 else
           "机制成立但成绩 11-12 → 对硬题做 ≥3 次重复后再判" if mech_ok else
           "机制未成立（旋钮在本栈无效/无咬合） → 该方向否决")
lines.append(f"- **预注册判读：{verdict}**")
lines.append("")
(ART / "think1k_vs_val20.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
