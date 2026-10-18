"""Correct the evaluation statistics: separate GPU quota billing, real wall time,
patch-generation (agent) time, and test (pytest) time per batch.

Reads quota snapshots, push/remote status, papermill execution logs, result durations,
traces, and test outputs. Writes artifacts/statistics_correction.md.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")
SUITES = ["single", "dev", "expand20", "expand20b", "promptv2", "noise20", "maxtok8k", "hygiene20", "val20", "ts28", "b28"]
PYTEST_RE = re.compile(r"in (\d+\.\d+)s")


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def exec_seconds(outputs):
    for name in ("execution.log",):
        log = outputs / name
        if not log.is_file():
            continue
        times = [float(m) for m in re.findall(r'"time":\s*([0-9.]+)', log.read_text(encoding="utf-8", errors="replace"))]
        if times:
            return max(times)
    return None


def batch_stats(suite):
    folder = ART / f"tool_control_v2_{suite}"
    outputs = ART / f"tool_control_v2_{suite}_outputs"
    results = load_json(outputs / "comparison_results.json")
    push = load_json(folder / "push.json")
    qb = load_json(folder / "quota_before.json")
    qa = load_json(folder / "quota_after.json")
    remote = load_json(outputs / "remote_status.json")
    if not results:
        return None
    n = len(results)
    wall_h = complete_h = None
    if push and remote:
        wall_h = (remote["checked_epoch"] - push["epoch"]) / 3600
    quota_h = (qa["used_hours"] - qb["used_hours"]) if (qb and qa) else None
    exec_s = exec_seconds(outputs)
    dur_sum = sum(r["duration_seconds"] for r in results)
    agent_elapsed = []
    pytest_sum = 0.0
    pytest_missing = 0
    for row in results:
        arm = f"tools_{row['max_tool_calls']}" if "repeat" not in row else f"tools_20_r{row['repeat']}"
        test_out = outputs / arm / row["task_id"] / "test_output.txt"
        if test_out.is_file():
            hits = PYTEST_RE.findall(test_out.read_text(encoding="utf-8", errors="replace"))
            if hits:
                pytest_sum += float(hits[-1])
            else:
                pytest_missing += 1
        trace = outputs / arm / "results" / "traces" / f"trace_{row['task_id']}.json"
        if trace.is_file():
            data = load_json(trace)
            last = 0.0
            for step in data.get("steps", []):
                for holder in ((step.get("observation") or {}), step):
                    extra = holder.get("extra") or {}
                    if isinstance(extra, dict) and extra.get("elapsed_s"):
                        last = max(last, float(extra["elapsed_s"]))
            if last:
                agent_elapsed.append(last)
    return {
        "suite": suite, "n": n, "quota_h": quota_h, "wall_h": wall_h, "exec_s": exec_s,
        "dur_sum_s": dur_sum, "agent_elapsed_sum_s": sum(agent_elapsed) if agent_elapsed else None,
        "agent_elapsed_n": len(agent_elapsed),
        "pytest_sum_s": pytest_sum, "pytest_missing": pytest_missing,
        "resolved": sum(1 for r in results if r["resolved"]),
    }


batches = [b for b in (batch_stats(s) for s in SUITES) if b]
lines = ["# 评测统计修正（配额计费 / 墙钟 / 补丁生成 / 测试时间分离）", "",
         "生成脚本 `scripts/eval_statistics.py`，数据源：push/quota 快照、remote_status、papermill 执行日志、result duration、trace、test_output。", ""]
lines.append("| suite | 任务运行 | resolved | 配额(h) | 墙钟 push→COMPLETE(h) | notebook 执行(s) | Σduration(s) | Σtrace周期(s) | Σpytest(s) | pytest缺失 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for b in batches:
    lines.append("| {suite} | {n} | {resolved} | {q} | {w} | {e} | {d} | {a} | {p} | {pm} |".format(
        suite=b["suite"], n=b["n"], resolved=b["resolved"],
        q="-" if b["quota_h"] is None else round(b["quota_h"], 3),
        w="-" if b["wall_h"] is None else round(b["wall_h"], 3),
        e="-" if b["exec_s"] is None else round(b["exec_s"]),
        d=round(b["dur_sum_s"]),
        a="-" if b["agent_elapsed_sum_s"] is None else round(b["agent_elapsed_sum_s"]),
        p=round(b["pytest_sum_s"], 1), pm=b["pytest_missing"]))
lines.append("")

if batches:
    ratios = [(b["suite"], b["quota_h"] / b["wall_h"]) for b in batches
              if b["quota_h"] and b["wall_h"] and b["wall_h"] > 0.05 and b["suite"] not in ("single", "maxtok8k")]
    if ratios:
        mean_ratio = sum(r for _, r in ratios) / len(ratios)
        lines.append(f"- 配额/墙钟比（排除状态写入延迟的 single 与 maxtok8k）：{', '.join(f'{s}={r:.2f}' for s, r in ratios)}；均值 {mean_ratio:.2f} → L4 按≈2× 计费；此前把配额小时当墙钟小时使用导致约 2× 高估。")
        lines.append("- 注：single 与 maxtok8k 的 `墙钟 push→COMPLETE` 含排队或重收集延迟（状态写入晚于实际完成），不代表运行时长，不参与比率与外推拟合。")
lines.append("- 口径说明：`Σduration` 来自 harness `result.duration_seconds`（含沙箱准备 + Agent 循环 + 补丁验证）；`Σtrace周期` 是 trace 最后一步的会话经过时间（纯 Agent 循环，不含验证）；`Σpytest` 为各任务最终 pytest 用时；比赛 12 小时上限包含 sandbox 设置、排除排队与补丁验证。")
lines.append("")

fit_pairs = {b["suite"]: b for b in batches}
if "promptv2" in fit_pairs and "expand20b" in fit_pairs and fit_pairs["promptv2"]["exec_s"] and fit_pairs["expand20b"]["exec_s"]:
    a, b_ = fit_pairs["promptv2"], fit_pairs["expand20b"]
    slope = (a["exec_s"] - b_["exec_s"]) / (a["n"] - b_["n"])
    intercept = b_["exec_s"] - b_["n"] * slope
    lines.append(f"## 修正后的 120 个隐藏任务外推（notebook 执行口径）")
    lines.append("")
    lines.append(f"- 线性拟合（promptv2 n={a['n']} 与 expand20b n={b_['n']}）：固定开销≈{intercept / 60:.1f} 分钟 + 每任务≈{slope / 60:.2f} 分钟（notebook 执行时间，含模型启动、Agent、验证与打包）。")
    total_120 = intercept + 120 * slope
    lines.append(f"- 120 任务外推（notebook 口径，含补丁验证）：≈{total_120 / 3600:.1f} 小时。")
    stage1_120 = intercept + 120 * (slope - (a["pytest_sum_s"] / a["n"]) - 20)
    lines.append(f"- 120 任务外推（Stage-1 口径，按估算剔除每任务 pytest {a['pytest_sum_s'] / a['n']:.0f}s 与约 20s 验证开销）：≈{stage1_120 / 3600:.1f} 小时。")
    lines.append("- 口径提示：上述外推依赖两个小批次（n=12 与 n=3）的线性拟合与估算的验证开销，隐藏集任务构成、并发度与容器准备方式未知，**只能作为量级参考，不能保证隐藏集实际耗时**；可比硬锚点：12h ÷ 120 任务 = 6 分钟/任务的上限，当前观测约 4.8 分钟/任务（notebook 口径）。")
    lines.append("- 修正说明：此前“≈15.4h 超限”的结论错误，原因是（a）把 L4 的 2× 计费配额小时直接当作墙钟小时（约 2× 高估），（b）未区分补丁验证时间（不计入 12 小时）。")
lines.append("")
(ART / "statistics_correction.md").write_text("\n".join(lines), encoding="utf-8")

for b in batches:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})
print()
print("report:", (ART / "statistics_correction.md").resolve())