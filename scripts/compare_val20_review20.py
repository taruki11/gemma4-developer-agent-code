"""review20 vs val20 comparison: outcomes plus behavioral audit metrics.

Behavioral metrics (never the model's own claim):
- review ran: a run_command with `git diff`/`git status` and the call index (in time = <= 16)
- audit-driven additions: source edits after the first review call; new files added post-review
- failed-edit recovery: file with a failed edit that later receives a successful edit
- coverage: distinct source files edited per run (vs baseline), for the multi-touchpoint tasks

Writes artifacts/review20_vs_val20.md.
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")
SCRATCH = ("repro", "test_", "check_", "verify_", "debug", "/tmp/")


def load(name):
    out = ART / f"tool_control_v2_{name}_outputs"
    rows = json.loads((out / "comparison_results.json").read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in rows}, out


def run_behavior(out, tid):
    trace = out / "tools_20" / "results" / "traces" / f"trace_{tid}.json"
    if not trace.is_file():
        return None
    data = json.loads(trace.read_text(encoding="utf-8"))
    idx = 0
    review_idx = None
    failed_files = set()
    ok_files = set()
    edits_after_review = 0
    files_after_review = set()
    files_before_review = set()
    for s in data.get("steps", []):
        for tc in s.get("tool_calls") or []:
            idx += 1
            fn = str(tc.get("function_name") or "")
            args = tc.get("arguments") or {}
            obs = str((s.get("observation") or {}).get("content") or "")
            if fn == "run_command":
                cmd = str(args.get("command") or "")
                if re.search(r"\bgit\s+(diff|status)", cmd) and review_idx is None:
                    review_idx = idx
            if fn in ("edit_file", "write_file"):
                fp = str(args.get("filepath") or args.get("file_path") or args.get("path") or "")
                scratch = any(m in fp for m in SCRATCH)
                if not scratch:
                    if review_idx is not None:
                        edits_after_review += 1
                        files_after_review.add(fp)
                    else:
                        files_before_review.add(fp)
                    if '"status": "ok"' in obs or obs.strip() == "":
                        ok_files.add(fp)
                    else:
                        failed_files.add(fp)
    recovery = failed_files & ok_files
    turns = []
    prev = 0.0
    completion = 0
    for s in data.get("steps", []):
        extra = s.get("extra") or {}
        elapsed = extra.get("elapsed_s") or prev
        turns.append(max(elapsed - prev, 0))
        prev = elapsed
    completion = data.get("final_metrics", {}).get("total_completion_tokens", 0)
    return {
        "review_idx": review_idx,
        "in_time": review_idx is not None and review_idx <= 16,
        "edits_after_review": edits_after_review,
        "new_files_after_review": sorted(files_after_review - files_before_review),
        "failed_edit_files": sorted(failed_files),
        "recovered_files": sorted(recovery),
        "source_files": sorted(ok_files | failed_files | files_after_review),
        "turns": len(turns),
        "max_turn": round(max(turns)) if turns else 0,
        "completion": completion,
    }


V, vout = load("val20")
R, rout = load("review20")
tasks = sorted(V)
base_n, cand_n = sum(V[t]["resolved"] for t in tasks), sum(R[t]["resolved"] for t in tasks)
gained = [t for t in tasks if R[t]["resolved"] and not V[t]["resolved"]]
lost = [t for t in tasks if V[t]["resolved"] and not R[t]["resolved"]]

lines = ["# review20 vs val20 对照（同 20 题、同 20 工具/5 分钟；唯一差异 = 提交前 diff 审查协议）", ""]
lines.append(f"- 基线 val20：**{base_n}/20**；候选 review20：**{cand_n}/20**（{cand_n - base_n:+d}）")
lines.append(f"- 获得：{gained or '无'}；丢失：{lost or '无'}")
lines.append("")
lines.append("| 任务 | val20 | review20 | 审查(调用#) | 审查后编辑 | 审查后新增文件 |")
lines.append("|---|---|---|---|---|---|")
behav = {}
for t in tasks:
    b = run_behavior(rout, t)
    behav[t] = b
    ridx = b["review_idx"] if b else None
    after = b["edits_after_review"] if b else 0
    newf = ",".join(f.split("/")[-1] for f in b["new_files_after_review"]) if b else ""
    star = " ✅" if b and b["in_time"] else ("" if b is None else " ⏰")
    lines.append(f"| {t} | {'✓' if V[t]['resolved'] else '✗'} | {'✓' if R[t]['resolved'] else '✗'} | {ridx if ridx else '未运行'}{star} | {after} | {newf} |")
lines.append("")

n_review = sum(1 for b in behav.values() if b and b["review_idx"])
n_intime = sum(1 for b in behav.values() if b and b["in_time"])
n_gapfill = sum(1 for b in behav.values() if b and b["edits_after_review"] > 0)
n_newfile = [t for t, b in behav.items() if b and b["new_files_after_review"]]
n_recover = [t for t, b in behav.items() if b and b["recovered_files"]]
lines.append("## 行为指标（以 trace 中的实际动作为准）")
lines.append("")
lines.append(f"- 实际执行过 `git diff/status` 的运行：**{n_review}/20**（其中按时（≤16 次调用）**{n_intime}**）")
lines.append(f"- 审查后又发生源码编辑的运行：**{n_gapfill}/20**")
lines.append(f"- 审查后编辑了**新文件**（此前未编辑过）的运行：{len(n_newfile)}——{'、'.join(n_newfile) or '无'}")
lines.append(f"- 失败编辑后同文件成功补齐的运行：{len(n_recover)}——{'、'.join(n_recover) or '无'}")
lines.append("")
cross = {"review_yes_pass": 0, "review_yes_fail": 0, "review_no_pass": 0, "review_no_fail": 0}
for t, b in behav.items():
    key = ("review_yes_" if b and b["review_idx"] else "review_no_") + ("pass" if R[t]["resolved"] else "fail")
    cross[key] += 1
lines.append(f"- 交叉：审查运行通过 {cross['review_yes_pass']}、未通过 {cross['review_yes_fail']}；未审查运行通过 {cross['review_no_pass']}、未通过 {cross['review_no_fail']}")
multi = ["fastapi_13713", "fastapi_13786", "fastapi_14099", "rich_3063"]
lines.append("")
lines.append("## 多触点任务的覆盖对比（失败复盘的 4 个'修改未完成'）")
lines.append("")
lines.append("| 任务 | val20 编辑的源文件 | review20 编辑的源文件 | 结果变化 |")
lines.append("|---|---|---|---|")
for t in multi:
    b = behav.get(t)
    bfiles = ", ".join(f.split("/")[-1] for f in (b["source_files"] if b else [])) or "-"
    base_patch = (vout / "tools_20" / t / "agent.patch")
    vfiles = re.findall(r"diff --git a/(\S+)", base_patch.read_text(encoding="utf-8", errors="replace")) if base_patch.is_file() else []
    vfiles = ", ".join(f.split("/")[-1] for f in vfiles) or "（空补丁）"
    lines.append(f"| {t} | {vfiles} | {bfiles} | {'✓ 转通过' if R[t]['resolved'] and not V[t]['resolved'] else ('✗ 仍失败' if not R[t]['resolved'] else '—')} |")
lines.append("")
bt = [b["turns"] for b in behav.values() if b]
lines.append("## 机制侧写")
lines.append("")
lines.append(f"- 回合数：val20 平均 {statistics.mean(b['turns'] for b in [run_behavior(vout, t) for t in tasks] if b):.1f} → review20 平均 {statistics.mean(bt):.1f}")
lines.append(f"- completion tokens：val20 {statistics.mean(b['completion'] for b in [run_behavior(vout, t) for t in tasks] if b):.0f} → review20 {statistics.mean(b['completion'] for b in behav.values() if b):.0f}")
lines.append("")
(ART / "review20_vs_val20.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))