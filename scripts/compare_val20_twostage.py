"""twostage20 vs val20 comparison: outcomes plus fixer/auditor behavior metrics.

Behavioral metrics (trace evidence, not self-report):
- treatment: arm agent.yaml is a SequentialAgent; fixer.yaml lacks submit_patch; auditor prompt present
- auditor ran: any trace step authored by the auditor
- who submitted: author of the submit_patch call
- edits after the first auditor step (gap-filling by the auditor)
- call split between fixer and auditor

Writes artifacts/twostage20_vs_val20.md.
"""
import json
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


def author_of(step):
    extra = step.get("extra") or {}
    return extra.get("author")


def analyze(out, tid):
    trace = out / "tools_20" / "results" / "traces" / f"trace_{tid}.json"
    if not trace.is_file():
        return None
    data = json.loads(trace.read_text(encoding="utf-8"))
    current_author = None
    calls = {"fixer": 0, "auditor": 0, "other": 0}
    first_auditor_call = None
    submit_author = None
    edits_after_auditor = 0
    files_after_auditor = set()
    idx = 0
    for s in data.get("steps", []):
        a = author_of(s)
        if a:
            current_author = "auditor" if a == "auditor" else ("fixer" if a in ("fixer", "swe_baseline_agent", "swe_two_stage_agent") else a)
        for tc in s.get("tool_calls") or []:
            idx += 1
            who = current_author if current_author in calls else "other"
            calls[who] += 1
            if who == "auditor" and first_auditor_call is None:
                first_auditor_call = idx
            fn = str(tc.get("function_name") or "")
            if fn == "submit_patch" and submit_author is None:
                submit_author = who
            if who == "auditor" and fn in ("edit_file", "write_file"):
                fp = str((tc.get("arguments") or {}).get("filepath") or (tc.get("arguments") or {}).get("file_path") or (tc.get("arguments") or {}).get("path") or "")
                if not any(m in fp for m in SCRATCH):
                    edits_after_auditor += 1
                    files_after_auditor.add(fp)
    return {"calls": calls, "first_auditor_call": first_auditor_call, "submit_author": submit_author,
            "auditor_edits": edits_after_auditor, "auditor_files": sorted(files_after_auditor),
            "total_calls": idx}


def main():
    V, vout = load("val20")
    T, tout = load("twostage20")
    tasks = sorted(V)
    base_n, cand_n = sum(V[t]["resolved"] for t in tasks), sum(T[t]["resolved"] for t in tasks)
    gained = [t for t in tasks if T[t]["resolved"] and not V[t]["resolved"]]
    lost = [t for t in tasks if V[t]["resolved"] and not T[t]["resolved"]]

    lines = ["# twostage20 vs val20 对照(同 20 题、同 20 工具/5 分钟;唯一差异 = fixer->auditor 两阶段结构)", ""]
    lines.append(f"- 基线 val20:**{base_n}/20**;候选 twostage20:**{cand_n}/20**({cand_n - base_n:+d})")
    lines.append(f"- 获得:{gained or '无'};丢失:{lost or '无'}")
    lines.append("")
    lines.append("| 任务 | val20 | twostage20 | 审计启动(调用#) | 审计后编辑 | 提交者 | fixer/auditor 调用 |")
    lines.append("|---|---|---|---|---|---|---|")
    behav = {}
    for t in tasks:
        b = analyze(tout, t)
        behav[t] = b
        idx_a = (f"{b['first_auditor_call']}" if b and b["first_auditor_call"] else ("—" if b else "n/a"))
        ae = b["auditor_edits"] if b else 0
        sa = b["submit_author"] or "无" if b else "n/a"
        split = f"{b['calls']['fixer']}/{b['calls']['auditor']}" if b else "n/a"
        lines.append(f"| {t} | {'✓' if V[t]['resolved'] else '✗'} | {'✓' if T[t]['resolved'] else '✗'} | {idx_a} | {ae} | {sa} | {split} |")
    lines.append("")

    n_auditor = sum(1 for b in behav.values() if b and b["first_auditor_call"])
    n_auditor_edit = [t for t, b in behav.items() if b and b["auditor_edits"] > 0]
    submitters = {}
    for b in behav.values():
        if b and b["submit_author"]:
            submitters[b["submit_author"]] = submitters.get(b["submit_author"], 0) + 1
    lines.append("## 行为指标(trace 证据)")
    lines.append("")
    lines.append(f"- **审计 Agent 实际运行**:{n_auditor}/20")
    lines.append(f"- **审计阶段发生编辑**:{len(n_auditor_edit)}/20 —— {'、'.join(n_auditor_edit) or '无'}")
    lines.append(f"- 提交者分布:{submitters or '无(自动回收)'}")
    fixer_calls = [b["calls"]["fixer"] for b in behav.values() if b]
    auditor_calls = [b["calls"]["auditor"] for b in behav.values() if b]
    lines.append(f"- 调用拆分:fixer 平均 {statistics.mean(fixer_calls):.1f},auditor 平均 {statistics.mean(auditor_calls):.1f}(总数均值 {statistics.mean([b['total_calls'] for b in behav.values() if b]):.1f})")
    cross = {"aud_pass": 0, "aud_fail": 0, "noaud_pass": 0, "noaud_fail": 0}
    for t, b in behav.items():
        key = ("aud_" if b and b["first_auditor_call"] else "noaud_") + ("pass" if T[t]["resolved"] else "fail")
        cross[key] += 1
    lines.append(f"- 交叉:审计运行通过 {cross['aud_pass']}、未通过 {cross['aud_fail']};审计未运行通过 {cross['noaud_pass']}、未通过 {cross['noaud_fail']}")
    lines.append("")
    multi = ["fastapi_13713", "fastapi_13786", "fastapi_14099", "rich_3063"]
    lines.append("## 多触点任务(failure_review 的 4 个'修改未完成')")
    lines.append("")
    lines.append("| 任务 | val20 源文件 | twostage20 源文件 | 审计后新增 | 结果 |")
    lines.append("|---|---|---|---|---|")
    import re
    for t in multi:
        p = vout / "tools_20" / t / "agent.patch"
        vf = re.findall(r"diff --git a/(\S+)", p.read_text(encoding="utf-8", errors="replace")) if p.is_file() else []
        tp = tout / "tools_20" / t / "agent.patch"
        tf = re.findall(r"diff --git a/(\S+)", tp.read_text(encoding="utf-8", errors="replace")) if tp.is_file() else []
        b = behav.get(t)
        new = ",".join(f.split("/")[-1] for f in (b["auditor_files"] if b else [])) or "—"
        res = "✓ 转通过" if T[t]["resolved"] and not V[t]["resolved"] else ("✗ 仍失败" if not T[t]["resolved"] else "—")
        lines.append(f"| {t} | {', '.join(f.split('/')[-1] for f in vf) or '(空)'} | {', '.join(f.split('/')[-1] for f in tf) or '(空)'} | {new} | {res} |")
    lines.append("")
    (ART / "twostage20_vs_val20.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


main()