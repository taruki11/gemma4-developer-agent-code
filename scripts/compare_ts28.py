"""ts28 vs val20 comparison on the 10 all-three-failed tasks, with treatment and behavior metrics.

Writes artifacts/ts28_vs_val20.md and applies the pre-registered decision rule.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")
TASKS = ["fastapi_13713", "fastapi_13786", "fastapi_13920", "fastapi_14099", "fastapi_14246",
         "requests_7328", "rich_3052", "rich_3061", "rich_3063", "rich_3105"]
SCRATCH = ("repro", "test_", "check_", "verify_", "debug", "/tmp/")


def load(name):
    out = ART / f"tool_control_v2_{name}_outputs"
    rows = json.loads((out / "comparison_results.json").read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in rows}, out


def analyze(out, tid, arm):
    trace = out / arm / "results" / "traces" / f"trace_{tid}.json"
    if not trace.is_file():
        return None
    data = json.loads(trace.read_text(encoding="utf-8"))
    author = None
    idx = 0
    calls = {"fixer": 0, "auditor": 0, "other": 0}
    first_auditor = None
    submit_author = None
    auditor_edits = 0
    auditor_files = set()
    for s in data.get("steps", []):
        a = (s.get("extra") or {}).get("author")
        if a:
            author = "auditor" if a == "auditor" else ("fixer" if a in ("fixer", "swe_two_stage_agent") else a)
        for tc in s.get("tool_calls") or []:
            idx += 1
            who = author if author in calls else "other"
            calls[who] += 1
            fn = str(tc.get("function_name") or "")
            if who == "auditor":
                if first_auditor is None:
                    first_auditor = idx
                if fn in ("edit_file", "write_file"):
                    fp = str((tc.get("arguments") or {}).get("filepath") or (tc.get("arguments") or {}).get("file_path") or "")
                    if not any(m in fp for m in SCRATCH):
                        auditor_edits += 1
                        auditor_files.add(fp)
            if fn == "submit_patch" and submit_author is None:
                submit_author = who
    return {"calls": calls, "first_auditor": first_auditor, "submit_author": submit_author,
            "auditor_edits": auditor_edits, "auditor_files": sorted(auditor_files), "total": idx}


def main():
    V, _ = load("val20")
    T, tout = load("ts28")
    arm = "tools_28"

    lines = ["# ts28 vs val20 对照(10 个三连败任务;两阶段结构 + 6.5 分钟/28 调用)", ""]
    tcfg = json.loads((tout / arm / "eval_config.json").read_text(encoding="utf-8"))
    sub_cfg = tout / arm / "submission_config"
    agent_yaml = (sub_cfg / "agent.yaml").read_text(encoding="utf-8")
    eval_yaml = (sub_cfg / "eval_config.yaml").read_text(encoding="utf-8")
    lines.append(f"- 处理核实:arm eval_config = {json.dumps({k: tcfg.get(k) for k in ('max_tool_calls', 'max_time_minutes', 'timeout_seconds')})};"
                 f" 提交内 agent.yaml SequentialAgent={'agent_class: SequentialAgent' in agent_yaml};"
                 f" 提交内 eval_config.yaml 28/6.5={'max_tool_calls: 28' in eval_yaml and 'max_time_minutes: 6.5' in eval_yaml}")
    lines.append("")

    converted = [t for t in TASKS if T[t]["resolved"]]
    n_conv = len(converted)
    lines.append(f"- **转换:{n_conv}/10**(val20 基线:0/10;review20、twostage20 同 10 题也全败)")
    lines.append(f"- 通过任务:{converted or '无'}")
    lines.append("")
    lines.append("| 任务 | val20 | ts28 | 审计启动 | fixer/auditor 调用 | 审计编辑 | 提交者 |")
    lines.append("|---|---|---|---|---|---|---|")
    behav = {}
    for t in TASKS:
        b = analyze(tout, t, arm)
        behav[t] = b
        lines.append(f"| {t} | {'✓' if V[t]['resolved'] else '✗'} | {'✓' if T[t]['resolved'] else '✗'} | "
                     f"{b['first_auditor'] if b and b['first_auditor'] else ('—' if b else 'n/a')} | "
                     f"{(str(b['calls']['fixer']) + '/' + str(b['calls']['auditor'])) if b else 'n/a'} | "
                     f"{b['auditor_edits'] if b else 0} | {b['submit_author'] or '无' if b else 'n/a'} |")
    lines.append("")
    n_aud = sum(1 for b in behav.values() if b and b["first_auditor"])
    edits_runs = [t for t, b in behav.items() if b and b["auditor_edits"] > 0]
    lines.append("## 行为指标")
    lines.append("")
    lines.append(f"- 审计运行:{n_aud}/10(twostage20 为 12/20)")
    lines.append(f"- 审计阶段编辑的运行:{len(edits_runs)}/10 —— {', '.join(edits_runs) or '无'}(twostage20 为 0/20)")
    if n_aud:
        lines.append(f"- auditor 平均调用 {statistics.mean([b['calls']['auditor'] for b in behav.values() if b]):.1f}(twostage20: 2.3);"
                     f" fixer 平均 {statistics.mean([b['calls']['fixer'] for b in behav.values() if b]):.1f}(twostage20: 17.4)")
    submitters = {}
    for b in behav.values():
        if b and b["submit_author"]:
            submitters[b["submit_author"]] = submitters.get(b["submit_author"], 0) + 1
    lines.append(f"- 提交者:{submitters or '无(自动回收)'}")
    conv_with_edits = [t for t in converted if behav.get(t) and behav[t]["auditor_edits"] > 0]
    lines.append("")
    if n_conv >= 3 and len(conv_with_edits) >= 2:
        verdict = "**机制+成绩双达标 → 采纳,进入全量确认(配额刷新后 20 题)**"
    elif n_conv <= 1:
        verdict = "**无转化 → 否决该方向(预算提高不足以救活审计)**"
    else:
        verdict = "**2 个转化 → 不确定,按预注册规则待配额刷新后重复**"
    lines.append(f"- 预注册判读:{verdict}")
    lines.append("")
    (ART / "ts28_vs_val20.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


main()