"""b28 vs val20 full adjudication: outcomes, config integrity, new-budget usage, 12h input.

Implements the b28_prereg.md metrics. Requires b28 outputs to exist; exits gracefully otherwise.
Writes artifacts/b28_vs_val20.md.
"""
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_class import classify  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")
SCRATCH = ("repro", "test_", "check_", "verify_", "debug", "/tmp/")
GRAPH_TOOLS = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}
PYTEST_RE = re.compile(r"in (\d+\.\d+)s")


def load(name):
    out = ART / f"tool_control_v2_{name}_outputs"
    rows = json.loads((out / "comparison_results.json").read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in rows}, out


def attribute(row, out, arm):
    """失败归因：与 run_tool_experiments.py 共用同一套规则，避免口径漂移。"""
    base = out / arm / row["task_id"]
    test_out = base / "test_output.txt"
    patch = base / "agent.patch"
    return classify(
        row,
        test_out.read_text(encoding="utf-8", errors="replace") if test_out.is_file() else "",
        patch.read_text(encoding="utf-8", errors="replace") if patch.is_file() else "",
    )


def pick(rows, out, arm, kind=None, category=None):
    out_list = []
    for t, r in rows.items():
        a = attribute(r, out, arm)
        if kind is not None and a["kind"] != kind:
            continue
        if category is not None and a["category"] != category:
            continue
        out_list.append(t)
    return out_list


def phase_stats(out, arm, tid):
    trace = out / arm / "results" / "traces" / f"trace_{tid}.json"
    if not trace.is_file():
        return None
    data = json.loads(trace.read_text(encoding="utf-8"))
    prev = 0.0
    calls = {}
    wall = {}
    raw = 0
    denied = 0
    failed = 0

    def add(key, d):
        calls[key] = calls.get(key, 0) + 1
        wall[key] = wall.get(key, 0.0) + d

    for s in data.get("steps", []):
        extra = s.get("extra") or {}
        elapsed = extra.get("elapsed_s") or prev
        d = max(elapsed - prev, 0)
        prev = elapsed
        tcs = s.get("tool_calls") or []
        obs = str((s.get("observation") or {}).get("content") or "")
        if '"error_type": "BudgetExceeded"' in obs:
            denied += 1
        if tcs:
            tc = tcs[0]
            raw += len(tcs)
            fn = str(tc.get("function_name") or "?")
            args = tc.get("arguments") or {}
            if fn == "read_file":
                key = "explore_read"
            elif fn == "run_command":
                cmd = str(args.get("command") or "")
                key = "test_run" if re.search(r"\bpytest\b", cmd) else ("script_run" if re.search(r"\.py", cmd) else "explore_shell")
            elif fn in ("edit_file", "write_file"):
                fp = str(args.get("filepath") or args.get("file_path") or args.get("path") or "")
                key = "script_write" if any(m in fp for m in SCRATCH) else "source_edit"
            elif fn in GRAPH_TOOLS:
                key = "graph_tool"
            elif fn == "get_status":
                key = "status"
            elif fn == "submit_patch":
                key = "submit"
            else:
                key = "other"
            add(key, d)
            if '"status": "error"' in obs:
                failed += 1
        else:
            add("model_only", d)
    return {"calls": calls, "wall": {k: round(v) for k, v in wall.items()}, "raw": raw,
            "denied": denied, "failed": failed}


def pytest_seconds(out, arm, tid):
    p = out / arm / tid / "test_output.txt"
    if not p.is_file():
        return None
    hits = PYTEST_RE.findall(p.read_text(encoding="utf-8", errors="replace"))
    return float(hits[-1]) if hits else None


def main():
    V, vout = load("val20")
    try:
        C, cout = load("b28")
    except FileNotFoundError:
        print("b28 outputs not collected yet; run after `--suite b28` completes.")
        return
    arm_c, arm_v = "tools_28", "tools_20"
    tasks = sorted(V)

    lines = ["# b28 vs val20 判读(单 Agent 联合预算 28/6.5 对 20/5;历史对照,非同期间重复)", ""]

    # config integrity
    cfg = json.loads((cout / arm_c / "eval_config.json").read_text(encoding="utf-8"))
    budget = cfg.get("budget", {})
    sub_eval = yaml.safe_load((cout / arm_c / "submission_config" / "eval_config.yaml").read_text(encoding="utf-8"))
    se = sub_eval.get("evaluation", {})
    consistency = {
        "arm_tool_calls": budget.get("tool_calls"), "file_tool_calls": se.get("max_tool_calls"),
        "arm_time_minutes": budget.get("time_minutes"), "file_time_minutes": se.get("max_time_minutes"),
        "arm_turns": budget.get("turns"), "file_turns": se.get("max_turns"),
        "arm_timeout": cfg.get("timeout_seconds"), "file_timeout": se.get("timeout_seconds"),
    }
    consistent = (consistency["arm_tool_calls"] == consistency["file_tool_calls"] == 28
                  and consistency["arm_time_minutes"] == consistency["file_time_minutes"] == 6.5
                  and consistency["arm_turns"] == consistency["file_turns"] == 50
                  and consistency["arm_timeout"] == consistency["file_timeout"] == 300)
    lines.append(f"- 配置一致性(arm eval_config.json vs 提交 eval_config.yaml): {'✓ 一致(300/28/6.5/50)' if consistent else '✗ 不一致 —— 按预注册该批次无效'} {consistency if not consistent else ''}")
    lines.append("")

    # outcomes
    base_n = sum(V[t]["resolved"] for t in tasks)
    cand_n = sum(C[t]["resolved"] for t in tasks)
    new_pass = [t for t in tasks if C[t]["resolved"] and not V[t]["resolved"]]
    lost = [t for t in tasks if V[t]["resolved"] and not C[t]["resolved"]]
    common_pass = [t for t in tasks if V[t]["resolved"] and C[t]["resolved"]]
    common_fail = [t for t in tasks if not V[t]["resolved"] and not C[t]["resolved"]]
    env_b28 = pick(C, cout, arm_c, kind="environment")
    env_v20 = pick(V, vout, arm_v, kind="environment")
    crash_b28 = pick(C, cout, arm_c, category="context_crash")
    crash_v20 = pick(V, vout, arm_v, category="context_crash")
    import_b28 = pick(C, cout, arm_c, category="patch_import")
    import_v20 = pick(V, vout, arm_v, category="patch_import")
    timeout_b28 = pick(C, cout, arm_c, category="session_timeout")
    timeout_v20 = pick(V, vout, arm_v, category="session_timeout")
    undiag_b28 = pick(C, cout, arm_c, kind="undiagnosed")
    undiag_v20 = pick(V, vout, arm_v, kind="undiagnosed")
    lines.append(f"- **resolved:val20 {base_n}/20 → b28 {cand_n}/20**")
    lines.append(f"- 共同通过 {len(common_pass)}、新增通过 {len(new_pass)}{(':' + ', '.join(new_pass)) if new_pass else ''}、丢失通过 {len(lost)}{(':' + ', '.join(lost)) if lost else ''}、共同失败 {len(common_fail)}")
    lines.append(f"- 确证环境故障(缺依赖/缺 fixture;单独列出,不计入能力判读):b28 {env_b28 or '无'};val20 {env_v20 or '无'}")
    lines.append(f"- 上下文超限(ContextWindowExceeded;Agent 侧失败):b28 {crash_b28 or '无'};val20 {crash_v20 or '无'}")
    lines.append(f"- 补丁破坏导入/收集(Agent 侧失败):b28 {import_b28 or '无'};val20 {import_v20 or '无'}")
    lines.append(f"- 会话超时(exit -1;Agent 侧失败):b28 {timeout_b28 or '无'};val20 {timeout_v20 or '无'}")
    lines.append(f"- 证据不足/待诊断(既非确证环境故障也非已定性 Agent 失败):b28 {undiag_b28 or '无'};val20 {undiag_v20 or '无'}")
    lines.append("")

    def summ(rows):
        return {
            "empty": sum(1 for t in tasks if rows[t]["patch_chars"] == 0),
            "budget_hits": sum(1 for t in tasks if (rows[t].get("error") or "").find("budget") >= 0 or "BudgetExceeded" in (rows[t].get("error") or "")),
            "timeouts": sum(1 for t in tasks if rows[t]["test_exit_code"] == -1),
            "dur_mean": round(statistics.mean(rows[t]["duration_seconds"] for t in tasks)),
            "dur_sum": round(sum(rows[t]["duration_seconds"] for t in tasks)),
            "counted_calls_sum": sum(rows[t]["tool_calls"] for t in tasks),
        }

    sv, sc = summ(V), summ(C)
    lines.append("## 题组汇总(同 20 题)")
    lines.append("")
    lines.append("| 指标 | val20 | b28 |")
    lines.append("|---|---|---|")
    lines.append(f"| 空补丁 | {sv['empty']} | {sc['empty']} |")
    lines.append(f"| Σ 计数调用 | {sv['counted_calls_sum']} | {sc['counted_calls_sum']} |")
    lines.append(f"| 超时行(exit -1) | {sv['timeouts']} | {sc['timeouts']} |")
    lines.append(f"| 平均 duration(s) | {sv['dur_mean']} | {sc['dur_mean']} |")
    lines.append(f"| Σ duration(s) | {sv['dur_sum']} | {sc['dur_sum']} |")
    lines.append("")

    # new-budget usage by phase (b28 - val20 deltas, sums over tasks)
    agg_v = {}
    agg_c = {}
    missing_trace = {"val20": [], "b28": []}
    for t in tasks:
        pv = phase_stats(vout, arm_v, t)
        pc = phase_stats(cout, arm_c, t)
        if not pv:
            # 不静默跳过：缺 trace 的题单独列出，避免"少算几题"被误读成预算下降
            missing_trace["val20"].append(t)
        if not pc:
            missing_trace["b28"].append(t)
        for k, v in (pv or {}).get("calls", {}).items():
            agg_v[k] = agg_v.get(k, 0) + v
        for k, v in (pc or {}).get("calls", {}).items():
            agg_c[k] = agg_c.get(k, 0) + v
    lines.append("## 新增预算的去向(同 20 题的阶段调用数合计;计数=工具执行,含被拒)")
    lines.append("")
    if missing_trace["val20"] or missing_trace["b28"]:
        lines.append(f"- ⚠ 缺 trace 未纳入阶段聚合:val20 {missing_trace['val20'] or '无'};b28 {missing_trace['b28'] or '无'}(该题结果与预算统计仍计入上方汇总表)")
        lines.append("")
    phases = sorted(set(agg_v) | set(agg_c))
    lines.append("| 阶段 | val20 | b28 | Δ |")
    lines.append("|---|---|---|---|")
    for k in phases:
        a, b = agg_v.get(k, 0), agg_c.get(k, 0)
        lines.append(f"| {k} | {a} | {b} | {b - a:+d} |")
    lines.append("")

    # call semantics
    raw_c = sum((phase_stats(cout, arm_c, t) or {}).get("raw", 0) for t in tasks)
    denied_c = sum((phase_stats(cout, arm_c, t) or {}).get("denied", 0) for t in tasks)
    failed_c = sum((phase_stats(cout, arm_c, t) or {}).get("failed", 0) for t in tasks)
    lines.append(f"- 调用口径:计数合计 {sc['counted_calls_sum']};trace 原始调用 {raw_c};被拒(BudgetExceeded) {denied_c};执行失败 {failed_c} —— 计数口径不含 get_status/submit_patch、参数校验失败与被拒调用。")
    lines.append("")

    # 12h inputs
    pytest_sum = sum((pytest_seconds(cout, arm_c, t) or 0) for t in tasks)
    missing = sum(1 for t in tasks if pytest_seconds(cout, arm_c, t) is None)
    exec_times = []
    log = cout / "execution.log"
    if log.is_file():
        times = [float(m) for m in re.findall(r'"time":\s*([0-9.]+)', log.read_text(encoding="utf-8", errors="replace"))]
        if times:
            exec_times.append(max(times))
    notebook_s = exec_times[0] if exec_times else None
    stage1_mean = (sc["dur_sum"] - pytest_sum - 20 * len(tasks)) / len(tasks)
    lines.append("## 12 小时上限评估输入(只用本批次实测;不把配额当墙钟)")
    lines.append("")
    lines.append(f"- notebook 执行总时 {('%.0f s' % notebook_s) if notebook_s else 'n/a'};Σduration(沙箱设置+Agent 循环+验证) {sc['dur_sum']} s;Σpytest(验证部分,不计入 12h) {pytest_sum:.0f} s(pytest 用时缺失 {missing} 题)。")
    lines.append(f"- Stage-1 近似口径每题均值 ≈ {stage1_mean:.0f} s(即 Σduration − Σpytest − 20 s/题验证开销)。")
    lines.append(f"- 若按此均值 ×120 + 假设启动 ~12 min ≈ **{((stage1_mean * 120) + 720) / 3600:.1f} h**——这只是依据不足的粗估(单批 20 题、隐藏集构成未知、启动摊销未知),**不得据此声称离 12h 上限还有余量**;可比硬锚点为 12h/120 = 6 min/题。")
    lines.append("")

    # ts28 comparison on the 10-subset
    subset = ["fastapi_13713", "fastapi_13786", "fastapi_13920", "fastapi_14099", "fastapi_14246",
              "requests_7328", "rich_3052", "rich_3061", "rich_3063", "rich_3105"]
    try:
        T, tout = load("ts28")
        t_conv = [t for t in subset if T[t]["resolved"]]
        b_conv = [t for t in subset if C[t]["resolved"]]
        lines.append("## 与 ts28 的结构对照(10 题子集;单次运行,不作显著性结论)")
        lines.append("")
        lines.append(f"- ts28(结构+预算)通过 {t_conv or '无'};b28(仅预算)通过 {b_conv or '无'}")
        if set(t_conv) == set(b_conv):
            lines.append("- 结构未显示出可辨识的额外价值(同子集结果一致);证据仅限单次运行,**证据不足以做方向性结论**。")
        else:
            lines.append("- 两者结果不同:仅为单次运行观察,证据不足以判断结构价值;需重复后再判。")
        lines.append("")
    except FileNotFoundError:
        lines.append("## 与 ts28 的结构对照\n\n- ts28 数据缺失,无法对照。\n")

    lines.append(f"- **预注册判读**:resolved={cand_n}/20 → " + (">=13:值得继续(仍需重复/未见确认)" if cand_n >= 13 else ("11-12:模糊,无结论" if cand_n >= 11 else "<=10:无收益,拒绝")) )
    lines.append("")
    (ART / "b28_vs_val20.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


main()