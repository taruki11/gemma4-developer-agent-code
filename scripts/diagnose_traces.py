"""Phase 0 diagnostics over collected traces, patches and adapters. Read-only, zero GPU.

Reads every collected run under artifacts/tool_control_v2_*_outputs, computes per-step
timings from cumulative elapsed_s diffs, classifies steps, audits patches and adapters,
and writes artifacts/diagnosis_report.md.
"""
import hashlib
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("artifacts")
SUITES = ["single", "dev", "expand20", "expand20b", "promptv2", "val20", "think1k", "review20", "twostage20", "ts28", "b28"]
ENV_MARKERS = [
    "recursive dependency involving fixture",
    "fixture 'httpbin' not found",
    "ModuleNotFoundError",
    "ImportError while importing test module",
    "ERROR at setup",
    "ERROR collecting",
]
SCRATCH_PREFIXES = ("repro", "test_", "check", "verify", "debug")
SLOW_THRESHOLD = 30.0
GRAPH_TOOLS = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}


def load_runs():
    runs = []
    for suite in SUITES:
        root = ART / f"tool_control_v2_{suite}_outputs"
        rpath = root / "comparison_results.json"
        if not rpath.is_file():
            continue
        for row in json.loads(rpath.read_text(encoding="utf-8")):
            arm = f"tools_{row['max_tool_calls']}"
            base = root / arm / row["task_id"]
            test_out = base / "test_output.txt"
            text = test_out.read_text(encoding="utf-8", errors="replace") if test_out.is_file() else ""
            runs.append({
                "suite": suite, "arm": arm, "task": row["task_id"],
                "resolved": bool(row["resolved"]), "exit": row["test_exit_code"],
                "error": row.get("error"), "patch_chars": row["patch_chars"],
                "tool_calls": row["tool_calls"], "duration": row["duration_seconds"],
                "trace": root / arm / "results" / "traces" / f"trace_{row['task_id']}.json",
                "patch": base / "agent.patch",
                "env_issue": any(m in text for m in ENV_MARKERS),
            })
    return runs


def step_time(step):
    obs = step.get("observation") or {}
    for holder in (obs, step):
        extra = holder.get("extra") or {}
        if isinstance(extra, dict) and extra.get("timestamp"):
            return float(extra["timestamp"]), float(extra.get("elapsed_s") or 0.0)
    for tc in step.get("tool_calls") or []:
        extra = tc.get("extra") or {}
        if extra.get("timestamp"):
            return float(extra["timestamp"]), float(extra.get("elapsed_s") or 0.0)
    return None


def classify(step):
    tcs = step.get("tool_calls") or []
    if not tcs:
        return "model_only", None, {}
    tc = tcs[0]
    fn = str(tc.get("function_name") or "?")
    args = tc.get("arguments") or {}
    if fn == "read_file":
        return "explore_read", fn, args
    if fn == "run_command":
        cmd = str(args.get("command") or "")
        if re.search(r"\bpytest\b", cmd):
            return "test_run", fn, args
        if re.search(r"\bpython3?(?:\s+-\S+)*\s+\S*\.py", cmd):
            return "script_run", fn, args
        return "explore_shell", fn, args
    if fn in ("edit_file", "write_file"):
        fp = str(args.get("filepath") or "")
        scratch = "/" not in fp and fp.startswith(SCRATCH_PREFIXES)
        return ("script_write" if scratch else "source_edit"), fn, args
    if fn in GRAPH_TOOLS:
        return "graph_tool", fn, args
    if fn == "get_status":
        return "status", fn, args
    if fn == "submit_patch":
        return "submit", fn, args
    if "agent" in fn.lower():
        return "agent_tool", fn, args
    return "other", fn, args


def analyze_trace(path, duration):
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = {
        "class_time": defaultdict(float), "class_calls": Counter(),
        "functions": Counter(), "index": 0,
        "first_edit_index": None, "first_edit_start_elapsed": None,
        "slow": [], "denied_on_edit": False, "denied_any": False,
        "submitted": False, "budget_warnings": 0, "last_elapsed": None,
        "failed_calls": Counter(), "validation_failed": 0, "denied_calls": 0,
    }
    prev_elapsed = None
    for step in data.get("steps", []):
        t = step_time(step)
        start_elapsed = prev_elapsed
        dur = None
        if t is not None:
            if prev_elapsed is not None:
                dur = max(0.0, t[1] - prev_elapsed)
            prev_elapsed = t[1]
        meta["last_elapsed"] = prev_elapsed
        cls, fn, args = classify(step)
        tcs = step.get("tool_calls") or []
        if tcs:
            meta["index"] += 1
        for tc in tcs:
            meta["functions"][str(tc.get("function_name") or "?")] += 1
        obs_text = str((step.get("observation") or {}).get("content") or "")
        meta["budget_warnings"] += "budget_warning" in obs_text
        denied = '"error_type": "BudgetExceeded"' in obs_text
        if denied:
            meta["denied_any"] = True
            meta["denied_calls"] += 1
            if cls in ("source_edit", "script_write"):
                meta["denied_on_edit"] = True
        elif '"error": "Invoking' in obs_text:
            meta["validation_failed"] += 1
        elif '"error_type":' in obs_text:
            meta["failed_calls"][cls] += 1
        if cls == "submit":
            meta["submitted"] = True
        landed = cls == "source_edit" and '"status": "ok"' in obs_text
        if landed and meta["first_edit_index"] is None:
            meta["first_edit_index"] = meta["index"]
            meta["first_edit_start_elapsed"] = start_elapsed
        if dur is not None:
            meta["class_time"][cls] += dur
            meta["class_calls"][cls] += 1
            if dur > SLOW_THRESHOLD:
                note = ""
                if cls in ("source_edit", "script_write"):
                    note = f"payload {len(str(args.get('new_string') or args.get('content') or ''))} chars"
                elif fn == "run_command":
                    note = "cmd: " + str(args.get("command") or "")[:70].replace("\n", " ")
                elif cls == "model_only":
                    note = "model/thinking only"
                else:
                    note = "light tool; likely model/thinking"
                meta["slow"].append({
                    "index": meta["index"], "cls": cls, "fn": fn,
                    "dur": round(dur, 1), "note": note,
                    "denied": denied, "step_id": step.get("step_id"),
                })
    meta["first_edit_fraction"] = (
        round(meta["first_edit_start_elapsed"] / duration, 3)
        if meta["first_edit_start_elapsed"] is not None and duration else None
    )
    return meta


def audit_patch(path):
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    files = re.findall(r"^diff --git a/(\S+) b/", text, flags=re.M)
    scratch = [f for f in files if "/" not in f and f.startswith(SCRATCH_PREFIXES)]
    return {"files": files, "scratch": scratch, "chars": len(text)}


def safetensors_summary(path):
    raw = path.read_bytes()
    (length,) = struct.unpack_from("<Q", raw, 0)
    header = json.loads(raw[8:8 + length].decode("utf-8"))
    base = 8 + length
    tensors = []
    for name, info in header.items():
        if not isinstance(info, dict) or "data_offsets" not in info:
            continue
        start, end = info["data_offsets"]
        blob = raw[base + start: base + end]
        tensors.append({
            "name": name, "dtype": info.get("dtype"), "shape": info.get("shape"),
            "bytes": len(blob), "all_zero": blob == b"\x00" * len(blob),
        })
    return {"sha256": hashlib.sha256(raw).hexdigest(), "len": len(raw), "tensors": tensors}


def main():
    runs = load_runs()
    traces = []
    for run in runs:
        if not run["trace"].is_file():
            run["meta"] = None
            continue
        run["meta"] = analyze_trace(run["trace"], run["duration"])
        traces.append(run)

    valid = [r for r in traces if not r["env_issue"]]
    primary = [r for r in valid if r["arm"] == "tools_20"]
    secondary = [r for r in valid if r["arm"] == "tools_10"]
    unresolved = [r for r in valid if not r["resolved"]]
    env_blocked = [r for r in traces if r["env_issue"]]

    lines = ["# Phase 0 诊断报告（只读，零 GPU）", "",
             f"覆盖 suite：{', '.join(SUITES)}；trace 总数 {len(traces)}（有效 {len(valid)}，环境阻塞 {len(env_blocked)}）。",
             "口径：每步耗时 = 相邻步骤 `elapsed_s`（自会话起点的累计值）之差；该差值包含模型生成 + 工具执行。",
             "首编辑 = 第一条 `status: ok` 的 `edit_file/write_file`（非草稿路径）；被拒绝或被截断的编辑不计入。", ""]

    lines.extend(["## 1. 逐运行概览", "",
                  "| suite | 任务 | arm | resolved | 总调用 | 首编辑(步号) | 首编辑前耗时占比 | 提交 | 会话时长(s) | 阶段耗时(s)：探索/草稿/编辑/测试 |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
    for run in sorted(primary + secondary, key=lambda r: (r["suite"], r["arm"], r["task"])):
        m = run["meta"]
        if m is None:
            lines.append(f"| {run['suite']} | {run['task']} | {run['arm']} | (无 trace) | | | | | | |")
            continue
        ct = m["class_time"]
        explore = ct.get("explore_read", 0) + ct.get("explore_shell", 0) + ct.get("graph_tool", 0)
        scratch = ct.get("script_write", 0) + ct.get("script_run", 0)
        edit = ct.get("source_edit", 0)
        test = ct.get("test_run", 0)
        frac = m["first_edit_fraction"]
        lines.append(
            f"| {run['suite']} | {run['task']} | {run['arm']} | {run['resolved']} | {run['tool_calls']} "
            f"| {m['first_edit_index']} | {'-' if frac is None else f'{frac:.0%}'} | {m['submitted']} "
            f"| {run['duration']:.0f} | {explore:.0f}/{scratch:.0f}/{edit:.0f}/{test:.0f} |")
    lines.append("")

    lines.extend(["## 2. 阶段耗时汇总（按结果分组）", ""])
    for label, group in (("通过", [r for r in valid if r["resolved"]]),
                         ("未通过", unresolved)):
        if not group:
            continue
        lines.append(f"### {label}（{len(group)} 个运行）")
        lines.append("")
        lines.append("| 类别 | 平均调用数 | 平均耗时(s) | 占会话时长 |")
        lines.append("|---|---|---|---|")
        cats = ["explore_read", "explore_shell", "script_write", "script_run",
                "source_edit", "test_run", "submit", "status", "graph_tool", "agent_tool", "model_only", "other"]
        grand = sum(sum((r["meta"]["class_time"]).get(c, 0) for c in cats) for r in group) or 1
        for cat in cats:
            calls = sum(r["meta"]["class_calls"].get(cat, 0) for r in group)
            total = sum(r["meta"]["class_time"].get(cat, 0) for r in group)
            if calls == 0 and total == 0:
                continue
            lines.append(f"| {cat} | {calls / len(group):.1f} | {total / len(group):.0f} | {total / grand:.0%} |")
        lines.append("")
        fracs = [r["meta"]["first_edit_fraction"] for r in group if r["meta"]["first_edit_fraction"] is not None]
        if fracs:
            lines.append(f"- 首编辑前耗时占比：平均 {sum(fracs) / len(fracs):.0%}（{min(fracs):.0%}–{max(fracs):.0%}）")
        no_edit = [r["task"] for r in group if r["meta"]["first_edit_index"] is None]
        if no_edit:
            lines.append(f"- 无落地源码编辑：{', '.join(no_edit)}")
        lines.append("")

    lines.extend(["### 失败调用（工具执行报错 / 参数校验失败）", ""])
    total_failed = Counter()
    total_validation = 0
    per_run_failed = []
    for r in valid:
        fc = r["meta"].get("failed_calls") or Counter()
        vf = r["meta"].get("validation_failed", 0)
        if sum(fc.values()) or vf:
            per_run_failed.append((r, fc, vf))
        total_failed.update(fc)
        total_validation += vf
    lines.append(f"- 工具执行报错合计 {sum(total_failed.values())} 次；参数校验失败合计 {total_validation} 次（多为 edit_file 缺 old_str）。")
    lines.append("")
    lines.append("| suite | 任务 | arm | 执行报错 | 参数校验失败 | 类别 |")
    lines.append("|---|---|---|---|---|---|")
    for r, fc, vf in sorted(per_run_failed, key=lambda x: -(sum(x[1].values()) + x[2])):
        lines.append(f"| {r['suite']} | {r['task']} | {r['arm']} | {sum(fc.values())} | {vf} | " +
                     "；".join(f"{k}×{v}" for k, v in fc.most_common()) + " |")
    lines.append("")

    lines.extend(["### 调用计数核对", "",
                  "差值为 0 表示 `raw - 免费(get_status/submit_patch) - 校验失败 - 被拒 = result.tool_calls` 成立。", ""])
    mismatches = []
    for r in valid:
        m = r["meta"]
        raw = sum(m["functions"].values())
        free = m["functions"].get("get_status", 0) + m["functions"].get("submit_patch", 0)
        diff = raw - free - m["validation_failed"] - m["denied_calls"] - r["tool_calls"]
        if diff != 0:
            mismatches.append((r, raw, free, m["validation_failed"], m["denied_calls"], diff))
    if mismatches:
        lines.append("| suite | 任务 | arm | raw | 免费 | 校验失败 | 被拒 | 差值 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r, raw, free, vf, dn, diff in mismatches:
            lines.append(f"| {r['suite']} | {r['task']} | {r['arm']} | {raw} | {free} | {vf} | {dn} | {diff} |")
        lines.append(f"- 上述为不吻合的运行（共 {len(mismatches)}）；其余运行差值均为 0。")
    else:
        lines.append("- 全部运行差值均为 0：免费调用、参数校验失败的调用与被拒调用都不计入工具预算。")
    lines.append("")

    lines.extend(["## 3. 慢步骤（>30s）", ""])
    for r in valid:
        slow = r["meta"]["slow"]
        if not slow:
            continue
        top = sorted(slow, key=lambda x: -x["dur"])[:3]
        entry = "；".join(f"步{s['index']} {s['cls']} {s['dur']:.0f}s（{s['note']}{'，被拒绝' if s['denied'] else ''}）" for s in top)
        lines.append(f"- {r['suite']}/{r['task']}：{entry}")
    lines.append("")

    lines.extend(["## 4. 工具使用普查", ""])
    all_fn = Counter()
    special = Counter()
    for r in valid:
        all_fn.update(r["meta"]["functions"])
        for fn, n in r["meta"]["functions"].items():
            if fn in GRAPH_TOOLS or fn == "get_status" or fn == "submit_patch" or "agent" in fn.lower():
                special[fn] += n
    lines.append("全部函数名与次数（有效运行合计）：")
    lines.append("")
    for fn, n in all_fn.most_common():
        lines.append(f"- `{fn}`：{n}")
    lines.append("")
    lines.append("重点工具的按 suite 分布：")
    lines.append("")
    lines.append("| suite | " + " | ".join(sorted(special)) + " |")
    lines.append("|" + "---|" * (len(special) + 1))
    for suite in SUITES:
        row = Counter()
        for r in valid:
            if r["suite"] != suite:
                continue
            for fn, n in r["meta"]["functions"].items():
                if fn in special:
                    row[fn] += n
        if sum(row.values()) == 0 and suite != "single":
            pass
        lines.append("| " + suite + " | " + " | ".join(str(row.get(fn, 0)) for fn in sorted(special)) + " |")
    lines.append("")

    lines.extend(["## 5. 补丁卫生审计", ""])
    pol = []
    for r in traces:
        audit = audit_patch(r["patch"])
        if audit is None:
            continue
        pol.append((r, audit))
    dirty = [x for x in pol if x[1]["scratch"]]
    lines.append(f"- 补丁总数 {len(pol)}；含草稿脚本的补丁 {len(dirty)}。")
    name_counter = Counter()
    for _, a in dirty:
        name_counter.update(a["scratch"])
    if name_counter:
        lines.append("- 草稿文件名分布：" + "；".join(f"{k}×{v}" for k, v in name_counter.most_common()))
    lines.append("")
    lines.append("| suite | 任务 | arm | 文件数 | 草稿文件 |")
    lines.append("|---|---|---|---|---|")
    for r, a in dirty:
        lines.append(f"| {r['suite']} | {r['task']} | {r['arm']} | {len(a['files'])} | {', '.join(a['scratch'])} |")
    lines.append("")

    lines.extend(["## 6. 未通过运行分类（有效运行）", "",
                  "| suite | 任务 | arm | 标签 | 备注 |", "|---|---|---|---|---|"])
    for r in sorted(unresolved, key=lambda r: (r["suite"], r["task"])):
        m = r["meta"]
        labels = []
        if m["first_edit_index"] is None:
            labels.append("无落地编辑")
        if m["denied_on_edit"]:
            labels.append("编辑被预算拒绝")
        elif m["denied_any"]:
            labels.append("预算拒绝(非编辑)")
        if not m["submitted"]:
            labels.append("未提交(自动回收)")
        err = str(r.get("error") or "").lower()
        if "timeout" in err or "exceeded" in err:
            labels.append("会话超时")
        if r["patch_chars"] == 0:
            labels.append("空补丁")
        if r["exit"] not in (0, -1, 124):
            labels.append("测试失败")
        if m["first_edit_index"] is not None and r["exit"] not in (0, -1, 124):
            labels.append("编辑未解决问题")
        lines.append(f"| {r['suite']} | {r['task']} | {r['arm']} | {'、'.join(labels) or '-'} | exit={r['exit']}，错误={r.get('error') or '-'} |")
    lines.append("")
    if env_blocked:
        lines.append("环境阻塞、已排除的运行：")
        for r in env_blocked:
            lines.append(f"- {r['suite']}/{r['task']}/{r['arm']}（测试收集失败，另见 INVALID_FASTAPI_ENV.json）")
        lines.append("")

    lines.extend(["## 7. 样例 adapter 检查", ""])
    adapters = {
        "main_lora": Path("data/official/sample_submission/adapters/main_lora/adapter_model.safetensors"),
        "tool_lora": Path("data/official/sample_submission/adapters/tool_lora/adapter_model.safetensors"),
    }
    summary = {}
    for name, path in adapters.items():
        if path.is_file():
            summary[name] = safetensors_summary(path)
    if len(summary) == 2:
        same = summary["main_lora"]["sha256"] == summary["tool_lora"]["sha256"]
        lines.append(f"- main_lora 与 tool_lora 字节一致：{same}（sha256 {summary['main_lora']['sha256'][:16]}…）")
        for t in summary["main_lora"]["tensors"]:
            lines.append(f"- {t['name']}：{t['dtype']} {t['shape']}，{t['bytes']} 字节，全零={t['all_zero']}")
    lines.append("")

    lines.extend(["## 8. 12 小时总预算的可行性估计（2026-10-13 修正口径）", ""])
    counts = Counter(r["suite"] for r in traces)

    def exec_seconds(suite):
        log = ART / f"tool_control_v2_{suite}_outputs" / "execution.log"
        if not log.is_file():
            return None
        times = [float(m) for m in re.findall(r'"time":\s*([0-9.]+)', log.read_text(encoding="utf-8", errors="replace"))]
        return max(times) if times else None

    execs = {s: exec_seconds(s) for s in SUITES}
    lines.append("| suite | 任务运行数 | notebook 执行(min) | 每任务(min) |")
    lines.append("|---|---|---|---|")
    for s in SUITES:
        if execs.get(s) is None:
            continue
        n = counts[s]
        lines.append(f"| {s} | {n} | {execs[s] / 60:.1f} | {execs[s] / n / 60:.2f} |")
    if execs.get("promptv2") and execs.get("expand20b"):
        p = (execs["promptv2"] - execs["expand20b"]) / (counts["promptv2"] - counts["expand20b"])
        startup = execs["expand20b"] - counts["expand20b"] * p
        total = (startup + 120 * p) / 3600
        lines.append("")
        lines.append(f"- 线性拟合（promptv2 与 expand20b，notebook 执行口径，含模型启动/Agent/补丁验证/打包）：固定开销≈{startup / 60:.0f} 分钟 + 每任务≈{p / 60:.2f} 分钟。")
        lines.append(f"- 外推 120 个隐藏任务：≈{total:.1f} 小时" + ("，高于 12 小时上限" if total > 12 else "，低于 12 小时上限") +
                     "（粗估：基于两个小批次的拟合与估算的验证开销，隐藏集构成未知，不能保证实际耗时）。剔除不计入 12 小时的补丁验证（Stage-1 口径）后更低，详见 artifacts/statistics_correction.md。")
        lines.append("- 修正说明：本节此前用 GPU 计费额度差外推（≈15.4h，错误）——L4 按 2× 计费（实测配额/墙钟≈1.85），且补丁验证不计入 12 小时；计费小时≠墙钟小时。")
    lines.append("")

    lines.extend(["## 9. 与既定根因的对照（复盘）", "",
                  "- fastapi_11355（expand20b）：补丁改 `get_typed_annotation`，官方测试 `test_stringified_annotations` 仍 422；本地探针已证明该补丁对依赖类 `__call__` 的字符串注解路径无效（上游修复应为 `inspect.signature(eval_str=True)`）。",
                  "- fastapi_13207（expand20b）：源码编辑（`fastapi/_compat/v2.py`）被 BudgetExceeded 拒绝，补丁只剩草稿脚本；上方分类表应显示“无落地编辑/编辑被拒绝”。",
                  "- fastapi_11194（dev 10 工具）与 httpx_3672（dev 10 工具）为“预算耗尽/会话超时”形态，与 20 工具组对照一致：20 次调用足以完成两者。",
                  ""])

    report = "\n".join(lines)
    (ART / "diagnosis_report.md").write_text(report, encoding="utf-8")

    print("runs:", len(runs), "| traces:", len(traces), "| valid:", len(valid),
          "| primary(20):", len(primary), "| secondary(10):", len(secondary),
          "| env-blocked:", len(env_blocked), "| unresolved:", len(unresolved))
    print("first-edit fractions (20-call valid):")
    for r in primary:
        frac = r["meta"]["first_edit_fraction"]
        print(f"  {r['suite']:9} {r['task']:15} resolved={str(r['resolved']):5} first_edit={r['meta']['first_edit_index']} frac={frac}")
    print("report:", (ART / "diagnosis_report.md").resolve())


if __name__ == "__main__":
    main()