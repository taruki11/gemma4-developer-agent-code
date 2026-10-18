"""Create a concise evidence report from collected results, without running jobs."""
import json
from pathlib import Path

root = Path('artifacts')
verification_path = root / 'verification_repaired_v6_outputs/verification_results.json'
verification = json.loads(verification_path.read_text(encoding='utf-8')) if verification_path.exists() else []
lines = ['# 开发实验结果', '', '本报告来自实际运行产物；没有训练或排行榜提交。', '',
         '## 官方补丁验证', '', '| 命令超时 | resolved | 测试退出码 | 总耗时 |',
         '|---|---|---|---|']
for row in verification:
    lines.append(f"| {row['command_timeout_seconds']}s | {row['resolved']} | {row['test_exit_code']} | {row['duration_seconds']:.2f}s |")
lines.extend(['', 'CPU v5 导入系统预装 requests，其通过结果已标记无效。CPU v6 增加 editable 安装和 workspace 导入断言；以上仅列 v6 实际结果。',
              '60 秒诊断表明，有补丁与空补丁均在无效域名 DNS 测试中等待；未删减测试。',
              '该补丁来自此前真实 Agent trace 的重放，非原始捕获文件；后续对照直接保存 result.agent_patch。', ''])
for suite in ('single', 'dev', 'expand20', 'expand20b', 'promptv2', 'val20', 'think1k', 'review20', 'twostage20', 'ts28', 'b28'):
    output = root / f'tool_control_v2_{suite}_outputs'
    result_path = output / 'comparison_results.json'
    lines.extend([f'## {suite} 工具预算对照', ''])
    if not result_path.is_file():
        lines.extend(['结果尚未收集完成。', ''])
        continue
    rows = json.loads(result_path.read_text(encoding='utf-8'))
    lines.extend(['每批次时间/调用预算见该批次 experiment.json(早期批次 5 分钟/20 调用;ts28 与 b28 为 6.5 分钟/28 调用);命令超时 300 秒、官方 Gemma 与样例 Agent 相同。', '',
                  '| 任务 | 工具上限 | 实际调用 | resolved | 测试退出码 | 补丁字符数 | 耗时 |',
                  '|---|---|---|---|---|---|---|'])
    for row in rows:
        lines.append(f"| {row['task_id']} | {row['max_tool_calls']} | {row['tool_calls']} | {row['resolved']} | {row['test_exit_code']} | {row['patch_chars']} | {row['duration_seconds']:.2f}s |")
    folder = root / f'tool_control_v2_{suite}'
    summary_path = output / 'comparison_summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for limit, arm in summary.items():
            lines.append(f"- 工具上限 {limit}：resolved {arm['resolved']}/{arm['tasks']}，空补丁 {arm['empty_patches']}，"
                         f"预算耗尽 {arm['budget_exhaustions']}，环境错误或命令超时 {arm['environment_errors_or_timeouts']}，"
                         f"总耗时 {arm['total_duration_seconds']:.2f}s")
    before_path, after_path = folder / 'quota_before.json', folder / 'quota_after.json'
    if before_path.exists() and after_path.exists():
        before, after = [json.loads(p.read_text()) for p in (before_path, after_path)]
        lines.extend(['', f"账号 GPU 额度用量变化：{after['used_hours'] - before['used_hours']:.3f} 小时。"])
    lines.append('')
combined_by_task = {}
for suite in ('dev', 'expand20', 'expand20b'):
    result_path = root / f'tool_control_v2_{suite}_outputs/comparison_results.json'
    if result_path.is_file():
        for row in json.loads(result_path.read_text(encoding='utf-8')):
            if row['max_tool_calls'] == 20:
                combined_by_task[row['task_id']] = row
combined = list(combined_by_task.values())
if combined and (root / 'tool_control_v2_expand20_outputs/comparison_results.json').is_file():
    families = {}
    for row in combined:
        bucket = families.setdefault(row['task_id'].split('_')[0], [0, 0])
        bucket[0] += int(row['resolved'])
        bucket[1] += 1
    lines.extend(['## 20 工具口径合并（dev + expand20 + expand20b）', '',
                  f"- 任务数 {len(combined)}：resolved {sum(r['resolved'] for r in combined)}，"
                  f"空补丁 {sum(r['patch_chars'] == 0 for r in combined)}，"
                  f"总耗时 {sum(r['duration_seconds'] for r in combined):.2f}s",
                  '- 按仓库：' + '；'.join(f"{name} {value[0]}/{value[1]}" for name, value in sorted(families.items())),
                  'expand20b 为 wheelhouse v4 修复环境后对三个 fastapi 任务的单臂重跑；合并时同任务以其结果为准。',
                  '合并仅为开发集描述性统计，不能代表隐藏集成绩。', ''])
prompt_path = root / 'tool_control_v2_promptv2_outputs/comparison_results.json'
if prompt_path.is_file():
    prompt_rows = [r for r in json.loads(prompt_path.read_text(encoding='utf-8')) if r['max_tool_calls'] == 20]
    baseline_map = {}
    for suite in ('dev', 'expand20', 'expand20b'):
        result_path = root / f'tool_control_v2_{suite}_outputs/comparison_results.json'
        if result_path.is_file():
            for row in json.loads(result_path.read_text(encoding='utf-8')):
                if row['max_tool_calls'] == 20:
                    baseline_map[row['task_id']] = row
    lines.extend(['## promptv2 vs 基线（同 12 任务、20 工具、单变量提示词）', '',
                  '| 任务 | 基线 resolved | promptv2 resolved | 基线调用 | promptv2 调用 |',
                  '|---|---|---|---|---|'])
    improved = regressed = 0
    for row in prompt_rows:
        base = baseline_map.get(row['task_id'])
        base_ok = base['resolved'] if base else None
        if base_ok is not None:
            improved += int(not base_ok and row['resolved'])
            regressed += int(base_ok and not row['resolved'])
        lines.append(f"| {row['task_id']} | {base_ok} | {row['resolved']} | {base['tool_calls'] if base else '-'} | {row['tool_calls']} |")
    task_ids = {r['task_id'] for r in prompt_rows}
    base_resolved = sum(1 for t in task_ids if baseline_map.get(t, {}).get('resolved'))
    lines.extend(['', f"- 基线 resolved {base_resolved}/12；promptv2 resolved {sum(r['resolved'] for r in prompt_rows)}/12；改善 {improved}，回退 {regressed}。",
                  '提示词为唯一变量（prompts/system.md 追加协议），n 小且顺序执行，仅作方向性证据。', ''])
def _rep_table(title, path, note):
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding='utf-8'))
    per = {}
    for row in rows:
        per.setdefault(row['task_id'], []).append(row)
    lines.extend([f'## {title}', '',
                  '| 任务 | 通过 | 三次结果(P/F) | 三次耗时(s) | 三次调用数 |',
                  '|---|---|---|---|---|'])
    for task, rs in per.items():
        rs = sorted(rs, key=lambda r: r.get('repeat', 0))
        lines.append(f"| {task} | {sum(bool(r['resolved']) for r in rs)}/{len(rs)} | "
                     + "".join("P" if r['resolved'] else "F" for r in rs)
                     + " | " + ",".join(str(round(r['duration_seconds'])) for r in rs)
                     + " | " + ",".join(str(r['tool_calls']) for r in rs) + " |")
    flaky = [t for t, rs in per.items() if 0 < sum(bool(r['resolved']) for r in rs) < len(rs)]
    stable_pass = [t for t, rs in per.items() if all(bool(r['resolved']) for r in rs)]
    stable_fail = [t for t, rs in per.items() if not any(bool(r['resolved']) for r in rs)]
    lines.extend(['', f"- 飘忽：{', '.join(flaky) if flaky else '无'}；稳定通过：{', '.join(stable_pass) if stable_pass else '无'}；稳定失败：{', '.join(stable_fail) if stable_fail else '无'}。",
                  f'- {note}', ''])
    return per


noise_per = _rep_table('noise20 噪声基线（4 个 fastapi 任务 ×3 次，20 工具）',
                       root / 'tool_control_v2_noise20_outputs/comparison_results.json',
                       '用途：为后续 A/B 提供噪声带；落在该范围内的差异不具判读价值。')
candidate_data = {}
for suite, label in (('hygiene20', 'system.md 追加工具卫生协议'),
                     ('maxtok8k', 'sampling.yaml max_output_tokens 8192（输入上限 16384→24576，针对上下文溢出崩溃）')):
    per = _rep_table(f'{suite}（4 个 fastapi 任务 ×3 次，20 工具）',
                     root / f'tool_control_v2_{suite}_outputs/comparison_results.json',
                     f'单变量：{label}。')
    if per:
        candidate_data[suite] = per
if noise_per and candidate_data:
    header = ' | '.join(candidate_data)
    lines.extend(['## 候选 vs noise20（按任务通过次数）', '',
                  f'| 任务 | noise 通过 | {header} |',
                  '|' + '---|' * (2 + len(candidate_data))])
    for task in noise_per:
        n = sum(bool(r['resolved']) for r in noise_per[task])
        cells = [f"{sum(bool(r['resolved']) for r in candidate_data[s].get(task, []))}/3" for s in candidate_data]
        lines.append(f"| {task} | {n}/3 | " + " | ".join(cells) + " |")
    lines.extend(['', '- 通过数差异需对照噪声带解读；机制证据看对应失败是否消失（崩溃数、失败调用、补丁卫生）。', ''])
lines.extend(['## 解释边界', '', '单任务与开发集（4+8 任务）结果仅用于筛选方向。样例采用随机采样，运行顺序与分组限制因果判断，不能据此声称统计显著提升或隐藏集分数。',
              '真实补丁、测试输出、完整结果位于 tools_10/tools_20 下；trace 位于各组 results/traces。', ''])
(root / 'experiment_report.md').write_text('\n'.join(lines), encoding='utf-8')
print((root / 'experiment_report.md').resolve())
