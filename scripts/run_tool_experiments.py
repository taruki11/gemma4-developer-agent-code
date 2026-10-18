"""Run quota-checked GPU comparisons only after CPU verification is usable."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_class import classify, counts  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--verification', type=Path, default=Path('artifacts/verification_repaired_v6_outputs/verification_results.json'))
parser.add_argument('--suite', choices=['single', 'dev', 'expand20', 'expand20b', 'promptv2', 'noise20', 'hygiene20', 'maxtok8k', 'val20', 'think1k', 'review20', 'twostage20', 'ts28', 'b28'], default='single')
parser.add_argument('--collect-only', action='store_true', help='Collect and summarize an already launched run without using more GPU')
args = parser.parse_args()
cli = str(Path(sys.executable).with_name('kaggle.exe'))
api = KaggleApi()
api.authenticate()


def environment_errors(output):
    """仅用于 CPU 预检门禁（reference 补丁口径、无 Agent 补丁）。

    批次汇总不再使用本函数：Agent 补丁破坏 import 造成的收集失败会被它误判为环境故障，
    批次侧归因统一走 `failure_class.classify()`。
    """
    return any(marker in output for marker in [
        'recursive dependency involving fixture', 'fixture \'httpbin\' not found',
        'ModuleNotFoundError', 'ImportError while importing test module',
        'ERROR at setup', 'ERROR collecting'])


def quota_snapshot(destination):
    response = api.quota_view()
    quota = response.gpu_quota
    if quota is None:
        raise RuntimeError('GPU quota could not be confirmed')
    used = quota.time_used.total_seconds() / 3600
    total = quota.total_time_allowed.total_seconds() / 3600
    state = {'checked_epoch': time.time(), 'used_hours': used, 'total_hours': total,
             'remaining_hours': max(0, total - used),
             'refresh_at': response.quota_refresh_time.isoformat() if response.quota_refresh_time else None}
    destination.write_text(json.dumps(state, indent=2))
    return state


verification = json.loads(args.verification.read_text(encoding='utf-8'))
audit = json.loads(args.verification.with_name('workspace_install_audit.json').read_text(encoding='utf-8'))
assert any(a['stage'] == 'import_origin' for a in audit), 'Missing workspace import-origin evidence'
assert all(a['exit_code'] == 0 for a in audit), 'Workspace installation or imports failed'
long_run = next(r for r in verification if r['command_timeout_seconds'] == 300)
if long_run['error'] or long_run['test_exit_code'] in (-1, 124) or environment_errors(long_run['test_output']):
    raise RuntimeError('CPU verifier still has environment errors or timeout; GPU experiments are paused')

if args.suite in ('dev', 'expand20'):
    environments = json.loads(Path('artifacts/dev_environment_v4_outputs/environment_check_results.json').read_text(encoding='utf-8'))
    assert {r['task_id'] for r in environments} == {'fastapi_11194', 'httpx_3672', 'rich_2725'}, 'Incomplete CPU environment check'
    assert all(r['resolved'] and r['test_exit_code'] == 0 for r in environments), 'CPU reference-only environment sanity check failed'
    prior = json.loads(Path('artifacts/tool_control_v2_single_outputs/comparison_summary.json').read_text(encoding='utf-8'))
    if any(s['environment_errors_or_timeouts'] for s in prior.values()) or all(s['empty_patches'] == s['tasks'] for s in prior.values()):
        raise RuntimeError('Single-task pipeline has not passed the expansion gate')

if args.suite == 'expand20':
    dev_summary = json.loads(Path('artifacts/tool_control_v2_dev_outputs/comparison_summary.json').read_text(encoding='utf-8'))
    if any(s['environment_errors_or_timeouts'] for s in dev_summary.values()):
        raise RuntimeError('Dev comparison has environment errors; expansion paused')

if args.suite in ('expand20b', 'promptv2', 'noise20', 'hygiene20', 'maxtok8k'):
    environments = json.loads(Path('artifacts/dev_environment_v5_outputs/environment_check_results.json').read_text(encoding='utf-8'))
    expected_tasks = {'fastapi_11194', 'httpx_3672', 'rich_2725', 'fastapi_11355', 'fastapi_12942', 'fastapi_13207'}
    assert {r['task_id'] for r in environments} == expected_tasks, 'Incomplete v5 environment check'
    assert all(r['resolved'] and r['test_exit_code'] == 0 for r in environments), 'v5 environment check has failures; fix before GPU'
    dev_summary = json.loads(Path('artifacts/tool_control_v2_dev_outputs/comparison_summary.json').read_text(encoding='utf-8'))
    if any(s['environment_errors_or_timeouts'] for s in dev_summary.values()):
        raise RuntimeError('Dev comparison has environment errors; expansion paused')

if args.suite in ('val20', 'think1k', 'review20', 'twostage20', 'b28'):
    v6 = json.loads(Path('artifacts/dev_environment_v6_outputs/environment_check_results.json').read_text(encoding='utf-8'))
    v7 = json.loads(Path('artifacts/dev_environment_v7_outputs/environment_check_results.json').read_text(encoding='utf-8'))
    final_tasks = {'fastapi_13537', 'requests_7502', 'rich_3043', 'fastapi_13713', 'requests_7205',
                   'rich_3052', 'fastapi_13786', 'requests_7309', 'rich_3061', 'fastapi_13920',
                   'requests_7315', 'rich_3063', 'fastapi_14077', 'requests_7328', 'rich_3105',
                   'fastapi_14099', 'requests_7427', 'rich_3067', 'fastapi_14246', 'requests_7433'}
    entries = {r['task_id']: r for r in v6 + v7}
    missing = final_tasks - set(entries)
    assert not missing, f'Missing environment entries: {missing}'
    bad = [t for t in final_tasks if not (entries[t]['resolved'] and entries[t]['test_exit_code'] == 0)]
    assert not bad, f'{args.suite} environment check failures: {bad}'

if args.suite == 'ts28':
    v6 = json.loads(Path('artifacts/dev_environment_v6_outputs/environment_check_results.json').read_text(encoding='utf-8'))
    v7 = json.loads(Path('artifacts/dev_environment_v7_outputs/environment_check_results.json').read_text(encoding='utf-8'))
    ts28_tasks = {'fastapi_13713', 'fastapi_13786', 'fastapi_13920', 'fastapi_14099', 'fastapi_14246',
                  'requests_7328', 'rich_3052', 'rich_3061', 'rich_3063', 'rich_3105'}
    entries = {r['task_id']: r for r in v6 + v7}
    missing = ts28_tasks - set(entries)
    assert not missing, f'Missing environment entries: {missing}'
    bad = [t for t in ts28_tasks if not (entries[t]['resolved'] and entries[t]['test_exit_code'] == 0)]
    assert not bad, f'ts28 environment check failures: {bad}'

for suite, runtime, required_hours in [('single', 2700, 1.5), ('dev', 5400, 3.0), ('expand20', 7200, 3.0), ('expand20b', 3600, 1.5), ('promptv2', 9000, 3.0), ('noise20', 9000, 3.0), ('hygiene20', 9000, 3.0), ('maxtok8k', 9000, 3.0), ('val20', 8100, 3.5), ('think1k', 8100, 3.5), ('review20', 8100, 3.5), ('twostage20', 8100, 3.5), ('ts28', 7200, 2.6), ('b28', 14400, 5.0)]:
    if suite != args.suite:
        continue
    folder = Path(f'artifacts/tool_control_v2_{suite}')
    output = Path(f'artifacts/tool_control_v2_{suite}_outputs')
    output.mkdir(exist_ok=True)
    if not args.collect_only:
        if (folder / 'push.json').is_file():
            raise RuntimeError('This batch was already launched; use --collect-only to avoid duplicate GPU use')
        subprocess.run([sys.executable, 'scripts/prepare_tool_experiment.py', '--suite', suite], check=True)
        quota = quota_snapshot(folder / 'quota_before.json')
        if quota['remaining_hours'] < required_hours:
            raise RuntimeError(f'Not enough confirmed free quota for {suite}: {quota}')
        pushed = subprocess.run([cli, 'kernels', 'push', '-p', str(folder), '-t', str(runtime),
                                 '--accelerator', 'NvidiaL4'], capture_output=True, text=True)
        (folder / 'push.json').write_text(json.dumps({'epoch': time.time(), 'exit_code': pushed.returncode,
                                                    'stdout': pushed.stdout, 'stderr': pushed.stderr}, indent=2))
        print(pushed.stdout, flush=True)
        pushed.check_returncode()
    kernel = json.loads((folder / 'kernel-metadata.json').read_text(encoding='utf-8'))['id']
    subprocess.run([sys.executable, 'scripts/collect_kaggle_run.py', '--wait', '--kernel', kernel,
                    '--output', str(output)], check=True)
    quota_snapshot(folder / 'quota_after.json')
    results = json.loads((output / 'comparison_results.json').read_text(encoding='utf-8'))
    expected_count = {'single': 2, 'dev': 8, 'expand20': 8, 'expand20b': 3, 'promptv2': 12, 'noise20': 12, 'hygiene20': 12, 'maxtok8k': 12, 'val20': 20, 'think1k': 20, 'review20': 20, 'twostage20': 20, 'ts28': 10, 'b28': 20}[suite]
    if len(results) != expected_count:
        raise RuntimeError(f'Incomplete {suite} results: {len(results)}/{expected_count}')
    summary = {}
    for tool_limit in ((10, 20) if suite in ('single', 'dev') else ((28,) if suite in ('ts28', 'b28') else (20,))):
        rows = [r for r in results if r['max_tool_calls'] == tool_limit]
        attributions = []
        budget_exhaustions = 0
        for row in rows:
            arm = output / (f"tools_20_r{row['repeat']}" if suite in ('noise20', 'hygiene20', 'maxtok8k') else f'tools_{tool_limit}')
            base = arm / row['task_id']
            test_path = base / 'test_output.txt'
            test_output = test_path.read_text(encoding='utf-8', errors='replace') if test_path.is_file() else ''
            patch_path = base / 'agent.patch'
            patch_text = patch_path.read_text(encoding='utf-8', errors='replace') if patch_path.is_file() else ''
            attribution = classify(row, test_output, patch_text)
            attribution['task_id'] = row['task_id']
            attributions.append(attribution)
            trace_path = arm / 'results' / 'traces' / f"trace_{row['task_id']}.json"
            if not trace_path.is_file():
                raise RuntimeError(f'Missing agent trace: {trace_path}')
            budget_exhaustions += int('BudgetExceeded' in trace_path.read_text(encoding='utf-8'))
        attribution_counts = counts(attributions)
        summary[str(tool_limit)] = {'tasks': len(rows), 'resolved': sum(r['resolved'] for r in rows),
            'empty_patches': sum(r['patch_chars'] == 0 for r in rows),
            'budget_exhaustions': budget_exhaustions,
            'environment_errors_or_timeouts': attribution_counts['environment_errors_or_timeouts'],
            'failure_attribution': attribution_counts,
            'failure_details': {a['task_id']: {'kind': a['kind'], 'category': a['category'], 'evidence': a['evidence']}
                                for a in attributions if a['kind'] not in ('none', 'tests_failed')},
            'total_duration_seconds': sum(r['duration_seconds'] for r in rows)}
    if suite in ('noise20', 'hygiene20', 'maxtok8k'):
        per_task = {}
        for row in results:
            per_task.setdefault(row['task_id'], []).append(bool(row['resolved']))
        summary['per_task_pass'] = {task: sum(v) for task, v in per_task.items()}
        summary['flaky'] = [task for task, v in per_task.items() if 0 < sum(v) < len(v)]
    (output / 'comparison_summary.json').write_text(json.dumps(summary, indent=2))
    arm_summaries = [value for key, value in summary.items() if isinstance(key, str) and key.isdigit()]
    print(json.dumps(summary), flush=True)
    # 只有"确证环境故障"（缺依赖/缺 fixture）才暂停后续 GPU 批次；
    # Agent 侧失败（上下文超限/补丁破坏导入/会话超时）与命令超时是实验数据本身，不暂停。
    if any(s['failure_attribution']['environment_errors'] for s in arm_summaries):
        raise RuntimeError('Environment errors detected; further GPU batches are paused')
    if any(s['failure_attribution']['undiagnosed_failures'] for s in arm_summaries):
        print('WARNING: 存在证据不足（待诊断）的失败，未计为环境故障；见 summary.failure_details', flush=True)
    if suite in ('single', 'expand20', 'expand20b', 'promptv2', 'noise20', 'hygiene20', 'maxtok8k') and all(s['empty_patches'] == s['tasks'] for s in arm_summaries):
        raise RuntimeError('Neither arm produced a patch; diagnose traces before expanding')
print(f'{args.suite} comparison complete. No training or scoring submission.', flush=True)
