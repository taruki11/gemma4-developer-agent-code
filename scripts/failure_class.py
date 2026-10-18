"""失败归因：把"Agent 侧失败 / 真正环境故障 / 超时 / 证据不足"分开。

共享给 `run_tool_experiments.py` 与 `compare_val20_b28.py`，避免两个脚本口径漂移。
此前 run 脚本只按 test_output 文本命中 `ERROR collecting` 就计为环境错误，
把 val20 的 fastapi_13786（Agent 补丁破坏 import 造成的收集失败）误记为环境故障。

归因规则（保守：证据不足一律标 `undiagnosed`，绝不默认算环境故障）：

- `context_crash`  → Agent：上下文超限（ContextWindowExceeded，输入 + 请求输出 > 32k）。
- `patch_import`   → Agent：补丁触碰的源码在导入/收集期报错（报错栈与补丁路径有交集），
                     或 `No module named` 指向仓库自身包（fastapi/requests/rich/httpx/...）。
- `session_timeout`→ Agent：会话时间预算耗尽（exit -1 / "exceeded session timeout"）。
- `command_timeout`→ 超时：harness 命令上限 300s（exit 124），不算环境故障也不算能力失败。
- `missing_dependency` → 环境：报告缺第三方依赖（模块名不属于仓库包）。
- `missing_fixture`    → 环境：环境缺 fixture（如 httpbin）。
- `undiagnosed`        → 证据不足：有异常标记但无法证明成因（例如报错栈与补丁无交集、
                        只有 "Sandbox execution error" 外壳）。
- `tests_failed`       → 普通未通过（exit 1/2 的断言失败），属能力结果本身。
- `none`               → 已通过（resolved 且 exit 0）。
"""
import re

# 仓库自身顶层包名：`No module named '<这些>'` 说明是仓库内导入被破坏，不是缺依赖。
REPO_ROOTS = {'fastapi', 'requests', 'rich', 'httpx', 'starlette', 'pydantic'}
FIXTURE_MARKERS = ("fixture 'httpbin' not found", 'recursive dependency involving fixture')
IMPORT_MARKERS = ('ImportError while importing test module', 'ERROR at setup', 'ERROR collecting')
MISSING_MODULE_RE = re.compile(r"No module named '([\w.]+)'")
TRACEBACK_PATH_RE = re.compile(r'^\s*([\w./\\-]+\.py):(\d+): in ', re.MULTILINE)
PATCH_PATH_RE = re.compile(r'^(?:\+\+\+|---) [ab]/(.+?)\s*$', re.MULTILINE)

AGENT_CATEGORIES = ('context_crash', 'patch_import', 'session_timeout')


def patch_paths(patch_text):
    """补丁里被改动的仓库相对路径集合。"""
    return {p.replace('\\', '/') for p in PATCH_PATH_RE.findall(patch_text or '')}


def traceback_paths(test_output):
    """pytest 报错栈里出现的 .py 路径集合（形如 `fastapi/security/api_key.py:20: in ...`）。"""
    return {p.replace('\\', '/') for p, _ in TRACEBACK_PATH_RE.findall(test_output or '')}


def _is_repo_module(module, task_id):
    root = (module or '').split('.')[0]
    if root in REPO_ROOTS:
        return True
    return bool(task_id) and root == task_id.split('_')[0]


def classify(row, test_output='', patch_text=''):
    """返回 {'kind', 'category', 'evidence'}。kind ∈ none/agent/environment/timeout/undiagnosed/tests_failed。"""
    result = {'kind': 'none', 'category': '', 'evidence': ''}
    error = row.get('error') or ''
    code = row.get('test_exit_code')
    if bool(row.get('resolved')) and code == 0:
        return result
    text = test_output or ''

    if 'ContextWindowExceeded' in error:
        return {'kind': 'agent', 'category': 'context_crash',
                'evidence': '上下文超限 ContextWindowExceeded（输入 + 请求输出 > 32768，Agent 侧）'}
    if code == 124:
        return {'kind': 'timeout', 'category': 'command_timeout',
                'evidence': '命令超时 300s（exit 124，harness 命令上限，非环境故障）'}
    if code == -1 or 'exceeded session timeout' in error.lower():
        return {'kind': 'agent', 'category': 'session_timeout',
                'evidence': '会话时间预算耗尽（exit -1 / exceeded session timeout）'}

    touched = patch_paths(patch_text)
    missing = MISSING_MODULE_RE.findall(text)
    if missing:
        foreign = sorted({m for m in missing if not _is_repo_module(m, row.get('task_id'))})
        if foreign:
            return {'kind': 'environment', 'category': 'missing_dependency',
                    'evidence': f'缺第三方依赖: {foreign}'}
        return {'kind': 'agent', 'category': 'patch_import',
                'evidence': f'仓库内导入被破坏: No module named {sorted(set(missing))}'}

    if any(m in text for m in FIXTURE_MARKERS):
        return {'kind': 'environment', 'category': 'missing_fixture',
                'evidence': '镜像缺 fixture（httpbin 等）'}

    if any(m in text for m in IMPORT_MARKERS):
        hits = sorted(traceback_paths(text) & touched)
        if hits:
            return {'kind': 'agent', 'category': 'patch_import',
                    'evidence': '补丁改动的源码在导入/收集期报错: ' + ', '.join(hits[:4])}
        if not touched:
            return {'kind': 'environment', 'category': 'missing_dependency',
                    'evidence': '无补丁（参考口径）下导入/收集失败'}
        return {'kind': 'undiagnosed', 'category': 'undiagnosed',
                'evidence': '导入/收集失败，但补丁路径与报错栈无交集，成因证据不足'}

    if 'Sandbox execution error' in error:
        return {'kind': 'undiagnosed', 'category': 'undiagnosed',
                'evidence': '仅有 Sandbox execution error 外壳，无归因证据: ' + error[:200]}

    return {'kind': 'tests_failed', 'category': 'tests_failed',
            'evidence': f'测试未通过（exit {code}），无环境/超限标记'}


def counts(rows_attributions):
    """把一批归因结果聚合成摘要计数（兼容旧键 environment_errors_or_timeouts）。"""
    get = lambda cat: sum(1 for a in rows_attributions if a['category'] == cat)
    environment = get('missing_dependency') + get('missing_fixture')
    command_timeouts = get('command_timeout')
    return {
        'environment_errors': environment,
        'command_timeouts': command_timeouts,
        'session_timeouts': get('session_timeout'),
        'context_crashes': get('context_crash'),
        'patch_broken_imports': get('patch_import'),
        'undiagnosed_failures': get('undiagnosed'),
        'agent_side_failures': get('context_crash') + get('patch_import') + get('session_timeout'),
        # 旧键保留：确证环境故障 + 命令超时（历史 summary 与 summarize_experiments.py 仍在读）
        'environment_errors_or_timeouts': environment + command_timeouts,
    }
