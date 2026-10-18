"""Prepare matched 10/20-call development runs from the proven official smoke."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
from twostage_layout import (
    AUDITOR_PROMPT as TWO_STAGE_AUDITOR_PROMPT,
    AUDITOR_YAML as TWO_STAGE_AUDITOR_YAML,
    FIXER_ADDENDUM as TWO_STAGE_FIXER_ADDENDUM,
    FIXER_YAML as TWO_STAGE_FIXER_YAML,
    ROOT_YAML as TWO_STAGE_ROOT_YAML,
)

parser = argparse.ArgumentParser()
parser.add_argument('--suite', choices=['single', 'dev', 'expand20', 'expand20b', 'promptv2', 'noise20', 'hygiene20', 'maxtok8k', 'val20', 'think1k', 'review20', 'twostage20', 'ts28', 'b28'], default='single')
args = parser.parse_args()
task_ids = {
    'single': ['requests_6589'],
    'dev': ['fastapi_11194', 'httpx_3672', 'requests_6589', 'rich_2725'],
    'expand20': ['fastapi_11355', 'requests_6592', 'rich_2943', 'fastapi_12942',
                 'requests_6629', 'rich_3006', 'fastapi_13207', 'requests_6644'],
    'expand20b': ['fastapi_11355', 'fastapi_12942', 'fastapi_13207'],
    'promptv2': ['fastapi_11194', 'httpx_3672', 'requests_6589', 'rich_2725',
                 'fastapi_11355', 'requests_6592', 'rich_2943', 'fastapi_12942',
                 'requests_6629', 'rich_3006', 'fastapi_13207', 'requests_6644'],
    'noise20': ['fastapi_11194', 'fastapi_11355', 'fastapi_12942', 'fastapi_13207'],
    'hygiene20': ['fastapi_11194', 'fastapi_11355', 'fastapi_12942', 'fastapi_13207'],
    'maxtok8k': ['fastapi_11194', 'fastapi_11355', 'fastapi_12942', 'fastapi_13207'],
    'val20': ['fastapi_13537', 'requests_7502', 'rich_3043', 'fastapi_13713', 'requests_7205',
              'rich_3052', 'fastapi_13786', 'requests_7309', 'rich_3061', 'fastapi_13920',
              'requests_7315', 'rich_3063', 'fastapi_14077', 'requests_7328', 'rich_3105',
              'fastapi_14099', 'requests_7427', 'rich_3067', 'fastapi_14246', 'requests_7433'],
    'think1k': ['fastapi_13537', 'requests_7502', 'rich_3043', 'fastapi_13713', 'requests_7205',
                'rich_3052', 'fastapi_13786', 'requests_7309', 'rich_3061', 'fastapi_13920',
                'requests_7315', 'rich_3063', 'fastapi_14077', 'requests_7328', 'rich_3105',
                'fastapi_14099', 'requests_7427', 'rich_3067', 'fastapi_14246', 'requests_7433'],
    'review20': ['fastapi_13537', 'requests_7502', 'rich_3043', 'fastapi_13713', 'requests_7205',
                 'rich_3052', 'fastapi_13786', 'requests_7309', 'rich_3061', 'fastapi_13920',
                 'requests_7315', 'rich_3063', 'fastapi_14077', 'requests_7328', 'rich_3105',
                 'fastapi_14099', 'requests_7427', 'rich_3067', 'fastapi_14246', 'requests_7433'],
    'twostage20': ['fastapi_13537', 'requests_7502', 'rich_3043', 'fastapi_13713', 'requests_7205',
                   'rich_3052', 'fastapi_13786', 'requests_7309', 'rich_3061', 'fastapi_13920',
                   'requests_7315', 'rich_3063', 'fastapi_14077', 'requests_7328', 'rich_3105',
                   'fastapi_14099', 'requests_7427', 'rich_3067', 'fastapi_14246', 'requests_7433'],
    'ts28': ['fastapi_13713', 'fastapi_13786', 'fastapi_13920', 'fastapi_14099', 'fastapi_14246',
             'requests_7328', 'rich_3052', 'rich_3061', 'rich_3063', 'rich_3105'],
    'b28': ['fastapi_13537', 'requests_7502', 'rich_3043', 'fastapi_13713', 'requests_7205',
            'rich_3052', 'fastapi_13786', 'requests_7309', 'rich_3061', 'fastapi_13920',
            'requests_7315', 'rich_3063', 'fastapi_14077', 'requests_7328', 'rich_3105',
            'fastapi_14099', 'requests_7427', 'rich_3067', 'fastapi_14246', 'requests_7433'],
}[args.suite]
arms = (10, 20) if args.suite in ('single', 'dev') else ((28,) if args.suite in ('ts28', 'b28') else (20,))

PROMPT_PROTOCOL = """
<!-- development addendum, appended for this experiment -->

## Budget & Verification Protocol

1. **Land the source edit early.** Once you have identified the target file and function, apply your best candidate fix with `edit_file` before continuing to explore. Do not postpone the edit until exploration is finished; the tool budget may not cover a late edit.
2. **Obey budget and time signals.** When a tool observation reports remaining calls (for example "Only N tool call(s) remaining") or your elapsed time exceeds about 3 minutes, stop exploring immediately: finish the pending source edit, run at most one quick targeted check, and call `submit_patch`.
3. **Verify end-to-end before submitting.** Reproduce the issue statement's exact scenario (the same request, command, or API call it describes) and show it failing before your fix and passing after it. Calling an internal helper directly with hand-built inputs is not an end-to-end reproduction. If you cannot demonstrate that the original failure is gone, keep fixing; do not submit an unverified patch.
"""

PROMPT_PROTOCOL_CODE = f'''
# Development experiment: append the budget & verification protocol to the system prompt.
import hashlib as _hashlib
import json as _json
PROMPT_PROTOCOL_TEXT = {PROMPT_PROTOCOL!r}
_prompt_path = AGENT_DIR / 'prompts' / 'system.md'
_original_prompt = _prompt_path.read_text(encoding='utf-8')
_prompt_path.write_text(_original_prompt + PROMPT_PROTOCOL_TEXT, encoding='utf-8')
(WORKING_DIR / 'prompt_protocol.json').write_text(_json.dumps({{
    'addendum': PROMPT_PROTOCOL_TEXT,
    'system_md_sha256_before': _hashlib.sha256(_original_prompt.encode('utf-8')).hexdigest(),
    'system_md_sha256_after': _hashlib.sha256((_original_prompt + PROMPT_PROTOCOL_TEXT).encode('utf-8')).hexdigest(),
}}, indent=2))
print('Prompt protocol appended and hashed.')
'''

HYGIENE_PROTOCOL = """
<!-- development addendum (tool hygiene), appended for this experiment -->

## Tool-Hygiene Protocol

1. **Make every edit call valid and incremental.** `edit_file` requires `filepath`, `old_string`, and `new_string`; `write_file` requires `filepath` and `content`. Always provide all required fields, keep each edit small, and if a call fails its parameter check, fix the missing field before retrying - never resend the same malformed call.
2. **Do not keep retrying failing shell commands.** If a `run_command` returns an error (for example `grep`), try at most one variation, then switch to `read_file` or the code-search tools instead of repeating the same command.
3. **Keep scratch scripts out of the repository.** Write reproduction or temporary scripts under `/tmp` (for example `/tmp/repro.py`), never in the repository root, so they cannot end up in the submitted patch.
"""

HYGIENE_PROTOCOL_CODE = f'''
# Development experiment: append the tool-hygiene protocol to the system prompt.
import hashlib as _hashlib
import json as _json
HYGIENE_PROTOCOL_TEXT = {HYGIENE_PROTOCOL!r}
_prompt_path = AGENT_DIR / 'prompts' / 'system.md'
_original_prompt = _prompt_path.read_text(encoding='utf-8')
_prompt_path.write_text(_original_prompt + HYGIENE_PROTOCOL_TEXT, encoding='utf-8')
(WORKING_DIR / 'prompt_hygiene.json').write_text(_json.dumps({{
    'addendum': HYGIENE_PROTOCOL_TEXT,
    'system_md_sha256_before': _hashlib.sha256(_original_prompt.encode('utf-8')).hexdigest(),
    'system_md_sha256_after': _hashlib.sha256((_original_prompt + HYGIENE_PROTOCOL_TEXT).encode('utf-8')).hexdigest(),
}}, indent=2))
print('Hygiene protocol appended and hashed.')
'''

REVIEW_PROTOCOL = """
<!-- development addendum (pre-submission diff audit), appended for this experiment -->

## Budgeted Diff Audit Before Submission

The most common failure of this configuration is an incomplete diff: a fix that covers only the first file it touches, a new parameter that no caller passes, or a planned edit that never actually landed. Before you call `submit_patch`, audit the diff itself - not your memory of it - while you still have calls left to fix what the audit finds.

### Timeline (20-call budget; this overrides "submit immediately" until the audit is done)
- Calls 1-12: locate the target and land your implementation edits.
- Call 14 at the latest: run the audit below with `git diff`. If you are about to submit before call 13, run the audit first.
- Calls 15-19: land any missing edits the audit found.
- Submit by call 20 (or as soon as the audit is clean). Never end a session without submitting a non-empty patch.

### The audit (read the REAL diff, not your memory)
Run `git diff` (for long diffs: `git diff --stat`, then `git diff -- <file>`) and check the actual hunks:

1. **Requirement coverage** - For every behavior, parameter, status code, error message, or signature the issue asks for, point to the hunk that implements it. If you cannot point to a hunk, the requirement is NOT covered.
2. **Wiring** - If you added or changed a parameter, argument, attribute, or function signature, search all call sites, constructors, and sibling classes (`grep -rn` or `search_similar_code`) and confirm each one passes or handles it. A new parameter that no caller passes is not wired, and a fix to one sibling class that ignores the other siblings is incomplete.
3. **Failed or missing edits** - If any earlier `edit_file`/`write_file` returned an error, or you intended an edit you never landed, verify the file content NOW contains the intended change (re-read it or grep for the new symbol). Land every absent intended change before submitting.
4. **Behavior evidence** - Name the exact targeted command (test or reproduction) that exercises the changed behavior and its observed result.

### After the audit
- Gaps found: land the missing edits FIRST, re-check with `git diff --stat` (one call), then submit.
- Empty diff at the audit deadline: stop exploring and land your best candidate fix in the most likely target file now, then re-run the audit.
- Clean audit: submit immediately. Do not add exploratory reading after a clean audit, and do not repeat the audit more than once.
"""

REVIEW_PROTOCOL_CODE = f'''
# Development experiment: append the pre-submission diff audit to the system prompt.
import hashlib as _hashlib
import json as _json
REVIEW_PROTOCOL_TEXT = {REVIEW_PROTOCOL!r}
_prompt_path = AGENT_DIR / 'prompts' / 'system.md'
_original_prompt = _prompt_path.read_text(encoding='utf-8')
_prompt_path.write_text(_original_prompt + REVIEW_PROTOCOL_TEXT, encoding='utf-8')
(WORKING_DIR / 'prompt_review.json').write_text(_json.dumps({{
    'addendum': REVIEW_PROTOCOL_TEXT,
    'system_md_sha256_before': _hashlib.sha256(_original_prompt.encode('utf-8')).hexdigest(),
    'system_md_sha256_after': _hashlib.sha256((_original_prompt + REVIEW_PROTOCOL_TEXT).encode('utf-8')).hexdigest(),
}}, indent=2))
print('Review protocol appended and hashed.')
'''

TWO_STAGE_CODE = f'''
# Development experiment: two-stage fixer->auditor submission layout.
import hashlib as _hashlib
import json as _json
TWO_STAGE_ROOT_YAML = {TWO_STAGE_ROOT_YAML!r}
TWO_STAGE_FIXER_YAML = {TWO_STAGE_FIXER_YAML!r}
TWO_STAGE_AUDITOR_YAML = {TWO_STAGE_AUDITOR_YAML!r}
TWO_STAGE_FIXER_ADDENDUM = {TWO_STAGE_FIXER_ADDENDUM!r}
TWO_STAGE_AUDITOR_PROMPT = {TWO_STAGE_AUDITOR_PROMPT!r}
_root_agent_path = AGENT_DIR / 'agent.yaml'
_original_root = _root_agent_path.read_text(encoding='utf-8')
_system_md = (AGENT_DIR / 'prompts' / 'system.md').read_text(encoding='utf-8')
(AGENT_DIR / 'prompts' / 'fixer.md').write_text(_system_md + TWO_STAGE_FIXER_ADDENDUM, encoding='utf-8')
(AGENT_DIR / 'prompts' / 'auditor.md').write_text(TWO_STAGE_AUDITOR_PROMPT, encoding='utf-8')
(AGENT_DIR / 'sub_agents' / 'fixer.yaml').write_text(TWO_STAGE_FIXER_YAML, encoding='utf-8')
(AGENT_DIR / 'sub_agents' / 'auditor.yaml').write_text(TWO_STAGE_AUDITOR_YAML, encoding='utf-8')
_root_agent_path.write_text(TWO_STAGE_ROOT_YAML, encoding='utf-8')
(WORKING_DIR / 'two_stage_layout.json').write_text(_json.dumps({{
    'root_agent_yaml': TWO_STAGE_ROOT_YAML,
    'root_agent_sha256_before': _hashlib.sha256(_original_root.encode('utf-8')).hexdigest(),
    'root_agent_sha256_after': _hashlib.sha256(TWO_STAGE_ROOT_YAML.encode('utf-8')).hexdigest(),
    'fixer_prompt_chars': len(_system_md + TWO_STAGE_FIXER_ADDENDUM),
    'auditor_prompt_chars': len(TWO_STAGE_AUDITOR_PROMPT),
}}, indent=2))
print('Two-stage fixer->auditor layout written (fixer has no submit_patch; auditor is the only submitter).')
'''

BUDGET28_EVAL_CONFIG_CODE = '''
# Development experiment: submission-level eval_config.yaml (baseline-consistent; used by ts28 only).
# NOTE: the original ts28 run actually executed with max_turns=500 (see RUNBOOK 2026-10-14 傍晚);
# this constant now keeps max_turns=50 so any regeneration stays baseline-consistent and file==config.
import yaml as _yaml
(AGENT_DIR / 'eval_config.yaml').write_text(_yaml.safe_dump(
    {'evaluation': {'timeout_seconds': 300, 'max_tool_calls': 28,
                    'max_time_minutes': 6.5, 'max_turns': 50}}, sort_keys=False), encoding='utf-8')
print('Submission eval_config.yaml written (28 calls / 6.5 min / 300 s / 50 turns).')
'''

NOISE_BLOCK = """# Noise baseline: three repetitions of the same tasks, one output arm per repetition.
import hashlib
import json
from dataclasses import asdict, replace

base_results = WORKING_DIR / 'results'
smoke_results = []
for rep in (1, 2, 3):
    arm = WORKING_DIR / f'tools_20_r{rep}'
    arm.mkdir(exist_ok=True)
    budget_data['evaluation']['max_tool_calls'] = 20
    budget_path.write_text(yaml.safe_dump(budget_data, sort_keys=False))
    arm_config = replace(eval_config, results_dir=arm / 'results', max_tool_calls=20)
    assert arm_config.budget.tool_calls == 20
    (arm / 'eval_config.json').write_text(json.dumps(asdict(arm_config), indent=2, default=str))
    shutil.copytree(AGENT_DIR, arm / 'submission_config', dirs_exist_ok=True)
    for idx, task in enumerate(SAMPLE_TASKS, start=1):
        evaluator = Evaluator(arm_config)
        result = run_sync(evaluator.evaluate_task, task=task, task_index=idx,
                          total_tasks=len(SAMPLE_TASKS))
        folder = arm / task.instance_id
        folder.mkdir(exist_ok=True)
        patch = result.agent_patch or ''
        (folder / 'agent.patch').write_text(patch, encoding='utf-8')
        (folder / 'test_output.txt').write_text(result.test_output or '', encoding='utf-8')
        (folder / 'result.json').write_text(result.model_dump_json(indent=2, exclude={'trace'}))
        entry = {'task_id': task.instance_id, 'max_tool_calls': 20, 'repeat': rep,
                 'resolved': result.resolved, 'test_exit_code': result.test_exit_code,
                 'error': result.error, 'patch_chars': len(patch),
                 'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                 'tool_calls': result.tool_calls, 'duration_seconds': result.duration_seconds}
        smoke_results.append(entry)
        (WORKING_DIR / 'comparison_results.json').write_text(json.dumps(smoke_results, indent=2))
        print(json.dumps(entry), flush=True)
        evaluator.docker.cleanup_all()
print('Noise baseline completed; no leaderboard submission or training.')
"""
source = Path('artifacts/budget5_run')
destination = Path(f'artifacts/tool_control_v2_{args.suite}')
destination.mkdir(exist_ok=True)
notebook = nbformat.read(source / 'smoke.ipynb', as_version=4)
dependency_cell = nbformat.read('artifacts/verification_run/verify.ipynb', as_version=4).cells[1].source
dependency_repair = dependency_cell[dependency_cell.index('# Development-only dependency repair:'):]
workspace_repair = Path('scripts/workspace_install_repair.py').read_text(encoding='utf-8')
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    code = cell.source
    if 'Wheelhouse installation complete.' in code:
        code += '\n' + dependency_repair
    if "budget_data['evaluation']['max_time_minutes'] = 5" in code:
        code = code.replace("budget_data['evaluation']['max_time_minutes'] = 5",
                            "budget_data['evaluation']['max_time_minutes'] = 5\nbudget_data['evaluation']['timeout_seconds'] = 300")
    if args.suite == 'promptv2' and 'shutil.copytree(SAMPLE_SUBMISSION_SRC, AGENT_DIR)' in code:
        code += PROMPT_PROTOCOL_CODE
    if args.suite == 'hygiene20' and 'shutil.copytree(SAMPLE_SUBMISSION_SRC, AGENT_DIR)' in code:
        code += HYGIENE_PROTOCOL_CODE
    if args.suite == 'review20' and 'shutil.copytree(SAMPLE_SUBMISSION_SRC, AGENT_DIR)' in code:
        code += REVIEW_PROTOCOL_CODE
    if args.suite == 'twostage20' and 'shutil.copytree(SAMPLE_SUBMISSION_SRC, AGENT_DIR)' in code:
        code += TWO_STAGE_CODE
    if args.suite == 'ts28' and 'shutil.copytree(SAMPLE_SUBMISSION_SRC, AGENT_DIR)' in code:
        code = code.replace("budget_data['evaluation']['max_time_minutes'] = 5",
                            "budget_data['evaluation']['max_time_minutes'] = 6.5")
        code += TWO_STAGE_CODE
        code += BUDGET28_EVAL_CONFIG_CODE
    if args.suite == 'b28' and 'shutil.copytree(SAMPLE_SUBMISSION_SRC, AGENT_DIR)' in code:
        code = code.replace("budget_data['evaluation']['max_time_minutes'] = 5",
                            "budget_data['evaluation']['max_time_minutes'] = 6.5")
    if args.suite == 'maxtok8k' and 'max_output_tokens: 16384' in code:
        code = code.replace('max_output_tokens: 16384', 'max_output_tokens: 8192')
    if args.suite == 'think1k' and 'thinking_budget: 4096' in code:
        code = code.replace('thinking_budget: 4096', 'thinking_budget: 1024')
    if 'evaluator = Evaluator(eval_config)' in code:
        code = workspace_repair + '\n' + code
        start = code.index('# Select a small subset')
        end = code.index('limits, gen_constraints', start)
        code = code[:start] + f"TASK_IDS = {task_ids!r}\nSAMPLE_TASKS = [next(t for t in tasks if t.instance_id == i) for i in TASK_IDS]\n" + code[end:]
        start = code.index('evaluator = Evaluator(eval_config)')
        arm_block = NOISE_BLOCK if args.suite in ('noise20', 'hygiene20', 'maxtok8k') else """# Matched arms share one model server; each task gets a fresh evaluator sandbox.
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, replace

base_results = WORKING_DIR / 'results'
smoke_results = []
predictions = []
for tool_limit in (10, 20):
    arm = WORKING_DIR / f'tools_{tool_limit}'
    arm.mkdir(exist_ok=True)
    budget_data['evaluation']['max_tool_calls'] = tool_limit
    budget_path.write_text(yaml.safe_dump(budget_data, sort_keys=False))
    arm_config = replace(eval_config, results_dir=arm / 'results', max_tool_calls=tool_limit)
    assert arm_config.budget.tool_calls == tool_limit
    (arm / 'eval_config.json').write_text(json.dumps(asdict(arm_config), indent=2, default=str))
    shutil.copytree(AGENT_DIR, arm / 'submission_config', dirs_exist_ok=True)
    for idx, task in enumerate(SAMPLE_TASKS, start=1):
        evaluator = Evaluator(arm_config)
        result = run_sync(evaluator.evaluate_task, task=task, task_index=idx,
                          total_tasks=len(SAMPLE_TASKS))
        folder = arm / task.instance_id
        folder.mkdir(exist_ok=True)
        patch = result.agent_patch or ''
        (folder / 'agent.patch').write_text(patch, encoding='utf-8')
        (folder / 'test_output.txt').write_text(result.test_output or '', encoding='utf-8')
        (folder / 'result.json').write_text(result.model_dump_json(indent=2, exclude={'trace'}))
        entry = {'task_id': task.instance_id, 'max_tool_calls': tool_limit,
                 'resolved': result.resolved, 'test_exit_code': result.test_exit_code,
                 'error': result.error, 'patch_chars': len(patch),
                 'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                 'tool_calls': result.tool_calls, 'duration_seconds': result.duration_seconds}
        smoke_results.append(entry)
        (WORKING_DIR / 'comparison_results.json').write_text(json.dumps(smoke_results, indent=2))
        print(json.dumps(entry), flush=True)
        evaluator.docker.cleanup_all()
print('Development comparison completed; no leaderboard submission or training.')
"""
        arm_block = arm_block.replace('for tool_limit in (10, 20):', f'for tool_limit in {arms!r}:')
        code = code[:start] + arm_block
    cell.source = code
    cell.outputs = []
    cell.execution_count = None
notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbformat.validate(notebook)
nbformat.write(notebook, destination / 'smoke.ipynb')
metadata = json.loads((source / 'kernel-metadata.json').read_text())
kernel_names = {'single': ('taruki/gemma4-tool-control-v2-single', 'Gemma4 Tool Control v2 single'),
                'dev': ('taruki/gemma4-tool-control-v2-dev', 'Gemma4 Tool Control v2 dev'),
                'expand20': ('taruki/gemma4-tool-control-v2-expand20', 'Gemma4 Tool Control v2 expand20'),
                'expand20b': ('taruki/gemma4-tool-control-v2-expand20b', 'Gemma4 Tool Control v2 expand20b'),
                'promptv2': ('taruki/gemma4-prompt-protocol-20', 'Gemma4 Prompt Protocol 20'),
                'noise20': ('taruki/gemma4-noise-20', 'Gemma4 Noise 20'),
                'hygiene20': ('taruki/gemma4-hygiene-20', 'Gemma4 Hygiene 20'),
                'maxtok8k': ('taruki/gemma4-maxtok8k-20', 'Gemma4 Maxtok8k 20'),
                'val20': ('taruki/gemma4-val20', 'Gemma4 Val20'),
                'think1k': ('taruki/gemma4-think1k', 'Gemma4 Think1k'),
                'review20': ('taruki/gemma4-review20', 'Gemma4 Review20'),
                'twostage20': ('taruki/gemma4-twostage20', 'Gemma4 Twostage20'),
                'ts28': ('taruki/gemma4-ts28', 'Gemma4 Ts28'),
                'b28': ('taruki/gemma4-b28', 'Gemma4 B28')}
metadata.update(id=kernel_names[args.suite][0], title=kernel_names[args.suite][1])
metadata['dataset_sources'].append('taruki/gemma4-requests-test-wheels')
(destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
manifest = {'kernel': metadata['id'], 'task_ids': task_ids, 'arms': list(arms),
            'max_time_minutes': 6.5 if args.suite in ('ts28', 'b28') else 5, 'timeout_seconds': 300, 'training': False,
            'model': 'gemma-4-31b-it-qat-w4a16-ct', 'model_source': metadata['model_sources'],
            'docker_image': metadata['docker_image'],
            'environment_repair': 'requests test fixture dependencies; applies equally to both arms',
            'workspace_import_assertion': True,
            'order': '10-call arm then 20-call arm; descriptive comparison, not statistical proof',
            'notebook_sha256': hashlib.sha256((destination / 'smoke.ipynb').read_bytes()).hexdigest()}
if args.suite == 'expand20':
    manifest['order'] = 'single 20-call arm; expansion of the dev set, descriptive only'
    manifest['selection'] = 'answer-blind round-robin across fastapi/requests/rich unused public tasks, instance_id ascending'
    manifest['reuses'] = 'artifacts/tool_control_v2_dev_outputs tools_20 rows for fastapi_11194/httpx_3672/requests_6589/rich_2725'
if args.suite == 'expand20b':
    manifest['order'] = 'single 20-call arm; environment-fix re-run of the three env-blocked fastapi tasks'
    manifest['selection'] = 'same answer-blind task list as expand20, restricted to the tasks blocked by missing test dependencies'
    manifest['environment_fix'] = 'wheelhouse v4 adds inline-snapshot==0.21.1 and dirty-equals==0.9.0; validated in artifacts/dev_environment_v5_outputs'
    manifest['reuses'] = 'artifacts/tool_control_v2_expand20_outputs tools_20 rows for requests_6592/rich_2943/requests_6629/rich_3006/requests_6644'
if args.suite == 'promptv2':
    manifest['order'] = 'single 20-call arm; prompt-protocol experiment on the same 12 development tasks'
    manifest['change'] = 'appended a Budget & Verification Protocol to prompts/system.md; all other config identical to the baseline runs'
    manifest['baseline'] = 'artifacts/tool_control_v2_dev_outputs + tool_control_v2_expand20_outputs + tool_control_v2_expand20b_outputs (20-call rows, unchanged prompt)'
    manifest['prompt_protocol'] = PROMPT_PROTOCOL
if args.suite == 'noise20':
    manifest['order'] = 'three repetitions of four fastapi tasks, one output arm per repetition, 20 calls'
    manifest['purpose'] = 'noise-floor estimation for judging later A/B; unchanged prompt and config'
if args.suite == 'hygiene20':
    manifest['order'] = 'three repetitions of four fastapi tasks, one output arm per repetition, 20 calls'
    manifest['change'] = 'appended a Tool-Hygiene Protocol to prompts/system.md; all other config identical to noise20'
    manifest['baseline'] = 'artifacts/tool_control_v2_noise20_outputs (same tasks and repetitions, unchanged prompt)'
    manifest['hygiene_protocol'] = HYGIENE_PROTOCOL
if args.suite == 'maxtok8k':
    manifest['order'] = 'three repetitions of four fastapi tasks, one output arm per repetition, 20 calls'
    manifest['change'] = 'configs/sampling.yaml max_output_tokens 16384 -> 8192 (input ceiling 16384 -> 24576 tokens; targets ContextWindowExceeded crashes seen in noise20/promptv2)'
    manifest['baseline'] = 'artifacts/tool_control_v2_noise20_outputs (same tasks and repetitions, unchanged config)'
if args.suite == 'val20':
    manifest['order'] = 'single 20-call arm over 20 holdout public tasks (7 fastapi / 7 requests / 6 rich), one run each'
    manifest['purpose'] = 'credible baseline on tasks never used for tuning; selection = answer-blind round-robin across repos from the unused pool, instance_id ascending'
    manifest['config'] = 'unchanged baseline: sample agent/prompt/adapters, 20 tools, 5 min per task, 300 s command timeout'
if args.suite == 'think1k':
    manifest['order'] = 'single 20-call arm over the SAME 20 val20 tasks, one run each'
    manifest['change'] = 'configs/sampling.yaml thinking_config.thinking_budget 4096 -> 1024; everything else identical to val20'
    manifest['baseline'] = 'artifacts/tool_control_v2_val20_outputs (same tasks and budget, unchanged config)'
    manifest['hypothesis'] = ('val20 failures spend 254 s vs 165 s of inter-turn wall in the same ~17 turns; single thinking bursts reach 84-150 s '
                              '(up to 11908 chars, including degenerate repetition loops). Capping thought tokens should free turns for edits within the 5-min budget.')
if args.suite == 'review20':
    manifest['order'] = 'single 20-call arm over the SAME 20 val20 tasks, one run each'
    manifest['change'] = 'appended a Budgeted Diff Audit Before Submission to prompts/system.md; everything else identical to val20'
    manifest['baseline'] = 'artifacts/tool_control_v2_val20_outputs (same tasks and budget, unchanged prompt)'
    manifest['hypothesis'] = ('the 9 common failures include 4 incomplete-coverage cases (fix stops at the first file, parameter never wired, failed edit submitted as-is). '
                              'A pre-submission audit of the real git diff, with a hard deadline (call 14) so gaps can still be fixed, should raise both coverage and pass count.')
    manifest['review_protocol'] = REVIEW_PROTOCOL
if args.suite == 'twostage20':
    manifest['order'] = 'single 20-call arm over the SAME 20 val20 tasks, one run each'
    manifest['change'] = ('root agent becomes SequentialAgent[fixer, auditor]: the fixer has the baseline prompt + a handoff addendum and NO submit_patch tool, '
                          'the auditor is the only agent with submit_patch and must read the real git diff, land gaps, and submit; everything else identical to val20')
    manifest['baseline'] = 'artifacts/tool_control_v2_val20_outputs (same tasks and budget, unchanged single-agent submission)'
    manifest['hypothesis'] = ('the review protocol as prompt text was ignored (audit ran 6/20, zero gap-filling). Making the audit a structural workflow stage '
                              '(fixer cannot submit) should force the review to happen and convert incomplete-coverage failures.')
    manifest['two_stage'] = {'root': TWO_STAGE_ROOT_YAML, 'fixer': TWO_STAGE_FIXER_YAML,
                             'auditor': TWO_STAGE_AUDITOR_YAML}
if args.suite == 'ts28':
    manifest['order'] = 'single arm over the 10 tasks that failed in all three prior configs (val20, review20, twostage20), one run each'
    manifest['change'] = ('two-stage fixer->auditor structure (same as twostage20) PLUS raised budgets: max_tool_calls 20 -> 28, max_time_minutes 5 -> 6.5 '
                          '(dev arm and submission eval_config.yaml), so the auditor stage gets room to act; everything else identical')
    manifest['baseline'] = 'artifacts/tool_control_v2_val20_outputs (same tasks: 0/10 resolved in val20, review20 and twostage20)'
    manifest['hypothesis'] = ('twostage20 proved the auditor runs (12/20) but is starved (2.3 calls, 0 edits, git diff budget-denied on 3 tasks). '
                              'Giving the two-stage workflow budget headroom should let the auditor read the diff and land gaps, converting coverage failures.')
    manifest['decision_rule'] = ('adopt-for-full-confirmation if >= 3/10 convert with auditor edits on >= 2 converted tasks; reject if <= 1 converts; '
                                 '2 converts -> repeat after quota refresh. Note: 6.5 min cap is a dev-measurement setting; final-submission budget needs its own 12h-cap review.')
if args.suite == 'b28':
    manifest['order'] = 'single 28-call arm over the SAME 20 val20 tasks, one run each; single-agent frozen baseline structure'
    manifest['change'] = ('JOINT budget adjustment only: max_tool_calls 20 -> 28 AND max_time_minutes 5 -> 6.5 (dev arm + submission eval_config.yaml). '
                          'Model, adapter, prompts, sampling, context, max_turns (50, unchanged), command timeout (300 s) all identical to val20. '
                          'No new prompt text, no review stage, no structure change.')
    manifest['baseline'] = ('historical comparison: artifacts/tool_control_v2_val20_outputs (10/20, single run estimate). '
                            'This is a HISTORICAL baseline, NOT a same-period repeat. The 13/20 threshold in the prereg is a decision rule for whether to keep investing, not a significance test.')
    manifest['config_consistency'] = ('The Evaluator config is built in cell 12 from the submission eval_config.yaml written in cell 6; '
                                      'the arm loop then rewrites the file with the same values it puts into the arm config, so file == Evaluator config. '
                                      'Post-run check: compare arm eval_config.json budget vs submission_config/eval_config.yaml (time/tool_calls/turns/timeout).')
    manifest['hypothesis'] = ('ts28 (two-stage + same joint budget) freed the auditor but converted only fastapi_14246, on the fixer side; that single conversion cannot be attributed to time alone. '
                              'Test the simpler arm: does the joint budget raise help the FROZEN BASELINE single-agent structure itself, without any structure or prompt change?')
    manifest['prereg'] = 'artifacts/b28_prereg.md (criteria and stop conditions, written before launch)'
    manifest['decision_rule'] = ('pre-registered: resolved >= 13/20 -> worth continuing (then repeats/full confirmation needed, not a claim yet); 11-12 -> ambiguous, no claim without >=3x repeats on the changed tasks; '
                                 '<= 10 -> no gain, reject. Joint budget change: no attribution to time vs calls separately.')
(destination / 'experiment.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
