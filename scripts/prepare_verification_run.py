"""CPU-only replay of the official verifier, with identical tests and patch."""
import argparse
import json
from pathlib import Path
import nbformat

parser = argparse.ArgumentParser()
parser.add_argument('--diagnostic', action='store_true')
args = parser.parse_args()
destination = Path('artifacts/verification_diagnostic_run' if args.diagnostic else 'artifacts/verification_run')
destination.mkdir(exist_ok=True)
official = json.loads(Path('artifacts/official_current/getting-started-gemma-4-developer-agent.ipynb').read_text())
install = next(''.join(c['source']) for c in official['cells'] if c['cell_type'] == 'code')
install += """
# Development-only dependency repair: official SubprocessManager skips
# install_test_dependencies, and the bundled wheels lack requests test fixtures.
test_wheel_candidates = [Path('/kaggle/input/gemma4-requests-test-wheels'),
                        Path('/kaggle/input/datasets/taruki/gemma4-requests-test-wheels')]
test_wheels = next((p for p in test_wheel_candidates if p.is_dir()), None)
assert test_wheels is not None, f'Missing private test wheelhouse; mounts={list(Path("/kaggle/input").iterdir())}'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                '--find-links', str(test_wheels), 'pytest-httpbin==2.0.0',
                'httpbin==0.10.2', 'trustme==1.2.1', 'PySocks==1.7.1',
                'Flask==2.2.5', 'Werkzeug==2.3.8', 'flasgger==0.9.7.1',
                'dirty-equals==0.9.0', 'inline-snapshot==0.21.1'], check=True)
importlib.invalidate_caches()
"""
patch = Path('artifacts/agent_patch_requests_6589.patch').read_text(encoding='utf-8')
workspace_repair = Path('scripts/workspace_install_repair.py').read_text(encoding='utf-8')
code = """import asyncio
import importlib.metadata
import json
import logging
import time
from pathlib import Path
from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks
import swegemma.harness.verification as verification

logging.basicConfig(level=logging.INFO)
data = Path('/kaggle/input/competitions/gemma-4-developer-agent')
working = Path('/kaggle/working')
(working / 'environment.json').write_text(json.dumps({
    'versions': {name: importlib.metadata.version(name) for name in
                 ['swegemma', 'adk-submission', 'adk-eval-core', 'pytest',
                  'pytest-httpbin', 'httpbin', 'Flask', 'Werkzeug', 'flasgger', 'trustme',
                  'inline-snapshot', 'dirty-equals']},
    'development_environment_repair': True,
    'tests_modified': False,
}, indent=2))
task = next(t for t in load_tasks(data / 'tasks.jsonl') if t.instance_id == 'requests_6589')
assert (data / 'snapshots' / 'requests_6589.tgz').is_file()
patch = PATCH_LITERAL
(working / 'agent.patch').write_text(patch)
audit = []
original_exec = verification.sandbox_exec
async def logged_exec(*args, **kwargs):
    start = time.perf_counter()
    result = await original_exec(*args, **kwargs)
    audit.append({'command': str(args[2]) if len(args) > 2 else str(kwargs.get('command')),
                  'timeout': kwargs.get('timeout'), 'exit_code': result.exit_code,
                  'seconds': time.perf_counter() - start,
                  'stdout': result.stdout, 'stderr': result.stderr})
    (working / 'sandbox_commands.json').write_text(json.dumps(audit, indent=2))
    return result
verification.sandbox_exec = logged_exec
results = []
for timeout in (60, 300):
    cfg = EvalConfig(tasks_path=data / 'tasks.jsonl', snapshots_dir=data / 'snapshots',
                     results_dir=working / f'verify_{timeout}', submission_dir=data / 'sample_submission',
                     models=ModelRegistry(), sandbox='subprocess', timeout_seconds=timeout,
                     wheels_dir=data / 'wheels')
    evaluator = Evaluator(cfg)
    start = time.perf_counter()
    result = await verification.verify_task(evaluator.docker, cfg, task,
                        data / 'snapshots' / 'requests_6589.tgz', agent_patch=patch, start_time=start)
    entry = {'command_timeout_seconds': timeout, 'resolved': result.resolved,
             'test_exit_code': result.test_exit_code, 'error': result.error,
             'duration_seconds': result.duration_seconds, 'test_output': result.test_output,
             'model_inference': False, 'reference_patch_used': False}
    entry['environment_repair'] = 'requests test fixture dependencies installed from private wheelhouse'
    results.append(entry)
    (working / 'verification_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(entry), flush=True)
print('Verification-only run complete; no model was served.')
""".replace('PATCH_LITERAL', repr(patch))
code = code.replace('logging.basicConfig(level=logging.INFO)', workspace_repair + '\nlogging.basicConfig(level=logging.INFO)')
if args.diagnostic:
    code = code.replace('result = await original_exec(*args, **kwargs)',
                        "if len(args) > 2 and '-m pytest ' in args[2]:\n"
                        "        args = (*args[:2], args[2] + ' -vv -o faulthandler_timeout=20', *args[3:])\n"
                        "    result = await original_exec(*args, **kwargs)")
    code = code.replace('for timeout in (60, 300):', "for timeout, patch_kind in ((60, 'agent'), (60, 'empty')):")
    code = code.replace('agent_patch=patch, start_time=start', "agent_patch=patch if patch_kind == 'agent' else '', start_time=start")
    code = code.replace("entry['environment_repair']", "entry['patch_kind'] = patch_kind\n    entry['environment_repair']")
notebook = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell('CPU-only official verification replay. Same agent patch and tests; 60s vs 300s command timeout. No inference or training.'),
    nbformat.v4.new_code_cell(install), nbformat.v4.new_code_cell(code)])
notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata['language_info'] = {'name': 'python'}
nbformat.validate(notebook)
nbformat.write(notebook, destination / 'verify.ipynb')
metadata = json.loads(Path('artifacts/official_current/kernel-metadata.json').read_text())
metadata.update({'id': 'taruki/gemma4-verification-replay', 'title': 'Gemma4 Verification Replay',
                 'code_file': 'verify.ipynb', 'is_private': True, 'enable_gpu': False,
                 'enable_internet': False, 'model_sources': [], 'keywords': []})
if args.diagnostic:
    metadata.update(id='taruki/gemma4-verification-diagnostic', title='Gemma4 Verification Diagnostic')
metadata.pop('id_no', None)
metadata.pop('machine_shape', None)
metadata['dataset_sources'].append('taruki/gemma4-requests-test-wheels')
(destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print('Prepared CPU-only verification; unchanged official verify_task and unchanged grading tests.')
