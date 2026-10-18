"""CPU environment sanity check with official reference patches, never an Agent input."""
import argparse
import json
from pathlib import Path

import nbformat

parser = argparse.ArgumentParser()
parser.add_argument('--destination', type=Path, default=Path('artifacts/dev_environment_v5_run'))
parser.add_argument('--tasks', default=('fastapi_11194,httpx_3672,rich_2725,'
                                        'fastapi_11355,fastapi_12942,fastapi_13207'))
args = parser.parse_args()
task_tuple = tuple(t.strip() for t in args.tasks.split(',') if t.strip())

source = Path('artifacts/verification_run')
destination = args.destination
destination.mkdir(exist_ok=True)
notebook = nbformat.read(source / 'verify.ipynb', as_version=4)
cell = notebook.cells[-1]
code = cell.source
start = code.index('patch = ')
end = code.index('audit = []', start)
code = code[:start] + code[end:]
code = code.replace('for timeout in (60, 300):',
    f"for task_id in {task_tuple!r}:\n"
    "    task = next(t for t in load_tasks(data / 'tasks.jsonl') if t.instance_id == task_id)\n"
    "    timeout = 300")
code = code.replace("data / 'snapshots' / 'requests_6589.tgz', agent_patch=patch", "data / 'snapshots' / f'{task.instance_id}.tgz', agent_patch=task.patch")
code = code.replace("'reference_patch_used': False", "'reference_patch_used': True, 'task_id': task.instance_id, 'agent_run': False")
code = code.replace("'verification_results.json'", "'environment_check_results.json'")
cell.source = code
notebook.cells[0].source = 'CPU environment sanity check using official reference patches only inside the verifier. No Agent, inference, training, or leaderboard submission.'
nbformat.validate(notebook)
nbformat.write(notebook, destination / 'verify.ipynb')
metadata = json.loads((source / 'kernel-metadata.json').read_text())
metadata.update(id='taruki/gemma4-dev-environment-check', title='Gemma4 Dev Environment Check')
(destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(f'Prepared reference-only CPU environment check for {len(task_tuple)} tasks; reference patches are never passed to an Agent.')