"""Single variable ablation of the official baseline task time budget."""
import hashlib
import json
from pathlib import Path
import nbformat

source = Path('artifacts/kaggle_run')
destination = Path('artifacts/budget5_run')
destination.mkdir(exist_ok=True)
notebook = json.loads((source / 'smoke.ipynb').read_text(encoding='utf-8'))
insertion = """# Single-variable experiment: only task wall-clock budget changes.
import yaml
budget_path = AGENT_DIR / 'eval_config.yaml'
budget_data = yaml.safe_load(budget_path.read_text())
assert budget_data['evaluation']['max_time_minutes'] == 1
assert budget_data['evaluation']['max_tool_calls'] == 10
budget_data['evaluation']['max_time_minutes'] = 5
budget_path.write_text(yaml.safe_dump(budget_data, sort_keys=False))
"""
count = 0
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source'])
    marker = '# Load tasks from the published dataset'
    if marker in code:
        code = code.replace(marker, insertion + '\n' + marker)
        count += 1
    cell['source'] = code.splitlines(keepends=True)
assert count == 1
(destination / 'smoke.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
nbformat.validate(nbformat.read(destination / 'smoke.ipynb', as_version=4))
metadata = json.loads((source / 'kernel-metadata.json').read_text())
metadata.update({'id': 'taruki/gemma4-budget5-smoke', 'title': 'Gemma4 Budget5 Smoke'})
(destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
manifest = {'baseline_kernel': 'taruki/gemma4-official-baseline-smoke',
            'experiment_kernel': metadata['id'], 'task_id': 'requests_6589',
            'changed_parameter': 'evaluation.max_time_minutes', 'baseline': 1, 'experiment': 5,
            'max_tool_calls': 10, 'training': False, 'model': 'gemma-4-31b-it-qat-w4a16-ct',
            'baseline_notebook_sha256': hashlib.sha256((source / 'smoke.ipynb').read_bytes()).hexdigest(),
            'experiment_notebook_sha256': hashlib.sha256((destination / 'smoke.ipynb').read_bytes()).hexdigest()}
(destination / 'experiment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
