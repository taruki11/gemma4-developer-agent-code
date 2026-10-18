"""Recover agent edits from successful tool observations, not reference answers."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

trace_path = Path('artifacts/budget5_outputs/results/traces/trace_requests_6589.json')
trace = json.loads(trace_path.read_text(encoding='utf-8'))
root = Path('artifacts/patch_replay/requests_6589')
if Path('artifacts/agent_patch_requests_6589.patch').exists():
    raise SystemExit('Recovered patch already exists; preserve prior result.')
if not root.exists():
    root.mkdir(parents=True)
    with tarfile.open('data/full/files/snapshots/requests_6589.tgz') as archive:
        archive.extractall(root, filter='data')
repo = next(p.parent for p in root.rglob('.git') if p.is_dir())
subprocess.run(['git', 'config', 'core.autocrlf', 'false'], cwd=repo, check=True)
# Match the official harness: snapshot setup edits become the clean baseline.
subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
subprocess.run(['git', '-c', 'user.name=Replay', '-c', 'user.email=replay@example.invalid',
                'commit', '--allow-empty', '-m', 'Official snapshot replay baseline'], cwd=repo, check=True, capture_output=True)
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip() == b''
replayed = []
for step in trace['steps']:
    calls = step.get('tool_calls') or []
    observation = step.get('observation') or {}
    content = observation.get('content', '')
    if not content:
        continue
    try:
        response = json.loads(content)
    except json.JSONDecodeError:
        continue
    if response.get('status') != 'ok':
        continue
    for call in calls:
        name, args = call['function_name'], call['arguments']
        if name not in {'edit_file', 'write_file'}:
            continue
        path = (repo / args['filepath']).resolve()
        assert path.is_relative_to(repo.resolve())
        if name == 'edit_file':
            old = path.read_text(encoding='utf-8')
            occurrences = old.count(args['old_string'])
            assert occurrences == 1 or args.get('allow_multiple'), occurrences
            new = old.replace(args['old_string'], args['new_string'])
        else:
            new = args['content']
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new, encoding='utf-8', newline='\n')
        replayed.append({'tool': name, 'path': args['filepath']})
subprocess.run(['git', 'add', '-N', '.'], cwd=repo, check=True)
patch = subprocess.check_output(['git', '-c', 'core.quotePath=false', 'diff', 'HEAD'], cwd=repo)
destination = Path('artifacts/agent_patch_requests_6589.patch')
destination.write_bytes(patch)
report = {'source_trace': str(trace_path), 'replayed_edits': replayed,
          'patch_bytes': len(patch), 'patch_characters': len(patch.decode('utf-8')),
          'patch_sha256': hashlib.sha256(patch).hexdigest(),
          'provenance': 'reconstructed from successful agent tools on official snapshot; not original captured diff',
          'reference_patch_used': False}
Path('artifacts/patch_replay/manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
