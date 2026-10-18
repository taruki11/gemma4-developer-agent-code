"""Development-only repair for the official subprocess backend's skipped install.

Executed inside a Kaggle notebook. Installs into each sandbox venv, verifies
import origin, and leaves tests and patch application to the official verifier.
"""
import json
import shlex
from pathlib import Path

import swegemma.harness.agent_runner as agent_runner
import swegemma.harness.container_setup as container_setup
import swegemma.harness.verification as verification

installation_audit = []
original_install = container_setup.install_editable_package


def install_workspace_package(manager, sandbox_id):
    if type(manager).__name__ not in {'SubprocessManager', 'SubprocessSandbox'}:
        return original_install(manager, sandbox_id)
    extra_wheelhouse = next(p for p in (
        Path('/kaggle/input/gemma4-requests-test-wheels'),
        Path('/kaggle/input/datasets/taruki/gemma4-requests-test-wheels'),
    ) if p.is_dir())
    extra_links = f'--find-links={shlex.quote(str(extra_wheelhouse))} '
    # The competition Dockerfile preinstalls these backends, whereas the
    # notebook host does not. Use the official competition wheels offline.
    backend_result = manager.exec(sandbox_id,
        'python3 -m pip install --no-index '
        '--find-links=/kaggle/input/competitions/gemma-4-developer-agent/wheels '
        + extra_links + 'pdm-backend hatchling poetry-core flit-core editables', timeout=300)
    installation_audit.append({'sandbox_id': sandbox_id, 'stage': 'build_backends',
                               'exit_code': backend_result.exit_code,
                               'stdout': backend_result.stdout, 'stderr': backend_result.stderr})
    Path('/kaggle/working/workspace_install_audit.json').write_text(json.dumps(installation_audit, indent=2))
    if backend_result.exit_code != 0:
        raise RuntimeError(f'Offline build backend installation failed: {backend_result.stderr}')
    installed = manager.exec(sandbox_id,
        'cd /workspace && python3 -m pip install --no-index '
        '--find-links=/kaggle/input/competitions/gemma-4-developer-agent/wheels '
        + extra_links + '--no-build-isolation -e /workspace', timeout=300)
    installation_audit.append({'sandbox_id': sandbox_id, 'stage': 'editable_install',
                               'exit_code': installed.exit_code,
                               'stdout': installed.stdout, 'stderr': installed.stderr})
    Path('/kaggle/working/workspace_install_audit.json').write_text(json.dumps(installation_audit, indent=2))
    if installed.exit_code != 0:
        raise RuntimeError(f'Workspace editable installation failed: {installed.stderr}')
    probe = '''import importlib, json, pathlib, sys
ws = pathlib.Path('/workspace').resolve()
assert pathlib.Path(sys.prefix).resolve() != pathlib.Path(sys.base_prefix).resolve(), 'Expected isolated sandbox venv'
module_names = [name for name in ('requests', 'httpx', 'fastapi', 'rich')
                if (ws / name / '__init__.py').exists() or (ws / 'src' / name / '__init__.py').exists()]
assert len(module_names) == 1, module_names
module = importlib.import_module(module_names[0])
origin = pathlib.Path(module.__file__).resolve()
assert origin.is_relative_to(ws), f'Wrong module origin: {origin}; workspace={ws}'
print(json.dumps({'module': module_names[0], 'origin': str(origin), 'workspace': str(ws)}))
'''
    checked = manager.exec(sandbox_id, 'cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -c ' + shlex.quote(probe), timeout=300)
    installation_audit.append({'sandbox_id': sandbox_id, 'stage': 'import_origin',
                               'exit_code': checked.exit_code,
                               'stdout': checked.stdout, 'stderr': checked.stderr})
    Path('/kaggle/working/workspace_install_audit.json').write_text(json.dumps(installation_audit, indent=2))
    if checked.exit_code != 0:
        raise RuntimeError(f'Workspace import-origin verification failed: {checked.stderr}')


container_setup.install_editable_package = install_workspace_package
agent_runner.install_editable_package = install_workspace_package
verification.install_editable_package = install_workspace_package
