"""Prepare the Phase 3 training-feasibility spike (LoRA smoke on the QAT checkpoint)."""
import json
from pathlib import Path

import nbformat

OUT = Path('artifacts/training_spike')
OUT.mkdir(exist_ok=True)

md = nbformat.v4.new_markdown_cell(
    'Phase 3 feasibility spike: attach an r=16 LoRA to gemma-4-31b-it-qat-w4a16-ct, run ~10 training steps on '
    'synthetic tokens, save the adapter, then try loading it in vLLM. Development-only; the real model is not '
    'trained here, and nothing is submitted.')

preflight = nbformat.v4.new_code_cell('''import json, torch
from pathlib import Path
paths = {
    'model': Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'),
    'metric_wheelhouse': Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
    'training_wheels': Path('/kaggle/input/datasets/taruki/gemma4-training-wheels'),
    'competition_wheels': Path('/kaggle/input/competitions/gemma-4-developer-agent/wheels'),
}
state = {k: v.exists() for k, v in paths.items()}
state['gpu_count'] = torch.cuda.device_count()
state['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps(state, indent=2))
assert all(v for k, v in state.items() if k != 'gpu_names'), state
assert state['gpu_count'] == 4
Path('/kaggle/working/preflight.json').write_text(json.dumps(state, indent=2))
''')

install = nbformat.v4.new_code_cell('''import glob, importlib, importlib.metadata, os, subprocess, sys
from pathlib import Path

WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
TRAINING_DIR = Path('/kaggle/input/datasets/taruki/gemma4-training-wheels')
COMPETITION_WHEELS = Path('/kaggle/input/competitions/gemma-4-developer-agent/wheels')

for pth_pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                    '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pth_pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass

tmp_whl = Path('/tmp/wheelhouse')
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    target_name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    target = tmp_whl / target_name
    if not target.exists():
        os.symlink(w, target)
wheels = sorted(str(w) for w in tmp_whl.glob('*.whl'))
print(f'Installing {len(wheels)} wheelhouse wheels...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)

print('Installing training wheels...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                '--find-links', str(TRAINING_DIR), '--find-links', str(COMPETITION_WHEELS),
                'peft==0.21.2', 'accelerate==1.15.0', 'huggingface_hub==1.33.0',
                'filelock==4.1.1', 'fsspec==2026.9.0', 'psutil==7.2.2', 'pyyaml==6.0.3',
                'tqdm==4.70.1'], check=True)
importlib.invalidate_caches()
for name in ('peft', 'accelerate', 'transformers', 'bitsandbytes', 'compressed-tensors', 'torch'):
    try:
        print(name, importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        print(name, 'NOT INSTALLED')
''')

train = nbformat.v4.new_code_cell('''import json, time, traceback
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
report = {'spike': 'lora-train-smoke', 'started': time.time()}
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    report['tokenizer_ok'] = True
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='auto')
    report['loaded'] = True
    qc = getattr(model.config, 'quantization_config', None)
    report['quantization_config'] = str(qc)[:400]
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    trainable = [p for p in model.parameters() if p.requires_grad]
    report['trainable_params'] = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    model.train()
    vocab = int(model.config.vocab_size)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for step in range(10):
        ids = torch.randint(0, vocab, (1, 512), device='cuda')
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(round(float(loss.detach()), 4))
        print('step', step, losses[-1], flush=True)
    report['losses'] = losses
    report['peak_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    adapter_dir = Path('/kaggle/working/spike_adapter')
    model.save_pretrained(str(adapter_dir))
    report['adapter_bytes'] = sum(p.stat().st_size for p in adapter_dir.rglob('*') if p.is_file())
    report['adapter_files'] = sorted(str(p.relative_to(adapter_dir)) for p in adapter_dir.rglob('*') if p.is_file())
    report['status'] = 'ok'
    del model
    torch.cuda.empty_cache()
except Exception as exc:
    report['status'] = 'fail'
    report['error'] = f'{type(exc).__name__}: {exc}'
    traceback.print_exc()
Path('/kaggle/working/training_spike.json').write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
''')

vllm_check = nbformat.v4.new_code_cell('''# Best-effort: serve the model with the spike adapter via vLLM and issue one generation.
import json, shutil, traceback
from pathlib import Path

report = json.loads(Path('/kaggle/working/training_spike.json').read_text())
if report.get('status') != 'ok':
    print('spike training failed; skipping vLLM check')
else:
    try:
        from adk_submission import VllmConfig, VllmServer, discover_adapters
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
        MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
        layout = Path('/kaggle/working/spike_layout')
        dest = layout / 'adapters' / 'spike_lora'
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path('/kaggle/working/spike_adapter'), dest)
        adapters = discover_adapters(str(layout), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        print('discovered adapters:', adapters)
        cfg = VllmConfig(model=str(MODEL), port=8001, host='127.0.0.1', tool_call_parser='gemma4',
                         reasoning_parser='gemma4', max_model_len=32768, dtype='bfloat16',
                         gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=True,
                         max_loras=8, max_lora_rank=128, tensor_parallel_size=4, startup_timeout=1200)
        server = VllmServer(cfg, adapter_manifest=adapters)
        server.start()
        import openai
        client = openai.OpenAI(base_url=server.base_url + '/v1', api_key='EMPTY')
        last_error = None
        for model_name in ('spike_lora', 'openai/spike_lora'):
            try:
                response = client.chat.completions.create(model=model_name,
                                                          messages=[{'role': 'user', 'content': 'Say OK.'}],
                                                          max_tokens=8, temperature=0.0)
                text = response.choices[0].message.content
                Path('/kaggle/working/vllm_check.json').write_text(
                    json.dumps({'status': 'ok', 'model_name': model_name, 'reply': text}, indent=2))
                print('vLLM adapter reply:', text)
                last_error = None
                break
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
        if last_error is not None:
            raise RuntimeError(last_error)
    except Exception as exc:
        Path('/kaggle/working/vllm_check.json').write_text(
            json.dumps({'status': 'fail', 'error': f'{type(exc).__name__}: {exc}'}, indent=2))
        traceback.print_exc()
''')

notebook = nbformat.v4.new_notebook(cells=[md, preflight, install, train, vllm_check])
notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbformat.validate(notebook)
nbformat.write(notebook, OUT / 'smoke.ipynb')

metadata = {
    'id': 'taruki/gemma4-training-spike',
    'title': 'Gemma4 Training Spike',
    'code_file': 'smoke.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': False,
    'keywords': ['gpu'],
    'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse', 'taruki/gemma4-training-wheels'],
    'kernel_sources': [],
    'competition_sources': ['gemma-4-developer-agent'],
    'model_sources': ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'],
    'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
    'machine_shape': 'NvidiaL4',
}
(OUT / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
manifest = {
    'kernel': metadata['id'],
    'purpose': 'Phase 3 feasibility gate: can a LoRA be trained on the w4a16-ct checkpoint and loaded by vLLM',
    'success_criteria': ['training steps complete and adapter saved', 'adapter < 400MB', 'vLLM loads the adapter (best-effort check)'],
    'training': 'development spike only; not the real training run',
}
(OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2))
print('prepared', OUT.resolve())