"""Generate the Gate 1 (training feasibility probe) Kaggle notebook and metadata.

Writes:
  artifacts/training_gate1/smoke.ipynb
  artifacts/training_gate1/kernel-metadata.json
  artifacts/training_gate1/experiment.json

The notebook is deterministic: rerunning this script byte-reproduces smoke.ipynb.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts" / "training_gate1"
KERNEL_ID = "taruki/gemma4-gate1-e4b"
KERNEL_TITLE = "Gemma4 Gate1 E4B"
RUNTIME_CAP_SECONDS = 4800
SOFT_DEADLINE_SECONDS = 3600

PROMPT = (
    'def add(a, b):\n'
    '    """Return the sum of a and b."""\n'
    '    return a + b\n'
    '\n'
    '\n'
    'def multiply(a, b):\n'
    '    """Return the product of a and b."""\n'
    '    return a * b\n'
    '\n'
    '\n'
    'def subtract(a, b):\n'
    '    """Return the difference of a and b."""\n'
    '    return a - b\n'
    '\n'
    '\n'
    'def divide(a, b):\n'
    '    """Return the quotient of a and b."""\n'
    '    if b == 0:\n'
    '        raise ValueError("division by zero")\n'
    '    return a / b\n'
    '\n'
    '\n'
    'def power(a, b):\n'
    '    """Return a raised to the power of b."""\n'
    '    return a ** b\n'
    '\n'
    '\n'
    'def modulo(a, b):\n'
    '    """Return the remainder of a divided by b."""\n'
    '    if b == 0:\n'
    '        raise ValueError("modulo by zero")\n'
    '    return a % b\n'
    '\n'
    '\n'
    'def negate(a):\n'
    '    """Return the negation of a."""\n'
    '    return -a\n'
    '\n'
    '\n'
    'def absolute(a):\n'
    '    """Return the absolute value of a."""\n'
    '    return abs(a)\n'
    '\n'
    '\n'
    'def clamp(value, low, high):\n'
    '    """Return value constrained to the inclusive range [low, high]."""\n'
    '    if low > high:\n'
    '        raise ValueError("low must not exceed high")\n'
    '    return max(low, min(high, value))\n'
)

CELL_0 = """# Gate 1: training feasibility probe on `gemma-4-e4b-it-qat-w4a16-ct`

Small-variant probe for the chain **PEFT -> compressed-tensors W4A16 base -> backprop -> save -> reload -> vLLM consumes the adapter**.

This is **not** SFT and **not** RL. It runs 8 AdamW steps on synthetic tokens, purely to produce a non-trivial adapter.
Passing Gate 1 only proves the mechanism on E4B; it does **not** prove 31B is feasible (Gate 2 does that).

Preregistered checks, tolerances and failure criteria: `artifacts/gate1_prereg.md`.
"""

CELL_1 = """## 1. Preflight: paths, GPU inventory, report scaffold

Every later cell is defensive: a failure records `ok=False` and continues, so the report is always written.
"""

CELL_1_CODE = r'''import json, os, platform, subprocess, time, traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

T0 = time.time()
SOFT_DEADLINE = T0 + 3600
WORK = Path("/kaggle/working")
REPORT_PATH = WORK / "gate1_report.json"

report = {
    "gate": "gate1",
    "kernel": "taruki/gemma4-gate1-e4b",
    "started_epoch": T0,
    "checks": {},
    "notes": [],
    "errors": [],
    "versions": {},
    "gpu": {},
    "timings": {},
}


def rec(cid, ok, value, detail=""):
    """Record one preregistered check. H-failures do not raise; they are reported."""
    report["checks"][cid] = {"ok": bool(ok), "value": value, "detail": str(detail)[:4000]}
    print(("PASS" if ok else "FAIL"), cid, "|", repr(value)[:400], "|", str(detail)[:300])
    return bool(ok)


def note(msg):
    report["notes"].append(str(msg)[:4000])
    print("NOTE:", str(msg)[:600])


def fail_soft(cid, exc, where):
    report["checks"][cid] = {
        "ok": False,
        "value": None,
        "detail": f"{type(exc).__name__}: {exc}",
    }
    report["errors"].append({"where": where, "error": f"{type(exc).__name__}: {exc}"})
    print("ERROR in", where, "->", type(exc).__name__, exc)
    traceback.print_exc()


def save_report():
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print("report written ->", REPORT_PATH)


def nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        return out.stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        return [f"nvidia-smi failed: {type(exc).__name__}: {exc}"]


MODEL_ROOT = Path("/kaggle/input/models/google/gemma-4/other")
E4B_SLUG = "gemma-4-e4b-it-qat-w4a16-ct"
cands = sorted(p for p in MODEL_ROOT.glob(E4B_SLUG + "/*") if p.is_dir())
MODEL = cands[-1] if cands else None
if MODEL is None:
    for alt in sorted(Path("/kaggle/input/models").rglob(E4B_SLUG + "/config.json")):
        MODEL = alt.parent
        break

paths = {
    "model": MODEL,
    "model_candidates": [str(p) for p in cands],
    "metric_wheelhouse": Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse"),
    "training_wheels": Path("/kaggle/input/datasets/taruki/gemma4-training-wheels"),
    "competition_wheels": Path("/kaggle/input/competitions/gemma-4-developer-agent/wheels"),
}
report["paths"] = {k: (str(v) if not isinstance(v, list) else v) if v is not None else None
                   for k, v in paths.items()}
print(json.dumps(report["paths"], indent=2))

# Fail fast and loudly: an unmounted model turns every later check into a misleading failure.
rec("PRE_MODEL", MODEL is not None and (MODEL / "config.json").is_file(),
    str(MODEL) if MODEL is not None else None,
    "model mount present; candidates=%s" % paths["model_candidates"])

try:
    import torch
    gpu_count = torch.cuda.device_count()
    report["gpu"] = {
        "count": gpu_count,
        "names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "total_memory_gib": [
            round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
            for i in range(gpu_count)
        ],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_smi": nvidia_smi(),
    }
    print(json.dumps(report["gpu"], indent=2))
except Exception as exc:  # noqa: BLE001
    fail_soft("PRE_GPU", exc, "preflight/gpu")

report["host"] = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "ram_gib": round(
        int(open("/proc/meminfo").read().split()[1]) / 1024**2, 2
    ) if os.path.exists("/proc/meminfo") else None,
}
print(json.dumps(report["host"], indent=2))

save_report()
print("preflight done in %.1fs" % (time.time() - T0))
'''

CELL_2 = """## 2. Install the official stack from the competition wheelhouse

Same install recipe as the earlier spike, so the versions match what the grader runs:
`transformers 5.13.1`, `vllm 0.19.1`, `compressed-tensors 0.15.0.1`, plus `peft 0.21.2` / `accelerate` from the training wheelhouse.
"""

CELL_2_CODE = r'''import glob, importlib, importlib.metadata, os, subprocess, sys
from pathlib import Path

WHEELHOUSE_DIR = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
TRAINING_DIR = Path("/kaggle/input/datasets/taruki/gemma4-training-wheels")
COMPETITION_WHEELS = Path("/kaggle/input/competitions/gemma-4-developer-agent/wheels")

try:
    for pth_pattern in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth",
                        "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
        for pth in glob.glob(pth_pattern):
            try:
                os.unlink(pth)
            except OSError:
                pass

    tmp_whl = Path("/tmp/wheelhouse")
    tmp_whl.mkdir(parents=True, exist_ok=True)
    for w in WHEELHOUSE_DIR.glob("*.whl"):
        if "cutlass" in w.name.lower():
            continue
        target_name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
        target = tmp_whl / target_name
        if not target.exists():
            os.symlink(w, target)
    wheels = sorted(str(w) for w in tmp_whl.glob("*.whl"))
    print(f"installing {len(wheels)} wheelhouse wheels ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--force-reinstall", *wheels], check=True)

    print("installing training wheels ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                    "--find-links", str(TRAINING_DIR),
                    "--find-links", str(COMPETITION_WHEELS),
                    "peft==0.21.2", "accelerate==1.15.0", "huggingface_hub==1.33.0",
                    "filelock==4.1.1", "fsspec==2026.9.0", "psutil==7.2.2",
                    "pyyaml==6.0.3", "tqdm==4.70.1"], check=True)

    importlib.invalidate_caches()
    for name in ("torch", "transformers", "vllm", "compressed-tensors", "peft",
                 "accelerate", "safetensors", "adk_submission", "swegemma"):
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][name] = "NOT INSTALLED"
    print(json.dumps(report["versions"], indent=2))
    report["timings"]["install_s"] = round(time.time() - T0, 1)
except Exception as exc:  # noqa: BLE001
    fail_soft("PRE_INSTALL", exc, "install")

save_report()
'''

CELL_3 = """## 3. Load the quantized base with `run_compressed=False`

A1 assertions on `architectures`; A2 confirms the weights really were decompressed to bf16;
A3 counts `Gemma4ClippableLinear` so we know the hazard is actually present in this variant.
"""

CELL_3_CODE = r'''import gc, json, re, time
from collections import Counter

import torch

T_LOAD = time.time()
model = None
tokenizer = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

    if MODEL is None:
        raise RuntimeError(
            "model not mounted at %s; check kernel model_sources (remote kernel-metadata "
            "showed model_sources=[] , meaning the push dropped the reference)" % MODEL_ROOT)

    print("model dir:", MODEL)
    report["model_files"] = sorted(p.name for p in MODEL.iterdir())[:40]

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    cfg = json.loads((MODEL / "config.json").read_text())
    report["architectures"] = cfg.get("architectures")
    report["text_layers"] = cfg["text_config"]["num_hidden_layers"]
    report["layer_types_count"] = dict(Counter(cfg["text_config"]["layer_types"]))
    report["attention_k_eq_v"] = cfg["text_config"].get("attention_k_eq_v")
    report["num_kv_shared_layers"] = cfg["text_config"].get("num_kv_shared_layers")

    quant_cfg = CompressedTensorsConfig(run_compressed=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_cfg,
    )
    model.eval()

    rec("A1",
        report["architectures"] == ["Gemma4ForConditionalGeneration"],
        report["architectures"],
        "expected Gemma4ForConditionalGeneration")

    # A2: decompression really happened -> bf16 floating weights + status decompressed
    # NOTE: str() on the enum yields "QuantizationStatus.DECOMPRESSED" (class name included),
    # so compare the trailing symbol rather than the bare word.
    status_counter = Counter()
    for _, m in model.named_modules():
        s = getattr(m, "quantization_status", None)
        if s is not None:
            status_counter[str(s).rsplit(".", 1)[-1].lower()] += 1
    report["quantization_status_counts"] = dict(status_counter)

    q_dtype = None
    for name, m in model.named_modules():
        if name.endswith("language_model.layers.0.self_attn.q_proj"):
            q_dtype = str(getattr(m, "weight", None).dtype) if getattr(m, "weight", None) is not None else None
            break
    hf_run_compressed = getattr(getattr(model, "hf_quantizer", None), "run_compressed", "missing")
    report["hf_quantizer_run_compressed"] = hf_run_compressed
    report["layer0_q_proj_dtype"] = q_dtype
    a2 = (
        q_dtype == "torch.bfloat16"
        and hf_run_compressed is False
        and all(v == "decompressed" for v in status_counter)
        and status_counter.get("decompressed", 0) > 0
    )
    rec("A2", a2, {"q_dtype": q_dtype, "run_compressed": hf_run_compressed,
                   "statuses": dict(status_counter)},
        "bf16 weights + all statuses decompressed")

    # A3: the hazard module must actually be present
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        clip_names = [n for n, m in model.named_modules() if isinstance(m, Gemma4ClippableLinear)]
    except Exception as exc:  # noqa: BLE001
        note("ClippableLinear import failed: %s" % exc)
        clip_names = []
    report["clippable_linear_count"] = len(clip_names)
    report["clippable_linear_sample"] = clip_names[:8]
    rec("A3", len(clip_names) > 0, len(clip_names),
        "hazard present (sample: %s)" % clip_names[:3])

    report["model_footprint_gib"] = round(model.get_memory_footprint() / 1024**3, 2)
    note("model footprint after decompress: %.2f GiB" % report["model_footprint_gib"])
except Exception as exc:  # noqa: BLE001
    fail_soft("A1", exc, "load")

report["timings"]["load_s"] = round(time.time() - T_LOAD, 1)
save_report()
'''

CELL_4 = """## 4. Attach LoRA with a regex anchored on `language_model`

A4/A5 assert the regex never escapes into the vision or audio tower (that is the whole point of the fix).
A6 checks the match count, A7 the injected layer count, A8 the adapter tensor count.
B1/B2 verify the base really is frozen and the trainable parameter count matches theory exactly.
"""

CELL_4_CODE = r'''import re
from collections import Counter

T_LORA = time.time()
TARGET_MODULE_REGEX = r".*\.language_model\.layers\.\d+\.self_attn\.(q|o)_proj$"
EXPECTED_TRAINABLE = 3325952

try:
    # The image ships torchao 0.10.0, below peft's TORCHAO_MINIMUM_VERSION (0.16.0).
    # peft's is_torchao_available() RAISES ImportError for that combination instead of
    # returning False, and LoraModel._create_new_module calls the torchao dispatcher
    # unconditionally for EVERY candidate target. Injection therefore dies on the first
    # matched module with "Found an incompatible version of torchao. Found version 0.10.0,
    # but only versions above 0.16.0 are supported" -- even though our targets are plain
    # nn.Linear, which peft supports natively and which have nothing to do with torchao.
    #
    # We use no torchao backend at all, so neutralise that one dispatcher for this process.
    # This is deliberately narrow: it patches a module-level name, not importlib and not the
    # installed environment, so the vLLM subprocess started later is unaffected.
    import peft.import_utils as _peft_import_utils
    import peft.tuners.lora.model as _peft_lora_model

    _peft_import_utils.is_torchao_available = lambda: False
    _peft_lora_model.is_torchao_available = lambda: False
    _peft_lora_model.dispatch_torchao = lambda *a, **k: None

    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import LoraLayer

    report["torchao_dispatcher_neutralised"] = True
    note("peft torchao dispatcher neutralised (image torchao 0.10.0 < peft minimum 0.16.0; "
         "no torchao backend is used by this probe)")

    matched = [n for n, _ in model.named_modules() if re.fullmatch(TARGET_MODULE_REGEX, n)]
    report["n_matched_modules"] = len(matched)
    report["matched_sample"] = matched[:6]
    report["n_text_layers"] = report.get("text_layers")

    violations = [n for n in matched if "language_model" not in n]
    tower_hits = [n for n in matched if ("vision_tower" in n or "audio_tower" in n)]
    expected_n = 2 * int(report.get("text_layers") or 0)

    rec("A4", len(violations) == 0, violations[:10], "modules without language_model in path")
    rec("A5", len(tower_hits) == 0, tower_hits[:10], "tower modules matched")
    rec("A6", len(matched) == expected_n, len(matched), "expected %d (q+o x layers)" % expected_n)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=False,
        use_rslora=False,
        modules_to_save=None,
        target_modules=TARGET_MODULE_REGEX,
    )
    model = get_peft_model(model, lora_cfg)

    lora_layers = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    report["n_lora_layers"] = len(lora_layers)
    wrapped_tower = [n for n in lora_layers if ("vision_tower" in n or "audio_tower" in n)]
    rec("A7", len(lora_layers) == expected_n and len(wrapped_tower) == 0,
        {"n_lora_layers": len(lora_layers), "wrapped_tower": wrapped_tower[:5]},
        "expected %d injected layers, no tower wrapped" % expected_n)

    lora_keys = sorted(k for k in model.state_dict() if "lora_" in k)
    report["n_adapter_tensors"] = len(lora_keys)
    report["adapter_key_sample"] = lora_keys[:4]
    suffixes = Counter(k.split(".lora_")[-1].split(".")[0] for k in lora_keys)
    report["adapter_tensor_kinds"] = dict(suffixes)
    rec("A8", len(lora_keys) == 2 * expected_n, len(lora_keys),
        "expected %d tensors (A+B per module)" % (2 * expected_n))

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    non_lora = [n for n, _ in trainable if "lora_" not in n]
    n_trainable = sum(p.numel() for _, p in trainable)
    report["n_trainable_params"] = n_trainable
    report["non_lora_trainable"] = non_lora[:20]
    rec("B1", len(non_lora) == 0, len(non_lora), "trainable params outside lora_")
    rec("B2", n_trainable == EXPECTED_TRAINABLE, n_trainable,
        "expected %d" % EXPECTED_TRAINABLE)

    report["target_module_regex"] = TARGET_MODULE_REGEX
    report["lora_config"] = {
        "r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none",
        "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False,
        "modules_to_save": None, "target_modules": TARGET_MODULE_REGEX,
    }
except Exception as exc:  # noqa: BLE001
    fail_soft("A7", exc, "lora_inject")

report["timings"]["lora_s"] = round(time.time() - T_LORA, 1)
save_report()
'''

CELL_5 = """## 5. Eight optimiser steps

Loss / gradient finiteness, `lora_B` gradient non-zero, and weight change are all recorded per step.
Peak memory per device and the parameter->device distribution are recorded for the report.
"""

CELL_5_CODE = r'''import math
import torch
import torch.nn.functional as F

T_TRAIN = time.time()
N_STEPS = 8
SEQ_LEN = 512
LR = 3e-3
SEED = 1234

try:
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    trainable = [p for _, p in lora_params]
    before = {n: p.detach().float().clone() for n, p in lora_params}

    report["device_map"] = {str(k): str(v) for k, v in
                            getattr(model, "hf_device_map", {}).items()}

    per_dev = {}
    for n, p in model.named_parameters():
        dev = str(p.device)
        slot = per_dev.setdefault(dev, {"params": 0, "gib": 0.0, "dtypes": set()})
        slot["params"] += p.numel()
        slot["gib"] += p.numel() * p.element_size() / 1024**3
        slot["dtypes"].add(str(p.dtype))
    report["param_device_distribution"] = {
        k: {"params": v["params"], "gib": round(v["gib"], 3), "dtypes": sorted(v["dtypes"])}
        for k, v in per_dev.items()
    }

    model.train()
    optim = torch.optim.AdamW(trainable, lr=LR)
    vocab = None
    for getter in (
        lambda: int(tokenizer.vocab_size),
        lambda: int(model.config.text_config.vocab_size),
        lambda: int(model.config.vocab_size),
    ):
        try:
            vocab = getter()
            break
        except Exception:  # noqa: BLE001
            continue
    if not vocab:
        raise RuntimeError("could not determine vocab size")
    report["vocab_size"] = vocab
    gen = torch.Generator().manual_seed(SEED)
    steps = []
    loss_ok = True
    grad_finite_ok = True
    b_grad_ok = True

    # `device_map="auto"` shards this model across multiple GPUs, so the embedding device
    # is NOT necessarily cuda:0. Feeding `.to("cuda")` (which means cuda:0) therefore raised
    # "Expected all tensors to be on the same device, but found at least two devices,
    # cuda:1 and cuda:2!" Place the inputs on whatever device actually holds the input
    # embedding, and assert they really agree before using them.
    def _input_device(model):
        for attr in ("get_input_embeddings",):
            getter = getattr(model, attr, None)
            if callable(getter):
                try:
                    emb = getter()
                    if emb is not None and hasattr(emb, "weight"):
                        return emb.weight.device
                except Exception:  # noqa: BLE001
                    pass
        try:
            for name, p in model.named_parameters():
                if "embed_tokens" in name:
                    return p.device
        except Exception:  # noqa: BLE001
            pass
        return torch.device("cuda:0")

    input_device = _input_device(model)
    report["train_input_device"] = str(input_device)
    note("training inputs placed on %s (device_map shards across %d GPUs)"
         % (input_device, torch.cuda.device_count()))

    # Sanity check: confirm the embedding and the first decoder layer agree, so a future
    # device_map change cannot silently reintroduce the mismatch.
    _embed_dev = None
    _layer_dev = None
    for _n, _p in model.named_parameters():
        if _embed_dev is None and "embed_tokens" in _n:
            _embed_dev = _p.device
        if _layer_dev is None and "layers.0." in _n:
            _layer_dev = _p.device
        if _embed_dev is not None and _layer_dev is not None:
            break
    report["embed_device"] = str(_embed_dev)
    report["first_layer_device"] = str(_layer_dev)

    for step in range(N_STEPS):
        ids = torch.randint(0, vocab, (1, SEQ_LEN), generator=gen).to(input_device)
        out = model(input_ids=ids)
        logits = out.logits
        # Under device_map="auto" the LM head can live on a different rank than the input
        # embedding, so the logits and the targets must be joined on a common device
        # explicitly; cross_entropy raises "Expected all tensors to be on the same device"
        # otherwise (that is exactly how v3 died).
        _targets = ids[:, 1:].reshape(-1).to(logits.device)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
            _targets,
        )
        loss.backward()

        grads = {n: p.grad for n, p in lora_params}
        finite = all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
        # Under device_map="auto" the 84 LoRA tensors are spread over several GPUs, so a
        # cross-tensor reduction must not be expressed as sum(t1, t2, ...): that makes
        # torch add scalars that live on different devices and dies with "Expected all
        # tensors to be on the same device". Reduce each tensor to a Python float first.
        b_grad_sum = float(sum(
            g.detach().float().abs().sum().item() for n, g in grads.items()
            if "lora_B" in n and g is not None
        ))
        a_grad_sum = float(sum(
            g.detach().float().abs().sum().item() for n, g in grads.items()
            if "lora_A" in n and g is not None
        ))
        loss_val = float(loss.detach())
        steps.append({"step": step, "loss": round(loss_val, 6),
                      "grad_finite": finite,
                      "lora_B_grad_abs_sum": round(b_grad_sum, 4),
                      "lora_A_grad_abs_sum": round(a_grad_sum, 4)})
        if not math.isfinite(loss_val):
            loss_ok = False
        if not finite:
            grad_finite_ok = False
        if not (b_grad_sum > 0.0):
            b_grad_ok = False
        print("step", step, "loss", round(loss_val, 5),
              "B_grad", round(b_grad_sum, 2), "A_grad", round(a_grad_sum, 2), flush=True)

        optim.step()
        optim.zero_grad(set_to_none=True)
        del out, logits, loss

    report["training_steps"] = steps
    rec("B3", loss_ok, [s["loss"] for s in steps], "all losses finite")
    rec("B4", grad_finite_ok, len(steps), "all LoRA grads finite")
    rec("B5", b_grad_ok, [s["lora_B_grad_abs_sum"] for s in steps],
        "lora_B grad non-zero at every step")

    after = {n: p.detach().float().clone() for n, p in lora_params}
    per_tensor = {}
    for n in before:
        d = (after[n] - before[n]).abs().max().item()
        per_tensor[n] = d
    max_delta = max(per_tensor.values()) if per_tensor else 0.0
    n_changed = sum(1 for v in per_tensor.values() if v > 0)
    # Same cross-device hazard as the per-step gradient reduction above.
    b_after_abs = float(sum(
        p.detach().float().abs().sum().item() for n, p in lora_params if "lora_B" in n
    ))
    report["max_weight_delta"] = max_delta
    report["n_tensors_changed"] = n_changed
    report["lora_B_abs_sum_after"] = b_after_abs
    report["delta_sample"] = dict(sorted(per_tensor.items(), key=lambda kv: -kv[1])[:5])
    rec("B6", max_delta > 0.0 and b_after_abs > 0.0,
        {"max_weight_delta": max_delta, "n_changed": n_changed,
         "lora_B_abs_sum_after": round(b_after_abs, 4)},
        "%d/%d tensors changed" % (n_changed, len(per_tensor)))

    peaks = {}
    for i in range(torch.cuda.device_count()):
        peaks[str(i)] = {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(i) / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(i) / 1024**3, 3),
            "total_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        }
    report["peak_memory_during_training"] = peaks
    over = [k for k, v in peaks.items() if v["peak_reserved_gib"] > v["total_gib"]]
    rec("D1", True, peaks, "per-device peak memory (training)")
    rec("D3", len(over) == 0, over, "no device exceeded capacity")
    model.eval()
except Exception as exc:  # noqa: BLE001
    fail_soft("B3", exc, "train")

report["timings"]["train_s"] = round(time.time() - T_TRAIN, 1)
save_report()
'''

CELL_6 = """## 6. Save, reload, and verify the adapter is actually consumed

C1 compares the safetensors on disk bit-for-bit against the in-memory tensors.
C2 reloads through PEFT into a second adapter slot and compares again.
E5 (checked here, while the model is alive) confirms every saved key maps onto a module that
PEFT actually wrapped with LoRA.

PEFT stores the same tensor under three different names depending on where it lives:
disk `....q_proj.lora_A.weight`, in-memory default slot `....q_proj.lora_A.default.weight`,
in-memory reload slot `....q_proj.lora_A.reloaded.weight`. All three are normalised to one
canonical key before comparison, otherwise a correct reload looks like a mismatch.
"""

CELL_6_CODE = r'''import json, re, shutil
from pathlib import Path

T_SAVE = time.time()
ADAPTER_TRAINED = WORK / "gate1_adapter"
LAYOUT = WORK / "gate1_layout"
CANON_RE = re.compile(r"(layers\.\d+\.self_attn\.(?:q|o)_proj)\.lora_([AB])\.")


def canon(name):
    """Canonical adapter key, independent of the PEFT adapter-name segment."""
    m = CANON_RE.search(name)
    return f"{m.group(1)}.lora_{m.group(2)}.weight" if m else None


try:
    from safetensors.torch import load_file, save_file

    # Guard: if LoRA injection failed, `model` is still the plain transformers model and
    # model.save_pretrained() would try to write the whole 15 GiB base (and blow up inside
    # transformers' revert_weight_conversion). Fail loudly instead of producing a bogus
    # "adapter" directory.
    from peft import PeftModel as _PeftModel
    if not isinstance(model, _PeftModel):
        raise RuntimeError(
            "model is %s, not a PeftModel: LoRA injection did not run, so there is no "
            "adapter to save (see the A7/B* results above)" % type(model).__name__)

    if ADAPTER_TRAINED.exists():
        shutil.rmtree(ADAPTER_TRAINED)
    model.save_pretrained(str(ADAPTER_TRAINED))
    adapter_files = sorted(p.name for p in ADAPTER_TRAINED.rglob("*") if p.is_file())
    adapter_bytes = sum(p.stat().st_size for p in ADAPTER_TRAINED.rglob("*") if p.is_file())
    report["adapter_files"] = adapter_files
    report["adapter_bytes"] = adapter_bytes
    rec("C3",
        "adapter_config.json" in adapter_files and "adapter_model.safetensors" in adapter_files,
        {"files": adapter_files, "bytes": adapter_bytes},
        "adapter artifacts present")

    on_disk = load_file(str(ADAPTER_TRAINED / "adapter_model.safetensors"))
    report["saved_tensor_count"] = len(on_disk)
    disk = {canon(k): v.float() for k, v in on_disk.items()}
    unparsed = sorted(k for k in on_disk if canon(k) is None)
    report["c1_unparsed_keys"] = unparsed[:10]
    report["c1_parsed_keys"] = sum(1 for k in on_disk if canon(k) is not None)

    mem = {}
    for n, p in model.named_parameters():
        if "lora_" in n:
            c = canon(n)
            if c is not None:
                mem[c] = p.detach().float().cpu()

    common = sorted(set(mem) & set(disk))
    missing_mem = sorted(set(disk) - set(mem))
    c1_max = max((disk[k] - mem[k]).abs().max().item() for k in common) if common else None
    report["c1_common_keys"] = len(common)
    report["c1_missing_in_memory"] = missing_mem[:10]
    rec("C1",
        bool(common) and c1_max == 0.0 and not missing_mem and not unparsed,
        {"compared": len(common), "max_abs_diff": c1_max,
         "missing": missing_mem[:5], "unparsed": unparsed[:5]},
        "disk == memory, bit-exact")

    # E5: every saved key must map onto a module that PEFT actually wrapped
    from peft.tuners.lora.layer import LoraLayer
    lora_module_names = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    distinct_modules = sorted({c.rsplit(".lora_", 1)[0] for c in disk if c})
    unknown = [p for p in distinct_modules
               if not any(n.endswith(p) for n in lora_module_names)]
    suffixes_used = sorted({p.split(".")[-1] for p in distinct_modules})
    a_keys = sum(1 for c in disk if c and ".lora_A." in c)
    b_keys = sum(1 for c in disk if c and ".lora_B." in c)
    rec("E5",
        len(distinct_modules) == report.get("n_matched_modules")
        and not unknown
        and set(suffixes_used) <= {"q_proj", "o_proj"}
        and a_keys == b_keys == len(distinct_modules),
        {"distinct_modules": len(distinct_modules),
         "expected": report.get("n_matched_modules"),
         "wrapped_lora_layers": len(lora_module_names),
         "unknown": unknown[:5],
         "suffixes": suffixes_used,
         "lora_A_keys": a_keys,
         "lora_B_keys": b_keys},
        "adapter keys map onto injected modules only")

    # C2: reload through PEFT into a second adapter slot
    try:
        model.load_adapter(str(ADAPTER_TRAINED), adapter_name="reloaded")
        reloaded = {}
        for n, p in model.named_parameters():
            if "lora_" in n and ".reloaded." in n:
                c = canon(n)
                if c is not None:
                    reloaded[c] = p.detach().float().cpu()
        pairs = [(reloaded[k], disk[k]) for k in sorted(set(reloaded) & set(disk))]
        c2_max = max((a - b).abs().max().item() for a, b in pairs) if pairs else None
        report["c2_compared"] = len(pairs)
        report["c2_reloaded_keys"] = len(reloaded)
        rec("C2",
            bool(pairs) and c2_max == 0.0 and len(pairs) == len(disk),
            {"compared": len(pairs), "disk_keys": len(disk), "max_abs_diff": c2_max},
            "PEFT reload == on-disk bytes")
    except Exception as exc:  # noqa: BLE001
        fail_soft("C2", exc, "peft_reload")

    # Build the serving layout: one real adapter + one zero-lora_B control
    if LAYOUT.exists():
        shutil.rmtree(LAYOUT)
    trained_dir = LAYOUT / "adapters" / "trained_gate1"
    zero_dir = LAYOUT / "adapters" / "zero_b_control"
    trained_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ADAPTER_TRAINED, trained_dir)
    shutil.copytree(ADAPTER_TRAINED, zero_dir)

    zd = load_file(str(zero_dir / "adapter_model.safetensors"))
    zeroed = {}
    n_zeroed = 0
    for k, v in zd.items():
        if "lora_B" in k:
            zeroed[k] = torch.zeros_like(v)
            n_zeroed += 1
        else:
            zeroed[k] = v
    save_file(zeroed, str(zero_dir / "adapter_model.safetensors"))
    report["zero_b_tensors_zeroed"] = n_zeroed
    b_abs_zero = float(sum(v.float().abs().sum() for k, v in zeroed.items() if "lora_B" in k))
    b_abs_zero_trained = float(sum(v.float().abs().sum() for k, v in on_disk.items()
                                   if "lora_B" in k))
    report["zero_b_abs_sum"] = b_abs_zero
    report["trained_b_abs_sum"] = b_abs_zero_trained
    rec("E4_prep", n_zeroed > 0 and b_abs_zero == 0.0 and b_abs_zero_trained > 0.0,
        {"n_zeroed": n_zeroed, "control_lora_B_abs_sum": b_abs_zero,
         "trained_lora_B_abs_sum": round(b_abs_zero_trained, 4)},
        "control adapter lora_B is exactly zero while trained one is not")
    report["layout_path"] = str(LAYOUT)
except Exception as exc:  # noqa: BLE001
    fail_soft("C1", exc, "save_reload")

report["timings"]["save_s"] = round(time.time() - T_SAVE, 1)
save_report()
'''

CELL_7 = """## 7. Start the official inference stack with both adapters registered

Frees the GPUs held by the training model first, then launches `VllmServer` through `adk_submission`
(the same wrapper the graded submission uses) and records the exact CLI it built.
"""

CELL_7_CODE = r'''import gc, json, shutil, subprocess, time
from pathlib import Path

T_VLLM = time.time()
server = None
try:
    report["elapsed_before_vllm_s"] = round(time.time() - T0, 1)
    if time.time() > SOFT_DEADLINE:
        raise RuntimeError(
            "soft deadline exceeded before vLLM start (elapsed %.0fs > %ds); "
            "aborting serve stage to stay inside the kernel cap"
            % (time.time() - T0, SOFT_DEADLINE - T0))

    # Release every reference that keeps the training graph alive on the GPUs.
    # `del model` alone is NOT enough: the optimiser holds AdamW exp_avg/exp_avg_sq
    # moments for every LoRA parameter (2x the adapter bytes, on device), and the
    # bookkeeping dicts in the previous cells still point at live CUDA tensors.
    # Left in place they caused: "Free memory on device cuda:1 (17.84/22.03 GiB) on
    # startup is less than desired GPU memory utilization (0.85, 18.73 GiB)".
    for _name in ("optim", "model", "lora_params", "trainable", "before", "after",
                  "grads", "per_tensor", "reloaded", "mem", "on_disk", "zeroed",
                  "steps", "out", "logits", "loss"):
        try:
            del globals()[_name]
        except KeyError:
            pass
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)

    free_gib = {}
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gib[str(i)] = {"free": round(free_b / 1024**3, 2),
                            "total": round(total_b / 1024**3, 2)}
    report["gpu_free_before_vllm"] = free_gib
    print("free memory before vLLM:", json.dumps(free_gib), flush=True)

    # Pick a utilisation that leaves headroom on the *least* free device.
    # vLLM requires free_memory >= gpu_memory_utilization * total_memory on every worker,
    # so take the minimum feasible fraction across devices and keep headroom for the
    # CUDA-graph capture phase, which allocates well beyond the static weights.
    util = 0.85
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        feasible = (free_b - 2.5 * 1024**3) / float(total_b)
        util = min(util, feasible)
    util = max(0.10, round(util, 2))
    report["vllm_gpu_memory_utilization"] = util

    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    adapters = discover_adapters(str(LAYOUT), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    report["discovered_adapters"] = {
        k: {"rank": getattr(v, "rank", None), "path": str(getattr(v, "path", ""))}
        for k, v in adapters.adapters.items()
    }
    print("adapters:", json.dumps(report["discovered_adapters"], indent=2))

    vcfg = VllmConfig(
        model=str(MODEL),
        port=8000,
        host="127.0.0.1",
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=util,
        enable_auto_tool_choice=True,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=128,
        tensor_parallel_size=4 if report["gpu"]["count"] >= 4 else report["gpu"]["count"],
        startup_timeout=1800,
        # CUDA-graph capture allocates far beyond the static weights; on a 22 GiB L4 the
        # default capture blew up at 53% ("Tried to allocate 20.00 MiB ... 3.94 MiB free")
        # even with utilisation 0.68. `ENFORCE_EAGER` disables graph capture entirely,
        # which is irrelevant to what this gate measures (adapter plumbing + logprobs).
        extra_args=["--enforce-eager"],
    )
    server = VllmServer(vcfg, adapter_manifest=adapters)
    cmd = server.build_cmd()
    report["vllm_cmd"] = cmd
    print("vLLM cmd:", " ".join(cmd))

    server.start()
    report["server_started"] = True
    report["base_url"] = server.base_url
    report["log_path"] = server.log_path
    report["nvidia_smi_after_start"] = nvidia_smi()
    print("server up in %.1fs -> %s" % (time.time() - T_VLLM, server.base_url))
    print("\n".join(report["nvidia_smi_after_start"]))
except Exception as exc:  # noqa: BLE001
    fail_soft("E7", exc, "vllm_start")

report["timings"]["vllm_start_s"] = round(time.time() - T_VLLM, 1)
save_report()
'''

CELL_8 = """## 8. Does the inference stack actually consume the adapter?

The decisive test is the **logprob** comparison, not generated text:

- **E2** noise floor — the same model queried twice must return identical logprobs.
- **E3** trained adapter vs base — must move logprobs by `>= 1e-3` and by more than `10x` the noise floor.
- **E4** zero-`lora_B` control vs base — must NOT move logprobs beyond the noise floor.

E3 without E4 would be worthless: any change could come from something other than the adapter weights.
E4 without E3 would be equally worthless: it cannot distinguish "adapter is correctly inert" from
"every adapter is silently ignored".

`prompt_logprobs=0` scores every prompt token with no sampling involved, which is far more sensitive
than comparing a few sampled tokens.
"""

CELL_8_CODE = r'''import json, time
import requests

T_E = time.time()
# adk_submission's server.base_url already ends in "/v1" (adk_submission/server.py:204
# returns f"http://{host}:{port}/v1"), so paths below are relative to that.
BASE = None
try:
    BASE = server.base_url if server is not None else None
except Exception:  # noqa: BLE001
    pass
report["api_base"] = BASE

PROMPT_TEXT = __PROMPT_JSON__


def api(path, payload, timeout=180):
    """BASE already ends with /v1, so callers pass only the suffix (e.g. '/completions')."""
    url = BASE.rstrip("/") + path
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def prompt_logprobs(model_name):
    body = {
        "model": model_name,
        "prompt": PROMPT_TEXT,
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 0,
        "prompt_logprobs": 0,
    }
    data = api("/completions", body)
    lp = data["choices"][0].get("prompt_logprobs")
    out = []
    if lp is None:
        return None
    for element in lp:
        if isinstance(element, dict) and element:
            first = next(iter(element.values()))
            if isinstance(first, dict):
                out.append(float(first.get("logprob")))
            else:
                out.append(float(first))
        else:
            out.append(None)
    return out


def compare(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    return max(abs(x - y) for x, y in pairs), len(pairs)


if BASE is None:
    rec("E1", False, None, "server never started")
else:
    try:
        listing = requests.get(BASE.rstrip("/") + "/models", timeout=60).json()
        ids = [m["id"] for m in listing.get("data", [])]
        report["served_models"] = ids
        wanted = {"trained_gate1", "zero_b_control", "adapter:trained_gate1", "adapter:zero_b_control"}
        hit = sorted(wanted & set(ids))
        base_present = any(not i.endswith(("_gate1", "_control")) and "adapter:" not in i for i in ids)
        rec("E1", len(hit) >= 2 and base_present, {"ids": ids, "adapter_aliases_found": hit},
            "base + adapter aliases listed")
    except Exception as exc:  # noqa: BLE001
        fail_soft("E1", exc, "models_list")

    try:
        base_name = None
        for candidate in report.get("served_models", []):
            if candidate.endswith(("_gate1", "_control")) or "adapter:" in candidate:
                continue
            base_name = candidate
            break
        report["base_model_alias"] = base_name

        base_a = prompt_logprobs(base_name)
        base_b = prompt_logprobs(base_name)
        noise, n_noise = compare(base_a, base_b)
        report["E2_positions"] = n_noise
        report["E2_noise_floor"] = noise
        rec("E2", noise is not None, {"noise": noise, "positions": n_noise},
            "same model twice")

        trained = prompt_logprobs("trained_gate1")
        zero_b = prompt_logprobs("zero_b_control")
        d_trained, n_t = compare(base_a, trained)
        d_zero, n_z = compare(base_a, zero_b)
        report["E3_delta"] = d_trained
        report["E3_positions"] = n_t
        report["E4_delta"] = d_zero
        report["E4_positions"] = n_z

        eff_noise = noise if (noise is not None and noise > 0.0) else 0.0
        e3_ok = d_trained is not None and d_trained >= 1e-3 and (
            eff_noise == 0.0 or d_trained > 10 * eff_noise)
        rec("E3", e3_ok, {"delta": d_trained, "positions": n_t, "noise": noise},
            "trained adapter must move logprobs")

        e4_ok = d_zero is not None and (d_zero <= eff_noise if eff_noise > 0 else d_zero == 0.0)
        rec("E4", e4_ok, {"delta": d_zero, "positions": n_z, "noise": noise},
            "zero-B control must stay within noise")

        gen_body = {
            "model": base_name,
            "prompt": PROMPT_TEXT,
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 0,
            "logprobs": 5,
        }
        gen = api("/completions", gen_body)
        choice = gen["choices"][0]
        text = choice.get("text") or ""
        gl = choice.get("logprobs") or {}
        report["E7_text"] = text[:200]
        report["E7_token_count"] = len(gl.get("tokens") or [])
        rec("E7", len(gl.get("tokens") or []) > 0, {"tokens": len(gl.get("tokens") or [])},
            "generation produced tokens")

        trained_gen = api("/completions", dict(gen_body, model="trained_gate1"))
        gl_t = trained_gen["choices"][0].get("logprobs") or {}
        tok_b = gl.get("tokens") or []
        tok_t = gl_t.get("tokens") or []
        lp_b = gl.get("token_logprobs") or []
        lp_t = gl_t.get("token_logprobs") or []
        n = min(len(lp_b), len(lp_t))
        gen_delta = max((abs(lp_b[i] - lp_t[i]) for i in range(n)
                         if lp_b[i] is not None and lp_t[i] is not None), default=None)
        report["E3_gen_token_delta"] = gen_delta
        report["E3_gen_tokens_identical"] = tok_b[:16] == tok_t[:16]
        note("generated-token logprob delta (base vs trained): %r; same tokens: %r"
             % (gen_delta, report["E3_gen_tokens_identical"]))
    except Exception as exc:  # noqa: BLE001
        fail_soft("E3", exc, "logprob_compare")

    # E8: tool calling through the official auto tool choice path
    TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }]
    attempts = [
        ("What is the weather in Paris right now? Use the get_weather tool.",
         {"enable_thinking": False}),
        ("You must call the function get_weather with city='Paris'. Do not answer in prose.",
         {"enable_thinking": False}),
        ("What is the weather in Paris right now? Use the get_weather tool.",
         {"enable_thinking": True}),
    ]
    e8_ok = False
    e8_log = []
    for idx, (user_msg, tkw) in enumerate(attempts):
        try:
            body = {
                "model": report.get("base_model_alias"),
                "messages": [{"role": "user", "content": user_msg}],
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": 256,
                "chat_template_kwargs": tkw,
            }
            data = api("/chat/completions", body)
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            names = [c.get("function", {}).get("name") for c in calls]
            e8_log.append({"attempt": idx, "tool_calls": names,
                           "content": (msg.get("content") or "")[:300]})
            if calls and any(nm == "get_weather" for nm in names):
                e8_ok = True
                break
        except Exception as exc:  # noqa: BLE001
            e8_log.append({"attempt": idx, "error": f"{type(exc).__name__}: {exc}"})
    report["E8_attempts"] = e8_log
    rec("E8", e8_ok, e8_log[-1] if e8_log else None, "tool call parsed at least once")

    # E6: no silent-drop warnings / no LoRA-unsupported errors in the server log
    try:
        log_text = Path(server.log_path).read_text(errors="replace")
        Path(WORK / "gate1_vllm_server.log").write_text(log_text)
        forbidden = [
            "is not in the model's supported LoRA target modules",
            "These parameters will be ignored",
            "does not support LoRA yet",
        ]
        found = [f for f in forbidden if f in log_text]
        report["E6_found"] = found
        report["vllm_log_bytes"] = len(log_text)
        lora_mentions = log_text.lower().count("lora")
        report["E6_lora_mentions"] = lora_mentions
        rec("E6", len(found) == 0, {"forbidden_found": found, "lora_mentions": lora_mentions},
            "no silent-drop / unsupported warnings")
    except Exception as exc:  # noqa: BLE001
        fail_soft("E6", exc, "vllm_log_scan")

report["timings"]["e_checks_s"] = round(time.time() - T_E, 1)
save_report()
'''

CELL_9 = """## 9. Finalise

Stop the server, capture the closing GPU state, and write the machine-readable verdict.
"""

CELL_9_CODE = r'''import json, time

try:
    report["nvidia_smi_before_stop"] = nvidia_smi()
    if server is not None:
        try:
            report["nvidia_smi_during_serve"] = nvidia_smi()
            server.stop()
            report["server_stopped"] = True
        except Exception as exc:  # noqa: BLE001
            note("server.stop failed: %s" % exc)
except Exception as exc:  # noqa: BLE001
    fail_soft("POST_STOP", exc, "finalise")

HARD_GROUPS = {
    "A": ["A1", "A2", "A4", "A5", "A6", "A7", "A8"],
    "B": ["B1", "B2", "B3", "B4", "B5", "B6"],
    "C": ["C1", "C2"],
    "D": ["D3"],
    "E": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
}
summary = {}
for group, ids in HARD_GROUPS.items():
    missing = [i for i in ids if i not in report["checks"]]
    failed = [i for i in ids if i in report["checks"] and not report["checks"][i]["ok"]]
    summary[group] = {"missing": missing, "failed": failed,
                      "ok": not missing and not failed}
report["verdict"] = {
    "groups": summary,
    "pass": all(v["ok"] for v in summary.values()),
}
report["finished_epoch"] = time.time()
report["wall_seconds"] = round(report["finished_epoch"] - report["started_epoch"], 1)
save_report()

print()
print("=" * 72)
print("VERDICT:", "PASS" if report["verdict"]["pass"] else "FAIL")
for g, v in summary.items():
    print("  %s: %s (missing=%s failed=%s)" % (g, "ok" if v["ok"] else "FAIL", v["missing"], v["failed"]))
print("wall: %.1f s" % report["wall_seconds"])
print("=" * 72)
'''


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build_notebook() -> dict:
    e_code = CELL_8_CODE.replace("__PROMPT_JSON__", json.dumps(PROMPT))
    cells = [
        md_cell(CELL_0),
        md_cell(CELL_1), code_cell(CELL_1_CODE),
        md_cell(CELL_2), code_cell(CELL_2_CODE),
        md_cell(CELL_3), code_cell(CELL_3_CODE),
        md_cell(CELL_4), code_cell(CELL_4_CODE),
        md_cell(CELL_5), code_cell(CELL_5_CODE),
        md_cell(CELL_6), code_cell(CELL_6_CODE),
        md_cell(CELL_7), code_cell(CELL_7_CODE),
        md_cell(CELL_8), code_cell(e_code),
        md_cell(CELL_9), code_cell(CELL_9_CODE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nb_path = DEST / "smoke.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    nb_sha = hashlib.sha256(nb_path.read_bytes()).hexdigest()

    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "smoke.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "metric/gemma-4-developer-agent-wheelhouse",
            "taruki/gemma4-training-wheels",
        ],
        "kernel_sources": [],
        "competition_sources": ["gemma-4-developer-agent"],
        "model_sources": ["google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4",
    }
    (DEST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    experiment = {
        "suite": "gate1",
        "kernel": KERNEL_ID,
        "purpose": "training feasibility probe (not SFT, not RL)",
        "base_model": "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2",
        "base_model_path_in_kernel": (
            "/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"),
        "runtime_cap_seconds": RUNTIME_CAP_SECONDS,
        "soft_deadline_seconds": SOFT_DEADLINE_SECONDS,
        "target_module_regex": r".*\.language_model\.layers\.\d+\.self_attn\.(q|o)_proj$",
        "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none",
                 "use_dora": False, "use_rslora": False, "modules_to_save": None},
        "training": {"steps": 8, "seq_len": 512, "batch": 1, "lr": 3e-3,
                     "optimizer": "AdamW", "seed": 1234,
                     "data": "synthetic random token ids (no dataset)"},
        "expected": {
            "n_matched_modules": 84,
            "n_trainable_params": 3325952,
            "n_adapter_tensors": 168,
        },
        "notebook_sha256": nb_sha,
        "prereg": "artifacts/gate1_prereg.md",
        "not_in_scope": ["full SFT", "RL", "31B training", "submission"],
    }
    (DEST / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("wrote", nb_path, "sha256", nb_sha)
    print("cells:", len(nb["cells"]))
    print("wrote", DEST / "kernel-metadata.json")
    print("wrote", DEST / "experiment.json")


if __name__ == "__main__":
    main()
