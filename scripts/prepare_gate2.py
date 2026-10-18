"""Generate the Gate 2 (31B) training-feasibility probe notebook.

Gate 2 answers one question only: does the *scoring* 31B checkpoint
(`gemma-4-31b-it-qat-w4a16-ct`, the only name in `ALLOWED_MODEL_NAMES`) accept a
LoRA injection, a real parameter update, and adapter consumption by the official
inference stack? It is not SFT and not RL.

Everything here reuses the fixes proven in Gate 1 (E4B), listed in
artifacts/gate1_result.md. The 31B-specific deltas are:

  * 60 text layers, 50 sliding + 10 full (full indices 5,11,...,59).
  * `attention_k_eq_v = True` and `v_proj is None` on every full layer, so HF's
    Gemma4TextAttention reuses the k_proj output as V. vLLM instead keeps a
    packed qkv_proj and leaves the missing v slice at zeros. Targeting k/v on
    those layers would therefore diverge silently between the two stacks, so the
    target set stays q_proj/o_proj only.
  * `num_kv_shared_layers = 0`, so every one of the 60 layers has q_proj and
    o_proj: 120 modules, 240 adapter tensors, 14,336,000 trainable parameters.
  * The base is 32.68B parameters / 60.9 GiB in bf16, i.e. ~15.2 GiB per L4.
    That leaves ~6 GiB of headroom per card, so training enables gradient
    checkpointing and reduces the cross-entropy over the 262,144-token
    vocabulary in sequence chunks.
  * Training and serving are separate stages, and one step at the planned
    training length (2048) is measured explicitly.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts" / "training_gate2"

KERNEL_ID = "taruki/gemma4-gate2-31b"
KERNEL_TITLE = "Gemma4 Gate2 31B"
RUNTIME_CAP_SECONDS = 7200
# Raised from 3600 s when the load path moved to the host: the CPU decompression
# pass is slower than the GPU one it replaces, and that cost lands before the
# serving stage. This constant is the single source of truth -- the notebook cell
# interpolates it rather than repeating the literal.
SOFT_DEADLINE_SECONDS = 4800
# Seconds of the kernel cap held back so the E group and the report write always
# have room, whatever the serving ladder does. 1500 s = 25 min.
SERVER_RESERVE_SECONDS = 1500
# Per-card headroom kept free when placing the decompressed bf16 shards. Weights
# may never consume the whole device: the CUDA context plus one 2048-token
# activation pass have to fit in what is left.
GPU_RESERVE_GIB = 3.0

BASE_SLUG = "gemma-4-31b-it-qat-w4a16-ct"
BASE_REF = "google/gemma-4/Other/" + BASE_SLUG + "/2"
BASE_PATH_IN_KERNEL = (
    "/kaggle/input/models/google/gemma-4/other/" + BASE_SLUG + "/2")

TARGET_MODULE_REGEX = r".*\.language_model\.layers\.\d+\.self_attn\.(q|o)_proj$"
LORA_R = 8
LORA_ALPHA = 16
N_TEXT_LAYERS = 60
N_FULL_LAYERS = 10
EXPECTED_MODULES = 2 * N_TEXT_LAYERS           # 120
EXPECTED_TENSORS = 4 * N_TEXT_LAYERS           # 240
SLIDING_PER_LAYER = 2 * LORA_R * (5376 + 32 * 256)     # 217,088
FULL_PER_LAYER = 2 * LORA_R * (5376 + 32 * 512)        # 348,160
EXPECTED_TRAINABLE = (50 * SLIDING_PER_LAYER) + (10 * FULL_PER_LAYER)  # 14,336,000

N_STEPS = 8
SEQ_LEN = 512
LONG_SEQ_LEN = 2048
LR = 3e-3
SEED = 1234

# Official scoring configuration: data/official/HARNESS_README.md lines 163-169
# (`setup_vllm_server` in scripts/inference.py).
OFFICIAL_SERVE = {
    "source": "data/official/HARNESS_README.md:163-169",
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.80,
    "max_model_len": 32768,
    "enable_auto_tool_choice": True,
    "tool_call_parser": "gemma4",
    "reasoning_parser": "gemma4",
    "enable_lora": True,
    "max_loras": 8,
    "max_lora_rank": 128,
    "enforce_eager": False,
}

PROMPT = '''
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


def divide(a: float, b: float) -> float:
    """Divide a by b, raising ValueError on division by zero."""
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b


def power(base: int, exponent: int) -> int:
    """Raise base to exponent."""
    return base ** exponent


def modulo(a: int, b: int) -> int:
    """Return a modulo b, raising ValueError when b is zero."""
    if b == 0:
        raise ValueError("cannot take modulo by zero")
    return a % b


def negate(value: int) -> int:
    """Return the additive inverse."""
    return -value


def absolute(value: int) -> int:
    """Return the absolute value."""
    return abs(value)


def clamp(value: int, low: int, high: int) -> int:
    """Clamp value into the inclusive range [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))
'''.strip()


CELL_0 = """# Gate 2 — 31B training / inference compatibility probe

**Scope.** The Kaggle scoring path accepts exactly one base model name,
`gemma-4-31b-it-qat-w4a16-ct` (`ALLOWED_MODEL_NAMES` in
`data/official/HARNESS_README.md:177`). Gate 1 proved the *mechanism* works on the
E4B variant; it did not prove it works here. This notebook tests the real
artifact: LoRA injection into the text layers, a handful of real parameter
updates, an adapter that survives save/reload byte-for-byte, and — the part that
actually matters — whether the official vLLM stack **consumes** that adapter.

**Not in scope.** No SFT, no RL, no dataset: the token ids below are synthetic
random integers. Nothing is submitted to the leaderboard. The frozen baseline
archive is never touched.

**Why `q_proj`/`o_proj` only.** `attention_k_eq_v` is `True` on the 31B and
`v_proj is None` on all ten full-attention layers, so HF's attention reuses the
`k_proj` output as V. vLLM keeps a packed `qkv_proj` and leaves the absent v
slice at zero, and skips it in `set_lora`. A LoRA on `k_proj`/`v_proj` of those
layers would therefore produce different numbers in training and in serving
*without raising anything*. `q_proj` and `o_proj` exist on all 60 layers with
identical semantics in both stacks, so that is the whole target set here.
"""

CELL_1 = """## 1. Preflight: model mount, GPUs, dependencies, runtime budget

`PRE_MODEL` fails loudly rather than silently training nothing: Kaggle's push
step drops an invalid `model_sources` reference without an error (that is how the
first Gate 1 run died), so the mount is verified before anything expensive
happens.
"""

CELL_1_CODE = r'''import gc, glob, json, math, os, platform, shutil, subprocess, time, traceback
from pathlib import Path

T0 = time.time()
RUNTIME_CAP_SECONDS = __RUNTIME_CAP_SECONDS__
# Soft deadline: stop *starting* new serving attempts past this point. Raised from
# 3600 s because the CPU-side decompression pass is slower than the GPU path it
# replaces; the extra 20 minutes is budgeted up front, not borrowed from the E
# group at the end.
SOFT_DEADLINE = T0 + __SOFT_DEADLINE_SECONDS__
# Server deadline: the latest moment a serving attempt may still be running, so
# that after it there is enough of the kernel cap left for the E group and the
# report write. The E group needs roughly 10 minutes; reserving __SERVER_RESERVE__
# keeps the run inside RUNTIME_CAP_SECONDS even when every start attempt burns its
# timeout.
SERVER_DEADLINE = T0 + (RUNTIME_CAP_SECONDS - __SERVER_RESERVE__)

WORK = Path("/kaggle/working")
REPORT_PATH = WORK / "gate2_report.json"

# Which checks each downstream stage owns. `_stage_guard` uses this to record a
# skipped stage as skipped rather than letting the checks go missing, which would
# be indistinguishable from a cell that never ran for some other reason.
STAGE_CHECKS = {
    "lora": ["A5", "A6", "A7", "A8", "B1", "B2", "GC1"],
    "train": ["B3", "B4", "B5", "B6", "B7", "B7b", "D1", "D2", "D3"],
    "save_reload": ["C1", "C2", "C3", "E5"],
    "vllm_start": ["D4", "E7"],
    "e_checks": ["E1", "E2", "E3", "E4", "E6", "E8", "E9"],
}


class StageSkipped(Exception):
    """Raised by `_stage_guard` when an upstream stage failed."""


# Stages that stop being attempted once an earlier one fails. A load failure
# blocks everything; a training failure blocks the adapter verification that would
# otherwise run against an adapter that was never trained.
STAGE_BLOCKED_BY = {
    "lora": ("load",),
    "train": ("load", "lora"),
    "save_reload": ("load", "train"),
    "vllm_start": ("load", "train"),
    "e_checks": ("load", "train"),
}


def _stage_guard(stage):
    """Stop a downstream stage when an upstream one failed.

    Without this, a failed load still walks into LoRA injection, an empty optimiser
    and a full vLLM server: v1 spent its remaining budget serving a base model with
    no adapter, then measured a 404 on /v1/completions and reported eight E checks
    that carried no information. The whole point of a staged gate is that a failure
    at stage N ends the run at stage N, and that the report says *skipped* rather
    than inventing failures for checks that never ran.

    Training failure blocks the adapter verification too (user requirement): an
    adapter whose training died mid-run is not the artifact the E group is supposed
    to measure, so comparing it against the base would answer a question nobody
    asked while still costing a full serving start-up.
    """
    blockers = STAGE_BLOCKED_BY.get(stage, ("load",))
    reason = None
    for cause in blockers:
        if report.get(cause + "_failed"):
            reason = report.get("blocked_by") or ("%s stage failed" % cause)
            break
    if reason is None:
        return
    for cid in STAGE_CHECKS.get(stage, []):
        if cid not in report["checks"]:
            report["checks"][cid] = {
                "ok": False, "value": None,
                "detail": "skipped: %s not reached because %s" % (stage, reason)}
    if stage not in report["skipped_stages"]:
        report["skipped_stages"].append(stage)
    print("SKIP %s :: %s" % (stage, reason), flush=True)
    save_report()
    raise StageSkipped(stage)

report = {
    "gate": "gate2_31b",
    "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown"),
    "started_epoch": time.time(),
    "runtime_cap_seconds": RUNTIME_CAP_SECONDS,
    "soft_deadline_seconds": SOFT_DEADLINE - T0,
    "server_deadline_seconds": SERVER_DEADLINE - T0,
    "checks": {},
    "notes": [],
    "errors": [],
    "load_ok": None,
    "load_failed": False,
    "lora_ok": None,
    "lora_failed": False,
    "train_ok": None,
    "train_failed": False,
    "blocked_by": None,
    "skipped_stages": [],
    "versions": {},
    "gpu": {},
    "host": {},
    "paths": {},
    "timings": {},
}


def save_report():
    try:
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    except Exception as exc:  # noqa: BLE001
        print("save_report failed: %s" % exc, flush=True)


def rec(cid, ok, value, detail=""):
    report["checks"][cid] = {"ok": bool(ok), "value": value, "detail": detail}
    print(("PASS " if ok else "FAIL ") + cid + " :: " + str(detail), flush=True)
    save_report()
    return bool(ok)


def note(msg):
    report["notes"].append(msg)
    print("NOTE " + str(msg), flush=True)


def fail_soft(cid, exc, where):
    msg = "%s: %s" % (type(exc).__name__, exc)
    report["checks"][cid] = {"ok": False, "value": None, "detail": msg}
    report["errors"].append({"where": where, "error": msg})
    print("FAIL " + cid + " :: " + msg, flush=True)
    traceback.print_exc()
    save_report()


def nvidia_smi():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=120)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        return ["nvidia-smi failed: %s" % exc]


# ---- model mount -----------------------------------------------------------
MODEL_ROOT = Path("/kaggle/input/models/google/gemma-4/other")
BASE_SLUG = "gemma-4-31b-it-qat-w4a16-ct"
MODEL = None
candidates = []
if MODEL_ROOT.is_dir():
    candidates = sorted(p for p in MODEL_ROOT.glob(BASE_SLUG + "/*") if p.is_dir())
    if candidates:
        MODEL = candidates[-1]
if MODEL is None:
    hits = sorted(Path("/kaggle/input/models").rglob(BASE_SLUG + "/config.json"))
    if hits:
        MODEL = hits[-1].parent

report["paths"] = {
    "model": str(MODEL) if MODEL else None,
    "model_candidates": [str(p) for p in candidates],
    "model_root_exists": MODEL_ROOT.is_dir(),
    "metric_wheelhouse": "/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse",
    "training_wheels": "/kaggle/input/datasets/taruki/gemma4-training-wheels",
    "competition_wheels": "/kaggle/input/competitions/gemma-4-developer-agent/wheels",
}
if MODEL is not None:
    usage = shutil.disk_usage(str(WORK))
    report["working_disk"] = {"total_gib": round(usage.total / 1024**3, 2),
                              "free_gib": round(usage.free / 1024**3, 2)}
    report["model_disk_gib"] = round(
        sum(p.stat().st_size for p in MODEL.rglob("*") if p.is_file()) / 1024**3, 2)

rec("PRE_MODEL",
    MODEL is not None and (MODEL / "config.json").is_file(),
    {"model": str(MODEL) if MODEL else None,
     "candidates": [p.name for p in candidates],
     "config_present": bool(MODEL is not None and (MODEL / "config.json").is_file())},
    "31B checkpoint mounted")

# ---- GPUs / host -----------------------------------------------------------
try:
    import torch

    report["gpu"] = {
        "count": torch.cuda.device_count(),
        "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "total_memory_gib": [round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                             for i in range(torch.cuda.device_count())],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_smi": nvidia_smi(),
    }
    note("torch %s / cuda %s / %d GPU(s)"
         % (torch.__version__, torch.version.cuda, torch.cuda.device_count()))
except Exception as exc:  # noqa: BLE001
    fail_soft("PRE_GPU", exc, "preflight_gpu")

ram_gib = None
try:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            ram_gib = round(int(line.split()[1]) / 1024**2, 2)
            break
except Exception:  # noqa: BLE001
    pass
report["host"] = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "ram_gib": ram_gib,
}

# ---- dependency snapshot (before install) ----------------------------------
import importlib.metadata as _md

report["versions_before_install"] = {}
for name in ("torch", "transformers", "vllm", "compressed-tensors", "peft", "accelerate"):
    try:
        report["versions_before_install"][name] = _md.version(name)
    except _md.PackageNotFoundError:
        report["versions_before_install"][name] = "NOT INSTALLED"
print(json.dumps(report["versions_before_install"], indent=2), flush=True)

note("pool = 4x L4 (22.03 GiB each); the 31B checkpoints needs ~15.2 GiB per card "
     "in bf16, so headroom is ~6 GiB per card")
save_report()
'''

CELL_2 = """## 2. Install the pinned wheel set (offline)

Identical to Gate 1: the competition wheelhouse supplies torch/transformers/vllm,
the training wheels supply PEFT, and the `cu128` filename normalisation keeps pip
from rejecting local-version wheels.
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
    print("installing %d wheelhouse wheels ..." % len(wheels), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--force-reinstall"] + wheels, check=True)

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

CELL_3 = """## 3. Load the 31B base decompressed to bf16

`CompressedTensorsConfig(run_compressed=False)` turns the W4A16 checkpoint into
plain bf16 tensors at load time without modifying any file on disk, which is what
makes the weights differentiable. A2 proves that happened; A3 records the
structural facts that force the `q_proj`/`o_proj`-only target set; A4 proves
nothing was silently offloaded to host RAM, which would otherwise look like a
successful load while destroying both the speed and the memory numbers this gate
reports.
"""

CELL_3_CODE = r'''import gc, json, re, time
from collections import Counter

import torch

T_LOAD = time.time()
model = None
tokenizer = None
_rss_peak = {"rss_gib": 0.0, "samples": 0}


def _rss_now_gib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except Exception:  # noqa: BLE001
        return None


def _start_rss_sampler():
    """Sample host RSS while the CPU load and decompression run.

    `resource.getrusage().ru_maxrss` alone cannot answer the question this gate
    asks. It is a high-water mark for the whole process, so a peak recorded after
    the load cannot be attributed to the load; and it never comes back down, so it
    cannot show whether the packed and unpacked representations were co-resident
    or sequential. Sampling gives both the peak *during* this cell and the shape of
    the curve.
    """
    import threading
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            v = _rss_now_gib()
            if v is not None:
                _rss_peak["samples"] += 1
                if v > _rss_peak["rss_gib"]:
                    _rss_peak["rss_gib"] = round(v, 2)
            stop.wait(0.5)

    threading.Thread(target=loop, daemon=True).start()
    return stop


def _rss_ru_maxrss_gib():
    try:
        import resource
        # Linux reports KiB, macOS reports bytes.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)
    except Exception:  # noqa: BLE001
        return None


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

    if MODEL is None:
        raise RuntimeError(
            "31B model not mounted under %s; check kernel model_sources (a dropped "
            "reference shows up as model_sources=[] on the remote kernel)"
            % MODEL_ROOT)

    print("model dir:", MODEL, flush=True)
    report["model_files"] = sorted(p.name for p in MODEL.iterdir())[:60]
    report["safetensors_shards"] = sorted(p.name for p in MODEL.glob("*.safetensors"))

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    cfg = json.loads((MODEL / "config.json").read_text())
    text_cfg = cfg["text_config"]
    report["architectures"] = cfg.get("architectures")
    report["text_layers"] = text_cfg["num_hidden_layers"]
    report["layer_types_count"] = dict(Counter(text_cfg["layer_types"]))
    report["attention_k_eq_v"] = text_cfg.get("attention_k_eq_v")
    report["num_kv_shared_layers"] = text_cfg.get("num_kv_shared_layers")
    report["hidden_size"] = text_cfg["hidden_size"]
    report["num_attention_heads"] = text_cfg["num_attention_heads"]
    report["head_dim"] = text_cfg["head_dim"]
    report["global_head_dim"] = text_cfg.get("global_head_dim")
    report["num_key_value_heads"] = text_cfg["num_key_value_heads"]
    report["num_global_key_value_heads"] = text_cfg.get("num_global_key_value_heads")
    report["vocab_size_config"] = text_cfg["vocab_size"]
    report["has_audio_tower"] = cfg.get("audio_config") is not None
    report["vision_use_clipped_linears"] = (cfg.get("vision_config") or {}).get("use_clipped_linears")
    full_idx = [i for i, t in enumerate(text_cfg["layer_types"]) if t != "sliding_attention"]
    report["full_attention_indices"] = full_idx

    rec("A1",
        report["architectures"] == ["Gemma4ForConditionalGeneration"]
        and report["text_layers"] == 60
        and report["layer_types_count"].get("full_attention") == 10,
        {"architectures": report["architectures"],
         "text_layers": report["text_layers"],
         "layer_types": report["layer_types_count"],
         "full_attention_indices": full_idx},
        "31B text stack: 60 layers, 10 full-attention")

    quant_cfg = CompressedTensorsConfig(run_compressed=False)
    n_gpu = torch.cuda.device_count()

    # ---- stage A: load AND decompress on the host ---------------------------
    # `run_compressed=False` keeps the packed int32 tensors and the unpacked bf16
    # tensors alive at the same time while the pass runs. This checkpoint holds
    # 21.70 GiB packed (410 `weight_packed` tensors, all in the text stack) that
    # become 60.88 GiB of bf16, so the pass needs roughly 82.6 GiB of headroom for
    # its duration. Four L4s give 88.1 GiB in total, which is why doing this on the
    # GPUs cannot work: v1 died with "Tried to allocate 442.00 MiB" at 62% of the
    # unpack, with 3.78 GiB reserved-but-unallocated on GPU 1 and every card
    # already holding its planned share. Host RAM here is 188.69 GiB, so the same
    # pass on the CPU is comfortable and, more importantly, leaves the GPUs to
    # receive only the finished bf16 shards.
    rss_stop = _start_rss_sampler()
    t_cpu_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        quantization_config=quant_cfg,
    )
    rss_stop.set()
    report["load_route"] = "cpu_decompress_then_dispatch"
    report["cpu_load_seconds"] = round(time.time() - t_cpu_load, 1)
    report["cpu_load_rss_peak_gib"] = _rss_peak["rss_gib"]
    report["cpu_load_rss_samples"] = _rss_peak["samples"]
    report["cpu_load_ru_maxrss_gib"] = _rss_ru_maxrss_gib()
    model.eval()
    print("CPU load+decompress: %.1fs, RSS peak %.2f GiB (%d samples)"
          % (report["cpu_load_seconds"], report["cpu_load_rss_peak_gib"],
             report["cpu_load_rss_samples"]), flush=True)
    print("devices after CPU load:", sorted({str(p.device) for p in model.parameters()}),
          flush=True)

    # A2: the weights really are bf16 floats now, not packed 4-bit groups.
    status_counter = Counter()
    for _, m in model.named_modules():
        s = getattr(m, "quantization_status", None)
        if s is not None:
            # str() on the enum is "QuantizationStatus.DECOMPRESSED", so compare
            # the trailing symbol rather than the bare word.
            status_counter[str(s).rsplit(".", 1)[-1].lower()] += 1
    report["quantization_status_counts"] = dict(status_counter)

    q_dtype = None
    for name, m in model.named_modules():
        if name.endswith("language_model.layers.0.self_attn.q_proj"):
            w = getattr(m, "weight", None)
            q_dtype = str(w.dtype) if w is not None else None
            break
    run_compressed = getattr(getattr(model, "hf_quantizer", None), "run_compressed", "missing")
    report["hf_quantizer_run_compressed"] = run_compressed
    report["layer0_q_proj_dtype"] = q_dtype
    rec("A2",
        q_dtype == "torch.bfloat16"
        and run_compressed is False
        and status_counter.get("decompressed", 0) > 0
        and status_counter.get("compressed", 0) == 0,
        {"q_dtype": q_dtype, "run_compressed": run_compressed,
         "statuses": dict(status_counter)},
        "all quantized modules decompressed to bf16")

    # ---- stage B: place the finished bf16 shards onto the four cards ---------
    # Weights are now plain bf16 on the host, so placement no longer has to survive
    # a decompression pass on the same card: whatever layout fits the bf16 bytes is
    # final. The plan comes from accelerate rather than from local arithmetic --
    # `get_balanced_memory` sizes the per-card budget and `infer_auto_device_map`
    # produces the map, both using the library's own module accounting. Each card
    # keeps GPU_RESERVE_GIB back for the CUDA context and for activations.
    GPU_RESERVE_GIB = __GPU_RESERVE_GIB__
    totals = [torch.cuda.get_device_properties(i).total_memory / 1024**3
              for i in range(n_gpu)]
    max_memory = {i: int((totals[i] - GPU_RESERVE_GIB) * 1024**3) for i in range(n_gpu)}
    report["gpu_reserve_gib"] = GPU_RESERVE_GIB
    report["max_memory_bytes"] = {str(k): v for k, v in max_memory.items()}

    from accelerate import dispatch_model as _dispatch_model
    from accelerate.utils import get_balanced_memory as _get_balanced_memory
    from accelerate.utils import infer_auto_device_map as _infer_auto_device_map

    no_split = list(getattr(model, "_no_split_modules", []) or [])
    report["no_split_modules"] = no_split
    balanced = _get_balanced_memory(model, max_memory=dict(max_memory),
                                   no_split_module_classes=no_split,
                                   dtype=torch.bfloat16)
    report["balanced_max_memory_bytes"] = {str(k): int(v) for k, v in balanced.items()}
    device_map = _infer_auto_device_map(model, max_memory=balanced,
                                        no_split_module_classes=no_split,
                                        dtype=torch.bfloat16)

    # A plan that reaches for the host is not this experiment: an offloaded layer
    # would still "load", then run at host speed and make every timing and memory
    # number meaningless. Refuse before dispatching, while nothing has moved yet.
    offloaded = {k: str(v) for k, v in device_map.items()
                 if str(v).startswith(("cpu", "disk"))}
    report["device_map"] = {str(k): str(v) for k, v in sorted(device_map.items())}
    report["device_map_modules"] = len(device_map)
    print("device_map over %d modules; offloaded=%d"
          % (len(device_map), len(offloaded)), flush=True)
    if offloaded:
        raise RuntimeError(
            "infer_auto_device_map offloaded %d module(s) to the host, e.g. %r; "
            "the 31B bf16 weights must all fit on the %d cards with %.1f GiB held "
            "back each" % (len(offloaded), list(offloaded.items())[:4], n_gpu,
                           GPU_RESERVE_GIB))

    # Planned per-card bytes, derived from the plan itself, so D5 can compare the
    # intention against the allocator's own reading. Tied tensors are counted once
    # (by object identity) because they are one storage, not two.
    _sizes = {}
    _counted = set()
    _modules = dict(model.named_modules())
    for name, dev in device_map.items():
        mod = _modules.get(name)
        if mod is None:
            continue
        # recurse=True: the map may name a parent module rather than each leaf.
        # `id()` dedup keeps that correct when both a parent and its child appear.
        for prm in mod.parameters(recurse=True):
            if id(prm) in _counted:
                continue
            _counted.add(id(prm))
            _sizes[str(dev)] = _sizes.get(str(dev), 0) + prm.numel() * prm.element_size()
        for buf in mod.buffers(recurse=True):
            if id(buf) in _counted:
                continue
            _counted.add(id(buf))
            _sizes[str(dev)] = _sizes.get(str(dev), 0) + buf.numel() * buf.element_size()
    report["planned_per_card_gib"] = {k: round(v / 1024**3, 3) for k, v in _sizes.items()}
    report["planned_total_gib"] = round(sum(_sizes.values()) / 1024**3, 3)

    model = _dispatch_model(model, device_map=device_map)
    model.eval()

    # A3: the reason this gate does not touch k_proj/v_proj.
    #
    # Match on the SUFFIX. `named_modules()` names the attention stack
    # `model.language_model.layers.N.self_attn.v_proj` because the stack hangs under
    # `.model`; a check keyed on `language_model.layers...` therefore matches no
    # layer at all, and "the set of layers missing v_proj" becomes "all of them",
    # which reads as a finding about the model when it is a defect here. Suffix
    # matching survives a prefix change, and `v_present` below refuses to let an
    # empty match masquerade as an answer.
    modules = dict(model.named_modules())
    v_present = sorted(
        int(m.group(1)) for m in (
            re.match(r".*language_model\.layers\.(\d+)\.self_attn\.v_proj$", n)
            for n in modules) if m)
    v_absent = [i for i in range(report["text_layers"]) if i not in set(v_present)]
    report["v_proj_absent_indices"] = v_absent
    report["v_proj_present_indices"] = v_present
    report["v_proj_sample_keys"] = sorted(
        n for n in modules if n.endswith("self_attn.v_proj"))[:3]
    rec("A3",
        report["attention_k_eq_v"] is True
        and report["num_kv_shared_layers"] == 0
        and bool(v_present)
        and v_absent == full_idx,
        {"attention_k_eq_v": report["attention_k_eq_v"],
         "num_kv_shared_layers": report["num_kv_shared_layers"],
         "v_proj_absent": v_absent,
         "v_proj_present_count": len(v_present),
         "v_proj_sample_keys": report["v_proj_sample_keys"],
         "full_attention_indices": full_idx},
        "v_proj exists on every sliding layer and is absent on exactly the full ones")

    # A4: every parameter AND every buffer must live on a GPU. `dispatch_model`
    # can legitimately leave things on the host -- a module missing from the map,
    # a buffer it does not consider a parameter, a buffer attached to a module the
    # hook only moves on forward. Any of those would "succeed" while silently
    # destroying both the speed and the memory numbers this gate reports, and a
    # CPU-resident buffer would resurface later as a device mismatch inside a
    # backward pass rather than as a load failure.
    hf_map = {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
    bad_modules = {k: v for k, v in hf_map.items()
                   if v.startswith("cpu") or v.startswith("disk")}
    param_devices = {}
    for n, prm in model.named_parameters():
        param_devices.setdefault(str(prm.device), []).append(n)
    buffer_devices = {}
    for n, buf in model.named_buffers():
        buffer_devices.setdefault(str(buf.device), []).append(n)
    non_cuda_params = sorted(d for d in param_devices if not d.startswith("cuda"))
    non_cuda_buffers = sorted(d for d in buffer_devices if not d.startswith("cuda"))

    # The 31B sets tie_word_embeddings=True, so lm_head.weight and
    # model.language_model.embed_tokens.weight are one storage. If a placement ever
    # split them, the move would quietly produce two independent matrices and the
    # LM head would stop matching the embedding table: no error, just wrong logits.
    # Record the relationship explicitly rather than trusting the plan.
    try:
        lm_head_w = model.get_output_embeddings().weight
        embed_w = model.get_input_embeddings().weight
        tied_shared = lm_head_w is embed_w
        tied_equal = True if tied_shared else bool(torch.equal(lm_head_w, embed_w))
        tied_devices = [str(lm_head_w.device), str(embed_w.device)]
    except Exception as exc:  # noqa: BLE001
        tied_shared, tied_equal = None, None
        tied_devices = ["%s: %s" % (type(exc).__name__, exc)]

    report["device_map"] = hf_map
    report["param_devices"] = {d: len(v) for d, v in param_devices.items()}
    report["buffer_devices"] = {d: len(v) for d, v in buffer_devices.items()}
    report["non_cuda_param_devices"] = non_cuda_params
    report["non_cuda_buffer_devices"] = non_cuda_buffers
    # A9: the tie itself, asserted on its own rather than folded into A4. Two
    # independent 262144x5376 matrices load without error and only show up as wrong
    # logits, so this is the one failure mode in the load stage that cannot be left
    # to a generic conjunction. Check three independent views: object identity, the
    # library's own tie detection, and value equality.
    try:
        from accelerate.utils import find_tied_parameters as _find_tied
        tied_groups = _find_tied(model)
    except Exception as exc:  # noqa: BLE001
        tied_groups = ["%s: %s" % (type(exc).__name__, exc)]

    def _names_of(group):
        return sorted(n for n in group if isinstance(n, str))

    tied_named = [g for g in tied_groups if isinstance(g, list) and len(g) > 1]
    tie_seen_by_accelerate = any(
        any("lm_head.weight" in n for n in g) and any("embed_tokens.weight" in n for n in g)
        for g in tied_named)
    tie_in_same_device = (tied_shared is True
                          and len(set(tied_devices)) == 1)
    expected_tie = bool(getattr(model.config, "tie_word_embeddings", True))

    report["tied_check"] = {
        "config_tie_word_embeddings": expected_tie,
        "is_same_storage": tied_shared,
        "values_equal": tied_equal,
        "devices": tied_devices,
        "same_device": len(set(tied_devices)) == 1,
        "accelerate_tied_groups": [_names_of(g) for g in tied_named],
        "tie_seen_by_accelerate": tie_seen_by_accelerate,
        "lm_head_data_ptr": getattr(lm_head_w, "data_ptr", lambda: None)(),
        "embed_data_ptr": getattr(embed_w, "data_ptr", lambda: None)(),
    }
    rec("A9",
        expected_tie is True
        and tied_shared is True
        and tied_equal is True
        and tie_in_same_device
        and tie_seen_by_accelerate,
        report["tied_check"],
        "lm_head.weight and embed_tokens.weight are one storage on one device")

    report["tied_embeddings"] = {"is_same_storage": tied_shared,
                                 "values_equal": tied_equal,
                                 "devices": tied_devices}
    rec("A4",
        (not bad_modules) and (not non_cuda_params) and (not non_cuda_buffers)
        and bool(hf_map) and tied_equal is True,
        {"offloaded": bad_modules,
         "non_cuda_params": non_cuda_params,
         "non_cuda_buffers": non_cuda_buffers,
         "param_counts_by_device": report["param_devices"],
         "buffer_counts_by_device": report["buffer_devices"],
         "tied_embeddings": report["tied_embeddings"]},
        "every parameter and buffer on a GPU, tied embedding pair kept together")

    # D5 (recorded, not gated): what the cards actually hold once the hooks are in
    # place. `planned_per_card_gib` is an intention derived from tensor sizes; this
    # is the allocator's own reading. If the two disagree by more than the reserve,
    # the plan is not the layout that actually happened.
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    actual = {}
    for i_gpu in range(n_gpu):
        free_b, total_b = torch.cuda.mem_get_info(i_gpu)
        actual[str(i_gpu)] = {
            "used_gib": round((total_b - free_b) / 1024**3, 3),
            "total_gib": round(total_b / 1024**3, 2),
            "torch_allocated_gib": round(torch.cuda.memory_allocated(i_gpu) / 1024**3, 3),
        }
    report["actual_gpu_usage_after_load"] = actual
    planned_map = report.get("planned_per_card_gib", {})
    report["plan_vs_actual_gib"] = {
        d: round(actual[d]["torch_allocated_gib"] - planned_map.get(d, 0.0), 3)
        for d in actual}
    rec("D5",
        all(v["torch_allocated_gib"] <= v["total_gib"] for v in actual.values()),
        {"actual": actual,
         "planned_per_card_gib": planned_map,
         "planned_total_gib": report.get("planned_total_gib"),
         "delta_gib": report["plan_vs_actual_gib"]},
        "measured per-card occupancy after dispatch")

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        clip = [n for n, m in model.named_modules() if isinstance(m, Gemma4ClippableLinear)]
    except Exception as exc:  # noqa: BLE001
        note("ClippableLinear import failed: %s" % exc)
        clip = []
    report["clippable_linear_count"] = len(clip)
    report["clippable_linear_sample"] = clip[:6]
    note("Gemma4ClippableLinear count: %d (informational; vision_config.use_clipped_linears "
         "is %r here and the target regex never reaches the tower)"
         % (len(clip), report["vision_use_clipped_linears"]))

    report["model_footprint_gib"] = round(model.get_memory_footprint() / 1024**3, 2)
    note("decompressed footprint: %.2f GiB" % report["model_footprint_gib"])
    save_report()
except Exception as exc:  # noqa: BLE001
    fail_soft("A1", exc, "load")

report["timings"]["load_s"] = round(time.time() - T_LOAD, 1)
# Fail-fast gate. Set once, read by every later cell. "Loaded" means the model is
# a dispatched, all-on-GPU, decompressed 31B -- not merely that no exception was
# raised. A run where the load fails must not go on to attach LoRA, train, or stand
# up the whole vLLM server: v1 did exactly that and spent its remaining budget
# serving a base model with no adapter, then measured a 404 on /v1/completions and
# reported eight E checks as failures that carried no information.
_loaded_ok = all(
    report["checks"].get(cid, {}).get("ok") is True
    for cid in ("A1", "A2", "A3", "A4", "A9"))
report["load_ok"] = _loaded_ok
if not _loaded_ok:
    report["load_failed"] = True
    report["blocked_by"] = "load stage failed (see A1-A4 and errors[where=load])"
    note("load_failed: LoRA, training, save and serving will be skipped")
print("load_ok:", _loaded_ok, flush=True)
save_report()
'''

CELL_4 = """## 4. Attach LoRA to the text attention projections only

The regex is anchored on `language_model`, so it cannot reach the vision tower
even if a future variant wraps its tower layers in `Gemma4ClippableLinear`. B1/B2
prove the base is frozen and that the trainable count matches the closed form
exactly — for the 31B that is 50 sliding layers at `2·r·(5376+8192)` plus 10 full
layers at `2·r·(5376+16384)` = `14,336,000`.

This cell also enables gradient checkpointing: at 60.9 GiB of weights the run has
only ~6 GiB of headroom per L4, which is not enough for the MLP activations of a
2048-token step (`21504 × 2048 × 2 B ≈ 88 MB per layer`, times 60 layers).
"""

CELL_4_CODE = r'''import re
import time

import torch

T_LORA = time.time()
TARGET_MODULE_REGEX = r".*\.language_model\.layers\.\d+\.self_attn\.(q|o)_proj$"
EXPECTED_MODULES = 120
EXPECTED_TRAINABLE = 14336000

try:
    _stage_guard("lora")

    # The image ships torchao 0.10.0, below peft's TORCHAO_MINIMUM_VERSION (0.16.0).
    # peft's is_torchao_available() RAISES ImportError for that combination instead
    # of returning False, and LoraModel._create_new_module calls the torchao
    # dispatcher unconditionally for EVERY candidate target. Injection therefore
    # dies on the first matched module with "Found an incompatible version of
    # torchao. Found version 0.10.0, but only versions above 0.16.0 are supported"
    # -- even though our targets are plain nn.Linear and no torchao backend is
    # used anywhere in this probe. This patches module-level names in this process
    # only; the installed environment is untouched, so the vLLM subprocess started
    # in stage 3 is unaffected.
    import peft.import_utils as _peft_import_utils
    import peft.tuners.lora.model as _peft_lora_model

    _peft_import_utils.is_torchao_available = lambda: False
    _peft_lora_model.is_torchao_available = lambda: False
    _peft_lora_model.dispatch_torchao = lambda *a, **k: None
    report["torchao_dispatcher_neutralised"] = True
    note("peft torchao dispatcher neutralised for this process (image torchao 0.10.0 "
         "< peft minimum 0.16.0; no torchao backend is used here)")

    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import LoraLayer

    modules = dict(model.named_modules())
    matched = [n for n in modules if re.fullmatch(TARGET_MODULE_REGEX, n)]
    report["n_matched_modules"] = len(matched)
    report["matched_sample"] = matched[:6]
    report["matched_suffixes"] = sorted({n.split(".")[-1] for n in matched})

    # A5: nothing outside language_model, and specifically nothing in a tower.
    outside = [n for n in matched if ".language_model." not in n]
    tower = [n for n in matched if "vision_tower" in n or "audio_tower" in n]
    types = sorted({type(modules[n]).__name__ for n in matched})
    rec("A5", (not outside) and (not tower),
        {"outside_language_model": outside[:5], "tower_hits": tower[:5]},
        "every match is a text-layer attention projection")
    note("matched module classes: %s" % types)
    report["matched_module_classes"] = types

    layer_ids = sorted({int(n.split("layers.")[1].split(".")[0]) for n in matched})
    rec("A6",
        len(matched) == EXPECTED_MODULES and layer_ids == list(range(60)),
        {"n_matched": len(matched), "expected": EXPECTED_MODULES,
         "layers_covered": len(layer_ids),
         "suffixes": report["matched_suffixes"]},
        "120 projections: q_proj and o_proj on all 60 layers")

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

    lora_modules = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    tower_lora = [n for n in lora_modules if "vision_tower" in n or "audio_tower" in n]
    report["n_lora_layers"] = len(lora_modules)
    report["lora_layer_suffixes"] = sorted({n.split(".")[-1] for n in lora_modules})
    rec("A7",
        len(lora_modules) == EXPECTED_MODULES and not tower_lora,
        {"n_lora_layers": len(lora_modules), "expected": EXPECTED_MODULES,
         "wrapped_tower": tower_lora[:5],
         "suffixes": report["lora_layer_suffixes"]},
        "120 LoraLayer wrappers, none in a tower")

    lora_keys = [n for n, _ in model.named_parameters() if "lora_" in n]
    report["n_lora_tensors"] = len(lora_keys)
    rec("A8", len(lora_keys) == 240,
        {"n_lora_tensors": len(lora_keys), "expected": 240},
        "240 tensors = 120 modules x lora_A/lora_B")

    # B1: the base must be fully frozen.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    non_lora = [n for n, _ in trainable if "lora_" not in n]
    report["n_trainable_params"] = sum(p.numel() for _, p in trainable)
    rec("B1", not non_lora,
        {"non_lora_trainable": non_lora[:10], "n_trainable_tensors": len(trainable)},
        "base is frozen, only lora_ parameters require grad")

    # B2: closed-form parameter count. sliding 2*r*(5376+8192), full 2*r*(5376+16384).
    rec("B2", report["n_trainable_params"] == EXPECTED_TRAINABLE,
        {"n_trainable_params": report["n_trainable_params"],
         "expected": EXPECTED_TRAINABLE,
         "sliding_per_layer": 2 * 8 * (5376 + 32 * 256),
         "full_per_layer": 2 * 8 * (5376 + 32 * 512)},
        "trainable count matches the closed form for r=8")

    # Gradient checkpointing: 60.9 GiB of weights leaves ~6 GiB per L4, far too
    # little for the un-checkpointed MLP activations of a 2048-token step.
    gc_ok = False
    gc_details = {}
    try:
        inner = model.base_model.model
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        n_on = sum(1 for m in model.modules()
                   if getattr(m, "gradient_checkpointing", False))
        gc_details = {"wrapped_class": type(inner).__name__,
                      "modules_with_flag": n_on,
                      "use_reentrant": False}
        gc_ok = n_on >= 60
    except Exception as exc:  # noqa: BLE001
        gc_details = {"error": "%s: %s" % (type(exc).__name__, exc)}
    report["gradient_checkpointing"] = gc_details
    rec("GC1", gc_ok, gc_details,
        "gradient checkpointing active on every decoder layer (use_reentrant=False)")

    # Gradient-checkpointing + frozen base needs input grads enabled or the whole
    # graph is dropped: with use_reentrant=False and no input requiring grad, the
    # checkpointed segments receive no gradient from the leaf tensors.
    try:
        model.enable_input_require_grads()
        report["input_require_grads"] = True
    except Exception as exc:  # noqa: BLE001
        report["input_require_grads"] = "%s: %s" % (type(exc).__name__, exc)
    note("input_require_grads: %r" % report["input_require_grads"])
    save_report()
except StageSkipped:
    pass
except Exception as exc:  # noqa: BLE001
    fail_soft("A7", exc, "lora_inject")

report["timings"]["lora_s"] = round(time.time() - T_LORA, 1)
# LoRA injection counts as done only when the adapter actually attached to the
# expected modules. An injection that raised, or that attached to the wrong count,
# leaves nothing worth training, so the next stage is refused rather than wasting a
# full training pass on an empty optimiser.
_lora_ids = ("A5", "A6", "A7", "A8", "B1", "B2")
_lora_ok = all(report["checks"].get(cid, {}).get("ok") is True for cid in _lora_ids)
report["lora_ok"] = _lora_ok
if not _lora_ok:
    report["lora_failed"] = True
    if not report.get("blocked_by"):
        report["blocked_by"] = "LoRA injection failed (see A5-A8/B1-B2)"
    note("lora_failed: training will be skipped")
print("lora_ok:", _lora_ok, flush=True)
save_report()
'''

CELL_5 = """## 5. Real parameter updates, including one planned-length step

Eight short steps at 512 tokens prove the optimiser loop, then a single step at
the planned training length of **2048** tokens proves the memory budget. The
cross-entropy over 262,144 logits is reduced in sequence chunks: float32 logits
for `2048 × 262144` are 2.1 GiB *by itself*, before the softmax buffer and its
gradient, which does not fit in the available headroom.

Expected starting loss: random token ids over a 262,144-entry vocabulary give
`ln(262144) ≈ 12.477`. The measured loss should sit near that and fall.

`lora_B` is zero-initialised, so the `lora_A` gradient at step 0 is **exactly**
zero by construction and only becomes non-zero from step 1 — B5 therefore gates
on `lora_B`, not on `lora_A`.
"""

CELL_5_CODE = r'''import gc
import math
import time

import torch
import torch.nn.functional as F

T_TRAIN = time.time()
N_STEPS = 8
SEQ_LEN = 512
LONG_SEQ_LEN = 2048
LR = 3e-3
SEED = 1234


def _reset_peaks():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def _peaks():
    out = {}
    for i in range(torch.cuda.device_count()):
        out[str(i)] = {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(i) / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(i) / 1024**3, 3),
            "total_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        }
    return out


def _input_device(m):
    getter = getattr(m, "get_input_embeddings", None)
    if callable(getter):
        try:
            emb = getter()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
        except Exception:  # noqa: BLE001
            pass
    for name, p in m.named_parameters():
        if "embed_tokens" in name:
            return p.device
    return torch.device("cuda:0")


def _ce_loss(logits, ids):
    """Next-token CE over an (B, L, V) logits tensor, avoiding a 1 GiB copy.

    The naive shift `F.cross_entropy(logits[:, :-1, :].reshape(-1, V), ids[:, 1:])`
    is not a view: slicing the sequence dimension breaks contiguity, so `.reshape`
    materialises a full copy of the logits. At 2048 tokens with a 262,144-entry
    vocabulary that copy alone is 1.07 GiB in bf16. Instead the *labels* are
    shifted (cheap, int64, same size as ids) and the logits are consumed in place
    with `.view(-1, V)`, which is free. The final position is masked with
    `ignore_index=-100` and the sum is divided by the L-1 real tokens, so the value
    is identical to the naive form.

    Left in bf16 deliberately: `cross_entropy` accumulates internally in float32,
    while an explicit `.float()` cast would add another 2.1 GiB of saved tensors.

    The labels are built on the logits' device on purpose. This model is sharded
    across four GPUs and the LM head does not have to sit on the same card as
    `embed_tokens`: Gate 1 measured `ids` landing on cuda:1 while the logits came
    from another card, and an in-place copy between devices raises "Expected all
    tensors to be on the same device". Moving the label slice is free; moving the
    logits would not be.
    """
    batch, length, vocab = logits.shape
    labels = torch.full((batch, length), -100, dtype=torch.long, device=logits.device)
    labels[:, :-1] = ids[:, 1:length].to(logits.device)
    total = F.cross_entropy(logits.view(-1, vocab), labels.view(-1),
                            reduction="sum", ignore_index=-100)
    return total / float(max(1, length - 1) * batch)


def _run_step(ids, optim):
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits
    loss = _ce_loss(logits, ids)
    loss_val = float(loss.detach())
    loss.backward()
    del loss
    grads = {n: p.grad for n, p in lora_params}
    finite = all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
    # The plan spreads the LoRA tensors across several GPUs, so every tensor is
    # reduced to a Python float first: float(sum(t1, t2, ...)) across devices raises
    # "Expected all tensors to be on the same device". This is the exact bug that
    # killed Gate 1 v3/v4 -- the reduction, not the training.
    b_grad = float(sum(g.detach().float().abs().sum().item()
                       for n, g in grads.items() if "lora_B" in n and g is not None))
    a_grad = float(sum(g.detach().float().abs().sum().item()
                       for n, g in grads.items() if "lora_A" in n and g is not None))
    optim.step()
    optim.zero_grad(set_to_none=True)
    del out, logits, grads
    return loss_val, finite, b_grad, a_grad


try:
    _stage_guard("train")

    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    trainable = [p for _, p in lora_params]
    before = {n: p.detach().float().clone() for n, p in lora_params}

    per_dev = {}
    for n, p in model.named_parameters():
        slot = per_dev.setdefault(str(p.device), {"params": 0, "gib": 0.0, "dtypes": set()})
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
    report["expected_random_loss"] = round(math.log(vocab), 4)

    gen = torch.Generator().manual_seed(SEED)
    input_device = _input_device(model)
    getter = getattr(model, "get_input_embeddings", None)
    report["train_input_device"] = str(input_device)
    try:
        report["embed_device"] = str(getter().weight.device) if callable(getter) else None
    except Exception:  # noqa: BLE001
        report["embed_device"] = None
    note("inputs placed on %s (device_map shards the model across %d GPUs)"
         % (input_device, torch.cuda.device_count()))

    # ---- phase 1: eight short steps ---------------------------------------
    _reset_peaks()
    t_short = time.time()
    steps = []
    loss_ok = True
    grad_finite_ok = True
    b_grad_ok = True
    for step in range(N_STEPS):
        ids = torch.randint(0, vocab, (1, SEQ_LEN), generator=gen).to(input_device)
        t_step = time.time()
        loss_val, finite, b_grad, a_grad = _run_step(ids, optim)
        dt = time.time() - t_step
        steps.append({"step": step, "seq_len": SEQ_LEN, "loss": round(loss_val, 6),
                      "grad_finite": finite,
                      "lora_B_grad_abs_sum": round(b_grad, 4),
                      "lora_A_grad_abs_sum": round(a_grad, 4),
                      "seconds": round(dt, 3)})
        if not math.isfinite(loss_val):
            loss_ok = False
        if not finite:
            grad_finite_ok = False
        if not b_grad > 0.0:
            b_grad_ok = False
        print("step %d seq %d loss %.5f B_grad %.2f A_grad %.2f %.2fs"
              % (step, SEQ_LEN, loss_val, b_grad, a_grad, dt), flush=True)
        del ids
        save_report()

    report["training_steps"] = steps
    report["short_phase_seconds"] = round(time.time() - t_short, 2)
    short_times = [s["seconds"] for s in steps]
    report["short_step_seconds"] = {
        "mean": round(sum(short_times) / len(short_times), 3),
        "min": min(short_times), "max": max(short_times)}
    report["peak_memory_short_phase"] = _peaks()

    rec("B3", loss_ok, [s["loss"] for s in steps], "all losses finite")
    rec("B4", grad_finite_ok, len(steps), "all LoRA gradients finite")
    rec("B5", b_grad_ok, [s["lora_B_grad_abs_sum"] for s in steps],
        "lora_B gradient non-zero at every step")

    # ---- phase 2: one step at the planned training length -------------------
    # 2048 is the planned training length and the point of this phase. If it OOMs
    # that is a real finding about the memory budget, so the failure is recorded
    # verbatim and the largest length that does fit is measured too -- reporting
    # "2048 failed" alone would leave the actual headroom unknown.
    _reset_peaks()
    t_long_total = time.time()
    long_report = None
    oom_errors = []
    for attempt_len in (LONG_SEQ_LEN, 1024, 512):
        ids_try = None
        try:
            torch.cuda.empty_cache()
            gc.collect()
            ids_try = torch.randint(0, vocab, (1, attempt_len),
                                    generator=gen).to(input_device)
            torch.cuda.synchronize()
            t_long = time.time()
            long_loss, long_finite, long_b, long_a = _run_step(ids_try, optim)
            torch.cuda.synchronize()
            long_dt = time.time() - t_long
            long_report = {
                "seq_len": attempt_len,
                "planned_seq_len": LONG_SEQ_LEN,
                "at_planned_length": attempt_len == LONG_SEQ_LEN,
                "loss": round(long_loss, 6),
                "seconds": round(long_dt, 3),
                "grad_finite": long_finite,
                "lora_B_grad_abs_sum": round(long_b, 4),
                "lora_A_grad_abs_sum": round(long_a, 4),
                "peak_memory": _peaks(),
            }
            print("long step seq %d loss %.5f B_grad %.2f %.2fs"
                  % (attempt_len, long_loss, long_b, long_dt), flush=True)
            del ids_try
            break
        except torch.OutOfMemoryError as exc:
            oom_errors.append({"seq_len": attempt_len,
                               "error": "%s: %s" % (type(exc).__name__, exc)})
            print("OOM at seq_len %d: %s" % (attempt_len, exc), flush=True)
            # A step that OOM'd inside backward has already written partial
            # gradients into p.grad. Without this reset the next (shorter) attempt
            # would accumulate on top of them and its "gradient is non-zero"
            # evidence would be contaminated by the failed attempt.
            try:
                optim.zero_grad(set_to_none=True)
            except Exception:  # noqa: BLE001
                for _n, _p in lora_params:
                    _p.grad = None
            try:
                del ids_try
            except Exception:  # noqa: BLE001
                pass
            gc.collect()
            torch.cuda.empty_cache()
            continue

    report["long_step"] = long_report
    report["long_step_oom"] = oom_errors
    report["long_phase_seconds"] = round(time.time() - t_long_total, 2)
    report["peak_memory_long_phase"] = _peaks()

    rec("B7",
        bool(long_report) and long_report["grad_finite"]
        and math.isfinite(long_report["loss"]) and long_report["lora_B_grad_abs_sum"] > 0.0,
        {"result": long_report, "oom": oom_errors},
        "forward+backward+optimiser step at a long sequence length")
    if long_report is not None and not long_report["at_planned_length"]:
        rec("B7b", False,
            {"achieved_seq_len": long_report["seq_len"], "planned": LONG_SEQ_LEN,
             "oom": [e["seq_len"] for e in oom_errors]},
            "planned training length %d did NOT fit; %d is the largest that ran"
            % (LONG_SEQ_LEN, long_report["seq_len"]))
    elif long_report is not None:
        rec("B7b", True, {"achieved_seq_len": long_report["seq_len"]},
            "planned training length reached")

    # ---- weight change ------------------------------------------------------
    after = {n: p.detach().float().clone() for n, p in lora_params}
    per_tensor = {n: (after[n] - before[n]).abs().max().item() for n in before}
    max_delta = max(per_tensor.values()) if per_tensor else 0.0
    n_changed = sum(1 for v in per_tensor.values() if v > 0)
    b_after_abs = float(sum(p.detach().float().abs().sum().item()
                            for n, p in lora_params if "lora_B" in n))
    report["max_weight_delta"] = max_delta
    report["n_tensors_changed"] = n_changed
    report["lora_B_abs_sum_after"] = b_after_abs
    report["delta_sample"] = dict(sorted(per_tensor.items(), key=lambda kv: -kv[1])[:5])
    rec("B6", max_delta > 0.0 and b_after_abs > 0.0,
        {"max_weight_delta": max_delta, "n_changed": n_changed,
         "n_tensors": len(per_tensor), "lora_B_abs_sum_after": round(b_after_abs, 4)},
        "%d/%d tensors changed after the optimiser steps" % (n_changed, len(per_tensor)))

    # ---- D group: memory + timing ------------------------------------------
    short_peaks = report["peak_memory_short_phase"]
    long_peaks = report["peak_memory_long_phase"]
    combined = {}
    for dev in sorted(set(short_peaks) | set(long_peaks)):
        s = short_peaks.get(dev, {})
        l = long_peaks.get(dev, {})
        combined[dev] = {
            "short_phase_peak_reserved_gib": s.get("peak_reserved_gib"),
            "long_phase_peak_reserved_gib": l.get("peak_reserved_gib"),
            "total_gib": (s.get("total_gib") or l.get("total_gib")),
        }
    report["peak_memory_by_phase"] = combined
    over = [d for d, v in combined.items()
            if (v["short_phase_peak_reserved_gib"] or 0) > v["total_gib"]
            or (v["long_phase_peak_reserved_gib"] or 0) > v["total_gib"]]

    rec("D1", bool(short_peaks) and bool(long_peaks), combined,
        "per-device peaks recorded for both the short and the planned-length phase")
    rec("D2", not over, over, "no device exceeded its capacity in either phase")
    rec("D3",
        report.get("long_step") is not None
        and report["long_step"]["seconds"] > 0
        and report["short_step_seconds"]["mean"] > 0,
        {"long_step_seconds": (report.get("long_step") or {}).get("seconds"),
         "long_step_seq_len": (report.get("long_step") or {}).get("seq_len"),
         "short_step_seconds": report["short_step_seconds"],
         "footprint_gib": report.get("model_footprint_gib")},
        "single-step timing recorded at both lengths")

    model.eval()
except StageSkipped:
    pass
except Exception as exc:  # noqa: BLE001
    fail_soft("B3", exc, "train")

report["timings"]["train_s"] = round(time.time() - T_TRAIN, 1)
# Training counts as done only when every training check actually passed. A cell
# that raised catches it here, and a cell that finished with a dead optimiser or a
# zero gradient is just as unusable as one that raised: either way the adapter on
# disk is not a trained adapter, and the E group must not go on to measure it.
_train_ids = ("B3", "B4", "B5", "B6", "B7")
_train_ok = all(report["checks"].get(cid, {}).get("ok") is True for cid in _train_ids)
report["train_ok"] = _train_ok
if not _train_ok:
    report["train_failed"] = True
    if not report.get("blocked_by"):
        report["blocked_by"] = "training stage failed (see B3-B7 and errors[where=train])"
    note("train_failed: adapter verification will be skipped")
print("train_ok:", _train_ok, flush=True)
save_report()
'''

CELL_6 = """## 6. Save, reload, and locate the adapter in the module tree

C1 compares the safetensors on disk bit-for-bit against the live tensors, C2
reloads through PEFT into a second slot and compares again, C3 records the
artifacts and their size against the competition's `< 3 GiB` unpacked budget, and
E5 confirms every saved key maps onto a module PEFT really wrapped.

PEFT stores the same tensor under three different names — disk
`….q_proj.lora_A.weight`, default slot `….q_proj.lora_A.default.weight`, reload
slot `….q_proj.lora_A.reloaded.weight` — so all three are normalised to one
canonical key first.

Two adapters are written: `trained_gate2` (the real result) and `zero_b_control`
(identical but with every `lora_B` zeroed, i.e. mathematically inert). Serving
both is what separates "the adapter is being used" from "the adapter is silently
ignored".
"""

CELL_6_CODE = r'''import json, re, shutil
import time

import torch
from pathlib import Path

T_SAVE = time.time()
ADAPTER_TRAINED = WORK / "gate2_adapter"
LAYOUT = WORK / "gate2_layout"
CANON_RE = re.compile(r"(layers\.\d+\.self_attn\.(?:q|o)_proj)\.lora_([AB])\.")


def canon(name):
    """Canonical adapter key, independent of the PEFT adapter-name segment."""
    m = CANON_RE.search(name)
    return "%s.lora_%s.weight" % (m.group(1), m.group(2)) if m else None


try:
    _stage_guard("save_reload")

    from safetensors.torch import load_file, save_file

    # Guard: if injection failed, `model` is still a plain transformers model and
    # save_pretrained() would try to write the whole 61 GiB base (and blow up in
    # transformers' revert_weight_conversion). Fail loudly instead of emitting a
    # bogus "adapter" directory.
    from peft import PeftModel as _PeftModel
    if not isinstance(model, _PeftModel):
        raise RuntimeError(
            "model is %s, not a PeftModel: LoRA injection did not run, so there is "
            "no adapter to save (see A5-A8 above)" % type(model).__name__)

    if ADAPTER_TRAINED.exists():
        shutil.rmtree(ADAPTER_TRAINED)
    model.save_pretrained(str(ADAPTER_TRAINED))
    adapter_files = sorted(p.name for p in ADAPTER_TRAINED.rglob("*") if p.is_file())
    adapter_bytes = sum(p.stat().st_size for p in ADAPTER_TRAINED.rglob("*") if p.is_file())
    report["adapter_files"] = adapter_files
    report["adapter_bytes"] = adapter_bytes
    report["adapter_mib"] = round(adapter_bytes / 1024**2, 2)
    rec("C3",
        "adapter_config.json" in adapter_files
        and "adapter_model.safetensors" in adapter_files
        and adapter_bytes < 3 * 1024**3,
        {"files": adapter_files, "mib": report["adapter_mib"],
         "budget_mib": round(3 * 1024, 1)},
        "adapter artifacts present and inside the submission size budget")

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
            if c is not None and c not in mem:
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

    # E5: every saved key must map onto a module PEFT actually wrapped.
    from peft.tuners.lora.layer import LoraLayer
    lora_module_names = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    distinct = sorted({c.rsplit(".lora_", 1)[0] for c in disk if c})
    unknown = [p for p in distinct
               if not any(n.endswith(p) for n in lora_module_names)]
    suffixes_used = sorted({p.split(".")[-1] for p in distinct})
    a_keys = sum(1 for c in disk if c and ".lora_A." in c)
    b_keys = sum(1 for c in disk if c and ".lora_B." in c)
    rec("E5",
        len(distinct) == report.get("n_matched_modules")
        and not unknown
        and set(suffixes_used) <= {"q_proj", "o_proj"}
        and a_keys == b_keys == len(distinct),
        {"distinct_modules": len(distinct),
         "expected": report.get("n_matched_modules"),
         "wrapped_lora_layers": len(lora_module_names),
         "unknown": unknown[:5],
         "suffixes": suffixes_used,
         "lora_A_keys": a_keys,
         "lora_B_keys": b_keys},
        "adapter keys map onto injected modules only")

    # C2: reload through PEFT into a second adapter slot.
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

    # Serving layout: one real adapter + one zero-lora_B control.
    if LAYOUT.exists():
        shutil.rmtree(LAYOUT)
    trained_dir = LAYOUT / "adapters" / "trained_gate2"
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
    b_abs_zero = float(sum(v.float().abs().sum().item() for k, v in zeroed.items()
                           if "lora_B" in k))
    b_abs_trained = float(sum(v.float().abs().sum().item() for k, v in on_disk.items()
                              if "lora_B" in k))
    report["zero_b_abs_sum"] = b_abs_zero
    report["trained_b_abs_sum"] = b_abs_trained
    rec("E4_prep", n_zeroed > 0 and b_abs_zero == 0.0 and b_abs_trained > 0.0,
        {"n_zeroed": n_zeroed, "control_lora_B_abs_sum": b_abs_zero,
         "trained_lora_B_abs_sum": round(b_abs_trained, 4)},
        "control adapter lora_B is exactly zero while the trained one is not")
    report["layout_path"] = str(LAYOUT)
    save_report()
except StageSkipped:
    pass
except Exception as exc:  # noqa: BLE001
    fail_soft("C1", exc, "save_reload")

report["timings"]["save_s"] = round(time.time() - T_SAVE, 1)
save_report()
'''

CELL_7 = """## 7. Stage 3: serve with the official stack, training model fully released

Stage separation is not cosmetic. The decompressed training model occupies
~15.2 GiB per L4, and the quantized serving copy needs ~4.5 GiB per L4 plus KV
cache; both resident at once does not fit in 22 GiB, and the failure mode is an
OOM inside vLLM rather than a clean error. Every reference that keeps the trained
graph alive is therefore dropped, and D4 refuses to start the server unless the
measured free memory is actually sufficient.

**`--enforce-eager` differs from the scoring configuration.** The official
`setup_vllm_server` (`data/official/HARNESS_README.md:163-169`) uses
`gpu_memory_utilization=0.80`, `max_model_len=32768`, `tensor_parallel_size=4`,
`max_loras=8`, `max_lora_rank=128`, and does **not** pass `--enforce-eager`.
Graph capture is disabled here because it allocates well beyond the static
weights on a 22 GiB L4 (Gate 1 died at 58% of capture with 3.94 MiB free). That
is irrelevant to what this gate measures — adapter plumbing and logprobs — but it
means a pass here proves the adapter path works **in eager mode at reduced
context**, not under the exact scoring launch flags.
"""

CELL_7_CODE = r'''import gc, json, time

import torch

T_VLLM = time.time()
server = None
OFFICIAL_SERVE = {
    "source": "data/official/HARNESS_README.md:163-169",
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.80,
    "max_model_len": 32768,
    "enable_auto_tool_choice": True,
    "tool_call_parser": "gemma4",
    "reasoning_parser": "gemma4",
    "enable_lora": True,
    "max_loras": 8,
    "max_lora_rank": 128,
    "enforce_eager": False,
}
try:
    _stage_guard("vllm_start")

    report["elapsed_before_vllm_s"] = round(time.time() - T0, 1)
    if time.time() > SOFT_DEADLINE:
        raise RuntimeError(
            "soft deadline exceeded before vLLM start (elapsed %.0fs)"
            % (time.time() - T0))

    # Drop every reference that keeps the training graph alive on the GPUs.
    # `del model` alone is not enough: the optimiser holds AdamW moments for each
    # LoRA parameter, and the bookkeeping dicts above still point at CUDA tensors.
    for _name in ("optim", "model", "lora_params", "trainable", "before", "after",
                  "grads", "reloaded", "mem", "on_disk", "zeroed", "zeroed_dict",
                  "steps", "out", "logits", "loss", "targets", "ids", "long_ids",
                  "per_tensor", "modules", "lora_modules", "zd"):
        try:
            del globals()[_name]
        except KeyError:
            pass
    gc.collect()
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    time.sleep(5)

    free_gib = {}
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gib[str(i)] = {"free": round(free_b / 1024**3, 2),
                            "total": round(total_b / 1024**3, 2)}
    report["gpu_free_before_vllm"] = free_gib
    report["nvidia_smi_after_unload"] = nvidia_smi()
    print("free memory before vLLM:", json.dumps(free_gib), flush=True)
    print("\n".join(report["nvidia_smi_after_unload"]), flush=True)

    min_free = min(v["free"] for v in free_gib.values()) if free_gib else 0.0
    # The quantized serving copy needs ~4.5 GiB per L4 for weights (the W4A16
    # checkpoint is 23.3 GB across 4 cards). The rest of the util budget is KV
    # cache, so 9 GiB free is enough for weights plus a useful cache. The
    # threshold is deliberately not higher: a stricter bound would refuse to
    # serve on a run that is actually fine, and losing the whole E group to a
    # conservative guess is a worse outcome than a slightly smaller KV cache.
    d4_ok = rec("D4", min_free >= 9.0,
                {"min_free_gib": min_free, "per_device": free_gib,
                 "footprint_before_unload_gib": report.get("model_footprint_gib")},
                "training model released enough memory for the serving copy")

    util = 0.80
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        util = min(util, (free_b - 2.0 * 1024**3) / float(total_b))
    util = max(0.10, round(util, 2))
    report["vllm_gpu_memory_utilization"] = util

    if not d4_ok:
        raise RuntimeError(
            "only %.2f GiB free on the tightest device; refusing to start vLLM "
            "(it would OOM inside weight loading, losing the evidence)" % min_free)

    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    adapters = discover_adapters(str(LAYOUT), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    report["discovered_adapters"] = {
        k: str(getattr(v, "path", v)) for k, v in adapters.adapters.items()
    }
    print("adapters:", json.dumps(report["discovered_adapters"], indent=2), flush=True)

    tp = 4 if report["gpu"]["count"] >= 4 else report["gpu"]["count"]

    # Start-ladder. The official 32768 context is tried first so the pass says as
    # much as possible about the scoring configuration; a smaller window is only
    # used if vLLM itself rejects the larger one (KV cache would not fit). Losing
    # the entire E group to one unlucky context length is not acceptable, but
    # neither is quietly defaulting to a small window, so every attempt is
    # recorded and the achieved value is reported.
    attempts = []
    lows = []
    if util >= 0.55:
        lows.append(32768)
    if util >= 0.35:
        lows.append(16384)
    if util >= 0.20:
        lows.append(8192)
    lows.append(4096)
    seen = []
    for candidate in lows:
        if candidate not in seen:
            seen.append(candidate)

    for attempt_len in seen:
        now = time.time()
        if now > SOFT_DEADLINE:
            attempts.append({"max_model_len": attempt_len,
                             "skipped": "soft deadline exceeded before this attempt"})
            break
        # `_wait_until_ready` polls process.poll() each interval, so a genuine
        # KV-cache rejection surfaces in seconds. The timeout therefore only has to
        # cover a real hang; bounding it per attempt keeps the whole ladder inside
        # the kernel cap instead of spending 4 x 2400 s on a dead server.
        startup_timeout = max(300, min(1500, int(SERVER_DEADLINE - now)))
        if SERVER_DEADLINE - now <= 0:
            attempts.append({"max_model_len": attempt_len,
                             "skipped": "server deadline reached before this attempt"})
            break
        vcfg = VllmConfig(
            model=str(MODEL),
            port=8000,
            host="127.0.0.1",
            tool_call_parser="gemma4",
            reasoning_parser="gemma4",
            max_model_len=attempt_len,
            dtype="bfloat16",
            gpu_memory_utilization=util,
            enable_auto_tool_choice=True,
            enable_lora=True,
            max_loras=8,
            max_lora_rank=128,
            tensor_parallel_size=tp,
            startup_timeout=startup_timeout,
            extra_args=["--enforce-eager"],
        )
        cand_server = VllmServer(vcfg, adapter_manifest=adapters)
        cmd = cand_server.build_cmd()
        print("vLLM attempt max_model_len=%d timeout=%ds" % (attempt_len, startup_timeout),
              flush=True)
        print("vLLM cmd:", " ".join(cmd), flush=True)
        try:
            cand_server.start()
            server = cand_server
            report["vllm_cmd"] = cmd
            report["vllm_max_model_len"] = attempt_len
            report["vllm_startup_timeout"] = startup_timeout
            attempts.append({"max_model_len": attempt_len, "ok": True,
                             "startup_timeout": startup_timeout})
            break
        except Exception as exc:  # noqa: BLE001
            msg = "%s: %s" % (type(exc).__name__, exc)
            attempts.append({"max_model_len": attempt_len, "ok": False,
                             "error": msg, "startup_timeout": startup_timeout})
            print("vLLM failed at max_model_len=%d: %s" % (attempt_len, msg), flush=True)
            try:
                cand_server.stop()
            except Exception:  # noqa: BLE001
                pass
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)

    report["vllm_start_attempts"] = attempts
    if server is None:
        raise RuntimeError(
            "vLLM failed to start at every attempted context length: %s"
            % json.dumps(attempts))

    max_model_len = report["vllm_max_model_len"]
    report["serve_config"] = {
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": util,
        "max_model_len": max_model_len,
        "enable_auto_tool_choice": True,
        "tool_call_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "enable_lora": True,
        "max_loras": 8,
        "max_lora_rank": 128,
        "enforce_eager": True,
    }
    differing = {k: {"official": OFFICIAL_SERVE[k], "ours": v}
                 for k, v in report["serve_config"].items()
                 if k in OFFICIAL_SERVE and OFFICIAL_SERVE[k] != v}
    report["serve_config_comparison"] = {
        "official": OFFICIAL_SERVE,
        "ours": report["serve_config"],
        "differing_keys": differing,
        "note": ("A pass proves the adapter path works in eager mode at %d context, "
                 "not under the exact scoring launch flags (max_model_len=32768, "
                 "gpu_memory_utilization=0.80, CUDA graphs enabled)." % max_model_len),
    }
    print("serve config differences:", json.dumps(differing, indent=2), flush=True)

    report["server_started"] = True
    report["base_url"] = server.base_url
    report["log_path"] = server.log_path
    report["nvidia_smi_after_start"] = nvidia_smi()
    print("server up in %.1fs -> %s" % (time.time() - T_VLLM, server.base_url), flush=True)
    print("\n".join(report["nvidia_smi_after_start"]), flush=True)
except StageSkipped:
    pass
except Exception as exc:  # noqa: BLE001
    fail_soft("E7", exc, "vllm_start")

report["timings"]["vllm_start_s"] = round(time.time() - T_VLLM, 1)
save_report()
'''

CELL_8 = """## 8. Does the serving stack consume the adapter?

The decisive measurement is the **prompt logprob**, not generated text:

- **E2** noise floor — the same model queried three times must return identical
  logprobs; this bounds every other comparison.
- **E3** trained adapter vs base — must move logprobs by at least `1e-3` and by
  more than ten times the noise floor.
- **E4** zero-`lora_B` control vs base — must **not** move logprobs.
- **E9** repeats the E3 comparison against a second base reference, so a pass
  cannot rest on one lucky call.

E3 without E4 is worthless: any difference could come from something other than
the adapter. E4 without E3 is equally worthless: it cannot distinguish "correctly
inert" from "every adapter is silently ignored".

`prompt_logprobs=0` scores every prompt token with no sampling, which is far more
sensitive than comparing a handful of generated tokens. **E6** then scans the
full server log for the strings vLLM emits when it drops LoRA weights.
"""

CELL_8_CODE = r'''import json, time

import requests

T_E = time.time()
# adk_submission's server.base_url already ends in "/v1", so paths are relative.
try:
    BASE = server.base_url if server is not None else None
except Exception:  # noqa: BLE001
    BASE = None
report["api_base"] = BASE

PROMPT_TEXT = __PROMPT_JSON__


def api(path, payload, timeout=600):
    url = BASE.rstrip("/") + path
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


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
    if lp is None:
        return None
    out = []
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
    if a is None or b is None:
        return None, 0
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    return max(abs(x - y) for x, y in pairs), len(pairs)


E_IDS = ["E1", "E2", "E3", "E4", "E6", "E7", "E8", "E9"]
if BASE is None:
    if report.get("load_failed") or report.get("train_failed"):
        # A blocked run must not report these as failures: "the check ran and the
        # numbers were wrong" and "the check never ran" are different findings, and
        # collapsing them is how v1's cascade got read as eight independent failures.
        _reason = report.get("blocked_by") or "an upstream stage failed"
        for _cid in E_IDS:
            if _cid not in report["checks"]:
                report["checks"][_cid] = {
                    "ok": False, "value": None,
                    "detail": "skipped: adapter comparison not reached because %s"
                              % _reason}
        if "e_checks" not in report["skipped_stages"]:
            report["skipped_stages"].append("e_checks")
        print("SKIP e_checks :: %s" % _reason, flush=True)
    else:
        # The server did not come up. That is one finding, recorded against the
        # serving stage (E7). These comparisons need a live endpoint and never ran,
        # so they are skipped rather than failed -- the verdict is still FAIL,
        # because a skipped check is not a passing one.
        _reason = "serving did not start; see E7 and vllm_start_attempts"
        for _cid in E_IDS:
            if _cid not in report["checks"]:
                report["checks"][_cid] = {
                    "ok": False, "value": None,
                    "detail": "skipped: %s" % _reason}
        if "e_checks" not in report["skipped_stages"]:
            report["skipped_stages"].append("e_checks")
        print("SKIP e_checks :: %s" % _reason, flush=True)
else:
    # ---- E1: model listing --------------------------------------------------
    try:
        listing = requests.get(BASE.rstrip("/") + "/models", timeout=180).json()
        ids = [m["id"] for m in listing.get("data", [])]
        report["served_models"] = ids
        wanted = {"trained_gate2", "zero_b_control",
                  "adapter:trained_gate2", "adapter:zero_b_control"}
        hit = sorted(wanted & set(ids))
        base_present = any(not i.endswith(("_gate2", "_control"))
                           and "adapter:" not in i for i in ids)
        rec("E1", len(hit) >= 2 and base_present,
            {"ids": ids, "adapter_aliases_found": hit},
            "base + both adapter aliases listed")
    except Exception as exc:  # noqa: BLE001
        fail_soft("E1", exc, "models_list")

    try:
        base_name = None
        for candidate in report.get("served_models", []):
            if candidate.endswith(("_gate2", "_control")) or "adapter:" in candidate:
                continue
            base_name = candidate
            break
        report["base_model_alias"] = base_name
        if base_name is None:
            raise RuntimeError("no base model alias in %r" % report.get("served_models"))

        # ---- E2: noise floor from three identical calls ----------------------
        base_a = prompt_logprobs(base_name)
        base_b = prompt_logprobs(base_name)
        base_c = prompt_logprobs(base_name)
        n1, npos1 = compare(base_a, base_b)
        n2, npos2 = compare(base_a, base_c)
        candidates = [v for v in (n1, n2) if v is not None]
        noise = max(candidates) if candidates else None
        report["E2_positions"] = max(npos1, npos2)
        report["E2_noise_floor"] = noise
        report["E2_pairwise"] = {"ab": n1, "ac": n2}
        rec("E2", noise is not None,
            {"noise": noise, "positions": report["E2_positions"],
             "pairwise": report["E2_pairwise"]},
            "same model, three calls: measurement noise floor")

        # ---- E3 / E4 / E9 ----------------------------------------------------
        trained = prompt_logprobs("trained_gate2")
        zero_b = prompt_logprobs("zero_b_control")
        d_trained, n_t = compare(base_a, trained)
        d_trained2, _ = compare(base_c, trained)
        d_zero, n_z = compare(base_a, zero_b)
        report["E3_delta"] = d_trained
        report["E3_delta_second_reference"] = d_trained2
        report["E3_positions"] = n_t
        report["E4_delta"] = d_zero
        report["E4_positions"] = n_z

        eff_noise = noise if (noise is not None and noise > 0.0) else 0.0
        e3_ok = d_trained is not None and d_trained >= 1e-3 and (
            eff_noise == 0.0 or d_trained > 10 * eff_noise)
        rec("E3", e3_ok, {"delta": d_trained, "positions": n_t, "noise": noise},
            "trained adapter must move logprobs")

        e4_ok = d_zero is not None and (
            d_zero <= eff_noise if eff_noise > 0 else d_zero == 0.0)
        rec("E4", e4_ok, {"delta": d_zero, "positions": n_z, "noise": noise},
            "zero-B control must stay within the noise floor")

        stable = (d_trained is not None and d_trained2 is not None
                  and d_trained2 >= 1e-3
                  and abs(d_trained - d_trained2) <= 0.05 * max(d_trained, d_trained2))
        rec("E9", stable,
            {"delta_ref_a": d_trained, "delta_ref_c": d_trained2,
             "relative_gap": (round(abs(d_trained - d_trained2)
                                    / max(d_trained, d_trained2), 6)
                              if d_trained and d_trained2 else None)},
            "trained-adapter delta reproducible against two base references")

        # ---- E7: generation with logprobs -----------------------------------
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
        report["E7_text"] = (choice.get("text") or "")[:200]
        gl = choice.get("logprobs") or {}
        report["E7_token_count"] = len(gl.get("tokens") or [])
        rec("E7", report["E7_token_count"] > 0,
            {"tokens": report["E7_token_count"], "text": report["E7_text"][:80]},
            "generation produced tokens")

        trained_gen = api("/completions", dict(gen_body, model="trained_gate2"))
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
        save_report()
    except Exception as exc:  # noqa: BLE001
        for _cid in ("E2", "E3", "E4", "E7", "E9"):
            if _cid not in report["checks"]:
                report["checks"][_cid] = {"ok": False, "value": None,
                                          "detail": "skipped after error: %s" % exc}
        fail_soft("E3", exc, "logprob_compare")

    # ---- E8: tool calling through the auto tool-choice path -----------------
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
                "max_tokens": 512,
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
            e8_log.append({"attempt": idx, "error": "%s: %s" % (type(exc).__name__, exc)})
    report["E8_attempts"] = e8_log
    rec("E8", e8_ok, e8_log[-1] if e8_log else None, "tool call parsed at least once")

    # ---- E6: no silent-drop warnings in the server log ----------------------
    try:
        log_text = Path(server.log_path).read_text(errors="replace")
        (WORK / "gate2_vllm_server.log").write_text(log_text)
        forbidden = [
            "is not in the model's supported LoRA target modules",
            "These parameters will be ignored",
            "does not support LoRA yet",
        ]
        found = [f for f in forbidden if f in log_text]
        report["E6_found"] = found
        report["vllm_log_bytes"] = len(log_text)
        report["E6_lora_mentions"] = log_text.lower().count("lora")
        rec("E6", len(found) == 0,
            {"forbidden_found": found, "lora_mentions": report["E6_lora_mentions"],
             "log_bytes": report["vllm_log_bytes"]},
            "no silent-drop / unsupported warnings")
    except Exception as exc:  # noqa: BLE001
        fail_soft("E6", exc, "vllm_log_scan")

report["timings"]["e_checks_s"] = round(time.time() - T_E, 1)
save_report()
'''

CELL_9 = """## 9. Finalise

Stop the server, capture the closing GPU state, and write the machine-readable
verdict. `D1` and `GC1` are recorded but not gated: they measure the setup rather
than the compatibility claim.
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
report["nvidia_smi_after_stop"] = nvidia_smi()

HARD_GROUPS = {
    "A": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
    "B": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B7b"],
    "C": ["C1", "C2", "C3"],
    "D": ["D2", "D3", "D4"],
    "E": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
}
# A skipped check counts against the verdict -- it did not pass -- but it is
# reported apart from a failing one, because "the load died and nothing downstream
# ran" and "the check ran and the numbers were wrong" are different findings. v1
# collapsed the two and its cascade was read as eight independent failures.
skipped_ids = {cid for cid, v in report["checks"].items()
               if isinstance(v.get("detail"), str)
               and v["detail"].startswith("skipped:")}
summary = {}
for group, ids in HARD_GROUPS.items():
    missing = [i for i in ids if i not in report["checks"]]
    failed = [i for i in ids
              if i in report["checks"] and not report["checks"][i]["ok"]
              and i not in skipped_ids]
    skipped = [i for i in ids if i in skipped_ids]
    summary[group] = {"missing": missing, "failed": failed, "skipped": skipped,
                      "ok": (not missing) and (not failed) and (not skipped)}
report["verdict"] = {"groups": summary,
                     "pass": all(v["ok"] for v in summary.values()),
                     "skipped_stages": list(report["skipped_stages"]),
                     "blocked_by": report.get("blocked_by")}
report["finished_epoch"] = time.time()
report["wall_seconds"] = round(report["finished_epoch"] - report["started_epoch"], 1)
save_report()

print()
print("=" * 72)
print("VERDICT:", "PASS" if report["verdict"]["pass"] else "FAIL")
if report["verdict"]["skipped_stages"]:
    print("  blocked at: %s" % (report.get("blocked_by") or "unknown stage"))
    print("  skipped stages: %s" % ", ".join(report["verdict"]["skipped_stages"]))
for g, v in summary.items():
    line = "  %s: %s (missing=%s failed=%s" % (g, "ok" if v["ok"] else "FAIL",
                                              v["missing"], v["failed"])
    if v["skipped"]:
        line += " skipped=%s" % v["skipped"]
    print(line + ")")
print("wall: %.1f s (cap %ds)" % (report["wall_seconds"], report["runtime_cap_seconds"]))
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
    # Interpolate the timing constants from the Python module above rather than
    # repeating literals in the cell strings. v1 shipped a notebook whose hard cap
    # and whose manifest disagreed with the prose in the pre-registration; a single
    # source of truth makes that class of drift impossible to introduce silently.
    cell1_code = (CELL_1_CODE
                  .replace("__RUNTIME_CAP_SECONDS__", str(RUNTIME_CAP_SECONDS))
                  .replace("__SOFT_DEADLINE_SECONDS__", str(SOFT_DEADLINE_SECONDS))
                  .replace("__SERVER_RESERVE__", str(SERVER_RESERVE_SECONDS)))
    # CELL_3 carries the load path; the GPU reserve is a single knob shared by the
    # placement plan and by the manifest, so both are interpolated from one literal.
    cell3_code = CELL_3_CODE.replace("__GPU_RESERVE_GIB__", repr(GPU_RESERVE_GIB))
    cells = [
        md_cell(CELL_0),
        md_cell(CELL_1), code_cell(cell1_code),
        md_cell(CELL_2), code_cell(CELL_2_CODE),
        md_cell(CELL_3), code_cell(cell3_code),
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
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nb_path = DEST / "smoke.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
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
        "model_sources": [BASE_REF],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4",
    }
    (DEST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    experiment = {
        "suite": "gate2",
        "kernel": KERNEL_ID,
        "purpose": "31B training/inference compatibility probe (not SFT, not RL)",
        "base_model": BASE_REF,
        "base_model_path_in_kernel": BASE_PATH_IN_KERNEL,
        "runtime_cap_seconds": RUNTIME_CAP_SECONDS,
        "soft_deadline_seconds": SOFT_DEADLINE_SECONDS,
        "server_reserve_seconds": SERVER_RESERVE_SECONDS,
        "server_deadline_seconds": RUNTIME_CAP_SECONDS - SERVER_RESERVE_SECONDS,
        "load_route": "cpu_decompress_then_dispatch",
        "placement_method": "accelerate.get_balanced_memory + infer_auto_device_map + dispatch_model",
        "gpu_weight_reserve_gib": GPU_RESERVE_GIB,
        "target_module_regex": TARGET_MODULE_REGEX,
        "lora": {"r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": 0.0,
                 "bias": "none", "use_dora": False, "use_rslora": False,
                 "modules_to_save": None},
        "training": {"steps": N_STEPS, "seq_len": SEQ_LEN,
                     "long_seq_len": LONG_SEQ_LEN, "batch": 1, "lr": LR,
                     "optimizer": "AdamW", "seed": SEED,
                     "gradient_checkpointing": True,
                     "data": "synthetic random token ids (no dataset)"},
        "expected": {
            "n_matched_modules": EXPECTED_MODULES,
            "n_trainable_params": EXPECTED_TRAINABLE,
            "n_adapter_tensors": EXPECTED_TENSORS,
            "sliding_per_layer": SLIDING_PER_LAYER,
            "full_per_layer": FULL_PER_LAYER,
            "full_attention_indices": [5, 11, 17, 23, 29, 35, 41, 47, 53, 59],
        },
        "official_serve_config": OFFICIAL_SERVE,
        "hard_checks": {
            "A": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
            "B": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B7b"],
            "C": ["C1", "C2", "C3"],
            "D": ["D2", "D3", "D4"],
            "E": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
        },
        "a9_tied_embeddings": {
            "what": "lm_head.weight and embed_tokens.weight are one storage, on one "
                    "device, and accelerate's own tie detection sees the pair",
            "why": "a split tie loads without error and only shows up as wrong "
                   "logits, so it cannot be left as one conjunct inside A4",
            "views": ["object identity", "value equality", "device equality",
                      "accelerate find_tied_parameters"],
        },
        "notebook_sha256": nb_sha,
        "prereg": "artifacts/gate2_prereg.md",
        "not_in_scope": ["full SFT", "RL", "k/v LoRA targets", "submission"],
    }
    (DEST / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("wrote", nb_path, "sha256", nb_sha)
    print("cells:", len(nb["cells"]))
    print("wrote", DEST / "kernel-metadata.json")
    print("wrote", DEST / "experiment.json")


if __name__ == "__main__":
    main()
