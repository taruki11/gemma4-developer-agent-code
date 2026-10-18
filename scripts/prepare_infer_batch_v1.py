"""Generate the Gate 2 inference-verification batch (v1).

This batch is deliberately separate from the training runs. It answers one question:
does the official scoring stack, started with the official parameters, actually consume
the two adapters that the v3 training run saved? It never loads the Hugging Face
training model, so the 60 GiB that blocked v3's serving stage simply is not there.

Judgement scope is inference compatibility only. A pass here says the adapter is real,
loadable and numerically active in vLLM; it says nothing about whether the agent is
better at fixing code.
"""

import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts" / "infer_batch_v1"

KERNEL_ID = "taruki/gemma4-infer-batch-v1"
KERNEL_TITLE = "Gemma4 Infer Batch v1"

# A serving-only run: install, hash, start, compare. The hard cap bounds quota spend
# even if a request hangs; the soft deadline leaves room to write the report.
RUNTIME_CAP_SECONDS = 3600
SOFT_DEADLINE_SECONDS = 2400
SERVER_START_TIMEOUT = 1500
REQUEST_TIMEOUT = 240

BASE_SLUG = "gemma-4-31b-it-qat-w4a16-ct"
BASE_REF = "google/gemma-4/Other/" + BASE_SLUG + "/2"
ADAPTER_DATASET = "taruki/gemma4-infer-batch-v1-adapters"

# The official scoring configuration, and the only configuration this batch tries.
# Source: data/official/HARNESS_README.md:163-169 (setup_vllm_server). enforce_eager is
# absent there, so it is absent here.
OFFICIAL_SERVE = {
    "source": "data/official/HARNESS_README.md:163-169",
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.80,
    "max_model_len": 32768,
    "enable_auto_tool_choice": True,
    "tool_call_parser": "gemma4",
    "reasoning_parser": "gemma4",
    # The official configuration sets this; VllmConfig defaults to None, which
    # would omit --default-chat-template-kwargs entirely and leave thinking off.
    "default_chat_template_kwargs": {"enable_thinking": True},
    "enable_lora": True,
    "max_loras": 8,
    "max_lora_rank": 128,
    "extra_args": [],
    "enforce_eager": False,
}

PROMPT = '''def add(a: int, b: int) -> int:
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
'''

# Strings vLLM emits when it decides some adapter tensors cannot be applied. Any of
# them means the adapter was partially or wholly dropped, which would make every
# logprob comparison below meaningless.
FORBIDDEN_WARNINGS = [
    "is not in the model's supported LoRA target modules",
    "These parameters will be ignored",
    "does not support LoRA yet",
]
# Positive evidence that an adapter was actually mounted.
LOADED_MARKER = "Loaded new LoRA adapter"


# --------------------------------------------------------------------------- cells
CELL_0 = """# Gate 2 inference batch v1

Two adapters, produced by the v3 training run, are loaded into the official vLLM
serving stack and measured against the base model.

- **Base model:** `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`
- **Adapters:** `trained_gate2` (240 tensors, 120 with nonzero `lora_B`) and
  `zero_b_control` (identical config, every `lora_B` zeroed)
- **Serving:** the official scoring parameters, **no** `--enforce-eager`
- **Scope:** inference compatibility only. Nothing here measures agent quality.

The Hugging Face training model is never loaded in this notebook; only the quantized
checkpoint is, and only by vLLM's own subprocess.
"""

CELL_1 = """## 1. Preflight

Locate the two inputs (the competition checkpoint and the adapter dataset) and set up
the report. Nothing is downloaded.
"""

CELL_1_CODE = r'''import json, os, platform, subprocess, sys, time
from pathlib import Path

T0 = time.time()
SOFT_DEADLINE = T0 + __SOFT_DEADLINE_SECONDS__
RUNTIME_CAP = __RUNTIME_CAP_SECONDS__
WORK = Path("/kaggle/working")
REPORT_PATH = WORK / "infer_batch_report.json"

report = {
    "batch": "gate2_infer_batch_v1",
    "kernel": __KERNEL_ID__,
    "started_epoch": time.time(),
    "runtime_cap_seconds": RUNTIME_CAP,
    "soft_deadline_seconds": __SOFT_DEADLINE_SECONDS__,
    "checks": {},
    "notes": [],
    "errors": [],
    "versions": {},
    "gpu": {},
    "host": {},
    "timings": {},
    "scope": ("inference compatibility only: does the official serving stack consume "
              "the v3 adapters? this does not measure code-fixing ability"),
}


def rec(cid, ok, value=None, detail=""):
    report["checks"][cid] = {"ok": bool(ok), "value": value, "detail": detail}
    print("%s %s :: %s" % ("PASS" if ok else "FAIL", cid, detail), flush=True)
    return bool(ok)


def note(msg):
    report["notes"].append(msg)
    print("NOTE %s" % msg, flush=True)


def fail_soft(cid, exc, where):
    import traceback
    report["checks"][cid] = {
        "ok": False, "value": None,
        "detail": "%s: %s" % (type(exc).__name__, exc),
    }
    report["errors"].append({"where": where,
                             "error": "%s: %s" % (type(exc).__name__, exc)})
    print("FAIL %s :: %s: %s" % (cid, type(exc).__name__, exc), flush=True)
    traceback.print_exc()


def save_report():
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print("wrote %s (%d bytes)" % (REPORT_PATH.name, REPORT_PATH.stat().st_size),
          flush=True)


def nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
        return [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return []


# ---- inputs -----------------------------------------------------------------
MODEL_ROOT = Path("/kaggle/input/models/google/gemma-4/other")
MODEL = None
try:
    cands = sorted(p for p in MODEL_ROOT.glob(__BASE_SLUG__ + "/*") if p.is_dir())
    if cands:
        MODEL = cands[-1]
except Exception:  # noqa: BLE001
    MODEL = None
if MODEL is None or not (MODEL / "config.json").is_file():
    hits = sorted(Path("/kaggle/input/models").rglob(__BASE_SLUG__ + "/config.json"))
    MODEL = hits[-1].parent if hits else None

ADAPTER_INPUT = None
for cand in sorted(Path("/kaggle/input").rglob("adapter_manifest.json")):
    parent = cand.parent
    if (parent / "adapters" / "trained_gate2" / "adapter_model.safetensors").is_file():
        ADAPTER_INPUT = parent
        break

report["paths"] = {
    "model": str(MODEL) if MODEL else None,
    "adapter_dataset": str(ADAPTER_INPUT) if ADAPTER_INPUT else None,
    "layout": str(WORK / "infer_layout"),
}
report["base_model_ref"] = __BASE_REF__
report["adapter_dataset_ref"] = __ADAPTER_DATASET__

rec("PRE_MODEL", MODEL is not None and (MODEL / "config.json").is_file(),
    report["paths"]["model"], "competition 31B checkpoint mounted")
rec("PRE_ADAPTERS", ADAPTER_INPUT is not None, report["paths"]["adapter_dataset"],
    "adapter dataset mounted")

# ---- environment ------------------------------------------------------------
try:
    import torch
    report["gpu"] = {
        "count": torch.cuda.device_count(),
        "names": [torch.cuda.get_device_name(i)
                  for i in range(torch.cuda.device_count())],
        "total_memory_gib": [round(torch.cuda.get_device_properties(i).total_memory
                                   / 1024 ** 3, 2)
                             for i in range(torch.cuda.device_count())],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_smi": nvidia_smi(),
    }
except Exception as exc:  # noqa: BLE001
    fail_soft("PRE_GPU", exc, "preflight_gpu")

report["host"] = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
}
try:
    with open("/proc/meminfo") as fh:
        report["host"]["ram_gib"] = round(
            int(fh.readlines()[1].split()[1]) / 1024 ** 2, 2)
except Exception:  # noqa: BLE001
    pass

# The model file list is recorded so the run's own inputs are auditable.
if MODEL is not None:
    report["model_files"] = sorted(p.name for p in MODEL.iterdir() if p.is_file())

note("torch %s / cuda %s / %d GPU(s)"
     % (report["gpu"].get("torch"), report["gpu"].get("cuda"),
        report["gpu"].get("count", 0)))
print(json.dumps(report["gpu"], indent=2, default=str), flush=True)
save_report()
'''

CELL_2 = """## 2. Install the serving stack

The same wheelhouse and pin set that the training gates used, so the versions under
test match the ones the earlier evidence came from.
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
        target_name = (w.name.replace("cu128", "+cu128")
                       if ("cu128" in w.name and "+" not in w.name) else w.name)
        target = tmp_whl / target_name
        if not target.exists():
            os.symlink(w, target)
    wheels = sorted(str(w) for w in tmp_whl.glob("*.whl"))
    print("installing %d wheelhouse wheels ..." % len(wheels), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--force-reinstall"] + wheels, check=True)

    print("installing pinned support wheels ...", flush=True)
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
    print(json.dumps(report["versions"], indent=2), flush=True)
    report["timings"]["install_s"] = round(time.time() - T0, 1)
except Exception as exc:  # noqa: BLE001
    fail_soft("PRE_INSTALL", exc, "install")

save_report()
'''

CELL_3 = """## 3. Adapter integrity, before and after the mount

Three hash sets must agree: the digests recorded when the dataset was assembled, the
digests of the files as mounted, and the digests after copying into the working
directory that vLLM will read. An upload that silently rewrote a safetensors file would
invalidate every comparison below without raising anything, so it is checked explicitly
rather than assumed.
"""

CELL_3_CODE = r'''import hashlib, json, shutil, time
from pathlib import Path

T_A = time.time()
EXPECTED = json.loads(__EXPECTED_MANIFEST_JSON__)
LABELS = ("trained_gate2", "zero_b_control")
FILES = ("adapter_config.json", "adapter_model.safetensors", "README.md")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- A1: what was mounted is what was assembled -----------------------------
mounted = {}
try:
    for label in LABELS:
        entry = {}
        for name in FILES:
            p = ADAPTER_INPUT / "adapters" / label / name
            if p.is_file():
                entry[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        mounted[label] = entry
    report["mounted_hashes"] = mounted
    mismatches = []
    for label in LABELS:
        for name, want in EXPECTED["adapters"][label].items():
            got = mounted.get(label, {}).get(name)
            if got is None:
                mismatches.append("%s/%s missing" % (label, name))
            elif got["sha256"] != want["sha256"]:
                mismatches.append("%s/%s sha differs (%s vs %s)"
                                  % (label, name, got["sha256"][:16],
                                     want["sha256"][:16]))
            elif got["bytes"] != want["bytes"]:
                mismatches.append("%s/%s size differs (%d vs %d)"
                                  % (label, name, got["bytes"], want["bytes"]))
    rec("A1", not mismatches, {"mismatches": mismatches,
                               "mounted": mounted},
        "mounted adapter files match the digests recorded at assembly")
except Exception as exc:  # noqa: BLE001
    fail_soft("A1", exc, "adapter_hash_mounted")

# ---- copy into the working layout vLLM will read ----------------------------
LAYOUT = WORK / "infer_layout"
try:
    if LAYOUT.exists():
        shutil.rmtree(LAYOUT)
    (LAYOUT / "adapters").mkdir(parents=True)
    for label in LABELS:
        shutil.copytree(ADAPTER_INPUT / "adapters" / label,
                        LAYOUT / "adapters" / label)
    report["layout_path"] = str(LAYOUT)
    report["notes"].append("copied %d adapter directories into %s"
                           % (len(LABELS), LAYOUT))
except Exception as exc:  # noqa: BLE001
    fail_soft("A2", exc, "adapter_copy")

# ---- A2: the copy is bit-identical -----------------------------------------
try:
    copied = {}
    for label in LABELS:
        entry = {}
        for name in FILES:
            p = LAYOUT / "adapters" / label / name
            if p.is_file():
                entry[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        copied[label] = entry
    report["layout_hashes"] = copied
    drift = []
    for label in LABELS:
        for name in FILES:
            a = mounted.get(label, {}).get(name, {}).get("sha256")
            b = copied.get(label, {}).get(name, {}).get("sha256")
            if a != b:
                drift.append("%s/%s mounted=%s copied=%s"
                             % (label, name, str(a)[:16], str(b)[:16]))
    rec("A2", not drift, {"drift": drift, "layout": copied},
        "the working copy is bit-identical to the mounted files")
except Exception as exc:  # noqa: BLE001
    fail_soft("A2", exc, "adapter_rehash")

# ---- A3: the configs describe the adapters we think they do ----------------
try:
    cfg_t = json.loads((LAYOUT / "adapters" / "trained_gate2" /
                        "adapter_config.json").read_text())
    cfg_z = json.loads((LAYOUT / "adapters" / "zero_b_control" /
                        "adapter_config.json").read_text())
    facts = {
        "r": cfg_t.get("r"),
        "lora_alpha": cfg_t.get("lora_alpha"),
        "target_modules": cfg_t.get("target_modules"),
        "peft_type": cfg_t.get("peft_type"),
        "task_type": cfg_t.get("task_type"),
        "use_dora": cfg_t.get("use_dora"),
        "base_model_name_or_path": cfg_t.get("base_model_name_or_path"),
        "configs_identical": cfg_t == cfg_z,
    }
    report["adapter_config"] = facts
    ok = (facts["r"] == 8
          and facts["lora_alpha"] == 16
          and facts["peft_type"] == "LORA"
          and facts["target_modules"]
          == r".*\.language_model\.layers\.\d+\.self_attn\.(q|o)_proj$"
          and facts["configs_identical"] is True
          and __BASE_SLUG__ in str(facts["base_model_name_or_path"]))
    rec("A3", ok, facts,
        "both configs declare r=8, q/o targets, and the competition base")
except Exception as exc:  # noqa: BLE001
    fail_soft("A3", exc, "adapter_config")

# ---- A4: trained has energy in lora_B, the control has none ----------------
try:
    import struct

    def header_of(path):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n).decode("utf-8"))
        head.pop("__metadata__", None)
        return head, 8 + n

    def lora_b_stats(path):
        head, data_start = header_of(path)
        keys = [k for k in head if ".lora_B." in k]
        nonzero = 0
        for k in keys:
            lo, hi = head[k]["data_offsets"]
            with open(path, "rb") as fh:
                fh.seek(data_start + lo)
                blob = fh.read(hi - lo)
            if any(blob):
                nonzero += 1
        # lora_A keys must also be present; a file with only lora_B would load as
        # zeros and look identical to the control.
        n_a = len([k for k in head if ".lora_A." in k])
        return {"n_lora_B": len(keys), "n_lora_B_nonzero": nonzero, "n_lora_A": n_a,
                "n_keys": len(head)}

    st_t = lora_b_stats(LAYOUT / "adapters" / "trained_gate2" /
                        "adapter_model.safetensors")
    st_z = lora_b_stats(LAYOUT / "adapters" / "zero_b_control" /
                        "adapter_model.safetensors")
    report["lora_b_stats"] = {"trained": st_t, "zero_b": st_z}
    ok = (st_t["n_lora_A"] == 120 and st_t["n_lora_B"] == 120
          and st_z["n_lora_A"] == 120 and st_z["n_lora_B"] == 120
          and st_t["n_lora_B_nonzero"] == 120
          and st_z["n_lora_B_nonzero"] == 0)
    rec("A4", ok, report["lora_b_stats"],
        "trained adapter has 120/120 nonzero lora_B; control has 0/120")
except Exception as exc:  # noqa: BLE001
    fail_soft("A4", exc, "adapter_contents")

report["timings"]["adapter_check_s"] = round(time.time() - T_A, 1)
save_report()
'''

CELL_4 = """## 4. Start the official serving stack

One attempt, with the official parameters and **no** `--enforce-eager`. This is the
point of the batch: the earlier gates added `--enforce-eager` to dodge a CUDA-graph OOM
caused by the training model still holding memory. Here nothing else holds the GPUs, so
the scoring configuration can be tested as it actually runs.

If the server does not start, the log is saved and the run stops — there is no
degradation ladder and no second attempt.
"""

CELL_4_CODE = r'''import json, time
from pathlib import Path

T_S = time.time()
server = None
adapters = None

OFFICIAL = json.loads(__OFFICIAL_SERVE_JSON__)
report["official_serve_config"] = OFFICIAL

try:
    from adk_submission import VllmConfig, VllmServer, discover_adapters

    adapters = discover_adapters(str(LAYOUT))
    found = {name: {"path": str(info.path), "bytes": int(info.size_bytes),
                    "rank": int(info.rank) if info.rank is not None else None}
             for name, info in adapters.adapters.items()}
    report["discovered_adapters"] = found
    rec("B0", set(found) == {"trained_gate2", "zero_b_control"}, found,
        "both adapters discovered by the official adapter discovery")

    startup_budget = max(300, min(__SERVER_START_TIMEOUT__,
                                  int(SOFT_DEADLINE - time.time())))
    vcfg = VllmConfig(
        model=str(MODEL),
        host="127.0.0.1",
        port=8000,
        tool_call_parser=OFFICIAL["tool_call_parser"],
        reasoning_parser=OFFICIAL["reasoning_parser"],
        default_chat_template_kwargs=OFFICIAL["default_chat_template_kwargs"],
        max_model_len=OFFICIAL["max_model_len"],
        dtype="bfloat16",
        gpu_memory_utilization=OFFICIAL["gpu_memory_utilization"],
        enable_auto_tool_choice=OFFICIAL["enable_auto_tool_choice"],
        enable_lora=OFFICIAL["enable_lora"],
        max_loras=OFFICIAL["max_loras"],
        max_lora_rank=OFFICIAL["max_lora_rank"],
        tensor_parallel_size=OFFICIAL["tensor_parallel_size"],
        startup_timeout=startup_budget,
        extra_args=list(OFFICIAL["extra_args"]),
    )
    server = VllmServer(vcfg, adapter_manifest=adapters)
    cmd = server.build_cmd()
    report["vllm_cmd"] = cmd
    report["vllm_startup_timeout"] = startup_budget
    print("command: %s" % " ".join(cmd), flush=True)

    # ---- B1: the command is the official one ------------------------------
    # The declared max_loras / max_lora_rank are advisory: the server sizes them from
    # the number of mounted adapters and the largest adapter rank. Recording both makes
    # that visible instead of letting "8" in the config imply 8 was used.
    def after(flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    actual = {
        "tensor_parallel_size": after("--tensor-parallel-size"),
        "gpu_memory_utilization": after("--gpu-memory-utilization"),
        "max_model_len": after("--max-model-len"),
        "max_loras": after("--max-loras"),
        "max_lora_rank": after("--max-lora-rank"),
        "enable_lora": "--enable-lora" in cmd,
        "enable_auto_tool_choice": "--enable-auto-tool-choice" in cmd,
        "tool_call_parser": after("--tool-call-parser"),
        "reasoning_parser": after("--reasoning-parser"),
        "default_chat_template_kwargs": after("--default-chat-template-kwargs"),
        "enforce_eager_present": "--enforce-eager" in cmd,
    }
    report["vllm_cmd_actual"] = actual
    report["declared_vs_actual"] = {
        "declared_max_loras": OFFICIAL["max_loras"],
        "actual_max_loras": actual["max_loras"],
        "declared_max_lora_rank": OFFICIAL["max_lora_rank"],
        "actual_max_lora_rank": actual["max_lora_rank"],
        "note": ("vLLM sizes --max-loras from the number of mounted adapters and "
                 "--max-lora-rank from the largest adapter rank, rounded up to a "
                 "supported bucket; the official stack behaves the same way"),
    }
    b1_ok = (actual["tensor_parallel_size"] == "4"
             and abs(float(actual["gpu_memory_utilization"]) - 0.80) < 1e-6
             and actual["max_model_len"] == "32768"
             and actual["enable_lora"] is True
             and actual["enable_auto_tool_choice"] is True
             and actual["tool_call_parser"] == "gemma4"
             and actual["reasoning_parser"] == "gemma4"
             and actual["default_chat_template_kwargs"] is not None
             and "enable_thinking" in actual["default_chat_template_kwargs"]
             and actual["enforce_eager_present"] is False)
    rec("B1", b1_ok, actual,
        "command carries the official parameters and no --enforce-eager")
    if not b1_ok:
        # Starting a command that is not the official one would answer a different
        # question than the one this batch asks, and would still cost a full GPU run.
        raise RuntimeError(
            "refusing to start: the built command does not match the official "
            "configuration; see B1")

    # ---- B2: start it ------------------------------------------------------
    t_start = time.time()
    server.start()
    report["timings"]["vllm_start_s"] = round(time.time() - t_start, 1)
    report["api_base"] = server.base_url
    report["nvidia_smi_during_serve"] = nvidia_smi()
    rec("B2", True, {"base_url": server.base_url,
                     "start_s": report["timings"]["vllm_start_s"],
                     "nvidia_smi": report["nvidia_smi_during_serve"]},
        "server started with the official configuration")
    save_report()
except Exception as exc:  # noqa: BLE001
    # Save the log first: a server that refuses to start is the interesting case, and
    # its log is the only evidence there will ever be for it.
    try:
        if server is not None:
            log_text = Path(server.log_path).read_text(errors="replace")
            (WORK / "infer_batch_vllm_server.log").write_text(log_text)
            report["vllm_log_bytes"] = len(log_text)
    except Exception as log_exc:  # noqa: BLE001
        report["notes"].append("could not dump server log: %s: %s"
                               % (type(log_exc).__name__, log_exc))
    report["serve_failed"] = True
    fail_soft("B2", exc, "vllm_start")
    try:
        if server is not None:
            server.stop()
    except Exception:  # noqa: BLE001
        pass
'''

CELL_5 = """## 5. Comparison

Base model, trained adapter and zero-B control are asked the same question, and the
`prompt_logprobs` are compared position by position.

The pair matters. A large delta for `trained_gate2` alone cannot distinguish "the
adapter is active" from "the two model names happen to hit different request paths".
A zero-B control that sits at zero shows the pipeline is inert when the adapter is
inert, which is what makes the trained delta mean something.
"""

CELL_5_CODE = r'''import json, re, time

import requests

T_C = time.time()
PROMPT_TEXT = __PROMPT_JSON__
BASE = report.get("api_base")
FORBIDDEN = __FORBIDDEN_JSON__
LOADED_MARKER = __LOADED_MARKER_JSON__


def api(path, payload, timeout=__REQUEST_TIMEOUT__):
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
    """Max absolute difference over positions where both sides produced a value."""
    if a is None or b is None:
        return None, 0
    if len(a) != len(b):
        return None, min(len(a), len(b))
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    return max(abs(x - y) for x, y in pairs), len(pairs)


if BASE is None:
    reason = "serving did not start; see B2 and the server log"
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        report["checks"].setdefault(cid, {
            "ok": False, "value": None, "detail": "skipped: %s" % reason})
    report["skipped_stage"] = "comparison"
    print("SKIP comparison :: %s" % reason, flush=True)
else:
    try:
        # ---- model listing -----------------------------------------------
        listing = requests.get(BASE.rstrip("/") + "/models", timeout=180).json()
        ids = [m["id"] for m in listing.get("data", [])]
        report["served_models"] = ids
        base_name = next((i for i in ids
                          if not i.endswith(("_gate2", "_control"))
                          and "adapter:" not in i), None)
        report["base_model_alias"] = base_name
        rec("B3", base_name is not None
            and "trained_gate2" in ids and "zero_b_control" in ids, {"ids": ids},
            "base model and both adapter aliases are served")

        # ---- C2: noise floor ----------------------------------------------------
        base_a = prompt_logprobs(base_name)
        base_b = prompt_logprobs(base_name)
        base_c = prompt_logprobs(base_name)
        n1, p1 = compare(base_a, base_b)
        n2, p2 = compare(base_a, base_c)
        cands = [v for v in (n1, n2) if v is not None]
        noise = max(cands) if cands else None
        positions = (len(base_a) if base_a else 0)
        report["E2_noise_floor"] = noise
        report["E2_positions"] = positions
        report["E2_pairwise"] = {"ab": n1, "ac": n2}
        rec("C2", noise is not None, {"noise": noise, "positions": positions,
                                     "pairwise": report["E2_pairwise"]},
            "same model called three times: measurement noise floor")

        # ---- C3 / C4 / C5: the comparison --------------------------------------
        trained = prompt_logprobs("trained_gate2")
        zero_b = prompt_logprobs("zero_b_control")
        d_t, nt = compare(base_a, trained)
        d_t2, _ = compare(base_c, trained)
        d_z, nz = compare(base_a, zero_b)

        # ---- C1: the three requests tokenized identically --------------------
        # prompt_logprobs has one entry per prompt position, so equal lengths mean
        # equal token counts. Shorter or longer would mean the adapter requests took a
        # different encoding path, which would make the per-position deltas below a
        # comparison of different tokens.
        lengths = {"base": len(base_a or []),
                   "trained": len(trained or []),
                   "zero_b": len(zero_b or [])}
        encodings_match = (lengths["base"] > 0
                           and lengths["base"] == lengths["trained"]
                           == lengths["zero_b"])
        rec("C1", encodings_match, lengths,
            "all three models returned one logprob per prompt position")

        report["C3_delta"] = d_t
        report["C3_delta_second_reference"] = d_t2
        report["C3_positions"] = nt
        report["C4_delta"] = d_z
        report["C4_positions"] = nz
        report["logprob_lengths"] = {"base": len(base_a or []),
                                     "trained": len(trained or []),
                                     "zero_b": len(zero_b or [])}
        report["logprob_head"] = {
            "base": (base_a or [])[:5],
            "trained": (trained or [])[:5],
            "zero_b": (zero_b or [])[:5],
        }

        eff_noise = noise if (noise is not None and noise > 0.0) else 0.0
        c3_ok = d_t is not None and d_t >= 1e-3 and (
            eff_noise == 0.0 or d_t > 10 * eff_noise)
        rec("C3", c3_ok,
            {"delta": d_t, "positions": nt, "noise": noise,
             "lengths_match": report["logprob_lengths"]},
            "trained adapter moves the logprobs")

        c4_ok = d_z is not None and (
            d_z <= eff_noise if eff_noise > 0 else d_z == 0.0)
        rec("C4", c4_ok, {"delta": d_z, "positions": nz, "noise": noise},
            "zero-B control stays at the noise floor")

        c5_ok = (d_t is not None and d_t2 is not None and d_t2 >= 1e-3
                 and abs(d_t - d_t2) <= 0.05 * max(d_t, d_t2))
        rec("C5", c5_ok,
            {"delta_ref_a": d_t, "delta_ref_c": d_t2,
             "relative_gap": (round(abs(d_t - d_t2) / max(d_t, d_t2), 6)
                              if d_t and d_t2 else None)},
            "trained delta reproduces against a second base reference")
        save_report()

        # ---- C6: did the weights load without being dropped? -------------------
        try:
            log_text = Path(server.log_path).read_text(errors="replace")
            (WORK / "infer_batch_vllm_server.log").write_text(log_text)
            found = [f for f in FORBIDDEN if f in log_text]
            loaded = len(re.findall(re.escape(LOADED_MARKER), log_text))
            warn_lora = [ln.strip() for ln in log_text.splitlines()
                         if ln.strip().startswith("WARNING") and "lora" in ln.lower()]
            report["C6_forbidden_found"] = found
            report["C6_loaded_marker_count"] = loaded
            report["C6_lora_warnings"] = warn_lora[:10]
            report["vllm_log_bytes"] = len(log_text)
            report["C6_lora_mentions"] = log_text.lower().count("lora")
            c6_ok = not found and loaded >= 2
            rec("C6", c6_ok,
                {"forbidden_found": found, "loaded_marker_count": loaded,
                 "lora_warnings": warn_lora[:10],
                 "lora_mentions": report["C6_lora_mentions"],
                 "log_bytes": report["vllm_log_bytes"]},
                "both adapters reported loaded and no tensor was ignored")
        except Exception as exc:  # noqa: BLE001
            fail_soft("C6", exc, "vllm_log_scan")

        # ---- C7: generation ----------------------------------------------------
        try:
            gen_body = {"model": base_name, "prompt": PROMPT_TEXT, "max_tokens": 16,
                        "temperature": 0.0, "seed": 0, "logprobs": 5}
            gen = api("/completions", gen_body)
            choice = gen["choices"][0]
            gl = choice.get("logprobs") or {}
            report["C7_text"] = (choice.get("text") or "")[:200]
            report["C7_token_count"] = len(gl.get("tokens") or [])
            gen_t = api("/completions", dict(gen_body, model="trained_gate2"))
            gl_t = gen_t["choices"][0].get("logprobs") or {}
            tok_b, tok_t = gl.get("tokens") or [], gl_t.get("tokens") or []
            lp_b, lp_t = gl.get("token_logprobs") or [], gl_t.get("token_logprobs") or []
            n = min(len(lp_b), len(lp_t))
            gen_delta = max((abs(lp_b[i] - lp_t[i]) for i in range(n)
                             if lp_b[i] is not None and lp_t[i] is not None),
                            default=None)
            report["C7_gen_token_delta"] = gen_delta
            report["C7_gen_tokens_identical"] = tok_b[:16] == tok_t[:16]
            report["C7_trained_text"] = (gen_t["choices"][0].get("text") or "")[:200]
            rec("C7", report["C7_token_count"] > 0,
                {"tokens": report["C7_token_count"], "text": report["C7_text"][:80],
                 "gen_token_delta": gen_delta,
                 "same_tokens": report["C7_gen_tokens_identical"]},
                "generation produced tokens; trained adapter also generated")
        except Exception as exc:  # noqa: BLE001
            fail_soft("C7", exc, "generation")

        # ---- C8: tool calling through the trained adapter ----------------------
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
        attempts_spec = [
            ("trained_gate2", "What is the weather in Paris right now? "
                              "Use the get_weather tool.", {"enable_thinking": False}),
            ("trained_gate2", "You must call the function get_weather with city='Paris'. "
                              "Do not answer in prose.", {"enable_thinking": False}),
            ("trained_gate2", "What is the weather in Paris right now? "
                              "Use the get_weather tool.", {"enable_thinking": True}),
            (None, "What is the weather in Paris right now? Use the get_weather tool.",
             {"enable_thinking": False}),
        ]
        c8_ok = False
        log = []
        trained_called = False
        for idx, (alias, user_msg, tkw) in enumerate(attempts_spec):
            target = alias or base_name
            try:
                body = {"model": target,
                        "messages": [{"role": "user", "content": user_msg}],
                        "tools": TOOLS, "tool_choice": "auto",
                        "temperature": 0.0, "seed": 0, "max_tokens": 512,
                        "chat_template_kwargs": tkw}
                data = api("/chat/completions", body)
                msg = data["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                names = [c.get("function", {}).get("name") for c in calls]
                log.append({"attempt": idx, "model": target, "tool_calls": names,
                            "content": (msg.get("content") or "")[:200]})
                if alias == "trained_gate2" and any(n == "get_weather" for n in names):
                    trained_called = True
                    if not c8_ok:
                        c8_ok = True
                        break
            except Exception as exc:  # noqa: BLE001
                log.append({"attempt": idx, "model": target,
                            "error": "%s: %s" % (type(exc).__name__, exc)})
        report["C8_attempts"] = log
        report["C8_trained_adapter_called_tool"] = trained_called
        rec("C8", c8_ok and trained_called, log,
            "the trained adapter produced a parsed tool call")

    except Exception as exc:  # noqa: BLE001
        # An exception here means the comparison did not finish. The checks that did
        # not run are recorded as skipped, not failed, so the verdict cannot read as
        # "eight independent failures".
        for _cid in ("C3", "C4", "C5", "C7", "C8"):
            if _cid not in report["checks"]:
                report["checks"][_cid] = {
                    "ok": False, "value": None,
                    "detail": "skipped after error: %s: %s"
                              % (type(exc).__name__, exc),
                }
        fail_soft("C3", exc, "comparison")
    finally:
        # Every path ends here: a server left running would hold all four GPUs until
        # the kernel is killed, and the next reader would have no shutdown evidence.
        try:
            if server is not None:
                server.stop()
                report["nvidia_smi_after_stop"] = nvidia_smi()
        except Exception as stop_exc:  # noqa: BLE001
            report["notes"].append("server stop raised: %s: %s"
                                   % (type(stop_exc).__name__, stop_exc))

report["timings"]["comparison_s"] = round(time.time() - T_C, 1)
save_report()

'''

CELL_6 = """## 6. Verdict

Skipped checks are not passes. A group counts as passing only when nothing in it is
missing, failed or skipped.
"""

CELL_6_CODE = r'''import json

HARD_GROUPS = {
    "A": ["A1", "A2", "A3", "A4"],
    "B": ["B1", "B2", "B3"],
    "C": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
}

skipped_ids = {
    cid for cid, v in report["checks"].items()
    if isinstance(v.get("detail"), str) and v["detail"].startswith("skipped:")
}

groups = {}
for name, ids in HARD_GROUPS.items():
    missing = [c for c in ids if c not in report["checks"]]
    failed = [c for c in ids if c in report["checks"]
              and report["checks"][c]["ok"] is not True
              and c not in skipped_ids]
    skipped = [c for c in ids if c in skipped_ids]
    groups[name] = {"missing": missing, "failed": failed, "skipped": skipped,
                    "ok": not missing and not failed and not skipped}
    print("%s: missing=%s failed=%s skipped=%s -> %s"
          % (name, missing, failed, skipped,
             "PASS" if groups[name]["ok"] else "FAIL"), flush=True)

report["verdict"] = {
    "groups": groups,
    "pass": all(g["ok"] for g in groups.values()),
    "scope": report["scope"],
    "inference_compatible_only": True,
    "does_not_establish_agent_improvement": True,
}
report["wall_seconds"] = round(time.time() - T0, 1)
report["finished_epoch"] = time.time()
save_report()

print()
print("VERDICT: %s" % ("PASS" if report["verdict"]["pass"] else "FAIL"))
for name, g in groups.items():
    if not g["ok"]:
        print("  blocked at group %s: failed=%s skipped=%s missing=%s"
              % (name, g["failed"], g["skipped"], g["missing"]))
print("wall %.1fs of %ds cap" % (report["wall_seconds"], RUNTIME_CAP))
'''


def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def build_notebook(expected_manifest):
    manifest_json = json.dumps(expected_manifest)
    official_json = json.dumps({k: v for k, v in OFFICIAL_SERVE.items()})

    def fill(src):
        return (src
                .replace("__SOFT_DEADLINE_SECONDS__", str(SOFT_DEADLINE_SECONDS))
                .replace("__RUNTIME_CAP_SECONDS__", str(RUNTIME_CAP_SECONDS))
                .replace("__SERVER_START_TIMEOUT__", str(SERVER_START_TIMEOUT))
                .replace("__REQUEST_TIMEOUT__", str(REQUEST_TIMEOUT))
                .replace("__KERNEL_ID__", json.dumps(KERNEL_ID))
                .replace("__BASE_SLUG__", json.dumps(BASE_SLUG))
                .replace("__BASE_REF__", json.dumps(BASE_REF))
                .replace("__ADAPTER_DATASET__", json.dumps(ADAPTER_DATASET))
                .replace("__EXPECTED_MANIFEST_JSON__", json.dumps(manifest_json))
                .replace("__OFFICIAL_SERVE_JSON__", json.dumps(official_json))
                .replace("__PROMPT_JSON__", json.dumps(PROMPT))
                .replace("__FORBIDDEN_JSON__", json.dumps(FORBIDDEN_WARNINGS))
                .replace("__LOADED_MARKER_JSON__", json.dumps(LOADED_MARKER)))

    cells = [
        md_cell(CELL_0),
        md_cell(CELL_1), code_cell(fill(CELL_1_CODE)),
        md_cell(CELL_2), code_cell(fill(CELL_2_CODE)),
        md_cell(CELL_3), code_cell(fill(CELL_3_CODE)),
        md_cell(CELL_4), code_cell(fill(CELL_4_CODE)),
        md_cell(CELL_5), code_cell(fill(CELL_5_CODE)),
        md_cell(CELL_6), code_cell(fill(CELL_6_CODE)),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"},
    }, "nbformat": 4, "nbformat_minor": 5}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / "artifacts" /
                           "infer_batch_v1_expected_manifest.json")
                          .read_text(encoding="utf-8"))
    nb = build_notebook(expected)
    nb_path = DEST / "infer_batch.ipynb"
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    nb_path.write_text(text, encoding="utf-8", newline="\n")
    nb_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "infer_batch.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": "NvidiaL4",
        "dataset_sources": [
            "metric/gemma-4-developer-agent-wheelhouse",
            "taruki/gemma4-training-wheels",
            ADAPTER_DATASET,
        ],
        "competition_sources": ["gemma-4-developer-agent"],
        "model_sources": [BASE_REF],
        "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                         "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"),
    }
    (DEST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")

    experiment = {
        "batch": "gate2_infer_batch_v1",
        "purpose": ("verify that the official serving stack consumes the two adapters "
                    "saved by Gate 2 v3; inference compatibility only"),
        "notebook_sha256": nb_sha,
        "prereg": "artifacts/infer_batch_v1_prereg.md",
        "base_model_ref": BASE_REF,
        "adapter_dataset": ADAPTER_DATASET,
        "official_serve_config": OFFICIAL_SERVE,
        "runtime_cap_seconds": RUNTIME_CAP_SECONDS,
        "soft_deadline_seconds": SOFT_DEADLINE_SECONDS,
        "server_start_timeout": SERVER_START_TIMEOUT,
        "request_timeout": REQUEST_TIMEOUT,
        "expected": {
            "adapters": ["trained_gate2", "zero_b_control"],
            "trained_lora_B_nonzero": 120,
            "control_lora_B_nonzero": 0,
            "adapter_tensors_each": 240,
            "c3_min_delta": 1e-3,
            "c3_noise_multiple": 10.0,
            "c6_loaded_marker_min": 2,
        },
        "not_in_scope": [
            "training (this batch trains nothing)",
            "agent code-fixing quality",
            "full SFT",
            "RL",
            "leaderboard submission",
        ],
        "does_not_establish_agent_improvement": True,
    }
    (DEST / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8", newline="\n")

    print("notebook   %s (%d cells)" % (nb_path.relative_to(ROOT), len(nb["cells"])))
    print("sha256     %s" % nb_sha)
    print("metadata   %s" % (DEST / "kernel-metadata.json").relative_to(ROOT))
    print("experiment %s" % (DEST / "experiment.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
