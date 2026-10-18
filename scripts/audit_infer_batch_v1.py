"""Audit the inference-batch notebook against the constraints the batch must satisfy.

Each assertion below corresponds to something the user asked for or to a way an earlier
run silently produced a wrong answer. The point is to fail here, on a laptop, rather
than after a GPU run has been paid for.
"""

import ast
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "artifacts" / "infer_batch_v1" / "infer_batch.ipynb"
EXP = json.loads((ROOT / "artifacts" / "infer_batch_v1" / "experiment.json")
                 .read_text(encoding="utf-8"))
META = json.loads((ROOT / "artifacts" / "infer_batch_v1" / "kernel-metadata.json")
                  .read_text(encoding="utf-8"))

nb = json.load(io.open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
full = "\n".join(code)

failures = []


def want(cond, msg):
    if not cond:
        failures.append(msg)
    print("%s %s" % ("ok  " if cond else "FAIL", msg))


# ---- shape -----------------------------------------------------------------
want(len(nb["cells"]) == 13, "13 cells (7 markdown + 6 code)")
want(len(code) == 6, "6 code cells")

# ---- nothing from the training stack --------------------------------------
want("AutoModelForCausalLM" not in full,
     "no Hugging Face model is loaded (the training model is not needed here)")
want("from_pretrained" not in full, "no from_pretrained call anywhere")
want("CompressedTensorsConfig" not in full,
     "no quantized-model loading in-process; only vLLM's subprocess loads weights")
want("get_peft_model" not in full, "no PEFT injection; adapters are served, not merged")
want("LoraConfig" not in full, "no LoRA config construction")

# ---- the serving configuration is the official one ------------------------
for name, value in (
    ("tensor_parallel_size", 4),
    ("gpu_memory_utilization", 0.80),
    ("max_model_len", 32768),
    ("enable_lora", True),
    ("max_loras", 8),
    ("max_lora_rank", 128),
):
    want(EXP["official_serve_config"][name] == value,
         "official_serve_config.%s == %r" % (name, value))
want(EXP["official_serve_config"]["enforce_eager"] is False,
     "the declared configuration does not use enforce-eager")
want(EXP["official_serve_config"]["extra_args"] == [],
     "no extra vLLM arguments are appended")
want(EXP["official_serve_config"]["default_chat_template_kwargs"]
     == {"enable_thinking": True},
     "official default_chat_template_kwargs carried")

# The command must not contain the flag, and that must be asserted, not just omitted.
want("--enforce-eager" not in re.sub(r'"[^"]*--enforce-eager[^"]*"', "", full),
     "--enforce-eager appears only inside quoted strings, never as a real argument")
want('"enforce_eager_present": "--enforce-eager" in cmd' in full,
     "the built command is inspected for enforce-eager")
want('and actual["enforce_eager_present"] is False' in full,
     "B1 requires enforce-eager to be absent")
want("refusing to start" in full,
     "a non-official command refuses to start instead of running anyway")

# ---- B1 checks the real command, not the declared config ------------------
want('after("--tensor-parallel-size")' in full, "B1 reads TP size from the command")
want('after("--gpu-memory-utilization")' in full, "B1 reads GPU util from the command")
want('actual["max_model_len"] == "32768"' in full, "B1 pins max-model-len 32768")
want("declared_vs_actual" in full,
     "declared vs actual max_loras/rank are both recorded, since vLLM resizes them")

# ---- adapter integrity before and after the mount ------------------------
# One definition plus one call site per stage (mounted, then copied).
want(full.count("sha256_file(") == 3, "adapters are hashed at both stages")
want('rec("A1"' in full and 'rec("A2"' in full,
     "A1 (mounted matches assembled) and A2 (copy matches mounted) both recorded")
want("adapter_manifest.json" in full, "the assembly-time manifest is read in-kernel")
want('rec("A4"' in full and "n_lora_B_nonzero" in full,
     "A4 counts nonzero lora_B tensors in both adapters")
want("== 120" in full and "== 0" in full,
     "A4 requires 120 nonzero in the trained adapter and 0 in the control")

# ---- the comparison pair --------------------------------------------------
want('rec("C3"' in full and 'rec("C4"' in full,
     "C3 (trained) and C4 (zero-B control) are both recorded")
want("eff_noise = noise if (noise is not None and noise > 0.0) else 0.0" in full,
     "the noise floor is normalised before it is used as a threshold")
want("d_t >= 1e-3" in full, "C3 requires a delta above the 1e-3 floor")
want("d_z == 0.0" in full, "C4 requires exact zero when the noise floor is zero")
want('rec("C5"' in full and "0.05 * max(d_t, d_t2)" in full,
     "C5 requires the delta to reproduce against a second base reference")
want('rec("C1"' in full and "encodings_match" in full,
     "C1 checks the three requests produced equal-length logprob vectors")
want("/tokenize" not in full,
     "no dependency on an endpoint whose presence in this vLLM build is unverified")

# ---- weight-consumption evidence -----------------------------------------
want("is not in the model's supported LoRA target modules" in full
     and "These parameters will be ignored" in full
     and "does not support LoRA yet" in full,
     "all three silent-drop warnings are scanned for")
want("Loaded new LoRA adapter" in full,
     "positive evidence (the adapter-loaded line) is required, not just absence of warnings")
want("loaded >= 2" in full, "C6 requires one loaded line per adapter")

# ---- tool calling through the trained adapter ----------------------------
want('"C8_trained_adapter_called_tool"' in full,
     "C8 records whether the trained adapter produced the tool call")
want('trained_called = False' in full and 'alias == "trained_gate2"' in full,
     "C8 only credits a tool call that came from the trained adapter")

# ---- fail-fast and shutdown ----------------------------------------------
want("finally:" in full, "the comparison body shuts the server down in a finally")
# Two legitimate call sites: cell 4 stops a server whose start() raised, cell 5's
# finally stops a server that came up. Neither can run after the other, because a
# failed start leaves api_base unset and skips the comparison branch entirely.
want(full.count("server.stop()") == 2, "both shutdown call sites present")
want("BASE is None" in full and "else:" in full,
     "the comparison branch is guarded on api_base, so the two stops are exclusive")
want("skipped:" in full, "unreached checks are marked skipped")
want('groups[name] = {"missing": missing, "failed": failed, "skipped": skipped,'
     in full, "the verdict separates missing, failed and skipped")
want('"ok": not missing and not failed and not skipped' in full,
     "a skipped check cannot make a group pass")

# ---- scope statement -----------------------------------------------------
want("does_not_establish_agent_improvement" in full,
     "the report states this batch does not measure agent improvement")
want("inference_compatible_only" in full, "the verdict is scoped to inference")

# ---- kernel metadata -----------------------------------------------------
want(META["model_sources"] == ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
     "the competition 31B checkpoint is the only mounted model")
want("taruki/gemma4-infer-batch-v1-adapters" in META["dataset_sources"],
     "the adapter dataset is mounted")
want(META["enable_internet"] is False, "internet stays disabled")
want(META["machine_shape"] == "NvidiaL4", "4x L4 shape")

# ---- limits ---------------------------------------------------------------
want(EXP["runtime_cap_seconds"] == 3600, "declared 1 hour cap")
want(EXP["server_start_timeout"] <= EXP["soft_deadline_seconds"],
     "the server start timeout fits inside the soft deadline")

# ---- syntax ---------------------------------------------------------------
for i, src in enumerate(code):
    try:
        ast.parse(src)
    except SyntaxError as exc:
        failures.append("cell %d does not parse: %s" % (i, exc))
print("%s all code cells parse" % ("ok  " if not failures else "FAIL"))

print()
if failures:
    print("AUDIT: FAIL (%d)" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("AUDIT: PASS")
