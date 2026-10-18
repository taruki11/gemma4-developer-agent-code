"""Compare the two 31B serving attempts on identical hardware.

Gate 2 v1 and this batch used the same checkpoint, the same TP=4, the same 0.80
utilization and the same 32768 context. The only difference is --enforce-eager. If v1
started and this one did not, then the flag is load-bearing on this hardware and the
official configuration does not fit as written -- which is the single most useful thing
this run can establish, so it needs to be checked rather than asserted.
"""

import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

v1 = json.load(io.open(ROOT / "artifacts" / "training_gate2_outputs" / "gate2_report.json",
                       encoding="utf-8"))
v3 = json.load(io.open(ROOT / "artifacts" / "training_gate2_outputs_v3" / "gate2_report.json",
                       encoding="utf-8"))
inf = json.load(io.open(ROOT / "artifacts" / "infer_batch_v1_outputs" / "infer_batch_report.json",
                        encoding="utf-8"))


def cmd_flags(cmd):
    if not cmd:
        return None
    out = {}
    for flag in ("--max-model-len", "--gpu-memory-utilization",
                 "--tensor-parallel-size", "--max-loras", "--max-lora-rank"):
        if flag in cmd:
            out[flag] = cmd[cmd.index(flag) + 1]
    out["--enforce-eager"] = "--enforce-eager" in cmd
    return out


print("=== Gate 2 v1 (training run, enforce-eager) ===")
print("cmd:", json.dumps(cmd_flags(v1.get("vllm_cmd")), indent=1))
print("started:", v1.get("api_base") is not None)
print("nvidia_smi_after_start:", json.dumps(v1.get("nvidia_smi_after_start")))
print("startup_attempts:", json.dumps(v1.get("vllm_start_attempts")))
print()
print("=== Gate 2 v3 (did not start) ===")
print("cmd:", json.dumps(cmd_flags(v3.get("vllm_cmd")), indent=1))
print("started:", v3.get("api_base") is not None)
print("vllm_gpu_memory_utilization:", v3.get("vllm_gpu_memory_utilization"))
print("D4 value:", json.dumps(v3.get("checks", {}).get("D4", {}).get("value")))
print()
print("=== Inference batch v1 (official config, no enforce-eager) ===")
print("cmd:", json.dumps(cmd_flags(inf.get("vllm_cmd")), indent=1))
print("started:", inf.get("api_base") is not None)
print("serve_failed:", inf.get("serve_failed"))
print("wall:", inf.get("wall_seconds"))
print("timings:", json.dumps(inf.get("timings")))
print()
print("gpu totals:", inf.get("gpu", {}).get("total_memory_gib"))

# Relative headroom the two configurations actually asked for.
tot = inf["gpu"]["total_memory_gib"][0]
print()
print("per-GPU total: %.2f GiB" % tot)
print("  0.80 utilization  => %.2f GiB budgeted" % (tot * 0.80))
print("  0.80 on 24 GB     => %.2f GiB (what the README assumes)"
      % (24.0 * 0.80))
