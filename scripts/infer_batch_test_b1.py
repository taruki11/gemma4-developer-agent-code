"""Rebuild the command exactly as the notebook does and assert B1 would pass.

This runs the same VllmConfig construction the notebook uses, so a mismatch between the
declared official configuration and what build_cmd emits is caught here rather than on
the GPU. If B1 would fail, the notebook now refuses to start at all -- so this test is
what makes that refusal safe to rely on.
"""

import json
import shutil
import tempfile
from pathlib import Path

from adk_submission import VllmConfig, VllmServer, discover_adapters

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "infer_batch_v1_dataset"
OFFICIAL = json.loads(
    (ROOT / "artifacts" / "infer_batch_v1" / "experiment.json")
    .read_text(encoding="utf-8"))["official_serve_config"]

with tempfile.TemporaryDirectory() as tmp:
    layout = Path(tmp) / "infer_layout"
    shutil.copytree(SRC, layout)
    manifest = discover_adapters(str(layout))
    vcfg = VllmConfig(
        model=str(layout),
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
        startup_timeout=1500,
        extra_args=list(OFFICIAL["extra_args"]),
    )
    cmd = VllmServer(vcfg, adapter_manifest=manifest).build_cmd()


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
print(json.dumps(actual, indent=2))

b1_ok = (
    actual["tensor_parallel_size"] == "4"
    and abs(float(actual["gpu_memory_utilization"]) - 0.80) < 1e-6
    and actual["max_model_len"] == "32768"
    and actual["enable_lora"] is True
    and actual["enable_auto_tool_choice"] is True
    and actual["tool_call_parser"] == "gemma4"
    and actual["reasoning_parser"] == "gemma4"
    and actual["default_chat_template_kwargs"] is not None
    and "enable_thinking" in actual["default_chat_template_kwargs"]
    and actual["enforce_eager_present"] is False
)
print()
print("B1 would pass:", b1_ok)
print("declared max_loras:", OFFICIAL["max_loras"], "-> actual:", actual["max_loras"])
print("declared max_lora_rank:", OFFICIAL["max_lora_rank"],
      "-> actual:", actual["max_lora_rank"])
assert b1_ok, "B1 gate would refuse to start with the official configuration"
