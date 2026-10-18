"""Print build_cmd so B1's assertions can be written against the real command.

B1 refuses to start the server when the built command is not the official one, so it
has to be right about what build_cmd produces: which flags appear, in what spelling, and
whether the adapter count overrides the declared --max-loras.
"""

import inspect
import json
import shutil
import tempfile
from pathlib import Path

from adk_submission import VllmConfig, VllmServer, discover_adapters

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "infer_batch_v1_dataset"

with tempfile.TemporaryDirectory() as tmp:
    layout = Path(tmp) / "infer_layout"
    shutil.copytree(SRC, layout)
    manifest = discover_adapters(str(layout))
    cfg = VllmConfig(
        model=str(layout),  # any path; only the command shape matters here
        host="127.0.0.1",
        port=8000,
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        max_model_len=32768,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        enable_auto_tool_choice=True,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=128,
        tensor_parallel_size=4,
        startup_timeout=1500,
        extra_args=[],
    )
    server = VllmServer(cfg, adapter_manifest=manifest)
    cmd = server.build_cmd()
    print(json.dumps(cmd, indent=1))
    print()
    print("enforce-eager present:", "--enforce-eager" in cmd)
    print("max-loras:", cmd[cmd.index("--max-loras") + 1] if "--max-loras" in cmd else None)
    print("max-lora-rank:",
          cmd[cmd.index("--max-lora-rank") + 1] if "--max-lora-rank" in cmd else None)
    print("lora-modules:", [c for c in cmd if c.startswith("trained_gate2=")
                            or c.startswith("zero_b_control=")])
    print()
    print("--- VllmConfig fields ---")
    print(json.dumps(sorted(cfg.__dataclass_fields__), indent=1))
