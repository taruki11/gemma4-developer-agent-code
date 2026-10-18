import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
v1 = json.load(io.open(ROOT / "artifacts/training_gate2_outputs/gate2_report.json",
                       encoding="utf-8"))
print("=== FULL Gate 2 v1 vllm_cmd ===")
print(json.dumps(v1.get("vllm_cmd"), indent=1))

print()
print("=== search swegemma / adk_eval_core for VllmConfig construction ===")
import adk_eval_core
import swegemma

for pkg in (adk_eval_core, swegemma):
    base = Path(pkg.__file__).parent
    for py in sorted(base.rglob("*.py")):
        txt = py.read_text(encoding="utf-8", errors="replace")
        if "VllmConfig" in txt or "gpu_memory_utilization" in txt or "max_num_seqs" in txt:
            print()
            print("---", py.relative_to(base.parent))
            for i, ln in enumerate(txt.splitlines(), 1):
                if re.search(r"VllmConfig|gpu_memory_utilization|max_num_seqs|"
                             r"max_model_len|enforce_eager|extra_args", ln):
                    print("  %4d| %s" % (i, ln.strip()[:160]))
