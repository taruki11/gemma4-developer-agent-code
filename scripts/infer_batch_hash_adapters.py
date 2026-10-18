"""Record SHA256 for the v3 adapters that this batch will upload.

These files are the only input to the new run. Their digests must be known before the
push so that what the GPU kernel actually reads can be compared afterwards: an upload
that silently rewrote a safetensors file would invalidate every logprob comparison in
this batch without raising anything.
"""

import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "artifacts" / "training_gate2_outputs_v3" / "gate2_layout"
ADAPTER = ROOT / "artifacts" / "training_gate2_outputs_v3" / "gate2_adapter"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out = {"source_of_adapters": str(LAYOUT.relative_to(ROOT)), "files": {}}

for label in ("trained_gate2", "zero_b_control"):
    d = LAYOUT / "adapters" / label
    entry = {}
    for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        f = d / name
        if f.is_file():
            entry[name] = {"sha256": sha(f), "bytes": f.stat().st_size}
    out["files"][label] = entry

# The original single-adapter copy must also be recorded and left untouched.
extra = {}
for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
    f = ADAPTER / name
    if f.is_file():
        extra[name] = {"sha256": sha(f), "bytes": f.stat().st_size}
out["original_adapter_copy"] = {"path": str(ADAPTER.relative_to(ROOT)), "files": extra}

# Config facts that the run will assert, read here so they are on record.
for label in ("trained_gate2", "zero_b_control"):
    cfg = json.load(io.open(LAYOUT / "adapters" / label / "adapter_config.json",
                            encoding="utf-8"))
    out.setdefault("config", {})[label] = {
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": cfg.get("target_modules"),
        "use_dora": cfg.get("use_dora"),
        "peft_type": cfg.get("peft_type"),
        "task_type": cfg.get("task_type"),
    }

# Do the two adapters differ only in lora_B values? Record the tensor digests so the
# zero-B control's provenance is explicit.
safetensors = {}
for label in ("trained_gate2", "zero_b_control"):
    f = LAYOUT / "adapters" / label / "adapter_model.safetensors"
    safetensors[label] = sha(f)
out["safetensors_differ"] = safetensors["trained_gate2"] != \
    safetensors["zero_b_control"]

dest = ROOT / "artifacts" / "infer_batch_v1_adapter_hashes.json"
dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

for label, files in out["files"].items():
    print(label)
    for name, meta in files.items():
        print("  %-28s %s  %d B" % (name, meta["sha256"][:32] + "...",
                                    meta["bytes"]))
print()
print("adapters differ (expected, zero-B control):", out["safetensors_differ"])
print("wrote", dest.relative_to(ROOT))
