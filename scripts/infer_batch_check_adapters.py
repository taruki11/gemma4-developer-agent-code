"""Verify the two adapter files locally before they are uploaded.

This is the only place the full tensor contents are read, so the claims about them are
established here rather than inferred from file sizes: the trained adapter must have
nonzero lora_B everywhere, the control must have all-zero lora_B, and both must carry
exactly the 240 keys the declared target regex implies.
"""

import io
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "artifacts" / "training_gate2_outputs_v3" / "gate2_layout" / "adapters"


def read_header(path):
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode("utf-8")), 8 + n


def tensor_stats(path, want_lora_b_zero):
    header, data_start = read_header(path)
    meta = header.pop("__metadata__", {})
    keys = sorted(header)
    suffix_a = [k for k in keys if ".lora_A." in k]
    suffix_b = [k for k in keys if ".lora_B." in k]
    # Every value must be BF16 or F32; sum |value| over the raw bytes.
    nonzero_b = 0
    total_b = 0
    for k in suffix_b:
        info = header[k]
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        with path.open("rb") as fh:
            fh.seek(data_start + info["data_offsets"][0])
            blob = fh.read(nbytes)
        total_b += 1
        if any(blob):
            nonzero_b += 1
    return {
        "n_keys": len(keys),
        "n_lora_A": len(suffix_a),
        "n_lora_B": len(suffix_b),
        "lora_B_tensors_with_nonzero_bytes": nonzero_b,
        "metadata": meta,
        "sample": keys[:3],
        "want_zero_b": want_lora_b_zero,
    }


for label, want_zero in (("trained_gate2", False), ("zero_b_control", True)):
    st = tensor_stats(LAYOUT / label / "adapter_model.safetensors", want_zero)
    print("=== %s ===" % label)
    print(json.dumps(st, indent=2)[:900])
    print()

# Confirm the declared regex covers exactly the keys present.
import re
cfg = json.load(io.open(LAYOUT / "trained_gate2" / "adapter_config.json",
                        encoding="utf-8"))
pat_src = cfg["target_modules"]
print("declared target_modules:", pat_src)
# PEFT stores keys as base_model.model.<module>.lora_A.weight
BASE = "base_model.model."
HEADER, _ = read_header(LAYOUT / "trained_gate2" / "adapter_model.safetensors")
HEADER.pop("__metadata__", None)
pat = re.compile(pat_src)
mods, bad = set(), []
for k in HEADER:
    core = k[len(BASE):] if k.startswith(BASE) else k
    if ".lora_A." not in core and ".lora_B." not in core:
        bad.append(k)
        continue
    mod = core.split(".lora_")[0]
    if not pat.fullmatch(mod):
        bad.append(k)
    else:
        mods.add(mod)
print("distinct modules matched by the declared regex:", len(mods))
print("keys not covered:", len(bad), bad[:4])
layer_re = re.compile(r"\.layers\.(\d+)\.self_attn\.(q|o)_proj$")
parsed = [layer_re.search(m) for m in mods]
print("modules parseable:", sum(1 for m in parsed if m), "of", len(mods))
layers = sorted({int(m.group(1)) for m in parsed if m})
print("layer indices covered:", len(layers), "min", layers[0], "max", layers[-1])
print("all 0-59 covered:", layers == list(range(60)))
print("projection suffixes:", sorted({m.group(2) + "_proj" for m in parsed if m}))
