"""Build the Kaggle dataset that carries the two v3 adapters into the inference batch.

The kernel cannot read files off this workstation, so the adapters travel as a dataset.
Everything that will be compared in the run is hashed here, at the moment the payload
is assembled, so the in-kernel check has a fixed reference to compare against.
"""

import hashlib
import io
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "training_gate2_outputs_v3" / "gate2_layout" / "adapters"
DEST = ROOT / "artifacts" / "infer_batch_v1_dataset"
DATASET_ID = "taruki/gemma4-infer-batch-v1-adapters"

FILES = ("adapter_config.json", "adapter_model.safetensors", "README.md")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if DEST.exists():
    shutil.rmtree(DEST)
(DEST / "adapters").mkdir(parents=True)

manifest = {"source": str(SRC.relative_to(ROOT)), "adapters": {}}

for label in ("trained_gate2", "zero_b_control"):
    out_dir = DEST / "adapters" / label
    out_dir.mkdir(parents=True)
    entry = {}
    for name in FILES:
        src = SRC / label / name
        if not src.is_file():
            continue
        shutil.copy2(src, out_dir / name)
        entry[name] = {"sha256": sha(src), "bytes": src.stat().st_size}
    manifest["adapters"][label] = entry

# The manifest is part of the payload so the kernel can hash what it actually sees;
# it is also kept locally as the reference copy.
(DEST / "adapter_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(ROOT / "artifacts" / "infer_batch_v1_expected_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

meta = {
    "title": "Gemma4 Infer Batch v1 Adapters",
    "id": DATASET_ID,
    "licenses": [{"name": "CC0-1.0"}],
}
(DEST / "dataset-metadata.json").write_text(
    json.dumps(meta, indent=2) + "\n", encoding="utf-8")

print("assembled", DEST.relative_to(ROOT))
total = 0
for label, files in manifest["adapters"].items():
    for name, m in files.items():
        total += m["bytes"]
        print("  %-16s %-26s %s %d B" % (label, name, m["sha256"][:16], m["bytes"]))
print("payload total %d B (%.1f MiB)" % (total, total / 1024**2))
