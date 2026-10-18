"""Dry-run discover_adapters against the exact layout the notebook will build.

The notebook copies the dataset's adapters/ into /kaggle/working/infer_layout and lets
adk_submission discover them there. If discovery renames them, or returns a different
count, the model aliases used in every comparison would be wrong -- and that failure
would only surface after the server was already up. Checking it here costs nothing.
"""

import json
import shutil
import tempfile
from pathlib import Path

from adk_submission import discover_adapters

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "infer_batch_v1_dataset"

with tempfile.TemporaryDirectory() as tmp:
    layout = Path(tmp) / "infer_layout"
    shutil.copytree(SRC, layout)
    # The notebook deletes the dataset metadata from the copied layout's parent only;
    # the adapters/ subtree is what matters here.
    manifest = discover_adapters(str(layout))
    found = {name: {"path": str(i.path), "bytes": int(i.size_bytes),
                    "rank": i.rank, "format": i.format}
             for name, i in manifest.adapters.items()}
    print(json.dumps(found, indent=2))
    assert set(found) == {"trained_gate2", "zero_b_control"}, found
    for name, info in found.items():
        assert Path(info["path"]).name == "adapter_model.safetensors", info
        assert info["rank"] == 8, (name, info["rank"])
        assert info["bytes"] == 57379688 if name == "trained_gate2" else True
    print()
    print("DISCOVERY: PASS")
    print("names:", sorted(found))
    print("ranks:", {k: v["rank"] for k, v in found.items()})
    print("primary files:", {k: Path(v["path"]).name for k, v in found.items()})
