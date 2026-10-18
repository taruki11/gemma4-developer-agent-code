"""Package the downloaded official sample. Does not replace official validation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("sample", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = args.sample.resolve()
if not (root / "agent.yaml").is_file():
    parser.error("official sample root must contain agent.yaml")
files = sorted(path for path in root.rglob("*") if path.is_file())
for path in files:
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        parser.error(f"unsafe path: {path.name}")
    if path.name in {".env", "kaggle.json", "access_token"}:
        parser.error("credential filename found in submission")
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root).as_posix())
with zipfile.ZipFile(args.output) as archive:
    assert "agent.yaml" in archive.namelist()
    assert archive.testzip() is None
report = {"archive": str(args.output.resolve()), "bytes": args.output.stat().st_size,
          "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
          "files": [path.relative_to(root).as_posix() for path in files],
          "official_validator_passed": False, "model_inference_completed": False}
args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
