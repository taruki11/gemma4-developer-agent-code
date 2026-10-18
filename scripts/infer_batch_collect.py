"""Collect the inference batch outputs into their own directory.

Kept separate from the training output directories so the v3 evidence is never
overwritten by a later batch.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "taruki/gemma4-infer-batch-v1"
OUT = ROOT / "artifacts" / "infer_batch_v1_outputs"
KAGGLE = ROOT / ".venv" / "Scripts" / "kaggle.exe"

if OUT.exists() and any(OUT.iterdir()):
    print("refusing to overwrite existing outputs in %s" % OUT.relative_to(ROOT))
    sys.exit(2)

OUT.mkdir(parents=True, exist_ok=True)
log = OUT / "collection.log"

for attempt in range(1, 7):
    cmd = [str(KAGGLE), "kernels", "output", KERNEL_ID, "-p", str(OUT)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    text = ("$ %s\n%s\n%s\nexit %d\n"
            % (" ".join(cmd), proc.stdout or "", proc.stderr or "", proc.returncode))
    with log.open("a", encoding="utf-8") as fh:
        fh.write(text)
    print("--- attempt %d exit %d ---" % (attempt, proc.returncode))
    print((proc.stdout or "")[-1200:])
    if proc.returncode == 0:
        break
    if attempt == 6:
        print("COLLECTION: FAILED after 6 attempts")
        sys.exit(1)
    time.sleep(30)

files = sorted(p.name for p in OUT.iterdir() if p.is_file())
print()
print("COLLECTED %d files:" % len(files))
for name in files:
    print("  %-40s %d B" % (name, (OUT / name).stat().st_size))
