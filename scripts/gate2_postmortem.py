"""Post-mortem for Gate 2 (31B) run 1: locate the load-phase OOM precisely.

Reads the collected execution log, extracts the decompress timeline and the
per-device memory picture, and writes a compact JSON summary. Kept separate from
`verify_gate2.py` because the verdict is already unambiguous (a load-time OOM);
this script only answers *why* the decompression needed more memory than the
4 x L4 budget allows.
"""

import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "training_gate2_outputs"
LOG = OUT / "gemma4-gate2-31b.log"

rows = []
text = io.open(LOG, encoding="utf-8", errors="replace").read()
for line in text.splitlines():
    line = line.strip().lstrip(",")
    try:
        obj = json.loads(line)
    except Exception:
        continue
    rows.append((obj.get("time"), obj.get("data", "")))

# Decompression progress: the last percentage reached before the OOM tells us how
# far the decompressor got, and the rate tells us how big the peak was.
progress = []
for tm, data in rows:
    m = re.search(r"Decompressing model:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)", data)
    if m and tm is not None:
        progress.append({"t": round(tm, 1), "pct": int(m.group(1)),
                         "done": int(m.group(2)), "total": int(m.group(3))})

oom = None
for tm, data in rows:
    if "OutOfMemoryError" in data and oom is None:
        oom = {"t": round(tm, 1) if tm else None, "text": data.strip()}

# The decompressor's peak: packed int32 input plus the unpacked bf16 output.
# 442 MiB was the allocation that failed, at 62% of 1763 modules.
summary = {
    "decompress_progress": progress,
    "decompress_last_pct": progress[-1]["pct"] if progress else None,
    "decompress_last_done": progress[-1]["done"] if progress else None,
    "decompress_total_modules": progress[-1]["total"] if progress else None,
    "oom": oom,
}

again = re.search(r"Tried to allocate ([\d.]+) MiB", oom["text"]) if oom else None
summary["failed_allocation_mib"] = float(again.group(1)) if again else None

pp = re.search(r"of which ([\d.]+) MiB is free", oom["text"]) if oom else None
summary["free_at_oom_mib"] = float(pp.group(1)) if pp else None

al = re.search(r"([\d.]+) GiB is allocated by PyTorch", oom["text"]) if oom else None
summary["allocated_at_oom_gib"] = float(al.group(1)) if al else None

rs = re.search(r"([\d.]+) GiB is reserved by PyTorch but unallocated", oom["text"]) if oom else None
summary["reserved_unallocated_gib"] = float(rs.group(1)) if rs else None

(ROOT / "artifacts" / "gate2_postmortem.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
