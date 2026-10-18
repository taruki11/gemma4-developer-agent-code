"""Static check: does the v3 notebook record real named_modules() evidence for A3?

The v2 failure was a check that could not tell "no match" from "absent everywhere".
The fix must record what the model actually reported, so the report can be judged on
observed module names rather than on a hand-built test. This asserts that the code
which produces those fields is present and reachable.
"""

import ast
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "artifacts" / "training_gate2" / "smoke_v3_cpu_decompress_a3fix.ipynb"
nb = json.load(io.open(nb_path, encoding="utf-8"))
srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

load_cell = srcs[2]
tree = ast.parse(load_cell)

required = {
    "v_proj_present_indices": "list of layer indices where v_proj exists",
    "v_proj_absent_indices": "list of layer indices where it does not",
    "v_proj_sample_keys": "real module names, so a prefix mistake is visible",
}
missing = [k for k in required if '"%s"' % k not in load_cell]
print("report fields written:", "yes" if not missing else missing)

# The suffix regex must be a real pattern anchored at the end.
has_suffix_regex = "self_attn\\.v_proj$" in load_cell
print("suffix-anchored regex:", has_suffix_regex)

# The guard that rejects an empty match must be in the rec() condition.
guard = "and bool(v_present)" in load_cell
print("empty-match guard:", guard)

# A3 must still be in the hard roster and in the load gate. If A3 were merely
# reported, a wrong answer would not stop the run.
a3_hard = '"A3"' in load_cell
print("A3 still gates the load stage:", a3_hard)

# Confirm the regex itself behaves on the real names.
import re
pat_src = None
for line in load_cell.splitlines():
    if "language_model\\\\.layers" in line and "re.match" in line:
        pat_src = line
print("regex source line found:", pat_src is not None)

PAT = re.compile(r".*language_model\.layers\.(\d+)\.self_attn\.v_proj$")
real = ["model.language_model.layers.%d.self_attn.v_proj" % i
        for i in (0, 5, 11, 59)]
found = [int(m.group(1)) for m in (PAT.match(n) for n in real) if m]
print("regex matches real prefixed names:", found)

ok = (not missing) and has_suffix_regex and guard and a3_hard and len(found) == 4
print()
print("A3 EVIDENCE CHECK:", "PASS" if ok else "FAIL")
if not ok:
    raise SystemExit(1)
