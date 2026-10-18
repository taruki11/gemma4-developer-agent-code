"""Verify execution order inside the training cell: B6 must be measured last."""
import io
import re

src = io.open("scripts/prepare_gate2.py", encoding="utf-8").read()
start = src.find("CELL_5_CODE")
end = src.find("CELL_6 =", start)
body = src[start:end]

marks = [
    "_reset_peaks()",
    "phase 2",
    "per_tensor = ",
    "rec(\"B6\"",
    "rec(\"B7\"",
    "model.eval()",
    "report[\"long_step\"]",
    "report[\"max_weight_delta\"]",
]
pos = {}
for m in marks:
    pos[m] = body.find(m)

for m in marks:
    print("%-28s at %s" % (m, pos[m]))

print()
# weight-delta capture must come after the optimiser has stepped at both lengths,
# otherwise B6 measures only the short phase.
ok_delta = pos["per_tensor = "] > pos["phase 2"]
ok_b6 = pos["rec(\"B6\""] > pos["per_tensor = "]
print("weight delta measured after long phase:", ok_delta)
print("B6 recorded after delta:", ok_b6)

# no negative positions
bad = [m for m, p in pos.items() if p < 0]
print("missing markers:", bad)
print()
print("RESULT:", "PASS" if ok_delta and ok_b6 and not bad else "FAIL")
