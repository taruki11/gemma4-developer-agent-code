"""Confirm every hard-check id is actually recorded somewhere in the notebook."""
import io
import json

nb = json.load(io.open("artifacts/training_gate2/smoke.ipynb", encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
allc = "\n".join(codes)

HARD = {
    "A": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
    "B": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B7b"],
    "C": ["C1", "C2", "C3"],
    "D": ["D2", "D3", "D4"],
    "E": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
}

QUOTES = ['"', "'", '"'.replace('"', "'")]


def recorded(cid, text):
    for q in ('"', "'"):
        if "rec(%s%s%s" % (q, cid, q) in text:
            return True
        if "fail_soft(%s%s%s" % (q, cid, q) in text:
            return True
    return False


missing = []
for group, ids in HARD.items():
    parts = []
    for cid in ids:
        ok = recorded(cid, allc)
        parts.append(cid + ":" + ("ok" if ok else "MISSING"))
        if not ok:
            missing.append(cid)
    print(group, " ".join(parts))

print()
for cid in ["E5", "C3", "GC1", "D1"]:
    where = [i for i, c in enumerate(codes) if recorded(cid, c)]
    print("%-4s recorded in code cell index(es): %s" % (cid, where))

print()
print("missing:", missing)
print("RESULT:", "PASS" if not missing else "FAIL")
