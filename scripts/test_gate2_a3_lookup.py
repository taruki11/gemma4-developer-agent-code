"""Test the A3 lookup against the real names the model produces.

The bug it fixes was a silent one: every layer looked like it was missing v_proj, so
the check "passed" a comparison between two full sets. This builds the exact module
names the 31B yields (with the `model.` prefix) and confirms the fixed lookup finds
v_proj on the sliding layers and not on the full ones -- and that an empty match is
rejected rather than treated as "absent everywhere".
"""

import re
import sys

FULL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
N_LAYERS = 60
PAT = re.compile(r".*language_model\.layers\.(\d+)\.self_attn\.v_proj$")


def names_for(prefix, have_v_proj):
    out = []
    for i in range(N_LAYERS):
        out.append("%slanguage_model.layers.%d.self_attn.q_proj" % (prefix, i))
        out.append("%slanguage_model.layers.%d.self_attn.k_proj" % (prefix, i))
        out.append("%slanguage_model.layers.%d.self_attn.o_proj" % (prefix, i))
        if i in have_v_proj:
            out.append("%slanguage_model.layers.%d.self_attn.v_proj" % (prefix, i))
    return out


def lookup(module_names):
    return sorted(int(m.group(1)) for m in
                  (PAT.match(n) for n in module_names) if m)


failures = []

# The real shape: v_proj on every sliding layer, absent on every full layer.
sliding = [i for i in range(N_LAYERS) if i not in FULL]
got = lookup(names_for("model.", sliding))
absent = [i for i in range(N_LAYERS) if i not in set(got)]
print("with 'model.' prefix -> found %d, absent %r" % (len(got), absent))
if got != sliding:
    failures.append("searching for v_proj found %r, expected the 50 sliding "
                    "layers" % got)
if absent != FULL:
    failures.append("absent set is %r, expected the 10 full layers" % absent)

# The bug: a hard-coded key without the prefix finds nothing at all.
old_hits = [i for i in range(N_LAYERS)
            if ("language_model.layers.%d.self_attn.v_proj" % i)
            in set(names_for("model.", sliding))]
print("old exact-key lookup -> found %d (the defect: 0)" % len(old_hits))
if old_hits:
    failures.append("the old lookup should have found nothing; if it now finds "
                    "layers, the regression is not what is being tested")

# Guard: no matches must not be read as "absent everywhere".
empty = lookup([])
print("no modules at all -> found %d, bool(v_present)=%s" % (len(empty), bool(empty)))
if empty:
    failures.append("an empty model should yield no matches")
if bool(empty):
    failures.append("A3's bool(v_present) guard would not reject an empty match")

print()
if failures:
    print("A3 LOOKUP TEST: FAIL")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("A3 LOOKUP TEST: PASS (finds 50 sliding layers, absent exactly on the 10 full)")
