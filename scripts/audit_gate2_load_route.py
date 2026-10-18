"""Re-check the load cell after switching to accelerate's planner.

The earlier audit asserted on `_plan_placement`, which no longer exists. What must
hold now is stronger: the plan must come from the library, the refusal of a
host-offloading plan must happen before dispatch, and the verification of tied
weights and buffers must still be in place.
"""

import ast
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "artifacts" / "training_gate2" / "smoke.ipynb"
nb = json.loads(io.open(NB, encoding="utf-8").read())
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
full = "\n".join(code_cells)

load_cell = next(s for s in code_cells
                 if "from_pretrained" in s and "quantization_config" in s)
tree = ast.parse(load_cell)

failures = []
notes = []


def want(cond, msg):
    (notes if cond else failures).append(("PASS  " if cond else "") + msg)


# ---- the plan comes from accelerate --------------------------------------
want("from accelerate.utils import get_balanced_memory" in load_cell,
     "load cell imports accelerate's memory balancer")
want("from accelerate.utils import infer_auto_device_map" in load_cell,
     "load cell imports accelerate's device-map inference")
want("_get_balanced_memory(" in load_cell, "balanced memory is computed by accelerate")
want("_infer_auto_device_map(" in load_cell, "device map is inferred by accelerate")
want("_plan_placement" not in full, "the hand-rolled planner is gone")
want("placement_plan" not in full, "the hand-rolled plan's report key is gone")
want("dispatch_model" in load_cell, "dispatch still goes through accelerate")
# `max_memory` must now be built from the cards' real totals and feed only the
# planner; the old form pinned a fixed "18GiB" string on every card.
want('"18GiB"' not in load_cell, "the fixed 18GiB per-card cap is gone")
want("get_device_properties(i).total_memory" in load_cell,
     "the per-card budget comes from the measured total")
want("GPU_RESERVE_GIB" in load_cell, "a per-card reserve is held back")
want("no_split_module_classes" in load_cell,
     "the model's own _no_split_modules is forwarded to the planner")

# ---- host decompression, no GPU-side unpack ------------------------------
dm = None
for node in ast.walk(tree):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoModelForCausalLM"):
        for k in node.keywords:
            if k.arg == "device_map":
                dm = k.value
want(isinstance(dm, ast.Constant) and dm.value == "cpu",
     "the model is decompressed on the host")

# ---- refuse a host-offloaded plan BEFORE dispatch ------------------------
i_offload_check = load_cell.index("infer_auto_device_map offloaded")
i_dispatch = load_cell.index("model = _dispatch_model(")
want(i_offload_check < i_dispatch,
     "an offloading plan is refused before any tensor moves")
want("raise RuntimeError" in load_cell[i_offload_check - 900:i_offload_check + 200],
     "the refusal raises rather than warning")

# ---- verification stays explicit ----------------------------------------
want("tied_embeddings" in load_cell, "tied embedding storage is verified")
want("non_cuda_buffers" in load_cell, "buffer placement is verified")
want("non_cuda_params" in load_cell, "parameter placement is verified")
# A9: the tied pair gets its own hard assertion, cross-checked against accelerate's
# own tie detection. A split tie loads without error and only shows wrong logits,
# so it cannot be left as one conjunct inside A4.
want('rec("A9"' in load_cell, "A9 asserts the tied embedding pair on its own")
want("find_tied_parameters" in load_cell,
     "the tie is cross-checked with accelerate's own view")
want("is_same_storage" in load_cell, "A9 records storage identity")
want("same_device" in load_cell, "A9 asserts both halves share a device")
want("tie_seen_by_accelerate" in load_cell, "A9 requires the library to see the tie")
want('"A9"' in full, "A9 participates in the hard verdict")
want("actual_gpu_usage_after_load" in load_cell,
     "the allocator's own per-card reading is recorded")
want("planned_per_card_gib" in load_cell, "the plan's per-card bytes are derived")
want("plan_vs_actual_gib" in load_cell, "plan and measurement are compared")
want("_start_rss_sampler()" in load_cell, "host RSS is sampled during decompression")
want("_rss_ru_maxrss_gib()" in load_cell, "ru_maxrss cross-check is recorded")

# ---- fail-fast survives -------------------------------------------------
for stage in ("lora", "train", "save_reload", "vllm_start"):
    want('_stage_guard("%s")' % stage in full, "stage %r is guarded" % stage)
want("STAGE_BLOCKED_BY" in full, "each stage declares what blocks it")
want('"train"' in full and "train_failed" in full,
     "a training failure blocks adapter verification")

print("\n".join(notes))
print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("LOAD-PATH AUDIT: PASS (%d assertions)" % len(notes))
