"""Feasibility check for the accelerate placement plan, computed offline.

`scripts/prepare_gate2.py` now hands placement to
`accelerate.utils.get_balanced_memory` + `infer_auto_device_map`. Both work purely
from tensor sizes, so their arithmetic can be reproduced here on the host against
the real 31B inventory, with no model and no GPU. The point is to catch a
capacity error before it costs a run, not to predict the exact map the run will
produce -- the run's own D5 records that.

What accelerate does (accelerate/utils/modeling.py:1018-1076):

    per_gpu = module_sizes[""] // num_devices
    buffer  = 1.25 * max(largest no-split block, mean leaf size)
    per_gpu += buffer
    for idx in all_but_last: max_memory[idx] = min(per_gpu, max_memory[idx])

The last card keeps whatever budget it was given, which is what absorbs the
remainder. Tied parameters are counted once, because `compute_module_sizes` walks
`named_parameters(recurse=True)` and that defaults to `remove_duplicate=True`.
"""

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

probe = json.loads(io.open(ROOT / "artifacts" / "training_feasibility_probe.json",
                           encoding="utf-8").read())
hdr = probe["safetensors_header"]
bf16_gib = hdr["bf16_gib"]
n_layers = probe["n_layers"]

# Measured on the Kaggle L4 pool.
L4_TOTAL_GIB = 22.03
N_GPU = 4
GPU_RESERVE_GIB = 3.0

print("31B bf16 footprint (headers):  %.3f GiB" % bf16_gib)
print("measured per-card capacity:    %.2f GiB x %d" % (L4_TOTAL_GIB, N_GPU))
print("weights budget per card:       %.2f GiB (reserve %.1f GiB)"
      % (L4_TOTAL_GIB - GPU_RESERVE_GIB, GPU_RESERVE_GIB))

caps = [(L4_TOTAL_GIB - GPU_RESERVE_GIB) for _ in range(N_GPU)]
total_capacity = sum(caps)
print("total weight capacity:         %.2f GiB" % total_capacity)

# ---- reproduce get_balanced_memory -----------------------------------------
# The largest indivisible block is one text decoder layer: Gemma4TextDecoderLayer
# is in _no_split_modules. Everything else (embeddings, lm_head, vision tower,
# final norm) is smaller. Split the footprint accordingly: the vision tower is
# stored unquantized bf16 and the 410 weight_packed tensors are all text.
vision_gib = 1.7
text_gib = bf16_gib - vision_gib
per_layer_gib = text_gib / n_layers
remainder_gib = bf16_gib - per_layer_gib * n_layers

largest_no_split = per_layer_gib
mean_leaf = (text_gib + vision_gib + remainder_gib) / (n_layers + 2)
buffer_gib = 1.25 * max(largest_no_split, mean_leaf)

per_gpu = bf16_gib / N_GPU + buffer_gib
print()
print("largest no-split block:        %.4f GiB (one decoder layer)" % largest_no_split)
print("mean leaf:                     %.4f GiB" % mean_leaf)
print("accelerate buffer (1.25x):     %.4f GiB" % buffer_gib)
print("per-GPU budget it would set:   %.4f GiB" % per_gpu)

planned = [min(per_gpu, caps[i]) for i in range(N_GPU - 1)] + [caps[-1]]
print()
for i, c in enumerate(planned):
    print("GPU %d planned ceiling: %.3f GiB" % (i, c))
print("total planned ceiling:         %.3f GiB" % sum(planned))

if sum(planned) < bf16_gib:
    print()
    print("PLACEMENT FEASIBILITY: FAIL -- ceilings total %.3f GiB but the weights "
          "are %.3f GiB; the planner would offload to the host"
          % (sum(planned), bf16_gib))
    sys.exit(1)

slack = sum(planned) - bf16_gib
print()
print("slack over the weights:        %.3f GiB" % slack)
print("tightest card's headroom:      %.3f GiB above the reserve"
      % min(c - GPU_RESERVE_GIB for c in planned))

# The plan is only usable if no single irreducable unit exceeds a card's ceiling.
if largest_no_split > min(planned):
    print("PLACEMENT FEASIBILITY: FAIL -- one decoder layer (%.3f GiB) exceeds the "
          "tightest ceiling (%.3f GiB)" % (largest_no_split, min(planned)))
    sys.exit(1)

print()
print("PLACEMENT FEASIBILITY: PASS")
print("  accelerate's own budget arithmetic covers the bf16 weights on 4 x L4")
print("  with %.2f GiB of slack, and no indivisible unit is larger than a ceiling."
      % slack)
