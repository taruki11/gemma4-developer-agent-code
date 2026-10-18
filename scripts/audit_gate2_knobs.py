"""Verify the exact memory/launch knobs landed in the generated notebook."""
import io
import json
import re
from pathlib import Path

nb = json.load(io.open("artifacts/training_gate2/smoke.ipynb", encoding="utf-8"))
code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

checks = {
    # Load path: the whole point of this revision is that decompression happens on
    # the host and placement is planned from measured sizes. The old squeeze knob
    # (`max_memory` at 85%) must be gone, not merely tolerated.
    "load on host": 'device_map="cpu"' in code,
    # `max_memory` now legitimately feeds accelerate's planner (it is the budget the
    # planner sizes against). What must be absent is the old squeeze-the-cap usage:
    # a bare percentage of total memory handed to from_pretrained.
    "no max_memory squeeze": "totals[i] * 0.85" not in code,
    "planner is accelerate": "_infer_auto_device_map(" in code,
    "balanced budget from accelerate": "_get_balanced_memory(" in code,
    "per-card reserve": "GPU_RESERVE_GIB = 3.0" in code,
    "dispatch via accelerate": "from accelerate import dispatch_model" in code,
    "buffer placement checked": "non_cuda_buffers" in code,
    "tied embeddings checked": "tied_embeddings" in code,
    "rss sampled": "_start_rss_sampler()" in code,
    "rss ru_maxrss cross-check": "_rss_ru_maxrss_gib()" in code,
    # Fail-fast.
    "stage guard": "_stage_guard(" in code,
    "load_failed flag": 'report["load_failed"]' in code,
    "train_failed flag": 'report["train_failed"]' in code,
    "skips not failures": "except StageSkipped" in code,
    # Timing constants, interpolated from the generator rather than repeated.
    "cap 7200": "RUNTIME_CAP_SECONDS = 7200" in code,
    "soft deadline 4800": "SOFT_DEADLINE = T0 + 4800" in code,
    "server reserve 1500": "SERVER_DEADLINE = T0 + (RUNTIME_CAP_SECONDS - 1500)" in code,
    "no leftover placeholder": "__RUNTIME_CAP_SECONDS__" not in code
                              and "__SOFT_DEADLINE_SECONDS__" not in code
                              and "__SERVER_RESERVE__" not in code
                              and "__GPU_RESERVE_GIB__" not in code,
    # Everything the earlier revision established and must not lose.
    "D4 threshold 9.0": "min_free >= 9.0" in code,
    "rec returns bool": "return bool(ok)" in code,
    "start ladder tries 32768": "32768" in code,
    "start ladder falls to 4096": "4096" in code,
    "enforce-eager": "--enforce-eager" in code,
    "grad ckpt kwargs": '"use_reentrant": False' in code,
    "LONG_SEQ_LEN 2048": "LONG_SEQ_LEN = 2048" in code,
    "OPTIMIZER AdamW": "AdamW(" in code,
    "zero_grad set_to_none": "set_to_none=True" in code,
    "domain: 31b slug": "gemma-4-31b-it-qat-w4a16-ct" in code,
    "torchao neutralised": "dispatch_torchao" in code,
    "quant status normalise": 'rsplit(".", 1)[-1].lower()' in code,
    "report gate2": "gate2_report.json" in code,
    "HARD groups B7b": '"B7b"' in code,
    "E9 second ref": '"E9"' in code,
}
bad = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("OK   " if v else "MISS ") + k)
print()
print("FAILED:", bad)

# Confirm no stale Gate 1 identifiers leaked in
stale = []
for needle in ["gate1_report", "gate1_adapter", "gate1_layout", "E4B_SLUG",
               "3325952", "gate1_vllm_server"]:
    if needle in code:
        stale.append(needle)
print("stale Gate 1 identifiers:", stale)
print()
print("RESULT:", "PASS" if not bad and not stale else "FAIL")
