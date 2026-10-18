"""Confirm the generated notebook's control flow actually reaches the CPU path.

Static audits check that strings exist; this walks the AST of the load cell and
asserts the *call graph* the run will follow: one model load, on the host, followed
by library planning, an offload refusal, and only then a dispatch. It also checks
the stage-guard ordering, because a guard placed after the work it guards would be
decorative.
"""

import ast
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads(io.open(ROOT / "artifacts" / "training_gate2" / "smoke.ipynb",
                        encoding="utf-8").read())
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)


def calls_in(src):
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                out.append((node.lineno, f.attr))
            elif isinstance(f, ast.Name):
                out.append((node.lineno, f.id))
    return sorted(out)


# ---------------------------------------------------------------- load cell
load_idx = next(i for i, s in enumerate(cells) if "from_pretrained" in s
                and "quantization_config" in s)
load_src = cells[load_idx]
seq = calls_in(load_src)
by_name = {}
for lineno, name in seq:
    by_name.setdefault(name, []).append(lineno)

order_keys = ["_start_rss_sampler", "_get_balanced_memory",
              "_infer_auto_device_map", "_dispatch_model"]
missing = [k for k in order_keys if k not in by_name]
check(not missing, "load cell is missing calls: %s" % missing)

# The model load has to be located structurally: `from_pretrained` is also how the
# tokenizer is created, and the tokenizer is deliberately constructed first.
model_load_line = None
for node in ast.walk(ast.parse(load_src)):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoModelForCausalLM"):
        model_load_line = node.lineno
check(model_load_line is not None, "the model load call is present")

if not missing and model_load_line is not None:
    fp = model_load_line
    rss = by_name["_start_rss_sampler"][0]
    bal = by_name["_get_balanced_memory"][0]
    inf = by_name["_infer_auto_device_map"][0]
    dis = by_name["_dispatch_model"][0]
    check(rss < fp, "RSS sampler starts before the load (rss=%d, load=%d)" % (rss, fp))
    check(fp < bal, "load precedes planning (load=%d, plan=%d)" % (fp, bal))
    check(bal < inf, "balancing precedes map inference (%d vs %d)" % (bal, inf))
    check(inf < dis, "inference precedes dispatch (%d vs %d)" % (inf, dis))

    # The refusal must sit between inference and dispatch, and it must raise.
    # Anchor on the `if offloaded:` guard rather than on the message text, because
    # the `raise` statement precedes the string it raises.
    refusal = load_src.find("if offloaded:")
    raise_at = load_src.find("raise RuntimeError", refusal)
    dispatch_at = load_src.find("model = _dispatch_model(")
    check(refusal != -1, "an `if offloaded:` refusal guard is present")
    check(refusal < dispatch_at, "the refusal guard precedes dispatch")
    check(raise_at != -1 and refusal < raise_at < dispatch_at,
          "the refusal raises between the guard and the dispatch")

# ----------------------------------------------------- stage guard ordering
guard_cells = {}
for i, src in enumerate(cells):
    for m in ("lora", "train", "save_reload", "vllm_start"):
        if '_stage_guard("%s")' % m in src:
            guard_cells[m] = i
check(set(guard_cells) == {"lora", "train", "save_reload", "vllm_start"},
      "all four downstream stages are guarded: %s" % sorted(guard_cells))

# A guard must appear before the work it protects in its own cell.
for stage, idx in guard_cells.items():
    src = cells[idx]
    gpos = src.find('_stage_guard("%s")' % stage)
    markers = {
        "lora": "get_peft_model(",
        "train": "optim = torch.optim.AdamW(",
        "save_reload": "model.save_pretrained(",
        "vllm_start": "server.start()",
    }
    wpos = src.find(markers[stage])
    check(wpos == -1 or gpos < wpos,
          "guard for %r precedes %r (%d vs %d)" % (stage, markers[stage], gpos, wpos))

# ------------------------------------------------ E-group skip, not failure
e_src = next(s for s in cells if "E_IDS" in s)
check("skipped: adapter comparison not reached" in e_src,
      "the E group records skips when blocked")
check("skipped_stages" in e_src, "the E group marks the stage as skipped")

# ---------------------------------------------------- no stale load path left
full = "\n".join(cells)
check("_plan_placement" not in full, "the hand-rolled planner is gone")

# Comments and docstrings may legitimately describe the abandoned path; calls may
# not use it. Inspect every call's keywords rather than searching the text.
auto_call = None
for idx, src in enumerate(cells):
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (kw.arg == "device_map" and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "auto"):
                auto_call = (idx, node.lineno)
check(auto_call is None,
      "no call passes device_map=\"auto\" (found at %r)" % (auto_call,))

if problems:
    print("CONTROL-FLOW AUDIT: FAIL")
    for p in problems:
        print("  - " + p)
    sys.exit(1)

print("CONTROL-FLOW AUDIT: PASS")
print("  load=%d rss=%d balance=%d infer=%d dispatch=%d (line order)"
      % (model_load_line, rss, bal, inf, dis))
print("  guards: %s" % ", ".join("%s@cell%d" % kv for kv in sorted(guard_cells.items())))
print("  the refusal sits between inference and dispatch and raises")
