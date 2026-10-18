"""Zero-GPU validation of the two-stage layout: build it, run the official static
validators, then compile it with the real adk_submission compiler using mock registries.

Asserts: root is SequentialAgent with [fixer, auditor]; the fixer has no submit_patch tool;
the auditor has submit_patch; both use the declared model.
Writes artifacts/twostage_layout_validation.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from twostage_layout import build_layout  # noqa: E402

from adk_submission import compile_submission  # noqa: E402
from adk_submission.registry import ModelRegistry  # noqa: E402

TOOL_NAMES = ["run_command", "read_file", "edit_file", "write_file", "get_status",
              "submit_patch", "get_code_neighbors", "search_similar_code", "get_code_subgraph"]
MODEL_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"


def dummy_tool(name):
    def fn(**kwargs):
        return {"status": "ok"}
    fn.__name__ = name
    return fn


report = {}
sample = Path("data/official/sample_submission")
destination = Path("artifacts/twostage_layout")
report["layout"] = build_layout(destination, sample)

models = ModelRegistry()
models.register(MODEL_ALIAS, "dummy-base-model")
models.register(f"adapter:main_lora", "dummy-base-model:main_lora")
models.register(f"adapter:tool_lora", "dummy-base-model:tool_lora")

tools = {name: dummy_tool(name) for name in TOOL_NAMES}
try:
    root = compile_submission(submission_dir=destination, tool_registry=tools, model_registry=models)
    report["compile_ok"] = True
except Exception as exc:  # noqa: BLE001
    report["compile_ok"] = False
    report["compile_error"] = f"{type(exc).__name__}: {exc}"
    Path("artifacts/twostage_layout_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(1)

report["root_class"] = type(root).__name__
report["root_name"] = root.name
subs = {a.name: a for a in root.sub_agents}
report["sub_agents"] = sorted(subs)
def tool_name(t):
    return getattr(t, "name", None) or getattr(t, "__name__", type(t).__name__)


report["fixer_tools"] = sorted(tool_name(t) for t in subs["fixer"].tools)
report["auditor_tools"] = sorted(tool_name(t) for t in subs["auditor"].tools)
report["fixer_has_submit"] = "submit_patch" in report["fixer_tools"]
report["auditor_has_submit"] = "submit_patch" in report["auditor_tools"]
report["fixer_model"] = str(getattr(subs["fixer"], "model", None))
report["auditor_model"] = str(getattr(subs["auditor"], "model", None))
report["fixer_instruction_chars"] = len(subs["fixer"].instruction)
report["auditor_instruction_chars"] = len(subs["auditor"].instruction)

assert report["root_class"] == "SequentialAgent", report["root_class"]
assert report["sub_agents"] == ["auditor", "fixer"], report["sub_agents"]
assert not report["fixer_has_submit"], "fixer must NOT have submit_patch"
assert report["auditor_has_submit"], "auditor must have submit_patch"
assert "main_lora" in report["fixer_model"], report["fixer_model"]

Path("artifacts/twostage_layout_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))