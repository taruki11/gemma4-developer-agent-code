"""Exercise verify_infer_batch_v1.py against hand-built reports.

The verifier must fail on a report that is merely missing its evidence, not only on one
with bad numbers. That is exactly the bug the earlier training verifier had, where
`not (x or [])` made a never-run check pass. Each scenario below is a report that a
plausible bug or an aborted run would produce.
"""

import copy
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify_infer_batch_v1.py"


def base_report():
    """A report that should pass: every check present and satisfied."""
    return {
        "batch": "gate2_infer_batch_v1",
        "kernel_version": 1,
        "wall_seconds": 500.0,
        "api_base": "http://127.0.0.1:8000/v1",
        "served_models": ["/mnt/model", "trained_gate2", "zero_b_control"],
        "mounted_hashes": {"trained_gate2": {"adapter_model.safetensors": {}},
                           "zero_b_control": {"adapter_model.safetensors": {}}},
        "layout_hashes": {"trained_gate2": {"adapter_model.safetensors": {}},
                          "zero_b_control": {"adapter_model.safetensors": {}}},
        "adapter_config": {"r": 8, "lora_alpha": 16, "peft_type": "LORA",
                           "configs_identical": True},
        "lora_b_stats": {"trained": {"n_lora_A": 120, "n_lora_B": 120,
                                     "n_lora_B_nonzero": 120},
                         "zero_b": {"n_lora_A": 120, "n_lora_B": 120,
                                    "n_lora_B_nonzero": 0}},
        "discovered_adapters": {"trained_gate2": {"rank": 8},
                                "zero_b_control": {"rank": 8}},
        "vllm_cmd_actual": {"tensor_parallel_size": "4",
                            "gpu_memory_utilization": "0.8",
                            "max_model_len": "32768",
                            "tool_call_parser": "gemma4",
                            "reasoning_parser": "gemma4",
                            "default_chat_template_kwargs": '{"enable_thinking": true}',
                            "enable_lora": True,
                            "enable_auto_tool_choice": True,
                            "enforce_eager_present": False},
        "E2_noise_floor": 0.0,
        "E2_positions": 336,
        "C3_delta": 0.42,
        "C3_delta_second_reference": 0.421,
        "C4_delta": 0.0,
        "C6_forbidden_found": [],
        "C6_loaded_marker_count": 2,
        "C7_token_count": 16,
        "C8_trained_adapter_called_tool": True,
        "checks": {
            "A1": {"ok": True, "value": {"mismatches": []}, "detail": "..."},
            "A2": {"ok": True, "value": {"drift": []}, "detail": "..."},
            "A3": {"ok": True, "value": {}, "detail": "..."},
            "A4": {"ok": True, "value": {}, "detail": "..."},
            "B0": {"ok": True, "value": {}, "detail": "..."},
            "B1": {"ok": True, "value": {}, "detail": "..."},
            "B2": {"ok": True, "value": {}, "detail": "..."},
            "B3": {"ok": True, "value": {}, "detail": "..."},
            "C1": {"ok": True, "value": {"base": 336, "trained": 336,
                                         "zero_b": 336}, "detail": "..."},
            "C2": {"ok": True, "value": {}, "detail": "..."},
            "C3": {"ok": True, "value": {}, "detail": "..."},
            "C4": {"ok": True, "value": {}, "detail": "..."},
            "C5": {"ok": True, "value": {}, "detail": "..."},
            "C6": {"ok": True, "value": {}, "detail": "..."},
            "C7": {"ok": True, "value": {}, "detail": "..."},
            "C8": {"ok": True, "value": {}, "detail": "..."},
        },
        "verdict": {"pass": True, "inference_compatible_only": True,
                    "does_not_establish_agent_improvement": True},
    }


SCENARIOS = []


def scenario(name, mutate, expect_fail):
    SCENARIOS.append((name, mutate, expect_fail))


scenario("healthy report passes", lambda r: r, False)

scenario("control moved (adapter not inert)",
         lambda r: r.update({"C4_delta": 0.31}), True)

scenario("trained adapter did nothing",
         lambda r: r.update({"C3_delta": 0.0, "C3_delta_second_reference": 0.0}), True)

scenario("enforce-eager was actually used",
         lambda r: r["vllm_cmd_actual"].update({"enforce_eager_present": True}), True)

scenario("only one adapter reported loaded",
         lambda r: r.update({"C6_loaded_marker_count": 1}), True)

scenario("silent-drop warning present",
         lambda r: r.update({"C6_forbidden_found": [
             "These parameters will be ignored"]}), True)

scenario("tool call came only from the base model",
         lambda r: r.update({"C8_trained_adapter_called_tool": False}), True)

scenario("lora_B stats missing entirely (never ran)",
         lambda r: r.update({"lora_b_stats": {}}), True)

scenario("mounted hashes missing, A1 claims ok",
         lambda r: r.update({"mounted_hashes": {}}), True)

scenario("A1 value has no mismatch key (empty-value pass attempt)",
         lambda r: r["checks"]["A1"].update({"value": {}}), True)

scenario("noise floor missing",
         lambda r: r.update({"E2_noise_floor": None}), True)

scenario("control delta null instead of zero",
         lambda r: r.update({"C4_delta": None}), True)

scenario("run was blocked before comparison (all C skipped)",
         lambda r: (r.update({"api_base": None, "C3_delta": None, "C4_delta": None,
                              "C6_loaded_marker_count": None,
                              "C8_trained_adapter_called_tool": None}),
                    [r["checks"].__setitem__(
                        cid, {"ok": False, "value": None,
                              "detail": "skipped: serving did not start"})
                     for cid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")]),
         True)


def main():
    failures = []
    for name, mutate, expect_fail in SCENARIOS:
        report = base_report()
        mutate(report)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            json.dump(report, fh)
            path = fh.name
        proc = subprocess.run([sys.executable, str(VERIFIER), path],
                              capture_output=True, text=True)
        got_fail = proc.returncode != 0
        ok = got_fail == expect_fail
        print("%s %-52s expected %s got %s"
              % ("ok  " if ok else "FAIL", name,
                 "FAIL" if expect_fail else "PASS",
                 "FAIL" if got_fail else "PASS"))
        if not ok:
            failures.append(name)
            print(proc.stdout)
            print(proc.stderr)
        Path(path).unlink()

    print()
    if failures:
        print("VERIFIER TEST: FAIL (%d)" % len(failures))
        return 1
    print("VERIFIER TEST: PASS (%d scenarios)" % len(SCENARIOS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
