"""Independently recompute the inference batch verdict from the raw report.

The notebook computes its own verdict. This recomputes the same judgement from the
recorded values, using tolerances stated here rather than read from the run, so a bug in
the notebook's verdict block cannot quietly certify itself.

It also refuses to treat absence of evidence as evidence of success: a check that is
missing, or whose value payload is empty where a value was required, fails.
"""

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Tolerances stated here, independently of the notebook.
C3_MIN_DELTA = 1e-3
C3_NOISE_MULTIPLE = 10.0
C5_MAX_RELATIVE_GAP = 0.05
C6_MIN_LOADED_MARKERS = 2
EXPECTED_MODULES = 120
EXPECTED_TENSORS = 240
EXPECTED_LORA_B_NONZERO = 120
EXPECTED_LORA_B_ZERO = 0

DEFAULT_REPORT = ROOT / "artifacts" / "infer_batch_v1_outputs" / "infer_batch_report.json"


def main(argv):
    report_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_REPORT
    report = json.load(io.open(report_path, encoding="utf-8"))

    checks = report.get("checks") or {}
    notes = report.get("notes") or []
    failures = []
    findings = []

    def check(cid, ok, detail):
        entry = {"id": cid, "ok": bool(ok), "detail": detail}
        findings.append(entry)
        if not ok:
            failures.append("%s: %s" % (cid, detail))
        return ok

    def val(cid):
        c = checks.get(cid)
        return c.get("value") if isinstance(c, dict) else None

    def present(cid):
        c = checks.get(cid)
        return isinstance(c, dict) and c.get("ok") is True

    # ---- the report must describe a run that happened ----------------------
    check("META.load_route",
          report.get("batch") == "gate2_infer_batch_v1",
          "report is from the expected batch")
    check("META.no_hf_model",
          "AutoModelForCausalLM" not in json.dumps(report),
          "no Hugging Face model was involved in the run")

    # ---- A: adapter identity ----------------------------------------------
    mounted = report.get("mounted_hashes") or {}
    layout = report.get("layout_hashes") or {}
    ok_a1 = present("A1") and bool(mounted)
    check("A1.mounted", ok_a1,
          "mounted hashes recorded: %r" % ({k: sorted(v) for k, v in mounted.items()},))
    # A1's own value carries the mismatch list; it must be empty, and that must be a
    # recorded empty list rather than a missing key.
    a1_val = val("A1") or {}
    check("A1.no_mismatch", a1_val.get("mismatches") == [],
          "mismatch list is present and empty: %r" % (a1_val.get("mismatches"),))

    a2_val = val("A2") or {}
    check("A2.copy", present("A2") and a2_val.get("drift") == [],
          "working copy has no drift: %r" % (a2_val.get("drift"),))

    cfg = report.get("adapter_config") or {}
    check("A3.config",
          present("A3")
          and cfg.get("r") == 8
          and cfg.get("lora_alpha") == 16
          and cfg.get("peft_type") == "LORA"
          and cfg.get("configs_identical") is True,
          "config declares r=8, alpha=16, LORA, and both files match: %r"
          % ({k: cfg.get(k) for k in ("r", "lora_alpha", "peft_type",
                                      "configs_identical")},))

    st = report.get("lora_b_stats") or {}
    st_t = st.get("trained") or {}
    st_z = st.get("zero_b") or {}
    check("A4.trained_energy",
          present("A4")
          and st_t.get("n_lora_B") == EXPECTED_MODULES
          and st_t.get("n_lora_B_nonzero") == EXPECTED_LORA_B_NONZERO
          and st_t.get("n_lora_A") == EXPECTED_MODULES,
          "trained adapter: %d/%d nonzero lora_B, %r lora_A"
          % (st_t.get("n_lora_B_nonzero", -1), st_t.get("n_lora_B", -1),
             st_t.get("n_lora_A")))
    check("A4.control_inert",
          st_z.get("n_lora_B") == EXPECTED_MODULES
          and st_z.get("n_lora_B_nonzero") == EXPECTED_LORA_B_ZERO,
          "control adapter: %d/%d nonzero lora_B"
          % (st_z.get("n_lora_B_nonzero", -1), st_z.get("n_lora_B", -1)))

    # ---- B: the official stack is what ran --------------------------------
    disc = report.get("discovered_adapters") or {}
    check("B0.discovery", set(disc) == {"trained_gate2", "zero_b_control"},
          "discovered adapters: %r" % (sorted(disc),))
    for name, info in disc.items():
        check("B0.rank.%s" % name, info.get("rank") == 8,
              "%s rank is 8 (got %r)" % (name, info.get("rank")))

    actual = report.get("vllm_cmd_actual") or {}
    check("B1.tp", actual.get("tensor_parallel_size") == "4",
          "TP size from the command: %r" % (actual.get("tensor_parallel_size"),))
    check("B1.util",
          actual.get("gpu_memory_utilization") is not None
          and abs(float(actual["gpu_memory_utilization"]) - 0.80) < 1e-6,
          "GPU util from the command: %r" % (actual.get("gpu_memory_utilization"),))
    check("B1.len", actual.get("max_model_len") == "32768",
          "max-model-len from the command: %r" % (actual.get("max_model_len"),))
    check("B1.parsers",
          actual.get("tool_call_parser") == "gemma4"
          and actual.get("reasoning_parser") == "gemma4",
          "parsers: %r / %r" % (actual.get("tool_call_parser"),
                                actual.get("reasoning_parser")))
    check("B1.chat_kwargs",
          "enable_thinking" in str(actual.get("default_chat_template_kwargs")),
          "chat template kwargs carried: %r"
          % (actual.get("default_chat_template_kwargs"),))
    # The point of the batch: enforce-eager must be provably absent from the command.
    check("B1.no_enforce_eager", actual.get("enforce_eager_present") is False,
          "enforce-eager absent from the command: %r"
          % (actual.get("enforce_eager_present"),))
    check("B2.started", present("B2") and bool(report.get("api_base")),
          "server started, api_base %r" % (report.get("api_base"),))
    check("B3.served",
          present("B3")
          and "trained_gate2" in (report.get("served_models") or [])
          and "zero_b_control" in (report.get("served_models") or []),
          "served models: %r" % (report.get("served_models"),))

    # ---- C: the measurements ----------------------------------------------
    check("C1.encodings", present("C1"),
          "equal logprob lengths across the three models: %r" % (val("C1"),))

    noise = report.get("E2_noise_floor")
    check("C2.noise_measured", noise is not None,
          "noise floor measured: %r over %r positions"
          % (noise, report.get("E2_positions")))

    d_t = report.get("C3_delta")
    d_t2 = report.get("C3_delta_second_reference")
    eff_noise = noise if (noise is not None and noise > 0.0) else 0.0
    c3_ok = (d_t is not None and d_t >= C3_MIN_DELTA
             and (eff_noise == 0.0 or d_t > C3_NOISE_MULTIPLE * eff_noise))
    check("C3.trained_moves", c3_ok,
          "trained delta %r vs noise %r (needs >= %r%s)"
          % (d_t, noise, C3_MIN_DELTA,
             "" if eff_noise == 0.0 else " and > %.0fx noise" % C3_NOISE_MULTIPLE))

    d_z = report.get("C4_delta")
    c4_ok = d_z is not None and (d_z <= eff_noise if eff_noise > 0 else d_z == 0.0)
    check("C4.control_inert", c4_ok,
          "control delta %r vs noise %r (needs <= noise, or exactly 0)"
          % (d_z, noise))

    c5_ok = (d_t is not None and d_t2 is not None and d_t2 >= C3_MIN_DELTA
             and abs(d_t - d_t2) <= C5_MAX_RELATIVE_GAP * max(d_t, d_t2))
    check("C5.reproduces", c5_ok,
          "deltas %r and %r against two base references (max gap %.0f%%)"
          % (d_t, d_t2, C5_MAX_RELATIVE_GAP * 100))

    found = report.get("C6_forbidden_found")
    markers = report.get("C6_loaded_marker_count")
    check("C6.no_forbidden", present("C6") and found == [],
          "silent-drop warnings found: %r" % (found,))
    check("C6.positive_evidence",
          isinstance(markers, int) and markers >= C6_MIN_LOADED_MARKERS,
          "adapters reported loaded: %r (needs >= %d)"
          % (markers, C6_MIN_LOADED_MARKERS))

    check("C7.generation", present("C7") and (report.get("C7_token_count") or 0) > 0,
          "generation produced %r tokens" % (report.get("C7_token_count"),))
    check("C8.tool_call",
          present("C8") and report.get("C8_trained_adapter_called_tool") is True,
          "the trained adapter produced a parsed tool call: %r"
          % (report.get("C8_trained_adapter_called_tool"),))

    # ---- scope -------------------------------------------------------------
    verdict = report.get("verdict") or {}
    check("SCOPE.inference_only", verdict.get("inference_compatible_only") is True,
          "the run declares inference-only scope")
    check("SCOPE.no_quality_claim",
          verdict.get("does_not_establish_agent_improvement") is True,
          "the run does not claim agent improvement")

    # ---- summary -----------------------------------------------------------
    skipped = [cid for cid, c in checks.items()
               if isinstance(c.get("detail"), str)
               and c["detail"].startswith("skipped:")]
    passed = sum(1 for f in findings if f["ok"])

    result = {
        "report": str(report_path),
        "kernel_version": report.get("kernel_version"),
        "wall_seconds": report.get("wall_seconds"),
        "checks_evaluated": len(findings),
        "checks_passed": passed,
        "checks_failed": len(failures),
        "skipped_in_report": sorted(skipped),
        "notebook_verdict_pass": verdict.get("pass"),
        "independent_verdict_pass": not failures,
        "failures": failures,
        "findings": findings,
        "key_numbers": {
            "c3_delta": d_t,
            "c3_delta_second_reference": d_t2,
            "c4_delta": d_z,
            "noise_floor": noise,
            "e2_positions": report.get("E2_positions"),
            "c6_loaded_markers": markers,
            "c7_tokens": report.get("C7_token_count"),
            "c7_gen_token_delta": report.get("C7_gen_token_delta"),
            "vllm_cmd_actual": actual,
            "declared_vs_actual": report.get("declared_vs_actual"),
        },
    }

    out = ROOT / "artifacts" / "infer_batch_v1_verification.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    print("INDEPENDENT VERDICT: %s" % ("PASS" if not failures else "FAIL"))
    print("  %d/%d checks passed" % (passed, len(findings)))
    if skipped:
        print("  skipped in report: %r" % (sorted(skipped),))
    print("  c3 delta %r / c4 delta %r / noise %r" % (d_t, d_z, noise))
    print("  wrote %s" % out.relative_to(ROOT))
    for f in failures:
        print("  FAIL", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
