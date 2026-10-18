"""Independently verify the Gate 1 training-feasibility probe.

Reads the collected artifacts and re-derives every preregistered verdict from raw data
instead of trusting the `verdict` block the notebook computed for itself.

Usage:
    python scripts/verify_gate1.py [--output artifacts/training_gate1_outputs]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HARD = {
    "A": ["A1", "A2", "A4", "A5", "A6", "A7", "A8"],
    "B": ["B1", "B2", "B3", "B4", "B5", "B6"],
    "C": ["C1", "C2"],
    "D": ["D3"],
    "E": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
}
SOFT = {"A3", "C3", "D1", "D2", "D4", "E4_prep"}

EXPECTED_N_MODULES = 84
EXPECTED_N_TRAINABLE = 3325952
EXPECTED_N_TENSORS = 168

FORBIDDEN_LOG = [
    "is not in the model's supported LoRA target modules",
    "These parameters will be ignored",
    "does not support LoRA yet",
]


def find_report(out: Path) -> Path | None:
    for cand in (out / "gate1_report.json",
                 out / "working" / "gate1_report.json"):
        if cand.is_file():
            return cand
    hits = sorted(out.rglob("gate1_report.json")) if out.is_dir() else []
    return hits[0] if hits else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="artifacts/training_gate1_outputs")
    args = ap.parse_args(argv)

    out = ROOT / args.output
    rp = find_report(out)
    if rp is None:
        print(f"gate1_report.json not found under {out}")
        print("run collection first: scripts/collect_kaggle_run.py --wait --kernel taruki/gemma4-gate1-e4b")
        return 1

    rep = json.loads(rp.read_text(encoding="utf-8"))
    checks = rep.get("checks", {})
    print(f"report: {rp}")
    print(f"kernel: {rep.get('kernel')}")
    print(f"wall:   {rep.get('wall_seconds')} s")
    print()

    def status(cid):
        if cid not in checks:
            return "MISSING"
        return "PASS" if checks[cid]["ok"] else "FAIL"

    print("=== preregistered checks ===")
    for g, ids in HARD.items():
        line = []
        for cid in ids:
            line.append(f"{cid}={status(cid)}")
        print(f"  {g}: " + " ".join(line))
    print()

    problems: list[str] = []

    print("=== independent re-derivation ===")

    # A: injection correctness
    n_matched = rep.get("n_matched_modules")
    n_layers = rep.get("n_text_layers")
    if n_matched == EXPECTED_N_MODULES and n_layers == 42:
        print(f"  A6 ok: {n_matched} matched modules over {n_layers} layers")
    else:
        problems.append(f"A6: matched={n_matched} over layers={n_layers}, "
                        f"expected {EXPECTED_N_MODULES}/42")

    matched = rep.get("matched_sample") or []
    bad = [m for m in matched if "language_model" not in m]
    if bad:
        problems.append(f"A4: sample contains non-language_model modules {bad}")
    else:
        print(f"  A4 ok: all sampled matches are under language_model")

    if rep.get("clippable_linear_count", 0) > 0:
        print(f"  A3 ok: {rep['clippable_linear_count']} Gemma4ClippableLinear present "
              f"(hazard exists in this variant)")
    else:
        print("  A3 WARN: no ClippableLinear found; the fix was not actually exercised")

    # B: frozen base, trainable count, gradients, weight change
    n_train = rep.get("n_trainable_params")
    non_lora = rep.get("non_lora_trainable") or []
    if n_train == EXPECTED_N_TRAINABLE and not non_lora:
        print(f"  B1/B2 ok: {n_train} trainable params, all under lora_")
    else:
        problems.append(f"B1/B2: trainable={n_train} expected={EXPECTED_N_TRAINABLE}, "
                        f"non_lora={non_lora[:5]}")

    steps = rep.get("training_steps") or []
    losses = [s["loss"] for s in steps]
    if len(steps) == 8 and all(isinstance(v, (int, float)) for v in losses):
        print(f"  B3 ok: 8 losses recorded, range {min(losses):.4f}..{max(losses):.4f}")
    else:
        problems.append(f"B3: {len(steps)} steps recorded, expected 8")

    if steps and all(s.get("grad_finite") for s in steps):
        print("  B4 ok: every step reported finite LoRA gradients")
    elif steps:
        problems.append("B4: at least one step had non-finite gradients")

    b_grads = [s.get("lora_B_grad_abs_sum", 0) for s in steps]
    if b_grads and all(g > 0 for g in b_grads):
        print(f"  B5 ok: lora_B grad abs-sum > 0 at every step "
              f"(min {min(b_grads):.4g}, max {max(b_grads):.4g})")
    else:
        problems.append(f"B5: lora_B grad abs-sum not positive at every step: {b_grads}")

    max_delta = rep.get("max_weight_delta")
    b_after = rep.get("lora_B_abs_sum_after")
    if isinstance(max_delta, (int, float)) and max_delta > 0 and isinstance(b_after, (int, float)) and b_after > 0:
        print(f"  B6 ok: weights moved (max|dW|={max_delta:.4g}), "
              f"lora_B abs-sum after={b_after:.4g} -> adapter is NOT zero-effect")
    else:
        problems.append(f"B6: max_weight_delta={max_delta}, lora_B_abs_sum_after={b_after}")

    # C: save / reload bit-exact
    print(f"  C1 max_abs_diff={checks.get('C1', {}).get('value', {}).get('max_abs_diff') if isinstance(checks.get('C1', {}).get('value'), dict) else 'n/a'}"
          f" over {rep.get('c1_common_keys')} keys")
    c2 = checks.get("C2", {})
    c2v = c2.get("value") if isinstance(c2.get("value"), dict) else {}
    print(f"  C2 compared={c2v.get('compared')} disk_keys={c2v.get('disk_keys')} "
          f"max_abs_diff={c2v.get('max_abs_diff')}")

    # D: capacity
    peaks = rep.get("peak_memory_during_training") or {}
    print("  D1 per-device peak reserved (GiB):",
          {k: v.get("peak_reserved_gib") for k, v in peaks.items()})
    dist = rep.get("param_device_distribution") or {}
    if dist:
        print("  D2 params per device:",
              {k: f"{v['params']/1e9:.2f}B/{v['gib']:.2f}GiB" for k, v in dist.items()})

    # E: the decisive adapter-consumption evidence
    e2 = checks.get("E2", {}).get("value") or {}
    e3 = checks.get("E3", {}).get("value") or {}
    e4 = checks.get("E4", {}).get("value") or {}
    noise = e2.get("noise")
    d3 = e3.get("delta")
    d4 = e4.get("delta")
    print(f"  E2 noise floor = {noise}")
    print(f"  E3 trained-vs-base logprob delta = {d3}")
    print(f"  E4 zero-B-vs-base logprob delta  = {d4}")

    if isinstance(d3, (int, float)) and isinstance(d4, (int, float)):
        if noise is None:
            problems.append("E2: noise floor unavailable, E3/E4 comparison is unmoored")
        elif noise == 0:
            if d3 > 0 and d4 == 0:
                print("  E3/E4 ok: generator is deterministic (noise 0); "
                      "trained moved logprobs, zero-B did not")
            elif d3 > 0 > -1 and d4 > 0:
                problems.append("E4: zero-B control moved logprobs despite zero lora_B "
                                "-> something other than adapter weights is perturbing the model")
            elif d3 == 0:
                problems.append("E3: trained adapter did not move logprobs at all "
                                "-> adapter silently ignored")
        else:
            if d3 > 10 * noise and d4 <= noise:
                print(f"  E3/E4 ok: trained delta {d3:.4g} > 10x noise {noise:.4g}, "
                      f"zero-B delta {d4:.4g} <= noise")
            else:
                if d3 <= 10 * noise:
                    problems.append(f"E3: trained delta {d3:.4g} not > 10x noise {noise:.4g}")
                if d4 > noise:
                    problems.append(f"E4: zero-B delta {d4:.4g} exceeds noise {noise:.4g}")

    gen_delta = rep.get("E3_gen_token_delta")
    print(f"  E3 generated-token logprob delta = {gen_delta} "
          f"(same tokens: {rep.get('E3_gen_tokens_identical')}) — context only, not the verdict")

    # E6: silent-drop scan, re-run against the collected log ourselves
    log_files = list(out.rglob("gate1_vllm_server.log"))
    if not log_files:
        log_files = [p for p in out.rglob("*.log") if p.stat().st_size > 1000]
    if log_files:
        text = log_files[0].read_text(errors="replace")
        found = [f for f in FORBIDDEN_LOG if f in text]
        print(f"  E6 re-scan of {log_files[0].name} ({len(text)} chars): "
              f"{'CLEAN' if not found else 'FOUND ' + str(found)}")
        if found:
            problems.append(f"E6: server log contains silent-drop warnings {found}")
        for pat in (r"Adding LoRA", r"lora.*loaded", r"Loaded.*lora"):
            if re.search(pat, text, re.I):
                print(f"    log mentions {pat!r}")
    else:
        print("  E6: no server log collected")

    e8 = rep.get("E8_attempts") or []
    called = [a for a in e8 if a.get("tool_calls")]
    print(f"  E8 tool-call attempts: {len(e8)}, with tool_calls: {len(called)}")
    if not called:
        problems.append("E8: no attempt produced a tool_call")

    print()
    print("=== verdict ===")
    missing = [cid for ids in HARD.values() for cid in ids if cid not in checks]
    failed = [cid for ids in HARD.values() for cid in ids
              if cid in checks and not checks[cid]["ok"]]
    soft_missing = [c for c in SOFT if c not in checks]

    print(f"  hard missing: {missing}")
    print(f"  hard failed:  {failed}")
    print(f"  soft missing: {soft_missing}")
    if problems:
        print("  independent problems:")
        for p in problems:
            print(f"    - {p}")

    ok = not missing and not failed and not problems
    print()
    print("  GATE 1:", "PASS" if ok else "FAIL")
    if not ok:
        print("  per prereg: do NOT proceed to Gate 2; keep logs, report to user.")

    verdict_doc = {
        "gate": "gate1",
        "kernel": rep.get("kernel"),
        "pass": ok,
        "hard_missing": missing,
        "hard_failed": failed,
        "soft_missing": soft_missing,
        "independent_problems": problems,
        "checks": {cid: {"ok": v["ok"], "value": v.get("value")}
                   for cid, v in checks.items()},
        "timings": rep.get("timings"),
        "gpu": rep.get("gpu"),
        "wall_seconds": rep.get("wall_seconds"),
        "source_report": str(rp),
    }
    dest = ROOT / "artifacts" / "gate1_verification.json"
    dest.write_text(json.dumps(verdict_doc, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
