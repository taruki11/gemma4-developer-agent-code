"""Independently verify the Gate 2 (31B) training-feasibility probe.

Reads the collected artefacts and re-derives every preregistered verdict from the
raw measurements rather than trusting the `verdict` block the notebook computed for
itself. The notebook's own verdict is printed for comparison, never used as input.

Usage:
    python scripts/verify_gate2.py [--output artifacts/training_gate2_outputs_v2]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Pre-registered roster (artifacts/gate2_prereg.md).
HARD = {
    "A": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
    "B": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B7b"],
    "C": ["C1", "C2", "C3"],
    "D": ["D2", "D3", "D4"],
    "E": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
}
SOFT = {"D1", "D5", "GC1", "E4_prep"}

# Derived from the 31B text config: 50 sliding + 10 full layers, q+o, r=8.
EXPECTED_N_MODULES = 120
EXPECTED_N_TRAINABLE = 14336000
EXPECTED_N_TENSORS = 240
EXPECTED_FULL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
SLIDING_PER_LAYER = 2 * 8 * (5376 + 32 * 256)     # 217,088
FULL_PER_LAYER = 2 * 8 * (5376 + 32 * 512)        # 348,160

FORBIDDEN_LOG = [
    "is not in the model's supported LoRA target modules",
    "These parameters will be ignored",
    "does not support LoRA yet",
]

# Pre-registered tolerances.
E3_MIN_DELTA = 1e-3
E3_NOISE_MULTIPLE = 10.0
E4_MAX_RELATIVE = 0.05
LONG_SEQ_TARGET = 2048
GPU_RESERVE_GIB = 3.0


def find_report(out: Path) -> Path | None:
    for cand in (out / "gate2_report.json", out / "working" / "gate2_report.json"):
        if cand.is_file():
            return cand
    hits = sorted(out.rglob("gate2_report.json")) if out.is_dir() else []
    return hits[0] if hits else None


def find_log(out: Path) -> Path | None:
    for pat in ("*vllm_server.log", "gate2_vllm_server.log"):
        hits = sorted(out.rglob(pat))
        if hits:
            return hits[0]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="artifacts/training_gate2_outputs_v2")
    args = ap.parse_args(argv)

    out = ROOT / args.output
    rp = find_report(out)
    if rp is None:
        print("gate2_report.json not found under %s" % out)
        print("run collection first: scripts/collect_kaggle_run.py "
              "--kernel taruki/gemma4-gate2-31b --output <dir>")
        return 1

    rep = json.loads(rp.read_text(encoding="utf-8"))
    checks = rep.get("checks", {})
    print("report:", rp)
    print("kernel:", rep.get("kernel"))
    print("wall:  ", rep.get("wall_seconds"), "s")
    print("load_route:", rep.get("load_route"))
    print()

    def status(cid):
        if cid not in checks:
            return "MISSING"
        detail = str(checks[cid].get("detail") or "")
        if detail.startswith("skipped:"):
            return "SKIPPED"
        return "PASS" if checks[cid].get("ok") else "FAIL"

    print("=== preregistered checks ===")
    for group, ids in HARD.items():
        print("  %s: %s" % (group, " ".join("%s=%s" % (c, status(c)) for c in ids)))
    print()

    problems = []
    notes = []

    def want(cond, ok_msg, bad_msg):
        if cond:
            notes.append(ok_msg)
        else:
            problems.append(bad_msg)

    print("=== independent re-derivation ===")

    # ---- load route and decompression ---------------------------------------
    want(rep.get("load_route") == "cpu_decompress_then_dispatch",
         "route ok: model was decompressed on the host",
         "route is %r, not the host-decompression path"
         % rep.get("load_route"))

    rss = rep.get("cpu_load_rss_peak_gib")
    ru = rep.get("cpu_load_ru_maxrss_gib")
    secs = rep.get("cpu_load_seconds")
    if rss is not None and secs is not None:
        # A 60.9 GiB bf16 model plus ~21.7 GiB of packed int32 cannot be
        # decompressed in less than the bf16 footprint alone.
        want(rss >= 60.0,
             "RSS peak %.2f GiB over %.1fs is consistent with holding the "
             "decompressed 60.9 GiB model" % (rss, secs),
             "RSS peak %.2f GiB is below the 60.9 GiB bf16 footprint -- the "
             "measurement cannot have covered the whole decompression" % rss)
        if ru is not None and ru > rss + 5.0:
            notes.append("note: ru_maxrss %.2f GiB exceeds sampled peak %.2f GiB"
                         % (ru, rss))
    else:
        problems.append("no RSS measurement recorded")

    # ---- A2: weights really are bf16 ---------------------------------------
    want(rep.get("layer0_q_proj_dtype") == "torch.bfloat16",
         "A2 ok: layer0 q_proj is bf16",
         "A2: layer0 q_proj dtype is %r" % rep.get("layer0_q_proj_dtype"))
    statuses = rep.get("quantization_status_counts") or {}
    want(statuses.get("decompressed", 0) > 0 and statuses.get("compressed", 0) == 0,
         "A2 ok: %d modules decompressed, 0 still compressed"
         % statuses.get("decompressed", 0),
         "A2: status counts %r" % statuses)

    # ---- A3: k_eq_v and the v_proj gap -------------------------------------
    want(rep.get("attention_k_eq_v") is True,
         "A3 ok: attention_k_eq_v is True, as the 31B config states",
         "A3: attention_k_eq_v is %r" % rep.get("attention_k_eq_v"))
    absent = rep.get("v_proj_absent_indices")
    want(absent == EXPECTED_FULL_LAYERS,
         "A3 ok: v_proj absent on exactly the 10 full layers %r" % (absent,),
         "A3: v_proj absent on %r, expected %r" % (absent, EXPECTED_FULL_LAYERS))

    # ---- A4 / A9: placement and the tie ------------------------------------
    pdev = rep.get("param_devices")
    non_cuda_p = rep.get("non_cuda_param_devices")
    if isinstance(pdev, dict) and pdev and isinstance(non_cuda_p, list):
        want(not non_cuda_p,
             "A4 ok: all %d parameters on %d GPU device(s)"
             % (sum(pdev.values()), len(pdev)),
             "A4: non-CUDA parameter devices %r" % (non_cuda_p,))
    else:
        problems.append("A4: param_devices=%r non_cuda=%r -- placement never "
                        "verified" % (pdev, non_cuda_p))
    bdev = rep.get("buffer_devices")
    non_cuda_b = rep.get("non_cuda_buffer_devices")
    if isinstance(bdev, dict) and bdev and isinstance(non_cuda_b, list):
        want(not non_cuda_b,
             "A4 ok: all %d buffers on GPU device(s)" % sum(bdev.values()),
             "A4: non-CUDA buffer devices %r" % (non_cuda_b,))
    else:
        problems.append("A4: buffer_devices=%r non_cuda=%r -- buffer placement "
                        "never verified" % (bdev, non_cuda_b))
    bad_map = {k: v for k, v in (rep.get("device_map") or {}).items()
               if str(v).startswith(("cpu", "disk"))}
    want(not bad_map, "A4 ok: no module offloaded to host/disk",
         "A4: offloaded modules %r" % (bad_map,))

    tie = rep.get("tied_check") or {}
    want(tie.get("is_same_storage") is True,
         "A9 ok: lm_head.weight is embed_tokens.weight (one storage)",
         "A9: is_same_storage=%r" % tie.get("is_same_storage"))
    want(tie.get("same_device") is True,
         "A9 ok: both halves on one device %r" % (tie.get("devices"),),
         "A9: devices %r" % (tie.get("devices"),))
    want(tie.get("tie_seen_by_accelerate") is True,
         "A9 ok: accelerate's find_tied_parameters sees the pair",
         "A9: accelerate did not report the tie: %r"
         % (tie.get("accelerate_tied_groups"),))
    want(tie.get("lm_head_data_ptr") is not None
         and tie.get("lm_head_data_ptr") == tie.get("embed_data_ptr"),
         "A9 ok: data_ptr identical (%r)" % tie.get("lm_head_data_ptr"),
         "A9: data_ptr differs (%r vs %r)"
         % (tie.get("lm_head_data_ptr"), tie.get("embed_data_ptr")))
    want(not (rep.get("errors") and
              any(e.get("where") == "load" for e in rep["errors"])),
         "load stage recorded no error",
         "load stage recorded an error: %r"
         % [e for e in rep.get("errors") or [] if e.get("where") == "load"])

    # ---- B: injection, frozen base, gradients, weight change ----------------
    want(rep.get("n_matched_modules") == EXPECTED_N_MODULES,
         "A6 ok: %d matched modules (2 x 60 layers)" % EXPECTED_N_MODULES,
         "A6: %r matched modules, expected %d"
         % (rep.get("n_matched_modules"), EXPECTED_N_MODULES))
    # Absent sample != clean sample: A5 only means something once injection ran.
    sample = rep.get("matched_sample")
    if isinstance(sample, list) and sample:
        off = [m for m in sample if "language_model" not in m]
        want(not off, "A5 ok: all %d sampled matches are under language_model"
             % len(sample),
             "A5: sample contains %r" % off)
        towers = [m for m in sample if "vision_tower" in m or "audio_tower" in m]
        want(not towers, "A5 ok: no tower modules in the sample",
             "A5: tower modules in the sample %r" % towers)
    else:
        problems.append("A5: no matched_sample recorded, so 'no module outside "
                        "language_model' is unestablished")
    want(rep.get("n_trainable_params") == EXPECTED_N_TRAINABLE,
         "B2 ok: %d trainable parameters, matching the closed-form 14,336,000"
         % EXPECTED_N_TRAINABLE,
         "B2: %r trainable, expected %d (50x%d + 10x%d = %d)"
         % (rep.get("n_trainable_params"), EXPECTED_N_TRAINABLE,
            SLIDING_PER_LAYER, FULL_PER_LAYER,
            50 * SLIDING_PER_LAYER + 10 * FULL_PER_LAYER))
    # The B1 check records this inside its own value block, not at top level.
    b1_val = checks.get("B1", {}).get("value") or {}
    non_lora = b1_val.get("non_lora_trainable")
    n_tr_tensors = b1_val.get("n_trainable_tensors")
    if isinstance(non_lora, list) and isinstance(n_tr_tensors, int):
        want(not non_lora,
             "B1 ok: nothing outside the %d LoRA tensors is trainable"
             % n_tr_tensors,
             "B1: non-LoRA trainable tensors %r" % (non_lora,))
    else:
        problems.append("B1: the check recorded non_lora_trainable=%r "
                        "n_trainable_tensors=%r -- 'the base is frozen' is "
                        "unestablished" % (non_lora, n_tr_tensors))

    steps = rep.get("training_steps") or rep.get("train_steps") or []
    if steps:
        losses = [s.get("loss") for s in steps]
        finite_losses = all(x is not None and math.isfinite(x) for x in losses)
        want(finite_losses, "B3 ok: all %d step losses finite" % len(steps),
             "B3: non-finite loss in %r" % losses)
        want(all(s.get("grad_finite") for s in steps),
             "B4 ok: every gradient tensor finite at every step",
             "B4: a step had a non-finite gradient")
        b_grads = [s.get("lora_B_grad_abs_sum") for s in steps]
        want(all(g is not None and g > 0 for g in b_grads),
             "B5 ok: lora_B gradient nonzero at every step (%s)"
             % ", ".join("%.1f" % g for g in b_grads),
             "B5: a step had zero lora_B gradient: %r" % b_grads)
        # Step 0 with a zero-initialised lora_B has exactly zero gradient on A.
        a_grads = [s.get("lora_A_grad_abs_sum") for s in steps]
        want(a_grads and a_grads[0] == 0.0,
             "note: lora_A gradient is 0.0 at step 0, as expected for a "
             "zero-initialised lora_B",
             "WARN: lora_A gradient at step 0 is %r, not the expected 0.0"
             % (a_grads[0] if a_grads else None))
        want(any(g for g in a_grads[1:]),
             "lora_A gradient becomes nonzero from step 1",
             "lora_A gradient never becomes nonzero")
        want(losses[-1] < losses[0],
             "loss fell %.3f -> %.3f over %d steps"
             % (losses[0], losses[-1], len(steps)),
             "loss did not fall: %.3f -> %.3f" % (losses[0], losses[-1]))
    else:
        problems.append("no training steps recorded")

    max_delta = rep.get("max_weight_delta")
    want(max_delta is not None and max_delta > 0,
         "B6 ok: weights changed by up to %.6f" % (max_delta or 0.0),
         "B6: max_weight_delta=%r" % max_delta)
    per_tensor = rep.get("per_tensor_delta") or {}
    if per_tensor:
        changed = [k for k, v in per_tensor.items() if v and v > 0]
        want(len(changed) == EXPECTED_N_TENSORS,
             "B6 ok: %d/%d adapter tensors changed"
             % (len(changed), EXPECTED_N_TENSORS),
             "B6: only %d/%d adapter tensors changed"
             % (len(changed), EXPECTED_N_TENSORS))

    # ---- B7 / B7b: the 2048-token step -------------------------------------
    long_step = rep.get("long_step") or {}
    reached = long_step.get("seq_len")
    want(long_step.get("at_planned_length") is True
         and long_step.get("planned_seq_len") == LONG_SEQ_TARGET,
         "B7b ok: a %d-token forward+backward+optimiser step completed (%.1fs)"
         % (LONG_SEQ_TARGET, long_step.get("seconds") or 0.0),
         "B7b: planned %d, reached %r (OOM ladder: %r)"
         % (LONG_SEQ_TARGET, reached, rep.get("long_step_oom")))
    if long_step:
        want(long_step.get("grad_finite") is True
             and (long_step.get("lora_B_grad_abs_sum") or 0) > 0,
             "B7 ok: the long step produced finite, nonzero lora_B gradients",
             "B7: long step gradient_finite=%r b_grad=%r"
             % (long_step.get("grad_finite"),
                long_step.get("lora_B_grad_abs_sum")))
        want((long_step.get("loss") or -1) >= 0
             and math.isfinite(long_step.get("loss") or -1),
             "B7 ok: long-step loss %.4f is finite" % (long_step.get("loss") or 0.0),
             "B7: long-step loss %r" % long_step.get("loss"))

    # ---- C: save and reload -------------------------------------------------
    c1 = checks.get("C1", {}).get("value") or {}
    want(c1.get("max_abs_diff") == 0.0 and not (c1.get("missing") or []),
         "C1 ok: %r keys bit-identical between memory and disk"
         % c1.get("compared"),
         "C1: %r" % (c1,))
    c2 = checks.get("C2", {}).get("value") or {}
    want(c2.get("max_abs_diff") == 0.0 and not (c2.get("missing") or []),
         "C2 ok: %r keys identical after a PEFT reload" % c2.get("compared"),
         "C2: %r" % (c2,))

    # ---- D: memory ----------------------------------------------------------
    peaks_by_phase = (rep.get("peak_memory_by_phase")
                      or rep.get("peak_memory_during_training"))
    short_peaks = rep.get("peak_memory_short_phase")
    long_peaks = rep.get("peak_memory_long_phase")
    if isinstance(peaks_by_phase, dict) and peaks_by_phase:
        want(True,
             "D1 recorded: per-card peaks for both phases (%d cards)"
             % len(peaks_by_phase), "")
    else:
        problems.append("D1 missing: no per-card peak memory recorded "
                        "(peak_memory_by_phase=%r)" % (peaks_by_phase,))
    d3 = checks.get("D3", {}).get("value") or {}
    want(checks.get("D3", {}).get("ok") is True,
         "D3 ok: no card exceeded its capacity; long step %r at %r tokens"
         % (d3.get("long_step_seconds"), d3.get("long_step_seq_len")),
         "D3: %r" % (d3,))

    # ---- E: served adapter is actually consumed -----------------------------
    e2 = checks.get("E2", {}).get("value") or {}
    noise = e2.get("delta")
    e3 = checks.get("E3", {}).get("value") or {}
    d_trained = e3.get("delta")
    e4 = checks.get("E4", {}).get("value") or {}
    d_zero = e4.get("delta")

    if d_trained is None:
        problems.append("E3 did not produce a delta; adapter consumption is "
                        "unverified")
    else:
        big_enough = d_trained >= E3_MIN_DELTA
        above_noise = (noise in (None, 0.0)) or (d_trained > E3_NOISE_MULTIPLE * noise)
        want(big_enough and above_noise,
             "E3 ok: trained adapter moves logprobs by %.6f (noise floor %r)"
             % (d_trained, noise),
             "E3: delta %.6f, noise %r, thresholds %r/%r"
             % (d_trained, noise, E3_MIN_DELTA, E3_NOISE_MULTIPLE))
        # E4 is the control that makes E3 meaningful: a zero lora_B must not move
        # the logprobs, otherwise E3's delta could come from anything.
        if d_zero is None:
            problems.append("E4 control did not produce a delta")
        else:
            want(d_zero == 0.0 or d_zero <= (noise or 0.0),
                 "E4 ok: zero-lora_B control moved nothing (%r), so E3's delta "
                 "is attributable to the trained weights" % d_zero,
                 "E4: zero-B control moved %.6f (noise %r) -- E3's delta is not "
                 "attributable to the adapter" % (d_zero, noise))
            if d_trained and d_zero and noise:
                rel = abs(d_trained - d_zero) / max(abs(d_trained), 1e-12)
                want(rel > 0.5, "E3/E4 separated by %.1f%%" % (100 * rel),
                     "E3 and E4 are within %.1f%% of each other" % (100 * rel))

    e5 = checks.get("E5", {}).get("value") or {}
    # Require positive evidence: an absent E5 must not read as "no unmapped keys".
    e5_keys = e5.get("lora_A_keys")
    e5_mods = e5.get("distinct_modules")
    want(isinstance(e5_keys, int) and e5_keys > 0
         and isinstance(e5_mods, int) and e5_mods > 0
         and not (e5.get("unknown") or []),
         "E5 ok: %r adapter keys all map to PEFT-wrapped modules (%r modules)"
         % (e5_keys, e5_mods),
         "E5: lora_A_keys=%r distinct_modules=%r unmapped=%r -- an absent or "
         "empty E5 does not establish that the saved keys are consumable"
         % (e5_keys, e5_mods, e5.get("unknown")))

    # ---- E6: the server log ------------------------------------------------
    log_path = find_log(out)
    if log_path is not None:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        found = [s for s in FORBIDDEN_LOG if s in text]
        want(not found,
             "E6 ok: %d chars of vLLM log, none of the unconsumed-weight "
             "warnings" % len(text),
             "E6: forbidden strings in the server log: %r" % found)
        # "lora" alone is too weak: `--enable-lora` is in the startup command
        # whether or not any adapter is ever mounted. Require the vLLM lines that
        # only appear once an adapter is actually loaded.
        low = text.lower()
        # The exact line vLLM emits when an adapter is registered, observed in
        # Gate 1's passing run: "serving.py:201 Loaded new LoRA adapter: name
        # 'trained_gate1', path '...'". Anything weaker would also match the
        # startup flags.
        mounted = sorted({m.group(0) for m in re.finditer(
            r"loaded new lora adapter: name '([^']+)'", low)})
        want(bool(mounted),
             "E6 ok: the log shows adapters being loaded (%r)" % (mounted,),
             "E6: the log never shows an adapter being loaded (only the flag) "
             "-- 'lora' appears in the startup command regardless")
    else:
        problems.append("no vLLM server log found to scan")

    e8 = checks.get("E8", {}).get("value") or {}
    want(checks.get("E8", {}).get("ok") is True,
         "E8 ok: tool call parsed as %r" % (e8.get("tool_name") or e8,),
         "E8: no tool call produced: %r" % (e8,))

    # ---- memory headline: every card within capacity -----------------------
    peaks = (rep.get("peak_memory_long_phase")
             or rep.get("peak_memory_short_phase")
             or rep.get("peak_memory_during_training") or {})
    over = {k: v for k, v in peaks.items()
            if (v.get("peak_reserved_gib") or 0) > (v.get("total_gib") or 1e9)}
    want(not over, "no card exceeded its total memory during training",
         "cards exceeded capacity: %r" % over)

    # ---- report -------------------------------------------------------------
    print()
    for msg in notes:
        print("  " + msg)
    print()

    missing = [c for ids in HARD.values() for c in ids if c not in checks]
    failed = [c for ids in HARD.values() for c in ids
              if c in checks and status(c) == "FAIL"]
    skipped = [c for ids in HARD.values() for c in ids
               if c in checks and status(c) == "SKIPPED"]
    soft_missing = sorted(c for c in SOFT if c not in checks)

    if problems:
        print("=== problems ===")
        for p in problems:
            print("  - " + p)
        print()

    print("hard missing:", missing or "none")
    print("hard failed :", failed or "none")
    print("hard skipped:", skipped or "none")
    print("soft missing:", soft_missing or "none")
    print()
    print("notebook's own verdict:",
          json.dumps((rep.get("verdict") or {}).get("pass"), default=str))

    ok = (not missing) and (not failed) and (not skipped) and (not problems)
    print()
    print("GATE 2:", "PASS" if ok else "FAIL")

    verdict_doc = {
        "gate": "gate2_31b",
        "kernel": rep.get("kernel"),
        "pass": ok,
        "hard_missing": missing,
        "hard_failed": failed,
        "hard_skipped": skipped,
        "soft_missing": soft_missing,
        "independent_problems": problems,
        "independent_notes": notes,
        "notebook_verdict": (rep.get("verdict") or {}).get("pass"),
        "checks": {cid: {"ok": v.get("ok"), "value": v.get("value"),
                         "detail": v.get("detail")}
                   for cid, v in checks.items()},
        "load": {
            "route": rep.get("load_route"),
            "cpu_load_seconds": rep.get("cpu_load_seconds"),
            "rss_peak_gib": rss,
            "ru_maxrss_gib": ru,
            "rss_samples": rep.get("cpu_load_rss_samples"),
            "planned_per_card_gib": rep.get("planned_per_card_gib"),
            "actual_after_load": rep.get("actual_gpu_usage_after_load"),
            "plan_vs_actual_gib": rep.get("plan_vs_actual_gib"),
        },
        "training": {
            "steps": len(steps),
            "losses": [s.get("loss") for s in steps],
            "lora_B_grad": [s.get("lora_B_grad_abs_sum") for s in steps],
            "lora_A_grad": [s.get("lora_A_grad_abs_sum") for s in steps],
            "short_step_seconds": rep.get("short_step_seconds"),
            "long_step": long_step,
            "long_step_oom": rep.get("long_step_oom"),
            "peak_memory": peaks,
            "peak_memory_short_phase": rep.get("peak_memory_short_phase"),
            "peak_memory_long_phase": rep.get("peak_memory_long_phase"),
            "max_weight_delta": max_delta,
            "n_tensors_changed": len([k for k, v in (per_tensor or {}).items()
                                      if v and v > 0]),
        },
        "adapter": {
            "discovered": rep.get("discovered_adapters"),
            "served_models": rep.get("served_models"),
            "E2_noise": noise,
            "E3_delta": d_trained,
            "E4_delta": d_zero,
            "E5": e5,
            "E6_forbidden_found": [],
        },
        "timings": rep.get("timings"),
        "wall_seconds": rep.get("wall_seconds"),
        "serve_config_comparison": rep.get("serve_config_comparison"),
        "source_report": str(rp),
    }
    dest = ROOT / "artifacts" / "gate2_verification.json"
    dest.write_text(json.dumps(verdict_doc, indent=2, default=str), encoding="utf-8")
    print("  wrote %s" % dest)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
