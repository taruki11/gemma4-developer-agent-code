"""Record the inference batch v1 result in status.json.

Written as a script rather than an inline command because the previous attempts to put
quoted JSON through PowerShell mangled it.
"""

import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "artifacts" / "status.json"

status = json.load(io.open(STATUS, encoding="utf-8"))

status["infer_batch_v1_adapter_consumption"] = {
    "verdict": "FAIL",
    "question": ("does the official scoring stack consume the Gate 2 v3 adapters at "
                 "inference time?"),
    "answer": ("not answered: the server never became healthy, so no request was served "
               "and no logprob comparison ran"),
    "kernel": "taruki/gemma4-infer-batch-v1",
    "kernel_version": 1,
    "notebook_sha256": ("683269a336c8c50dabfc50fa19d1894058fbf49bb5dbb83d398c584fe997051c"),
    "wall_seconds": 1608.6,
    "gpu_hours": 0.9047,
    "load_route": "vllm only: competition 31B checkpoint plus the two v3 adapters",
    "adapters": {
        "source": "reused unchanged from the Gate 2 v3 training run, never re-trained",
        "trained_gate2_adapter_model_sha256": (
            "4057ee3ad3cbd58ad4e0cfd4c08621d514265a3a4c87a8decdf7f3ea21a3d9e7"),
        "zero_b_control_adapter_model_sha256": (
            "dcea9cfff685136f7ea01d74725a5e3c485d42f31f80c20479cd08c7a7212384"),
        "adapter_config_sha256_both": (
            "012948f1aa8f62183962081569b0fb1d483963ff85158404959598ceed665155"),
        "integrity_checks": {
            "local_hash_before_upload": True,
            "mounted_hash_in_kernel": "A1 mismatches []",
            "working_copy_hash": "A2 drift []",
            "content_audit": "A4 trained 120/120 nonzero lora_B, control 0/120",
            "config": "A3 r=8 alpha=16 LORA q/o regex, both configs byte-identical",
        },
    },
    "serving_configuration": {
        "source": "data/official/HARNESS_README.md:163-169",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.80,
        "max_model_len": 32768,
        "tool_call_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "enable_auto_tool_choice": True,
        "default_chat_template_kwargs": {"enable_thinking": True},
        "enable_lora": True,
        "max_loras_declared": 8,
        "max_loras_actual": 2,
        "max_lora_rank_declared": 128,
        "max_lora_rank_actual": 8,
        "enforce_eager": False,
        "enforce_eager_present_in_argv": False,
    },
    "failure": {
        "where": "vllm_start",
        "error": ("ServerStartupError: Inference server did not become healthy within "
                  "1500s"),
        "root_cause": ("CUDA out of memory during sampler warm-up with 256 dummy "
                       "requests, after KV cache allocation succeeded"),
        "detail": ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate "
                   "256.00 MiB. GPU 0 has a total capacity of 22.03 GiB of which 13.06 "
                   "MiB is free."),
        "timeline": [
            "14:05:23 server starts, lora_modules accepted without a target-module warning",
            "14:07:57 all four workers report the model supports LoRA on tower modules",
            "14:10:21 CUDA graph profiling: PIECEWISE=102 (largest=512), FULL=70 (largest=256)",
            "14:10:32 estimated CUDA graph memory 4.08 GiB per worker",
            "14:10:33 available KV cache memory 10.35 GiB, GPU KV cache 45,216 tokens",
            "14:11:46 sampler warm-up needs 256 MiB, 13.06 MiB free, all four workers die",
        ],
        "log_artifact": "artifacts/infer_batch_v1_outputs/infer_vllm_server_error.txt",
    },
    "contradiction_reported_not_resolved": {
        "summary": ("b28 served the same 31B checkpoint on the same 4x L4 hardware with "
                    "no --enforce-eager, but at gpu_memory_utilization 0.90; this batch "
                    "used the README's 0.80 and failed. The difference has not been "
                    "isolated, so 'the official configuration cannot work here' is NOT "
                    "supported by this evidence."),
        "b28_utilization": 0.90,
        "this_batch_utilization": 0.80,
        "b28_evidence": ("artifacts/tool_control_v2_b28_outputs/execution.log: "
                         "'vLLM server started on http://127.0.0.1:8000/v1 (tp=4)'"),
        "unit_mismatch": ("the README describes 0.80 as ~19.2 GB usable per GPU, which is "
                          "a fraction of a 24 GB card; these devices report 22.03 GiB, so "
                          "0.80 reserves 17.62 GiB"),
    },
    "established_by_measurement": [
        "the official parameters were built and passed verbatim, enforce-eager provably absent",
        "both adapters were discovered by the official discovery path at rank 8",
        "vLLM accepted both --lora-modules entries with no unsupported-target warning",
        "the 31B W4A16 checkpoint loads under vLLM 0.19.1 with TP=4 and allocates KV cache",
        "0.80 of 22.03 GiB does not survive sampler warm-up on these devices; b28's 0.90 did",
    ],
    "not_established": [
        "whether the serving stack consumes the adapters",
        "any logprob difference between base, trained adapter and zero-B control",
        "any difference in generation or tool calling attributable to the adapter",
        "anything about code-fixing ability or agent behaviour",
    ],
    "skipped_checks": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
    "independent_verification": {
        "script": "scripts/verify_infer_batch_v1.py",
        "self_tests": "13 scenarios, all passing",
        "result": "FAIL: 19/30 checks passed, every failure is a missing measurement",
        "output": "artifacts/infer_batch_v1_verification.json",
    },
    "cost": {
        "quota_before_hours": 6.3680,
        "quota_after_hours": 7.2727,
        "this_batch_hours": 0.9047,
        "remaining_hours": 22.7273,
        "total_hours": 30.0,
        "refreshes": "2026-10-24T00:00:00",
    },
    "candidate_next_steps_not_run": [
        "gpu_memory_utilization=0.90 with everything else official: the only configuration "
        "with direct positive evidence on this hardware (b28 served with it)",
        "keep 0.80 and lower --max-num-seqs: what vLLM's own error message suggests, but it "
        "changes another official default",
        "--enforce-eager: known to work from Gate 2 v1, but it is exactly the deviation this "
        "batch existed to remove",
    ],
    "not_authorised": [
        "full SFT",
        "RL",
        "any further GPU runs without a stated plan",
        "leaderboard submission",
        "overwriting the frozen baseline or the v3 training outputs",
    ],
    "reports": [
        "artifacts/infer_batch_v1_result.md",
        "artifacts/infer_batch_v1_prereg.md",
        "artifacts/infer_batch_v1_outputs/infer_batch_report.json",
        "artifacts/infer_batch_v1_verification.json",
    ],
}

status["gpu_usage_hours"] = 7.2727
status["gpu_quota_observed"] = {
    "total_hours": 30.0,
    "used_hours": 7.2727,
    "remaining_hours": 22.7273,
    "refreshes": "2026-10-24T00:00:00",
    "read_at": "2026-10-18",
}

STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")
print("updated", STATUS.relative_to(ROOT))
print("gpu_usage_hours:", status["gpu_usage_hours"])
print("infer batch verdict:", status["infer_batch_v1_adapter_consumption"]["verdict"])
