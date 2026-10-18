"""Run official directory, YAML, and schema validators (no inference)."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from adk_submission import SubmissionLimits, discover_adapters, discover_declared_models, validate_directory
from adk_submission.schema import SandboxedAgentConfig
from adk_submission.yaml_loader import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("submission", type=Path)
parser.add_argument("report", type=Path)
args = parser.parse_args()
start = time.perf_counter()
report = {"adk_submission": importlib.metadata.version("adk-submission"),
          "google_adk": importlib.metadata.version("google-adk"),
          "submission": str(args.submission.resolve()),
          "model_inference_completed": False, "agent_compilation_completed": False}
try:
    root = args.submission.resolve()
    limits = SubmissionLimits(max_total_size_bytes=3 * 1024**3, max_file_count=10000,
                              max_yaml_files=1000, max_yaml_size_bytes=50 * 1024**2,
                              allowed_file_extensions=frozenset({".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}))
    validate_directory(root, limits=limits)
    report["directory_validated"] = True
    models = discover_declared_models(root)
    assert models == {"gemma-4-31b-it-qat-w4a16-ct"}, models
    report["declared_models"] = sorted(models)
    validated = []
    for path in sorted(root.rglob("*.yaml")):
        raw = load_yaml(path, root, limits=limits)
        if "name" in raw and ("model" in raw or "agent_class" in raw):
            SandboxedAgentConfig.model_validate(raw)
            validated.append(path.relative_to(root).as_posix())
    discover_adapters(root, adapter_extensions={".safetensors"})
    report["agent_schemas"] = validated
    report["official_static_validation_passed"] = True
except Exception as error:
    report["official_static_validation_passed"] = False
    report["error_type"] = type(error).__name__
    report["error"] = str(error)
report["duration_seconds"] = time.perf_counter() - start
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["official_static_validation_passed"] else 1)
