"""Create an agent-visible dev set without reference patches or grading tests."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tasks", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--count", type=int, default=4)
args = parser.parse_args()
if args.count < 1:
    parser.error("--count must be positive")
rows = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
rows.sort(key=lambda row: row["instance_id"])
# Choose one per repository before taking additional tasks, without examining answers.
selected, repos = [], set()
for row in rows:
    if row["repo"] not in repos:
        selected.append(row)
        repos.add(row["repo"])
    if len(selected) == args.count:
        break
for row in rows:
    if len(selected) >= args.count:
        break
    if row not in selected:
        selected.append(row)
args.output.mkdir(parents=True, exist_ok=True)
allowed = ("instance_id", "repo", "base_commit", "problem_statement", "hints_text")
agent_rows = [{key: row[key] for key in allowed if key in row} for row in selected]
(args.output / "agent_tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in agent_rows), encoding="utf-8")
manifest = {
    "source": str(args.tasks.resolve()),
    "source_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
    "selection": "instance_id alphabetical, first per repo, then fill; answers not used",
    "task_ids": [row["instance_id"] for row in selected],
    "agent_fields": list(allowed),
    "reference_patch_exposed": False,
    "test_patch_exposed": False,
}
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))

