"""Two-stage (fixer -> auditor) submission layout for the budget-complete direction.

The root agent becomes a SequentialAgent with two sub-agents:
  - fixer: baseline system prompt + a handoff addendum; has all tools EXCEPT submit_patch,
    so it structurally cannot submit before the auditor runs.
  - auditor: the only agent with submit_patch; instructed to read the real git diff,
    verify coverage/wiring/failed edits, land gaps, then submit.

The sample prompts/system.md is copied to prompts/fixer.md with the addendum appended,
so the fixer's instruction matches the frozen baseline prompt plus the handoff text.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT_YAML = """name: swe_two_stage_agent
agent_class: SequentialAgent
description: "Two-stage workflow: fixer then auditor; only the auditor can submit the patch."
sub_agents:
  - config_path: sub_agents/fixer.yaml
  - config_path: sub_agents/auditor.yaml
"""

FIXER_ADDENDUM = """
<!-- development experiment addendum (two-stage workflow: you are the FIXER) -->

## Two-Stage Workflow: You Are the Fixer (the Auditor Runs After You)

You work in a two-stage workflow. After you finish, an auditor agent reviews the actual
git diff, lands missing pieces, and calls `submit_patch`. You do NOT have the
`submit_patch` tool - do not try to submit.

- FIRST run `git diff --stat` (one call) to check the workspace state. If an earlier
  attempt already landed a candidate fix, do not redo it.
- Then land your best candidate fix for the issue: correct location, and when the change
  touches a shared pattern (a parameter, a base class, sibling classes, callers), cover
  the counterparts you can identify - the auditor will verify, but it can only patch
  small gaps.
- This overrides the "submit immediately" step of this prompt: run at most one targeted
  verification, then finish your turn.
- Finish early: aim to end your turn by your 12th tool call. The auditor needs roughly
  6 calls to review, land gaps, and submit.
- End your turn with a short status: files changed, what you verified, what is uncertain
  or unchecked for the auditor.
"""

AUDITOR_PROMPT = """You are the AUDITOR in a two-stage code-fix workflow. The fixer agent has finished.
You are the ONLY agent that can submit the patch, and you must submit before you end.

Your job: verify the ACTUAL changes against the issue, land every missing piece, then
submit. The run has failed if you end without submitting.

## Procedure (about 6-8 tool calls)

1. Run `git diff --stat` and then `git diff` (or `git diff -- <file>` for long diffs) -
   read the REAL hunks, not the fixer's summary.
2. Requirement coverage: for every behavior, parameter, status code, error message, or
   signature the issue asks for, point to the hunk that implements it. If you cannot
   point to a hunk, the requirement is NOT covered - land it now.
3. Wiring: if the diff adds or changes a parameter, argument, attribute, or function
   signature, grep all call sites, constructors, and sibling classes and confirm each
   one passes or handles it. A new parameter that no caller passes is not wired; a fix
   to one sibling that ignores the other siblings is incomplete.
4. Failed or missing pieces: if the fixer's status mentions an edit, or the diff is
   missing something the issue clearly needs, verify the file content NOW and land the
   change.
5. After your edits: run `git diff --stat` once more, then call `submit_patch` and end
   with a one-line summary.

- Never modify or create test files, and never run bare pytest; targeted commands only.
- Do not start broad new exploration or redesign the fix; you are here to complete and
  submit.
- If the diff is empty: apply your own best candidate fix for the issue first, then
  submit. Never end without a non-empty patch.
"""

FIXER_YAML = """name: fixer
description: "Fixes the issue and hands off to the auditor; cannot submit."
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
instruction: !include ../prompts/fixer.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include ../configs/sampling.yaml
disallow_transfer_to_parent: true
disallow_transfer_to_peers: true
"""

AUDITOR_YAML = """name: auditor
description: "Audits the diff and is the only agent allowed to submit."
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
instruction: !include ../prompts/auditor.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
generate_content_config: !include ../configs/sampling.yaml
disallow_transfer_to_parent: true
disallow_transfer_to_peers: true
"""


def build_layout(destination: Path, sample_source: Path) -> dict:
    """Materialize the two-stage layout from the sample submission tree."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(sample_source, destination)

    system_md = (destination / "prompts" / "system.md").read_text(encoding="utf-8")
    root_before = (destination / "agent.yaml").read_text(encoding="utf-8")
    (destination / "prompts" / "fixer.md").write_text(system_md + FIXER_ADDENDUM, encoding="utf-8")
    (destination / "prompts" / "auditor.md").write_text(AUDITOR_PROMPT, encoding="utf-8")
    (destination / "sub_agents" / "fixer.yaml").write_text(FIXER_YAML, encoding="utf-8")
    (destination / "sub_agents" / "auditor.yaml").write_text(AUDITOR_YAML, encoding="utf-8")
    (destination / "agent.yaml").write_text(ROOT_YAML, encoding="utf-8")

    return {
        "destination": str(destination),
        "root_agent_sha256_before": hashlib.sha256(root_before.encode("utf-8")).hexdigest(),
        "root_agent_sha256_after": hashlib.sha256(ROOT_YAML.encode("utf-8")).hexdigest(),
        "fixer_prompt_chars": len(system_md + FIXER_ADDENDUM),
        "auditor_prompt_chars": len(AUDITOR_PROMPT),
    }
