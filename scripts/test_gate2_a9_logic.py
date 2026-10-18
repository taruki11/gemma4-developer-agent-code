"""Check the A9 / tied-embedding logic in isolation, by running the real snippet.

The check only executes inside a GPU run, so a mistake in it would surface after a
two-hour wait. This builds the smallest plausible stand-ins -- a model with a real
tied pair, and one whose tie has been broken -- and runs the notebook's own A9 code
against both, so the check is proven to distinguish them here.
"""

import ast
import io
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads(io.open(ROOT / "artifacts" / "training_gate2" / "smoke.ipynb",
                        encoding="utf-8").read())
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
load = next(s for s in cells if "from_pretrained" in s and "quantization_config" in s)

# Lift the A9 block verbatim: from the "A9:" comment to just before A4's `rec`.
start = load.index("    # A9:")
end = load.index('    rec("A4",')
snippet = textwrap.dedent(load[start:end])

# The snippet references names the load cell defines; supply them.
import types


class FakeTensor:
    """Just enough of a tensor to answer what A9 asks: identity, device, ptr."""

    def __init__(self, name, device="cuda:1"):
        self.name = name
        self.device = device

    def data_ptr(self):
        return id(self)


class Config:
    tie_word_embeddings = True


class InEmbed:
    def __init__(self, w):
        self.weight = w


class OutEmbed:
    def __init__(self, w):
        self.weight = w


class FakeModel:
    def __init__(self, shared):
        w = FakeTensor("embed_tokens.weight")
        self._in = InEmbed(w)
        self._out = OutEmbed(w if shared else FakeTensor("lm_head.weight"))
        self.config = Config()

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out


def run(shared, find_tied_impl):
    checks = {}

    def rec(cid, ok, value, detail=""):
        checks[cid] = {"ok": bool(ok), "value": value, "detail": detail}
        return bool(ok)

    model = FakeModel(shared)
    # `accelerate` is not installed locally; inject the name the snippet imports.
    fake_acc = types.ModuleType("accelerate.utils")
    fake_acc.find_tied_parameters = find_tied_impl
    ns = {
        "model": model,
        "rec": rec,
        "report": {"checks": {}},
        "torch": __import__("types").SimpleNamespace(
            equal=lambda a, b: a is b),
        "lm_head_w": model.get_output_embeddings().weight,
        "embed_w": model.get_input_embeddings().weight,
        "tied_shared": model.get_output_embeddings().weight
                        is model.get_input_embeddings().weight,
        "tied_equal": model.get_output_embeddings().weight
                       is model.get_input_embeddings().weight,
        "tied_devices": [str(model.get_output_embeddings().weight.device),
                         str(model.get_input_embeddings().weight.device)],
        "sys": sys,
    }
    # Redirect the snippet's import to the stub. The stub has to be in the snippet's
    # namespace too: without it the replacement raises NameError, and the snippet's
    # own `except Exception` swallows that as "no tie detected".
    ns["find_tied_impl"] = find_tied_impl
    body = snippet.replace(
        "from accelerate.utils import find_tied_parameters as _find_tied",
        "_find_tied = find_tied_impl")
    exec(body, ns)  # noqa: S102 -- deliberately running the notebook's own logic
    return checks


# accelerate sees the tie when the two names land in one group.
def tied_detect_real(model):
    return [["model.language_model.embed_tokens.weight", "lm_head.weight"]]


def tied_detect_none(model):
    return [["model.language_model.embed_tokens.weight"]]


failures = []

checks = run(True, tied_detect_real)
a9 = checks.get("A9", {}).get("ok")
print("tied model, tie detected        -> A9=%s" % a9)
if a9 is not True:
    failures.append("a correctly tied model must pass A9")

checks = run(False, tied_detect_none)
a9 = checks.get("A9", {}).get("ok")
print("broken tie, tie not detected    -> A9=%s" % a9)
if a9 is not False:
    failures.append("a broken tie must fail A9")

checks = run(True, tied_detect_none)
a9 = checks.get("A9", {}).get("ok")
print("tied storage, library blind     -> A9=%s" % a9)
if a9 is not False:
    failures.append("a tie the library cannot see must fail A9")

print()
if failures:
    print("A9 LOGIC TEST: FAIL")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("A9 LOGIC TEST: PASS (3 cases)")
