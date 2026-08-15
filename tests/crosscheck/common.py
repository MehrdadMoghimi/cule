"""Shared plumbing for the upstream cross-checks.

Each `check_*.py` script loads the authors' official implementation out of
`third_party/upstream/`, transplants weights between it and this fork's port,
and reports a per-component max absolute difference. Nothing here is imported by
the trainers or by the normal test suite.
"""

import argparse
import contextlib
import importlib.util
import os
import sys

import numpy as np
import torch

# The cross-checks compare numerics, so the GPU runs in full fp32: TF32 costs
# ~1e-3 relative accuracy on convolutions and matmuls, which would swamp any
# real difference. The trainers themselves keep the repository's defaults.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEANRL_DIR = os.path.join(REPO_ROOT, "cleanrl")
UPSTREAM_DIR = os.path.join(REPO_ROOT, "third_party", "upstream")


def upstream(name):
    """Absolute path to a cloned upstream, or exit with instructions."""
    path = os.path.join(UPSTREAM_DIR, name)
    if not os.path.isdir(path):
        sys.exit(
            f"{name} is not cloned.\n"
            f"Run: python tests/crosscheck/clone_upstreams.py {name}"
        )
    return path


def load_trainer(name):
    """Import `cleanrl/<name>.py` without running its training loop."""
    for path in (CLEANRL_DIR, REPO_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        f"_trainer_{name}", os.path.join(CLEANRL_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def sys_path(*paths):
    """Temporarily prepend paths, so upstream's flat `import models` resolves."""
    added = [path for path in paths if path not in sys.path]
    for path in reversed(added):
        sys.path.insert(0, path)
    try:
        yield
    finally:
        for path in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


def load_module(name, path):
    """Import a single upstream file under an explicit module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class Report:
    """Collects per-component differences and prints a verdict."""

    def __init__(self, title, tolerance=2e-5):
        self.title = title
        self.tolerance = tolerance
        self.rows = []

    def check(self, name, ours, theirs, tolerance=None):
        ours, theirs = as_numpy(ours), as_numpy(theirs)
        limit = self.tolerance if tolerance is None else tolerance
        if ours.shape != theirs.shape:
            self.rows.append((name, float("inf"), False, f"shape {ours.shape} vs {theirs.shape}"))
            return False
        scale = max(float(np.abs(theirs).max()), 1.0)
        difference = float(np.abs(ours - theirs).max())
        passed = bool(np.isfinite(difference)) and difference <= limit * scale
        self.rows.append((name, difference, passed, ""))
        return passed

    def note(self, name, passed, detail=""):
        self.rows.append((name, 0.0, bool(passed), detail))
        return passed

    @property
    def ok(self):
        return all(row[2] for row in self.rows)

    def print(self):
        width = max(len(row[0]) for row in self.rows) + 2
        print(f"\n=== {self.title} ===")
        for name, difference, passed, detail in self.rows:
            mark = "OK  " if passed else "FAIL"
            value = "-" if detail else f"max|diff| = {difference:.3e}"
            print(f"  {mark} {name:<{width}} {value} {detail}")
        print(f"  -> {'ALL MATCH' if self.ok else 'MISMATCH'} "
              f"({sum(row[2] for row in self.rows)}/{len(self.rows)})")
        return self.ok


def parse_devices():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="all", choices=["cpu", "cuda", "all"])
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.device == "all":
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    else:
        devices = [arguments.device]
    return devices, arguments.seed


def copy_matching(destination, source, rename=(), allow_missing=()):
    """Copy `source`'s state dict into `destination`, applying key renames.

    `allow_missing` names buffers that exist on our side but are plain Python
    attributes upstream (so they never appear in its state dict); those are
    compared separately by the caller.
    """
    state = {}
    for key, value in source.state_dict().items():
        for old, new in rename:
            key = key.replace(old, new)
        state[key] = value.clone()
    missing, unexpected = destination.load_state_dict(state, strict=False)
    missing = [key for key in missing if key not in allow_missing]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch; missing={missing} unexpected={unexpected}")
    return destination
