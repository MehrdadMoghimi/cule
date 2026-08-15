"""Clone the official implementations the cross-checks diff against.

Checkouts land in `third_party/upstream/` and are gitignored: streaming-drl and
MRQ are CC BY-NC 4.0, incompatible with this repository's license, and the rest
would just be vendored copies. Nothing here is imported by the trainers.
"""

import argparse
import os
import subprocess
import sys

UPSTREAM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "upstream")
)

# name -> (url, sparse paths or None for a full shallow clone)
REPOS = {
    "MRQ": ("https://github.com/facebookresearch/MRQ.git", None),
    "disco_rl": ("https://github.com/google-deepmind/disco_rl.git", None),
    "BTR": ("https://github.com/VIPTankz/BTR.git", None),
    "hadamax": ("https://github.com/jacobkooi/hadamax.git", None),
    "streaming-drl": ("https://github.com/mohmdelsayed/streaming-drl.git", None),
    "FQF": ("https://github.com/microsoft/FQF.git", None),
    "spr": ("https://github.com/mila-iqia/spr.git", None),
    "dopamine": ("https://github.com/google/dopamine.git", None),
    "google-research": (
        "https://github.com/google-research/google-research.git",
        ["munchausen_rl", "bigger_better_faster"],
    ),
    "relative-value-learning": ("https://github.com/Hauf3n/relative-value-learning.git", None),
    # World models. dreamerv3-torch is the PyTorch reproduction the DreamerV3
    # cross-check runs against; danijar/dreamerv3 itself is JAX.
    "dreamerv3-torch": ("https://github.com/NM512/dreamerv3-torch.git", None),
    "storm": ("https://github.com/weipu-zhang/STORM.git", None),
    "twister": ("https://github.com/burchim/TWISTER.git", None),
    "simulus": ("https://github.com/leor-c/Simulus.git", None),
    # Cloned for completeness; as of 2026-08-08 this repository holds only a
    # LICENSE and a README saying the code lands by 2026-08-15, so
    # cleanrl/endpoint_ddqn_atari.py has no numerical cross-check yet.
    "endpoint-replay": ("https://github.com/panahiparham/endpoint-replay.git", None),
}


def clone(name, url, sparse):
    target = os.path.join(UPSTREAM_DIR, name)
    if os.path.exists(os.path.join(target, ".git")):
        print(f"{name}: already cloned")
        return True
    os.makedirs(UPSTREAM_DIR, exist_ok=True)
    try:
        if sparse:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, target],
                check=True, capture_output=True, text=True, timeout=1800)
            subprocess.run(["git", "-C", target, "sparse-checkout", "set", *sparse],
                           check=True, capture_output=True, text=True, timeout=1800)
        else:
            subprocess.run(["git", "clone", "--depth", "1", url, target],
                           check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as error:
        print(f"{name}: FAILED\n{error.stderr[-500:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"{name}: TIMEOUT")
        return False
    print(f"{name}: cloned")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=None,
                        help="repositories to clone; default is all of them")
    arguments = parser.parse_args()
    names = arguments.names or list(REPOS)
    ok = all(clone(name, *REPOS[name]) for name in names)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
