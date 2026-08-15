# Upstream checkouts (not committed)

`tests/crosscheck/` diffs this fork's ports against the authors' official
implementations. Those implementations are cloned into this directory and are
deliberately **not** committed — `streaming-drl` and `MRQ` are CC BY-NC 4.0,
incompatible with this repository's license, and the rest would just be vendored
copies.

Populate:

    python tests/crosscheck/clone_upstreams.py

Run a cross-check:

    python tests/crosscheck/check_mrq.py --device cuda
    python tests/crosscheck/run_all.py
