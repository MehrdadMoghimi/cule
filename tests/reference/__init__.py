"""Vendored reference implementations used only by the equivalence tests.

Each module in this package is a faithful transcription of a published
implementation, kept here so the equivalence tests are hermetic (no network
access, no optional JAX/Flax install).  Nothing in this package is imported by
the trainers in `cleanrl/`; it exists purely so the tests can assert that our
ports reproduce the reference numerics.

Provenance and licence of each file is recorded in its own header.  Reference
code is only vendored where the upstream licence permits redistribution; for
upstream projects that do not (MR.Q is CC-BY-NC, streaming-drl publishes no
licence), the corresponding tests check the algorithm's published equations and
invariants instead of diffing against copied source.
"""
