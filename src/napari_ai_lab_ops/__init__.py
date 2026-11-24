"""Ops that napari-ai-lab runs in scikit-ops' environments.

A small op collection (scikit-ops design 0022), shipped inside the
napari-ai-lab wheel. Each op declares a scikit-ops environment -- ``pytorch``
for micro_sam -- and runs in that environment's worker, which imports this
module from where the host found it.

Kept apart from ``napari_ai_lab`` on purpose: a worker importing
``napari_ai_lab.anything`` would run that package's ``__init__``, which pulls
in napari and Qt, built for the host's Python. Nothing here imports more than
numpy and opspec at module level; each op imports its own stack inside.
"""
