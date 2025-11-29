"""spikeOFC package.

This package implements the Factorized Innovation Spike-Coding estimator that
combines the closed-form SCN substrate with Bio-OFC style local learning.

Modules follow the layout defined in `docs/DATA_STRUCTURES.md`.
"""

from . import (
    lti,
    scn_core,
    spikeOFC_model,
    learning,
    delay,
    loop,
    logging as logging_utils,
)

__all__ = [
    "lti",
    "scn_core",
    "spikeOFC_model",
    "learning",
    "delay",
    "loop",
    "logging_utils",
]

