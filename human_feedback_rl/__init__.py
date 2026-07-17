"""Reward learning from human feedback (preferences and demonstrations) for SUMO driving.

The package exposes one algorithm, :class:`HybridAlgorithm`: with both
feedback sources active it is the hybrid method; with ``demo_weight=0`` it is
the preference-only baseline and with ``total_queries=0`` the
demonstration-only baseline.
"""

__version__ = "0.4.0"

from human_feedback_rl.algorithms import HybridAlgorithm

__all__ = [
    "HybridAlgorithm",
    "__version__",
]
