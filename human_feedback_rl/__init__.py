"""Human-feedback RL (preference- and demonstration-based reward learning, DAgger) for SUMO driving."""

__version__ = "0.3.0"

from human_feedback_rl.algorithms import (
    DaggerAlgorithm,
    HybridAlgorithm,
)

__all__ = [
    "DaggerAlgorithm",
    "HybridAlgorithm",
    "__version__",
]
