"""
Most general base class for all human-feedback RL algorithms.

Owns the three resources shared across every algorithm family:
  - environment, agent, logger, rng.

Concrete algorithm families (reward learning, imitation learning, …) inherit
from this and add their own components on top.
"""

import sys
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
from stable_baselines3.common.logger import HumanOutputFormat

from human_feedback_rl.common.loggers import Logger, WandbWriter


class BaseAlgorithm(ABC):
    """
    General base for all human-feedback RL algorithms.

    Provides:
      * ``env``    – the (vectorised) training environment.
      * ``agent``  – the policy being trained.
      * ``logger`` – SB3 Logger wired to WandB and a human-readable sink.
      * ``rng``    – shared NumPy random generator.

    Logger configuration (two complementary mechanisms):
      * ``log_folder``     – directory where SB3 Logger writes CSV/TensorBoard
                             files.  ``None`` disables file logging (default).
      * ``output_formats`` – explicit list of SB3 output-format objects.
                             When provided, overrides ``_output_formats()``.
                             When ``None``, ``_output_formats()`` is called
                             (the class-level override hook).

    Subclasses call ``super().__init__(env, agent, ...)`` and get all of the
    above for free.
    """

    def __init__(
        self,
        env,
        agent,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
    ):
        self.env    = env
        self.agent  = agent
        self.rng    = rng if rng is not None else np.random.default_rng()
        self.logger = Logger(
            folder=log_folder,
            output_formats=output_formats if output_formats is not None else self._output_formats(),
        )

    @property
    def venv(self):
        """Alias for self.env (kept for VecEnv-oriented algorithms)."""
        return self.env

    def _output_formats(self) -> list:
        """Default logger sinks used when no output_formats are passed to __init__.

        Override in a subclass to change the default sinks for that algorithm
        family without having to pass output_formats at every instantiation.
        """
        return [HumanOutputFormat(sys.stdout), WandbWriter()]

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        """Run the full training pipeline and return the trained agent."""
        ...
