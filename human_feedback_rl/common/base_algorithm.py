"""
Most general base class for all human-feedback RL algorithms.

Owns the three resources shared across every algorithm family:
  - environment, agent, logger, rng.

Concrete algorithm families (reward learning, imitation learning, …) inherit
from this and add their own components on top.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from human_feedback_rl.common.loggers import Logger, WandbWriter, make_human_output_format


class BaseAlgorithm(ABC):
    """Base for the algorithms: env, agent, logger and rng.

    log_folder points the SB3 logger at a directory, or disables file logging when
    None. output_formats overrides the formats chosen by _output_formats().
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
        return [make_human_output_format(), WandbWriter()]

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        """Run the full training pipeline and return the trained agent."""
        ...
