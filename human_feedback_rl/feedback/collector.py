"""
PreferenceCollector — encapsulates the segment buffer and pair-sampling logic
used by the preference worker subprocess.

Extracted from _preference_worker in scripts/train_christiano.py.
"""

import functools
import queue
import time

import numpy as np

from human_feedback_rl.feedback.sampling import (
    sample_pair_random,
    sample_pair_by_disagreement,
)
from human_feedback_rl.reward_models.ensemble import RewardPredictorEnsemble
from human_feedback_rl.reward_models.networks import SumoRewardNetwork


class PreferenceCollector:
    """
    Manages the segment circular buffer and reward-predictor-based pair selection
    for the preference worker.

    Args:
        config:                          Hydra DictConfig (preferences.*, training.*, resources.*)
        reward_predictor_checkpoint_dir: directory where the main process saves RP checkpoints
        observation_dim:                 obs_dim used to instantiate the RP ensemble
    """

    def __init__(self, config, reward_predictor_checkpoint_dir: str, observation_dim: int):
        self._config = config
        self._checkpoint_dir = reward_predictor_checkpoint_dir
        self._max_segs = config.preferences.max_segs
        self._disagreement_candidates = config.preferences.disagreement_candidates
        self._max_query_interval = config.preferences.max_query_interval
        self._total_env_steps_target = config.training.total_env_steps

        self._segment_buffer = []
        self._buffer_write_idx = 0
        self._total_labeled = 0

        # Inference-only RP for disagreement-based pair selection.
        self._rp = RewardPredictorEnsemble(
            core_network=functools.partial(SumoRewardNetwork, obs_dim=observation_dim),
            log_dir=None,
            device=config.resources.device,
        )
        self._rp_loaded = False

    # ------------------------------------------------------------------

    def add_segment(self, seg) -> None:
        """Insert a segment into the circular buffer."""
        if len(self._segment_buffer) < self._max_segs:
            self._segment_buffer.append(seg)
        else:
            self._segment_buffer[self._buffer_write_idx % self._max_segs] = seg
            self._buffer_write_idx += 1

    def drain_pipe(self, segment_pipe, max_drain: int = 8) -> None:
        """Try to get up to max_drain segments from segment_pipe; stop on empty."""
        for _ in range(max_drain):
            try:
                seg = segment_pipe.get(timeout=0.5)
                self.add_segment(seg)
            except queue.Empty:
                break

    def refresh_rp(self, ready_event) -> None:
        """Load or reload the latest RP checkpoint for disagreement scoring."""
        if not self._rp_loaded and ready_event.is_set():
            latest = RewardPredictorEnsemble.latest_checkpoint(self._checkpoint_dir)
            if latest:
                try:
                    self._rp.load(latest)
                    self._rp_loaded = True
                except Exception:
                    pass
        elif (
            self._rp_loaded
            and self._total_labeled % 50 == 0
            and self._total_labeled > 0
        ):
            latest = RewardPredictorEnsemble.latest_checkpoint(self._checkpoint_dir)
            if latest:
                try:
                    self._rp.load(latest)
                except Exception:
                    pass

    def sample_pair(self):
        """
        Return a (seg1, seg2) pair.
        Uses disagreement-based selection when RP is loaded, random otherwise.
        Returns None if the buffer has fewer than 2 segments.
        """
        if len(self._segment_buffer) < 2:
            return None

        if self._rp_loaded:
            return sample_pair_by_disagreement(
                self._segment_buffer,
                self._rp,
                n_candidates=self._disagreement_candidates,
            )
        else:
            return sample_pair_random(self._segment_buffer)

    def on_labeled(self, shared_env_steps, ready_event) -> None:
        """
        Increment the labeled counter and apply query-annealing sleep.

        Query annealing: after the RP is ready the inter-query sleep grows
        linearly with env steps so the policy has time to explore before being
        judged (Christiano et al. Section 3.2).
        """
        self._total_labeled += 1

        if ready_event.is_set() and self._total_env_steps_target > 0:
            fraction = min(
                1.0,
                shared_env_steps.value / self._total_env_steps_target,
            )
            sleep_s = self._max_query_interval * fraction
            if sleep_s > 0:
                time.sleep(sleep_s)
