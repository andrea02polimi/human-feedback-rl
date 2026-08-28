"""How far the replay buffer has drifted from the current reward."""

import numpy as np

PREFIX = "replay_relabel_debug"


class ReplayStalenessMixin:
    """Distance between stored rewards and what the model predicts now."""

    def _log_replay_reward_staleness(self, batch_size: int = 2048) -> None:
        """Compare the rewards SAC has in its buffer with today's predictions."""
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        if replay_buffer is None or not hasattr(replay_buffer, "sample_reward_staleness"):
            return
        batch = replay_buffer.sample_reward_staleness(batch_size, self._debug_rng)
        if batch is None:
            return

        stored, current = batch
        delta = current - stored
        abs_delta = np.abs(delta)
        stored_std = float(np.std(stored))
        current_std = float(np.std(current))
        denominator = current_std if current_std > 1e-8 else 1.0
        # Whether the critic sees the current reward at all, or older ones.
        relabel_enabled = float(getattr(replay_buffer, "relabel_rewards", False))

        self._record_staleness({
            "sample_size": len(stored),
            "stored_reward_mean": float(np.mean(stored)),
            "current_reward_mean": float(np.mean(current)),
            "stored_reward_std": stored_std,
            "current_reward_std": current_std,
            "delta_mean": float(np.mean(delta)),
            "delta_std": float(np.std(delta)),
            "delta_abs_mean": float(np.mean(abs_delta)),
            "delta_abs_p95": float(np.percentile(abs_delta, 95)),
            "staleness_ratio": float(np.mean(abs_delta) / denominator),
            "sign_flip_frac": float(np.mean(np.sign(stored) != np.sign(current))),
            "relabel_enabled": relabel_enabled,
            "critic_uses_current_reward": relabel_enabled,
        })

        if stored_std > 1e-8 and current_std > 1e-8:
            correlation = np.corrcoef(stored, current)[0, 1]
            self._record_staleness({"stored_current_corr": float(correlation)})

    def _record_staleness(self, values: dict) -> None:
        for name, value in values.items():
            self.logger.record(f"{PREFIX}/{name}", value, exclude="stdout")
