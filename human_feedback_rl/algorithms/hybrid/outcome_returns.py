"""Returns broken down by how the episode ended."""

import numpy as np
import torch as th

from human_feedback_rl.common.status import STATUS_NAMES

#: Terminal statuses only: a trajectory still running has not ended in anything.
TERMINAL_STATUS_NAMES = {
    index: name for index, name in enumerate(STATUS_NAMES) if name != "running"
}

FIELDS = ("raw_return", "norm_return", "disc_return",
          "mean_step_reward", "terminal_reward", "length")


def terminal_outcome(trajectory) -> str | None:
    """The status a trajectory ended in, or None if it did not end."""
    if len(trajectory) == 0:
        return None
    # An all-zero status would otherwise be read as "arrived".
    last = np.asarray(trajectory[-1].next_status, dtype=np.float64)
    if not trajectory[-1].done or not np.isclose(last.sum(), 1.0):
        return None
    return TERMINAL_STATUS_NAMES.get(int(np.argmax(last)))


class OutcomeReturnsMixin:
    """Mean predicted and true return per ego status."""

    def _log_outcome_returns(self) -> None:
        """Return under the *current* reward, by terminal outcome.

        ``disc_return`` is the discounted sum of the current normalized reward.
        It matches what SAC optimizes only when ``relabel_rewards=True``, and it
        leaves out the entropy term, so it is a proxy rather than SAC's exact
        objective. It is still enough to say something useful: if offroad scores
        higher than arrived, the reward is the problem, not SAC.

        For ``timeout`` the proxy is partial. SAC treats a timeout as a
        truncation and bootstraps the value beyond the episode, while this sum
        stops at the last step. The comparison holds for arrived, offroad and
        collided, which are true terminations.
        """
        if not self.trajectories:
            return

        for name, rows in self._returns_by_outcome().items():
            # Log the count even when zero, so "no episodes of this kind" reads
            # differently from "the metric did not run".
            self.logger.record(f"reward/outcome/{name}/count", len(rows), exclude="stdout")
            if not rows:
                continue
            values = np.asarray(rows, dtype=np.float64)
            for i, field in enumerate(FIELDS):
                self.logger.record(
                    f"reward/outcome/{name}/{field}", float(values[:, i].mean()),
                    exclude="stdout",
                )

    def _returns_by_outcome(self) -> dict:
        """One row of reward statistics per finished trajectory, grouped by outcome."""
        gamma = float(getattr(self.agent, "gamma", 1.0))
        buckets = {name: [] for name in TERMINAL_STATUS_NAMES.values()}
        self.reward_model.eval()
        with th.no_grad():
            for trajectory in self.trajectories:
                name = terminal_outcome(trajectory)
                if name is not None:
                    buckets[name].append(self._reward_summary(trajectory, gamma))
        self.reward_model.train()
        return buckets

    def _reward_summary(self, trajectory, gamma: float) -> tuple:
        """Raw, normalized and discounted return of one trajectory, and its shape."""
        obs = np.array([t.observation for t in trajectory], dtype=np.float32)
        acts = np.array([t.action for t in trajectory], dtype=np.float32)
        status = np.array([t.next_status for t in trajectory], dtype=np.float32)
        done = np.array([float(t.done) for t in trajectory], dtype=np.float32)

        raw = self.reward_model.predict_unnormalized(obs, acts, status, done)
        norm = self.reward_model.predict(obs, acts, status, done)
        discounts = gamma ** np.arange(len(norm), dtype=np.float64)
        return (
            float(raw.sum()), float(norm.sum()), float(np.sum(norm * discounts)),
            float(norm.mean()), float(norm[-1]), float(len(trajectory)),
        )
