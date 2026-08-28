import numpy as np

from human_feedback_rl.common.trajectory_generators import policy_action_log_probs


class ImitationMetricsMixin:
    """Agent-vs-expert imitation error metrics used by ``DemoAlgorithm``."""

    def _log_expert_imitation_errors(self) -> None:
        """Agent-versus-expert error over the whole expert dataset.

        action_rmse compares the expert action with the agent's deterministic one.
        expert_action_nll is the cross-entropy of the expert actions under the policy,
        a KL surrogate: the dataset carries no expert action density.
        """
        observations, expert_actions = self._flatten_expert_transitions()
        if len(observations) == 0:
            return

        agent_actions, _ = self.agent.predict(observations, deterministic=True)
        agent_actions = np.asarray(agent_actions, dtype=np.float64).reshape(len(observations), -1)
        expert_actions = expert_actions.reshape(len(observations), -1)

        squared_error = (agent_actions - expert_actions) ** 2
        self.logger.record("imitation/action_rmse", float(np.sqrt(np.mean(squared_error))))

        per_dim_rmse = np.sqrt(np.mean(squared_error, axis=0))
        for dim, value in enumerate(per_dim_rmse):
            self.logger.record(f"imitation/action_rmse_dim{dim}", float(value))

        # KL surrogate: -E_expert[log pi_agent(a | s)] via the SAC-aware helper,
        # which handles the tanh-squash / action-scaling coordinate change.
        log_probs = np.asarray(
            policy_action_log_probs(self.agent, observations, expert_actions),
            dtype=np.float64,
        )
        # Expert actions far outside the agent's current policy can produce
        # non-finite log-probs; the NLL is computed on the finite ones only.
        log_probs = log_probs[np.isfinite(log_probs)]
        if len(log_probs):
            self.logger.record("imitation/expert_action_nll", float(-log_probs.mean()))

    def _flatten_expert_transitions(self):
        """Flatten expert trajectories to stacked ``(observations, actions)`` arrays."""
        observations, actions = [], []
        for trajectory in self.expert_trajectories:
            for transition in trajectory:
                if transition.observation is None or transition.action is None:
                    continue
                observation = np.asarray(transition.observation, dtype=np.float64).reshape(-1)
                action = np.asarray(transition.action, dtype=np.float64).reshape(-1)
                if np.isfinite(observation).all() and np.isfinite(action).all():
                    observations.append(observation)
                    actions.append(action)
        if not observations:
            empty = np.empty((0, 0), dtype=np.float64)
            return empty, empty
        return np.stack(observations), np.stack(actions)
