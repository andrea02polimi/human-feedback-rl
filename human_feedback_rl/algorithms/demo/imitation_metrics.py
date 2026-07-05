import numpy as np

from human_feedback_rl.common.trajectory_generators import policy_action_log_probs


class ImitationMetricsMixin:

    def _log_expert_imitation_errors(self) -> None:
        """Direct agent-vs-expert imitation errors over the expert dataset.

        Two metrics, both computed on every expert transition (trajectories are
        flattened to single ``(state, action)`` pairs):

        * ``imitation/action_rmse`` -- RMSE between the expert action and the agent's deterministic action (the SAC actor's mode; the continuous analogue of ``argmax``) for the same state.
        * ``imitation/expert_action_nll (Negative Log Likelihood)`` -- mean negative log-likelihood of the expert actions under the agent policy. This is the cross-entropy ``H(expert, agent) = KL(expert || agent) + H(expert)``, i.e. a KL surrogate that differs from the true KL only by the expert's entropy(constant w.r.t. the agent). The literal KL is not computable from samples alone because the dataset provides no expert action density.
        """
        observations, expert_actions = self._flatten_expert_transitions()
        if len(observations) == 0:
            return

        # SAC has no argmax; the deterministic action is the actor's mode. guardare common/distributions righe 74-77
        agent_actions, _ = self.agent.predict(observations, deterministic=True)
        # RESHAPE TO 2D arrays for RMSE computation, in case the actions are multi-dimensional.
        agent_actions = np.asarray(agent_actions, dtype=np.float64).reshape(len(observations), -1)
        expert_actions = expert_actions.reshape(len(observations), -1)

        squared_error = (agent_actions - expert_actions) ** 2
        self.logger.record("imitation/action_rmse", float(np.sqrt(np.mean(squared_error))))

        # Stampa su terminale l'RMSE per singola dimensione di 10 azioni casuali, a ogni iterazione.
        batch_size = min(10, len(squared_error))
        batch_indices = np.random.default_rng(0).choice(
            len(squared_error), size=batch_size, replace=False
        )
        # Errore per dimensione (per una singola azione == |agente - esperto| su ciascuna dimensione).
        per_dim_error = np.sqrt(squared_error[batch_indices])
        print(f"[imitation] action agent vs expert (and RMSE) per dimension of {batch_size} casual actions:")
        for index, errors in zip(batch_indices, per_dim_error):
            agent_formatted = ", ".join(f"{value:.6f}" for value in agent_actions[index])
            expert_formatted = ", ".join(f"{value:.6f}" for value in expert_actions[index])
            error_formatted = ", ".join(f"{value:.6f}" for value in errors)
            print(f"  action {index}:")
            print(f"    agent:  [{agent_formatted}]")
            print(f"    expert: [{expert_formatted}]")
            print(f"    rmse:    [{error_formatted}]")
        per_dim_rmse = np.sqrt(np.mean(squared_error, axis=0))
        for dim, value in enumerate(per_dim_rmse):
            self.logger.record(f"imitation/action_rmse_dim{dim}", float(value))

        # KL surrogate: -E_expert[log pi_agent(a | s)] via the SAC-aware helper,
        # which handles the tanh-squash / action-scaling coordinate change.
        log_probs = np.asarray(
            policy_action_log_probs(self.agent, observations, expert_actions),
            dtype=np.float64,
        )
        # Only consider finite log-probabilities, which can be violated if the agent's policy is very far from the expert's actions. np.isfinite() is used to filter out any non-finite values before computing the mean log-probability. This ensures that the metric is computed only on valid log-probabilities, avoiding issues with NaN or infinite values that could arise from numerical instability or extreme policy outputs.
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