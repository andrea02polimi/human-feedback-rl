"""Direct expert/agent distribution-comparison diagnostics."""

import time

import numpy as np
from scipy.stats import rankdata

from human_feedback_rl.common.trajectory_generators import policy_action_log_probs


IMITATION_MAX_TRANSITIONS_PER_CLASS = 5_000
IMITATION_CLASSIFIER_STEPS = 100
IMITATION_CLASSIFIER_LR = 0.1
IMITATION_CLASSIFIER_L2 = 1e-4


class ImitationMetricsMixin:
    """Classifier two-sample diagnostics used by ``DemoAlgorithm``."""

    def _log_imitation_diagnostics(self) -> None:
        t0 = time.perf_counter()
        auc = self._state_action_classifier_auc(
            self.expert_trajectories, self.trajectories
        )
        if auc is not None:
            self.logger.record("imitation/state_action_auc", auc)
        self.logger.record_sum("time/imitation_diagnostics", time.perf_counter() - t0)

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
        batch_indices = np.random.default_rng().choice(
            len(squared_error), size=batch_size, replace=False
        )
        # Errore per dimensione (per una singola azione == |agente - esperto| su ciascuna dimensione).
        per_dim_error = np.sqrt(squared_error[batch_indices])
        print(f"[imitation] action_rmse per dimensione di {batch_size} azioni casuali:")
        for index, errors in zip(batch_indices, per_dim_error):
            formatted = ", ".join(f"{value:.6f}" for value in errors)
            print(f"  azione {index}: [{formatted}]")
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

    @classmethod
    def _state_action_classifier_auc(cls, expert_trajectories, agent_trajectories):
        expert_split = cls._trajectory_train_validation_split(expert_trajectories, seed=0)
        agent_split = cls._trajectory_train_validation_split(agent_trajectories, seed=1)
        if expert_split is None or agent_split is None:
            return None

        expert_train = cls._state_action_features(expert_split[0])
        expert_validation = cls._state_action_features(expert_split[1])
        agent_train = cls._state_action_features(agent_split[0])
        agent_validation = cls._state_action_features(agent_split[1])
        feature_sets = (expert_train, expert_validation, agent_train, agent_validation)
        if any(len(features) == 0 for features in feature_sets):
            return None
        if len({features.shape[1] for features in feature_sets}) != 1:
            return None

        expert_train, agent_train = cls._balance_feature_classes(expert_train, agent_train)
        expert_validation, agent_validation = cls._balance_feature_classes(
            expert_validation, agent_validation
        )

        train_x = np.concatenate([expert_train, agent_train], axis=0)
        train_y = np.concatenate([
            np.ones(len(expert_train), dtype=np.float64),
            np.zeros(len(agent_train), dtype=np.float64),
        ])
        validation_x = np.concatenate([expert_validation, agent_validation], axis=0)
        validation_y = np.concatenate([
            np.ones(len(expert_validation), dtype=np.float64),
            np.zeros(len(agent_validation), dtype=np.float64),
        ])

        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        train_x = (train_x - mean) / std
        validation_x = (validation_x - mean) / std

        weights = np.zeros(train_x.shape[1], dtype=np.float64)
        bias = 0.0
        for _ in range(cls.IMITATION_CLASSIFIER_STEPS):
            logits = np.clip(train_x @ weights + bias, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            residual = probabilities - train_y
            grad_weights = (
                train_x.T @ residual / len(train_x)
                + cls.IMITATION_CLASSIFIER_L2 * weights
            )
            grad_bias = float(residual.mean())
            weights -= cls.IMITATION_CLASSIFIER_LR * grad_weights
            bias -= cls.IMITATION_CLASSIFIER_LR * grad_bias

        scores = validation_x @ weights + bias
        return cls._binary_auc(validation_y, scores)

    @staticmethod
    def _trajectory_train_validation_split(trajectories, seed: int):
        trajectories = [trajectory for trajectory in trajectories if len(trajectory)]
        if len(trajectories) < 2:
            return None
        indices = np.arange(len(trajectories))
        np.random.default_rng(seed).shuffle(indices)
        validation_size = max(1, int(round(0.2 * len(indices))))
        validation_size = min(validation_size, len(indices) - 1)
        validation_indices = set(indices[:validation_size].tolist())
        train = [
            trajectory for index, trajectory in enumerate(trajectories)
            if index not in validation_indices
        ]
        validation = [
            trajectory for index, trajectory in enumerate(trajectories)
            if index in validation_indices
        ]
        return train, validation

    @staticmethod
    def _state_action_features(trajectories) -> np.ndarray:
        rows = []
        for trajectory in trajectories:
            for transition in trajectory:
                if transition.observation is None or transition.action is None:
                    continue
                observation = np.asarray(transition.observation, dtype=np.float64).reshape(-1)
                action = np.asarray(transition.action, dtype=np.float64).reshape(-1)
                row = np.concatenate([observation, action])
                if np.isfinite(row).all():
                    rows.append(row)
        if not rows:
            return np.empty((0, 0), dtype=np.float64)
        try:
            return np.stack(rows)
        except ValueError:
            return np.empty((0, 0), dtype=np.float64)

    @classmethod
    def _balance_feature_classes(cls, positive, negative):
        size = min(len(positive), len(negative), cls.IMITATION_MAX_TRANSITIONS_PER_CLASS)
        positive_indices = np.linspace(0, len(positive) - 1, size, dtype=int)
        negative_indices = np.linspace(0, len(negative) - 1, size, dtype=int)
        return positive[positive_indices], negative[negative_indices]

    @staticmethod
    def _binary_auc(labels: np.ndarray, scores: np.ndarray):
        positive = labels == 1
        n_positive = int(positive.sum())
        n_negative = int((~positive).sum())
        if n_positive == 0 or n_negative == 0:
            return None
        ranks = rankdata(scores, method="average")
        auc = (
            ranks[positive].sum() - n_positive * (n_positive + 1) / 2
        ) / (n_positive * n_negative)
        return float(auc)
