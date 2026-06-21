"""Direct expert/agent distribution-comparison diagnostics."""

import time

import numpy as np
from scipy.stats import rankdata


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
