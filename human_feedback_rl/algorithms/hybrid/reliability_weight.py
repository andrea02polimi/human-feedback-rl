"""Estimating alpha once per iteration, and publishing what it is made of."""

import time
import numpy as np
from human_feedback_rl.algorithms.hybrid.alpha_estimation import (
    ALPHA_MIN_PREFS,
    estimate_alpha,
)
from human_feedback_rl.common.preference_losses import preference_labels_tensor


class ReliabilityWeightMixin:
    """Owns this iteration's alpha: when it is measured, and what it logs."""

    def _alpha_weight(self, member) -> float:
        """Weight on the demonstration channel for the current iteration.

        Read-only: the value comes from ``_estimate_alpha``, which runs once at
        the start of reward training, at the parameters where the weight will
        be applied. Falls back to 1 (demonstrations only) when there is no
        estimate for this member.
        """
        estimate = self._alpha_current.get(id(member))
        return 1.0 if estimate is None else estimate.alpha

    def _alpha_is_active(self) -> bool:
        """True when alpha is estimated rather than pinned to the fallback."""
        estimates = [
            self._alpha_current.get(id(member)) for member in self.reward_model.members
        ]
        present = [e for e in estimates if e is not None]
        return bool(present) and all(not e.pinned for e in present)

    def _estimate_alpha(self) -> None:
        """Estimate this iteration's alpha, before any gradient step.

        The weight describes the noise at a point in parameter space, so it is measured
        where it will be applied. The rollout it needs comes from the diagnostics RNG,
        so measuring never moves the training draws.
        """
        if self.gcl_fusion == "norm_balance":
            return                       # that fusion does not use alpha
        self._alpha_current = {}
        if self.demo_weight <= 0.0 or not self.trajectories:
            return                       # no demonstration channel
        if self.loss_type != "demo_2":
            raise NotImplementedError(
                "alpha estimation implements the demo_2 per-sample "
                f"decomposition; got loss_type={self.loss_type!r}."
            )

        pref_batch = (
            self.dataset_train.get_all() if len(self.dataset_train) else None
        )
        n_model = min(self.batch_size_model, len(self.trajectories))
        model_indices = self._grad_probe_rng.choice(
            len(self.trajectories), size=n_model, replace=False
        )
        model_trajs = [self.trajectories[i] for i in model_indices]

        t0 = time.perf_counter()
        for member in self.reward_model.members:
            params = [p for p in member.parameters() if p.requires_grad]
            smooth = (
                self._smoothed_labels(preference_labels_tensor(pref_batch.preferences))
                if pref_batch is not None else None
            )
            self._alpha_current[id(member)] = estimate_alpha(
                member,
                params,
                pref_batch,
                smooth,
                self.expert_trajectories,
                model_trajs,
                batch_size_pref=self.batch_size_pref,
                batch_size_expert=self.batch_size_expert,
                min_prefs=ALPHA_MIN_PREFS,
                eps=self.alpha_eps,
            )
        self.logger.record("time/estimate_alpha", time.perf_counter() - t0)
        self._log_alpha_estimate()

    def _log_alpha_estimate(self) -> None:
        """Publish the two dispersions that make up alpha, and their parts.

        ``alpha/S_*`` is the sanity check: it is the variance of the gradient
        the optimizer actually applies, so it has to fall as the budget grows.
        """
        estimates = [
            self._alpha_current.get(id(member)) for member in self.reward_model.members
        ]
        estimates = [e for e in estimates if e is not None]
        if not estimates:
            return
        self.logger.record(
            "reward/hybrid_alpha",
            float(np.mean([e.alpha for e in estimates])),
            exclude="stdout",
        )
        self.logger.record(
            "reward/hybrid_alpha_active",
            float(self._alpha_is_active()),
            exclude="stdout",
        )
        for name, channel in (("pref", "pref"), ("demo", "demo")):
            values = [getattr(e, channel) for e in estimates]
            values = [v for v in values if v is not None]
            if not values:
                continue
            for key, attr in (
                ("V", "process_var"),
                ("S", "mean_var"),
                ("cv2", "cv2"),
                ("gradmean_norm_sq", "mean_norm_sq"),
                ("n", "n"),
                ("batch", "batch"),
            ):
                self.logger.record(
                    f"alpha/{key}_{name}",
                    float(np.mean([getattr(v, attr) for v in values])),
                    exclude="stdout",
                )
