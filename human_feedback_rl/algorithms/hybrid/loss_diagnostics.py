"""What the reward losses look like while they are being optimised."""

import numpy as np
import torch as th


class LossDiagnosticsMixin:
    """Per-iteration diagnostics of the demonstration and preference losses."""

    def _log_reward_loss_diagnostics(self) -> None:
        """Record the demonstration loss and the quantities it is built from."""
        if not self.trajectories:
            return

        self.reward_model.eval()
        with th.no_grad():
            expert_returns, model_returns, _, _ = self._diagnostic_returns(self.reward_model)
            all_returns = th.cat([expert_returns, model_returns], dim=0)
            if not th.isfinite(all_returns).all():
                raise FloatingPointError("Non-finite trajectory returns in reward diagnostics.")

            expert_term = -expert_returns.mean()
            model_mean = model_returns.mean()
            margin = expert_returns.mean() - model_mean
            abs_mean = all_returns.abs().mean()
            return_std = all_returns.std(unbiased=False)

            self._record_losses({
                "expert_return_mean": float(expert_returns.mean()),
                "model_return_mean": float(model_mean),
                "expert_model_margin": float(margin),
                "return_std": float(return_std),
                "return_abs_mean": float(abs_mean),
                "return_min": float(all_returns.min()),
                "return_max": float(all_returns.max()),
            })

            if self.loss_type == "demo_1":
                self._record_losses({
                    "loss": float(expert_term + model_mean),
                    "demo_1_margin": float(margin),
                    "demo_1_scale_std": float(return_std),
                    "demo_1_scale_abs": float(abs_mean),
                })
            else:
                self._log_demo_2_partition(expert_term, expert_returns, all_returns)
        self.reward_model.train()

    def _log_demo_2_partition(self, expert_term, expert_returns, all_returns) -> None:
        """How much of the demo_2 partition the experts take.

        The softmax mass and the effective sample size say whether the estimate
        rests on the whole batch or on a handful of trajectories.
        """
        partition = th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))
        weights = th.softmax(all_returns, dim=0)
        n_expert = len(expert_returns)
        top1_weight, top1_index = weights.max(dim=0)
        ess = 1.0 / weights.pow(2).sum()

        self._record_losses({
            "loss": float(expert_term + partition),
            "demo_2_partition_all": float(partition),
            "demo_2_expert_softmax_mass": float(weights[:n_expert].sum()),
            "demo_2_model_softmax_mass": float(weights[n_expert:].sum()),
            "demo_2_top1_softmax_weight": float(top1_weight),
            "demo_2_top1_is_expert": float((top1_index < n_expert).item()),
            "demo_2_effective_sample_size": float(ess),
        })
        self._log_ess_fraction("reward/demo_2", ess, len(weights))

    def _record_losses(self, values: dict) -> None:
        for name, value in values.items():
            self.logger.record(f"reward/{name}", value, exclude="stdout")

    def _log_ess_fraction(self, log_prefix: str, ess: th.Tensor, n_items: int) -> None:
        """Warn when the partition rests on too few trajectories."""
        ess_fraction = float(ess) / n_items
        self.logger.record(
            f"{log_prefix}_effective_sample_fraction", ess_fraction, exclude="stdout"
        )
        if ess_fraction < 0.1:
            self.logger.warn(
                f"Low effective sample fraction for {self.loss_type}: {ess_fraction:.3f}"
            )
