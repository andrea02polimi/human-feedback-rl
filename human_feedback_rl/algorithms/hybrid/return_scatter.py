"""Predicted return against true return, as a W&B scatter plot."""

import numpy as np
import wandb

OUTCOME_COLORS = {
    "arrived": "tab:green",
    "collided": "tab:red",
    "offroad": "tab:orange",
    "timeout": "tab:blue",
    "other": "tab:gray",
}


def align_to_true_scale(pred_steps, true_steps, running, lengths) -> np.ndarray:
    """Predicted returns, shifted and scaled onto the true-reward scale.

    The affine is fitted on running steps only, so the terminal rewards stay the
    signal we want to reconstruct. One shift and one scale are shared by every
    trajectory, so the ranking between trajectories does not move: only the
    units on the x axis change.

    No temperature enters here. It belongs to the oracle, not to the algorithm,
    and the affine would cancel it anyway, being fitted on the very predictions
    it would be applied to.
    """
    flat_pred = np.concatenate(pred_steps)
    flat_true = np.concatenate(true_steps)
    flat_running = np.concatenate(running)
    reference = flat_running if flat_running.any() else np.ones(len(flat_pred), dtype=bool)

    pred_std = flat_pred[reference].std()
    scale = (flat_true[reference].std() / pred_std) if pred_std > 1e-8 else 1.0
    shift = flat_true[reference].mean() - scale * flat_pred[reference].mean()

    return np.asarray([scale * p.sum() + shift * n for p, n in zip(pred_steps, lengths)])


def draw_scatter(pred_returns, true_returns, outcomes, log_class: str, iteration: int):
    """One point per trajectory, coloured by how the episode ended."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(5, 5))
    for outcome, color in OUTCOME_COLORS.items():
        mask = outcomes == outcome
        if mask.any():
            ax.scatter(pred_returns[mask], true_returns[mask],
                       color=color, label=outcome, alpha=0.7)

    ax.legend(title="Final outcome")
    ax.set_xlabel("predicted return (aligned to true, running-step affine)")
    ax.set_ylabel("true return")
    ax.set_title(f"{log_class} iter {iteration}")

    # Fix the y limits to the true returns, so the slider compares like frames.
    low, high = min(true_returns), max(true_returns)
    pad = 0.05 * (high - low) if high > low else 1.0
    ax.set_ylim(low - pad, high + pad)
    return figure


class ReturnScatterMixin:
    """One scatter point per trajectory, coloured by outcome."""

    def _log_return_scatter(self, trajectories, log_class: str, iteration: int) -> None:
        """Log the scatter as an image, so every iteration is kept.

        A ``wandb.Image`` rather than ``wandb.plot.scatter``: the media panel
        keeps one frame per step and the slider scrubs through them, instead of
        each new plot overwriting the last.
        """
        if wandb.run is None or len(trajectories) < 2:
            return

        pred_returns, true_returns, outcomes = self._scatter_points(trajectories)
        figure = draw_scatter(pred_returns, true_returns, outcomes, log_class, iteration)
        try:
            wandb.log({
                "iterations": iteration,
                f"{log_class}/return_scatter": wandb.Image(figure),
            }, commit=True)
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)

    def _scatter_points(self, trajectories):
        """Predicted return, true return and outcome, one per trajectory."""
        outcome_names = {
            self.STATUS_ARRIVED: "arrived",
            self.STATUS_COLLIDED: "collided",
            self.STATUS_OFFROAD: "offroad",
            self.STATUS_TIMEOUT: "timeout",
        }
        true_returns, outcomes = [], []
        pred_steps, true_steps, running, lengths = [], [], [], []

        for trajectory in trajectories:
            true_rewards, pred_rewards, _, status = self._run_reward_inference_with_std(trajectory)
            true_returns.append(float(true_rewards.sum()))
            pred_steps.append(np.asarray(pred_rewards, dtype=np.float64))
            true_steps.append(np.asarray(true_rewards, dtype=np.float64))
            running.append(status[:, self.STATUS_RUNNING] == 1)
            lengths.append(len(pred_rewards))

            final = np.asarray(trajectory[-1].next_status)
            outcomes.append(outcome_names.get(int(np.argmax(final)), "other")
                            if np.isclose(final.sum(), 1.0) else "other")

        pred_returns = align_to_true_scale(pred_steps, true_steps, running, lengths)
        return pred_returns, np.asarray(true_returns), np.asarray(outcomes)
