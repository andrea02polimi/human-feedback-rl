"""Everything the algorithm logs about the reward it is learning.

None of it changes training. The parts live in one file each:

    loss_diagnostics    what the two losses look like while being optimised
    reward_validation   how well the reward ranks and separates trajectories
    return_scatter      predicted return against true return
    outcome_returns     returns broken down by how the episode ended
    replay_staleness    drift between stored rewards and current predictions
"""

from human_feedback_rl.algorithms.hybrid.loss_diagnostics import LossDiagnosticsMixin
from human_feedback_rl.algorithms.hybrid.outcome_returns import OutcomeReturnsMixin
from human_feedback_rl.algorithms.hybrid.replay_staleness import ReplayStalenessMixin
from human_feedback_rl.algorithms.hybrid.return_scatter import ReturnScatterMixin
from human_feedback_rl.algorithms.hybrid.reward_validation import RewardValidationMixin


class RewardDiagnosticsMixin(
    LossDiagnosticsMixin,
    RewardValidationMixin,
    ReturnScatterMixin,
    OutcomeReturnsMixin,
    ReplayStalenessMixin,
):
    """Non-training reward diagnostics used by ``HybridAlgorithm``."""
