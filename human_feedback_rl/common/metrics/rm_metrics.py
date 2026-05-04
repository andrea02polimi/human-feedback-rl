import numpy as np
import torch as th
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..reward_nets import RewardEnsemble
    from ..types import Trajectory
    from ..loggers import MainLogger


def log_ensemble_uncertainty(
    reward_model: "RewardEnsemble",
    trajectories: List["Trajectory"],
    logger: "MainLogger",
) -> None:
    """
    Log per-trajectory ensemble disagreement (mean and std of reward_std across
    trajectories).  Requires at least 2 ensemble members; no-ops otherwise.

    Keys: rm/uncertainty_mean, rm/uncertainty_std
    """
    if not hasattr(reward_model, "members") or len(reward_model.members) < 2:
        return

    uncertainties: List[float] = []
    for traj in trajectories:
        obs  = np.array([t.observation for t in traj], dtype=np.float32)
        acts = np.array([t.action      for t in traj], dtype=np.float32)
        _, std = reward_model.predict_mean_std(obs, acts)  # (T,)
        uncertainties.append(float(std.mean()))

    logger.record("rm/uncertainty_mean", float(np.mean(uncertainties)))
    logger.record("rm/uncertainty_std",  float(np.std(uncertainties)))