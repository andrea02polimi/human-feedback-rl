import numpy as np
from scipy.stats import pearsonr
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..loggers import MainLogger


def log_hacking_signals(
    true_rewards: List[float],
    model_rewards: List[float],
    logger: "MainLogger",
) -> None:
    """
    Compute and record reward-hacking detection signals per sampling batch.

    Keys logged:
        hack/reward_gap          — mean(model) - mean(true)  [positive = model over-estimates]
        hack/exploitation_index  — gap / std(true)           [>1.5 is a strong hacking signal]
        hack/true_reward_mean
        hack/true_reward_std
        hack/model_reward_mean
        hack/true_vs_model_corr  — Pearson r(true, model)    [<0.7 indicates decorrelation]
    """
    true_arr  = np.array(true_rewards,  dtype=np.float32)
    model_arr = np.array(model_rewards, dtype=np.float32)

    true_mean  = float(np.mean(true_arr))
    true_std   = float(np.std(true_arr))
    model_mean = float(np.mean(model_arr))
    gap        = model_mean - true_mean

    logger.record("hack/reward_gap",         gap)
    logger.record("hack/exploitation_index", gap / (true_std + 1e-8))
    logger.record("hack/true_reward_mean",   true_mean)
    logger.record("hack/true_reward_std",    true_std)
    logger.record("hack/model_reward_mean",  model_mean)

    if len(true_arr) > 2 and true_std > 1e-6 and np.std(model_arr) > 1e-6:
        corr, _ = pearsonr(true_arr, model_arr)
        if not np.isnan(corr):
            logger.record("hack/true_vs_model_corr", float(corr))