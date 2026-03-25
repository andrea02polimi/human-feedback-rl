"""
Segment pair sampling strategies for preference collection.

"""

import random

import numpy as np


def sample_pair_random(segment_buffer):
    """Randomly sample two different segments from the buffer."""
    if len(segment_buffer) < 2:
        return None
    first_index, second_index = random.sample(range(len(segment_buffer)), 2)
    return segment_buffer[first_index], segment_buffer[second_index]


def disagreement_score(segment, reward_predictor) -> float:
    """
    Variance of per-ensemble-member total segment rewards.

    Higher variance means the ensemble members disagree more about the value of
    the segment — these are the most informative pairs to label (Section 3.2 of
    Christiano et al.).
    """
    frames  = np.array(segment.frames, dtype=np.float32)                    # (T, obs_dim)
    actions = np.array(getattr(segment, "actions", [0] * len(segment.frames)))  # (T,) or (T, act_dim)
    raw = reward_predictor.raw_rewards(frames, actions)    # (n_preds, T)
    member_totals = raw.sum(axis=-1).flatten()             # (n_preds,)
    return float(np.var(member_totals))


def sample_pair_by_disagreement(segment_buffer, reward_predictor, n_candidates: int):
    """
    Return the segment pair with highest ensemble disagreement from n_candidates
    random candidates (Christiano et al. Section 3.2).

    Disagreement is the sum of per-segment variances across ensemble members.
    Pairs on which the ensemble disagrees most are the most informative to label.
    Falls back to the best pair found even if disagreement is zero (untrained RP).
    """
    if len(segment_buffer) < 2:
        return None
    best_pair = None
    best_score = -1.0
    for _ in range(n_candidates):
        i, j = random.sample(range(len(segment_buffer)), 2)
        seg_a, seg_b = segment_buffer[i], segment_buffer[j]
        score = disagreement_score(seg_a, reward_predictor) + disagreement_score(
            seg_b, reward_predictor
        )
        if score > best_score:
            best_score = score
            best_pair = (seg_a, seg_b)
    return best_pair
