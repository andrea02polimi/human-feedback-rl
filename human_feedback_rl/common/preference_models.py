from typing import Sequence

import numpy as np
import torch as th
import torch.nn as nn

from . import reward_nets
from .types import FragmentPair, Trajectory
import torch as th
import torch.nn as nn


class PreferenceModelFromReward(nn.Module):
    def __init__(
        self,
        reward_model: reward_nets.RewardNet,
    ):
        super().__init__()
        self.reward_model = reward_model

    def forward(self, fragment_pairs: Sequence[FragmentPair]) -> th.Tensor:
        probs = []

        for fragment_pair in fragment_pairs:
            rews1 = self.fragment_rewards(fragment_pair.frag1)
            rews2 = self.fragment_rewards(fragment_pair.frag2)
            probs.append(self.probability(rews1, rews2))

        return th.stack(probs)

    def fragment_rewards(self, fragment: Trajectory) -> th.Tensor:
        state, action, next_state, done = self._fragment_arrays(fragment)
        state_th, action_th, next_state_th, done_th = self.reward_model.preprocess(
            state,
            action,
            next_state,
            done,
        )
        return self.reward_model(state_th, action_th, next_state_th, done_th)

    def _fragment_arrays(
        self,
        fragment: Trajectory,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not fragment.transitions:
            raise ValueError("fragment must contain at least one transition")

        obs = np.stack([transition.obs for transition in fragment.transitions])
        actions = np.asarray([transition.action for transition in fragment.transitions])

        next_obs = np.empty_like(obs)
        if len(fragment.transitions) > 1:
            next_obs[:-1] = obs[1:]
        next_obs[-1] = obs[-1]

        done = np.zeros(len(fragment.transitions), dtype=np.float32)
        done[-1] = 1.0

        return obs, actions, next_obs, done

    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        return1 = rews1.sum()
        return2 = rews2.sum()
        return th.sigmoid(return1 - return2)



# def bradley_terry_loss(
#     sum_rewards_1,
#     sum_rewards_2,
#     preferences,
#     noise_probability: float = 0.1,
#     eps: float = 1e-8,
#     xp=th,
# ):
#     """Cross-entropy loss based on Bradley-Terry model."""
    
#     # Bradley-Terry probability with noise
#     logit = sum_rewards_1 - sum_rewards_2
#     probs = 1 / (1 + xp.exp(-logit))  # P(seg1 > seg2)
    
#     # Add noise: 10% chance of random response
#     probs = noise_probability * 0.5 + (1 - noise_probability) * probs
    
#     # Cross-entropy with preferences
#     loss = -(preferences[:, 0] * xp.log(probs + eps) + 
#              preferences[:, 1] * xp.log(1 - probs + eps))
    
#     return loss.mean()


# class BradleyTerryLoss(nn.Module):
#     """Cross-entropy loss based on Bradley-Terry model."""
    
#     def __init__(self, noise_probability: float = 0.1):
#         super().__init__()
#         self.noise_prob = noise_probability
        
#     def forward(self, 
#                 segment_rewards_1: th.Tensor,  # (batch, seq_len)
#                 segment_rewards_2: th.Tensor,
#                 preferences: th.Tensor):       # (batch, 2)
        
#         # Sum rewards over trajectory segments
#         sum_rewards_1 = segment_rewards_1.sum(dim=1)  # (batch,)
#         sum_rewards_2 = segment_rewards_2.sum(dim=1)
        
#         return bradley_terry_loss(
#             sum_rewards_1=sum_rewards_1,
#             sum_rewards_2=sum_rewards_2,
#             preferences=preferences,
#             noise_probability=self.noise_prob,
#             xp=th,
#         )