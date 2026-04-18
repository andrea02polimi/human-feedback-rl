"""
ChristianoDemoAlgorithm: extends ChristianoAlgorithm with an expert demo loss.

The reward model loss becomes:
    L_total = L_preference + λ * L_demo
where:
    L_demo = -mean_t R_hat(o_t^expert, a_t^expert)

Expert segments are collected during each agent rollout: for every trajectory
visited by the agent we query the expert policy on the same observations and
record the actions it would have taken.  This keeps the expert dataset aligned
with the state distribution the agent actually visits.

Usage
-----
    from stable_baselines3 import SAC
    expert = SAC.load("expert.zip", env=env)

    algo = ChristianoDemoAlgorithm(
        env=env,
        agent=sac_agent,
        rng=rng,
        expert_policy=expert,
        demo_loss_weight=0.1,
    )
    algo.train(total_timesteps=500_000)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from human_feedback_rl.common import (
    EnsembleRewardModel,
    RewardNet,
    Segment,
    Trajectory,
    Transition,
)
from human_feedback_rl.common.core import Preference

from .christiano_algorithm import ChristianoAlgorithm


# ---------------------------------------------------------------------------
# Expert dataset
# ---------------------------------------------------------------------------


class ExpertDataset:
    """
    Fixed-size circular buffer of expert Segment objects.

    Populated during agent rollouts by querying the expert policy at the states
    the agent visits.  The buffer is never reset during training.
    """

    def __init__(self, max_size: int = 5000):
        self._segments: List[Segment] = []
        self.max_size = max_size

    def add(self, segment: Segment) -> None:
        self._segments.append(segment)
        if len(self._segments) > self.max_size:
            self._segments.pop(0)

    def sample(self, n: int, rng: np.random.Generator) -> List[Segment]:
        if not self._segments:
            return []
        idxs = rng.integers(len(self._segments), size=n)
        return [self._segments[i] for i in idxs]

    def __len__(self) -> int:
        return len(self._segments)


# ---------------------------------------------------------------------------
# Reward model with demo loss
# ---------------------------------------------------------------------------


class EnsembleRewardModelWithDemo(EnsembleRewardModel):
    """
    Adds a demo loss term to the standard Bradley-Terry preference loss.

    At each gradient step:
        loss = L_preference + demo_loss_weight * L_demo
    where:
        L_demo = -mean_{seg in batch} mean_t R_hat(o_t^expert, a_t^expert)

    The demo loss always uses the per-step mean (not sum) to be length-agnostic,
    regardless of the normalize_by_length setting of the preference loss.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_networks: int = 3,
        hidden_size: int = 256,
        lr: float = 3e-4,
        l2_reg: float = 1e-4,
        normalize_by_length: bool = False,
        device: str = "cpu",
        expert_dataset: Optional[ExpertDataset] = None,
        demo_loss_weight: float = 0.1,
        expert_batch_size: int = 32,
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_networks=n_networks,
            hidden_size=hidden_size,
            lr=lr,
            l2_reg=l2_reg,
            normalize_by_length=normalize_by_length,
            device=device,
        )
        self.expert_dataset = expert_dataset
        self.demo_loss_weight = demo_loss_weight
        self.expert_batch_size = expert_batch_size

    def train(
        self,
        dataset,
        n_steps: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        if len(dataset) < batch_size:
            return {}

        has_expert = (
            self.expert_dataset is not None
            and len(self.expert_dataset) >= self.expert_batch_size
        )

        total_pref_loss = 0.0
        total_demo_loss = 0.0

        for net, optimizer in zip(self.networks, self.optimizers):
            net.train()
            net_pref = 0.0
            net_demo = 0.0

            for _ in range(n_steps):
                optimizer.zero_grad()

                pref_batch = dataset.sample(batch_size, rng)
                loss = self._preference_loss(net, pref_batch)
                net_pref += loss.item()

                if has_expert:
                    expert_batch = self.expert_dataset.sample(self.expert_batch_size, rng)
                    demo_loss = self._demo_loss(net, expert_batch)
                    net_demo += demo_loss.item()
                    loss = loss + self.demo_loss_weight * demo_loss

                loss.backward()
                optimizer.step()

            net.eval()
            total_pref_loss += net_pref / n_steps
            if has_expert:
                total_demo_loss += net_demo / n_steps

        metrics: Dict[str, float] = {
            "reward_model/loss": total_pref_loss / self.n_networks
        }
        if has_expert:
            metrics["reward_model/demo_loss"] = total_demo_loss / self.n_networks

        return metrics

    def _demo_loss(self, net: RewardNet, expert_segments: List[Segment]) -> torch.Tensor:
        """L_demo = -log σ(mean_t R_hat(o_t^expert, a_t^expert))."""
        total = torch.tensor(0.0, device=self.device)
        for seg in expert_segments:
            obs_t = torch.tensor(seg.obs, dtype=torch.float32, device=self.device)
            act_t = torch.tensor(seg.actions, dtype=torch.float32, device=self.device)
            r_mean = net(obs_t, act_t).mean()
            total = total + F.logsigmoid(r_mean)
        return -total / len(expert_segments)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


class ChristianoDemoAlgorithm(ChristianoAlgorithm):
    """
    ChristianoAlgorithm extended with expert demonstration loss and SFT pre-training.

    Parameters
    ----------
    expert_policy : stable_baselines3 policy (pre-loaded)
        Expert policy used to label agent-visited states.  Must expose
        predict(obs, deterministic=True) returning (actions, states).
    demo_loss_weight : float
        Weight λ for the demo loss term in the reward model.
    expert_dataset_max_size : int
        Maximum number of expert segments in the buffer.
    expert_batch_size : int
        Number of expert segments sampled per RM gradient step.

    All other parameters are forwarded unchanged to ChristianoAlgorithm.
    """

    def __init__(
        self,
        env,
        agent,
        rng: np.random.Generator,
        expert_policy,
        demo_loss_weight: float = 0.1,
        expert_dataset_max_size: int = 5000,
        expert_batch_size: int = 32,
        # ── forwarded to base class ────────────────────────────────────
        reward_model_n_networks: int = 3,
        reward_model_hidden_size: int = 256,
        reward_model_lr: float = 3e-4,
        reward_model_l2: float = 1e-4,
        segment_length: Optional[int] = 50,
        episode_length_estimate: int = 200,
        preference_dataset_max_size: int = 3000,
        query_schedule: str = "constant",
        device: str = "cpu",
    ):
        super().__init__(
            env=env,
            agent=agent,
            rng=rng,
            reward_model_n_networks=reward_model_n_networks,
            reward_model_hidden_size=reward_model_hidden_size,
            reward_model_lr=reward_model_lr,
            reward_model_l2=reward_model_l2,
            segment_length=segment_length,
            episode_length_estimate=episode_length_estimate,
            preference_dataset_max_size=preference_dataset_max_size,
            query_schedule=query_schedule,
            device=device,
        )

        self.expert_policy = expert_policy
        self.expert_dataset = ExpertDataset(max_size=expert_dataset_max_size)

        # Replace the base reward model with the demo-aware version.
        obs_dim: int = env.observation_space.shape[0]
        action_dim: int = env.action_space.shape[0]
        self.reward_model = EnsembleRewardModelWithDemo(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_networks=reward_model_n_networks,
            hidden_size=reward_model_hidden_size,
            lr=reward_model_lr,
            l2_reg=reward_model_l2,
            normalize_by_length=(segment_length is None),
            device=device,
            expert_dataset=self.expert_dataset,
            demo_loss_weight=demo_loss_weight,
            expert_batch_size=expert_batch_size,
        )

        # SFT hyper-parameters set during train()
        self._n_sft_steps: int = 200
        self._sft_batch_size: int = 32

    # ------------------------------------------------------------------
    # Rollout overrides — collect expert segments after each rollout
    # ------------------------------------------------------------------

    def _collect_rollout_off_policy(
        self, n_steps_per_env: int
    ) -> Tuple[List[Trajectory], Dict[str, float]]:
        trajectories, stats = super()._collect_rollout_off_policy(n_steps_per_env)
        self._collect_expert_segments(trajectories)
        return trajectories, stats

    def _collect_rollout_on_policy(
        self, n_steps_per_env: int
    ) -> Tuple[List[Trajectory], Dict[str, float]]:
        trajectories, stats = super()._collect_rollout_on_policy(n_steps_per_env)
        self._collect_expert_segments(trajectories)
        return trajectories, stats

    # ------------------------------------------------------------------
    # Expert segment collection
    # ------------------------------------------------------------------

    def _collect_expert_segments(self, trajectories: List[Trajectory]) -> None:
        """
        For each eligible trajectory, query the expert policy on the agent's
        visited observations and store the resulting (obs, expert_action) pairs
        as an expert Segment in self.expert_dataset.

        Eligibility:
        - Full-episode mode (segment_length=None): only completed episodes
          (last transition has done=True).
        - Fixed-length mode: trajectories with at least segment_length steps;
          a random window of exactly segment_length is sampled.
        """
        for traj in trajectories:
            if not traj.transitions:
                continue

            if self.segment_length is None:
                if not traj.transitions[-1].done:
                    continue
                selected = traj.transitions
            else:
                if len(traj) < self.segment_length:
                    continue
                max_start = len(traj) - self.segment_length
                start = int(self.rng.integers(0, max_start + 1))
                selected = traj.transitions[start : start + self.segment_length]

            obs_batch = np.stack([t.obs for t in selected])
            expert_actions = self.expert_policy.predict(obs_batch)

            expert_transitions = [
                Transition(
                    obs=selected[i].obs,
                    action=expert_actions[i].copy(),
                    true_reward=selected[i].true_reward,
                    done=selected[i].done,
                )
                for i in range(len(selected))
            ]
            self.expert_dataset.add(Segment(expert_transitions))

    # ------------------------------------------------------------------
    # SFT pre-training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        n_initial_queries: int = 200,
        n_queries_per_iter: int = 10,
        reward_model_train_steps: int = 200,
        reward_model_batch_size: int = 64,
        n_sft_steps: int = 500,
        sft_batch_size: int = 32,
    ) -> None:
        self._n_sft_steps = n_sft_steps
        self._sft_batch_size = sft_batch_size
        super().train(
            total_timesteps=total_timesteps,
            n_initial_queries=n_initial_queries,
            n_queries_per_iter=n_queries_per_iter,
            reward_model_train_steps=reward_model_train_steps,
            reward_model_batch_size=reward_model_batch_size,
        )

    def _pretraining_phase(
        self,
        n_initial_queries: int,
        reward_model_train_steps: int,
        reward_model_batch_size: int,
    ) -> None:
        # 1. Raccolta demo esperte
        seg_len = self.segment_length or self._episode_length_estimate
        n_expert_steps = int(np.ceil(
            1.5 * self._sft_batch_size * seg_len / self.env.num_envs
        ))
        print(f"[Pre-training] Collecting expert demos ({n_expert_steps} steps/env)...")
        self._collect_expert_demos(n_expert_steps)
        print(f"[Pre-training] Expert dataset: {len(self.expert_dataset)} segments")

        # 2. SFT dell'agente tramite BC
        print(f"[Pre-training] SFT: {self._n_sft_steps} BC steps...")
        self._sft_phase(self._n_sft_steps, self._sft_batch_size)

        # 3+4. Raccolta preferenze + pre-training RM (parent)
        super()._pretraining_phase(n_initial_queries, reward_model_train_steps, reward_model_batch_size)

    def _collect_expert_demos(self, n_steps_per_env: int) -> None:
        """Roll out the expert for n_steps_per_env steps, storing Segments in expert_dataset.

        Segments are sliced with the same logic as _collect_expert_segments:
        - segment_length=None: one Segment per completed episode
        - segment_length=L: non-overlapping windows of length L per episode
        """
        obs = self.env.reset()
        self.agent._last_obs = obs

        active: List[List[Transition]] = [[] for _ in range(self.env.num_envs)]

        for _ in range(n_steps_per_env):
            expert_actions = self.expert_policy.predict(obs)
            next_obs, true_rewards, dones, _ = self.env.step(expert_actions)

            for i in range(self.env.num_envs):
                active[i].append(Transition(
                    obs=obs[i].copy(),
                    action=expert_actions[i].copy(),
                    true_reward=float(true_rewards[i]),
                    done=bool(dones[i]),
                ))
                if dones[i]:
                    self._store_expert_episode(active[i])
                    active[i] = []

            obs = next_obs

        # flush partial trajectories long enough for at least one segment
        for traj in active:
            if self.segment_length is None:
                if traj:
                    self.expert_dataset.add(Segment(traj))
            elif len(traj) >= self.segment_length:
                self._store_expert_episode(traj)

        self.agent._last_obs = obs

    def _store_expert_episode(self, transitions: List[Transition]) -> None:
        """Slice a completed episode into non-overlapping Segments and add to expert_dataset."""
        if self.segment_length is None:
            self.expert_dataset.add(Segment(transitions))
        else:
            start = 0
            while start + self.segment_length <= len(transitions):
                self.expert_dataset.add(Segment(transitions[start : start + self.segment_length]))
                start += self.segment_length

    def _sft_phase(self, n_sft_steps: int, sft_batch_size: int) -> None:
        """Behavioral cloning: minimize MSE between policy's deterministic actions and expert actions."""
        if len(self.expert_dataset) < sft_batch_size:
            print("[Pre-training] Not enough expert segments for SFT, skipping.")
            return

        bc_loss_val = float("nan")
        for _ in range(n_sft_steps):
            batch = self.expert_dataset.sample(sft_batch_size, self.rng)
            obs_np = np.concatenate([seg.obs for seg in batch])
            actions_np = np.concatenate([seg.actions for seg in batch])

            obs_t = obs_as_tensor(obs_np, self.agent.device)
            expert_actions_t = torch.tensor(actions_np, dtype=torch.float32, device=self.agent.device)

            if isinstance(self.agent, OffPolicyAlgorithm):
                mean_actions, _, _ = self.agent.actor.get_action_dist_params(obs_t)
                pred_actions = torch.tanh(mean_actions)
                optimizer = self.agent.actor.optimizer
            else:  # PPO
                features = self.agent.policy.extract_features(obs_t)
                latent_pi, _ = self.agent.policy.mlp_extractor(features)
                pred_actions = self.agent.policy.action_net(latent_pi)
                optimizer = self.agent.policy.optimizer

            bc_loss = F.mse_loss(pred_actions, expert_actions_t)
            optimizer.zero_grad()
            bc_loss.backward()
            optimizer.step()
            bc_loss_val = bc_loss.item()

        print(f"[Pre-training] SFT complete, final bc_loss={bc_loss_val:.4f}")
