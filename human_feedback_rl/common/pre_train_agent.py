import numpy as np
import torch as th
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional

from human_feedback_rl.algorithms import ChristianoAlgorithm
from human_feedback_rl.common import MainLogger

try:
    from sumo_gym_ego import EgoStatus
    _HAS_EGO_STATUS = True
except ImportError:
    EgoStatus = None
    _HAS_EGO_STATUS = False

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


class PreTrainAgent:
    """
    Standalone DAgger pre-trainer for SB3 agents (SAC or PPO).

    Workflow:
      1. DAgger behavioral cloning (actor only).
      2. Evaluation of the pretrained actor.
      3. SAC only: fill replay buffer + train critic with actor frozen.
      4. Save agent zip to disk.

    The saved zip can be loaded and used as the starting agent
    for ChristianoAlgorithm.
    """

    def __init__(
        self,
        env,
        agent,
        rng: np.random.Generator,
        expert_policy,
        segment_length: Optional[int] = 50,
    ):
        self.env = env
        self.agent = agent
        self.rng = rng
        self.segment_length = segment_length
        self.expert_policy = expert_policy

        self.logger = MainLogger()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        save_path: str,
        n_critic_warmup_rollout_steps: int = 10_000,
        n_critic_warmup_train_steps: int = 1_000,
    ) -> None:
        """Run the full pre-training pipeline and save the agent zip.

        Args:
            save_path: destination path for the zip (without .zip extension).
            n_critic_warmup_rollout_steps: SAC only – env steps to fill replay
                buffer after DAgger (uses true env rewards).
            n_critic_warmup_train_steps: SAC only – critic-only gradient steps
                after rollout collection (actor frozen throughout).
        """
        print("[PreTrainAgent] Starting DAgger pre-training...")
        self._dagger_sft_phase()

        print("[PreTrainAgent] Evaluating pretrained agent (100 episodes)...")
        self._eval_pretrained_agent()

        if isinstance(self.agent, OffPolicyAlgorithm):
            print(
                f"[PreTrainAgent] SAC: critic warm-up "
                f"({n_critic_warmup_rollout_steps} rollout steps, "
                f"{n_critic_warmup_train_steps} train steps, actor frozen)..."
            )
            self._warm_start_critic_sac(
                n_critic_warmup_rollout_steps,
                n_critic_warmup_train_steps,
            )

        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(dest)
        print(f"[PreTrainAgent] Saved pretrained agent to {dest}.zip")

    # ------------------------------------------------------------------ #
    # DAgger                                                               #
    # ------------------------------------------------------------------ #

    def _dagger_sft_phase(self) -> None:
        """DAgger: agent rollouts labelled by expert, BC on aggregated dataset.

        Round 0: beta=1.0 (pure expert).
        Round k: beta = 0.7^k (growing agent mix).

        Logs pretrain/dagger_mean_ep_len and pretrain/dagger_bc_loss per round.
        """
        n_rounds = 10
        n_episodes_per_env = 25
        bc_epochs_per_round = 10
        bc_batch_size = 256
        beta_decay = 0.7

        dagger_obs: List[np.ndarray] = []
        dagger_actions: List[np.ndarray] = []

        obs = self.env.reset()
        dones = np.zeros(self.env.num_envs, dtype=bool)

        for round_idx in range(n_rounds):
            beta = beta_decay ** round_idx
            episodes_per_env = [0] * self.env.num_envs
            step_counter = np.zeros(self.env.num_envs, dtype=int)
            round_ep_lens: List[int] = []

            # --- collect: agent acts, expert labels ---
            while min(episodes_per_env) < n_episodes_per_env:
                agent_actions, _ = self.agent.predict(obs, deterministic=False)
                expert_actions = self.expert_policy.predict(obs)

                use_expert = self.rng.random(self.env.num_envs) < beta
                executed = np.where(use_expert[:, np.newaxis], expert_actions, agent_actions)

                next_obs, _, dones, _ = self.env.step(executed)
                step_counter += 1

                for i in range(self.env.num_envs):
                    dagger_obs.append(obs[i].copy())
                    dagger_actions.append(expert_actions[i].copy())
                    if dones[i]:
                        round_ep_lens.append(int(step_counter[i]))
                        step_counter[i] = 0
                        episodes_per_env[i] += 1

                obs = next_obs

            mean_ep_len = float(np.mean(round_ep_lens)) if round_ep_lens else float("nan")

            # --- BC on aggregated dataset ---
            bc_loss_val = float("nan")
            if len(dagger_obs) >= bc_batch_size:
                obs_np = np.stack(dagger_obs)
                act_np = np.stack(dagger_actions)

                n_steps = bc_epochs_per_round * len(dagger_obs) // bc_batch_size
                for _ in range(max(n_steps, 1)):
                    idxs = self.rng.integers(len(dagger_obs), size=bc_batch_size)
                    obs_b = th.tensor(obs_np[idxs], dtype=th.float32, device=self.agent.device)
                    act_b = th.tensor(act_np[idxs], dtype=th.float32, device=self.agent.device)

                    if isinstance(self.agent, OffPolicyAlgorithm):
                        mean_actions, log_std, _ = self.agent.actor.get_action_dist_params(obs_b)
                        self.agent.actor.action_dist.proba_distribution(mean_actions, log_std)
                        log_prob = self.agent.actor.action_dist.log_prob(act_b)
                        optimizer = self.agent.actor.optimizer
                        loss = -log_prob.mean()
                    else:
                        _, log_prob, _ = self.agent.policy.evaluate_actions(obs_b, act_b)
                        optimizer = self.agent.policy.optimizer
                        loss = -log_prob.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    bc_loss_val = loss.item()

                    if isinstance(self.agent, OnPolicyAlgorithm):
                        with th.no_grad():
                            self.agent.policy.log_std.data.clamp_(min=-1.0)

            self.logger.record("pretrain/dagger_mean_ep_len", mean_ep_len)
            self.logger.record("pretrain/dagger_bc_loss", bc_loss_val)
            self.logger.dump()

            print(
                f"[DAgger] round {round_idx + 1}/{n_rounds}  beta={beta:.2f}"
                f"  dataset={len(dagger_obs)}  mean_ep_len={mean_ep_len:.1f}"
                f"  bc_loss={bc_loss_val:.4f}"
            )

        obs = self.env.reset()
        if isinstance(self.agent, OnPolicyAlgorithm):
            self.agent._last_episode_starts = np.array(dones)

    # ------------------------------------------------------------------ #
    # SAC critic warm-up                                                  #
    # ------------------------------------------------------------------ #

    def _warm_start_critic_sac(self, n_rollout_steps: int, n_train_steps: int) -> None:
        """Fill replay buffer with pretrained-actor transitions, then train
        only the critic for n_train_steps (actor weights never updated).

        Uses true environment rewards so the critic gets a meaningful
        Q-value initialization before Christiano RL starts.
        """
        agent = self.agent

        # ---- Phase A: collect transitions --------------------------------
        obs = self.env.reset()
        for step in range(n_rollout_steps):
            with th.no_grad():
                obs_t = obs_as_tensor(obs, agent.device)
                actions_t, _ = agent.actor.action_log_prob(obs_t)
                actions = actions_t.cpu().numpy()

            next_obs, rewards, dones, infos = self.env.step(actions)
            agent.replay_buffer.add(obs, next_obs, actions, rewards, dones, infos)
            obs = next_obs

            if (step + 1) % 2_000 == 0:
                print(f"[Critic warm-up] rollout {step + 1}/{n_rollout_steps}  "
                      f"buffer={agent.replay_buffer.size()}")

        agent._last_obs = obs

        # ---- Phase B: critic-only gradient steps -------------------------
        if agent.replay_buffer.size() < agent.batch_size:
            print("[Critic warm-up] replay buffer too small, skipping critic training.")
            return

        first_loss = last_loss = float("nan")
        for step in range(n_train_steps):
            replay_data = agent.replay_buffer.sample(
                agent.batch_size, env=agent._vec_normalize_env
            )

            with th.no_grad():
                next_actions, next_log_prob = agent.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q = th.cat(
                    agent.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q, _ = th.min(next_q, dim=1, keepdim=True)

                if agent.ent_coef_optimizer is not None and agent.log_ent_coef is not None:
                    ent_coef = th.exp(agent.log_ent_coef.detach())
                else:
                    ent_coef = agent.ent_coef_tensor

                target_q = (
                    replay_data.rewards
                    + (1.0 - replay_data.dones.float()) * agent.gamma
                    * (next_q - ent_coef * next_log_prob.reshape(-1, 1))
                )

            current_qs = agent.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(q, target_q) for q in current_qs)

            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            loss_val = critic_loss.item()
            if step == 0:
                first_loss = loss_val
            last_loss = loss_val

        print(
            f"[Critic warm-up] Done.  critic_loss: {first_loss:.4f} → {last_loss:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def _eval_pretrained_agent(self, n_eval_episodes: int = 100) -> None:
        """Evaluate agent deterministically; logs to pretrain_eval/ on wandb."""
        n_envs = self.env.num_envs
        episodes_done = 0
        n_collisions = n_off_road = n_timeouts = n_successes = 0
        ep_true_rewards: List[float] = []
        ep_lengths: List[int] = []

        obs = self.env.reset()
        ep_reward = np.zeros(n_envs)
        ep_length = np.zeros(n_envs, dtype=int)

        while episodes_done < n_eval_episodes:
            actions, _ = self.agent.predict(obs, deterministic=True)
            obs, true_rewards, dones, infos = self.env.step(actions)
            ep_reward += true_rewards
            ep_length += 1

            for i in range(n_envs):
                if dones[i]:
                    episodes_done += 1
                    ep_true_rewards.append(float(ep_reward[i]))
                    ep_lengths.append(int(ep_length[i]))
                    ep_reward[i] = 0.0
                    ep_length[i] = 0
                    if _HAS_EGO_STATUS:
                        ego_status = infos[i].get("ego_status")
                        if ego_status is not None:
                            n_collisions += int(ego_status == EgoStatus.COLLIDED.value)
                            n_off_road   += int(ego_status == EgoStatus.OFF_ROAD.value)
                            n_timeouts   += int(ego_status == EgoStatus.TIMEOUT.value)
                            n_successes  += int(ego_status == EgoStatus.ARRIVED.value)

        n = len(ep_true_rewards)
        metrics: Dict[str, float] = {
            "pretrain_eval/mean_true_reward": float(np.mean(ep_true_rewards)),
            "pretrain_eval/mean_ep_length":   float(np.mean(ep_lengths)),
        }
        if n > 0 and _HAS_EGO_STATUS:
            metrics["pretrain_eval/success_rate"]   = n_successes / n
            metrics["pretrain_eval/collision_rate"] = n_collisions / n
            metrics["pretrain_eval/off_road_rate"]  = n_off_road / n
            metrics["pretrain_eval/timeout_rate"]   = n_timeouts / n

        for k, v in metrics.items():
            self.logger.record(k, v)
        self.logger.dump()

        print(
            f"[Pretrain eval] {n} episodes | "
            f"success={metrics.get('pretrain_eval/success_rate', float('nan')):.2f} | "
            f"mean_reward={metrics['pretrain_eval/mean_true_reward']:.2f} | "
            f"mean_ep_len={metrics['pretrain_eval/mean_ep_length']:.1f}"
        )

