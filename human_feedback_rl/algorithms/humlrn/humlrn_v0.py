"""
HumLrn Algorithm
================
Pipeline ibrida preferenze + dimostrazioni per RLHF senza reward ambientale.

Fasi per iterazione:
  1. Collect rollouts con agente corrente
  2. Fragment → segment pairs
  3. Query oracolo esperto per preferenze tra coppie
  4. Collect dimostrazioni dall'esperto
  5. Train reward model: loss_pref + lambda_demo * loss_demo
  6. Train agente su env con reward model

Sia le preferenze che le dimostrazioni vengono fornite da un modello esperto
pre-trainato con SB3 (DQN, PPO, ecc.). Il reward ambientale non è mai usato.
"""

from typing import Any, List

import numpy as np
import torch
import wandb

from human_feedback_rl.common import (
    ActiveFragmenter,
    BaseAlgorithm,
    EnsembleRewardModel,
    EnvRewardWrapper,
    InverseSchedule,
    Preference,
    PreferenceDataset,
    PreferenceModelFromReward,
    PrefixLogger,
    SegmentPair,
    Trajectory,
    Transition,
    UnifiedLogger,
)
from .reward_trainer_humlrn import RewardTrainerHumLrn
from .demo_dataset import DemonstrationDataset


class HumLrnAlgorithm(BaseAlgorithm):
    """
    RLHF con feedback misto: preferenze + dimostrazioni.

    Il reward model viene addestrato combinando:
      - Cross-entropy su preferenze tra coppie di segmenti (Bradley-Terry)
      - Margin ranking loss su dimostrazioni esperto vs segmenti agente

    L'agente viene poi addestrato su un env wrappato con il reward model.
    """

    def __init__(
        self,
        env,
        agent,
        expert,
        n_ensembles: int,
        segment_length: int,
        device: str = "cpu",
        lr_reward_model: float = 1e-4,
        max_dataset_size: int = 10_000,
        reward_model_batch_size: int = 32,
        reward_training_epochs: int = 10,
        # Preference schedule
        num_pairs_initial: int = 100,
        num_pairs_final: int = 0,
        decay_pairs_schedule: float = 1.0,
        # Demonstration parameters
        use_demonstrations: bool = True,
        num_demo_episodes_per_iter: int = 10,
        lambda_demo: float = 1.0,
        demo_margin: float = 1.0,
    ):
        self.env = env
        self.agent = agent
        self.expert = expert
        self.reward_training_epochs = reward_training_epochs
        self.use_demonstrations = use_demonstrations
        self.num_demo_episodes_per_iter = num_demo_episodes_per_iter

        self.logger = UnifiedLogger()

        discrete = hasattr(env.action_space, "n")
        self.discrete_actions = discrete
        self.reward_model = EnsembleRewardModel(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n if discrete else env.action_space.shape[0],
            n_ensembles=n_ensembles,
            lr=lr_reward_model,
            device=device,
            discrete_actions=discrete,
        )

        self.preference_model = PreferenceModelFromReward(self.reward_model)

        self.reward_trainer = RewardTrainerHumLrn(
            preference_model=self.preference_model,
            logger=PrefixLogger(self.logger),
            batch_size=reward_model_batch_size,
            num_epochs=reward_training_epochs,
            lambda_demo=lambda_demo,
            demo_margin=demo_margin,
        )

        self.fragmenter = ActiveFragmenter(
            reward_model=self.reward_model,
            segment_length=segment_length,
        )

        self.preference_dataset = PreferenceDataset(max_dataset_size)
        self.preference_dataset_val = PreferenceDataset(max_dataset_size)
        self.demo_dataset = DemonstrationDataset(max_dataset_size)

        env_reward_wrapper = EnvRewardWrapper(self.env, self.reward_model)
        self.agent.set_env(env_reward_wrapper)
        self.agent.set_logger(SB3BridgeLogger(self.logger, key_map={"train/loss": "agent/train_loss"}))

        if wandb.run is not None:
            wandb.define_metric("reward_model/*", step_metric="timescales/global_reward_trainer_epochs")
            wandb.define_metric("agent/*",        step_metric="timescales/global_agent_steps")
            wandb.define_metric("rollout/*",       step_metric="timescales/iterations")

        self.schedule_num_pairs = InverseSchedule(
            initial_value=num_pairs_initial,
            final_value=num_pairs_final,
            decay_rate=decay_pairs_schedule,
        )

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self, total_timesteps: int = 1_000_000, num_iterations: int = 10) -> Any:
        per_iter_timesteps = int(total_timesteps / num_iterations)
        global_agent_steps = 0
        global_reward_trainer_epochs = 0

        for it in range(num_iterations):
            print(f"\n=== Iteration {it+1}/{num_iterations} ===")
            progress_remaining = 1 - it / num_iterations

            # 1) Collect rollouts with current agent
            print("[1/6] Collecting rollouts...")
            num_pairs = int(self.schedule_num_pairs(progress_remaining))
            tot_rollout_timesteps = num_pairs * self.fragmenter.segment_length * 2
            trajectories = self._collect_rollout(tot_rollout_timesteps)
            
            # DEBUG 
            avg_ep_length = float(np.mean([len(t.transitions) for t in trajectories]))
            avg_true_reward = float(np.mean([t.total_reward() for t in trajectories]))
            print(f"      trajectories={len(trajectories)}  avg_ep_length={avg_ep_length:.1f}  avg_true_reward={avg_true_reward:.3f}")
            # FINE 

            # 2) Fragment trajectories into segment pairs
            print("[2/6] Fragmenting trajectories...")
            segment_pairs = self.fragmenter.fragment(trajectories=trajectories, num_pairs=num_pairs)
            # DEBUG
            print(f"      segment_pairs={len(segment_pairs)} (requested {num_pairs})")
            # FINE
            
            # 3) Query expert for preferences
            print("[3/6] Querying expert for preferences...")
            preferences = self._query_preferences(segment_pairs)

            train_pairs, train_prefs, val_pairs, val_prefs = self._train_val_split(
                segment_pairs, preferences
            )
            self.preference_dataset.push(train_pairs, train_prefs)
            self.preference_dataset_val.push(val_pairs, val_prefs)
            # DEBUG
            print(f"      dataset — train={len(self.preference_dataset)}  val={len(self.preference_dataset_val)}")
            # FINE

            # 4) Collect expert demonstrations (optional)
            if self.use_demonstrations:
                print("[4/6] Collecting expert demonstrations...")
                demo_trajectories = self._collect_expert_demonstrations(self.num_demo_episodes_per_iter)
                self.demo_dataset.push(demo_trajectories)
                # DEBUG
                print(f"      demo_dataset={len(self.demo_dataset)}")
                # FINE
            else:
                print("[4/6] Skipping demonstrations (use_demonstrations=False).")
                demo_trajectories = []

            # 5) Train reward model (preference loss + optional demonstration loss)
            print("[5/6] Training reward model...")
            loss = self.reward_trainer.train(
                self.preference_dataset,
                self.demo_dataset if self.use_demonstrations else None,
            )
            val_loss = self.reward_trainer.evaluate(self.preference_dataset_val)
            # DEBUG
            acc_train = self._compute_preference_accuracy(train_pairs, train_prefs)
            acc_val   = self._compute_preference_accuracy(val_pairs, val_prefs)
            acc_train_global = self._compute_preference_accuracy(
                self.preference_dataset.pairs, self.preference_dataset.preferences)
            print(f"      loss={loss:.4f}  val_loss={val_loss:.4f}")
            print(f"      accuracy — train_current={acc_train:.2f}  val_current={acc_val:.2f}  train_global={acc_train_global:.2f}")
            # FINE

            global_reward_trainer_epochs += self.reward_training_epochs

            # 6) Train agent
            print("[6/6] Training agent...")
            self.agent.learn(total_timesteps=per_iter_timesteps, reset_num_timesteps=False)
            global_agent_steps += per_iter_timesteps
            avg_model_reward = self._compute_avg_model_reward(trajectories)
            # DEBUG
            print(f"      avg_model_reward={avg_model_reward:.3f}  avg_true_reward={avg_true_reward:.3f}")
            # FINE

            self.logger.record("rollout/num_pairs",     num_pairs)
            self.logger.record("rollout/avg_ep_length", avg_ep_length)
            self.logger.record("rollout/num_demos",     len(demo_trajectories))

            self.logger.record("reward_model/training_loss",   loss)
            self.logger.record("reward_model/validation_loss", val_loss)
            self.logger.record("reward_model/accuracy_train",
                               self._compute_preference_accuracy(train_pairs, train_prefs))
            self.logger.record("reward_model/accuracy_val",
                               self._compute_preference_accuracy(val_pairs, val_prefs))

            self.logger.record("agent/avg_ep_length",  avg_ep_length)
            self.logger.record("agent/avg_ep_reward",  avg_model_reward)

            self.logger.record("timescales/iterations",                   it)
            self.logger.record("timescales/global_reward_trainer_epochs", global_reward_trainer_epochs)
            self.logger.record("timescales/global_agent_steps",           global_agent_steps)

            self.logger.dump()

        return self.agent

    # -----------------------------------------------------------------------
    # Expert interaction
    # -----------------------------------------------------------------------

    def _query_preferences(self, segment_pairs: List[SegmentPair]) -> List[Preference]:
        """
        Query the expert oracle for preferences between segment pairs.

        The expert scores each segment using its internal value function:
          - DQN:     sum of Q(obs_t, action_t) over the segment
          - PPO/A2C: sum of V(obs_t) over the segment

        The segment with higher cumulative score is preferred.
        Scores are normalized by segment length to handle variable-length segments.
        """
        return [
            Preference((1, 0) if self._expert_segment_score(pair.seg1) >= self._expert_segment_score(pair.seg2)
                       else (0, 1))
            for pair in segment_pairs
        ]

    def _expert_segment_score(self, segment: Trajectory) -> float:
        """
        Score a segment using the expert's value estimates.

        - DQN:     policy.q_net(obs) → Q(obs, action_taken) per step
        - PPO/A2C: policy.evaluate_actions(obs, actions) → V(obs) per step

        Returns mean score per step to normalise across segment lengths.
        """
        policy = (
            getattr(self.expert, "policy", None)
            or getattr(getattr(self.expert, "model", None), "policy", None)
            or self.expert
        )
        device = getattr(policy, "device", torch.device("cpu"))

        obs = np.stack([t.obs for t in segment.transitions])
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            if hasattr(policy, "q_net"):
                # DQN: Q(obs, action_taken)
                q_values = policy.q_net(obs_t)  # (T, n_actions)
                actions_t = torch.tensor(
                    [t.action for t in segment.transitions],
                    dtype=torch.long, device=device,
                )
                scores = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (T,)
            else:
                # PPO / A2C: log π_expert(a|s) — measures how much the expert
                # approves of the actions taken, depends on both obs and actions
                actions_t = torch.as_tensor(
                    np.array([t.action for t in segment.transitions], dtype=np.float32),
                    device=device,
                )
                _, log_probs, _ = policy.evaluate_actions(obs_t, actions_t)
                scores = log_probs  # (T,)

        return float(scores.mean().item())

    def _collect_expert_demonstrations(self, n_episodes: int) -> List[Trajectory]:
        """
        Collect demonstration trajectories by running the expert policy.

        TODO: optionally filter demonstrations by quality (e.g. top-k by episode length).
        """
        # TODO: implement rollout collection using self.expert
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Rollout collection (same as ChristianoAlgorithm)
    # -----------------------------------------------------------------------

    def _collect_rollout(self, total_timesteps_target: int) -> List[Trajectory]:
        num_envs = self.env.num_envs
        current_transitions = [[] for _ in range(num_envs)]
        trajectories: List[Trajectory] = []
        total_timesteps = 0

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        while total_timesteps < total_timesteps_target:
            action, _ = self.agent.predict(obs, deterministic=False)
            step_result = self.env.step(action)

            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated | truncated
            else:
                next_obs, reward, done, _ = step_result

            for i in range(num_envs):
                current_transitions[i].append(
                    Transition(
                        obs=obs[i].copy(),
                        action=int(action[i]) if self.discrete_actions else action[i].copy(),
                        reward=float(reward[i]),
                    )
                )
                if done[i]:
                    trajectories.append(Trajectory(current_transitions[i]))
                    current_transitions[i] = []

            obs = next_obs
            total_timesteps += num_envs

        for i in range(num_envs):
            if current_transitions[i]:
                trajectories.append(Trajectory(current_transitions[i]))

        return trajectories

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _compute_preference_accuracy(
        self, segment_pairs: List[SegmentPair], preferences: List[Preference]
    ) -> float:
        if not preferences:
            return 0.0
        correct = sum(
            1 for pair, pref in zip(segment_pairs, preferences)
            if (self.preference_model.preference_probs(pair.seg1, pair.seg2).label[0] > 0.5) ==
               (pref.label[0] > 0.5)
        )
        return correct / len(preferences)

    def _compute_avg_model_reward(self, trajectories: List[Trajectory]) -> float:
        ep_model_rewards = [
            float(self.reward_model.predict(
                np.array([t.obs for t in traj.transitions]),
                np.array([t.action for t in traj.transitions]),
            ).sum())
            for traj in trajectories
        ]
        return float(np.mean(ep_model_rewards))

    def _train_val_split(
        self,
        pairs: List[SegmentPair],
        preferences: List[Preference],
        split_ratio: float = 0.7,
    ):
        n = len(pairs)
        indices = np.random.permutation(n)
        split_idx = int(n * split_ratio)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        return (
            [pairs[i] for i in train_idx],
            [preferences[i] for i in train_idx],
            [pairs[i] for i in val_idx],
            [preferences[i] for i in val_idx],
        )