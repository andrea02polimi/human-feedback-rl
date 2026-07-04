from typing import Optional

import numpy as np
import torch

from human_feedback_rl.common import Transition, Trajectory
from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.loggers import (
    WandbWriter,
    make_human_output_format,
)

try:
    from sumo_rl_ego.policies.base_policy import Policy as RuleBasedPolicy
    from sumo_rl_ego.policies.model_policy import ModelPolicy as _ModelPolicy
except ImportError:
    RuleBasedPolicy = None
    _ModelPolicy    = None

try:
    from sumo_gym_ego import EgoStatus
except ImportError:
    EgoStatus = None


class DaggerAlgorithm(BaseAlgorithm):
    """
    Dataset Aggregation (DAgger) — iterative imitation learning.

    At each round:
      1. Collect episodes mixing expert (probability β) and agent actions.
      2. Aggregate all expert-labeled transitions into a growing dataset.
      3. Run behaviour cloning (negative log-likelihood) on the full dataset.
      4. Evaluate the agent deterministically.

    β decays exponentially so the agent gradually takes over from the expert.
    """

    def __init__(
        self,
        env,
        agent,
        expert,
        bc_epochs: int = 5,
        bc_batch_size: int = 64,
        bc_lr: float = 1e-3,
        n_eval_episodes: int = 5,
        n_expert_rollout_episodes: int = 5,
        beta_decay: float = 0.7,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[list] = None,
    ):
        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.expert          = expert
        self.dataset         = []
        self.dataset_expert  = []
        self.bc_epochs       = bc_epochs
        self.bc_batch_size   = bc_batch_size
        self.n_eval_episodes = n_eval_episodes
        self.n_expert_rollout_episodes = n_expert_rollout_episodes
        self.beta_decay      = beta_decay
        self.discrete_actions = hasattr(env.action_space, "n")

        # True only for hand-coded rule-based experts (e.g. FastPolicy).
        # ModelPolicy wraps SB3 models and must use the vectorised branch.
        self._expert_is_rule_based = (
            RuleBasedPolicy is not None
            and isinstance(expert, RuleBasedPolicy)
            and (_ModelPolicy is None or not isinstance(expert, _ModelPolicy))
        )

        self._optimizer = torch.optim.Adam(agent.parameters(), lr=bc_lr)

    def _output_formats(self) -> list:
        return [make_human_output_format(), WandbWriter()]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_rounds: int, num_episodes: int):
        cumulative_bc_epochs    = 0
        cumulative_eval_episodes = 0

        # Fixed held-out expert dataset (pure-expert rollouts, beta=1) collected
        # once so the imitation error is measured against a stable reference
        # instead of the growing, mixture-visited aggregated dataset.
        print(f"[setup] Collecting expert rollouts ({self.n_expert_rollout_episodes} episodes)...")
        expert_trajectories, _ = self._collect_trajectories(self.n_expert_rollout_episodes, beta=1.0)
        for traj in expert_trajectories:
            self.dataset_expert.extend(traj)
        print(f"[setup] Expert dataset collected ({len(self.dataset_expert)} transitions).")

        for round_idx in range(n_rounds):
            beta = self._beta_schedule(round_idx)
            print(f"\n=== Round {round_idx + 1}/{n_rounds} (beta={beta:.3f}) ===")

            # 1) Collect trajectories mixing expert and agent
            print(f"[1/3] Collecting trajectories ({num_episodes} episodes)...")
            trajectories, collect_stats = self._collect_trajectories(num_episodes, beta)
            print(
                f"      round_reward={collect_stats['round_reward']:.3f}  "
                f"disagreement={collect_stats['disagreement_rate']:.2%}  "
                f"expert_usage={collect_stats['expert_usage']:.3f}"
            )

            # 2) Aggregate dataset (Trajectory is a list subclass of Transition)
            for traj in trajectories:
                self.dataset.extend(traj)
            print(f"[2/3] Dataset aggregated (total transitions: {len(self.dataset)})...")

            # 3) Behaviour cloning on aggregated dataset
            print(f"[3/3] BC training ({self.bc_epochs} epochs, dataset size: {len(self.dataset)})...")
            bc_stats = self._bc_train()
            print(
                f"      loss={bc_stats['loss']:.4f}  "
                f"entropy={bc_stats['entropy']:.4f}  "
                f"grad_norm={bc_stats['grad_norm']:.4f}"
            )

            # 4) Evaluate agent
            eval_stats = self._evaluate(self.n_eval_episodes)
            event_rates = eval_stats["event_rates"]
            print(
                f"      eval — mean_reward={eval_stats['mean_ep_reward']:.3f}  "
                f"mean_length={eval_stats['mean_ep_length']:.1f}"
            )
            print(
                f"      events — success={event_rates['successes']:.2%}  "
                f"collision={event_rates['collisions']:.2%}  "
                f"off_road={event_rates['off_road']:.2%}  "
                f"timeout={event_rates['timeouts']:.2%}"
            )

            # 5) Imitation errors against the fixed expert dataset
            imitation_stats = self._log_imitation_errors()
            print(
                f"      imitation — nll={imitation_stats['expert_action_nll']:.4f}  "
                f"{imitation_stats['error_name']}={imitation_stats['action_error']:.4f}"
            )

            cumulative_bc_epochs     += self.bc_epochs
            cumulative_eval_episodes += self.n_eval_episodes

            self.logger.record("dagger/beta",              beta)
            self.logger.record("dagger/dataset_size",      len(self.dataset))
            self.logger.record("dagger/expert_usage",      collect_stats["expert_usage"])
            self.logger.record("dagger/disagreement_rate", collect_stats["disagreement_rate"])
            self.logger.record("dagger/round_reward",      collect_stats["round_reward"])

            self.logger.record("bc/loss",          bc_stats["loss"])
            self.logger.record("bc/log_prob_mean", bc_stats["log_prob_mean"])
            self.logger.record("bc/entropy",       bc_stats["entropy"])

            self.logger.record("train/grad_norm", bc_stats["grad_norm"])
            self.logger.record("train/lr",        bc_stats["lr"])

            self.logger.record("eval/mean_ep_reward", eval_stats["mean_ep_reward"])
            self.logger.record("eval/mean_ep_length", eval_stats["mean_ep_length"])

            self.logger.record("eval/event_rate/successes",  event_rates["successes"])
            self.logger.record("eval/event_rate/collisions", event_rates["collisions"])
            self.logger.record("eval/event_rate/off_road",   event_rates["off_road"])
            self.logger.record("eval/event_rate/timeouts",   event_rates["timeouts"])

            self.logger.record("iterations",      round_idx + 1)
            self.logger.record("num_rounds",      round_idx + 1)
            self.logger.record("bc_epochs",       cumulative_bc_epochs)
            self.logger.record("n_eval_episodes", cumulative_eval_episodes)
            self.logger.dump()

        return self.agent

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _beta_schedule(self, round_idx: int) -> float:
        """Exponential decay: β=1 at round 0 (pure expert), approaches 0."""
        return self.beta_decay ** round_idx

    def _collect_trajectories(self, num_episodes: int, beta: float):
        trajectories       = []
        total_reward       = 0.0
        total_steps        = 0
        total_disagreements = 0

        for _ in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs

            episode_data = []
            done = np.zeros(self.env.num_envs, dtype=bool)

            while not done[0]:
                agent_action, _ = self.agent.predict(obs, deterministic=False)

                if self._expert_is_rule_based:
                    expert_action_scalar = self.expert.predict(obs[0])
                    expert_action_vec    = np.array([expert_action_scalar] * self.env.num_envs)
                    if self.discrete_actions:
                        disagrees = int(agent_action[0]) != int(expert_action_scalar)
                    else:
                        disagrees = not np.allclose(agent_action[0], expert_action_scalar, atol=0.1)
                    expert_action_for_transition = (
                        int(expert_action_scalar) if self.discrete_actions
                        else np.asarray(expert_action_scalar, dtype=np.float32)
                    )
                else:
                    expert_action_vec = self.expert.predict(obs)
                    if self.discrete_actions:
                        disagrees = int(agent_action[0]) != int(expert_action_vec[0])
                    else:
                        disagrees = not np.allclose(agent_action[0], expert_action_vec[0], atol=0.1)
                    expert_action_for_transition = (
                        int(expert_action_vec[0]) if self.discrete_actions
                        else expert_action_vec[0].copy()
                    )

                total_disagreements += int(disagrees)
                total_steps         += 1

                executed_action = expert_action_vec if self.rng.random() < beta else agent_action

                step_result = self.env.step(executed_action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated | truncated
                else:
                    next_obs, reward, done, _ = step_result

                total_reward += float(reward[0])
                episode_data.append(
                    Transition(
                        observation=obs[0].copy(),
                        action=expert_action_for_transition,
                        true_reward=float(reward[0]),
                    )
                )
                obs = next_obs

            trajectories.append(Trajectory(episode_data))

        collect_stats = {
            "round_reward":      total_reward / num_episodes,
            "disagreement_rate": total_disagreements / max(total_steps, 1),
            "expert_usage":      beta,
        }
        return trajectories, collect_stats

    def _bc_train(self) -> dict:
        """Behaviour cloning: minimize negative log-likelihood on expert actions."""
        empty_stats = {
            "loss": 0.0, "log_prob_mean": 0.0, "entropy": 0.0,
            "grad_norm": 0.0, "lr": self._get_lr(),
        }
        if not self.dataset:
            return empty_stats

        obs_t = torch.as_tensor(np.stack([t.observation for t in self.dataset]).astype(np.float32))
        if self.discrete_actions:
            act_t = torch.as_tensor(np.array([t.action for t in self.dataset], dtype=np.int64))
        else:
            act_t = torch.as_tensor(np.stack([t.action for t in self.dataset]).astype(np.float32))

        total_loss = total_log_prob = total_entropy = total_grad_norm = 0.0
        n_steps = 0

        for _ in range(self.bc_epochs):
            perm = torch.randperm(len(obs_t))
            for start in range(0, len(perm), self.bc_batch_size):
                batch_idx = perm[start : start + self.bc_batch_size]
                obs_b, act_b = obs_t[batch_idx], act_t[batch_idx]

                _, log_prob, entropy = self.agent.evaluate_actions(obs_b, act_b)
                loss = -log_prob.mean()

                self._optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), max_norm=float("inf")
                )
                self._optimizer.step()

                total_loss     += loss.item()
                total_log_prob += log_prob.mean().item()
                total_entropy  += entropy.mean().item() if entropy is not None else 0.0
                total_grad_norm += grad_norm.item()
                n_steps += 1

        n = max(n_steps, 1)
        return {
            "loss":          total_loss / n,
            "log_prob_mean": total_log_prob / n,
            "entropy":       total_entropy / n,
            "grad_norm":     total_grad_norm / n,
            "lr":            self._get_lr(),
        }

    def _log_imitation_errors(self) -> dict:
        """Agent-vs-expert imitation errors over the fixed expert dataset.

        Records two metrics on the ``imitation`` logger (and thus W&B):

        * ``imitation/expert_action_nll`` — mean negative log-likelihood of the
          expert actions under the agent policy (KL surrogate: it equals the
          cross-entropy H(expert, agent), differing from the true KL only by the
          expert's entropy, constant w.r.t. the agent).
        * ``imitation/action_rmse`` (continuous) or ``imitation/action_accuracy``
          (discrete) — deterministic agent action vs the expert action.
        """
        empty_stats = {
            "expert_action_nll": 0.0,
            "action_error": 0.0,
            "error_name": "action_accuracy" if self.discrete_actions else "action_rmse",
        }
        if not self.dataset_expert:
            return empty_stats

        obs_np = np.stack([t.observation for t in self.dataset_expert]).astype(np.float32)
        obs_t = torch.as_tensor(obs_np)
        if self.discrete_actions:
            act_t = torch.as_tensor(np.array([t.action for t in self.dataset_expert], dtype=np.int64))
        else:
            act_t = torch.as_tensor(np.stack([t.action for t in self.dataset_expert]).astype(np.float32))

        # NLL surrogate via the same evaluate_actions path used for BC training.
        with torch.no_grad():
            _, log_prob, _ = self.agent.evaluate_actions(obs_t, act_t) # questo metodo è ereditato da SB3 self.agent è impostato in base_algorithm.py e in test_dagger.py l'agente è un BCPolicy che estende ActorCriticPolicy di sb3 e non fa override di evaluate_actions quindi viene chiamato quello di ActorCriticPolicy 
        finite = torch.isfinite(log_prob)
        nll = float(-log_prob[finite].mean().item()) if finite.any() else 0.0

        # Deterministic action error: accuracy for discrete, RMSE for continuous.
        agent_actions, _ = self.agent.predict(obs_np, deterministic=True)
        if self.discrete_actions:
            expert_actions = act_t.numpy()
            action_error = float(np.mean(agent_actions.reshape(-1) == expert_actions.reshape(-1)))
            error_name = "action_accuracy"
        else:
            expert_actions = act_t.numpy().reshape(len(obs_np), -1)
            agent_actions = np.asarray(agent_actions, dtype=np.float64).reshape(len(obs_np), -1)
            action_error = float(np.sqrt(np.mean((agent_actions - expert_actions) ** 2)))
            error_name = "action_rmse"

        self.logger.record("imitation/expert_action_nll", nll)
        self.logger.record(f"imitation/{error_name}", action_error)

        return {"expert_action_nll": nll, "action_error": action_error, "error_name": error_name}

    def _evaluate(self, n_eval_episodes: int) -> dict:
        """Run the agent deterministically for n_eval_episodes and return stats.

        Alongside reward/length, tallies the four mutually exclusive terminal
        outcomes (arrived, collided, off_road, timeout) into per-episode rates.
        """
        total_reward = 0.0
        total_length = 0
        events = {"successes": 0, "collisions": 0, "off_road": 0, "timeouts": 0}

        for _ in range(n_eval_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            done      = np.zeros(self.env.num_envs, dtype=bool)
            ep_reward = 0.0
            ep_length = 0
            info      = {}

            while not done[0]:
                action, _ = self.agent.predict(obs, deterministic=True)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, infos = step_result
                    done = terminated | truncated
                else:
                    obs, reward, done, infos = step_result
                info = infos[0] if len(infos) else {}
                ep_reward += float(reward[0])
                ep_length += 1

            total_reward += ep_reward
            total_length += ep_length

            if EgoStatus is not None:
                ego_status = info.get("ego_status", EgoStatus.RUNNING)
                events["successes"]  += int(ego_status == EgoStatus.ARRIVED.value)
                events["collisions"] += int(ego_status == EgoStatus.COLLIDED.value)
                events["off_road"]   += int(ego_status == EgoStatus.OFF_ROAD.value)
                events["timeouts"]   += int(ego_status == EgoStatus.TIMEOUT.value)

        return {
            "mean_ep_reward": total_reward / n_eval_episodes,
            "mean_ep_length": total_length / n_eval_episodes,
            "event_rates": {k: v / n_eval_episodes for k, v in events.items()},
        }

    def _get_lr(self) -> float:
        return self._optimizer.param_groups[0]["lr"]
