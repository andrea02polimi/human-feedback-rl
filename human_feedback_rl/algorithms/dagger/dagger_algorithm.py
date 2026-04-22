import numpy as np
import torch

from human_feedback_rl.common import BaseAlgorithm, Transition, Trajectory
from human_feedback_rl.common.loggers import MainLogger, PrefixWrapper

try:
    from sumo_rl_ego.policies.base_policy import Policy as RuleBasedPolicy
    from sumo_rl_ego.policies.model_policy import ModelPolicy as _ModelPolicy
except ImportError:
    RuleBasedPolicy = None
    _ModelPolicy = None


class DaggerAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        env,
        agent,
        expert,
        bc_epochs: int = 5,
        bc_batch_size: int = 64,
        bc_lr: float = 1e-3,
        n_eval_episodes: int = 5,
        beta_decay: float = 0.7,
    ):
        self.env = env
        self.agent = agent
        self.expert = expert
        self.dataset = []
        self.bc_epochs = bc_epochs
        self.bc_batch_size = bc_batch_size
        self.n_eval_episodes = n_eval_episodes
        self.beta_decay = beta_decay
        self.discrete_actions = hasattr(env.action_space, "n") # per capire se azioni discrete
        # True only for hand-coded rule-based experts (e.g. FastPolicy).
        # ModelPolicy wraps SB3 models and must use the vectorised branch.
        self._expert_is_rule_based = (
            RuleBasedPolicy is not None
            and isinstance(expert, RuleBasedPolicy)
            and (_ModelPolicy is None or not isinstance(expert, _ModelPolicy))
        )
        self._optimizer = torch.optim.Adam(agent.parameters(), lr=bc_lr)

        self._logger = MainLogger()
        self._bc_log = PrefixWrapper(self._logger, prefix="bc")
        self._dagger_log = PrefixWrapper(self._logger, prefix="dagger")
        self._train_log = PrefixWrapper(self._logger, prefix="train")
        self._eval_log = PrefixWrapper(self._logger, prefix="eval")

    def _beta_schedule(self, round_idx: int) -> float:
        """Exponential decay: beta=1.0 at round 0 (pure expert), approaches 0."""
        return self.beta_decay ** round_idx

    def train(self, n_rounds: int, num_episodes: int):
        cumulative_bc_epochs = 0
        cumulative_eval_episodes = 0

        for round_idx in range(n_rounds):
            beta = self._beta_schedule(round_idx)
            print(f"\n=== Round {round_idx + 1}/{n_rounds} (beta={beta:.3f}) ===")

            # 1) Collect trajectories mixing expert and agent
            print(f"[1/3] Collecting trajectories ({num_episodes} episodes)...")
            trajectories, collect_stats = self._collect_trajectories(num_episodes, beta)
            print(f"      round_reward={collect_stats['round_reward']:.3f}  disagreement={collect_stats['disagreement_rate']:.2%}  expert_usage={collect_stats['expert_usage']:.3f}")

            # 2) Aggregate dataset
            for traj in trajectories:
                self.dataset.extend(traj.transitions)
            print(f"[2/3] Dataset aggregated (total transitions: {len(self.dataset)})...")

            # 3) Behavior cloning on aggregated dataset
            print(f"[3/3] BC training ({self.bc_epochs} epochs, dataset size: {len(self.dataset)})...")
            bc_stats = self._bc_train()
            print(f"      loss={bc_stats['loss']:.4f}  entropy={bc_stats['entropy']:.4f}  grad_norm={bc_stats['grad_norm']:.4f}")

            # 4) Evaluate agent
            eval_stats = self._evaluate(self.n_eval_episodes)
            print(f"      eval — mean_reward={eval_stats['mean_ep_reward']:.3f}  mean_length={eval_stats['mean_ep_length']:.1f}")

            cumulative_bc_epochs += self.bc_epochs
            cumulative_eval_episodes += self.n_eval_episodes

            self._dagger_log.record("beta",              beta)
            self._dagger_log.record("dataset_size",      len(self.dataset))
            self._dagger_log.record("expert_usage",      collect_stats["expert_usage"])
            self._dagger_log.record("disagreement_rate", collect_stats["disagreement_rate"])
            self._dagger_log.record("round_reward",      collect_stats["round_reward"])

            self._bc_log.record("loss",         bc_stats["loss"])
            self._bc_log.record("log_prob_mean", bc_stats["log_prob_mean"])
            self._bc_log.record("entropy",       bc_stats["entropy"])

            self._train_log.record("grad_norm", bc_stats["grad_norm"])
            self._train_log.record("lr",         bc_stats["lr"])

            self._eval_log.record("mean_ep_reward",  eval_stats["mean_ep_reward"])
            self._eval_log.record("mean_ep_length",  eval_stats["mean_ep_length"])

            self._logger.record("num_rounds",       round_idx + 1)
            self._logger.record("bc_epochs",        cumulative_bc_epochs)
            self._logger.record("n_eval_episodes",  cumulative_eval_episodes)

            self._logger.dump()

        return self.agent

    def _collect_trajectories(self, num_episodes: int, beta: float):
        trajectories = []
        total_reward = 0.0
        total_steps = 0
        total_disagreements = 0

        for _ in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):  # gymnasium-style (obs, infos)
                obs, _ = obs

            episode_data = []
            done = np.zeros(self.env.num_envs, dtype=bool)

            while not done[0]:
                agent_action, _ = self.agent.predict(obs, deterministic=False)

                if self._expert_is_rule_based:
                    # Rule-based experts expect a single flat obs and return a scalar action
                    expert_action_scalar = self.expert.predict(obs[0])
                    expert_action_vec = np.array([expert_action_scalar] * self.env.num_envs)
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
                total_steps += 1

                executed_action = expert_action_vec if np.random.random() < beta else agent_action

                step_result = self.env.step(executed_action)
                if len(step_result) == 5:  # gymnasium VecEnv
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated | truncated
                else:  # SB3 VecEnv
                    next_obs, reward, done, _ = step_result

                total_reward += float(reward[0])
                episode_data.append(
                    Transition(obs=obs[0].copy(),
                               action=expert_action_for_transition,
                               reward=float(reward[0]))
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
        """Behavior cloning: minimize negative log-likelihood on expert actions."""
        empty_stats = {
            "loss": 0.0, "log_prob_mean": 0.0, "entropy": 0.0,
            "grad_norm": 0.0, "lr": self._get_lr(),
        }
        if not self.dataset:
            return empty_stats

        obs_t = torch.as_tensor(np.stack([t.obs for t in self.dataset]).astype(np.float32))
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

                total_loss += loss.item()
                total_log_prob += log_prob.mean().item()
                total_entropy += entropy.mean().item() if entropy is not None else 0.0
                total_grad_norm += grad_norm.item()
                n_steps += 1

        n = max(n_steps, 1)
        return {
            "loss":         total_loss / n,
            "log_prob_mean": total_log_prob / n,
            "entropy":       total_entropy / n,
            "grad_norm":     total_grad_norm / n,
            "lr":            self._get_lr(),
        }

    def _evaluate(self, n_eval_episodes: int) -> dict:
        """Run agent deterministically for n_eval_episodes."""
        total_reward = 0.0
        total_length = 0

        for _ in range(n_eval_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            done = np.zeros(self.env.num_envs, dtype=bool)
            ep_reward = 0.0
            ep_length = 0

            while not done[0]:
                action, _ = self.agent.predict(obs, deterministic=True)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated | truncated
                else:
                    obs, reward, done, _ = step_result
                ep_reward += float(reward[0])
                ep_length += 1

            total_reward += ep_reward
            total_length += ep_length

        return {
            "mean_ep_reward": total_reward / n_eval_episodes,
            "mean_ep_length": total_length / n_eval_episodes,
        }

    def _get_lr(self) -> float:
        return self._optimizer.param_groups[0]["lr"]