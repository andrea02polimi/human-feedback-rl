import os
import sys
import pickle
import random
from pathlib import Path

import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf

import sumo_rl_ego as sre
import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

AGENT_CLASSES = {"PPO": PPO, "SAC": SAC}

from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.reward_nets import SumoRewardNet
from human_feedback_rl.common.replay_buffers import RelabelReplayBuffer
from human_feedback_rl.common.loggers import (
    Logger,
    WandbWriter,
    PrefixedLogger,
    ExcludeFormatLogger,
    NullLogger,
)


class DemoAlgorithmV2():
    def __init__(
        self,
        env,
        agent,
        reward_model,
        expert_trajectories,
        lr_rew,
        gradient_steps_rew,
        batch_size_expert,
        batch_size_model,
        log_interval=25,
        logger=None,
        rng=None,
    ):
        self.env = env
        self.rng = rng if rng is not None else np.random.default_rng()
        self.logger = logger if logger is not None else NullLogger()

        # Route every metric the SB3 agent records (including those emitted by the
        # CustomLoggingCallback, e.g. event_rate/successes) through a logger that
        # prefixes the keys with "agent/" and is wired to WandB. The result is
        # logged to wandb as "agent/event_rate/successes". The stdout sink is
        # excluded so PPO's internal table is not duplicated on the console.
        agent.set_logger(ExcludeFormatLogger(PrefixedLogger(self.logger, "agent"), exclude="stdout"))

        self.agent = TrajectoryGeneratorFromAgent(
            venv=env,
            agent=agent,
            reward_model=reward_model,
            rng=self.rng,
            logger=self.logger,
        )
        self.reward_model = reward_model
        self.optimizer = th.optim.Adam(reward_model.parameters(), lr=lr_rew)

        self.gradient_steps_rew = gradient_steps_rew
        self.batch_size_expert = batch_size_expert
        self.batch_size_model = batch_size_model
        self.log_interval = log_interval
        self.expert_trajectories = list(expert_trajectories)

    def train(self,
              total_timesteps,
              timesteps_per_iteration,
              checkpoint_dir=None,
              checkpoint_interval=10,):
        num_iterations = total_timesteps // timesteps_per_iteration
        for iteration in range(num_iterations):
            print(f"=== Iteration {iteration+1}/{num_iterations} ===")
            trajectories = self._collect_trajectories(timesteps_per_iteration)
            self._update_reward_model(trajectories)
            # Reward model just changed: recompute the rewards already stored in an
            # off-policy replay buffer (no-op for on-policy agents like PPO).
            self.agent.relabel_replay_buffer()
            self._update_policy(timesteps_per_iteration)

            self.logger.record("iterations", iteration)
            self.logger.record("agent/time/total_timesteps", self.agent.agent.num_timesteps)
            self.logger.dump()

            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, iteration + 1)

        return self.agent.agent

    def _collect_trajectories(self, timesteps_per_iteration):
        trajectories = self.agent.sample(agent_steps=timesteps_per_iteration)
        return trajectories

    def _update_reward_model(self, trajectories):
        for step in range(self.gradient_steps_rew):
            # Expert mini-batch
            exp_idx = self.rng.choice(len(self.expert_trajectories), size=self.batch_size_expert, replace=True)
            expert_returns = self._sum_rewards(self.expert_trajectories, exp_idx)

            # Model mini-batch
            agent_idx = self.rng.choice(len(trajectories), size=self.batch_size_model, replace=True)
            agent_returns = self._sum_rewards(trajectories, agent_idx)

            total_returns = th.cat([expert_returns, agent_returns])
            log_z = th.logsumexp(total_returns, dim=0) - th.log(th.tensor(len(exp_idx) + len(agent_idx), dtype=th.float32))

            # minimize -agent_returns.mean() + expert_returns.mean()
            loss = -th.mean(expert_returns) + th.mean(agent_returns) + log_z
            self.optimizer.zero_grad()
            loss.backward()

            # Total L2 norm of the gradient over all reward-model parameters,
            # computed after backward() and before the optimizer step.
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.reward_model.parameters(), max_norm=float("inf")
            )

            self.optimizer.step()

            self.logger.record_mean("reward_model/loss", loss.item())
            self.logger.record_mean("reward_model/grad_norm", grad_norm.item())

    def _update_policy(self, timesteps_per_iteration):
        self.agent.train(steps=timesteps_per_iteration, log_interval=self.log_interval)

    def _sum_rewards(self, trajectories, idx) -> th.Tensor:
        """Return a (len(idx),) tensor of per-trajectory reward-model returns (with grad)."""
        return th.stack([self._traj_sum_reward(trajectories[i]) for i in idx])

    def _traj_sum_reward(self, traj) -> th.Tensor:
        """Sum of per-step reward-model outputs over a trajectory (supports gradients)."""
        obs         = th.tensor(np.array([t.observation  for t in traj]), dtype=th.float32)
        actions     = th.tensor(np.array([t.action       for t in traj]), dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status  for t in traj]), dtype=th.float32)
        done        = th.tensor(np.array([float(t.done)  for t in traj]), dtype=th.float32)
        return self.reward_model(obs, actions, next_status, done).sum()

    def save_checkpoint(self, checkpoint_dir, iteration):
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}")
        os.makedirs(ckpt_path, exist_ok=True)
        th.save(self.reward_model.state_dict(), os.path.join(ckpt_path, "reward_model.pt"))
        self.agent.agent.save(os.path.join(ckpt_path, "agent"))
        print(f"  checkpoint saved in {ckpt_path}")


# ==========

def make_run_dir(output_dir: Path, name: str) -> Path:
    candidate = output_dir / name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    i = 1
    while True:
        candidate = output_dir / f"{name}_{i:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        i += 1


def get_name(cfg):
    seed = cfg.run.seed
    group_name = "demo_v2"
    run_name = group_name + f" seed={seed}"
    return group_name, run_name


@hydra.main(version_base=None, config_path=".", config_name="demo_algorithm_v2")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)

    group_name, run_name = get_name(cfg)
    run_dir = make_run_dir(Path(cfg.run.output_dir), run_name)

    config = OmegaConf.to_container(cfg, resolve=True)
    config["group_name"] = group_name

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=config,
        group=group_name,
        name=run_name,
        tags=OmegaConf.to_container(cfg.wandb.tags, resolve=True) if cfg.wandb.tags else None,
        dir=str(run_dir),
    )

    # Shared logger wired to WandB; the agent's metrics flow into it via the
    # "agent/" PrefixedLogger set inside DemoAlgorithmV2.
    logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout), WandbWriter()])

    print(OmegaConf.to_yaml(cfg))

    print(f"Loading expert trajectories...")
    expert_trajectories_path = Path(__file__).resolve().parents[3] / "data_for_training/expert_trajectories.pkl"

    with open(expert_trajectories_path, "rb") as f:
        expert_trajectories = pickle.load(f)

    print(f"Loaded {len(expert_trajectories)} expert trajectories")

    print("Creating environment...")
    env = sre.make_vec_env(cfg.env.id, n_envs=cfg.env.n_envs, base_seed=seed, **OmegaConf.to_container(cfg.env.kwargs, resolve=True))

    print(f"Initializing agent ({cfg.agent.type})...")
    try:
        agent_cls = AGENT_CLASSES[cfg.agent.type]
    except KeyError:
        raise ValueError(
            f"Unknown agent type '{cfg.agent.type}'. Available: {list(AGENT_CLASSES)}"
        )
    agent_kwargs = OmegaConf.to_container(cfg.agent.kwargs, resolve=True)
    if issubclass(agent_cls, OffPolicyAlgorithm):
        # Relabellable buffer so SAC's stored rewards can be recomputed each time the
        # reward model is updated (see DemoAlgorithmV2.train).
        agent_kwargs.setdefault("replay_buffer_class", RelabelReplayBuffer)
    agent = agent_cls(env=env, seed=seed, **agent_kwargs)

    print("Initializing reward model...")
    algo_kwargs = OmegaConf.to_container(cfg.algo.kwargs, resolve=True)
    reward_model_kwargs = algo_kwargs.pop("reward_model_kwargs")
    reward_model = SumoRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **reward_model_kwargs,
    )

    print("Initializing algorithm...")
    algo = DemoAlgorithmV2(
        env=env,
        agent=agent,
        reward_model=reward_model,
        expert_trajectories=expert_trajectories,
        logger=logger,
        rng=rng,
        **algo_kwargs,
    )

    print("Starting training...")
    train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
    train_kwargs["checkpoint_dir"] = str(run_dir)
    agent = algo.train(**train_kwargs)


if __name__ == "__main__":
    main()
