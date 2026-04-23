import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import RandomFragmenter
from human_feedback_rl.common.loggers import MainLogger, PrefixWrapper
from human_feedback_rl.common.reward_nets import RewardEnsemble, SimpleRewardNet
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.preference_models import PreferenceModelFromReward
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward

import torch as th



def make_reward_ensemble(venv: VecEnv, n_ensembles: int = 3) -> RewardEnsemble:
    obs_space = venv.observation_space
    act_space = venv.action_space

    members = [
        SimpleRewardNet(obs_space, act_space)
        for _ in range(n_ensembles)
    ]

    return RewardEnsemble(obs_space, act_space, members)


QUERY_SCHEDULES: Dict[str, Callable[[float], float]] = {
    "constant": lambda t: 1.0,
    "hyperbolic": lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


class ChristianoAlgorithm:
    def __init__(
        self,
        env,
        agent,
        lr_rew: float = 0.01,
        batch_size_rew: int = 100,
        n_ephochs_rew: int = 10,
        n_ensembles_rew: int = 4,
        n_iterations: int = 10,
        train_comparison_frac: int = 0.1,
        fragment_length: int = 10,
        transition_oversampling: float = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        query_schedule: Union[str, Callable[[float], float]] = "hyperbolic",
        comparison_queue_size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.batch_size_rew = batch_size_rew
        self.n_ephochs_rew = n_ephochs_rew
        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.n_iterations = n_iterations
        self.transition_oversampling = transition_oversampling
        self._iteration = 0
        self.rng = rng
        
        self.logger = MainLogger()
            
        self.query_schedule = QUERY_SCHEDULES[query_schedule]
        self.query_schedule_name = query_schedule

        self.reward_model = make_reward_ensemble(env, n_ensembles=n_ensembles_rew)

        self.preference_model = PreferenceModelFromReward(self.reward_model)

        self.optimizer = th.optim.Adam(
            self.reward_model.parameters(), 
            lr=lr_rew,
        )

        self.trajectory_generator = TrajectoryGeneratorFromAgent(
            agent=agent,
            reward_model=self.reward_model,
            venv=env,
            rng=rng if rng is not None else np.random.default_rng(),
            logger=PrefixWrapper(self.logger, "agent"),
        )

        self.fragmenter = RandomFragmenter(
            logger=PrefixWrapper(self.logger, "fragmenter"),
            rng=rng if rng is not None else np.random.default_rng(),
        )

        self.dataset = PreferenceDataset(
            queue_size=comparison_queue_size,
            train_frac=train_comparison_frac,
        )

        self.preference_gatherer = PreferenceGathererFromReward()



    def train(self, 
            total_timesteps: int = 100_000, 
            total_comparisons: int = 10_000,
        ) -> Any:
        
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons = total_comparisons - initial_comparisons

        # Compute the number of comparisons to request at each iteration in advance.
        t_vec = np.linspace(0, 1, self.n_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        probs = weights / weights.sum()
        shares = np.round(probs * total_comparisons).astype(int)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"- Query {self.query_schedule_name} schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.n_iterations,
        )

        for i, num_pairs in enumerate(schedule):
            self.logger.log(f"\nIteration {i+1}/{len(schedule)}")

            ##########################
            # Gather new preferences #
            ##########################
            num_steps = math.ceil(self.transition_oversampling * 2 * num_pairs * self.fragment_length)
            
            self.logger.log(f"- Collecting {2 * num_pairs} fragments ({num_steps} transitions)")
            trajectories = self.trajectory_generator.sample(num_steps)

            self.logger.log("- Creating fragment pairs")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)

            self.logger.log("- Gathering preferences")
            preferences = self.preference_gatherer(fragments)

            self.dataset.push(fragments, preferences, i)
            self.logger.log(f"- Dataset now contains {len(self.dataset)} comparisons ({self.dataset.train_frac:.0%} used for training)")


            ##########################
            # Train the reward model #
            ##########################

            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier

            self.logger.log(f"- Training reward model for {epoch_multiplier*self.n_ephochs_rew} epochs")
            self.train_reward_model(epoch_multiplier, decay=0.01)


            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration

            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if i == self.n_iterations - 1:
                num_steps += extra_timesteps
                
            self.logger.log(f"- Training agent for {num_steps} timesteps")
            self.trajectory_generator.train(steps=num_steps)

            self.logger.record("iteration", i)
            self.logger.dump()

            self._iteration += 1


    def train_reward_model(self, epoch_multiplier: float = 1.0, decay: float = 0.01):
        total_epochs = max(1, int(round(self.n_ephochs_rew * epoch_multiplier)))

        for epoch in range(total_epochs):
            for batch in self.dataset.get(self.batch_size_rew):
                
                bt_probs = self.preference_model(batch.fragment_pairs)  # (batch_size, 2)
                
                labels = th.tensor(
                    [[p.pref1, p.pref2] for p in batch.preferences],
                    dtype=th.float32
                )  # (batch_size, 2)

                t = th.tensor(batch.timestamps, dtype=th.float32)
                t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-8)
                weights = th.exp(decay * t_normalized)
                weights = weights / weights.sum()

                per_pair_loss = -(labels * bt_probs.log()).sum(dim=1)
                loss = (weights * per_pair_loss).sum()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # logs
        train_loss, train_acc = self._evaluate_reward_model(split="train")
        val_loss, val_acc     = self._evaluate_reward_model(split="val")
        
        self.logger.record("reward/train_loss", train_loss)
        self.logger.record("reward/train_acc",  train_acc)
        self.logger.record("reward/val_loss",   val_loss)
        self.logger.record("reward/val_acc",    val_acc)

    
    def _evaluate_reward_model(self, split: str) -> tuple[float, float]:
        data = self.dataset.get_train() if split == "train" else self.dataset.get_val()
        fragment_pairs, preferences, _ = zip(*data)

        self.preference_model.eval()
        with th.no_grad():
            bt_probs = self.preference_model(list(fragment_pairs))  # (N, 2)
        self.preference_model.train()

        labels = th.tensor(
            [[p.pref1, p.pref2] for p in preferences],
            dtype=th.float32
        )

        loss = -(labels * bt_probs.log()).sum(dim=1).mean().item()
        acc  = (bt_probs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()

        return loss, acc
    
