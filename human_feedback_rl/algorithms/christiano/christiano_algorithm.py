import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import RandomFragmenter
from human_feedback_rl.common.loggers import MainLogger, PrefixWrapper
from human_feedback_rl.common.reward_nets import RewardEnsemble, SumoSimpleRewardNet
from human_feedback_rl.common.reward_trainers import EnsembleRewardTrainer
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent




def make_reward_ensemble(venv: VecEnv, n_members: int = 3) -> RewardEnsemble:
    obs_space = venv.observation_space
    act_space = venv.action_space

    members = [
        SumoSimpleRewardNet(obs_space, act_space)
        for _ in range(n_members)
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
        n_ensembles_reward_model: int = 4,
        lr_reward_model: int = 0.01,
        batch_size_reward_trainer: int = 32,
        num_iterations: int = 10,
        train_comparison_frac: int = 0.1,
        fragment_length: int = 100,
        transition_oversampling: float = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        query_schedule: Union[str, Callable[[float], float]] = "hyperbolic",
        comparison_queue_size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.num_iterations = num_iterations
        self.transition_oversampling = transition_oversampling
        self.train_comparison_frac = train_comparison_frac
        self._iteration = 0
        self.rng = rng
        
        self.logger = MainLogger()
            
        self.query_schedule = QUERY_SCHEDULES[query_schedule]
        self.query_schedule_name = query_schedule

        self.reward_model = make_reward_ensemble(env, n_members=5)

        self.reward_trainer = EnsembleRewardTrainer(
            reward_model=self.reward_model,
            logger=PrefixWrapper(self.logger, "reward"),
            batch_size=batch_size_reward_trainer,
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



    def train(self, 
            total_timesteps: int = 100_000, 
            total_comparisons: int = 10_000,
        ) -> Any:
        
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons = total_comparisons - initial_comparisons

        # Compute the number of comparisons to request at each iteration in advance.
        t_vec = np.linspace(0, 1, self.num_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        probs = weights / weights.sum()
        shares = np.round(probs * total_comparisons).astype(int)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"- Query {self.query_schedule_name} schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.num_iterations,
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
            #fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)

            self.logger.log("- Gathering preferences")
            #preferences = self.preference_gatherer(fragments)

            #self.dataset.push(fragments, preferences)
            self.logger.log(f"- Dataset now contains {len(self.dataset)} comparisons ({self.dataset.train_frac:.0%} used for training)")


            ##########################
            # Train the reward model #
            ##########################

            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier

            self.logger.log(f"- Training reward model for {epoch_multiplier*self.reward_trainer.epochs} epochs")
            #self.reward_trainer.train(self.dataset.get_train(), epoch_multiplier=epoch_multiplier)


            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration

            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps
                
            self.logger.log(f"- Training agent for {num_steps} timesteps")
            #self.trajectory_generator.train(steps=num_steps)

            self._iteration += 1


