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
        lr_rew: float = 0.1,
        batch_size_rew: int = 100,
        n_ephochs_rew: int = 3,
        n_ensembles_rew: int = 4,
        n_iterations: int = 100,
        train_comparison_frac: int = 0.7,
        fragment_length: int = 1,
        transition_oversampling: int = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: int = 5,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        comparison_queue_size: int = 1_000_000,
        rng: Optional[np.random.Generator] = np.random.default_rng(),
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

        self.optimizers = [
            th.optim.Adam(member.parameters(), lr=lr_rew, weight_decay=1e-4)
            for member in self.reward_model.members
        ]

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

            # Debug: preference label distribution (are labels balanced?)
            pref1_vals = [p.pref1 for p in preferences]
            n_prefer_1 = sum(1 for v in pref1_vals if v > 0.5)
            n_prefer_2 = sum(1 for v in pref1_vals if v < 0.5)
            n_ties     = sum(1 for v in pref1_vals if v == 0.5)
            print(
                f"[DEBUG Train iter={i}] preferences: prefer_frag1={n_prefer_1} prefer_frag2={n_prefer_2} ties={n_ties} "
                f"(total={len(preferences)})"
            )


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

        return self.trajectory_generator.agent


    def _weight_norm(self, member) -> float:
        return float(sum(p.data.norm().item() for p in member.parameters()))

    def train_reward_model(self, epoch_multiplier: float = 1.0, decay: float = 0.01):
        total_epochs = max(1, int(round(self.n_ephochs_rew * epoch_multiplier)))

        train_data = self.dataset.get_train()
        if not train_data:
            return

        # Compute time-decay weights over the full training set once
        t_vals = np.array([item[2] for item in train_data], dtype=np.float32)
        t_normalized = (t_vals - t_vals.min()) / (t_vals.max() - t_vals.min() + 1e-8)
        all_weights = np.exp(decay * t_normalized)  # (N,) unnormalized

        norms_before = [self._weight_norm(m) for m in self.reward_model.members]
        print(
            f"[DEBUG RM iter={self._iteration}] train_data={len(train_data)} total_epochs={total_epochs} "
            f"weight_norms_before={[f'{n:.2f}' for n in norms_before]}"
        )

        all_member_losses = []  # track loss trajectory for each member

        for mi, (member, optimizer) in enumerate(zip(self.reward_model.members, self.optimizers)):
            member.train()
            step_losses = []
            grad_step = 0
            for epoch in range(total_epochs):
                # Independent random permutation per member per epoch
                indices = self.rng.permutation(len(train_data))
                for start in range(0, len(indices), self.batch_size_rew):
                    batch_idx = indices[start : start + self.batch_size_rew]

                    fragment_pairs = [train_data[i][0] for i in batch_idx]
                    preferences    = [train_data[i][1] for i in batch_idx]

                    w = all_weights[batch_idx]
                    w = w / w.sum()
                    batch_weights = th.tensor(w, dtype=th.float32)

                    r1_list, r2_list = [], []
                    for pair in fragment_pairs:
                        obs1 = th.tensor(np.array([t.observation for t in pair.frag1]), dtype=th.float32)
                        act1 = th.tensor(np.array([t.action      for t in pair.frag1]), dtype=th.float32)
                        obs2 = th.tensor(np.array([t.observation for t in pair.frag2]), dtype=th.float32)
                        act2 = th.tensor(np.array([t.action      for t in pair.frag2]), dtype=th.float32)
                        r1_list.append(member(obs1, act1).sum())
                        r2_list.append(member(obs2, act2).sum())

                    logits = th.stack(r1_list) - th.stack(r2_list)
                    labels = th.tensor([p.pref1 for p in preferences], dtype=th.float32)

                    per_pair_loss = th.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction='none'
                    )
                    loss = (batch_weights * per_pair_loss).sum()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step_losses.append(loss.item())

                    # print logit scale at first step to detect explosion early
                    if grad_step == 0 and mi == 0:
                        print(
                            f"[DEBUG RM iter={self._iteration}] epoch=0 step=0 member=0 | "
                            f"logits: min={logits.min().item():.3f} max={logits.max().item():.3f} "
                            f"abs_mean={logits.abs().mean().item():.3f} | loss={loss.item():.4f}"
                        )
                    grad_step += 1

            member.eval()
            all_member_losses.append(step_losses)

        norms_after = [self._weight_norm(m) for m in self.reward_model.members]
        # Summarize loss trajectory per member: first, mid, last
        loss_summary = []
        for losses in all_member_losses:
            if losses:
                mid = losses[len(losses)//2]
                loss_summary.append(f"[{losses[0]:.3f}->{mid:.3f}->{losses[-1]:.3f}]")
        print(
            f"[DEBUG RM iter={self._iteration}] DONE | "
            f"loss first->mid->last per member: {loss_summary} | "
            f"weight_norms_after={[f'{n:.2f}' for n in norms_after]} "
            f"(delta={[f'{a-b:.2f}' for a,b in zip(norms_after, norms_before)]})"
        )

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

