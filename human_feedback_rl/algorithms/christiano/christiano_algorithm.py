import math
from typing import Any, Callable, List, Optional, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr

from human_feedback_rl.common.types import FragmentPair, Trajectory
from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import RandomFragmenter
from human_feedback_rl.common.loggers import MainLogger, PrefixWrapper
from human_feedback_rl.common.reward_nets import RewardEnsemble, make_reward_ensemble
from human_feedback_rl.common.schedules import QUERY_SCHEDULES
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.preference_models import PreferenceModelFromReward
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.algorithms.christiano._shared import (
    save_reward_model as _save_reward_model,
    collect_debug_data as _collect_debug_data,
    compute_time_decay_weights,
    build_bootstrap_indices,
)

import torch as th
import wandb



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
        use_reward_reg: bool = True,
        reward_mean_reg: float = 0.1,
        rng: Optional[np.random.Generator] = np.random.default_rng(),
    ):
        self.batch_size_rew = batch_size_rew
        self.use_reward_reg = use_reward_reg
        self.reward_mean_reg = reward_mean_reg
        self.n_ephochs_rew = n_ephochs_rew
        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.n_iterations = n_iterations
        self.transition_oversampling = transition_oversampling
        self._iteration = 0
        self._rm_global_epoch = 0
        self._cumulative_timesteps = 0
        self.rng = rng

        self.logger    = MainLogger()   # iteration-level metrics (env/, agent/, hack/)
        self.rm_logger = MainLogger()   # per-epoch RM metrics (rm/)

        if wandb.run is not None:
            wandb.define_metric("rm/*",   step_metric="rm/epoch")
            wandb.define_metric("agent/*",  step_metric="total_timesteps")
            wandb.define_metric("env/*",  step_metric="iteration")
            wandb.define_metric("hack/*", step_metric="iteration")
            wandb.define_metric("total_timesteps")

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
            logger=self.logger,
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

            # Update cumulative timesteps
            self._cumulative_timesteps += num_steps

            self.logger.record("iteration", i)
            self.logger.record("total_timesteps", self._cumulative_timesteps)
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

        all_weights = compute_time_decay_weights(train_data, timestamp_idx=2, decay=decay)

        norms_before = [self._weight_norm(m) for m in self.reward_model.members]
        print(
            f"[DEBUG RM iter={self._iteration}] train_data={len(train_data)} total_epochs={total_epochs} "
            f"weight_norms_before={[f'{n:.2f}' for n in norms_before]}"
        )

        for member in self.reward_model.members:
            member.train()

        grad_steps   = [0] * len(self.reward_model.members)
        step_losses  = [[] for _ in self.reward_model.members]

        n_train = len(train_data)
        bootstrap_indices = build_bootstrap_indices(self.rng, n_train, len(self.reward_model.members))

        for epoch in range(total_epochs):
            epoch_pref_losses = []
            epoch_reg_losses  = []
            for mi, (member, optimizer) in enumerate(zip(self.reward_model.members, self.optimizers)):
                boot_idx = bootstrap_indices[mi]
                perm = self.rng.permutation(len(boot_idx))
                for start in range(0, len(perm), self.batch_size_rew):
                    batch_idx = boot_idx[perm[start : start + self.batch_size_rew]]

                    fragment_pairs = [train_data[i][0] for i in batch_idx]
                    preferences    = [train_data[i][1] for i in batch_idx]

                    w = all_weights[batch_idx]
                    w = w / w.sum()
                    batch_weights = th.tensor(w, dtype=th.float32)

                    r1_list, r2_list, all_rewards = [], [], []
                    step_rewards_list = []
                    for pair in fragment_pairs:
                        obs1 = th.tensor(np.array([t.observation for t in pair.frag1]), dtype=th.float32)
                        act1 = th.tensor(np.array([t.action      for t in pair.frag1]), dtype=th.float32)
                        obs2 = th.tensor(np.array([t.observation for t in pair.frag2]), dtype=th.float32)
                        act2 = th.tensor(np.array([t.action      for t in pair.frag2]), dtype=th.float32)
                        r1_raw = member(obs1, act1)
                        r2_raw = member(obs2, act2)
                        r1_list.append(r1_raw.sum())
                        r2_list.append(r2_raw.sum())

                        # Creiamo la lista per i singoli step SOLO se la regolarizzazione è attiva
                        # per risparmiare memoria ed evitare computazioni nel grafo se non serve.
                        if self.use_reward_reg:
                            step_rewards_list.append(r1_raw)
                            step_rewards_list.append(r2_raw)

                    logits = th.stack(r1_list) - th.stack(r2_list)
                    labels = th.tensor([p.pref1 for p in preferences], dtype=th.float32)

                    per_pair_loss = th.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction='none'
                    )

                    # loss base (binary cross entropy pesata)
                    pref_loss = (batch_weights * per_pair_loss).sum()

                    # Applichiamo la regolarizzazione in modo condizionale
                    if self.use_reward_reg:
                        all_step_rewards = th.cat(step_rewards_list)
                        reg_loss = self.reward_mean_reg * all_step_rewards.mean().pow(2)
                        loss = pref_loss + reg_loss
                        epoch_reg_losses.append(reg_loss.item())
                    else:
                        loss = pref_loss
                        epoch_reg_losses.append(0.0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step_losses[mi].append(loss.item())
                    epoch_pref_losses.append(pref_loss.item())

                    # print logit scale at first step to detect explosion early
                    if grad_steps[mi] == 0 and mi == 0 and epoch == 0:
                        print(
                            f"[DEBUG RM iter={self._iteration}] epoch=0 step=0 member=0 | "
                            f"logits: min={logits.min().item():.3f} max={logits.max().item():.3f} "
                            f"abs_mean={logits.abs().mean().item():.3f} | loss={loss.item():.4f}"
                        )
                    grad_steps[mi] += 1

            train_loss, _ = self._evaluate_reward_model(split="train")
            val_loss, _   = self._evaluate_reward_model(split="val")
            _, val_spearman = self._evaluate_reward_correlation(split="val")

            mean_pref_loss = float(np.mean(epoch_pref_losses))
            mean_reg_loss  = float(np.mean(epoch_reg_losses)) if epoch_reg_losses else 0.0

            self.rm_logger.record("rm/epoch",            self._rm_global_epoch)
            self.rm_logger.record("rm/val_loss",         val_loss)
            self.rm_logger.record("rm/overfit_gap_loss", val_loss - train_loss)
            self.rm_logger.record("rm/val_spearman",     val_spearman)
            self.rm_logger.record("rm/loss_total",       mean_pref_loss + mean_reg_loss)
            self.rm_logger.dump()
            self._rm_global_epoch += 1

        for member in self.reward_model.members:
            member.eval()

        norms_after = [self._weight_norm(m) for m in self.reward_model.members]
        # Summarize loss trajectory per member: first, mid, last
        loss_summary = []
        for losses in step_losses:
            if losses:
                mid = losses[len(losses)//2]
                loss_summary.append(f"[{losses[0]:.3f}->{mid:.3f}->{losses[-1]:.3f}]")
        print(
            f"[DEBUG RM iter={self._iteration}] DONE | "
            f"loss first->mid->last per member: {loss_summary} | "
            f"weight_norms_after={[f'{n:.2f}' for n in norms_after]} "
            f"(delta={[f'{a-b:.2f}' for a,b in zip(norms_after, norms_before)]})"
        )


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

    def _evaluate_reward_model_seg1(self, split: str) -> tuple[float, float]:
        """Evaluate the reward model on single-step pairs, identical method to
        _evaluate_reward_model but with fragments of length 1.

        Each original fragment pair (frag1, frag2) is expanded into len(frag) single-step
        pairs (frag1[i], frag2[i]), each inheriting the same dataset label p.pref1.
        When fragment_length=1 this is identical to _evaluate_reward_model.
        """
        data = self.dataset.get_train() if split == "train" else self.dataset.get_val()
        fragment_pairs, preferences, _ = zip(*data)

        single_step_pairs = []
        for pair, _ in zip(fragment_pairs, preferences):
            for t1, t2 in zip(pair.frag1, pair.frag2):
                single_step_pairs.append(FragmentPair(
                    frag1=Trajectory([t1]),
                    frag2=Trajectory([t2]),
                ))

        self.preference_model.eval()
        with th.no_grad():
            bt_probs = self.preference_model(single_step_pairs)
        self.preference_model.train()

        label_list = []
        for pair in single_step_pairs:
            r1 = pair.frag1[0].true_reward
            r2 = pair.frag2[0].true_reward
            if r1 > r2:
                label_list.append([1.0, 0.0])
            elif r2 > r1:
                label_list.append([0.0, 1.0])
            else:
                label_list.append([0.5, 0.5])
        labels = th.tensor(label_list, dtype=th.float32)

        loss = -(labels * bt_probs.log()).sum(dim=1).mean().item()
        acc  = (bt_probs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()

        return loss, acc

    def _evaluate_reward_correlation(self, split: str) -> tuple[float, float]:
        """
        Calcola la correlazione (Pearson e Spearman) tra la True Reward e la
        Predicted Reward per tutti i frammenti nel dataset indicato.
        """
        data = self.dataset.get_train() if split == "train" else self.dataset.get_val()
        if not data:
            return 0.0, 0.0

        fragment_pairs, _, _ = zip(*data)

        # Estraiamo tutti i frammenti individuali dalle coppie
        fragments = []
        for pair in fragment_pairs:
            fragments.append(pair.frag1)
            fragments.append(pair.frag2)

        # Calcoliamo la true_reward per ogni frammento
        true_rewards = [f.total_reward() for f in fragments]

        self.reward_model.eval()
        device = next(self.reward_model.parameters()).device

        pred_rewards = []
        with th.no_grad():
            for f in fragments:
                obs = th.tensor(np.array([t.observation for t in f]), dtype=th.float32, device=device)
                act = th.tensor(np.array([t.action for t in f]), dtype=th.float32, device=device)
                # Calcoliamo la predicted reward (somma sui timestep)
                pred_rew = self.reward_model(obs, act).sum().item()
                pred_rewards.append(pred_rew)

        self.reward_model.train()

        # Sicurezza matematica: se tutte le true_reward o pred_reward sono uguali
        # (varianza zero), scipy lancia un warning e ritorna NaN. Ritorniamo 0.
        if np.std(true_rewards) < 1e-6 or np.std(pred_rewards) < 1e-6:
            return 0.0, 0.0

        p_corr, _ = pearsonr(true_rewards, pred_rewards)
        s_corr, _ = spearmanr(true_rewards, pred_rewards)

        # Protezione addizionale contro eventuali NaN derivanti da scipy
        if np.isnan(p_corr): p_corr = 0.0
        if np.isnan(s_corr): s_corr = 0.0

        return float(p_corr), float(s_corr)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save_reward_model(self, path) -> None:
        _save_reward_model(self.reward_model, path)

    def collect_debug_data(self, n_steps: int = 2000) -> dict:
        return _collect_debug_data(self.trajectory_generator, self.reward_model, n_steps)


