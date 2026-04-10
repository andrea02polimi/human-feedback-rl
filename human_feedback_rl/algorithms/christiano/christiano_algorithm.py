import math
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

import numpy as np
from scipy import special
import wandb

from human_feedback_rl.common import (
    ActiveFragmenter,
    AgentTrainer,
    EnsembleRewardModel,
    Preference,
    PreferenceDataset,
    PreferenceModelFromReward,
    PrefixLogger,
    SegmentPair,
    Trajectory,
    UnifiedLogger,
)
from .preference_trainer import RewardTrainerChristiano


# ---------------------------------------------------------------------------
# Query schedules  (mirrors imitation's QUERY_SCHEDULES)
# ---------------------------------------------------------------------------

QUERY_SCHEDULES = {
    "constant":          lambda _: 1.0,
    "hyperbolic":        lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t ** 2),
}


def _oric(x: np.ndarray) -> np.ndarray:
    """Optimal Rounding with Integer Constraints.

    Rounds floats to integers while preserving the total sum: elements with
    the largest fractional parts are rounded up.  Mirrors ``imitation.util.util.oric``.

    Questo tipo di funzione viene usato per:

    distribuire campioni tra ambienti (es. VecEnv)
    allocare batch sizes interi
    mantenere proporzioni probabilistiche → conteggi interi
    """
    floored = np.floor(x).astype(int) # x = [1.2, 3.7, 4.4] -> floored = [1, 3. 4]
    deficit = int(round(x.sum())) - floored.sum() # somma desiderata finale - somma attuale dopo floor
    # definit ci dice quanti +1 dobbiamo distribuire
    top_indices = np.argsort(x - floored)[::-1][:deficit]
    floored[top_indices] += 1
    return floored


# ---------------------------------------------------------------------------
# SyntheticGatherer
# ---------------------------------------------------------------------------

class SyntheticGatherer:
    """Computes synthetic preferences using ground-truth environment rewards.

    Mirrors ``imitation.algorithms.preference_comparisons.SyntheticGatherer``
    but operates on our :class:`~.SegmentPair` / :class:`~.Trajectory` data
    structures instead of imitation's ``TrajectoryWithRew``.

    The generated label for a pair is the probability that ``seg1`` is
    preferred over ``seg2``:

    * ``temperature > 0`` (default): Boltzmann-rational probability
      ``P = 1 / (1 + exp((R2 − R1) / T))``, optionally hard-sampled via
      Bernoulli(P) when ``sample=True``.
    * ``temperature == 0``: deterministic argmax — ``(sign(R1 − R2) + 1) / 2``.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        discount_factor: float = 1.0,
        sample: bool = True,
        rng: Optional[np.random.Generator] = None,
        threshold: float = 50.0,
    ) -> None:
        """Initialize the synthetic gatherer.

        Args:
            temperature: softmax temperature. ``0`` → deterministic argmax.
            discount_factor: discount factor for computing trajectory returns.
                Default ``1.0`` = undiscounted sums (as in the DRLHP paper).
            sample: if ``True`` (default), hard 0 / 1 labels sampled from
                Bernoulli(P); if ``False``, returns the soft probability.
            rng: random number generator, required when ``temperature > 0``
                and ``sample=True``.
            threshold: clip logit differences to ``±threshold`` before the
                softmax to avoid numerical overflow.  Default ``50`` mirrors
                imitation.

        Raises:
            ValueError: if ``sample=True`` and ``rng`` is not provided.
        """
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = rng
        self.threshold = threshold

        if self.sample and self.rng is None:
            raise ValueError("If `sample` is True, then `rng` must be provided.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        segment_pairs: List[SegmentPair],
    ) -> tuple[np.ndarray, float]:
        """Gather preference labels for a batch of segment pairs.

        Mirrors ``SyntheticGatherer.__call__`` in imitation.

        Args:
            segment_pairs: batch of segment pairs to label.

        Returns:
            preferences: float32 array of shape ``(len(segment_pairs),)``.
                Each value is the probability that ``seg1`` is preferred.
                When ``sample=True`` values are in ``{0.0, 1.0}``.
            entropy: mean binary entropy of the *soft* preference distribution
                (computed before hard sampling).  Useful as a diversity metric.
        """
        if not segment_pairs:
            return np.array([], dtype=np.float32), 0.0

        returns1, returns2 = self._reward_sums(segment_pairs)

        if self.temperature == 0:
            # Deterministic branch — mirrors imitation's temperature=0 path.
            prefs = ((np.sign(returns1 - returns2) + 1) / 2).astype(np.float32)
            # Entropy of a deterministic distribution is 0 everywhere except at
            # ties (prefs == 0.5), which we handle via xlogy's 0·log(0) = 0 rule.
            entropy = float(
                -(
                    special.xlogy(prefs, prefs)
                    + special.xlogy(1.0 - prefs, 1.0 - prefs)
                ).mean()
            )
            return prefs, entropy

        # Boltzmann branch — mirrors imitation's default (temperature=1) path.
        r1_t = returns1 / self.temperature
        r2_t = returns2 / self.temperature
        returns_diff = np.clip(r2_t - r1_t, -self.threshold, self.threshold)
        model_probs = (1.0 / (1.0 + np.exp(returns_diff))).astype(np.float32)

        # Binary entropy using scipy.special.xlogy so that 0·log(0) = 0 (mirrors
        # imitation's ``-(xlogy(model_probs, model_probs) + xlogy(...)).mean()``).
        entropy = float(
            -(
                special.xlogy(model_probs, model_probs)
                + special.xlogy(1.0 - model_probs, 1.0 - model_probs)
            ).mean()
        )

        if self.sample:
            assert self.rng is not None
            return self.rng.binomial(n=1, p=model_probs).astype(np.float32), entropy
        return model_probs, entropy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reward_sums(
        self, segment_pairs: List[SegmentPair]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (discounted) return sums for both segments in each pair.

        Mirrors ``SyntheticGatherer._reward_sums`` in imitation.
        """
        rews1, rews2 = [], []
        for pair in segment_pairs:
            rews1.append(self._discounted_sum(pair.seg1))
            rews2.append(self._discounted_sum(pair.seg2))
        return (
            np.array(rews1, dtype=np.float32),
            np.array(rews2, dtype=np.float32),
        )

    def _discounted_sum(self, segment) -> float:
        """Compute the (discounted) sum of rewards for a single segment.

        Mirrors ``imitation.data.rollout.discounted_sum``.
        """
        rewards = [t.reward for t in segment.transitions]
        if self.discount_factor == 1.0:
            return float(sum(rewards))
        total = 0.0
        discount = 1.0
        for r in rewards:
            total += discount * r
            discount *= self.discount_factor
        return total


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_best_checkpoint(
    agent,
    checkpoint_dir: str,
    avg_true_reward: float,
    iteration: int,
) -> None:
    """Overwrite the best-model checkpoint with the current agent."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / "best_model"
    agent.save(model_path)
    print(
        f"      [checkpoint] New best avg_true_reward={avg_true_reward:.3f} "
        f"at iteration {iteration + 1} → {model_path}.zip"
    )
    if wandb.run is not None:
        wandb.run.summary["best_avg_true_reward"] = avg_true_reward
        wandb.run.summary["best_model_iteration"] = iteration + 1


# ---------------------------------------------------------------------------
# Main algorithm — mirrors PreferenceComparisons
# ---------------------------------------------------------------------------

class ChristianoAlgorithm:
    """Reward learning via preference comparisons (Christiano et al., 2017).

    Mirrors ``imitation.algorithms.preference_comparisons.PreferenceComparisons``
    using our custom components for trajectory generation, fragmenting, and
    reward-model training.

    Training alternates between five phases (per iteration):

    1. **Collect** trajectories (true env rewards) via :class:`~.AgentTrainer`.
    2. **Fragment** trajectories into segment pairs via :class:`~.ActiveFragmenter`.
    3. **Gather** preference labels via :class:`SyntheticGatherer`.
    4. **Train** the ensemble reward model via :class:`~.RewardTrainerChristiano`.
    5. **Train** the RL agent on model rewards via :class:`~.AgentTrainer`.

    The comparison budget is distributed across iterations via a query schedule
    (constant / hyperbolic / inverse-quadratic), with a larger allocation and
    longer reward-model training on iteration 0 — exactly as in imitation.
    """

    def __init__(
        self,
        env,
        agent,
        num_iterations: int,
        n_ensembles: int,
        segment_length: int,
        device: str = "cpu",
        # Reward model
        lr_reward_model: float = 1e-4,
        comparison_queue_size: Optional[int] = None,
        # Reward trainer (mirrors BasicRewardTrainer / EnsembleTrainer kwargs)
        reward_trainer_epochs: int = 1,
        reward_model_batch_size: int = 32,
        # Comparison / query schedule
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        transition_oversampling: float = 1.0,
        query_schedule: Union[str, Callable] = "hyperbolic",
        # SyntheticGatherer parameters
        preference_temperature: float = 1.0,
        preference_sample: bool = True,
        preference_discount_factor: float = 1.0,
        preference_threshold: float = 50.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize the preference-comparison trainer.

        Args:
            env: vectorised environment.
            agent: SB3 algorithm (PPO, SAC, …) used as the RL agent.
            num_iterations: number of *main* training iterations (agent +
                reward-model updates).  The actual loop runs
                ``num_iterations + 1`` times: one extra for the initial
                reward-model warm-up.  Mirrors imitation's ``num_iterations``.
            n_ensembles: number of reward-model ensemble members.
            segment_length: length (in timesteps) of each trajectory fragment
                used for preference queries.
            device: torch device for the reward model.
            lr_reward_model: learning rate for the reward model optimizers.
            comparison_queue_size: maximum number of comparisons stored in the
                dataset (FIFO eviction).  ``None`` = unbounded.
            reward_trainer_epochs: base number of reward-model training epochs
                per iteration.  Scaled by ``initial_epoch_multiplier`` on
                iteration 0.  Mirrors ``BasicRewardTrainer(epochs=...)``.
            reward_model_batch_size: mini-batch size for reward-model training.
            initial_comparison_frac: fraction of ``total_comparisons`` gathered
                before any agent training (warm-up phase).
            initial_epoch_multiplier: reward-model epoch multiplier applied on
                iteration 0 so the model is calibrated before the agent trains.
            transition_oversampling: oversample factor for trajectory collection.
                Exactly ``ceil(oversampling × 2 × num_pairs × segment_length)``
                transitions are requested from ``AgentTrainer.sample()`` —
                the same formula imitation uses before calling
                ``trajectory_generator.sample(num_steps)``.
            query_schedule: how to distribute the remaining comparisons across
                the main iterations.  One of ``"constant"``, ``"hyperbolic"``,
                ``"inverse_quadratic"``, or a callable ``t ∈ [0,1] → weight``.
            preference_temperature: softmax temperature for the synthetic
                gatherer (``0`` = deterministic argmax).
            preference_sample: if ``True``, hard 0/1 labels sampled from
                Bernoulli; if ``False``, returns soft probabilities.
            preference_discount_factor: discount factor for segment return sums
                (default ``1.0`` = undiscounted, as in the DRLHP paper).
            preference_threshold: numerical clipping threshold for logit
                differences inside the gatherer.
            rng: random number generator (required when
                ``preference_sample=True``).

        Raises:
            ValueError: if ``preference_sample=True`` and ``rng`` is not given,
                or if ``query_schedule`` is an unknown string.
        """
        self.logger = UnifiedLogger()
        self.num_iterations = num_iterations
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.transition_oversampling = transition_oversampling
        self._iteration = 0  # global iteration counter (survives multiple train() calls)

        # ---- Reward model + preference model --------------------------------
        discrete = hasattr(env.action_space, "n")
        self.reward_model = EnsembleRewardModel(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n if discrete else env.action_space.shape[0],
            n_ensembles=n_ensembles,
            lr=lr_reward_model,
            device=device,
            discrete_actions=discrete,
        )
        self.preference_model = PreferenceModelFromReward(self.reward_model)

        # ---- Reward trainer (mirrors EnsembleTrainer) -----------------------
        self.reward_trainer = RewardTrainerChristiano(
            self.preference_model,
            epochs=reward_trainer_epochs,
            batch_size=reward_model_batch_size,
            logger=PrefixLogger(self.logger, "reward_model"),
        )

        # ---- Fragmenter (mirrors ActiveSelectionFragmenter) -----------------
        self.fragmenter = ActiveFragmenter(
            reward_model=self.reward_model,
            segment_length=segment_length,
        )

        # ---- Preference gatherer (mirrors SyntheticGatherer) ----------------
        self.preference_gatherer = SyntheticGatherer(
            temperature=preference_temperature,
            discount_factor=preference_discount_factor,
            sample=preference_sample,
            rng=rng,
            threshold=preference_threshold,
        )

        # ---- Dataset (mirrors PreferenceDataset with comparison_queue_size) -
        self.dataset = PreferenceDataset(
            comparison_queue_size if comparison_queue_size is not None else int(1e10)
        )

        # ---- Segment length (needed to convert pairs → transitions for sample)
        self.segment_length = segment_length

        # ---- Agent trainer (mirrors imitation's AgentTrainer) ---------------
        self.agent_trainer = AgentTrainer(
            agent=agent,
            venv=env,
            reward_model=self.reward_model,
            segment_length=segment_length,
            agent_logger=PrefixLogger(self.logger, "agent"),
        )

        # ---- Query schedule -------------------------------------------------
        if callable(query_schedule):
            self.query_schedule = query_schedule
        elif query_schedule in QUERY_SCHEDULES:
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(
                f"Unknown query schedule: {query_schedule!r}. "
                f"Choose from {list(QUERY_SCHEDULES.keys())} or pass a callable."
            )

    # -----------------------------------------------------------------------
    # Training loop — mirrors PreferenceComparisons.train()
    # -----------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: Optional[Callable[[int], None]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Train reward model and agent interleaved over ``num_iterations`` loops.

        Mirrors ``PreferenceComparisons.train(total_timesteps, total_comparisons,
        callback)`` exactly.

        Comparison budget distribution (matches imitation):

        * ``initial_comparison_frac × total_comparisons`` pairs gathered at
          iteration 0, reward model trained for
          ``initial_epoch_multiplier × reward_trainer_epochs`` epochs.
        * Remaining budget distributed across the ``num_iterations`` main
          iterations via the query schedule (ORIC rounding).

        Timestep distribution (matches imitation):

        * ``timesteps_per_iteration = total_timesteps // num_iterations``
        * Extra remainder timesteps are added at iteration
          ``num_iterations − 1`` (the last of the *main* iterations).

        Args:
            total_timesteps: total environment timesteps for agent training.
            total_comparisons: total preference pairs to gather across all
                iterations.
            callback: called at the end of every iteration with the global
                iteration index.
            checkpoint_dir: if given, the best agent (by avg true reward) is
                saved here after each agent-training phase.

        Returns:
            ``{"reward_loss": float, "reward_accuracy": float}`` — loss and
            accuracy on the training dataset after the final iteration (mirrors
            imitation's return value).
        """
        import time

        # --- Comparison schedule (exact imitation logic) --------------------
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        remaining_comparisons = total_comparisons - initial_comparisons

        # linspace(0, 1, num_iterations) → num_iterations weights for the
        # main iterations (mirrors imitation exactly).
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized / unnormalized.sum()
        shares = _oric(probs * remaining_comparisons)
        # schedule[0] = warm-up; schedule[1..num_iterations] = main iterations.
        schedule = [initial_comparisons] + shares.tolist()

        # --- Timestep distribution (exact imitation logic) ------------------
        # Total agent steps split evenly; remainder added at iteration
        # num_iterations − 1 (mirrors imitation's divmod approach).
        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps, self.num_iterations
        )

        _W = 62  # terminal column width for separators
        print(
            f"\n{'═' * _W}\n"
            f" PreferenceComparisons  │  "
            f"iterations={len(schedule)}  "
            f"comparisons={total_comparisons}  "
            f"timesteps={total_timesteps:,}\n"
            f" schedule (pairs/iter): {schedule}\n"
            f"{'═' * _W}"
        )

        reward_loss: Optional[float] = None
        reward_accuracy: Optional[float] = None
        best_avg_true_reward = -np.inf
        agent_global_timesteps = 0
        t_iter_start = time.time()

        for i, num_pairs in enumerate(schedule):
            t_iter_start = time.time()
            is_warmup = (i == 0)
            label = "warm-up" if is_warmup else f"iter {i}/{self.num_iterations}"
            print(
                f"\n{'─' * _W}\n"
                f" [{label}]  pairs={num_pairs}  "
                f"global_iter={self._iteration + 1}\n"
                f"{'─' * _W}"
            )

            # ---- 1) Collect trajectories (true env rewards) ----------------
            # Budget expressed in *transitions* — mirrors imitation exactly:
            #   num_steps = ceil(oversampling × 2 × num_pairs × fragment_length)
            # AgentTrainer.sample(n_steps) counts transitions (not segments).
            n_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.segment_length
            )
            t0 = time.time()
            print(f"[1/5] Collecting rollouts  (budget={n_steps} transitions) …")
            trajectories = self.agent_trainer.sample(n_steps)
            t_collect = time.time() - t0

            ep_lengths   = [len(t.transitions) for t in trajectories]
            true_rewards = [t.total_reward()    for t in trajectories]
            n_episodes       = len(trajectories)
            n_transitions    = sum(ep_lengths)
            avg_ep_length    = float(np.mean(ep_lengths))
            avg_true_reward  = float(np.mean(true_rewards))
            print(
                f"      episodes={n_episodes}  transitions={n_transitions}  "
                f"ep_len=[{min(ep_lengths):.0f}|{avg_ep_length:.1f}|{max(ep_lengths):.0f}]  "
                f"true_rew=[{min(true_rewards):.2f}|{avg_true_reward:.2f}|{max(true_rewards):.2f}]"
                f"  [{t_collect:.1f}s]"
            )

            # ---- 2) Fragment trajectories into segment pairs ----------------
            t0 = time.time()
            print(f"[2/5] Fragmenting  (→ {num_pairs} pairs requested) …")
            segment_pairs = self.fragmenter.fragment(
                trajectories=trajectories, num_pairs=num_pairs
            )
            t_frag = time.time() - t0
            print(
                f"      pairs={len(segment_pairs)}  "
                f"(oversampled {n_transitions} transitions → top by variance)"
                f"  [{t_frag:.1f}s]"
            )

            # ---- 3) Gather preferences via SyntheticGatherer ---------------
            # Mirrors: preferences = self.preference_gatherer(fragments)
            t0 = time.time()
            print("[3/5] Gathering preferences …")
            preferences_array, pref_entropy = self.preference_gatherer(segment_pairs)
            t_pref = time.time() - t0
            debug_accuracy = self._compute_preference_accuracy(
                segment_pairs, preferences_array
            )
            print(
                f"      entropy={pref_entropy:.3f}  "
                f"label_acc={debug_accuracy:.3f}"
                f"  [{t_pref:.2f}s]"
            )

            # Convert float32 array → List[Preference] for our dataset/trainer.
            preferences = [
                Preference((float(p), float(1.0 - p))) for p in preferences_array
            ]
            self.dataset.push(segment_pairs, preferences)

            # ---- 4) Train reward model -------------------------------------
            # Epoch multiplier: warm-up iteration trains longer so the model is
            # calibrated before the agent begins learning on it.
            epoch_multiplier = self.initial_epoch_multiplier if is_warmup else 1.0
            num_rm_epochs = max(1, round(self.reward_trainer.epochs * epoch_multiplier))
            t0 = time.time()
            print(
                f"[4/5] Training reward model  "
                f"(epochs={num_rm_epochs}, ×{epoch_multiplier:.0f})  "
                f"dataset={len(self.dataset)} …"
            )
            rm_metrics = self.reward_trainer.train(
                self.dataset, epoch_multiplier=epoch_multiplier
            )
            t_rm = time.time() - t0
            # Reset reward-normalisation stats so the agent wrapper adapts to
            # the updated reward scale (Christiano et al. §2.2).
            self.agent_trainer.reset_reward_stats()

            # rm_metrics = {"loss": ..., "accuracy": ...} from the last epoch.
            # Use directly instead of calling evaluate() again (saves one pass).
            reward_loss     = rm_metrics["loss"]
            reward_accuracy = rm_metrics["accuracy"]
            acc_current = self._compute_preference_accuracy(
                segment_pairs, preferences_array
            )
            print(
                f"      loss={reward_loss:.4f}  "
                f"accuracy_batch={reward_accuracy:.3f}  "
                f"accuracy_current={acc_current:.3f}"
                f"  [{t_rm:.1f}s]"
            )

            # ---- 5) Train agent --------------------------------------------
            # Mirrors: self.trajectory_generator.train(steps=num_steps)
            # Extra remainder timesteps added at iteration num_iterations − 1
            # (penultimate loop step — exact imitation behaviour).
            num_agent_steps = timesteps_per_iteration
            if i == self.num_iterations - 1:
                num_agent_steps += extra_timesteps

            t0 = time.time()
            print(f"[5/5] Training agent  ({num_agent_steps:,} steps) …")
            self.agent_trainer.train(steps=num_agent_steps)
            t_agent = time.time() - t0
            agent_global_timesteps += num_agent_steps
            print(
                f"      total_agent_steps={agent_global_timesteps:,}"
                f"  [{t_agent:.1f}s]"
            )

            # ---- Checkpoint ------------------------------------------------
            if avg_true_reward > best_avg_true_reward:
                best_avg_true_reward = avg_true_reward
                if checkpoint_dir is not None:
                    _save_best_checkpoint(
                        agent=self.agent_trainer.agent,
                        checkpoint_dir=checkpoint_dir,
                        avg_true_reward=avg_true_reward,
                        iteration=i,
                    )
                self.logger.record("agent/best_avg_true_reward", best_avg_true_reward)

            # ---- Summary line ----------------------------------------------
            avg_model_reward = self._compute_avg_model_reward(trajectories)
            t_total = time.time() - t_iter_start
            print(
                f"  ↳ true_rew={avg_true_reward:.3f}  "
                f"model_rew={avg_model_reward:.3f}  "
                f"best_true_rew={best_avg_true_reward:.3f}  "
                f"[iter {t_total:.1f}s total]"
            )

            # ---- WandB logging ---------------------------------------------
            self.logger.record("iterations",                          self._iteration + 1)
            self.logger.record("rollout/num_pairs",                   num_pairs)
            self.logger.record("rollout/n_episodes",                  n_episodes)
            self.logger.record("rollout/n_transitions",               n_transitions)
            self.logger.record("rollout/avg_true_reward",             avg_true_reward)
            self.logger.record("rollout/min_true_reward",             float(min(true_rewards)))
            self.logger.record("rollout/max_true_reward",             float(max(true_rewards)))
            self.logger.record("rollout/avg_model_reward",            avg_model_reward)
            self.logger.record("rollout/avg_ep_length",               avg_ep_length)
            self.logger.record("rollout/preference_entropy",          pref_entropy)

            self.logger.record("reward_model/epoch_multiplier",       epoch_multiplier)
            self.logger.record("reward_model/num_epochs",             num_rm_epochs)
            self.logger.record("reward_model/loss_train",             reward_loss)
            self.logger.record("reward_model/accuracy_batch",         reward_accuracy)
            self.logger.record("reward_model/accuracy_current",       acc_current)
            self.logger.record("reward_model/debug_accuracy",         debug_accuracy)
            self.logger.record("reward_model/dataset_size",           len(self.dataset))

            self.logger.record("agent/time/total_timesteps",          agent_global_timesteps)

            self.logger.dump()

            # ---- Callback (mirrors imitation's callback(self._iteration)) --
            if callback:
                callback(self._iteration)
            self._iteration += 1

        return {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}

    # -----------------------------------------------------------------------
    # Preference helpers
    # -----------------------------------------------------------------------

    def _compute_preference_accuracy(
        self,
        segment_pairs: List[SegmentPair],
        preferences,
    ) -> float:
        """Fraction of pairs where the reward model agrees with the labels.

        Works with both a ``List[Preference]`` and a ``np.ndarray`` of float32
        probabilities (probability that ``seg1`` is preferred).
        """
        if not len(preferences):
            return 0.0

        def _label(p) -> float:
            """Normalise to a float probability that seg1 is preferred."""
            if isinstance(p, Preference):
                return float(p.label[0])
            return float(p)

        correct = sum(
            1
            for pair, pref in zip(segment_pairs, preferences)
            if (
                self.preference_model.preference_probs(
                    pair.seg1, pair.seg2
                ).label[0]
                > 0.5
            )
            == (_label(pref) > 0.5)
        )
        return correct / len(preferences)

    def _compute_avg_model_reward(self, trajectories: List[Trajectory]) -> float:
        """Mean per-episode model reward across trajectories."""
        ep_rewards = [
            float(
                self.reward_model.predict(
                    np.array([t.obs    for t in traj.transitions]),
                    np.array([t.action for t in traj.transitions]),
                ).sum()
            )
            for traj in trajectories
        ]
        return float(np.mean(ep_rewards))