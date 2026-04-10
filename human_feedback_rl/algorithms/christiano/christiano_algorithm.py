"""Christiano et al. 2017 — reward learning via preference comparisons.

Uses imitation's PreferenceComparisons directly.  All sumo-specific logic is
confined to MlpRewardNet; the rest of the pipeline is standard imitation.

Reward normalization
--------------------
imitation's RewardVecEnvWrapper does NOT normalize rewards.  Without
normalization the reward scale from the ensemble can vary arbitrarily between
iterations, making PPO's value function hard to fit and the policy gradient too
noisy (symptom: action std stays near its initial value of 1.0).

We therefore wrap the ensemble with NormalizedRewardNet(RunningNorm) for the
RL path only.  The preference training path keeps the raw ensemble so that
the BCE loss is computed on unnormalized reward magnitudes (as intended by
Christiano et al.).  The two paths diverge at construction time:

    RL agent     : NormalizedRewardNet(ensemble [+ AddSTDRewardWrapper])
    Pref training: PreferenceModel(ensemble)   — raw, via EnsembleTrainer
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv

from imitation.algorithms.preference_comparisons import (
    AgentTrainer,
    CrossEntropyRewardLoss,
    EnsembleTrainer,
    PreferenceComparisons,
    PreferenceModel,
    RandomFragmenter,
    SyntheticGatherer,
    QUERY_SCHEDULES,
)
from imitation.rewards.reward_nets import (
    AddSTDRewardWrapper,
    NormalizedRewardNet,
    RewardEnsemble,
    RewardNet,
)
from imitation.util import networks as imit_networks


# ---------------------------------------------------------------------------
# Re-export for backward-compat with christiano/__init__.py
# ---------------------------------------------------------------------------
__all__ = [
    "ChristianoAlgorithm",
    "MlpRewardNet",
    "SyntheticGatherer",
    "QUERY_SCHEDULES",
]


# ---------------------------------------------------------------------------
# Reward network
# ---------------------------------------------------------------------------

class MlpRewardNet(RewardNet):
    """MLP reward network for sumo-rl-ego.

    Predicts a scalar reward from (obs_t, action_t).
    next_state and done are accepted to satisfy the RewardNet interface but
    are intentionally ignored (as in Christiano et al. 2017, where reward
    is a function of the current transition only).

    Architecture mirrors the custom EnsembleRewardModel that was previously
    used: two hidden layers with ReLU activations and dropout.

    Args:
        observation_space: gym observation space of the environment.
        action_space: gym action space of the environment.
        hidden_dim: number of units in each hidden layer.
        dropout: dropout probability applied after each hidden ReLU.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        # normalize_images=False: sumo obs are float vectors, not images.
        super().__init__(observation_space, action_space, normalize_images=False)

        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-transition rewards.

        Args:
            state: (B, obs_dim) float tensor — current observation.
            action: (B, act_dim) float tensor — action taken.
            next_state: ignored (kept for interface compatibility).
            done: ignored (kept for interface compatibility).

        Returns:
            rewards: (B,) float tensor.
        """
        x = torch.cat(
            [state.float().flatten(start_dim=1),
             action.float().flatten(start_dim=1)],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

class ChristianoAlgorithm:
    """Reward learning via preference comparisons (Christiano et al. 2017).

    Thin wrapper that assembles all components and delegates entirely to
    imitation's ``PreferenceComparisons``.  The wrapper exists so that
    ``train.py`` can instantiate it with a flat kwargs dict from the
    Hydra config without knowing about imitation internals.

    Component mapping to imitation:
        reward model       →  MlpRewardNet × n_ensembles → RewardEnsemble
        pref model/trainer →  PreferenceModel(ensemble) + EnsembleTrainer
                              (bagging, one BasicRewardTrainer per member)
                              trained on RAW (unnormalized) ensemble outputs
        RL reward          →  NormalizedRewardNet(RunningNorm)(ensemble [+STD])
                              normalized running mean/std → passed to AgentTrainer
        preference gatherer→  SyntheticGatherer (ground-truth env rewards)
        fragmenter         →  RandomFragmenter
        trajectory gen.    →  AgentTrainer (wraps SB3 agent)
        outer loop         →  PreferenceComparisons.train()

    Args:
        env: vectorized environment (VecEnv).
        agent: SB3 algorithm instance (PPO, SAC, …) already constructed.
        rng: numpy random generator used to seed all stochastic components.
        n_ensembles: number of reward network members in the ensemble.
        hidden_dim: hidden layer width for each MlpRewardNet.
        lr_reward_model: Adam learning rate for reward model training.
        device: torch device string ("cpu" or "cuda").
        fragment_length: number of timesteps in each trajectory fragment.
        comparison_queue_size: FIFO buffer size for preference dataset
            (None = unlimited).
        transition_oversampling: factor by which to over-collect transitions
            before fragmenting (>1 reduces fragment overlap).
        initial_comparison_frac: fraction of total_comparisons gathered
            before any agent training (reward model warm-up).
        initial_epoch_multiplier: reward trainer epoch multiplier on
            iteration 0 (warm-up phase).
        query_schedule: comparison budget schedule across iterations.
            One of "constant", "hyperbolic", "inverse_quadratic".
        num_iterations: number of (reward-train → agent-train) cycles.
        reward_trainer_epochs: base epochs per reward trainer call.
        reward_model_batch_size: batch size for reward trainer.
        preference_temperature: SyntheticGatherer softmax temperature.
            0 = deterministic argmax; 1 = standard Boltzmann sampling.
        preference_sample: if True, sample hard 0/1 labels from the
            Bernoulli; if False, return raw probabilities.
        preference_discount_factor: discount applied when summing rewards
            over a fragment to compute preference probability.
        preference_threshold: logit-diff clipping in SyntheticGatherer.
        pessimism: if > 0, the reward seen by the RL agent is
            mean − pessimism × std  (AddSTDRewardWrapper with alpha<0),
            penalising regions where ensemble members disagree.
            The reward model is still trained on raw member predictions.
    """

    def __init__(
        self,
        env: VecEnv,
        agent,
        rng: np.random.Generator,
        # reward model
        n_ensembles: int = 3,
        hidden_dim: int = 256,
        lr_reward_model: float = 3e-4,
        device: str = "cpu",
        # fragments & query schedule
        fragment_length: int = 10,
        comparison_queue_size: Optional[int] = None,
        transition_oversampling: float = 1.0,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        query_schedule: Union[str] = "hyperbolic",
        num_iterations: int = 20,
        # reward trainer
        reward_trainer_epochs: int = 4,
        reward_model_batch_size: int = 64,
        # synthetic gatherer
        preference_temperature: float = 1.0,
        preference_sample: bool = True,
        preference_discount_factor: float = 1.0,
        preference_threshold: float = 50.0,
        # pessimism / OOD penalty
        pessimism: float = 0.0,
    ) -> None:
        obs_space = env.observation_space
        act_space = env.action_space

        # ------------------------------------------------------------------
        # 1. Build reward ensemble  (used for preference training — raw)
        # ------------------------------------------------------------------
        members = [
            MlpRewardNet(obs_space, act_space, hidden_dim=hidden_dim).to(device)
            for _ in range(n_ensembles)
        ]
        ensemble = RewardEnsemble(obs_space, act_space, members)

        # ------------------------------------------------------------------
        # 2. Preference model & loss  (trained on raw ensemble)
        # ------------------------------------------------------------------
        # PreferenceModel receives the raw ensemble directly.  This is
        # deliberate: the BCE loss must see unnormalized reward magnitudes so
        # that the preference probability is calibrated on the original scale.
        # (NormalizedRewardNet is applied only to the RL path below.)
        preference_model = PreferenceModel(ensemble)
        loss = CrossEntropyRewardLoss()

        # ------------------------------------------------------------------
        # 3. Reward trainer  (EnsembleTrainer — bagging per member)
        # ------------------------------------------------------------------
        reward_trainer = EnsembleTrainer(
            preference_model=preference_model,
            loss=loss,
            rng=rng,
            epochs=reward_trainer_epochs,
            batch_size=reward_model_batch_size,
            lr=lr_reward_model,
        )

        # ------------------------------------------------------------------
        # 4. RL reward net  (normalized — used only by the RL agent)
        # ------------------------------------------------------------------
        # Christiano et al. 2017, Section 2.2: rewards are normalized to
        # approximately unit variance before being passed to the RL agent.
        # imitation's RewardVecEnvWrapper does NOT normalize; we therefore
        # wrap explicitly with NormalizedRewardNet(RunningNorm).
        #
        # Optionally prepend AddSTDRewardWrapper for pessimism-based OOD
        # penalization: reward seen by agent = mean − pessimism × std.
        #
        # Training path:  PreferenceModel(ensemble)  — raw, no normalization
        # RL path:        NormalizedRewardNet(ensemble [+STD])  — normalized
        if pessimism > 0.0:
            _rl_base: RewardNet = AddSTDRewardWrapper(
                ensemble, default_alpha=-pessimism
            )
        else:
            _rl_base = ensemble

        rl_reward_net = NormalizedRewardNet(
            _rl_base, normalize_output_layer=imit_networks.RunningNorm
        )

        # ------------------------------------------------------------------
        # 5. Preference gatherer  (synthetic, using ground-truth env rewards)
        # ------------------------------------------------------------------
        gatherer = SyntheticGatherer(
            temperature=preference_temperature,
            discount_factor=preference_discount_factor,
            sample=preference_sample,
            rng=rng,
            threshold=preference_threshold,
        )

        # ------------------------------------------------------------------
        # 6. Fragmenter  (uniform random sampling)
        # ------------------------------------------------------------------
        fragmenter = RandomFragmenter(rng=rng)

        # ------------------------------------------------------------------
        # 7. Trajectory generator  (imitation's AgentTrainer)
        # ------------------------------------------------------------------
        # AgentTrainer wraps the SB3 agent with:
        #   BufferingWrapper    — records trajectories with ORIGINAL env rewards
        #   RewardVecEnvWrapper — replaces env rewards with rl_reward_net.predict_processed
        #                         (= normalized ensemble mean, or mean - pessimism*std)
        # The BufferingWrapper sits inside the reward wrapper so that the
        # trajectories passed to SyntheticGatherer carry the ground-truth rewards.
        trajectory_generator = AgentTrainer(
            algorithm=agent,
            reward_fn=rl_reward_net,  # normalized; converted to predict_processed inside
            venv=env,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # 8. PreferenceComparisons  (imitation's main training loop)
        # ------------------------------------------------------------------
        # We supply all three optional components (fragmenter, gatherer,
        # reward_trainer) so we must NOT pass rng to PreferenceComparisons
        # (imitation raises ValueError if both rng and all three are given).
        #
        # reward_model is stored as self.model but never used when a custom
        # reward_trainer is provided — passing rl_reward_net is fine here.
        self._pc = PreferenceComparisons(
            trajectory_generator=trajectory_generator,
            reward_model=rl_reward_net,
            num_iterations=num_iterations,
            fragmenter=fragmenter,
            preference_gatherer=gatherer,
            reward_trainer=reward_trainer,
            comparison_queue_size=comparison_queue_size,
            fragment_length=fragment_length,
            transition_oversampling=transition_oversampling,
            initial_comparison_frac=initial_comparison_frac,
            initial_epoch_multiplier=initial_epoch_multiplier,
            query_schedule=query_schedule,
            # sumo episodes vary in length (crash / arrive / timeout);
            # allow_variable_horizon=True disables the fixed-horizon check.
            allow_variable_horizon=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        checkpoint_dir: Optional[str] = None,
    ) -> dict:
        """Run the full Christiano training loop.

        Args:
            total_timesteps: total environment steps allocated for agent
                training across all iterations.
            total_comparisons: total number of preference comparisons to
                collect (split across iterations by query_schedule).
            checkpoint_dir: unused; kept for API compatibility with train.py.

        Returns:
            dict with keys "reward_loss" and "reward_accuracy" (final
            iteration values from imitation's PreferenceComparisons.train).
        """
        return self._pc.train(
            total_timesteps=total_timesteps,
            total_comparisons=total_comparisons,
        )
