# human-feedback-rl

Reward learning from demonstrations and preferences, for simulated driving in
[SUMO](https://eclipse.dev/sumo/) through the `sumo-rl-ego` environment.

The package exposes **one algorithm**, `HybridAlgorithm`. It learns a reward
model from two feedback sources and trains an SB3 agent on it. The baselines
are not separate code: they are the same algorithm with one channel switched
off.

| configuration | what it is |
|---|---|
| both channels active | the hybrid method: demonstration IRL loss and Bradley-Terry preference loss on one shared reward net |
| `demo_weight=0` | preference-only baseline (Christiano-style; soft or sampled binary labels) |
| `total_queries=0` | demonstration-only baseline (`demo_1`, difference of means, or `demo_2`, MaxEnt surrogate) |
| `demo_mode="preferences"` | literature hybrid (Ibarz et al. 2018): demonstrations as implicit preferences |

The library imports and tests **without SUMO**: the environment interface is
any SB3 `VecEnv` reporting `info["ego_status"]` as a string (see
`common/status.py`).

## Combining the two gradients

`gcl_fusion` decides how the demonstration and preference gradients become one
update:

`alpha_norm_single_adam`
: each gradient is reduced to its direction and the two are mixed by a
  reliability weight, `g = (1-α)·ĝ_pref + α·ĝ_demo`, then handed to a single
  Adam. The weight is estimated every iteration from how much each channel
  scatters: `α = CV²_pref / (CV²_pref + CV²_demo)`. Noisy comparisons push
  weight onto the demonstrations; as they become informative it shifts back.
  Below five comparisons the preference dispersion is not estimable and α stays
  pinned to 1. See `algorithms/hybrid/alpha_estimation.py`.

`norm_balance`
: the demonstration gradient is rescaled to `demo_weight` times the preference
  gradient norm and added, `g = g_pref + s·g_demo`. It is what the
  single-channel baselines use, and the ablation the reliability weight is
  measured against.

## Package layout

Algorithm code lives in `algorithms/`, shared infrastructure in `common/`. The
entry point, the Hydra configuration and the analysis live in the **parent
repository**.

`HybridAlgorithm` is assembled from mixins, one file each. The class itself
keeps only what holds the run together: the constructor, the RNG streams and
the training loop.

```
human_feedback_rl/
├── algorithms/
│   ├── hybrid_algorithm.py       # HybridAlgorithm: constructor and training loop
│   ├── hybrid_algorithm_pseudocode.md
│   └── hybrid/
│       ├── feedback_collection.py   # asking the oracle, counting what comes back
│       ├── reward_model_training.py # fitting the reward to the feedback so far
│       ├── gradient_fusion.py       # two gradients into one optimizer step
│       ├── reliability_weight.py    # estimating alpha once per iteration
│       ├── alpha_estimation.py      # the maths behind alpha
│       ├── demonstration_losses.py  # demo_1 / demo_2, and batch sampling
│       ├── reward_training.py       # gradient norms, reward normalization
│       ├── imitation_metrics.py     # agent-versus-expert error
│       ├── reward_diagnostics.py    # composes the five files below
│       ├── loss_diagnostics.py      #   the losses while they are optimised
│       ├── reward_validation.py     #   ranking and outcome separation
│       ├── return_scatter.py        #   predicted return against true return
│       ├── outcome_returns.py       #   returns by how the episode ended
│       └── replay_staleness.py      #   drift in the replay buffer
└── common/
    ├── base_algorithm.py         # env + agent + logger + rng
    ├── base_reward_learning_algorithm.py  # reward model, rollouts,
    │                             #   query schedule, checkpointing
    ├── status.py                 # single source of truth for the 7 ego statuses
    ├── types.py                  # Transition, Trajectory, FragmentPair, Preference
    ├── preference_losses.py      # Bradley-Terry probs / NLL / accuracy
    ├── reward_nets.py            # SumoRewardNet, RewardEnsemble, NormalizedRewardNet
    ├── fragmenters.py            # random / active (ensemble-disagreement) sampling
    ├── gatherers.py              # synthetic preference oracle
    ├── datasets.py               # circular PreferenceDataset with bootstrap()
    ├── demo_subsampling.py       # same budget -> same demonstrations
    ├── env_wrappers.py           # predicted reward, trajectory buffering,
    │                             #   epsilon-exploration policy
    ├── trajectory_generators.py  # rollout + SB3 training on predicted rewards
    ├── replay_buffers.py         # reward relabelling and staleness
    ├── loggers.py                # SB3 logger extensions, W&B and JSONL
    ├── batching.py               # differentiable fragment reward sums
    └── custom_logging_callback.py
```

> **Checkpoint compatibility:** saved checkpoints pickle objects from
> `common/datasets.py`, `common/types.py` and `common/replay_buffers.py` —
> those module paths must not move.

## Conventions worth knowing

- **Ego status** is a 7-way one-hot `[arrived, collided, offroad, timeout,
  running, teleported, removed_unknown]`; every index used anywhere comes from
  `common/status.py`.
- **Reward normalization** is agent-only: reward-model training and evaluation
  always use the raw `forward()`; the agent consumes `predict()`, which applies
  the persistent `(x - mean) / std` transform fitted on recent rollouts.
- **Trajectories** are recorded by `EnvBufferingWrapper` during both rollout and
  SB3 training, and popped by the algorithm each iteration.
- **Three RNG streams** — query, oracle, train — are spawned from the master
  seed. Sharing one made the feedback depend on how many gradient steps the
  reward model took, which put the optimization hyperparameters inside the
  comparison. The alpha estimate draws from a fourth generator on a fixed seed,
  so measuring never moves the training draws.
- **Oracle versus learner**: `pref_temperature` and `preference_fragment_length`
  describe the synthetic annotator. They define the problem, not the learner.
- **All feedback is trained on**: there is no held-out preference split.

## Installation

```bash
# from the parent repository root
pip install -e sumo-rl-ego/
pip install -e 'human-feedback-rl/[dev]'
```

Every constructor parameter is exposed as `algo.kwargs.<name>`, and every
`train()` parameter as `train.kwargs.<name>`, in the parent repository's Hydra
configuration.

## Tests

```bash
python -m pytest tests/
```

The suite runs without SUMO: `tests/conftest.py` provides a fake SB3 `VecEnv`
emitting `info["ego_status"]`.

## Metrics

All metrics go to W&B through `common/loggers.py`. `iterations` is the x-axis
for reward-model and rollout metrics, `agent/time/total_timesteps` for agent
metrics; `configure_wandb_metrics` keeps the auto-generated workspace small
(see `VISIBLE_METRICS`).

Two things are worth watching. `rollout/mean_model_reward` rising while
`rollout/mean_true_reward` stalls means the policy is exploiting the reward
model. And with soft labels at a high `pref_temperature` the BT loss sits at
its ln(2) floor even when learning is going well — read
`reward/acc_pref_train` and `reward_val/.../pred_true/pearson_*` instead.

`alpha/S_pref` and `alpha/S_demo` are the sanity check on the reliability
weight: they are the variance of the gradient actually applied, so they should
fall as the feedback budget grows.
