# human-feedback-rl

Reward learning from human feedback for simulated autonomous driving in
[SUMO](https://eclipse.dev/sumo/) via the `sumo-rl-ego` environment.

The package exposes **one algorithm**, `HybridAlgorithm`, that learns a reward
model from two feedback sources and trains an SB3 agent on it:

| Configuration | Meaning |
|---|---|
| both sources active | the **hybrid** method: demonstration IRL loss + Bradley-Terry preference loss, fused on one shared reward net with norm-balanced gradients (`demo_weight`) |
| `demo_weight=0` | **preference-only** baseline (Christiano-style; soft or sampled binary labels) |
| `total_queries=0` | **demonstration-only** baseline (loss `demo_1` = difference of means, or `demo_2` = MaxEnt surrogate) |
| `demo_mode="preferences"` | literature hybrid baseline (Ibarz et al. 2018): demonstrations as implicit preferences |

The library is importable and testable **without SUMO**: the environment
interface is any SB3 `VecEnv` that reports `info["ego_status"]` as a string
(see `common/status.py`).

---

## Package layout

Algorithm-specific code lives in `algorithms/`, shared infrastructure in
`common/`. Scripts, Hydra configs and launchers live in the **parent
repository** (`sumo-human-feedback-rl/`).

```
human_feedback_rl/
├── algorithms/
│   ├── hybrid_algorithm.py       # HybridAlgorithm — the training loop (all arms)
│   ├── hybrid_algorithm_pseudocode.md
│   └── hybrid/                   # its building blocks (mixins)
│       ├── demonstration_losses.py  # demo_1 / demo_2 + sampling & dispatch
│       ├── reward_training.py       # optimization helpers + agent-reward normalization
│       ├── reward_diagnostics.py    # validation/ranking/replay-staleness logging
│       └── imitation_metrics.py     # agent-vs-expert RMSE and NLL
└── common/
    ├── base_algorithm.py         # env + agent + logger + rng
    ├── base_reward_learning_algorithm.py  # reward model, rollouts, validation,
    │                             #   query schedule, checkpointing (shared base)
    ├── status.py                 # single source of truth for the 7 ego statuses
    ├── types.py                  # Transition, Trajectory, FragmentPair, Preference
    ├── preference_losses.py      # Bradley-Terry probs / NLL / accuracy
    ├── reward_nets.py            # SumoRewardNet, RewardEnsemble, NormalizedRewardNet
    ├── fragmenters.py            # random / active (ensemble-disagreement) pair sampling
    ├── gatherers.py              # synthetic preference oracle (binary/soft/bernoulli)
    ├── datasets.py               # circular PreferenceDataset with bootstrap()
    ├── env_wrappers.py           # predicted-reward wrapper, trajectory buffering,
    │                             #   epsilon-exploration policy wrapper
    ├── trajectory_generators.py  # rollout + SB3 training on predicted rewards
    ├── replay_buffers.py         # reward relabelling + staleness diagnostics
    ├── loggers.py                # SB3 logger extensions + W&B/JSONL wiring
    ├── batching.py               # differentiable fragment reward sums
    └── custom_logging_callback.py
```

> **Checkpoint compatibility:** saved checkpoints pickle objects from
> `common/datasets.py`, `common/types.py` and `common/replay_buffers.py` —
> those module paths must not move.

## Key conventions

- **Ego status** is a 7-way one-hot `[arrived, collided, offroad, timeout,
  running, teleported, removed_unknown]`; every index used anywhere comes from
  `common/status.py`.
- **Reward normalization** is agent-only: reward-model training and evaluation
  always use the raw `forward()`; the agent consumes `predict()`, which applies
  the persistent `(x - mean) / std` transform fitted on recent rollouts.
- **Trajectories** are recorded by `EnvBufferingWrapper` during both rollout
  and SB3 training, and popped by the algorithm each iteration.
- **Oracle vs learner**: `pref_temperature` (label softness) and
  `preference_fragment_length` describe the synthetic annotator — they define
  the problem, not the learner, and are never tuned.

## Installation

```bash
# From the parent repo root (sumo-human-feedback-rl/)
pip install -e sumo-rl-ego/          # only needed to run the SUMO scripts
pip install -e 'human-feedback-rl/[dev]'
```

## Training (from the parent repo root)

```bash
python scripts/train_hybrid_sac.py           # every arm, via Hydra overrides
MODE=pref_only ./launchers/run_hybrid_sac.sh
```

All constructor parameters are exposed as `algo.kwargs.<name>` and all
`train()` parameters as `train.kwargs.<name>` in `configs/train_hybrid_sac.yaml`,
overridable on the command line (Hydra). The Optuna tuning campaign, budget
curves and final multi-seed runs are documented in `docs/tuning-server-guide.md`;
the analysis pipeline in `docs/analysis-pipeline-guide.md`; planned extensions
in `docs/extensions-roadmap.md`.

## Tests

```bash
cd human-feedback-rl
python -m pytest tests/
```

The suite runs without SUMO: `tests/conftest.py` provides a fake SB3 `VecEnv`
emitting `info["ego_status"]`.

## Metrics

All metrics go to W&B via `common/loggers.py`. `iterations` is the x-axis for
reward-model/rollout metrics, `agent/time/total_timesteps` for agent metrics;
`configure_wandb_metrics` keeps the auto-generated workspace small (see
`VISIBLE_METRICS`).

Reward-hacking check: `rollout/mean_model_reward` rising while
`rollout/mean_true_reward` stalls means the policy is exploiting the reward
model. `reward_val/*/spearman_returns` and the `reward_val/*` MAE/gap metrics
track how well the learned reward ranks and separates outcomes.
