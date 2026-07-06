# human-feedback-rl

Human-feedback reinforcement-learning algorithms for simulated autonomous driving in [SUMO](https://eclipse.dev/sumo/) via the `sumo-rl-ego` environment.

Three algorithm families share a common infrastructure:

| Algorithm | Feedback signal | Agent |
|---|---|---|
| `PreferenceAlgorithm` | Preferences over trajectory-fragment pairs (Christiano et al. 2017), Bradley-Terry ensemble reward model | SB3 PPO (or any on-policy SB3 algo) |
| `DemoAlgorithm` | Expert demonstration trajectories, MaxEnt-IRL-style reward learning (several loss variants) | SB3 SAC/PPO, optional reward relabelling replay buffer |
| `DaggerAlgorithm` | Interactive expert action labels (DAgger), behaviour cloning | `BCPolicy` (SB3 ActorCriticPolicy) |

The library is importable and testable **without SUMO**: the environment interface is any SB3 `VecEnv` that reports `info["ego_status"]` as a string (see `common/status.py`).

---

## Package layout

Algorithm-agnostic code lives in `common/`, algorithm-specific code in `algorithms/`. Scripts, Hydra configs and launchers live in the **parent repository** (`sumo-human-feedback-rl/`).

```
human_feedback_rl/
├── algorithms/
│   ├── preference_algorithm.py   # PreferenceAlgorithm — owns the query-schedule loop
│   ├── demo_algorithm.py         # DemoAlgorithm — owns the demo IRL loop
│   ├── dagger_algorithm.py       # DaggerAlgorithm — DAgger rounds + BC training
│   └── demo/                     # DemoAlgorithm mixins
│       ├── losses.py             # pure loss functions + sampling/dispatch mixin
│       ├── reward_training.py    # per-member optimization + agent-reward normalization
│       ├── reward_diagnostics.py # loss/ESS/ranking/replay-staleness logging
│       └── imitation_metrics.py  # agent-vs-expert RMSE and NLL
└── common/
    ├── base_algorithm.py         # env + agent + logger + rng
    ├── base_reward_learning_algorithm.py  # reward model, rollouts, validation,
    │                             #   query schedule, checkpointing (shared base)
    ├── status.py                 # single source of truth for the 7 ego statuses
    ├── types.py                  # Transition, Trajectory, FragmentPair, Preference
    ├── losses.py                 # Bradley-Terry probs / NLL / accuracy
    ├── reward_nets.py            # SumoRewardNet, RewardEnsemble, NormalizedRewardNet
    ├── fragmenters.py            # random / high-variance fragment-pair sampling
    ├── gatherers.py              # synthetic preference oracle (binary/soft/bernoulli)
    ├── datasets.py               # circular PreferenceDataset with bootstrap()
    ├── env_wrappers.py           # predicted-reward wrapper, trajectory buffering,
    │                             #   epsilon-exploration policy wrapper
    ├── trajectory_generators.py  # rollout + SB3 training on predicted rewards,
    │                             #   policy log-prob evaluation (PPO/SAC-aware)
    ├── replay_buffers.py         # reward relabelling + staleness diagnostics
    ├── loggers.py                # SB3 logger extensions + W&B wiring
    └── base_policy.py            # BCPolicy
```

## Key conventions

- **Ego status** is a 7-way one-hot `[arrived, collided, offroad, timeout, running, teleported, removed_unknown]`; every index used anywhere comes from `common/status.py`.
- **Reward normalization** is agent-only: reward-model training and evaluation always use the raw `forward()`; the agent consumes `predict()`, which applies the persistent `(x - mean) / std` transform fitted on recent rollouts. `NormalizedRewardNet` wraps the ensemble once (members are raw networks). Checkpoints from the older per-member-normalized layout still load.
- **Trajectories** are recorded by `EnvBufferingWrapper` during both rollout and SB3 training, and popped by the algorithms each iteration.

## Installation

```bash
# From the parent repo root (sumo-human-feedback-rl/)
pip install -e sumo-rl-ego/          # only needed to run the SUMO scripts
pip install -e 'human-feedback-rl/[dev]'
```

Runtime dependencies are declared in `pyproject.toml`. Hydra/OmegaConf/tqdm belong to the parent repo's scripts, not to this library.

## Training (from the parent repo root)

```bash
python scripts/test_chri_PPO.py          # preference-based (Christiano)
python scripts/test_demo_SAC.py          # demo IRL + SAC
python scripts/test_demo_PPO.py          # demo IRL + PPO
python scripts/test_dagger.py            # DAgger
```

All constructor parameters are exposed as `algo.kwargs.<name>` and all `train()` parameters as `train.kwargs.<name>` in the corresponding `configs/*.yaml`, overridable on the command line (Hydra). Batch launchers live in `launchers/`.

Notable `DemoAlgorithm` options: `loss_type` (`maxent`, `maxent_2`, `demo`, `demo_corrected`, `maxent_corrected` — the latter requires a dedicated `rollout_env`), `relabel_rewards` (recompute replay-buffer rewards with the current model at sampling time), `normalize_agent_reward`.

Notable `PreferenceAlgorithm` options: `labels_type` (`binary`, `soft`, `binary_bernoulli`), `fragmenter_type` (`random`, `active` = ensemble-disagreement), `query_schedule` (`constant`, `hyperbolic`, `inverse_quadratic`).

## Tests

```bash
cd human-feedback-rl
python -m pytest tests/
```

The suite runs without SUMO: `tests/conftest.py` provides a fake SB3 `VecEnv` emitting `info["ego_status"]`.

## Metrics

All metrics go to W&B via `common/loggers.py`. `iterations` is the x-axis for reward-model/rollout metrics, `agent/time/total_timesteps` for agent metrics; `configure_wandb_metrics` keeps the auto-generated workspace small (see `VISIBLE_METRICS`).

Reward-hacking check: `rollout/mean_model_reward` rising while `rollout/mean_true_reward` stalls means the policy is exploiting the reward model. `reward_val/*/spearman_returns` and the `reward_val/*` MAE/gap metrics track how well the learned reward ranks and separates outcomes.
