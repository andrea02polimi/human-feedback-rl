# human-feedback-rl

Implementation of the **Christiano et al. (2017) RLHF pipeline** applied to simulated autonomous driving in [SUMO](https://eclipse.dev/sumo/) via the `sumo-rl-ego` environment.

An A2C policy is trained entirely on rewards predicted by a learned reward model, which is itself trained from human (or synthetic expert) preferences over short trajectory segments.

---

## Repository structure

This package is designed as a library modelled after [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3): algorithm-agnostic code lives in `common/`, algorithm-specific code lives in `algorithms/<name>/`. Scripts and configs live in the **parent repository** (`sumo-human-feedback-rl/`).

```
human-feedback-rl/
├── human_feedback_rl/
│   ├── algorithms/
│   │   ├── base_trainer.py              # BaseTrainer ABC
│   │   └── christiano/
│   │       ├── christiano.py            # ChristianoRLHF — SB3-like class, train(output_dir)
│   │       └── policy_worker.py         # _policy_worker (SB3 A2C, Christiano-specific)
│   └── common/
│       ├── workers.py                   # _preference_worker, _demonstration_worker,
│       │                                #   _demo_preference_worker (algorithm-agnostic)
│       ├── segment.py                   # Segment dataclass
│       ├── pref_db.py                   # PrefDB (circular DB) + PrefBuffer (async recv thread)
│       ├── demo_db.py                   # DemoDatabase (circular DB for margin ranking loss)
│       ├── sampling.py                  # Random and disagreement-based pair selection
│       ├── preference_collector.py      # PreferenceCollector — segment buffer + pair sampling
│       ├── demonstration_collector.py   # DemonstrationCollector — expert segment pairing
│       ├── callbacks.py                 # SegmentCollectorCallback (SB3 training callback)
│       ├── wrappers.py                  # PredictedRewardVecWrapper (replaces env rewards)
│       ├── oracles/
│       │   ├── base.py                  # BaseOracle ABC
│       │   ├── expert.py                # ExpertOracle (env_reward | qnet, hard | soft labels)
│       │   ├── human.py                 # HumanOracle (terminal + pyglet visualisation)
│       │   └── factory.py              # build_oracle(config) factory
│       ├── reward_predictor/
│       │   ├── networks.py              # SumoRewardNetwork (MLP backbone)
│       │   └── ensemble.py             # RewardPredictorEnsemble (training + inference + wandb)
│       └── utils/
│           ├── env_setup.py             # build_env_and_expert, build_single_env, build_demo_env_and_expert
│           ├── running_stats.py         # RunningStat (Welford's online mean/std)
│           ├── itertools.py             # batch_iter
│           └── checkpoints.py          # keep_latest_checkpoints, drain_demo_pipe
├── pyproject.toml
├── requirements.txt
└── README.md
```

Scripts and configs (entry points, Hydra configs, experiment launcher) are in the parent repo:

```
sumo-human-feedback-rl/
├── scripts/
│   ├── train.py          # Hydra entry point — instantiates ChristianoRLHF and calls train()
│   ├── eval.py           # Headless evaluation with matplotlib plots
│   └── play.py           # SUMO GUI visualisation
├── configs/
│   └── train.yaml        # Full flat config (all ChristianoRLHF params + wandb + hydra)
└── run_experiments.sh    # Batch launcher for remote server (taskset + numactl)
```

---

## Algorithm

The pipeline runs three concurrent processes:

1. **Policy process** — Generates trajectory segments with a random policy (Phase 1), then trains an SB3 A2C agent using rewards predicted by the reward model (Phase 2+). Sends segments to the preference worker.
2. **Preference process** — Samples segment pairs, labels them via an oracle, and sends labeled pairs to the main process via `preference_pipe`.
3. **Main process** — Maintains `PrefDB` train/val databases, retrains the reward predictor ensemble when enough new preferences have arrived, saves checkpoints, and logs metrics to wandb.

When `use_demonstrations=True`, a fourth **demonstration process** runs the expert DQN, pairs its segments with agent segments, and feeds them into `DemoDatabase` for an additional margin ranking loss.

When `use_demo_preferences=True`, a fourth **demo-preference process** runs the expert DQN, builds expert-correction segments, and injects `(expert_frames, agent_frames, (1.0, 0.0))` directly into `preference_pipe` — expert corrections treated as hard preferences, no separate loss term needed.

### Reward predictor loss

```
use_demonstrations=True:
  L_total = L_pref + demo_weight × L_demo

  L_pref  — soft cross-entropy on (seg1, seg2, p1, p2) preference pairs
  L_demo  — margin ranking loss: mean( relu( margin − (Σr_expert − Σr_agent) ) )

use_demo_preferences=True:
  L_total = L_pref  (expert corrections enter as (1.0, 0.0) preference pairs)
```

### Preference labeling modes (`label_mode`)

| `label_mode` | Label format | Description |
|---|---|---|
| `hard` | `(1,0)`, `(0,1)`, `(0.5,0.5)` | Original Christiano et al. — discrete preference |
| `soft` | `(p, 1−p)` with `p = softmax(score)` | Preserves oracle confidence as soft label |

### Oracles

| `oracle` value | Description |
|---|---|
| `env_reward` | Prefers the segment with higher sum of true environment rewards |
| `qnet` | Prefers the segment with higher sum of V(s) = max_a Q(s,a) from the expert DQN |
| `human` | Interactive: shows segments in a terminal/pyglet window and prompts for human input |

---

## Installation

```bash
# From the parent repo root (sumo-human-feedback-rl/)
pip install -e sumo-rl-ego/
pip install -e human-feedback-rl/
```

> `sumo-rl-ego` must be installed first as `human-feedback-rl` depends on it.

---

## Training

All training commands are run from the **parent repo root** (`sumo-human-feedback-rl/`).

```bash
# Default: env_reward oracle, hard labels
python scripts/train.py

# Soft labels
python scripts/train.py christiano.label_mode=soft

# Expert Q-net oracle
python scripts/train.py christiano.oracle=qnet

# Demo preferences (expert corrections injected as hard preference pairs)
python scripts/train.py \
    christiano.use_demo_preferences=true \
    christiano.db_train_maxlen=6000 \
    christiano.db_val_maxlen=1500

# Expert demo margin ranking loss (original demo path)
python scripts/train.py christiano.use_demonstrations=true

# Soft labels + wandb tags
python scripts/train.py christiano.label_mode=soft "wandb.tags=[soft,env_reward]"
```

Hydra saves the run output to `output/christiano/YYYY-MM-DD_HH-MM-SS/`:

```
output/christiano/2026-03-18_14-30-00/
├── config.yaml                      # Full resolved config (reused by eval/play)
├── models/policy_christiano.zip     # Final A2C policy (SB3 format)
└── reward_predictor_checkpoints/    # RP .pt checkpoints (latest 2 kept)
```

### Key config parameters

All parameters can be overridden on the command line as `christiano.<param>=<value>`.

| Parameter | Default | Description |
|---|---|---|
| `christiano.oracle` | `env_reward` | Preference oracle |
| `christiano.label_mode` | `hard` | `hard` or `soft` preference labels |
| `christiano.use_demonstrations` | `false` | Enable expert demo margin ranking loss |
| `christiano.use_demo_preferences` | `false` | Inject expert corrections as preference pairs |
| `christiano.n_reward_predictors` | `3` | Ensemble size |
| `christiano.initial_prefs` | `500` | Preferences collected before RP pretraining |
| `christiano.segment_len` | `25` | Frames per segment |
| `christiano.db_train_maxlen` | `3000` | Circular buffer size for train PrefDB |
| `christiano.db_val_maxlen` | `750` | Circular buffer size for val PrefDB |
| `christiano.total_env_steps` | `1000000` | Total A2C environment steps (Phase 2+) |
| `christiano.rp_val_interval` | `50` | Validate RP every N gradient steps; also controls Phase 3 retrain cadence |
| `christiano.torch_num_threads` | `2` | Max PyTorch intra-op threads per process |
| `expert_model` | `highway_discrete_dqn_v2_1` | Expert DQN directory (relative to parent repo root) |

---

## Evaluation

```bash
python scripts/eval.py run_dir=output/christiano/2026-03-18_14-30-00

# Override number of episodes
python scripts/eval.py run_dir=output/christiano/... eval.episodes=200
```

Generates matplotlib plots (reward per episode, episode length, aggregate bar plot) saved to `<run_dir>/eval_plots/`.

---

## GUI playback

Requires a SUMO installation with `traci`.

```bash
python scripts/play.py run_dir=output/christiano/2026-03-18_14-30-00

# Optional overrides
python scripts/play.py run_dir=... eval.episodes=5 play.step_delay=0.05
```

---

## Batch experiments (remote server)

```bash
bash run_experiments.sh
```

Launches 3 experiments in background, pinned to CPU cores 18-26 via `taskset` and `numactl`, with all output to `/storage/fis3`.

---

## Weights & Biases

All metrics are logged to [wandb](https://wandb.ai). Configure in `configs/train.yaml` or override on the command line:

```bash
python scripts/train.py wandb.project=my-project wandb.entity=my-org "wandb.tags=[exp1]"
```

### Tracked metrics

| Key | Description |
|---|---|
| `policy/mean_predicted_rew` | Mean predicted reward over the last rollout |
| `policy/std_predicted_rew` | Std of predicted rewards over the last rollout |
| `policy/mean_episode_avg_true_rew` | Mean true env reward per completed episode (reward hacking signal: should rise together with `mean_predicted_rew`) |
| `rp/train_loss` | Reward predictor MLE training loss |
| `rp/train_demo_margin_loss` | Margin ranking loss (`use_demonstrations` only) |
| `rp/val_loss` | Reward predictor validation loss |
| `rp/accuracy` | Reward predictor preference accuracy on validation set |
| `rp/mean_disagreement` | Mean std of per-segment rewards across ensemble members (high = uncertain RP) |
| `train_db_size` | Number of preferences in train PrefDB |
| `val_db_size` | Number of preferences in val PrefDB |
| `pref_db_size` | Total preferences (train + val) |
| `demo_db_size` | Number of demo pairs in DemoDatabase (`use_demonstrations` only) |

**Diagnosing reward hacking:** if `policy/mean_predicted_rew` rises but `policy/mean_episode_avg_true_rew` does not, the policy has found a reward model exploit. High `rp/mean_disagreement` indicates the reward model is uncertain and more preferences are needed.

---

## Performance notes (multi-core servers)

- `torch_num_threads` controls `torch.set_num_threads()` **and** `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` for all subprocesses. Set to `floor(n_assigned_cores / n_processes)`.
- On Linux, `forkserver` is used instead of `spawn` for faster subprocess startup without the deadlock risk of `fork` (wandb and PyTorch create threads before workers are spawned).
- Phase 3 RP retraining is data-driven: retrain only when `rp_val_interval` new preferences have arrived since the last retrain, preventing both busy-wait and redundant retraining on stale data.
