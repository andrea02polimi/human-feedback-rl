# human-feedback-rl

Implementation of the **Christiano et al. (2017) RLHF pipeline** applied to simulated autonomous driving in [SUMO](https://eclipse.dev/sumo/) via the `sumo-rl-ego` environment.

An A2C policy is trained entirely on rewards predicted by a learned reward model, which is itself trained from human (or synthetic expert) preferences over short trajectory segments.

---

## Repository structure

```
human-feedback-rl/
├── human_feedback_rl/
│   ├── algorithms/
│   │   ├── base_trainer.py          # BaseTrainer ABC
│   │   └── christiano/
│   │       ├── trainer.py           # ChristianoTrainer — main orchestration
│   │       └── workers.py           # Three async worker functions (policy, preference, demo)
│   ├── feedback/
│   │   ├── segment.py               # Segment dataclass
│   │   ├── pref_db.py               # PrefDB (circular DB) + PrefBuffer (async recv thread)
│   │   ├── sampling.py              # Random and disagreement-based pair selection
│   │   ├── collector.py             # PreferenceCollector — segment buffer + pair sampling
│   │   ├── demonstrations.py        # DemonstrationCollector — expert segment pairing
│   │   └── oracles/
│   │       ├── base.py              # BaseOracle ABC
│   │       ├── expert.py            # ExpertOracle (env_reward | qnet modes)
│   │       ├── human.py             # HumanOracle (terminal + pyglet visualisation)
│   │       └── factory.py           # build_oracle(config) factory
│   ├── reward_models/
│   │   ├── networks.py              # SumoRewardNetwork (MLP backbone)
│   │   └── ensemble.py              # RewardPredictorEnsemble (training + inference)
│   ├── policy/
│   │   ├── wrappers.py              # PredictedRewardVecWrapper (replaces env rewards)
│   │   └── callbacks.py             # SegmentCollectorCallback (SB3 training callback)
│   └── utils/
│       ├── env_setup.py             # build_env_and_expert, build_single_env
│       ├── running_stats.py         # RunningStat (Welford's online mean/std)
│       ├── itertools.py             # batch_iter
│       └── checkpoints.py           # keep_latest_checkpoints, drain_demo_pipe
├── configs/
│   ├── train.yaml                   # Base training config (Hydra)
│   ├── eval.yaml                    # Evaluation config
│   ├── play.yaml                    # GUI playback config
│   └── algorithm/
│       ├── christiano_env_reward.yaml   # Config 1: env reward oracle
│       ├── christiano_qnet.yaml         # Config 2: expert Q-net oracle
│       └── christiano_demo.yaml         # Config 3: env reward oracle + demonstrations
└── scripts/
    ├── train_christiano.py          # Training entry point
    ├── eval.py                      # Headless evaluation
    └── play.py                      # SUMO GUI visualisation
```

---

## Algorithm

The pipeline runs three concurrent activities:

1. **Policy process** — Generates trajectory segments with a random policy (Phase 1), then trains an SB3 A2C agent using rewards predicted by the reward model (Phase 2).
2. **Preference process** — Samples segment pairs from the policy process, labels them via an oracle, and sends labeled pairs to the main process.
3. **Main process** — Maintains the preference databases, trains the reward predictor ensemble, and saves checkpoints.

When demonstrations are enabled a fourth **demonstration process** runs the expert DQN in parallel, collects expert segments, pairs them with agent segments, and feeds them into a separate demonstration database used to compute an additional margin ranking loss.

### Reward predictor loss

```
L_total = L_pref + demo_weight × L_demo

L_pref  — soft cross-entropy on (seg1, seg2, p1, p2) preference pairs
          (soft labels preserve oracle confidence, e.g. 0.7/0.3)

L_demo  — margin ranking loss: mean( relu( margin − (Σr_expert − Σr_agent) ) )
          (gradient is zero once expert already beats agent by `margin`)
```

### Oracles

| `oracle` value | Description |
|---|---|
| `env_reward` | Prefers the segment with higher sum of true environment rewards |
| `qnet` | Prefers the segment with higher sum of V(s) = max_a Q(s,a) from the expert DQN |
| `human` | Interactive: shows segments in a pyglet window and prompts for human input |

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

All scripts are run from `human-feedback-rl/`.

### Three experimental configurations

```bash
# Config 1 — Christiano et al. original: preferences labeled by environment reward
python scripts/train_christiano.py

# Config 2 — Preferences labeled by expert Q-net value V(s) = max_a Q(s,a)
python scripts/train_christiano.py algorithm=christiano_qnet

# Config 3 — Preferences + expert demonstration pairs (margin ranking loss)
python scripts/train_christiano.py algorithm=christiano_demo
```

Hydra saves the run to `experiments/christiano/YYYY-MM-DD/HH-MM-SS/`:

```
experiments/christiano/2026-03-16/17-05-21/
├── config/config.yaml               # Full resolved config
├── models/policy_christiano.zip     # Final A2C policy (SB3 format)
├── reward_predictor_checkpoints/    # RP .pt checkpoints (latest 2 kept)
├── reward_predictor/                # TensorBoard logs for RP
│   ├── train/
│   └── val/
└── policy/                          # TensorBoard logs for A2C
```

### Key config parameters

| Parameter | Default | Description |
|---|---|---|
| `preferences.initial_prefs` | 500 | Preferences collected before RP pretraining |
| `preferences.segment_len` | 25 | Frames per segment |
| `preferences.oracle` | `env_reward` | Preference oracle (set by algorithm config) |
| `preferences.use_demonstrations` | `false` | Enable demonstration loss (set by algorithm config) |
| `reward_predictor.n_preds` | 3 | Ensemble size |
| `reward_predictor.demo_weight` | 1.0 | Weight of L_demo relative to L_pref |
| `reward_predictor.demo_margin` | 1.0 | Margin for ranking loss |
| `training.total_env_steps` | 100000 | Total A2C environment steps |
| `env.expert_model` | `highway_discrete_dqn_v2_1` | Expert model directory (relative to repo root) |

Override any parameter on the command line:

```bash
python scripts/train_christiano.py training.total_env_steps=500000 reward_predictor.n_preds=5
```

---

## Evaluation

```bash
python scripts/eval.py \
    run.dir=experiments/christiano/2026-03-16/17-05-21 \
    agent.model=experiments/christiano/2026-03-16/17-05-21/models/policy_christiano

# Override number of episodes
python scripts/eval.py run.dir=... agent.model=... eval.episodes=200
```

Prints per-episode metrics from the environment's `metrics_tracker` at the end.

---

## GUI playback

Requires a SUMO installation with `traci`.

```bash
python scripts/play.py \
    run.dir=experiments/christiano/2026-03-16/17-05-21 \
    agent.model=experiments/christiano/2026-03-16/17-05-21/models/policy_christiano

# Optional overrides
python scripts/play.py run.dir=... agent.model=... eval.episodes=5 play.step_delay=0.05
```

---

## Weights & Biases

All metrics are logged to [wandb](https://wandb.ai). Configure the project in `configs/train.yaml`:

```yaml
wandb:
  project: "sumo-rlhf"
  entity: null   # your wandb username/org, or null for default
  tags: []
```

Override on the command line:

```bash
python scripts/train_christiano.py wandb.project=my-project wandb.entity=my-org
```

Tracked metrics:

| Key | Description |
|---|---|
| `rollout/ep_rew_mean` | Mean episode reward (predicted rewards, from SB3) |
| `train/loss` | A2C training loss (from SB3) |
| `rp/train/loss` | Reward predictor training loss |
| `rp/train/demo_margin_loss` | Margin ranking loss (Config 3 only) |
| `rp/val/loss` | Reward predictor validation loss |
| `rp/val/accuracy` | Reward predictor validation accuracy |
| `prefs/train_db_size` | Number of preferences in the training database |
| `prefs/val_db_size` | Number of preferences in the validation database |
| `prefs/total_received` | Total labeled preferences received so far |
