"""
Worker functions for the Christiano et al. RLHF pipeline.

These are top-level functions (not methods) because they are passed to
multiprocessing.Process(target=...) and must be picklable.
"""

import functools
import os
import sys

import numpy as np
from omegaconf import OmegaConf

from human_feedback_rl.reward_models.ensemble import RewardPredictorEnsemble
from human_feedback_rl.reward_models.networks import SumoRewardNetwork
from human_feedback_rl.policy.wrappers import PredictedRewardVecWrapper
from human_feedback_rl.policy.callbacks import SegmentCollectorCallback
from human_feedback_rl.feedback.segment import Segment
from human_feedback_rl.feedback.preference_collector import PreferenceCollector
from human_feedback_rl.feedback.demonstration_collector import DemonstrationCollector
from human_feedback_rl.feedback.oracles.factory import build_oracle
from human_feedback_rl.utils.env_setup import build_env_and_expert, build_single_env, build_demo_env_and_expert


# ─────────────────────────────────────────────────────────────────────────────
# Policy worker
# ─────────────────────────────────────────────────────────────────────────────


def _policy_worker(
    config_dict,
    segment_pipe,
    reward_predictor_ready_event,
    shutdown_event,
    reward_predictor_checkpoint_dir,
    policy_checkpoint_path,
    log_directory,
    shared_env_steps,
    agent_demo_pipe=None,
    policy_metrics_queue=None,
    a2c_steps=None,
):
    """
    Subprocess responsible for policy training via SB3 A2C.

    Phase 1 (before RP ready):
      Runs random-action rollouts to generate Segment objects for initial
      preference collection.  No gradient updates — the paper requires no RL
      training before the reward predictor is pretrained.

    Phase 2+ (after RP ready):
      Wraps the VecEnv with PredictedRewardVecWrapper so SB3 A2C trains on
      predicted rewards only (env rewards never exposed to the policy).
      SegmentCollectorCallback handles segment generation, RP reloading,
      checkpoint saving, and shared_env_steps updates.
    """
    from pathlib import Path
    from stable_baselines3 import A2C as SB3A2C
    from stable_baselines3.common.vec_env import VecMonitor

    config = OmegaConf.create(config_dict)

    env, _ = build_env_and_expert(config)

    initial_obs     = np.asarray(env.reset())
    num_envs        = initial_obs.shape[0] if initial_obs.ndim > 1 else 1
    observation_dim = initial_obs.shape[-1]

    reward_predictor = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=observation_dim),
        log_dir=None,
        device=config.resources.device,
    )

    segment_length         = config.preferences.segment_len
    total_env_steps_target = config.training.total_env_steps

    # ── Phase 1: random rollouts for segment generation (no A2C updates) ──────
    print("[policy] Phase 1 — generating segments with random policy…", flush=True)

    current_obs             = initial_obs if initial_obs.ndim > 1 else initial_obs[np.newaxis]
    current_segment_frames  = [[] for _ in range(num_envs)]
    current_segment_rewards = [[] for _ in range(num_envs)]
    total_env_steps_phase1  = 0

    while not reward_predictor_ready_event.is_set() and not shutdown_event.is_set():
        actions  = np.array([env.action_space.sample() for _ in range(num_envs)])
        next_obs_raw, env_rewards_raw, dones_raw, _ = env.step(actions)
        next_obs    = np.asarray(next_obs_raw)
        if next_obs.ndim == 1:
            next_obs = next_obs[np.newaxis]
        dones       = np.asarray(dones_raw, dtype=bool)
        env_rewards = np.asarray(env_rewards_raw, dtype=np.float32)

        total_env_steps_phase1 += num_envs
        shared_env_steps.value  = total_env_steps_phase1

        for env_idx in range(num_envs):
            current_segment_frames[env_idx].append(current_obs[env_idx].copy())
            current_segment_rewards[env_idx].append(float(env_rewards[env_idx]))
            if (
                len(current_segment_frames[env_idx]) >= segment_length
                or dones[env_idx]
            ):
                frames  = current_segment_frames[env_idx]
                rewards = current_segment_rewards[env_idx]
                while len(frames) < segment_length:
                    frames.append(frames[-1].copy())
                    rewards.append(0.0)
                seg = Segment(frames[:segment_length])
                seg.env_rewards = rewards[:segment_length]
                try:
                    segment_pipe.put(seg, block=False)
                except Exception:
                    pass
                if agent_demo_pipe is not None:
                    try:
                        agent_demo_pipe.put(seg, block=False)
                    except Exception:
                        pass
                current_segment_frames[env_idx]  = []
                current_segment_rewards[env_idx] = []

        current_obs = next_obs

    if shutdown_event.is_set():
        env.close()
        return

    # ── Phase 2+: RP ready — SB3 A2C with predicted rewards ──────────────────
    latest = RewardPredictorEnsemble.latest_checkpoint(reward_predictor_checkpoint_dir)
    if latest:
        reward_predictor.load(latest)

    reward_wrapper = PredictedRewardVecWrapper(env, reward_predictor, reward_predictor_ready_event)
    wrapped_env    = VecMonitor(reward_wrapper)

    # total_env_steps_target is the budget for A2C training only.
    # Phase 1 random rollouts are not counted against it.
    remaining_steps = total_env_steps_target if total_env_steps_target > 0 else int(1e9)

    callback = SegmentCollectorCallback(
        segment_pipe=segment_pipe,
        segment_length=segment_length,
        n_envs=num_envs,
        shutdown_event=shutdown_event,
        reward_predictor_ready_event=reward_predictor_ready_event,
        reward_predictor_wrapper=reward_wrapper,
        reward_predictor_checkpoint_dir=reward_predictor_checkpoint_dir,
        reload_interval=config.training.reward_predictor_reload_interval,
        save_interval=config.training.policy_save_interval,
        policy_checkpoint_path=policy_checkpoint_path,
        shared_env_steps=shared_env_steps,
        env_steps_offset=total_env_steps_phase1,
        agent_demo_pipe=agent_demo_pipe,
    )
    callback.metrics_queue = policy_metrics_queue
    callback.a2c_steps     = a2c_steps

    a2c = SB3A2C(
        "MlpPolicy",
        wrapped_env,
        learning_rate=config.policy.lr,
        gamma=config.policy.gamma,
        n_steps=config.policy.rollout_steps,
        ent_coef=config.policy.entropy_coef,
        vf_coef=config.policy.value_coef,
        max_grad_norm=config.policy.max_gradient_norm,
        policy_kwargs={"net_arch": [64, 64]},
        device=config.resources.device,
        verbose=0,
    )

    print("[policy] SB3 A2C started — training with predicted rewards.", flush=True)

    a2c.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        reset_num_timesteps=True,
    )

    # Save final policy checkpoint (SB3 format, adds .zip automatically).
    checkpoint_path = Path(policy_checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    a2c.save(str(checkpoint_path))
    print(f"[policy] Policy saved to {checkpoint_path}.zip", flush=True)

    if not shutdown_event.is_set():
        shutdown_event.set()

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Demo preference worker
# ─────────────────────────────────────────────────────────────────────────────


def _demo_preference_worker(
    config_dict,
    agent_demo_pipe,
    preference_pipe,
    shutdown_event,
):
    """
    Subprocess responsible for turning expert-correction pairs into preferences.

    For each agent segment received from agent_demo_pipe:
      1. Builds an expert-correction segment via DemonstrationCollector.
      2. Sends (expert_frames, agent_frames, (1.0, 0.0)) to preference_pipe.

    The label (1.0, 0.0) encodes that the expert correction is always preferred
    over the agent segment. PrefBuffer routes these into train/val PrefDB
    identically to oracle-labeled pairs, so reward model training uses only
    the standard Christiano MLE loss — no separate margin loss.

    Runs continuously through Phase 1 and Phase 3, so demo-based preferences
    keep flowing alongside oracle preferences at all times.
    """
    import queue as _queue

    config = OmegaConf.create(config_dict)
    env, expert_model = build_demo_env_and_expert(config)
    env.reset()

    collector = DemonstrationCollector(config.preferences.segment_len)

    print("[demo_pref] Demo preference worker started.", flush=True)

    while not shutdown_event.is_set():
        try:
            agent_seg = agent_demo_pipe.get(timeout=1.0)
        except _queue.Empty:
            continue

        expert_frames = collector.create_expert_correction(
            agent_seg.frames, expert_model, env
        )

        try:
            # Always prefer expert correction over agent segment.
            preference_pipe.put(
                (expert_frames, agent_seg.frames, (1.0, 0.0)),
                block=False,
            )
        except Exception:
            pass

    env.close()
# ─────────────────────────────────────────────────────────────────────────────
# Preference worker
# ─────────────────────────────────────────────────────────────────────────────


def _preference_worker(
    config_dict,
    segment_pipe,
    preference_pipe,
    reward_predictor_ready_event,
    shutdown_event,
    reward_predictor_checkpoint_dir,
    shared_env_steps,
):
    """
    Subprocess responsible for labeling segment pairs.

    Thin wrapper: delegates buffer management and RP loading to
    PreferenceCollector, and oracle labeling to the configured oracle.
    """
    sys.stdin = os.fdopen(0)

    config = OmegaConf.create(config_dict)
    oracle = build_oracle(config)

    # Need observation_dim to instantiate the RP for disagreement scoring.
    env = build_single_env(config)
    observation_dim = env.observation_space.shape[0]
    env.close()

    collector = PreferenceCollector(
        config, reward_predictor_checkpoint_dir, observation_dim
    )

    while not shutdown_event.is_set():
        collector.drain_pipe(segment_pipe)
        collector.refresh_rp(reward_predictor_ready_event)
        pair = collector.sample_pair()
        if pair is None:
            continue
        seg1, seg2 = pair
        pref = oracle.label(seg1, seg2)
        if pref is not None:
            preference_pipe.put((seg1.frames, seg2.frames, pref))
            collector.on_labeled(shared_env_steps, reward_predictor_ready_event)


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration worker
# ─────────────────────────────────────────────────────────────────────────────


def _demonstration_worker(
    config_dict,
    agent_demo_pipe,
    demo_pipe,
    shutdown_event,
):
    """
    Subprocess responsible for generating expert-correction demonstration pairs.

    For each agent segment received, queries the expert policy on each agent
    observation to get the expert's action, steps the demo environment, and
    collects the resulting observations as the expert-correction segment.

    Sends (expert_correction_frames, agent_frames) to demo_pipe.
    """
    import queue as _queue

    config = OmegaConf.create(config_dict)
    env, expert_model = build_demo_env_and_expert(config)
    env.reset()

    collector = DemonstrationCollector(config.preferences.segment_len)

    print("[demo] Demonstration worker started.", flush=True)

    while not shutdown_event.is_set():
        try:
            agent_seg = agent_demo_pipe.get(timeout=1.0)
        except _queue.Empty:
            continue

        expert_frames = collector.create_expert_correction(
            agent_seg.frames, expert_model, env
        )

        try:
            demo_pipe.put((expert_frames, agent_seg.frames), block=False)
        except Exception:
            pass

    env.close()
