"""
Worker functions for the Christiano et al. RLHF pipeline.

_policy_worker is defined here because it is tightly coupled to SB3 A2C
(SegmentCollectorCallback, PredictedRewardVecWrapper, VecMonitor).

The algorithm-agnostic workers (_preference_worker, _demonstration_worker,
_demo_preference_worker) live in human_feedback_rl.common.workers and are
re-exported here for convenience so christiano.py has a single import point.

These are top-level functions (not methods) because they are passed to
multiprocessing.Process(target=...) and must be picklable.
"""

import functools

import numpy as np
from omegaconf import OmegaConf

from human_feedback_rl.common.reward_predictor.ensemble import RewardPredictorEnsemble
from human_feedback_rl.common.reward_predictor.networks import SumoRewardNetwork
from human_feedback_rl.common.wrappers import PredictedRewardVecWrapper
from human_feedback_rl.common.callbacks import SegmentCollectorCallback
from human_feedback_rl.common.segment import Segment
from human_feedback_rl.common.utils.env_setup import build_env_and_expert

# Re-export common workers so christiano.py imports from a single location.
from human_feedback_rl.common.workers import (          # noqa: F401
    _preference_worker,
    _demonstration_worker,
    _demo_preference_worker,
    _set_thread_limits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Policy worker  (SB3 A2C — Christiano-specific)
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
    import torch
    from pathlib import Path
    from stable_baselines3 import A2C as SB3A2C
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    config = OmegaConf.create(config_dict)

    # Limit all CPU thread pools so multiple spawned processes don't saturate the CPU.
    _set_thread_limits(config.resources.torch_num_threads)

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

    current_obs              = initial_obs if initial_obs.ndim > 1 else initial_obs[np.newaxis]
    current_segment_frames   = [[] for _ in range(num_envs)]
    current_segment_rewards  = [[] for _ in range(num_envs)]
    current_segment_actions  = [[] for _ in range(num_envs)]
    total_env_steps_phase1   = 0

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
            current_segment_actions[env_idx].append(int(actions[env_idx]))
            if (
                len(current_segment_frames[env_idx]) >= segment_length
                or dones[env_idx]
            ):
                frames  = current_segment_frames[env_idx]
                rewards = current_segment_rewards[env_idx]
                acts    = current_segment_actions[env_idx]
                while len(frames) < segment_length:
                    frames.append(frames[-1].copy())
                    rewards.append(0.0)
                    acts.append(acts[-1])
                seg = Segment(frames[:segment_length])
                seg.env_rewards = rewards[:segment_length]
                seg.actions     = acts[:segment_length]
                try:
                    segment_pipe.put(seg, block=False)
                except Exception:
                    pass
                if agent_demo_pipe is not None:
                    try:
                        agent_demo_pipe.put(seg, block=False)
                    except Exception:
                        pass
                current_segment_frames[env_idx]   = []
                current_segment_rewards[env_idx]  = []
                current_segment_actions[env_idx]  = []

        current_obs = next_obs

    if shutdown_event.is_set():
        env.close()
        return

    # ── Phase 2+: RP ready — SB3 A2C with predicted rewards ──────────────────
    latest = RewardPredictorEnsemble.latest_checkpoint(reward_predictor_checkpoint_dir)
    if latest:
        reward_predictor.load(latest)

    reward_wrapper = PredictedRewardVecWrapper(env, reward_predictor, reward_predictor_ready_event)
    # Normalise predicted rewards online (running return variance, per-rollout).
    # Isolates the value function from RP scale changes between checkpoint reloads.
    # norm_obs=False: observations are already structured for the RP — don't touch them.
    # clip_reward: prevents extreme values during early RP training.
    norm_wrapper   = VecNormalize(
        reward_wrapper,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=config.policy.gamma,
    )
    wrapped_env    = VecMonitor(norm_wrapper)

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
