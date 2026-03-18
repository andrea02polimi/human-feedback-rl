"""
Reusable worker functions for asynchronous RLHF pipelines.

These workers are algorithm-agnostic: they handle preference labeling and
demonstration generation independently of which RL algorithm is used for
policy optimisation. Any RLHF algorithm can import and reuse them.

All functions are top-level (not methods) because they are passed to
multiprocessing.Process(target=...) and must be picklable.
"""

import os
import sys
import time

import torch
from omegaconf import OmegaConf

from human_feedback_rl.common.preference_collector import PreferenceCollector
from human_feedback_rl.common.demonstration_collector import DemonstrationCollector
from human_feedback_rl.common.oracles.factory import build_oracle
from human_feedback_rl.common.utils.env_setup import build_single_env, build_demo_env_and_expert


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

    Compatible with any RLHF algorithm that produces Segment objects on
    segment_pipe and consumes (frames1, frames2, pref) tuples from preference_pipe.
    """
    sys.stdin = os.fdopen(0)

    config = OmegaConf.create(config_dict)

    # Limit PyTorch threads so multiple spawned processes don't saturate the CPU.
    torch.set_num_threads(config.resources.torch_num_threads)

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
            # No segments available yet — yield CPU instead of spinning.
            time.sleep(0.05)
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

    Sends (expert_correction_frames, agent_frames) to demo_pipe for use with
    the margin ranking loss (DemoDatabase path).
    """
    import queue as _queue

    config = OmegaConf.create(config_dict)

    # Limit PyTorch threads (used for expert model inference).
    torch.set_num_threads(config.resources.torch_num_threads)

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

    # Limit PyTorch threads (used for expert model inference).
    torch.set_num_threads(config.resources.torch_num_threads)

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
