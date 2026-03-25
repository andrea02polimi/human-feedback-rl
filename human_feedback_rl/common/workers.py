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


def _set_thread_limits(n: int) -> None:
    """Pin all CPU thread pools to n threads to prevent oversubscription."""
    s = str(n)
    os.environ["OMP_NUM_THREADS"]     = s
    os.environ["MKL_NUM_THREADS"]     = s
    os.environ["OPENBLAS_NUM_THREADS"] = s
    os.environ["NUMEXPR_NUM_THREADS"] = s
    torch.set_num_threads(n)

from human_feedback_rl.common.preference_collector import PreferenceCollector
from human_feedback_rl.common.demonstration_collector import DemonstrationCollector
from human_feedback_rl.common.oracles.factory import build_oracle
from human_feedback_rl.common.utils.env_setup import build_single_env, build_expert_only


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
    segment_pipe and consumes (frames1, actions1, frames2, actions2, pref, source)
    tuples from preference_pipe.
    """
    sys.stdin = os.fdopen(0)

    config = OmegaConf.create(config_dict)

    # Limit all CPU thread pools so multiple spawned processes don't saturate the CPU.
    _set_thread_limits(config.resources.torch_num_threads)

    oracle = build_oracle(config)

    # Need observation_dim and action space info to instantiate the RP for disagreement scoring.
    env = build_single_env(config)
    observation_dim    = env.observation_space.shape[0]
    is_discrete        = hasattr(env.action_space, "n")
    action_feature_dim = env.action_space.n if is_discrete else env.action_space.shape[0]
    env.close()

    collector = PreferenceCollector(
        config, reward_predictor_checkpoint_dir, observation_dim,
        is_discrete=is_discrete, action_feature_dim=action_feature_dim,
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
            preference_pipe.put((
                seg1.frames, getattr(seg1, "actions", []),
                seg2.frames, getattr(seg2, "actions", []),
                pref, "oracle",
            ))
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
    import numpy as np

    config = OmegaConf.create(config_dict)

    # Limit all CPU thread pools (used for expert model inference).
    _set_thread_limits(config.resources.torch_num_threads)

    env, expert_model = build_expert_only(config)
    env.close()  # env not needed after loading the expert model weights

    collector = DemonstrationCollector()

    print("[demo] Demonstration worker started.", flush=True)

    while not shutdown_event.is_set():
        try:
            agent_seg = agent_demo_pipe.get(timeout=1.0)
        except _queue.Empty:
            continue

        agent_frames, expert_actions = collector.create_expert_correction(
            agent_seg.frames, expert_model
        )
        agent_actions = getattr(agent_seg, "actions", [0] * len(agent_seg.frames))

        try:
            demo_pipe.put(
                (np.array(agent_frames), np.array(expert_actions), np.array(agent_actions)),
                block=False,
            )
        except Exception:
            pass


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
    import numpy as np

    config = OmegaConf.create(config_dict)

    # Limit all CPU thread pools (used for expert model inference).
    _set_thread_limits(config.resources.torch_num_threads)

    env, expert_model = build_expert_only(config)
    env.close()  # env not needed after loading the expert model weights

    collector = DemonstrationCollector()

    print("[demo_pref] Demo preference worker started.", flush=True)

    while not shutdown_event.is_set():
        try:
            agent_seg = agent_demo_pipe.get(timeout=1.0)
        except _queue.Empty:
            continue

        agent_frames, expert_actions = collector.create_expert_correction(
            agent_seg.frames, expert_model
        )
        agent_actions = getattr(agent_seg, "actions", [0] * len(agent_seg.frames))
        frames_arr    = np.array(agent_frames)
        exp_act_arr   = np.array(expert_actions)
        ag_act_arr    = np.array(agent_actions)

        try:
            # seg1 = expert correction (frames + expert actions) is preferred (1.0, 0.0)
            # seg2 = agent segment    (frames + agent  actions)
            preference_pipe.put(
                (frames_arr, exp_act_arr, frames_arr, ag_act_arr, (1.0, 0.0), "demo"),
                block=False,
            )
        except Exception:
            pass
