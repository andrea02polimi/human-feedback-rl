"""
Christiano et al. (2017) — Learning from Human Preferences
Asynchronous A2C implementation for the SUMO highway environment.

Three concurrent activities running in separate processes:

  Policy process  — runs A2C rollouts, generates individual trajectory
                    segments sent to segment_pipe, and updates the policy
                    ONLY with rewards from the reward predictor (never with
                    environment rewards — the environment reward is unknown,
                    as required by the paper).

  Preference process — accumulates segments from segment_pipe in a circular
                    buffer, randomly samples pairs, labels them via the
                    configured oracle (DQN expert or human terminal), and
                    forwards labeled triples to preference_pipe.

  Main process    — manages PrefDB, trains the RewardPredictorEnsemble on
                    incoming preferences, and saves checkpoints to disk.

Communication:
  segment_pipe                : Queue[Segment]
                                  policy → preference
  preference_pipe             : Queue[(frames, frames, pref)]
                                  preference → main (PrefBuffer)
  reward_predictor_ready_event: mp.Event
                                  main → policy  (reward predictor ready)
  shutdown_event              : mp.Event
                                  main → all     (time to stop)
  filesystem                  : reward_predictor_checkpoints/
                                  main → policy  (reward predictor weights)
                                models/policy_christiano.zip
                                  policy → eval / play
"""

import functools
import multiprocessing as mp
import os
import queue
import random
import signal
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C as SB3A2C
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import tqdm

from learning_from_human_preferences.preferences.pref_db import PrefDB, PrefBuffer, Segment
from learning_from_human_preferences.reward_model.reward_predictor import (
    RewardPredictorEnsemble,
)

from human_feedback_rl.christiano.expert_pref_interface import ExpertPrefInterface
from human_feedback_rl.christiano.human_pref_interface import HumanPrefInterface
from human_feedback_rl.christiano.reward_network import SumoRewardNetwork
from human_feedback_rl.christiano.sb3_components import (
    PredictedRewardVecWrapper,
    SegmentCollectorCallback,
)
from human_feedback_rl.utils.env_setup import build_env_and_expert, build_single_env


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _keep_latest_checkpoints(checkpoint_dir: str, keep: int = 2) -> None:
    """Delete old reward predictor checkpoints, keeping only the `keep` most recent."""
    if not os.path.exists(checkpoint_dir):
        return
    files = sorted(
        [f for f in os.listdir(checkpoint_dir)
         if f.startswith("reward_predictor_") and f.endswith(".pt")],
        key=lambda f: int(f[len("reward_predictor_"):-len(".pt")]),
    )
    for old_file in files[:-keep]:
        os.remove(os.path.join(checkpoint_dir, old_file))


# ─────────────────────────────────────────────────────────────────────────────
# Worker: SB3 A2C policy training + segment generation
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

    segment_length          = config.preferences.segment_len
    total_env_steps_target  = config.training.total_env_steps

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

    wrapped_env = VecMonitor(
        PredictedRewardVecWrapper(env, reward_predictor, reward_predictor_ready_event)
    )

    remaining_steps = (
        max(1, total_env_steps_target - total_env_steps_phase1)
        if total_env_steps_target > 0
        else int(1e9)
    )

    callback = SegmentCollectorCallback(
        segment_pipe=segment_pipe,
        segment_length=segment_length,
        n_envs=num_envs,
        shutdown_event=shutdown_event,
        reward_predictor_ready_event=reward_predictor_ready_event,
        reward_predictor_wrapper=wrapped_env,
        reward_predictor_checkpoint_dir=reward_predictor_checkpoint_dir,
        reload_interval=config.training.reward_predictor_reload_interval,
        save_interval=config.training.policy_save_interval,
        policy_checkpoint_path=policy_checkpoint_path,
        shared_env_steps=shared_env_steps,
        env_steps_offset=total_env_steps_phase1,
    )

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
        tensorboard_log=log_directory,
        device=config.resources.device,
        verbose=0,
    )

    print("[policy] SB3 A2C started — training with predicted rewards.", flush=True)

    a2c.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        tb_log_name="policy",
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
# Worker: preference labeling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_pair(segment_buffer):
    """Randomly sample two different segments from the buffer."""
    if len(segment_buffer) < 2:
        return None
    first_index, second_index = random.sample(range(len(segment_buffer)), 2)
    return segment_buffer[first_index], segment_buffer[second_index]


def _disagreement_score(segment, reward_predictor):
    """
    Variance of per-ensemble-member total segment rewards.

    Higher variance means the ensemble members disagree more about the value of
    the segment — these are the most informative pairs to label (Section 3.2 of
    Christiano et al.).
    """
    frames = np.array(segment.frames, dtype=np.float32)  # (segment_len, obs_dim)
    raw    = reward_predictor.raw_rewards(frames)         # (n_preds, segment_len)
    member_totals = raw.sum(axis=-1).flatten()            # (n_preds,)
    return float(np.var(member_totals))


def _sample_pair_by_disagreement(segment_buffer, reward_predictor, n_candidates: int):
    """
    Return the segment pair with highest ensemble disagreement from n_candidates
    random candidates (Christiano et al. Section 3.2).

    Disagreement is the sum of per-segment variances across ensemble members.
    Pairs on which the ensemble disagrees most are the most informative to label.
    Falls back to the best pair found even if disagreement is zero (untrained RP).
    """
    if len(segment_buffer) < 2:
        return None
    best_pair  = None
    best_score = -1.0
    for _ in range(n_candidates):
        i, j   = random.sample(range(len(segment_buffer)), 2)
        seg_a, seg_b = segment_buffer[i], segment_buffer[j]
        score  = _disagreement_score(seg_a, reward_predictor) + _disagreement_score(seg_b, reward_predictor)
        if score > best_score:
            best_score = score
            best_pair  = (seg_a, seg_b)
    return best_pair


def _preference_worker(
    config_dict,
    segment_pipe,
    preference_pipe,
    reward_predictor_ready_event,
    shutdown_event,
    reward_predictor_checkpoint_dir,
    log_directory,
    shared_env_steps,
):
    """
    Subprocess responsible for labeling segment pairs.

    Receives individual Segment objects from segment_pipe, accumulates them in
    a circular buffer, then:
      • Before the reward predictor is ready: samples pairs randomly.
      • After the reward predictor is ready: samples the highest-disagreement
        pair from `disagreement_candidates` random candidates (Section 3.2 of
        Christiano et al.) and reloads the latest RP checkpoint periodically.

    Query annealing: after `initial_prefs` labels the inter-query sleep grows
    linearly so that the RL policy has time to explore before being judged
    (Christiano et al. Section 3.2).
    """
    sys.stdin = os.fdopen(0)

    config = OmegaConf.create(config_dict)

    if config.preferences.oracle == "expert":
        # Oracle uses true environment rewards (seg.env_rewards) — no DQN needed.
        # To re-enable q-net scoring, uncomment the block below and comment out
        # the two lines that follow it.
        # ── Q-net oracle (DQN expert) — disabled ──────────────────────────────
        # env, expert_model = build_env_and_expert(config)
        # observation_dim   = env.observation_space.shape[0]
        # env.close()
        # interface = ExpertPrefInterface(
        #     expert_model=expert_model,
        #     max_segs=config.preferences.max_segs,
        #     log_dir=log_directory,
        # )
        # ── Env-reward oracle (current) ───────────────────────────────────────
        env = build_single_env(config)
        observation_dim = env.observation_space.shape[0]
        env.close()
        interface = ExpertPrefInterface(
            max_segs=config.preferences.max_segs,
            log_dir=log_directory,
        )
    else:
        env = build_single_env(config)
        observation_dim = env.observation_space.shape[0]
        env.close()
        interface = HumanPrefInterface(
            max_segs=config.preferences.max_segs,
            log_dir=log_directory,
        )

    # Inference-only reward predictor for disagreement-based pair selection.
    # Weights come from checkpoints saved by the main process — never trained here.
    rp_for_disagreement = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=observation_dim),
        log_dir=None,
        device=config.resources.device,
    )
    rp_loaded = False

    segment_buffer     = []
    buffer_write_index = 0
    total_labeled      = 0

    while not shutdown_event.is_set():

        # ── Load / refresh reward predictor for disagreement scoring ──────────
        if not rp_loaded and reward_predictor_ready_event.is_set():
            latest = RewardPredictorEnsemble.latest_checkpoint(reward_predictor_checkpoint_dir)
            if latest:
                try:
                    rp_for_disagreement.load(latest)
                    rp_loaded = True
                except Exception:
                    pass
        elif rp_loaded and total_labeled % 50 == 0 and total_labeled > 0:
            latest = RewardPredictorEnsemble.latest_checkpoint(reward_predictor_checkpoint_dir)
            if latest:
                try:
                    rp_for_disagreement.load(latest)
                except Exception:
                    pass

        # Drain up to 8 new segments from segment_pipe
        for _ in range(8):
            try:
                segment = segment_pipe.get(timeout=0.5)
                if len(segment_buffer) < config.preferences.max_segs:
                    segment_buffer.append(segment)
                else:
                    segment_buffer[buffer_write_index % config.preferences.max_segs] = segment
                    buffer_write_index += 1
            except queue.Empty:
                break

        if len(segment_buffer) < 2:
            continue

        # Disagreement-based selection when RP available, random otherwise
        if rp_loaded:
            pair = _sample_pair_by_disagreement(
                segment_buffer,
                rp_for_disagreement,
                n_candidates=config.preferences.disagreement_candidates,
            )
        else:
            pair = _sample_pair(segment_buffer)

        if pair is None:
            continue

        segment_1, segment_2 = pair
        preference = interface.ask_user(segment_1, segment_2)
        if preference is not None:
            preference_pipe.put((segment_1.frames, segment_2.frames, preference))
            total_labeled += 1

            # Query annealing: label rate decays linearly with environment steps
            # (only after the RP is ready, i.e. Phase 3).  At 0 env steps the
            # query rate is maximum; at total_env_steps it reaches max_query_interval
            # seconds between queries — matching the paper's "label rate decays
            # with environment steps" schedule.
            if (
                reward_predictor_ready_event.is_set()
                and config.training.total_env_steps > 0
            ):
                fraction = min(
                    1.0,
                    shared_env_steps.value / config.training.total_env_steps,
                )
                sleep_s = config.preferences.max_query_interval * fraction
                if sleep_s > 0:
                    time.sleep(sleep_s)


# ─────────────────────────────────────────────────────────────────────────────
# Main process
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_christiano.yaml",
)
def main(cfg: DictConfig):

    mp.set_start_method("spawn", force=True)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    run_directory                    = Path(HydraConfig.get().runtime.output_dir)
    reward_predictor_checkpoint_dir  = str(run_directory / "reward_predictor_checkpoints")
    policy_checkpoint_path           = str(run_directory / "models" / "policy_christiano")
    preference_interface_log_dir     = str(run_directory / "pref_interface")

    # ── Communication channels ────────────────────────────────────────────────
    segment_pipe                = Queue(maxsize=cfg.preferences.seg_pipe_maxsize)
    preference_pipe             = Queue()
    reward_predictor_ready_event = mp.Event()   # main → policy: reward predictor ready
    shutdown_event              = mp.Event()    # main → all:    time to stop

    # ── Preference databases (owned by main process) ──────────────────────────
    train_database      = PrefDB(maxlen=cfg.preferences.db_train_maxlen)
    validation_database = PrefDB(maxlen=cfg.preferences.db_val_maxlen)
    preference_buffer   = PrefBuffer(
        train_database,
        validation_database,
        log_dir=str(run_directory / "pref_buffer"),
    )
    preference_buffer.start_recv_thread(preference_pipe)

    # ── Reward predictor (trained in main process) ────────────────────────────
    temp_env        = build_single_env(cfg)
    observation_dim = temp_env.observation_space.shape[0]
    temp_env.close()

    reward_predictor = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=observation_dim),
        lr=cfg.reward_predictor.lr,
        n_preds=cfg.reward_predictor.n_preds,
        log_dir=str(run_directory),
        device=cfg.resources.device,
    )

    # ── Launch worker processes ───────────────────────────────────────────────
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Shared counter so the preference worker can read env steps for annealing.
    shared_env_steps = mp.Value("l", 0)

    policy_process = Process(
        target=_policy_worker,
        args=(
            config_dict,
            segment_pipe,
            reward_predictor_ready_event,
            shutdown_event,
            reward_predictor_checkpoint_dir,
            policy_checkpoint_path,
            str(run_directory),
            shared_env_steps,
        ),
    )
    preference_process = Process(
        target=_preference_worker,
        args=(
            config_dict,
            segment_pipe,
            preference_pipe,
            reward_predictor_ready_event,
            shutdown_event,
            reward_predictor_checkpoint_dir,
            preference_interface_log_dir,
            shared_env_steps,
        ),
    )

    policy_process.start()
    preference_process.start()

    def _shutdown(*_):
        print("\n[main] Shutting down…", flush=True)
        shutdown_event.set()
        policy_process.join(timeout=15)
        preference_process.join(timeout=15)
        # Force-terminate any worker still alive after the grace period.
        for proc in (policy_process, preference_process):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
        preference_buffer.stop_recv_thread()
        print(f"[main] Policy saved to {policy_checkpoint_path}")
        os._exit(0)   # bypass atexit so non-daemon workers don't block exit

    signal.signal(signal.SIGINT, _shutdown)

    # ── Phase 1: collect initial preferences (random/untrained policy) ────────
    target_preferences = cfg.preferences.initial_prefs
    print(f"\n[Phase 1] Collecting {target_preferences} initial preferences…")
    with tqdm(total=target_preferences, desc="preferences", unit="pref", ncols=80) as progress_bar:
        previous_count = 0
        while True:
            current_train_db, current_val_db = preference_buffer.get_dbs()
            current_count = len(current_train_db)
            if current_count > previous_count:
                progress_bar.update(current_count - previous_count)
                previous_count = current_count
            if current_count >= target_preferences and len(current_val_db) > 0:
                break
            time.sleep(2.0)
    train_db, val_db = preference_buffer.get_dbs()
    print(f"  Done — train={len(train_db)}, validation={len(val_db)}")

    # ── Phase 2: pretrain reward predictor ────────────────────────────────────
    print("\n[Phase 2] Pretraining reward predictor…")
    reward_predictor.train(train_db, val_db, val_interval=cfg.reward_predictor.val_interval)
    reward_predictor.save()
    _keep_latest_checkpoints(reward_predictor_checkpoint_dir)
    reward_predictor_ready_event.set()
    print("  Reward predictor ready — A2C training with predicted rewards unlocked.")

    # ── Phase 3: continuous reward predictor retraining ───────────────────────
    # The RP retrains as fast as new preferences arrive and keeps running until
    # the policy worker sets shutdown_event (i.e. total_env_steps reached).
    print("\n[Phase 3] Reward predictor training continuously…")

    rp_retrain_count = 0
    while not shutdown_event.is_set():
        train_db, val_db = preference_buffer.get_dbs()
        if len(train_db) == 0 or len(val_db) == 0:
            time.sleep(1.0)
            continue

        reward_predictor.train(
            train_db, val_db, val_interval=cfg.reward_predictor.val_interval
        )
        reward_predictor.save()
        _keep_latest_checkpoints(reward_predictor_checkpoint_dir)
        rp_retrain_count += 1
        print(
            f"[rp] retrain #{rp_retrain_count}"
            f"  train={len(train_db)}  val={len(val_db)}",
            flush=True,
        )

    # ── Shutdown ───────────────────────────────────────────────────────────────
    _shutdown()


if __name__ == "__main__":
    main()
