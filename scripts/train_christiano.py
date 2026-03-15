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
                                models/policy_christiano.pt
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
import torch.nn.functional as F
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from learning_from_human_preferences.preferences.pref_db import PrefDB, PrefBuffer, Segment
from learning_from_human_preferences.reward_model.reward_predictor import (
    RewardPredictorEnsemble,
)

from human_feedback_rl.agents.policy_network import AgentPolicyNetwork
from human_feedback_rl.christiano.expert_pref_interface import ExpertPrefInterface
from human_feedback_rl.christiano.human_pref_interface import HumanPrefInterface
from human_feedback_rl.christiano.reward_network import SumoRewardNetwork
from human_feedback_rl.utils.env_setup import build_env_and_expert, build_single_env


# ─────────────────────────────────────────────────────────────────────────────
# A2C helpers
# ─────────────────────────────────────────────────────────────────────────────

def _discount_with_dones(rewards, dones, discount_factor: float):
    """
    Compute discounted returns with episode-end masking.

    When the last step did NOT end the episode, the caller should append the
    bootstrap value to `rewards` and 0 to `dones`, then drop the last element
    of the returned list.
    """
    returns = []
    cumulative_return = 0.0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        cumulative_return = reward + discount_factor * cumulative_return * (1.0 - float(done))
        returns.append(cumulative_return)
    return returns[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# Worker: A2C policy training + segment generation
# ─────────────────────────────────────────────────────────────────────────────

def _policy_worker(
    config_dict,
    segment_pipe,
    reward_predictor_ready_event,
    shutdown_event,
    reward_predictor_checkpoint_dir,
    policy_checkpoint_path,
    log_directory,
):
    """
    Subprocess responsible for:
      1. Running A2C rollouts and sending individual Segment objects to
         segment_pipe so that the preference process can label them.
      2. Updating the policy with PREDICTED rewards from the reward predictor
         — never with environment rewards (the paper assumes the environment
         reward is unknown).  Policy gradient updates are skipped entirely
         until reward_predictor_ready_event is set and a checkpoint is
         available.
    """
    config = OmegaConf.create(config_dict)

    env, _ = build_env_and_expert(config)

    initial_observations = np.asarray(env.reset())
    num_envs        = initial_observations.shape[0] if initial_observations.ndim > 1 else 1
    observation_dim = initial_observations.shape[-1]
    num_actions     = env.action_space.n

    policy    = AgentPolicyNetwork(observation_dim, num_actions)
    optimizer = optim.Adam(policy.parameters(), lr=config.policy.lr)

    # Inference-only reward predictor in this process (no checkpoint writes,
    # no TensorBoard).  Rewards produced by reward_predictor.reward() are
    # normalised to zero mean and constant standard deviation (× 0.05) as
    # described in Section 2.3 of Christiano et al.
    reward_predictor = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=observation_dim),
        log_dir=None,
        device=config.resources.device,
    )

    writer = SummaryWriter(os.path.join(log_directory, "policy"))

    # Current observations: shape (num_envs, observation_dim)
    current_observations = (
        initial_observations
        if initial_observations.ndim > 1
        else initial_observations[np.newaxis]
    )
    episode_dones = np.zeros(num_envs, dtype=bool)

    # Segment accumulation for environment 0 only (mirrors run.py's Runner)
    current_segment_frames = []

    rollout_steps         = config.policy.rollout_steps
    discount_factor       = config.policy.gamma
    entropy_coefficient   = config.policy.entropy_coef
    value_coefficient     = config.policy.value_coef
    max_gradient_norm     = config.policy.max_gradient_norm
    segment_length        = config.preferences.segment_len

    reward_predictor_ready = False
    training_step          = 0

    while not shutdown_event.is_set():

        # ── Load / refresh reward predictor ───────────────────────────────────
        if not reward_predictor_ready and reward_predictor_ready_event.is_set():
            latest_checkpoint = RewardPredictorEnsemble.latest_checkpoint(
                reward_predictor_checkpoint_dir
            )
            if latest_checkpoint:
                try:
                    reward_predictor.load(latest_checkpoint)
                    reward_predictor_ready = True
                    print(
                        "[policy] reward predictor loaded — "
                        "A2C training with predicted rewards started.",
                        flush=True,
                    )
                except Exception as error:
                    print(f"[policy] reward predictor load failed: {error}", flush=True)

        if (
            reward_predictor_ready
            and training_step % config.training.reward_predictor_reload_interval == 0
            and training_step > 0
        ):
            latest_checkpoint = RewardPredictorEnsemble.latest_checkpoint(
                reward_predictor_checkpoint_dir
            )
            if latest_checkpoint:
                try:
                    reward_predictor.load(latest_checkpoint)
                except Exception:
                    pass

        # ── Collect rollout_steps of experience ───────────────────────────────
        rollout_observations = []   # (rollout_steps, num_envs, observation_dim)
        rollout_actions      = []   # (rollout_steps, num_envs)
        rollout_values       = []   # (rollout_steps, num_envs)
        rollout_dones        = []   # (rollout_steps, num_envs)

        for _ in range(rollout_steps):
            observations_tensor = torch.as_tensor(
                current_observations, dtype=torch.float32
            )
            with torch.no_grad():
                action_logits, state_values = policy(observations_tensor)

            action_distribution = torch.distributions.Categorical(logits=action_logits)
            sampled_actions     = action_distribution.sample()

            rollout_observations.append(current_observations.copy())
            rollout_actions.append(sampled_actions.numpy())
            rollout_values.append(state_values.numpy())
            rollout_dones.append(episode_dones.copy())

            next_observations_raw, _env_rewards, dones_raw, _ = env.step(
                sampled_actions.numpy()
            )
            next_observations = np.asarray(next_observations_raw)
            if next_observations.ndim == 1:
                next_observations = next_observations[np.newaxis]
            episode_dones = np.asarray(dones_raw, dtype=bool)

            # ── Segment generation (environment 0, mirrors run.py Runner) ─────
            current_segment_frames.append(current_observations[0].copy())
            if len(current_segment_frames) >= segment_length or episode_dones[0]:
                while len(current_segment_frames) < segment_length:
                    current_segment_frames.append(current_segment_frames[-1].copy())
                completed_segment = Segment(current_segment_frames[:segment_length])
                try:
                    segment_pipe.put(completed_segment, block=False)
                except Exception:
                    pass
                current_segment_frames = []

            current_observations = next_observations

        # ── A2C update — skipped until reward predictor is ready ──────────────
        # The paper assumes the environment reward is unknown.  Policy gradient
        # updates only start once the reward predictor has been pretrained.
        if not reward_predictor_ready:
            continue

        # ── Replace env rewards with normalised predicted rewards ─────────────
        # reward_predictor.reward() normalises rewards to zero mean and constant
        # standard deviation (scaled by 0.05) using running statistics, as
        # described in Section 2.3 of Christiano et al. (2017).
        observations_array = np.stack(rollout_observations, axis=0)   # (T, N, obs_dim)
        flat_observations  = observations_array.reshape(-1, observation_dim)
        predicted_rewards  = reward_predictor.reward(flat_observations)   # (T*N,)
        if not np.all(np.isfinite(predicted_rewards)):
            predicted_rewards = np.zeros_like(predicted_rewards)
        predicted_rewards_grid = predicted_rewards.reshape(rollout_steps, num_envs)

        # ── Bootstrap last state value ────────────────────────────────────────
        with torch.no_grad():
            _, last_state_values = policy(
                torch.as_tensor(current_observations, dtype=torch.float32)
            )
        last_state_values_np = last_state_values.numpy()   # (num_envs,)

        dones_array = np.stack(rollout_dones, axis=0)      # (rollout_steps, num_envs)

        # ── Discounted returns per environment (mirrors run.py Runner.run) ────
        discounted_returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        for env_index in range(num_envs):
            rewards_for_env = predicted_rewards_grid[:, env_index].tolist()
            dones_for_env   = dones_array[:, env_index].tolist()
            if not dones_for_env[-1]:
                rewards_for_env = rewards_for_env + [float(last_state_values_np[env_index])]
                dones_for_env   = dones_for_env   + [0]
                discounted_returns[:, env_index] = _discount_with_dones(
                    rewards_for_env, dones_for_env, discount_factor
                )[:-1]
            else:
                discounted_returns[:, env_index] = _discount_with_dones(
                    rewards_for_env, dones_for_env, discount_factor
                )

        # ── A2C loss = policy loss + value loss − entropy bonus ───────────────
        flat_observations_tensor = torch.as_tensor(
            observations_array.reshape(-1, observation_dim), dtype=torch.float32
        )
        flat_actions_tensor  = torch.as_tensor(
            np.concatenate(rollout_actions), dtype=torch.long
        )
        flat_returns_tensor  = torch.as_tensor(
            discounted_returns.flatten(), dtype=torch.float32
        )
        flat_values_tensor   = torch.as_tensor(
            np.concatenate(rollout_values), dtype=torch.float32
        )

        advantages = (flat_returns_tensor - flat_values_tensor).detach()

        new_logits, new_values = policy(flat_observations_tensor)
        action_distribution    = torch.distributions.Categorical(logits=new_logits)
        log_probabilities      = action_distribution.log_prob(flat_actions_tensor)
        entropy                = action_distribution.entropy().mean()

        policy_loss = -(log_probabilities * advantages).mean()
        value_loss  = F.mse_loss(new_values, flat_returns_tensor)
        total_loss  = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy

        if torch.isfinite(total_loss):
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_gradient_norm)
            optimizer.step()

        # ── TensorBoard ───────────────────────────────────────────────────────
        writer.add_scalar("policy/policy_loss",  policy_loss.item(),                   training_step)
        writer.add_scalar("policy/value_loss",   value_loss.item(),                    training_step)
        writer.add_scalar("policy/entropy",      entropy.item(),                       training_step)
        writer.add_scalar("policy/avg_return",   float(discounted_returns.mean()),     training_step)

        if training_step % config.training.policy_save_interval == 0:
            checkpoint_path = Path(policy_checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), checkpoint_path)
            print(
                f"[policy] step={training_step}"
                f"  avg_return={discounted_returns.mean():.3f}"
                f"  policy_loss={policy_loss.item():.4f}"
                f"  value_loss={value_loss.item():.4f}",
                flush=True,
            )

        training_step += 1

    # Final checkpoint before exit
    checkpoint_path = Path(policy_checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_path)
    writer.close()
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


def _preference_worker(
    config_dict,
    segment_pipe,
    preference_pipe,
    shutdown_event,
    log_directory,
):
    """
    Subprocess responsible for labeling segment pairs.

    Receives individual Segment objects from segment_pipe, accumulates them in
    a circular buffer (mirrors PrefInterface.receive_segments), randomly
    samples pairs (mirrors PrefInterface.sample_segment_pair), queries the
    oracle, and forwards labeled triples to preference_pipe.
    """
    sys.stdin = os.fdopen(0)

    config = OmegaConf.create(config_dict)

    if config.preferences.oracle == "expert":
        env, expert_model = build_env_and_expert(config)
        env.close()      # only q_net weights are needed for segment scoring
        interface = ExpertPrefInterface(
            expert_model=expert_model,
            max_segs=config.preferences.max_segs,
            log_dir=log_directory,
        )
    else:
        interface = HumanPrefInterface(
            max_segs=config.preferences.max_segs,
            log_dir=log_directory,
        )

    segment_buffer    = []
    buffer_write_index = 0

    while not shutdown_event.is_set():

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

        pair = _sample_pair(segment_buffer)
        if pair is None:
            continue

        segment_1, segment_2 = pair
        preference = interface.ask_user(segment_1, segment_2)
        if preference is not None:
            preference_pipe.put((segment_1.frames, segment_2.frames, preference))


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
    policy_checkpoint_path           = str(run_directory / "models" / "policy_christiano.pt")
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
        ),
    )
    preference_process = Process(
        target=_preference_worker,
        args=(
            config_dict,
            segment_pipe,
            preference_pipe,
            shutdown_event,
            preference_interface_log_dir,
        ),
    )

    policy_process.start()
    preference_process.start()

    def _shutdown(*_):
        print("\n[main] Shutting down…", flush=True)
        shutdown_event.set()
        policy_process.join(timeout=15)
        preference_process.join(timeout=15)
        preference_buffer.stop_recv_thread()
        print(f"[main] Policy saved to {policy_checkpoint_path}")
        sys.exit(0)

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
    reward_predictor_ready_event.set()
    print("  Reward predictor ready — A2C training with predicted rewards unlocked.")

    # ── Phase 3: continuous reward predictor retraining ───────────────────────
    # Mirrors run.py: retrain as fast as new preferences arrive, no fixed sleep.
    # Stops after reward_predictor_iterations completed retrainings.
    print(
        f"\n[Phase 3] Reward predictor loop — "
        f"{cfg.training.reward_predictor_iterations} iterations"
    )

    completed_iterations = 0
    with tqdm(
        total=cfg.training.reward_predictor_iterations,
        desc="rp iterations",
        unit="iter",
        ncols=80,
    ) as progress_bar:
        while completed_iterations < cfg.training.reward_predictor_iterations:
            train_db, val_db = preference_buffer.get_dbs()
            if len(train_db) == 0 or len(val_db) == 0:
                progress_bar.set_postfix({"status": "waiting for data"})
                time.sleep(1.0)
                continue

            reward_predictor.train(
                train_db, val_db, val_interval=cfg.reward_predictor.val_interval
            )
            reward_predictor.save()
            completed_iterations += 1
            progress_bar.set_postfix({"train": len(train_db), "val": len(val_db)})
            progress_bar.update(1)

    # ── Shutdown ───────────────────────────────────────────────────────────────
    _shutdown()


if __name__ == "__main__":
    main()
