"""
Christiano et al. (2017) — Learning from Human Preferences
Asynchronous A2C implementation for the SUMO highway environment.

Three concurrent activities running in separate processes:

  Policy process  — runs A2C rollouts over `nsteps` steps, generates
                    individual trajectory segments → seg_pipe, and uses the
                    reward predictor's predicted rewards (once ready) instead
                    of environment rewards.  Mirrors run.py's _training_worker
                    + Runner.

  Pref process    — accumulates segments from seg_pipe in a circular buffer,
                    randomly samples pairs, labels them via the configured
                    oracle (DQN expert or human terminal), and forwards labeled
                    triples to pref_pipe.  Mirrors run.py's _pref_interface_worker.

  Main process    — manages PrefDB, trains the RewardPredictorEnsemble on
                    incoming preferences, and saves checkpoints to disk.
                    Mirrors run.py's _train_policy_with_preferences.

Communication:
  seg_pipe   : Queue[Segment]                    policy → pref
  pref_pipe  : Queue[(frames, frames, pref)]     pref   → main (PrefBuffer)
  start_event: mp.Event                          main   → policy  (rp ready)
  stop_event : mp.Event                          main   → all     (shutdown)
  filesystem : reward_predictor_checkpoints/     main   → policy  (rp weights)
               models/policy_christiano.pt       policy → eval
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

def _discount_with_dones(rewards, dones, gamma: float):
    """
    Compute discounted returns with episode-end masking.

    If the last element of dones is 0 (episode not ended), the caller should
    append the bootstrap value to rewards and 0 to dones before calling, then
    drop the last element of the result.
    """
    returns = []
    R = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - float(d))
        returns.append(R)
    return returns[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# Worker: A2C policy training + segment generation
# ─────────────────────────────────────────────────────────────────────────────

def _policy_worker(cfg_dict, seg_pipe, start_event, stop_event,
                   rp_ckpt_dir, policy_ckpt_path, log_dir):
    """
    Subprocess responsible for:
      1. Running nsteps A2C rollouts and sending individual Segment objects
         (env 0 only, like run.py's Runner) to seg_pipe.
      2. Using environment rewards before the reward predictor is ready, then
         switching to predicted rewards once start_event is set and a checkpoint
         is available.
    """
    cfg = OmegaConf.create(cfg_dict)

    env, _ = build_env_and_expert(cfg)

    raw       = np.asarray(env.reset())
    n_envs    = raw.shape[0] if raw.ndim > 1 else 1
    obs_dim   = raw.shape[-1]
    n_actions = env.action_space.n

    policy    = AgentPolicyNetwork(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.policy.lr)

    # Inference-only reward predictor in this process (no TensorBoard writes)
    rp = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=obs_dim),
        log_dir=None,
        device=cfg.resources.device,
    )

    writer = SummaryWriter(os.path.join(log_dir, "policy"))

    # Current observations: (n_envs, obs_dim)
    current_obs = raw if raw.ndim > 1 else raw[np.newaxis]
    dones       = np.zeros(n_envs, dtype=bool)

    # Segment accumulation for env 0 (mirrors run.py's Runner.update_segment_buffer)
    seg_frames = []

    nsteps        = cfg.policy.nsteps
    gamma         = cfg.policy.gamma
    ent_coef      = cfg.policy.ent_coef
    vf_coef       = cfg.policy.vf_coef
    max_grad_norm = cfg.policy.max_grad_norm
    segment_len   = cfg.preferences.segment_len

    rp_ready = False
    step     = 0

    while not stop_event.is_set():

        # ── Load / refresh reward predictor ───────────────────────────────────
        if not rp_ready and start_event.is_set():
            latest = RewardPredictorEnsemble.latest_checkpoint(rp_ckpt_dir)
            if latest:
                try:
                    rp.load(latest)
                    rp_ready = True
                    print("[policy] rp loaded — switching to predicted rewards.",
                          flush=True)
                except Exception as e:
                    print(f"[policy] rp load failed: {e}", flush=True)

        if rp_ready and step % cfg.training.rp_reload_interval == 0 and step > 0:
            latest = RewardPredictorEnsemble.latest_checkpoint(rp_ckpt_dir)
            if latest:
                try:
                    rp.load(latest)
                except Exception:
                    pass

        # ── Collect nsteps rollout ─────────────────────────────────────────────
        mb_obs     = []   # (nsteps, n_envs, obs_dim)
        mb_actions = []   # (nsteps, n_envs)
        mb_values  = []   # (nsteps, n_envs)
        mb_rewards = []   # (nsteps, n_envs)
        mb_dones   = []   # (nsteps, n_envs)

        for _ in range(nsteps):
            obs_t = torch.as_tensor(current_obs, dtype=torch.float32)
            with torch.no_grad():
                logits, values = policy(obs_t)   # (n_envs, n_actions), (n_envs,)

            dist    = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()              # (n_envs,)

            mb_obs.append(current_obs.copy())
            mb_actions.append(actions.numpy())
            mb_values.append(values.numpy())
            mb_dones.append(dones.copy())

            next_obs_raw, env_rewards, dones_raw, _ = env.step(actions.numpy())
            next_obs    = np.asarray(next_obs_raw)
            if next_obs.ndim == 1:
                next_obs = next_obs[np.newaxis]
            dones       = np.asarray(dones_raw, dtype=bool)
            env_rewards = np.asarray(env_rewards).flatten()
            mb_rewards.append(env_rewards)

            # ── Segment generation (env 0 only, like run.py's Runner) ─────────
            seg_frames.append(current_obs[0].copy())
            if len(seg_frames) >= segment_len or dones[0]:
                while len(seg_frames) < segment_len:
                    seg_frames.append(seg_frames[-1].copy())
                seg = Segment(seg_frames[:segment_len])
                try:
                    seg_pipe.put(seg, block=False)
                except Exception:
                    pass
                seg_frames = []

            current_obs = next_obs

        # ── Replace env rewards with predicted rewards (when rp is ready) ─────
        mb_obs_arr = np.stack(mb_obs, axis=0)    # (nsteps, n_envs, obs_dim)

        if rp_ready:
            obs_flat  = mb_obs_arr.reshape(-1, obs_dim)
            predicted = rp.reward(obs_flat)              # (nsteps*n_envs,)
            if not np.all(np.isfinite(predicted)):
                predicted = np.zeros_like(predicted)
            mb_rewards_arr = predicted.reshape(nsteps, n_envs)
        else:
            mb_rewards_arr = np.stack(mb_rewards, axis=0)

        # ── Bootstrap last value ───────────────────────────────────────────────
        with torch.no_grad():
            _, last_values = policy(
                torch.as_tensor(current_obs, dtype=torch.float32)
            )
        last_values_np = last_values.numpy()          # (n_envs,)

        mb_dones_arr  = np.stack(mb_dones, axis=0)   # (nsteps, n_envs)

        # ── Discounted returns per env (mirrors run.py's Runner.run) ──────────
        returns = np.zeros((nsteps, n_envs), dtype=np.float32)
        for env_i in range(n_envs):
            rew_i  = mb_rewards_arr[:, env_i].tolist()
            done_i = mb_dones_arr[:, env_i].tolist()
            if not done_i[-1]:
                rew_i  = rew_i  + [float(last_values_np[env_i])]
                done_i = done_i + [0]
                returns[:, env_i] = _discount_with_dones(rew_i, done_i, gamma)[:-1]
            else:
                returns[:, env_i] = _discount_with_dones(rew_i, done_i, gamma)

        # ── A2C update ─────────────────────────────────────────────────────────
        obs_flat     = mb_obs_arr.reshape(-1, obs_dim)        # (T*N, obs_dim)
        actions_flat = np.concatenate(mb_actions)             # (T*N,)
        returns_flat = returns.flatten()                      # (T*N,)
        values_flat  = np.concatenate(mb_values)              # (T*N,)

        obs_t     = torch.as_tensor(obs_flat,     dtype=torch.float32)
        actions_t = torch.as_tensor(actions_flat, dtype=torch.long)
        returns_t = torch.as_tensor(returns_flat, dtype=torch.float32)
        values_t  = torch.as_tensor(values_flat,  dtype=torch.float32)

        advantages = (returns_t - values_t).detach()

        logits_new, values_new = policy(obs_t)
        dist_new   = torch.distributions.Categorical(logits=logits_new)
        log_probs  = dist_new.log_prob(actions_t)
        entropy    = dist_new.entropy().mean()

        policy_loss = -(log_probs * advantages).mean()
        value_loss  = F.mse_loss(values_new, returns_t)
        loss        = policy_loss + vf_coef * value_loss - ent_coef * entropy

        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        # ── TensorBoard ───────────────────────────────────────────────────────
        writer.add_scalar("policy/policy_loss", policy_loss.item(),         step)
        writer.add_scalar("policy/value_loss",  value_loss.item(),          step)
        writer.add_scalar("policy/entropy",     entropy.item(),             step)
        writer.add_scalar("policy/avg_return",  float(returns_flat.mean()), step)

        if step % cfg.training.policy_save_interval == 0:
            path = Path(policy_ckpt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), path)
            print(
                f"[policy] step={step}"
                f"  avg_return={returns_flat.mean():.3f}"
                f"  policy_loss={policy_loss.item():.4f}"
                f"  value_loss={value_loss.item():.4f}",
                flush=True,
            )

        step += 1

    # Final save before exit
    path = Path(policy_ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path)
    writer.close()
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Worker: preference labeling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_pair(segments):
    """Randomly sample two different segments from the buffer."""
    if len(segments) < 2:
        return None
    i, j = random.sample(range(len(segments)), 2)
    return segments[i], segments[j]


def _pref_worker(cfg_dict, seg_pipe, pref_pipe, stop_event, log_dir):
    """
    Subprocess responsible for labeling segment pairs.

    Receives individual Segment objects from seg_pipe, accumulates them in a
    circular buffer (mirrors PrefInterface.receive_segments), randomly samples
    pairs (mirrors PrefInterface.sample_segment_pair), queries the oracle, and
    forwards labeled triples to pref_pipe.
    """
    sys.stdin = os.fdopen(0)

    cfg = OmegaConf.create(cfg_dict)

    if cfg.preferences.oracle == "expert":
        env, expert_model = build_env_and_expert(cfg)
        env.close()      # only q_net weights are needed for scoring
        interface = ExpertPrefInterface(
            expert_model=expert_model,
            max_segs=cfg.preferences.max_segs,
            log_dir=log_dir,
        )
    else:
        interface = HumanPrefInterface(
            max_segs=cfg.preferences.max_segs,
            log_dir=log_dir,
        )

    segments  = []
    seg_index = 0   # circular buffer write pointer

    while not stop_event.is_set():

        # Drain up to 8 new segments from seg_pipe
        for _ in range(8):
            try:
                seg = seg_pipe.get(timeout=0.5)
                if len(segments) < cfg.preferences.max_segs:
                    segments.append(seg)
                else:
                    segments[seg_index % cfg.preferences.max_segs] = seg
                    seg_index += 1
            except queue.Empty:
                break

        if len(segments) < 2:
            continue

        pair = _sample_pair(segments)
        if pair is None:
            continue

        seg1, seg2 = pair
        pref = interface.ask_user(seg1, seg2)
        if pref is not None:
            pref_pipe.put((seg1.frames, seg2.frames, pref))


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

    run_dir     = Path(HydraConfig.get().runtime.output_dir)
    rp_ckpt_dir = str(run_dir / "reward_predictor_checkpoints")
    policy_path = str(run_dir / "models" / "policy_christiano.pt")
    pref_log    = str(run_dir / "pref_interface")

    # ── Communication ─────────────────────────────────────────────────────────
    seg_pipe    = Queue(maxsize=cfg.preferences.seg_pipe_maxsize)
    pref_pipe   = Queue()
    start_event = mp.Event()   # main → policy: rp checkpoint is ready
    stop_event  = mp.Event()   # main → all:    time to shut down

    # ── Preference databases (owned by main process) ──────────────────────────
    db_train = PrefDB(maxlen=cfg.preferences.db_train_maxlen)
    db_val   = PrefDB(maxlen=cfg.preferences.db_val_maxlen)
    buf = PrefBuffer(db_train, db_val, log_dir=str(run_dir / "pref_buffer"))
    buf.start_recv_thread(pref_pipe)

    # ── Reward predictor (trained in main process) ────────────────────────────
    tmp_env = build_single_env(cfg)
    obs_dim = tmp_env.observation_space.shape[0]
    tmp_env.close()

    rp = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=obs_dim),
        lr=cfg.reward_predictor.lr,
        n_preds=cfg.reward_predictor.n_preds,
        log_dir=str(run_dir),
        device=cfg.resources.device,
    )

    # ── Launch worker processes ───────────────────────────────────────────────
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    policy_proc = Process(
        target=_policy_worker,
        args=(cfg_dict, seg_pipe, start_event, stop_event,
              rp_ckpt_dir, policy_path, str(run_dir)),
    )
    pref_proc = Process(
        target=_pref_worker,
        args=(cfg_dict, seg_pipe, pref_pipe, stop_event, pref_log),
    )

    policy_proc.start()
    pref_proc.start()

    def _shutdown(*_):
        print("\n[main] Shutting down…", flush=True)
        stop_event.set()
        policy_proc.join(timeout=15)
        pref_proc.join(timeout=15)
        buf.stop_recv_thread()
        print(f"[main] Policy saved to {policy_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    # ── Phase 1: collect initial preferences ──────────────────────────────────
    target = cfg.preferences.initial_prefs
    print(f"\n[Phase 1] Collecting {target} initial preferences…")
    with tqdm(total=target, desc="preferences", unit="pref", ncols=80) as pbar:
        prev = 0
        while True:
            t_db, v_db = buf.get_dbs()
            current = len(t_db)
            if current > prev:
                pbar.update(current - prev)
                prev = current
            if current >= target and len(v_db) > 0:
                break
            time.sleep(2.0)
    t_db, v_db = buf.get_dbs()
    print(f"  Done — train={len(t_db)}, val={len(v_db)}")

    # ── Phase 2: pretrain reward predictor ────────────────────────────────────
    print("\n[Phase 2] Pretraining reward predictor…")
    rp.train(t_db, v_db, val_interval=cfg.reward_predictor.val_interval)
    rp.save()
    start_event.set()
    print("  Reward predictor ready — A2C training with predicted rewards unlocked.")

    # ── Phase 3: continuous reward predictor retraining ───────────────────────
    # Mirrors run.py: train as fast as new preferences arrive (no fixed sleep).
    # Stops after rp_iterations completed retrainings, then shuts down policy.
    print(f"\n[Phase 3] Reward predictor loop — {cfg.training.rp_iterations} iterations")

    completed = 0
    with tqdm(total=cfg.training.rp_iterations, desc="rp iterations", unit="iter", ncols=80) as pbar:
        while completed < cfg.training.rp_iterations:
            t_db, v_db = buf.get_dbs()
            if len(t_db) == 0 or len(v_db) == 0:
                pbar.set_postfix({"status": "waiting for data"})
                time.sleep(1.0)
                continue

            rp.train(t_db, v_db, val_interval=cfg.reward_predictor.val_interval)
            rp.save()
            completed += 1
            pbar.set_postfix({"train": len(t_db), "val": len(v_db)})
            pbar.update(1)

    # ── Shutdown ───────────────────────────────────────────────────────────────
    _shutdown()


if __name__ == "__main__":
    main()
