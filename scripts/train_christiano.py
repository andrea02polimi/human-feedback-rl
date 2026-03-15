"""
Christiano et al. (2017) — Learning from Human Preferences
Asynchronous implementation for the SUMO highway environment.

Three concurrent activities running in separate processes:

  Policy process  — continuously generates trajectory-segment pairs and,
                    once the reward predictor is ready, trains the policy
                    via REINFORCE using the latest reward predictor checkpoint.

  Pref process    — reads segment pairs from seg_pipe, labels them via the
                    configured oracle (DQN expert or human terminal), and
                    forwards labeled triples to pref_pipe.

  Main process    — manages PrefDB, trains the RewardPredictorEnsemble on
                    incoming preferences, and saves checkpoints to disk.

Communication:
  seg_pipe  : Queue[(Segment, Segment)]          policy → pref
  pref_pipe : Queue[(frames, frames, pref)]      pref   → main (PrefBuffer)
  start_event: mp.Event                          main   → policy  (rp ready)
  stop_event : mp.Event                          main   → all     (shutdown)
  filesystem : reward_predictor_checkpoints/     main   → policy  (rp weights)
               models/policy_christiano.pt       policy → eval
"""

import functools
import multiprocessing as mp
import os
import signal
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from learning_from_human_preferences.preferences.pref_db import PrefDB, PrefBuffer
from learning_from_human_preferences.reward_model.reward_predictor import (
    RewardPredictorEnsemble,
)

from human_feedback_rl.agents.policy_network import AgentPolicyNetwork
from human_feedback_rl.christiano.expert_pref_interface import ExpertPrefInterface
from human_feedback_rl.christiano.human_pref_interface import HumanPrefInterface
from human_feedback_rl.christiano.reward_network import SumoRewardNetwork
from human_feedback_rl.christiano.segment_collector import SegmentCollector
from human_feedback_rl.utils.env_setup import build_env_and_expert, build_single_env


# ─────────────────────────────────────────────────────────────────────────────
# Policy helpers  (SB3 VecEnv-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _policy_episode(env, policy, rp, n_envs: int):
    """
    Run one episode following `policy`, score each step with `rp`, and return
    (total_predicted_reward, [(log_prob, predicted_reward), ...]).
    """
    obs = env.reset()                              # SB3 VecEnv: no info tuple
    obs = np.asarray(obs)
    obs = obs[0] if obs.ndim > 1 else obs          # follow env 0

    done = False
    log_probs, obs_list = [], []

    while not done:
        state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(state)
        dist = torch.distributions.Categorical(torch.softmax(logits, dim=1))
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        obs_list.append(obs.copy())

        # SB3 VecEnv: array of actions → (obs, rewards, dones, infos)
        next_obs, _, dones, _ = env.step(np.array([action.item()] * n_envs))
        done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
        next_obs = np.asarray(next_obs)
        obs = next_obs[0] if next_obs.ndim > 1 else next_obs

    obs_batch = np.stack(obs_list, axis=0)      # (T, obs_dim)
    rewards_np = rp.reward(obs_batch)           # (T,) — normalised by rp

    if not np.all(np.isfinite(rewards_np)):     # guard for under-trained rp
        rewards_np = np.zeros_like(rewards_np)
    rewards_np = np.clip(rewards_np, -10.0, 10.0)

    return rewards_np.sum().item(), list(zip(log_probs, rewards_np.tolist()))


def _train_policy_episode(env, policy, optimizer, rp, gamma: float, n_envs: int):
    """One REINFORCE update using the learned reward predictor."""
    total_reward, transitions = _policy_episode(env, policy, rp, n_envs)

    returns, G = [], 0.0
    for _, r in reversed(transitions):
        G = r + gamma * G
        returns.insert(0, G)

    returns_t = torch.tensor(returns, dtype=torch.float32)
    if returns_t.numel() > 1:
        std = returns_t.std(correction=0)          # population std, never NaN
        returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)

    loss = sum(
        -lp * G_t for (lp, _), G_t in zip(transitions, returns_t)
    ) / len(transitions)

    if not torch.isfinite(loss):
        return total_reward, 0.0

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()

    return total_reward, loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Worker: policy training + segment generation
# ─────────────────────────────────────────────────────────────────────────────

def _policy_worker(cfg_dict, seg_pipe, start_event, stop_event,
                   rp_ckpt_dir, policy_ckpt_path, log_dir):
    """
    Subprocess responsible for:
      1. Continuously collecting trajectory-segment pairs → seg_pipe.
      2. Once start_event is set, training the policy with the latest rp
         checkpoint and saving the policy periodically to policy_ckpt_path.
    """
    cfg = OmegaConf.create(cfg_dict)

    env, _ = build_env_and_expert(cfg)

    raw = np.asarray(env.reset())
    n_envs  = raw.shape[0] if raw.ndim > 1 else 1
    obs_dim = raw.shape[-1]
    n_actions = env.action_space.n

    policy   = AgentPolicyNetwork(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.policy.lr)
    collector = SegmentCollector(env, segment_len=cfg.preferences.segment_len)

    # Inference-only rp in this process (no TensorBoard)
    rp = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=obs_dim),
        log_dir=None,
        device=cfg.resources.device,
    )

    writer = SummaryWriter(os.path.join(log_dir, "policy"))

    rp_ready = False
    step = 0

    while not stop_event.is_set():

        # ── Always generate segment pairs (feeds pref process) ──────────────
        try:
            seg1, _ = collector.collect_segment(policy)
            seg2, _ = collector.collect_segment(policy)
            seg_pipe.put((seg1, seg2), block=True, timeout=2)
        except Exception:
            pass

        # ── Wait for reward predictor pretraining to finish ─────────────────
        if not rp_ready and start_event.is_set():
            latest = RewardPredictorEnsemble.latest_checkpoint(rp_ckpt_dir)
            if latest:
                try:
                    rp.load(latest)
                    rp_ready = True
                    print("[policy] rp loaded — policy training started.", flush=True)
                except Exception as e:
                    print(f"[policy] rp load failed: {e}", flush=True)

        # ── Policy training ──────────────────────────────────────────────────
        if rp_ready:

            # Reload latest rp checkpoint every N steps
            if step % cfg.training.rp_reload_interval == 0 and step > 0:
                latest = RewardPredictorEnsemble.latest_checkpoint(rp_ckpt_dir)
                if latest:
                    try:
                        rp.load(latest)
                    except Exception:
                        pass

            ep_reward, ep_loss = 0.0, 0.0
            for _ in range(cfg.training.policy_episodes_per_step):
                r, l = _train_policy_episode(
                    env, policy, optimizer, rp, cfg.policy.gamma, n_envs
                )
                ep_reward += r
                ep_loss += l

            # _train_policy_episode ran env.reset()/step() directly on the shared
            # env, so the SegmentCollector's cached obs is now stale.
            collector._current_obs = None

            n = cfg.training.policy_episodes_per_step
            avg_reward = ep_reward / n
            avg_loss   = ep_loss   / n

            writer.add_scalar("policy/avg_reward", avg_reward, step)
            writer.add_scalar("policy/avg_loss",   avg_loss,   step)

            # Periodic checkpoint
            if step % cfg.training.policy_save_interval == 0:
                path = Path(policy_ckpt_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(policy.state_dict(), path)
                print(
                    f"[policy] step={step}"
                    f"  avg_reward={avg_reward:.3f}"
                    f"  avg_loss={avg_loss:.4f}",
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

def _pref_worker(cfg_dict, seg_pipe, pref_pipe, stop_event, log_dir):
    """
    Subprocess responsible for labeling segment pairs.

    Reads (seg1, seg2) from seg_pipe, queries the configured oracle
    (DQN expert or human via terminal), and sends
    (seg1.frames, seg2.frames, pref) to pref_pipe.
    """
    # Re-open stdin so that HumanPrefInterface can call input() in a
    # subprocess started with the "spawn" method.
    sys.stdin = os.fdopen(0)

    cfg = OmegaConf.create(cfg_dict)

    if cfg.preferences.oracle == "expert":
        env, expert_model = build_env_and_expert(cfg)
        env.close()      # env only needed to load the DQN; Q-net is standalone
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

    while not stop_event.is_set():
        try:
            seg1, seg2 = seg_pipe.get(timeout=2)
        except Exception:
            continue

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

    # ── Communication ────────────────────────────────────────────────────────
    seg_pipe    = Queue(maxsize=cfg.preferences.seg_pipe_maxsize)
    pref_pipe   = Queue()
    start_event = mp.Event()   # main → policy: rp checkpoint is ready
    stop_event  = mp.Event()   # main → all:    time to shut down

    # ── Preference databases (owned by main process) ─────────────────────────
    db_train = PrefDB(maxlen=cfg.preferences.db_train_maxlen)
    db_val   = PrefDB(maxlen=cfg.preferences.db_val_maxlen)
    buf = PrefBuffer(db_train, db_val, log_dir=str(run_dir / "pref_buffer"))
    buf.start_recv_thread(pref_pipe)

    # ── Reward predictor (trained in main process) ───────────────────────────
    # Use a single-env temporarily to discover obs_dim without spawning 4 SUMO
    # instances in the main process.
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
        args=(cfg_dict, seg_pipe, start_event, stop_event, rp_ckpt_dir, policy_path,
              str(run_dir)),
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

    # ── Phase 1: collect initial preferences ─────────────────────────────────
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

    # ── Phase 2: pretrain reward predictor ───────────────────────────────────
    print("\n[Phase 2] Pretraining reward predictor…")
    rp.train(t_db, v_db, val_interval=cfg.reward_predictor.val_interval)
    rp.save()
    start_event.set()
    print("  Reward predictor ready — policy training unlocked.")

    # ── Phase 3: continuous reward predictor retraining ──────────────────────
    print(f"\n[Phase 3] Reward predictor loop — {cfg.training.rp_iterations} iterations")

    with tqdm(total=cfg.training.rp_iterations, desc="rp iterations", unit="iter", ncols=80) as pbar:
        for i in range(1, cfg.training.rp_iterations + 1):
            time.sleep(cfg.training.rp_train_interval_sec)

            t_db, v_db = buf.get_dbs()
            if len(t_db) == 0 or len(v_db) == 0:
                pbar.set_postfix({"status": "waiting for data"})
                pbar.update(1)
                continue

            rp.train(t_db, v_db, val_interval=cfg.reward_predictor.val_interval)
            rp.save()
            pbar.set_postfix({"train": len(t_db), "val": len(v_db)})
            pbar.update(1)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _shutdown()


if __name__ == "__main__":
    main()
