"""
Christiano et al. (2017) — Learning from Human Preferences
Adapted for the SUMO highway environment with a DQN expert as preference oracle.

Pipeline:
  Phase 1 — collect `initial_prefs` trajectory-segment pairs and label them
             with the expert Q-network.
  Phase 2 — pretrain the RewardPredictorEnsemble on the collected PrefDB.
  Phase 3 — iteratively:
               (a) collect more preferences and add to PrefDB,
               (b) retrain the reward predictor,
               (c) run REINFORCE policy training using the learned reward.
"""

import functools

import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from learning_from_human_preferences.preferences.pref_db import PrefDB
from learning_from_human_preferences.reward_model.reward_predictor import (
    RewardPredictorEnsemble,
)

from human_feedback_rl.christiano.reward_network import SumoRewardNetwork
from human_feedback_rl.christiano.expert_pref_interface import ExpertPrefInterface
from human_feedback_rl.christiano.segment_collector import SegmentCollector
from human_feedback_rl.agents.policy_network import AgentPolicyNetwork
from human_feedback_rl.utils.env_setup import build_env_and_expert


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_preferences(collector, policy, pref_interface, db_train, db_val, n: int):
    """Collect `n` preferences and split 80/20 into train/val PrefDB."""
    collected = 0
    for _ in range(n):
        seg1, _ = collector.collect_segment(policy)
        seg2, _ = collector.collect_segment(policy)

        pref = pref_interface.ask_user(seg1, seg2)
        if pref is None:
            continue

        if np.random.random() < 0.8:
            db_train.append(seg1.frames, seg2.frames, pref)
        else:
            db_val.append(seg1.frames, seg2.frames, pref)

        collected += 1

    return collected


def _policy_episode(env, policy, rp) -> tuple[float, list]:
    """
    Run one episode with `policy`, compute per-step rewards from `rp`, and
    return (total_predicted_reward, list_of_(log_prob, predicted_reward) pairs).

    Assumes env is an SB3 VecEnv: reset() returns obs directly, step() takes
    an array of actions and returns (obs, rewards, dones, infos).
    """
    obs = env.reset()                        # SB3 VecEnv: no info tuple
    obs = np.asarray(obs)
    n_envs = obs.shape[0] if obs.ndim > 1 else 1
    obs = obs[0] if obs.ndim > 1 else obs    # follow env 0

    done = False
    log_probs = []
    obs_list = []

    while not done:
        state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(state)
        probs_dist = torch.distributions.Categorical(torch.softmax(logits, dim=1))
        action = probs_dist.sample()

        log_probs.append(probs_dist.log_prob(action))
        obs_list.append(obs.copy())

        # SB3 VecEnv: pass array of actions, returns 4-tuple
        actions = np.array([action.item()] * n_envs)
        next_obs, _, dones, _ = env.step(actions)
        done = bool(dones[0]) if hasattr(dones, '__len__') else bool(dones)

        next_obs = np.asarray(next_obs)
        obs = next_obs[0] if next_obs.ndim > 1 else next_obs

    obs_batch = np.stack(obs_list, axis=0)       # (T, obs_dim)
    rewards_np = rp.reward(obs_batch)            # (T,) — normalised by rp

    # Guard against NaN/Inf from an under-trained predictor
    if not np.all(np.isfinite(rewards_np)):
        rewards_np = np.zeros_like(rewards_np)

    # Clip to avoid exploding returns
    rewards_np = np.clip(rewards_np, -10.0, 10.0)

    return rewards_np.sum().item(), list(zip(log_probs, rewards_np.tolist()))


def _train_policy_episode(env, policy, optimizer, rp, gamma: float) -> tuple[float, float]:
    """One REINFORCE episode using the learned reward predictor."""
    total_reward, transitions = _policy_episode(env, policy, rp)

    # Compute discounted returns
    returns = []
    G = 0.0
    for _, r in reversed(transitions):
        G = r + gamma * G
        returns.insert(0, G)

    returns_t = torch.tensor(returns, dtype=torch.float32)

    # Normalise only when there is more than one step (std is NaN for n=1)
    if returns_t.numel() > 1:
        std = returns_t.std(correction=0)  # population std, never NaN
        returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)

    loss = sum(
        -lp * G_t for (lp, _), G_t in zip(transitions, returns_t)
    ) / len(transitions)

    # Skip update if loss is NaN (e.g. reward predictor not yet stable)
    if not torch.isfinite(loss):
        return total_reward, 0.0

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()

    return total_reward, loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="train_christiano.yaml")
def main(cfg: DictConfig):

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # ── Setup ──────────────────────────────────────────────────────────────
    env, expert_model = build_env_and_expert(cfg)

    obs_dim = env.observation_space.shape[-1]

    # Reward predictor ensemble (Christiano et al., n_preds=3 by default)
    rp = RewardPredictorEnsemble(
        core_network=functools.partial(SumoRewardNetwork, obs_dim=obs_dim),
        lr=cfg.reward_predictor.lr,
        n_preds=cfg.reward_predictor.n_preds,
        log_dir=str(run_dir),
        device=cfg.resources.device,
    )

    # Preference databases
    db_train = PrefDB(maxlen=cfg.preferences.db_train_maxlen)
    db_val = PrefDB(maxlen=cfg.preferences.db_val_maxlen)

    # Policy (REINFORCE with entropy regularisation)
    n_actions = env.action_space.n
    policy = AgentPolicyNetwork(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.policy.lr)

    # Segment collector (continuous stream over the same environment)
    collector = SegmentCollector(env, segment_len=cfg.preferences.segment_len)

    # Expert preference interface (uses DQN Q-values, no human in the loop)
    pref_interface = ExpertPrefInterface(
        expert_model=expert_model,
        max_segs=cfg.preferences.max_segs,
        log_dir=str(run_dir / "pref_interface"),
    )

    # ── Phase 1: initial preference collection ──────────────────────────────
    print(f"\n[Phase 1] Collecting {cfg.preferences.initial_prefs} initial preferences…")
    collected = _collect_preferences(
        collector, policy, pref_interface,
        db_train, db_val,
        n=cfg.preferences.initial_prefs,
    )
    print(f"  Done — train={len(db_train)}, val={len(db_val)}")

    # ── Phase 2: pretrain reward predictor ──────────────────────────────────
    print("\n[Phase 2] Pretraining reward predictor…")
    rp.train(db_train, db_val, val_interval=cfg.reward_predictor.val_interval)
    rp.save()
    print("  Reward predictor pretrained and saved.")

    # ── Phase 3: interleaved policy + reward predictor training ────────────
    print(f"\n[Phase 3] Main loop — {cfg.training.iterations} iterations")

    for iteration in range(1, cfg.training.iterations + 1):

        # (a) collect new preferences
        new_prefs = _collect_preferences(
            collector, policy, pref_interface,
            db_train, db_val,
            n=cfg.preferences.prefs_per_iter,
        )

        # (b) retrain reward predictor
        rp.train(db_train, db_val, val_interval=cfg.reward_predictor.val_interval)
        rp.save()

        # (c) policy training
        total_reward = 0.0
        total_loss = 0.0
        n_ep = cfg.training.policy_episodes_per_iter

        for _ in range(n_ep):
            ep_reward, ep_loss = _train_policy_episode(
                env, policy, optimizer, rp, gamma=cfg.policy.gamma
            )
            total_reward += ep_reward
            total_loss += ep_loss

        print(
            f"  Iter {iteration:3d}/{cfg.training.iterations}"
            f" | new_prefs={new_prefs}"
            f" | db_train={len(db_train)}, db_val={len(db_val)}"
            f" | avg_pred_reward={total_reward / n_ep:.3f}"
            f" | avg_loss={total_loss / n_ep:.4f}"
        )

    # ── Save policy ─────────────────────────────────────────────────────────
    policy_path = run_dir / "models" / "policy_christiano.pt"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), policy_path)
    print(f"\nPolicy saved to {policy_path}")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
