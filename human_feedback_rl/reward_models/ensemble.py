"""
Custom RewardPredictorEnsemble with combined preference + demonstration loss.

Drop-in replacement for learning_from_human_preferences.reward_model.reward_predictor.
Interface is intentionally identical: reward(), raw_rewards(), train(),
save(), load(), latest_checkpoint().

Training loss
─────────────
    L_total = L_pref + demo_weight × L_demo

    L_pref  — soft cross-entropy on (s1, s2, p1, p2) preference pairs.
              Uses the soft label (p1, p2) directly instead of argmax, which
              preserves the oracle's confidence (e.g. 0.7 / 0.3 vs 1 / 0).
              This is more faithful to Christiano et al. 2017 equation (1).

    L_demo  — margin ranking loss on (expert_seg, agent_seg) pairs:
                  mean( relu( margin − (Σr_expert − Σr_agent) ) )
              Gradient is zero once expert reward exceeds agent reward by
              `margin`, making training stable and scale-agnostic.
"""

import os
import os.path as osp
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from human_feedback_rl.utils.running_stats import RunningStat
from human_feedback_rl.utils.itertools import batch_iter


class RewardPredictorEnsemble:
    """
    Ensemble of reward predictors.

    Args:
        core_network  callable() → nn.Module  (e.g. functools.partial(SumoRewardNetwork, obs_dim=N))
        lr            Adam learning rate
        n_preds       ensemble size (paper: 3)
        log_dir       experiment dir for TensorBoard + checkpoints; None = inference-only
        device        torch device string
    """

    def __init__(
        self,
        core_network,
        lr: float = 1e-4,
        n_preds: int = 1,
        log_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device  = device
        self.n_preds = n_preds
        self.n_steps = 0
        self.r_norm  = RunningStat(shape=n_preds)   # running mean/std for reward normalisation

        self.models     = [core_network().to(device) for _ in range(n_preds)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]

        if log_dir is not None:
            self._log           = True
            self.checkpoint_dir = osp.join(log_dir, "reward_predictor_checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self._log           = False
            self.checkpoint_dir = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def raw_rewards(self, obs: np.ndarray) -> np.ndarray:
        """
        Per-member reward for each observation (no normalisation).
        obs: (N, obs_dim)  →  (n_preds, N)
        Used by disagreement-based pair selection.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        rs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                rs.append(model(obs_t).cpu().numpy())
        return np.array(rs)   # (n_preds, N)

    def reward(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalised mean ensemble reward (running z-score, scaled × 0.05).
        obs: (N, obs_dim)  →  (N,)
        """
        ensemble_rs = self.raw_rewards(obs)   # (n_preds, N)
        ensemble_rs = ensemble_rs.T           # (N, n_preds)
        for step_reward in ensemble_rs:
            self.r_norm.push(step_reward)
        ensemble_rs -= self.r_norm.mean
        ensemble_rs /= (self.r_norm.std + 1e-12)
        ensemble_rs *= 0.05
        return np.mean(ensemble_rs.T, axis=0)   # (N,)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        pref_db,
        val_db,
        demo_db=None,
        val_interval: int  = 50,
        demo_weight: float = 1.0,
        demo_margin: float = 1.0,
    ) -> None:
        """
        One pass over pref_db (and optionally demo_db).

        Args:
            pref_db      PrefDB — labeled preference pairs (train split)
            val_db       PrefDB — preference pairs (validation split)
            demo_db      PrefDB or None — expert-vs-agent demonstration pairs
            val_interval log validation metrics every N gradient steps
            demo_weight  weight of L_demo relative to L_pref
            demo_margin  margin for the ranking loss (reward units)
        """
        if len(pref_db) == 0:
            return

        # Pre-fetch demo pairs once per training pass for efficient sampling.
        demo_pairs = self._extract_pairs(demo_db) if (demo_db is not None and len(demo_db) > 0) else []
        random.shuffle(demo_pairs)
        demo_idx = [0]   # list so the inner closure can mutate it

        def _next_demo_batch(size: int):
            """Return `size` demo pairs, cycling through the shuffled list."""
            if not demo_pairs:
                return []
            out = []
            for _ in range(size):
                if demo_idx[0] >= len(demo_pairs):
                    random.shuffle(demo_pairs)
                    demo_idx[0] = 0
                out.append(demo_pairs[demo_idx[0]])
                demo_idx[0] += 1
            return out

        for batch in batch_iter(pref_db.preferences, 32, shuffle=True):

            # Same demo batch shared across all ensemble members for this step.
            demo_batch = _next_demo_batch(len(batch))

            for model, optimizer in zip(self.models, self.optimizers):
                model.train()
                optimizer.zero_grad()

                # ── Preference loss (soft cross-entropy) ──────────────────────
                s1s   = torch.tensor(
                    [pref_db.segments[k1] for k1, k2, _ in batch],
                    dtype=torch.float32,
                ).to(self.device)   # (B, T, obs_dim)
                s2s   = torch.tensor(
                    [pref_db.segments[k2] for k1, k2, _ in batch],
                    dtype=torch.float32,
                ).to(self.device)
                prefs = torch.tensor(
                    [p for _, _, p in batch],
                    dtype=torch.float32,
                ).to(self.device)   # (B, 2)

                B, T = s1s.shape[:2]
                r1 = model(s1s.view(B * T, -1)).view(B, T).sum(dim=1)   # (B,)
                r2 = model(s2s.view(B * T, -1)).view(B, T).sum(dim=1)   # (B,)

                # Soft cross-entropy: H(p, softmax([r1, r2]))
                log_probs = F.log_softmax(torch.stack([r1, r2], dim=1), dim=1)   # (B, 2)
                L_pref    = -(prefs * log_probs).sum(dim=1).mean()

                # ── Demo margin ranking loss ───────────────────────────────────
                L_demo = torch.tensor(0.0, device=self.device)
                if demo_batch:
                    exp_t = torch.tensor(
                        [e for e, a, _ in demo_batch],
                        dtype=torch.float32,
                    ).to(self.device)   # (B, T, obs_dim)
                    ag_t  = torch.tensor(
                        [a for e, a, _ in demo_batch],
                        dtype=torch.float32,
                    ).to(self.device)

                    Bd, Td = exp_t.shape[:2]
                    r_exp = model(exp_t.view(Bd * Td, -1)).view(Bd, Td).sum(dim=1)   # (B,)
                    r_ag  = model(ag_t.view(Bd * Td, -1)).view(Bd, Td).sum(dim=1)    # (B,)
                    L_demo = F.relu(demo_margin - (r_exp - r_ag)).mean()

                loss = L_pref + demo_weight * L_demo
                loss.backward()
                optimizer.step()

            self.n_steps += 1

            if val_interval > 0 and self.n_steps % val_interval == 0:
                self._val_step(val_db)

                if self._log and wandb.run is not None:
                    log_dict = {"rp/train/loss": loss.item(), "rp_step": self.n_steps}
                    if demo_batch:
                        log_dict["rp/train/demo_margin_loss"] = L_demo.item()
                    wandb.log(log_dict)

    def _extract_pairs(self, db) -> list:
        """Return [(seg1_frames, seg2_frames, pref), ...] from a PrefDB."""
        return [(db.segments[k1], db.segments[k2], p) for k1, k2, p in db.preferences]

    def _val_step(self, val_db) -> None:
        if len(val_db) == 0:
            return
        batch_size = min(32, len(val_db.preferences))
        idxs  = np.random.choice(len(val_db.preferences), batch_size, replace=False)
        batch = [val_db.preferences[i] for i in idxs]

        total_loss = 0.0
        total_acc  = 0.0
        for model in self.models:
            model.eval()
            with torch.no_grad():
                s1s   = torch.tensor(
                    [val_db.segments[k1] for k1, k2, _ in batch], dtype=torch.float32
                ).to(self.device)
                s2s   = torch.tensor(
                    [val_db.segments[k2] for k1, k2, _ in batch], dtype=torch.float32
                ).to(self.device)
                prefs = torch.tensor(
                    [p for _, _, p in batch], dtype=torch.float32
                ).to(self.device)

                B, T = s1s.shape[:2]
                r1 = model(s1s.view(B * T, -1)).view(B, T).sum(dim=1)
                r2 = model(s2s.view(B * T, -1)).view(B, T).sum(dim=1)

                log_probs  = F.log_softmax(torch.stack([r1, r2], dim=1), dim=1)
                loss       = -(prefs * log_probs).sum(dim=1).mean()
                acc        = (torch.stack([r1, r2], dim=1).argmax(dim=1) == prefs.argmax(dim=1)).float().mean()

                total_loss += loss.item()
                total_acc  += acc.item()

        if self._log and wandb.run is not None:
            wandb.log({
                "rp/val/loss":     total_loss / self.n_preds,
                "rp/val/accuracy": total_acc  / self.n_preds,
                "rp_step":         self.n_steps,
            })

    # ── Checkpointing ──────────────────────────────────────────────────────────

    def save(self) -> str:
        if self.checkpoint_dir is None:
            raise RuntimeError("Cannot save: log_dir was not set")
        path = osp.join(self.checkpoint_dir, f"reward_predictor_{self.n_steps}.pt")
        torch.save({
            "step":   self.n_steps,
            "models": [m.state_dict() for m in self.models],
        }, path)
        print(f"Saved reward predictor checkpoint to {path}")
        return path

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=True, map_location=self.device)
        for m, s in zip(self.models, state["models"]):
            m.load_state_dict(s)
        self.n_steps = state["step"]
        print(f"Loaded reward predictor checkpoint from {path}")

    @staticmethod
    def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
        if not osp.exists(ckpt_dir):
            return None
        files = [
            f for f in os.listdir(ckpt_dir)
            if f.startswith("reward_predictor_") and f.endswith(".pt")
        ]
        if not files:
            return None
        files.sort(key=lambda f: int(f[len("reward_predictor_"):-len(".pt")]))
        return osp.join(ckpt_dir, files[-1])
