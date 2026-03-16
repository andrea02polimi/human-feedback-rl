"""
Evaluate a policy trained with train_christiano.py.

Usage:
    python scripts/eval.py \
        run.dir=experiments/christiano/2026-03-15/16-45-23 \
        agent.model=experiments/christiano/2026-03-15/16-45-23/models/policy_christiano

    # override number of evaluation episodes
    python scripts/eval.py run.dir=... eval.episodes=100
"""

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import sumo_rl_ego as sre
from stable_baselines3 import A2C as SB3A2C


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(log: dict):
    """Pretty-print the environment's metric groups."""
    groups: dict = {}
    for key, value in log.items():
        group, name = key.split("/", 1)
        groups.setdefault(group, {})[name] = value

    for group, items in groups.items():
        print(f"[{group}]")
        for name, value in items.items():
            if isinstance(value, float):
                value = round(value, 4)
            print(f"  {name:26s}: {value}")
        print()


def _run_episodes(env, policy, episodes: int):
    """Roll out `policy` for `episodes` episodes (single gymnasium env)."""
    print(f"\n=== Evaluating — {episodes} episodes ===")

    for _ in tqdm(range(episodes)):
        obs, _ = env.reset()          # gymnasium API: (obs, info)
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # PROJECT_ROOT = human-feedback-rl/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # ── Load training config to reproduce the exact environment ──────────────
    run_dir = PROJECT_ROOT / cfg.run.dir
    train_cfg = OmegaConf.load(run_dir / "config" / "config.yaml")

    print(f"[eval] Training run : {run_dir}")
    print(f"[eval] Scenario     : {train_cfg.env.scenario}")

    # ── Build single (non-vectorized) environment ─────────────────────────────
    env = sre.make_env(train_cfg.env.scenario, seed=train_cfg.seed)

    # ── Load policy (SB3 A2C .zip format) ────────────────────────────────────
    agent_path = PROJECT_ROOT / cfg.agent.model
    print(f"[eval] Policy path  : {agent_path}")

    policy = SB3A2C.load(str(agent_path), device="cpu")

    # ── Run evaluation ────────────────────────────────────────────────────────
    _run_episodes(env, policy, cfg.eval.episodes)

    # ── Print environment metrics ─────────────────────────────────────────────
    log = env.metrics_tracker.get_log_metrics()
    _print_metrics(log)

    env.close()


if __name__ == "__main__":
    main()
