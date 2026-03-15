"""
Visualise a policy trained with train_christiano.py in the SUMO GUI.

Usage:
    python scripts/play.py \
        agent.model=experiments/christiano/2026-03-15/HH-MM-SS/models/policy_christiano.pt \
        run.dir=experiments/christiano/2026-03-15/HH-MM-SS

Optional overrides:
    eval.episodes=5          number of episodes to play (default: infinite loop)
    play.step_delay=0.0      seconds to wait between steps (0 = as fast as possible)
    play.interactive=false   if true, press Enter before each step (like play.py in sumo-rl-ego)
"""

import time

import hydra
import torch
import traci
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import sumo_rl_ego as sre
from human_feedback_rl.agents.policy_network import AgentPolicyNetwork


@hydra.main(version_base=None, config_path="../configs", config_name="play.yaml")
def main(cfg: DictConfig):

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # ── Reproduce training environment ───────────────────────────────────────
    run_dir    = PROJECT_ROOT / cfg.run.dir
    train_cfg  = OmegaConf.load(run_dir / "config" / "config.yaml")

    print(f"[play] Scenario : {train_cfg.env.scenario}")
    print(f"[play] Policy   : {cfg.agent.model}")

    env = sre.make_env(train_cfg.env.scenario, seed=train_cfg.seed, use_gui=True)

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = AgentPolicyNetwork(obs_dim, n_actions)
    agent_path = PROJECT_ROOT / cfg.agent.model
    policy.load_state_dict(torch.load(agent_path, map_location="cpu"))
    policy.eval()

    # ── GUI setup ─────────────────────────────────────────────────────────────
    obs, _ = env.reset()
    traci.gui.setSchema("View #0", "real world")
    traci.gui.trackVehicle("View #0", "ego")
    traci.gui.setZoom("View #0", 1000)

    max_episodes = cfg.eval.get("episodes", None)   # None = run forever
    episode = 0

    print("\nSUMO GUI aperta. Ctrl+C per uscire.\n")

    # ── Rollout loop ─────────────────────────────────────────────────────────
    while max_episodes is None or episode < max_episodes:

        terminated = truncated = False
        ep_reward  = 0.0
        step       = 0

        while not (terminated or truncated):

            if cfg.play.interactive:
                input(f"  [ep={episode} step={step}] Press Enter to step…")

            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy(state)
            action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step      += 1

            if cfg.play.step_delay > 0:
                time.sleep(cfg.play.step_delay)

        status = info.get("status", "?")
        event  = info.get("event",  "?")
        print(f"[ep={episode}] steps={step}  reward={ep_reward:.2f}"
              f"  status={status}  event={event}")

        episode += 1
        obs, _ = env.reset()

        # Re-centre camera after reset (vehicle ID changes on new episode)
        try:
            traci.gui.trackVehicle("View #0", "ego")
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()
