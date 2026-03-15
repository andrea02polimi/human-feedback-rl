import hydra
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import sumo_rl_ego as sre

from human_feedback_rl.agents.policy_network import AgentPolicyNetwork


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def print_log(log):

    groups = {}

    for key, value in log.items():

        group, name = key.split("/", 1)

        if group not in groups:
            groups[group] = {}

        groups[group][name] = value

    for group, items in groups.items():

        print(f"[{group}]")

        for name, value in items.items():

            if isinstance(value, float):
                value = round(value, 3)

            print(f"  {name:20s} : {value}")

        print()


# ------------------------------------------------------------
# Agent evaluation
# ------------------------------------------------------------

def evaluate_agent(env, policy, episodes):

    print("\n=== Agent evaluation ===")

    for _ in tqdm(range(episodes)):

        obs, _ = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):

            state_tensor = torch.tensor(
                obs,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                logits = policy(state_tensor)

            action = torch.argmax(logits, dim=1).item()

            obs, _, terminated, truncated, _ = env.step(action)


# ------------------------------------------------------------
# Hydra main
# ------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    PROJECT_ROOT = Path(__file__).resolve().parents[0]

    # ------------------------------------------------------------
    # Load training configuration
    # ------------------------------------------------------------

    run_dir = PROJECT_ROOT / cfg.run.dir

    config_path = run_dir / "config" / "config.yaml"

    print("[Eval] Loading run config from:", config_path)

    train_cfg = OmegaConf.load(config_path)

    # ------------------------------------------------------------
    # Build environment exactly like training
    # ------------------------------------------------------------

    env = sre.make_env(
        train_cfg.env.scenario,
        seed=train_cfg.seed
    )

    # ------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = AgentPolicyNetwork(obs_dim, n_actions)

    agent_path = PROJECT_ROOT / cfg.agent.model

    print("[Agent] Loading policy from:", agent_path)

    policy.load_state_dict(
        torch.load(agent_path, map_location="cpu")
    )

    policy.eval()

    # ------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------

    evaluate_agent(
        env,
        policy,
        cfg.eval.episodes
    )

    # ------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------

    log = env.metrics_tracker.get_log_metrics()

    print_log(log)

    env.close()


if __name__ == "__main__":
    main()