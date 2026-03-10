from pathlib import Path

import torch
from tqdm import tqdm

from human_feedback_rl.agent_policy_network import AgentPolicyNetwork
from human_feedback_rl.utils.env_setup import build_env_and_expert


ROOT = Path(__file__).resolve().parents[2]

EXPERT_MODEL = ROOT / "2026-03-09_17-15-41_train_highway_fast_DQN/model"
AGENT_MODEL = ROOT / "models/policy_final.pt"

EPISODES = 20


def evaluate_agent(env, policy, episodes):

    print("\n=== Evaluating trained agent ===")

    pbar = tqdm(total=episodes, desc="Agent Episodes")

    for _ in range(episodes):

        obs, _ = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:

            state = obs[0] if hasattr(obs, "shape") and len(obs.shape) > 1 else obs

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                logits = policy(state_tensor)

            action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)

        pbar.update(1)

    pbar.close()

    print("\nAgent metrics:")
    env.metrics_tracker.print_log_metrics()


def evaluate_expert(env, expert_model, episodes):

    print("\n=== Evaluating expert ===")

    pbar = tqdm(total=episodes, desc="Expert Episodes")

    for _ in range(episodes):

        obs, _ = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:

            action, _ = expert_model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

        pbar.update(1)

    pbar.close()

    print("\nExpert metrics:")
    env.metrics_tracker.print_log_metrics()


def main():

    # -------------------------
    # Build env to get spaces
    # -------------------------
    env_tmp, expert_model = build_env_and_expert(EXPERT_MODEL)

    obs_dim = env_tmp.observation_space.shape[0]
    n_actions = env_tmp.action_space.n

    env_tmp.close()

    # -------------------------
    # Load trained agent
    # -------------------------
    policy = AgentPolicyNetwork(obs_dim, n_actions)

    policy.load_state_dict(
        torch.load(AGENT_MODEL, map_location="cpu")
    )

    policy.eval()

    # -------------------------
    # Evaluate agent
    # -------------------------
    env_agent, _ = build_env_and_expert(EXPERT_MODEL)

    evaluate_agent(env_agent, policy, EPISODES)

    env_agent.close()

    # -------------------------
    # Evaluate expert
    # -------------------------
    env_expert, expert_model = build_env_and_expert(EXPERT_MODEL)

    evaluate_expert(env_expert, expert_model, EPISODES)

    env_expert.close()


if __name__ == "__main__":
    main()