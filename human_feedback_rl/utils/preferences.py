import torch
from human_feedback_rl.core import Step


def collect_preferences(env, policy, pref_expert, dataset, episodes=200):

    for _ in range(episodes):

        obs, _ = env.reset()
        done = False

        while not done:

            state = torch.tensor(obs).float()

            n_actions = env.action_space.n

            a0 = torch.randint(0, n_actions, (1,)).item()
            a1 = torch.randint(0, n_actions, (1,)).item()

            step0 = Step(state, a0)
            step1 = Step(state, a1)

            probs = pref_expert.query([step0, step1])

            dataset.add(step0, step1, probs)

            logits = policy(state.unsqueeze(0))

            action = torch.argmax(logits, dim=1).item()

            obs, _, terminated, truncated, _ = env.step(action)

            done = terminated or truncated