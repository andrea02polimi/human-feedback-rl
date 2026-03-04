import gymnasium as gym
from stable_baselines3 import DQN

from src.concrete_experts.Concrete_demonstration_expert import (
    ConcreteStepDemonstrationExpert,
)
from src.Core import Step


def main():
    env = gym.make("SumoEnv-v0")  # or wrapper
    expert_model = DQN.load("expert_policies/")

    expert = ConcreteStepDemonstrationExpert(env, expert_model)

    agent = DQN("MlpPolicy", env, verbose=1)

    state, _ = env.reset()
    for _ in range(10000):
        action, _ = agent.predict(state)

        step = Step(state, action)
        feedback = expert.query(step)
        expert_action = feedback.action

        print("Agent:", action, "Expert:", expert_action)

        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.learn(total_timesteps=1, reset_num_timesteps=False)
        state = next_state

        if terminated or truncated:
            state, _ = env.reset()


if __name__ == "__main__":
    main()