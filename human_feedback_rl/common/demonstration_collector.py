"""
DemonstrationCollector — expert-corrected segment generation.

For each agent segment received, the expert policy is queried on each agent
observation to decide an action. That action is applied in the demo environment
and the resulting observation is collected. This produces an "expert correction"
segment where:
  - actions are chosen by the expert reacting to the agent's observed states
  - the resulting observations are what those expert actions lead to

The (expert_correction_frames, agent_frames) pair is sent to demo_pipe and
stored in DemoDatabase for the margin ranking loss:
    L_demo = mean( relu( margin − (Σr_expert_correction − Σr_agent) ) )
"""

import numpy as np


class DemonstrationCollector:
    """
    Produces expert-correction segments paired with agent segments.

    For each agent segment:
      1. For each observation in the segment, query the expert for an action.
      2. Step the demo environment with that action.
      3. Collect the resulting observation as the expert-correction frame.

    Args:
        segment_length: number of frames per segment (must match policy worker)
    """

    def __init__(self, segment_length: int):
        self._segment_length = segment_length

    def create_expert_correction(
        self,
        agent_frames: list,
        expert_model,
        env,
    ) -> list:
        """
        Build an expert-correction segment from an agent segment.

        For each agent observation, the expert decides an action and the demo
        environment is stepped. The resulting observations form the expert
        correction trajectory.

        Args:
            agent_frames: list of np.ndarray observations from the agent segment
            expert_model: SB3 model with .predict()
            env:          1-env SB3 VecEnv used exclusively by the demo worker

        Returns:
            list of np.ndarray — expert-correction observations,
            length == segment_length (last frame repeated if episode ends early)
        """
        expert_frames = []

        for agent_obs in agent_frames:
            # Expert decides action based on the agent's observation.
            obs_batch = np.asarray(agent_obs, dtype=np.float32)[np.newaxis]  # (1, obs_dim)
            action, _ = expert_model.predict(obs_batch, deterministic=True)

            # Step the demo env — the resulting observation is the expert frame.
            next_obs, _, dones, _ = env.step(action)
            next_obs = np.asarray(next_obs)
            expert_frames.append(next_obs[0].copy())

            if dones[0]:
                env.reset()

        # Pad to segment_length in case the episode ended early.
        while len(expert_frames) < self._segment_length:
            expert_frames.append(expert_frames[-1].copy())

        return expert_frames[: self._segment_length]
