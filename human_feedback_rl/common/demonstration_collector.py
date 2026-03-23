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
    Produces expert rollout segments paired with agent segments.

    For each agent segment, the expert runs in its own environment (demo env)
    using the demo env's own observations to select actions — NOT the agent's
    observations. This produces coherent expert trajectories where every action
    is appropriate for the state the demo env is actually in.

    The demo env state is maintained across calls so the expert builds up
    continuous trajectories rather than restarting every segment.

    Args:
        segment_length: number of frames per segment (must match policy worker)
    """

    def __init__(self, segment_length: int):
        self._segment_length = segment_length
        self._current_obs = None  # demo env current obs, maintained across calls

    def create_expert_correction(
        self,
        agent_frames: list,
        expert_model,
        env,
    ) -> list:
        """
        Build a pure expert rollout segment of the same length as agent_frames.

        The expert acts based on the demo env's own observations at each step.
        agent_frames is used only to determine the segment length.

        Args:
            agent_frames: list of np.ndarray — used only for segment length
            expert_model: SB3 model with .predict()
            env:          1-env SB3 VecEnv used exclusively by the demo worker

        Returns:
            list of np.ndarray — expert rollout observations,
            length == segment_length (last frame repeated if episode ends early)
        """
        if self._current_obs is None:
            self._current_obs = np.asarray(env.reset(), dtype=np.float32)

        expert_frames = []
        T = len(agent_frames)

        for _ in range(T):
            action, _ = expert_model.predict(self._current_obs, deterministic=True)
            next_obs, _, dones, _ = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            expert_frames.append(next_obs[0].copy())
            self._current_obs = next_obs
            if dones[0]:
                self._current_obs = np.asarray(env.reset(), dtype=np.float32)

        # Pad to segment_length in case the episode ended early.
        while len(expert_frames) < self._segment_length:
            expert_frames.append(expert_frames[-1].copy())

        return expert_frames[: self._segment_length]
