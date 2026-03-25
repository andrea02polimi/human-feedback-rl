"""
DemonstrationCollector — expert-corrected action generation.

For each agent segment received, queries the expert policy on each agent
observation to produce the expert's preferred action at that state.

The result is an "expert correction" where:
  - observations are the agent's own observations (from its actual trajectory)
  - actions are what the expert would have chosen for those observations

This produces (agent_frames, expert_actions, agent_actions) triples used by:
  - DemoDatabase + margin ranking loss (use_demonstrations path):
        L_demo = mean( relu( margin − (RP(obs, a_expert) − RP(obs, a_agent)) ) )
  - PrefBuffer + preference loss (use_demo_preferences path):
        label (1.0, 0.0) encodes that expert actions are preferred
"""

import numpy as np


class DemonstrationCollector:
    """
    Produces expert-corrected action sequences for agent segments.

    No environment stepping is needed: the expert is queried frame-by-frame
    on the agent's observations to produce its preferred action at each state.
    """

    def create_expert_correction(self, agent_frames, expert_model):
        """
        For each observation in agent_frames, query the expert for its action.

        Args:
            agent_frames: list of np.ndarray — agent's observed states
            expert_model: SB3 model with .predict()

        Returns:
            agent_frames:   same list as input (observations are unchanged)
            expert_actions: list of int — expert's preferred action per frame
        """
        expert_actions = []
        for frame in agent_frames:
            obs = np.asarray(frame, dtype=np.float32)[np.newaxis]  # (1, obs_dim)
            action, _ = expert_model.predict(obs, deterministic=True)
            raw = action[0]
            a   = np.asarray(raw)
            # Discrete (DQN): action[0] is a scalar int → store as int.
            # Continuous (PPO): action[0] is a float array → store as float32 array.
            expert_actions.append(int(raw) if a.ndim == 0 else a.copy().astype(np.float32))
        return list(agent_frames), expert_actions
