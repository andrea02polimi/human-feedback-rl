import numpy as np
from typing import Callable, List

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from .reward_nets import RewardEnsemble
from .status import ego_status_to_onehot
from .types import Trajectory, Transition


# ---------------------------------------------------------------------------
# Environment reward wrapper
# ---------------------------------------------------------------------------

class EnvRewardWrapper(VecEnvWrapper):
    """
    VecEnvWrapper that replaces environment rewards with EnsembleRewardModel
    predictions. Uses the pre-step observation for reward prediction to align
    with how the reward predictor was trained on (obs_t, a_t) pairs.

    Agent-only reward transformations are owned by ``reward_model.predict``;
    reward-model training uses ``forward`` and remains unaffected.
    """

    def __init__(self, venv: VecEnv, reward_model: RewardEnsemble):
        super().__init__(venv)
        self.reward_model = reward_model
        self._obs: np.ndarray | None = None
        self._actions: np.ndarray | None = None

    def reset(self):
        obs = self.venv.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, true_rew, dones, infos = self.venv.step_wait()

        if self._obs is not None and self._actions is not None:
            next_status = np.array([ego_status_to_onehot(i.get("ego_status", "running")) for i in infos])
            predicted_rew = self.reward_model.predict(self._obs, self._actions, next_status, dones.astype(np.float32))
        else:
            predicted_rew = np.zeros(len(obs), dtype=np.float32)

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, predicted_rew, dones, infos

class EnvBufferingWrapper(VecEnvWrapper):
    """VecEnvWrapper that records rollout transitions and groups them into Trajectory objects."""

    def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
        super().__init__(venv)
        self.error_on_premature_reset = error_on_premature_reset

        self._initialized = False
        self._saved_actions = None
        self._saved_log_probs = None
        self._recording_mask = np.ones(self.num_envs, dtype=bool)

        # Completed (terminated) trajectories.
        self._finished_trajectories: List[Trajectory] = []

        # In-progress trajectory, one per parallel env.
        self._partial_trajectories: List[Trajectory] = []

        # Timestep counter per parallel env.
        self._timesteps: np.ndarray | None = None

        # Last observation seen per env.
        self._last_obs = None

    def is_empty(self):
        return len(self._finished_trajectories) == 0

    def step_async(self, actions):
        assert self._initialized, "Call reset() before stepping."
        assert self._saved_actions is None, "step_async called twice without step_wait."
        self._saved_actions = actions
        self.venv.step_async(actions)

    def set_log_probs(self, log_probs: np.ndarray) -> None:
        """Attach action log-probabilities to the next buffered transitions."""
        log_probs = np.asarray(log_probs, dtype=np.float32).reshape(-1)
        if len(log_probs) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} log-probs, got {len(log_probs)}.")
        self._saved_log_probs = log_probs

    def set_recording_mask(self, mask: np.ndarray) -> None:
        """Select which parallel environments are recorded on the next step."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != self.num_envs:
            raise ValueError(f"Expected a mask of length {self.num_envs}, got {len(mask)}.")
        self._recording_mask = mask

    def step_wait(self):
        """Step the env and record one Transition per parallel env."""
        assert self._initialized, "Call reset() before stepping."
        assert self._saved_actions is not None, "step_wait called before step_async."

        actions = self._saved_actions
        self._saved_actions = None
        log_probs = self._saved_log_probs
        self._saved_log_probs = None

        obs, true_rew, dones, infos = self.venv.step_wait()

        self._timesteps += self._recording_mask.astype(int)

        for i in range(self.num_envs):
            if not self._recording_mask[i]:
                continue
            transition = Transition(
                observation=self._last_obs[i],
                action=actions[i],
                true_reward=float(true_rew[i]),
                next_status=ego_status_to_onehot(infos[i].get("ego_status", "running")),
                done=bool(dones[i]),
                log_policy_prob=None if log_probs is None else float(log_probs[i]),
            )

            self._partial_trajectories[i].add_transition(transition)

            if dones[i]:
                # Episode finished: store the completed trajectory and start a new one.
                self._finished_trajectories.append(self._partial_trajectories[i])
                self._partial_trajectories[i] = Trajectory()
                self._timesteps[i] = 0

        self._last_obs = obs
        return obs, true_rew, dones, infos

    def pop_finished_trajectories(self) -> List[Trajectory]:
        """Return and clear the trajectories completed since the last pop."""
        trajectories = self._finished_trajectories
        self._finished_trajectories = []
        return trajectories

    def reset(self, **kwargs):
        """Riparte da capo, verificando che non ci sia nulla da perdere.

        Due cose andrebbero perse in silenzio, e la guardia copre entrambe:

        * traiettorie CONCLUSE non ancora lette da ``pop_finished_trajectories``;
        * un episodio IN CORSO, la cui parte gia' raccolta vive in
          ``_partial_trajectories``.

        Il secondo caso e' quello dell'ambiente condiviso: SAC lascia quasi
        sempre un episodio a meta', e chi legge il buffer ha appena svuotato le
        concluse, quindi una guardia sulle sole concluse non scatterebbe mai
        proprio quando servirebbe. Il percorso normale non resetta piu' a
        episodio aperto (``rollout_agent`` prosegue da ``start_obs``); questa
        e' la rete che impedisce a un percorso futuro di reintrodurre la
        perdita senza accorgersene.
        """
        if self._initialized and self.error_on_premature_reset:
            if len(self._finished_trajectories) > 0:
                raise RuntimeError(
                    "reset() called before the buffered trajectories were read."
                )
            aperti = [i for i, t in enumerate(self._partial_trajectories) if len(t) > 0]
            if aperti:
                lunghezze = [len(self._partial_trajectories[i]) for i in aperti]
                raise RuntimeError(
                    "reset() called while episodes are still in progress in envs "
                    f"{aperti} ({lunghezze} transitions would be discarded). "
                    "Continue the rollout from the current observation instead "
                    "(rollout_agent(..., start_obs=...))."
                )

        self._initialized = True
        self._saved_actions = None
        self._saved_log_probs = None
        self._recording_mask = np.ones(self.num_envs, dtype=bool)

        obs = self.venv.reset(**kwargs)
        self._last_obs = obs

        self._timesteps = np.zeros(self.num_envs, dtype=int)
        self._partial_trajectories = [Trajectory() for _ in range(self.num_envs)]
        self._finished_trajectories = []

        return obs



class PolicyExplorationWrapper:
    """Epsilon-greedy exploration wrapper around a policy.

    On each `predict` call, with probability `exploration_eps` it samples random
    actions from the env's action space, otherwise it defers to the wrapped policy.
    Only stateless policies are supported.
    """

    def __init__(
        self,
        venv: VecEnv,
        policy: Callable,
        exploration_eps: float,
        rng: np.random.Generator,
    ):
        """
        Args:
            venv: vectorized env, used to sample random actions.
            policy: wrapped policy; must be callable and return (actions, state).
            exploration_eps: probability of sampling a random action per call.
            rng: random generator driving all random choices.
        """
        self.wrapped_policy = policy
        self.venv = venv
        self.exploration_eps = exploration_eps
        self.rng = rng

        # Seed the action space so random sampling is also driven by rng.
        seed = int(self.rng.integers(0, 2**31 - 1))
        self.venv.action_space.seed(seed)

    def predict(self, observation: np.ndarray, **kwargs) -> tuple:
        if self.rng.random() < self.exploration_eps:
            num_envs = len(observation)
            actions = np.stack([self.venv.action_space.sample() for _ in range(num_envs)])
            return actions, None
        return self.wrapped_policy.predict(observation, **kwargs)

    def action_log_prob(self, observation: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Log-density under the epsilon mixture, independent of sampled branch."""
        from .trajectory_generators import policy_action_log_probs

        policy_log_prob = policy_action_log_probs(self.wrapped_policy, observation, actions)
        action_space = self.venv.action_space
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            uniform_log_prob = -float(np.log(action_space.high - action_space.low).sum())
        elif hasattr(action_space, "n"):
            uniform_log_prob = -float(np.log(action_space.n))
        else:
            raise TypeError(f"Unsupported exploration action space: {type(action_space).__name__}")

        eps = self.exploration_eps
        if eps <= 0:
            return policy_log_prob
        if eps >= 1:
            return np.full_like(policy_log_prob, uniform_log_prob)
        return np.logaddexp(
            np.log1p(-eps) + policy_log_prob,
            np.log(eps) + uniform_log_prob,
        )
