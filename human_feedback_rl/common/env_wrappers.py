import numpy as np
from typing import List, Tuple, Any, Dict, Callable, Union, Optional

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from .reward_nets import RewardEnsemble
from .types import Trajectory, Transition

# order: arrived, collided, off_road, timeout, running, teleported, removed_unknown
_STATUS_ONEHOT = {
    "arrived":         np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "collided":        np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "offroad":         np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "timeout":         np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "running":         np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    "teleported":      np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
    "removed_unknown": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
}

def ego_status_to_onehot(status: str) -> np.ndarray:
    return _STATUS_ONEHOT.get(status, _STATUS_ONEHOT["running"])

class _RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        for x in values.flat:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.var += (x - self.mean) * delta  # M2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        s = float(np.sqrt(max(0.0, self.var / (self.count - 1))))
        return s if s > 0 else 1.0


# ---------------------------------------------------------------------------
# Environment reward wrapper
# ---------------------------------------------------------------------------

class EnvRewardWrapper(VecEnvWrapper):
    """
    VecEnvWrapper that replaces environment rewards with EnsembleRewardModel
    predictions. Uses the pre-step observation for reward prediction to align
    with how the reward predictor was trained on (obs_t, a_t) pairs.

    Rewards are normalized to mean 0 / std 1 via a running estimate before
    being passed to the agent (Christiano et al. 2017, Section 2.2).
    """

    def __init__(self, venv: VecEnv, reward_model: RewardEnsemble, policy=None):
        super().__init__(venv)
        self.reward_model = reward_model
        self.policy = policy
        self._obs: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._reward_stats = _RunningMeanStd()

        if hasattr(self.reward_model, "set_policy"):
            self.reward_model.set_policy(policy)

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
            done_signal = dones.astype(np.float32)
            predicted_rew = self.reward_model.predict(self._obs, self._actions, next_status, done_signal)
            self._reward_stats.update(predicted_rew)
            predicted_rew = (predicted_rew - self._reward_stats.mean) / (self._reward_stats.std + 1e-8)
        else:
            predicted_rew = np.zeros(len(obs), dtype=np.float32)

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, predicted_rew, dones, infos
    



class EnvBufferingWrapper(VecEnvWrapper):
    """
    Wrapper per VecEnv che salva le transizioni raccolte durante i rollout
    e le organizza in Trajectory.

    Usa le classi:
    - Transition
    - Trajectory
    - Fragment (= Trajectory)
    """

    def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
        super().__init__(venv)
        self.error_on_premature_reset = error_on_premature_reset

        self._initialized = False
        self._saved_actions = None

        # traiettorie complete terminate
        self._finished_trajectories: List[Trajectory] = []

        # frammenti/traiettorie correnti, una per env parallelo
        self._partial_trajectories: List[Trajectory] = []

        # contatore timestep per ogni env parallelo
        self._timesteps: np.ndarray | None = None

        # ultima osservazione vista per ogni env
        self._last_obs = None

    def is_empty(self):
        return len(self._finished_trajectories) == 0

    def step_async(self, actions):
        assert self._initialized, "Call reset() before stepping."
        assert self._saved_actions is None, "step_async called twice without step_wait."
        self._saved_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        """
        Esegue lo step e salva una Transition per ogni env parallelo.
        """
        assert self._initialized, "Call reset() before stepping."
        assert self._saved_actions is not None, "step_wait called before step_async."

        actions = self._saved_actions
        self._saved_actions = None

        obs, true_rew, dones, infos = self.venv.step_wait()

        self._timesteps += 1

        for i in range(self.num_envs):
            transition = Transition(
                observation=self._last_obs[i],
                action=actions[i],
                true_reward=float(true_rew[i]),
                next_observation=obs[i],
                next_status=ego_status_to_onehot(infos[i].get("ego_status", "running")),
                done=bool(dones[i]),
            )

            self._partial_trajectories[i].add_transition(transition)

            if dones[i]:
                # episodio finito: salva traiettoria completa
                self._finished_trajectories.append(self._partial_trajectories[i])

                # ricomincia una nuova traiettoria vuota per quell'env
                self._partial_trajectories[i] = Trajectory()
                self._timesteps[i] = 0

        self._last_obs = obs
        return obs, true_rew, dones, infos

    def pop_finished_trajectories(self) -> List[Trajectory]:
        """
        Restituisce solo le traiettorie complete terminate dall'ultimo pop.
        """

        if len(self._finished_trajectories) == 0:
            return []

        trajectories = self._finished_trajectories

        self._finished_trajectories = []

        return trajectories

    def reset(self, **kwargs):
        if (
            self._initialized
            and self.error_on_premature_reset
            and len(self._finished_trajectories) > 0
        ):
            raise RuntimeError("reset() chiamato prima di aver letto le traiettorie salvate.")

        self._initialized = True
        self._saved_actions = None

        obs = self.venv.reset(**kwargs)
        self._last_obs = obs

        self._timesteps = np.zeros(self.num_envs, dtype=int)
        self._partial_trajectories = [Trajectory() for _ in range(self.num_envs)]
        self._finished_trajectories = []

        return obs



class PolicyExplorationWrapper:
    """
    Wrapper che rende una policy più esplorativa.

    In ogni momento usa una delle due policy possibili:
    1. la policy vera (wrapped policy)
    2. una policy casuale che campiona azioni a caso

    Dopo ogni chiamata, con probabilità `switch_prob`,
    decide se cambiare policy corrente.
    Quando cambia, sceglie:
    - policy casuale con probabilità `exploration_eps`
    - policy vera con probabilità `1 - exploration_eps`

    Limitazione:
    supporta solo policy stateless, cioè senza stato ricorrente.
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
            policy:
                La policy da wrappare. Deve essere callable e restituire:
                (actions, state), dove state deve essere None.
            venv:
                Ambiente vettorizzato, usato per campionare azioni casuali.
            exploration_eps:
                Probabilità di scegliere la policy casuale quando avviene uno switch.
            rng:
                Generatore random usato per tutte le scelte casuali.
        """
        self.wrapped_policy = policy
        self.venv = venv
        self.exploration_eps = exploration_eps
        self.rng = rng

        # Seed dell'action space, così anche il sampling casuale dipende da rng
        seed = int(self.rng.integers(0, 2**31 - 1))
        self.venv.action_space.seed(seed)

    def predict(self, observation: np.ndarray, **kwargs) -> tuple:
        if self.rng.random() < self.exploration_eps:
            num_envs = len(observation)
            actions = np.stack([self.venv.action_space.sample() for _ in range(num_envs)])
            return actions, None
        return self.wrapped_policy.predict(observation, **kwargs)
