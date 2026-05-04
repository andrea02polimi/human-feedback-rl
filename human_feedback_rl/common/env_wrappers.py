import numpy as np
from typing import List, Tuple, Any, Dict, Callable, Union, Optional

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from .reward_nets import RewardEnsemble
from .types import Trajectory, Transition

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
        return float(np.sqrt(self.var / (self.count - 1))) or 1.0


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

    def __init__(self, venv: VecEnv, reward_model: RewardEnsemble):
        super().__init__(venv)
        self.reward_model = reward_model
        self._obs: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._debug_step = 0

        self.rms = _RunningMeanStd()

    def reset(self):
        obs = self.venv.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, true_rew, dones, infos = self.venv.step_wait()

        predicted_rew = self.reward_model.predict(self._obs, self._actions)

        self.rms.update(predicted_rew)

        # Dividiamo per la std, ma NON sottraiamo la media (self.rms.mean).
        # Sottrarre una costante continua dalla reward cambia l'MDP penalizzando
        # artificiosamente la sopravvivenza o le azioni.
        normalized_rew = predicted_rew / (self.rms.std + 1e-8)

        self._debug_step += 1
        if self._debug_step % 500 == 0:
            print(
                f"[DEBUG EnvRew] step={self._debug_step:6d} | "
                f"raw=[{predicted_rew.min():.3f}, {predicted_rew.max():.3f}] mean={predicted_rew.mean():.3f} | "
                f"ensemble_std={predicted_rew.std():.3f} | "
            )

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, normalized_rew, dones, infos
    



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
                observation=self._last_obs[i],      # o_t
                action=actions[i],          # a_t
                true_reward=float(true_rew[i]),   # r_t
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
    - policy casuale con probabilità `random_prob`
    - policy vera con probabilità `1 - random_prob`

    Limitazione:
    supporta solo policy stateless, cioè senza stato ricorrente.
    """

    def __init__(
        self,
        policy: Callable,
        venv: VecEnv,
        random_prob: float,
        rng: np.random.Generator,
    ):
        """
        Args:
            policy:
                La policy da wrappare. Deve essere callable e restituire:
                (actions, state), dove state deve essere None.
            venv:
                Ambiente vettorizzato, usato per campionare azioni casuali.
            random_prob:
                Probabilità di scegliere la policy casuale quando avviene uno switch.
            rng:
                Generatore random usato per tutte le scelte casuali.
        """
        self.wrapped_policy = policy
        self.venv = venv
        self.random_prob = random_prob
        self.rng = rng

        # Seed dell'action space, così anche il sampling casuale dipende da rng
        seed = int(self.rng.integers(0, 2**31 - 1))
        self.venv.action_space.seed(seed)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.random_prob:
            num_envs = len(observation)
            actions = [self.venv.action_space.sample() for _ in range(num_envs)]
            actions = np.stack(actions, axis=0)
            return actions
        return self.wrapped_policy.predict(observation)