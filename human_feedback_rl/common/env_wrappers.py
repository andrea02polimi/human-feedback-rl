import numpy as np
from typing import List, Tuple, Any, Dict, Callable, Union, Optional

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from .reward_nets import RewardEnsemble
from . import types

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
        self._reward_stats = _RunningMeanStd()

    def reset_stats(self):
        if self._reward_stats.count > 1:
            new_count = self._reward_stats.count // 2
            self._reward_stats.var = self._reward_stats.var * (new_count - 1) / (self._reward_stats.count - 1)
            self._reward_stats.count = new_count

    def reset(self):
        obs = self.venv.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _env_rewards, dones, infos = self.venv.step_wait()

        if self._obs is not None and self._actions is not None:
            rewards = self.reward_model.predict(self._obs, self._actions)
            self._reward_stats.update(rewards)
            rewards = ((rewards - self._reward_stats.mean) / self._reward_stats.std).astype(np.float32)
        else:
            rewards = np.zeros(len(obs), dtype=np.float32)

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, rewards, dones, infos
    



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
        self._finished_trajectories: List[types.Trajectory] = []

        # frammenti/traiettorie correnti, una per env parallelo
        self._partial_trajectories: List[types.Trajectory] = []

        # lunghezze episodi terminati
        self._ep_lens: List[int] = []

        # contatore timestep per ogni env parallelo
        self._timesteps: np.ndarray | None = None

        # numero totale di transizioni accumulate dall'ultimo pop
        self.n_transitions: int = 0

        # ultima osservazione vista per ogni env
        self._last_obs = None

    def reset(self, **kwargs):
        """
        Resetta l'ambiente e inizializza una traiettoria vuota per ogni env.
        """
        if self._initialized and self.error_on_premature_reset and self.n_transitions > 0:
            raise RuntimeError(
                "BufferingWrapper reset() called before buffered samples were accessed"
            )

        obs = self.venv.reset(**kwargs)

        self._initialized = True
        self._saved_actions = None
        self._finished_trajectories = []
        self._partial_trajectories = [types.Trajectory(transitions=[]) for _ in range(self.num_envs)]
        self._ep_lens = []
        self._timesteps = np.zeros(self.num_envs, dtype=int)
        self.n_transitions = 0
        self._last_obs = obs

        return obs

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

        obs, rewards, dones, infos = self.venv.step_wait()

        self.n_transitions += self.num_envs
        self._timesteps += 1

        for i in range(self.num_envs):
            transition = types.Transition(
                obs=self._last_obs[i],      # o_t
                action=actions[i],          # a_t
                reward=float(rewards[i]),   # r_t
            )

            self._partial_trajectories[i].add_transition(transition)

            if dones[i]:
                # episodio finito: salva traiettoria completa
                self._finished_trajectories.append(self._partial_trajectories[i])
                self._ep_lens.append(int(self._timesteps[i]))

                # ricomincia una nuova traiettoria vuota per quell'env
                self._partial_trajectories[i] = types.Trajectory(transitions=[])
                self._timesteps[i] = 0

        self._last_obs = obs
        return obs, rewards, dones, infos

    def pop_finished_trajectories(self) -> Tuple[List[types.Trajectory], List[int]]:
        """
        Restituisce solo le traiettorie complete terminate dall'ultimo pop.
        """
        trajectories = self._finished_trajectories
        ep_lens = self._ep_lens

        self._finished_trajectories = []
        self._ep_lens = []
        self.n_transitions = sum(len(traj.transitions) for traj in self._partial_trajectories)

        return trajectories, ep_lens

    def pop_trajectories(self) -> Tuple[List[types.Trajectory], List[int]]:
        """
        Restituisce:
        - tutte le traiettorie complete
        - tutti i frammenti correnti non vuoti

        I frammenti correnti vengono restituiti ma NON persi: qui li svuotiamo
        e ripartiamo da traiettorie vuote.
        """
        if self.n_transitions == 0:
            return [], []

        trajectories = list(self._finished_trajectories)
        ep_lens = list(self._ep_lens)

        # aggiungi anche i frammenti correnti non vuoti
        for traj in self._partial_trajectories:
            if traj.length() > 0:
                trajectories.append(traj)

        # reset buffer interno
        self._finished_trajectories = []
        self._ep_lens = []
        self._partial_trajectories = [types.Trajectory(transitions=[]) for _ in range(self.num_envs)]
        self.n_transitions = 0

        return trajectories, ep_lens

    def pop_transitions(self) -> List[types.Transition]:
        """
        Restituisce tutte le transizioni raccolte dall'ultimo pop
        come lista piatta di Transition.
        """
        if self.n_transitions == 0:
            raise RuntimeError("Called pop_transitions on an empty BufferingWrapper")

        trajectories, _ = self.pop_trajectories()

        transitions: List[types.Transition] = []
        for traj in trajectories:
            transitions.extend(traj.transitions)

        return transitions



class EnvExplorationWrapper:
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
        switch_prob: float,
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
            switch_prob:
                Probabilità di cambiare policy dopo ogni chiamata.
            rng:
                Generatore random usato per tutte le scelte casuali.
        """
        self.wrapped_policy = policy
        self.venv = venv
        self.random_prob = random_prob
        self.switch_prob = switch_prob
        self.rng = rng

        # Seed dell'action space, così anche il sampling casuale dipende da rng
        seed = int(self.rng.integers(0, 2**31 - 1))
        self.venv.action_space.seed(seed)

        # La policy attualmente attiva: inizialmente la impostiamo
        # e poi la scegliamo davvero con _switch()
        self.current_policy = self.wrapped_policy
        self._switch()

    def _random_policy(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]],
        episode_start: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, None]:
        """
        Policy casuale: ignora input e campiona un'azione random per ogni env.
        """
        del state, episode_start

        num_envs = len(obs)
        actions = [self.venv.action_space.sample() for _ in range(num_envs)]
        actions = np.stack(actions, axis=0)
        return actions, None

    def _switch(self) -> None:
        """
        Sceglie una nuova policy corrente:
        - random policy con probabilità random_prob
        - wrapped policy altrimenti
        """
        if self.rng.random() < self.random_prob:
            self.current_policy = self._random_policy
        else:
            self.current_policy = self.wrapped_policy

    def __call__(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        input_state: Optional[Tuple[np.ndarray, ...]],
        episode_start: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, None]:
        """
        Calcola l'azione usando la policy corrente.

        Dopo aver prodotto l'azione, con probabilità switch_prob
        cambia policy per la chiamata successiva.

        Non supporta policy stateful/ricorrenti.
        """
        del episode_start

        if input_state is not None:
            raise ValueError(
                "ExplorationWrapper does not support stateful policies."
            )

        actions, output_state = self.current_policy(observation, None, None)

        if output_state is not None:
            raise ValueError(
                "ExplorationWrapper does not support stateful policies."
            )

        # Con una certa probabilità cambia policy per il prossimo step
        if self.rng.random() < self.switch_prob:
            self._switch()

        return actions, None