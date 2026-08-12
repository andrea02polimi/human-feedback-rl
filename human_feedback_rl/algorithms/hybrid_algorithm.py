"""Hybrid reward learning from demonstrations and preferences.

The single reward-learning algorithm of the package: with both feedback
sources active it is the hybrid method; with ``demo_weight=0`` it degenerates
to the preference-only baseline, with ``total_queries=0`` to the
demonstration-only baseline.

Two integration mechanisms for the demonstration signal (``demo_mode``):

* ``"gcl"`` — the demo IRL loss (``demo_1`` difference-of-means or ``demo_2``
  MaxEnt surrogate) is combined with the Bradley-Terry preference loss on one
  shared reward net. The demo gradient is norm-balanced against the
  preference gradient (``demo_weight`` = desired demo/preference
  gradient-strength ratio).
* ``"preferences"`` — demonstrations enter as preference pairs (expert
  fragment preferred over agent fragment, Ibarz et al. 2018): a single
  Bradley-Terry objective on mixed batches, no scale conflict by
  construction. This mode doubles as the LITERATURE HYBRID BASELINE
  ("demonstrations as implicit preferences") for the thesis comparison —
  it is fully implemented and tested; to run it as an experiment arm see
  the ``ibarz`` placeholder in ``scripts/tune_hybrid_sac.py``.

Health metrics: with soft oracle labels at high ``pref_temperature`` the BT
loss sits at its ln(2) cross-entropy floor even when learning succeeds —
watch ``reward/acc_pref_val`` and ``reward_val/.../pred_true/pearson_*``
instead. Adam weight decay at 1e-2 overwhelms the BT gradient and collapses
the reward net (diagnosed 2026-07-05).
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.algorithms.hybrid.alpha_estimation import estimate_alpha
from human_feedback_rl.algorithms.hybrid.imitation_metrics import ImitationMetricsMixin
from human_feedback_rl.algorithms.hybrid.demonstration_losses import (
    VALID_LOSSES,
    RewardLossMixin,
)
from human_feedback_rl.algorithms.hybrid.reward_diagnostics import RewardDiagnosticsMixin
from human_feedback_rl.algorithms.hybrid.reward_training import RewardTrainingMixin
from human_feedback_rl.common.base_reward_learning_algorithm import (
    QUERY_SCHEDULES,
    BaseRewardLearningAlgorithm,
)
from human_feedback_rl.common.batching import fragment_avg_rewards
from human_feedback_rl.common.datasets import PreferenceBatch, PreferenceDataset
from human_feedback_rl.common.fragmenters import make_pair_fragmenter
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.preference_losses import (
    bradley_terry_probs,
    evaluate_preference_batch,
    preference_labels_tensor,
    preference_nll,
)
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.types import Preference, Trajectory

VALID_DEMO_MODES = ("gcl", "preferences")

# Come i due gradienti diventano un update.
#   norm_balance            somma bilanciata di norma (ricetta storica)
#   alpha_norm_single_adam  direzioni unitarie combinate con alpha, UN solo Adam
VALID_GCL_FUSIONS = ("norm_balance", "alpha_norm_single_adam")

# Sotto questo numero di confronti distinti la dispersione delle preferenze non
# e' stimabile: alpha resta fissato a 1 (tutto il peso alle dimostrazioni).
ALPHA_MIN_UNIQUE_PREFS = 5


class HybridAlgorithm(
    RewardLossMixin,
    RewardTrainingMixin,
    ImitationMetricsMixin,
    RewardDiagnosticsMixin,
    BaseRewardLearningAlgorithm,
):
    """Trains one reward model from preferences and/or demonstrations.

    See the module docstring for the two ``demo_mode`` mechanisms and the
    degenerate single-source baselines. In ``"gcl"`` mode the two losses are
    combined with a norm-balanced sum: the demo gradient is rescaled so its
    norm is ``demo_weight`` times the preference gradient's norm before the
    two are added.
    """

    VALID_LOSSES = VALID_LOSSES

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: List[Trajectory],
        loss_type: str = "demo_2",
        lr_rew: float = 0.001,
        gradient_steps_rew: int = 10,
        batch_size_expert: int = 32,
        batch_size_model: int = 64,
        batch_size_pref: int = 32,
        l2_rew: float = 0.0001,
        pref_temperature: float = 20.0,
        preference_fragment_length: int = 1,
        fragmenter_type: str = "random",
        labels_type: str = "binary",
        comparison_queue_size: int = 1_000_000,
        total_queries: int = 10_000,
        initial_queries: int = 0,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        demo_mode: str = "gcl",
        demo_weight: float = 1.0,
        max_balance_scale: float = 100.0,
        balance_eps: float = 1e-8,
        gcl_fusion: str = "norm_balance",
        alpha_eps: float = 1e-8,
        label_smoothing: float = 0.0,
        demo_pref_pairs_per_iteration: int = 64,
        demo_pref_batch_fraction: float = 0.5,
        initial_agent_timesteps: int = 0,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        reward_model_kwargs: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[dict] = None,
        rollout_env=None,
        relabel_rewards: bool = True,
        normalize_agent_reward: bool = True,
        agent_log_timestep_interval: Optional[int] = None,
    ):
        if not expert_trajectories:
            raise ValueError("expert_trajectories must be a non-empty list of Trajectory objects.")
        if loss_type not in self.VALID_LOSSES:
            raise ValueError(f"loss_type must be one of {self.VALID_LOSSES}, got {loss_type!r}.")
        if gradient_steps_rew <= 0:
            raise ValueError("gradient_steps_rew must be positive.")
        if batch_size_expert <= 0 or batch_size_model <= 0:
            raise ValueError("Reward-model batch sizes must be positive.")
        if batch_size_pref <= 0:
            raise ValueError("batch_size_pref must be positive.")
        if preference_fragment_length <= 0:
            raise ValueError("preference_fragment_length must be positive.")
        if demo_weight < 0:
            raise ValueError("demo_weight must be non-negative.")
        if max_balance_scale <= 0 or balance_eps <= 0:
            raise ValueError("max_balance_scale and balance_eps must be positive.")
        if pref_temperature <= 0:
            raise ValueError("pref_temperature must be positive.")
        if demo_mode not in VALID_DEMO_MODES:
            raise ValueError(f"demo_mode must be one of {VALID_DEMO_MODES}, got {demo_mode!r}.")
        if gcl_fusion not in VALID_GCL_FUSIONS:
            raise ValueError(f"gcl_fusion must be one of {VALID_GCL_FUSIONS}, got {gcl_fusion!r}.")
        if alpha_eps <= 0:
            raise ValueError("alpha_eps must be positive.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if demo_pref_pairs_per_iteration < 0:
            raise ValueError("demo_pref_pairs_per_iteration must be non-negative.")
        if not 0 <= demo_pref_batch_fraction <= 1:
            raise ValueError("demo_pref_batch_fraction must be in [0, 1].")

        reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        super().__init__(
            env=env,
            agent=agent,
            reward_model=reward_model,
            exploration_frac=exploration_frac,
            exploration_eps=exploration_eps,
            rng=rng,
            log_folder=log_folder,
            output_formats=output_formats,
            debug_dataset=debug_dataset,
            sampling_venv=rollout_env,
            agent_log_timestep_interval=agent_log_timestep_interval,
        )

        self.expert_trajectories = list(expert_trajectories)
        self.loss_type = loss_type
        self.gradient_steps_rew = gradient_steps_rew
        self.batch_size_expert = batch_size_expert
        self.batch_size_model = batch_size_model
        self.initial_agent_timesteps = initial_agent_timesteps
        self.relabel_rewards = relabel_rewards
        self.normalize_agent_reward = normalize_agent_reward
        self._debug_rng = np.random.default_rng(0)
        self._debug_trajectories = self._split_into_trajectories(self.debug_dataset)

        replay_buffer = getattr(agent, "replay_buffer", None)
        if replay_buffer is not None:
            if hasattr(replay_buffer, "set_reward_model"):
                replay_buffer.set_reward_model(self.reward_model)
                replay_buffer.set_relabel_rewards(relabel_rewards)
            elif relabel_rewards:
                raise ValueError("relabel_rewards=True requires RewardRelabelReplayBuffer.")

        self.optimizers = [
            th.optim.Adam(member.parameters(), lr=lr_rew, weight_decay=l2_rew)
            for member in self.reward_model.members
        ]

        self.batch_size_pref = batch_size_pref
        self.preference_fragment_length = int(preference_fragment_length)
        self.total_queries = total_queries
        self.initial_queries = initial_queries
        self.query_schedule_name = query_schedule if isinstance(query_schedule, str) else "callable"
        if isinstance(query_schedule, str):
            if query_schedule not in QUERY_SCHEDULES:
                raise ValueError(f"Unknown query_schedule: {query_schedule!r}.")
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            self.query_schedule = query_schedule

        self.demo_mode = demo_mode
        self.demo_weight = float(demo_weight)
        # --- EXTENSION PLACEHOLDER: learned demo_weight via Adam ------------
        # Planned (phase 5, first tested on hybrid + soft preferences): make
        # the demo/preference balance a learnable parameter instead of a fixed
        # hyperparameter. Sketch (disabled until implemented):
        #   self.log_demo_weight = th.nn.Parameter(
        #       th.tensor(math.log(self.demo_weight)))       # log-space -> w>0
        #   self.demo_weight_optimizer = th.optim.Adam(
        #       [self.log_demo_weight], lr=demo_weight_lr)   # new kwarg
        # The weight used in ``_reward_step`` then becomes
        # ``exp(self.log_demo_weight)`` and is updated once per reward-model
        # step (see the twin placeholder in ``_reward_step``). With the
        # feature off, behaviour must stay identical to the constant weight.
        # Design notes: docs/extensions-roadmap.md.
        self.max_balance_scale = float(max_balance_scale)
        self.balance_eps = float(balance_eps)
        self.gcl_fusion = gcl_fusion
        self.alpha_eps = float(alpha_eps)
        self.label_smoothing = float(label_smoothing)
        self.labels_type = labels_type
        # Stima di alpha dell'iterazione corrente, una per membro (id -> AlphaEstimate).
        self._alpha_current = {}
        # RNG separato per le diagnostiche: estrarre il rollout per la stima non
        # deve consumare lo stato di quello usato dal training.
        self._grad_probe_rng = np.random.default_rng(12345)
        self.demo_pref_pairs_per_iteration = int(demo_pref_pairs_per_iteration)
        self.demo_pref_batch_fraction = float(demo_pref_batch_fraction)

        self.fragmenter = make_pair_fragmenter(
            fragmenter_type, rng=self.rng, logger=self.logger, reward_ensemble=self.reward_model
        )
        # Oracle label softness is a property of the (synthetic) annotator,
        # NOT of the demo IRL loss: it gets its own temperature.
        self.preference_gatherer = PreferenceGathererFromReward(
            logger=self.logger,
            labels_type=labels_type,
            temperature=pref_temperature,
            rng=self.rng,
        )
        self.dataset_train = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)
        # Expert-vs-agent pairs (demo_mode="preferences" only).
        self.dataset_demo_prefs_train = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 1_000_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        scatter_interval: Optional[int] = None,
    ) -> Any:
        """Run hybrid reward learning and agent training."""
        if scatter_interval is None:
            scatter_interval = 10
        if scatter_interval < 0:
            raise ValueError("scatter_interval must be non-negative.")

        n_iterations = int(total_timesteps / timesteps_per_iteration)
        # Il budget di query sta con i suoi fratelli (initial_queries,
        # query_schedule) fra i kwargs dell'algoritmo. Prima era anche un
        # parametro di train(), che quando presente vinceva in silenzio: due
        # posti per lo stesso numero, e nessun modo di accorgersi di averne
        # impostato solo uno.
        schedule = self.build_query_schedule(n_iterations, self.total_queries)
        self._n_training_iterations = n_iterations

        if self.initial_agent_timesteps > 0:
            print(f"- Collecting {self.initial_agent_timesteps} bootstrap transitions")
            self.trajectories = self.sample_rollout(self.initial_agent_timesteps)
            bootstrap_queries = self.initial_queries
            self._collect_feedback(bootstrap_queries)
            if schedule:
                schedule[0] = max(schedule[0] - bootstrap_queries, 0)
            print("- Bootstrapping reward model")
            self._train_reward_model()
            self._update_agent_reward_normalization()
            self._refresh_replay_relabel_cache()
            print(f"- Pre-warming agent for {self.initial_agent_timesteps} timesteps on learned reward")
            self.train_agent(self.initial_agent_timesteps, log_interval)

        for iteration, num_queries in enumerate(schedule):
            self.iteration = iteration
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration}/{n_iterations - 1}")

            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(
                f"- Collecting {timesteps_per_iteration} agent + "
                f"{exploration_steps} exploration transitions"
            )
            self.trajectories = self.sample_rollout(timesteps_per_iteration, exploration_steps)
            self._collect_feedback(num_queries)

            self._log_expert_imitation_errors()
            all_transitions = [transition for traj in self.trajectories for transition in traj]
            self._log_validation_snapshot(all_transitions, "pre_update")

            print("- Training hybrid reward model")
            self._train_reward_model()
            self._update_agent_reward_normalization()
            self._log_validation_snapshot(all_transitions, "post_update")
            self._log_outcome_returns()
            self._log_replay_reward_staleness()

            should_log_scatter = self.debug_dataset and scatter_interval > 0 and (
                iteration % scatter_interval == 0 or iteration == n_iterations - 1
            )
            if should_log_scatter:
                self._log_return_scatter(
                    self._debug_trajectories,
                    "reward_val/debug_dataset/post_update",
                    iteration,
                )

            self._refresh_replay_relabel_cache()
            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            self.train_agent(timesteps_per_iteration, log_interval)

            self.log_iteration(t_iter)
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, iteration + 1)

        return self.trajectory_generator.agent

    # ------------------------------------------------------------------
    # Feedback collection
    # ------------------------------------------------------------------

    def _collect_feedback(self, num_queries: int) -> None:
        self._collect_preference_feedback(num_queries)
        if self.demo_mode == "preferences":
            self._collect_demo_preference_pairs(self.demo_pref_pairs_per_iteration)

    def _collect_preference_feedback(self, num_queries: int) -> None:
        if num_queries <= 0:
            return
        fragments = self.fragmenter(
            self.trajectories,
            self.preference_fragment_length,
            num_queries,
        )
        preferences = self.preference_gatherer(fragments)
        self.dataset_train.push(fragments, preferences)
        self.logger.record("dataset/n_train", len(self.dataset_train), exclude="stdout")

    def _collect_demo_preference_pairs(self, num_pairs: int) -> None:
        """Expert fragment > agent fragment pairs (Ibarz et al. 2018).

        The expert is preferred by assumption — no reward signal is used.
        """
        if num_pairs <= 0 or not self.trajectories:
            return
        from human_feedback_rl.common.types import FragmentPair

        expert_frags = self._single_fragmenter(
            self.expert_trajectories, self.preference_fragment_length, num_pairs
        )
        agent_frags = self._single_fragmenter(
            self.trajectories, self.preference_fragment_length, num_pairs
        )
        pairs = [FragmentPair(e, a) for e, a in zip(expert_frags, agent_frags)]
        preferences = [Preference(1.0, 0.0) for _ in pairs]
        self.dataset_demo_prefs_train.push(pairs, preferences)
        self.logger.record(
            "dataset/n_demo_prefs_train", len(self.dataset_demo_prefs_train), exclude="stdout"
        )

    # ------------------------------------------------------------------
    # Reward-model training
    # ------------------------------------------------------------------

    def _train_reward_model(self) -> None:
        if not self.trajectories:
            return
        if self.demo_mode == "preferences":
            self._train_reward_model_pure_preferences()
        else:
            self._train_reward_model_gcl()

    def _train_reward_model_gcl(self) -> None:
        """BT + GCL on the shared net with norm-balanced gradient fusion.

        Degenerate cases stay well-defined: with an empty preference dataset
        only the demo loss trains (demo-only arm); with ``demo_weight == 0``
        only the preference loss trains (pref-only arm).
        """
        has_prefs = len(self.dataset_train) > 0
        demo_weight = self.demo_weight
        if not has_prefs and demo_weight == 0.0:
            return

        self.logger.record("reward/demo_weight", demo_weight, exclude="stdout")
        # Prima di qualunque passo: il peso deve descrivere questo theta.
        self._estimate_alpha()
        t0 = time.perf_counter()

        def member_step(member, optimizer):
            member.train()
            boot_dataset = self.dataset_train.bootstrap() if has_prefs else None
            # Un alpha per membro per tutta l'iterazione, stimato sopra su
            # questi stessi parametri.
            alpha = self._alpha_weight(member)
            stats = []
            for _ in range(self.gradient_steps_rew):
                pref_loss = (
                    self._preference_loss(member, boot_dataset.sample(self.batch_size_pref))
                    if boot_dataset is not None
                    else None
                )
                demo_loss = self._reward_loss(member) if demo_weight > 0.0 else None
                stats.append(
                    self._reward_step(member, optimizer, pref_loss, demo_loss, alpha=alpha)
                )
            return stats

        all_stats = [s for stats in self.train_reward_members(member_step) for s in stats]
        t_train = time.perf_counter() - t0

        self._log_reward_loss_diagnostics()
        self._log_preference_diagnostics()
        self._log_hybrid_step_stats(all_stats)
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)

    def _train_reward_model_pure_preferences(self) -> None:
        """Single BT objective on mixed oracle + expert-vs-agent batches."""
        has_oracle = len(self.dataset_train) > 0
        has_demo = len(self.dataset_demo_prefs_train) > 0
        if not has_oracle and not has_demo:
            return

        t0 = time.perf_counter()
        n_demo = round(self.batch_size_pref * self.demo_pref_batch_fraction)
        if not has_demo:
            n_demo = 0
        if not has_oracle:
            n_demo = self.batch_size_pref
        n_oracle = self.batch_size_pref - n_demo

        def member_step(member, optimizer):
            member.train()
            boot_oracle = self.dataset_train.bootstrap() if n_oracle else None
            boot_demo = self.dataset_demo_prefs_train.bootstrap() if n_demo else None
            losses = []
            for _ in range(self.gradient_steps_rew):
                parts = []
                if boot_oracle is not None:
                    parts.append(boot_oracle.sample(n_oracle))
                if boot_demo is not None:
                    parts.append(boot_demo.sample(n_demo))
                batch = PreferenceBatch(
                    [pair for part in parts for pair in part.fragment_pairs],
                    [pref for part in parts for pref in part.preferences],
                )
                loss = self._preference_loss(member, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            return losses

        all_losses = [l for losses in self.train_reward_members(member_step) for l in losses]
        t_train = time.perf_counter() - t0

        self._log_preference_diagnostics()
        self.logger.record("reward/hybrid_pref_loss", float(np.mean(all_losses)), exclude="stdout")
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)

    # ------------------------------------------------------------------
    # Balanced gradient step
    # ------------------------------------------------------------------

    def _reward_step(self, member, optimizer, pref_loss, demo_loss,
                     alpha=None) -> Dict[str, float]:
        """One optimizer step from the preference and/or demo losses.

        With both losses present, the two gradients are computed separately,
        the demo gradient is norm-balanced to ``demo_weight`` times the
        preference gradient, and the composed gradient (an exact per-parameter
        linear combination ``g_pref + scale * g_demo``) is written back before
        ``step()``.
        """
        nan = float("nan")
        stats = {
            "pref_loss": nan, "demo_loss": nan, "scale": nan, "pref_norm": nan,
            "demo_norm": nan, "grad_norm": nan, "alpha": nan,
        }

        if pref_loss is None and demo_loss is None:
            return stats

        if demo_loss is None:  # pref-only arm
            optimizer.zero_grad()
            pref_loss.backward()
            stats.update(pref_loss=float(pref_loss.detach()), grad_norm=self._grad_norm(member))
            optimizer.step()
            return stats
        if pref_loss is None:  # demo-only arm
            optimizer.zero_grad()
            demo_loss.backward()
            stats.update(demo_loss=float(demo_loss.detach()), grad_norm=self._grad_norm(member))
            optimizer.step()
            return stats

        params = list(member.parameters())
        optimizer.zero_grad()
        pref_loss.backward()
        g_pref = [None if p.grad is None else p.grad.detach().clone() for p in params]
        optimizer.zero_grad()
        demo_loss.backward()
        g_demo = [None if p.grad is None else p.grad.detach().clone() for p in params]

        flat_pref = self._flatten(g_pref, params)
        flat_demo = self._flatten(g_demo, params)
        pref_norm = float(flat_pref.norm())
        demo_norm = float(flat_demo.norm())

        if self.gcl_fusion == "alpha_norm_single_adam":
            # g_fin = (1 - a) * g_p/||g_p|| + a * g_d/||g_d||, poi UN solo Adam.
            # Le norme dei due canali vengono buttate via di proposito:
            # sopravvive solo la direzione, ed e' per questo che alpha deve
            # essere adimensionale (da qui la normalizzazione per ||g||^2 in
            # CV^2).
            if alpha is None:
                alpha = self._alpha_weight(member)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            unit_pref = flat_pref / (pref_norm + self.balance_eps)
            unit_demo = flat_demo / (demo_norm + self.balance_eps)
            g_fin = (1.0 - alpha) * unit_pref + alpha * unit_demo
            self._set_flat_grad(params, g_fin)
            optimizer.step()
            stats.update(
                pref_loss=float(pref_loss.detach()),
                demo_loss=float(demo_loss.detach()),
                pref_norm=pref_norm, demo_norm=demo_norm,
                grad_norm=float(g_fin.norm()), alpha=alpha,
            )
            return stats

        scale = min(
            self.demo_weight * pref_norm / (demo_norm + self.balance_eps),
            self.max_balance_scale,
        )
        # --- EXTENSION PLACEHOLDER: learned demo_weight via Adam ------------
        # With the learnable weight enabled (see __init__), this becomes:
        #   demo_weight = float(th.exp(self.log_demo_weight))
        #   scale = min(demo_weight * pref_norm / (demo_norm + eps), max_scale)
        # and, after ``optimizer.step()`` below, the weight takes its own Adam
        # step on a validation signal — e.g. the Bradley-Terry loss of the
        # updated member on a held-out preference batch
        # (a held-out preference batch, when one is available), with the
        # gradient wrt log_demo_weight obtained by differentiating through the
        # mixing coefficient (unrolled one-step or finite differences).

        for p, gp, gd in zip(params, g_pref, g_demo):
            if gp is None and gd is None:
                p.grad = None
                continue
            grad = th.zeros_like(p)
            if gp is not None:
                grad += gp
            if gd is not None:
                grad += scale * gd
            p.grad = grad

        grad_norm = self._grad_norm(member)
        optimizer.step()

        stats.update(
            pref_loss=float(pref_loss.detach()),
            demo_loss=float(demo_loss.detach()),
            scale=scale, pref_norm=pref_norm, demo_norm=demo_norm, grad_norm=grad_norm,
        )
        return stats

    # ------------------------------------------------------------------
    # Peso di affidabilita' (alpha)
    # ------------------------------------------------------------------

    def _alpha_weight(self, member) -> float:
        """Peso sul canale dimostrazioni per l'iterazione corrente.

        Sola lettura: il valore lo produce ``_estimate_alpha``, che gira una
        volta all'inizio del training del reward sui parametri dove il peso
        verra' applicato. Fallback a 1 (sole dimostrazioni) quando non esiste
        una stima per questo membro.
        """
        estimate = self._alpha_current.get(id(member))
        return 1.0 if estimate is None else estimate.alpha

    def _alpha_is_active(self) -> bool:
        """True quando alpha e' stimato e non fissato al fallback."""
        estimates = [
            self._alpha_current.get(id(member)) for member in self.reward_model.members
        ]
        present = [e for e in estimates if e is not None]
        return bool(present) and all(not e.pinned for e in present)

    def _estimate_alpha(self) -> None:
        """Stima l'alpha di questa iterazione, PRIMA di ogni passo di gradiente.

        Il momento conta: il peso dice quanto e' rumoroso il gradiente di ogni
        canale IN UN PUNTO DEI PARAMETRI, quindi va misurato dove viene usato.
        Misurarlo dopo l'update e applicarlo all'iterazione successiva
        significherebbe descrivere un punto diverso. Alla prima chiamata theta
        e' l'inizializzazione random: e' una misura legittima, e li' N_p e'
        quasi sempre sotto la soglia, quindi alpha resta fissato a 1.

        Il rollout che serve alla loss dimostrazioni e' estratto UNA volta e
        condiviso da tutti i campioni: non e' feedback, quindi il suo rumore di
        campionamento non deve finire nella varianza del canale. Viene estratto
        dall'RNG delle diagnostiche, cosi' la stima non perturba mai le
        estrazioni del training.
        """
        if self.gcl_fusion == "norm_balance":
            return                       # quella fusione non usa alpha
        self._alpha_current = {}
        if self.demo_weight <= 0.0 or not self.trajectories:
            return                       # canale dimostrazioni assente
        if self.loss_type != "demo_2":
            raise NotImplementedError(
                "alpha estimation implements the demo_2 per-sample "
                f"decomposition; got loss_type={self.loss_type!r}."
            )

        pref_batch = (
            self.dataset_train.get_all() if len(self.dataset_train) else None
        )
        n_model = min(self.batch_size_model, len(self.trajectories))
        model_indices = self._grad_probe_rng.choice(
            len(self.trajectories), size=n_model, replace=False
        )
        model_trajs = [self.trajectories[i] for i in model_indices]

        t0 = time.perf_counter()
        for member in self.reward_model.members:
            params = [p for p in member.parameters() if p.requires_grad]
            smooth = (
                self._smoothed_labels(preference_labels_tensor(pref_batch.preferences))
                if pref_batch is not None else None
            )
            self._alpha_current[id(member)] = estimate_alpha(
                member,
                params,
                pref_batch,
                smooth,
                self.expert_trajectories,
                model_trajs,
                batch_size_pref=self.batch_size_pref,
                batch_size_expert=self.batch_size_expert,
                min_unique_prefs=ALPHA_MIN_UNIQUE_PREFS,
                eps=self.alpha_eps,
            )
        self.logger.record("time/estimate_alpha", time.perf_counter() - t0)
        self._log_alpha_estimate()

    def _log_alpha_estimate(self) -> None:
        """Pubblica le due dispersioni che formano alpha, e i loro ingredienti.

        ``alpha/S_*`` e' il sanity check: e' la varianza del gradiente che
        l'ottimizzatore applica davvero, quindi deve calare al crescere del
        budget.
        """
        estimates = [
            self._alpha_current.get(id(member)) for member in self.reward_model.members
        ]
        estimates = [e for e in estimates if e is not None]
        if not estimates:
            return
        self.logger.record(
            "reward/hybrid_alpha",
            float(np.mean([e.alpha for e in estimates])),
            exclude="stdout",
        )
        self.logger.record(
            "reward/hybrid_alpha_active",
            float(self._alpha_is_active()),
            exclude="stdout",
        )
        for name, channel in (("pref", "pref"), ("demo", "demo")):
            values = [getattr(e, channel) for e in estimates]
            values = [v for v in values if v is not None]
            if not values:
                continue
            for key, attr in (
                ("V", "process_var"),
                ("S", "mean_var"),
                ("cv2", "cv2"),
                ("gradmean_norm_sq", "mean_norm_sq"),
                ("n", "n"),
                ("batch", "batch"),
            ):
                self.logger.record(
                    f"alpha/{key}_{name}",
                    float(np.mean([getattr(v, attr) for v in values])),
                    exclude="stdout",
                )

    @staticmethod
    def _set_flat_grad(params, direction: th.Tensor) -> None:
        """Scrive un vettore piatto sui ``.grad`` cosi' che ``step()`` lo usi."""
        offset = 0
        for p in params:
            k = p.numel()
            p.grad = direction[offset:offset + k].view_as(p).clone()
            offset += k

    @staticmethod
    def _flatten(grads, params) -> th.Tensor:
        parts = [
            g.reshape(-1) if g is not None else th.zeros(p.numel())
            for g, p in zip(grads, params)
        ]
        return th.cat(parts) if parts else th.zeros(0)

    def _log_hybrid_step_stats(self, all_stats: List[Dict[str, float]]) -> None:
        def nanmean(key):
            values = np.asarray([s[key] for s in all_stats], dtype=float)
            finite = values[np.isfinite(values)]
            return float(finite.mean()) if finite.size else None

        pairs = {
            "reward/hybrid_pref_loss": "pref_loss",
            "reward/hybrid_demo_loss": "demo_loss",
            "reward/hybrid_demo_scale": "scale",
            "reward/grad_norm_pref": "pref_norm",
            "reward/grad_norm_demo": "demo_norm",
            "reward/grad_norm": "grad_norm",
        }
        for log_key, stat_key in pairs.items():
            value = nanmean(stat_key)
            if value is not None:
                self.logger.record(log_key, value, exclude="stdout")

        norms = {k: nanmean(k2) for k, k2 in (("pref", "pref_norm"), ("demo", "demo_norm"))}
        if norms["pref"] is not None and norms["demo"] is not None:
            self.logger.record(
                "reward/grad_norm_demo_pref_ratio",
                norms["demo"] / (norms["pref"] + self.balance_eps),
                exclude="stdout",
            )
        grad_norms = [s["grad_norm"] for s in all_stats if np.isfinite(s["grad_norm"])]
        if grad_norms:
            self.logger.record("reward/grad_norm_max", float(np.max(grad_norms)), exclude="stdout")

    # ------------------------------------------------------------------
    # Preference loss / diagnostics
    # ------------------------------------------------------------------

    def _preference_loss(self, member, batch) -> th.Tensor:
        r1 = fragment_avg_rewards(member, [p.frag1 for p in batch.fragment_pairs])
        r2 = fragment_avg_rewards(member, [p.frag2 for p in batch.fragment_pairs])
        labels = self._smoothed_labels(preference_labels_tensor(batch.preferences))
        return preference_nll(bradley_terry_probs(r1, r2), labels)

    def _smoothed_labels(self, labels: th.Tensor) -> th.Tensor:
        """Sposta le etichette verso 1/2 di ``label_smoothing``.

        Serve solo alle etichette binarie campionate: la cross-entropy contro un
        target in {0, 1} ha minimo in Delta = +-inf, quindi con pochi confronti
        il modello li separa arbitrariamente e li memorizza. Con target
        y' = (1-eps) y + eps/2 il minimo torna finito, Delta* = logit(y'), e la
        loss ha di nuovo un pavimento H(y') > 0.

        Le etichette ``soft`` sono gia' probabilita' dell'oracolo: hanno un
        ottimo finito e un pavimento loro, e spostarle introdurrebbe un bias
        senza correggere nulla. Restano intatte.

        Applicato qui e non alla generazione (``gatherers.py``) di proposito: il
        dataset conserva le etichette vere, quindi le diagnostiche restano
        calcolate sul target grezzo.
        """
        if self.label_smoothing <= 0.0 or self.labels_type != "binary_bernoulli":
            return labels
        eps = self.label_smoothing
        return (1.0 - eps) * labels + eps / 2.0

    def _log_preference_diagnostics(self) -> None:
        """Fit diagnostics on the training comparisons.

        Non c'e' piu' un validation set: il budget di feedback e' la risorsa
        scarsa dello studio, e tenerne da parte una quota la sottraeva
        all'addestramento senza che nessuna decisione dipendesse davvero da
        quella misura. Le chiavi ``*_val`` spariscono di conseguenza.
        """
        if not len(self.dataset_train):
            return
        train_loss, train_acc = evaluate_preference_batch(
            self.reward_model, self.dataset_train.get_all()
        )
        self.logger.record("reward/loss_pref_train", train_loss, exclude="stdout")
        self.logger.record("reward/acc_pref_train", train_acc, exclude="stdout")
        if self.demo_mode == "preferences" and len(self.dataset_demo_prefs_train):
            _, demo_acc = evaluate_preference_batch(
                self.reward_model, self.dataset_demo_prefs_train.get_all()
            )
            self.logger.record("reward/acc_demo_pref_train", demo_acc, exclude="stdout")

    def _refresh_replay_relabel_cache(self) -> None:
        """Relabel the replay buffer once per iteration (the model is frozen during learn).

        Must run after ``_update_agent_reward_normalization``: cached rewards
        use the final normalization statistics for this iteration.
        """
        if not self.relabel_rewards:
            return
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "refresh_relabel_cache"):
            replay_buffer.refresh_relabel_cache()

    def _save_checkpoint_extras(self, ckpt_path: str, iteration: int) -> None:
        """Persist reward-training state, the replay buffer and the datasets."""
        th.save(
            {
                "iteration": iteration,
                "loss_type": self.loss_type,
                "relabel_rewards": self.relabel_rewards,
                "normalize_agent_reward": self.normalize_agent_reward,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            },
            os.path.join(ckpt_path, "reward_training.pt"),
        )
        agent = self.trajectory_generator.agent
        if hasattr(agent, "save_replay_buffer"):
            agent.save_replay_buffer(os.path.join(ckpt_path, "replay_buffer.pkl"))
        th.save(
            {
                "iteration": iteration,
                "demo_mode": self.demo_mode,
                "demo_weight": self.demo_weight,
                "preference_fragment_length": self.preference_fragment_length,
                "dataset_train": self.dataset_train,
                "dataset_demo_prefs_train": self.dataset_demo_prefs_train,
            },
            os.path.join(ckpt_path, "hybrid_training.pt"),
        )
