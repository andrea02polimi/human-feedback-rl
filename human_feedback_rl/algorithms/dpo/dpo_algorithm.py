"""
DPO Algorithm — Direct Preference Optimization (Rafailov et al., 2023)
=======================================================================
Adattamento per RL continuo su (obs, action) trajectories.

Idea centrale:
  Invece di imparare esplicitamente un reward model R(s,a), DPO usa la policy
  stessa come reward implicito rispetto a una policy di riferimento fissa:

      r_θ(τ) = log π_θ(τ) - log π_ref(τ)
             = Σ_t [ log π_θ(a_t | s_t) - log π_ref(a_t | s_t) ]

  La loss DPO (sigmoid variant, Eq. 7 del paper) è:

      L_DPO(π_θ; π_ref) = -E_(τ_w, τ_l) [
          log σ( β * (r_θ(τ_w) - r_θ(τ_l)) )
      ]

  dove τ_w = chosen (winner), τ_l = rejected (loser).

Pipeline per iterazione:
  1. Collect rollouts con l'agente corrente
  2. Fragment → segment pairs
  3. Query expert per preferenze → (chosen, rejected)
  4. DPO training su chosen/rejected pairs
"""

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from human_feedback_rl.common import (
    ActiveFragmenter,
    BaseAlgorithm,
    Preference,
    PreferenceDataset,
    PrefixLogger,
    SegmentPair,
    Trajectory,
    Transition,
    UnifiedLogger,
    InverseSchedule,
)


class DPOAlgorithm(BaseAlgorithm):
    """
    DPO per RL con segmenti (obs, action).

    La policy π_θ è un agente SB3 (DQN, PPO, ecc.) con metodo evaluate_actions.
    La policy di riferimento π_ref è una copia frozen dell'agente all'inizio del training.
    """

    def __init__(
        self,
        env,
        agent,
        expert,
        segment_length: int,
        device: str = "cpu",
        # DPO hyperparameters
        beta: float = 0.1,
        loss_type: str = "sigmoid",   # "sigmoid" | "hinge" | "ipo"
        dpo_lr: float = 1e-4,
        dpo_epochs: int = 5,
        dpo_batch_size: int = 32,
        # Preference schedule
        num_pairs_initial: int = 100,
        num_pairs_final: int = 0,
        decay_pairs_schedule: float = 1.0,
        max_dataset_size: int = 10_000,
        n_eval_episodes: int = 5,
    ):
        self.env = env
        self.agent = agent
        self.expert = expert
        self.device = device
        self.beta = beta
        self.loss_type = loss_type
        self.dpo_epochs = dpo_epochs
        self.dpo_batch_size = dpo_batch_size
        self.n_eval_episodes = n_eval_episodes

        self.discrete_actions = hasattr(env.action_space, "n")

        self._logger = UnifiedLogger()
        self._dpo_log  = PrefixLogger(self._logger, prefix="dpo")
        self._eval_log = PrefixLogger(self._logger, prefix="eval")

        # TODO: costruire l'ottimizzatore per π_θ
        #   self._optimizer = torch.optim.Adam(agent.policy.parameters(), lr=dpo_lr)
        self._optimizer = None  # TODO

        # TODO: costruire π_ref come copia frozen della policy iniziale
        #   import copy
        #   self._ref_policy = copy.deepcopy(agent.policy)
        #   for p in self._ref_policy.parameters(): p.requires_grad = False
        self._ref_policy = None  # TODO

        self.fragmenter = ActiveFragmenter(
            # TODO: DPO non ha un reward model esplicito.
            # Opzione 1: random fragmenter (stub reward_model=None)
            # Opzione 2: usare r_θ(τ) = log π_θ(τ) - log π_ref(τ) come proxy score
            reward_model=None,  # TODO
            segment_length=segment_length,
        )

        self.preference_dataset     = PreferenceDataset(max_dataset_size)
        self.preference_dataset_val = PreferenceDataset(max_dataset_size)

        # Buffer DPO: lista di (chosen_traj, rejected_traj)
        self._dpo_buffer: List[Tuple[Trajectory, Trajectory]] = []

        self.schedule_num_pairs = InverseSchedule(
            initial_value=num_pairs_initial,
            final_value=num_pairs_final,
            decay_rate=decay_pairs_schedule,
        )

        if wandb.run is not None:
            wandb.define_metric("dpo/*",  step_metric="timescales/iterations")
            wandb.define_metric("eval/*", step_metric="timescales/iterations")

    # =========================================================================
    # Main training loop
    # =========================================================================

    def train(self, total_timesteps: int = 1_000_000, num_iterations: int = 10) -> Any:
        per_iter_timesteps = int(total_timesteps / num_iterations)

        for it in range(num_iterations):
            print(f"\n=== DPO Iteration {it+1}/{num_iterations} ===")
            progress_remaining = 1 - it / num_iterations
            num_pairs = int(self.schedule_num_pairs(progress_remaining))

            # ------------------------------------------------------------------
            # Step 1 — Collect rollouts con l'agente corrente
            # ------------------------------------------------------------------
            print("[1/4] Collecting rollouts...")
            # TODO: tot_rollout_timesteps = num_pairs * self.fragmenter.segment_length * 2
            # TODO: trajectories = self._collect_rollout(tot_rollout_timesteps)
            trajectories: List[Trajectory] = []  # TODO

            # ------------------------------------------------------------------
            # Step 2 — Fragment → segment pairs
            # ------------------------------------------------------------------
            print("[2/4] Fragmenting trajectories...")
            # TODO: segment_pairs = self.fragmenter.fragment(trajectories, num_pairs)
            segment_pairs: List[SegmentPair] = []  # TODO

            # ------------------------------------------------------------------
            # Step 3 — Query expert per preferenze → costruisci (chosen, rejected)
            # ------------------------------------------------------------------
            print("[3/4] Querying expert for preferences...")
            # TODO: preferences = self._query_preferences(segment_pairs)
            # TODO: new_pairs = self._build_chosen_rejected_pairs(segment_pairs, preferences)
            # TODO: self._dpo_buffer.extend(new_pairs)
            # TODO: train/val split e push in self.preference_dataset / val

            # ------------------------------------------------------------------
            # Step 4 — DPO training
            # ------------------------------------------------------------------
            print("[4/4] DPO training...")
            dpo_stats = self._dpo_train()

            # ------------------------------------------------------------------
            # Evaluation + logging
            # ------------------------------------------------------------------
            eval_stats = self._evaluate(self.n_eval_episodes)

            self._dpo_log.record("loss",            dpo_stats.get("loss", 0.0))
            self._dpo_log.record("chosen_reward",   dpo_stats.get("chosen_reward", 0.0))
            self._dpo_log.record("rejected_reward", dpo_stats.get("rejected_reward", 0.0))
            self._dpo_log.record("accuracy",        dpo_stats.get("accuracy", 0.0))
            self._dpo_log.record("margin",          dpo_stats.get("margin", 0.0))
            self._dpo_log.record("buffer_size",     len(self._dpo_buffer))

            self._eval_log.record("mean_ep_reward", eval_stats["mean_ep_reward"])
            self._eval_log.record("mean_ep_length", eval_stats["mean_ep_length"])

            self._logger.record("timescales/iterations", it)
            self._logger.dump()

        return self.agent

    # =========================================================================
    # DPO training core
    # =========================================================================

    def _dpo_train(self) -> dict:
        """
        Ottimizza π_θ con la DPO loss su tutte le coppie (τ_w, τ_l) nel buffer.

        Per ogni coppia:
          1. log π_θ(τ_w), log π_θ(τ_l)        via _segment_logprob (con grad)
          2. log π_ref(τ_w), log π_ref(τ_l)     via _segment_logprob su _ref_policy (no grad)
          3. Implicit reward (log-ratio):
               r_θ(τ) = log π_θ(τ) - log π_ref(τ)
          4. delta_score = r_θ(τ_w) - r_θ(τ_l)
          5. loss = _dpo_loss(delta_score)

        Returns:
            dict con loss, chosen_reward, rejected_reward, accuracy, margin
        """
        if not self._dpo_buffer:
            return {"loss": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0,
                    "accuracy": 0.0, "margin": 0.0}

        # TODO: implementare il training loop
        #
        # total_loss = chosen_r_sum = rejected_r_sum = correct = 0.0
        # n_steps = 0
        #
        # for epoch in range(self.dpo_epochs):
        #     perm = torch.randperm(len(self._dpo_buffer))
        #     for batch_idx in _chunks(perm, self.dpo_batch_size):
        #
        #         chosen_logps   = torch.stack([self._segment_logprob(self._dpo_buffer[i][0]) for i in batch_idx])
        #         rejected_logps = torch.stack([self._segment_logprob(self._dpo_buffer[i][1]) for i in batch_idx])
        #
        #         with torch.no_grad():
        #             ref_chosen_logps   = torch.stack([self._ref_segment_logprob(self._dpo_buffer[i][0]) for i in batch_idx])
        #             ref_rejected_logps = torch.stack([self._ref_segment_logprob(self._dpo_buffer[i][1]) for i in batch_idx])
        #
        #         chosen_logratios   = chosen_logps   - ref_chosen_logps    # r_θ(τ_w)
        #         rejected_logratios = rejected_logps - ref_rejected_logps  # r_θ(τ_l)
        #
        #         delta_score = chosen_logratios - rejected_logratios       # Eq. 7
        #
        #         loss = self._dpo_loss(delta_score)
        #
        #         self._optimizer.zero_grad()
        #         loss.backward()
        #         self._optimizer.step()
        #
        #         total_loss      += loss.item()
        #         chosen_r_sum    += chosen_logratios.mean().item()
        #         rejected_r_sum  += rejected_logratios.mean().item()
        #         correct         += (delta_score > 0).float().mean().item()
        #         n_steps         += 1
        #
        # n = max(n_steps, 1)
        # return {
        #     "loss":            total_loss     / n,
        #     "chosen_reward":   chosen_r_sum   / n,
        #     "rejected_reward": rejected_r_sum / n,
        #     "accuracy":        correct         / n,
        #     "margin":         (chosen_r_sum - rejected_r_sum) / n,
        # }

        raise NotImplementedError  # TODO

    def _dpo_loss(self, delta_score: torch.Tensor) -> torch.Tensor:
        """
        Calcola la DPO loss dato delta_score = r_θ(τ_w) - r_θ(τ_l).

        Varianti (Rafailov et al. e follow-up):
          sigmoid (DPO, Eq. 7):  -log σ(β * delta_score)
          hinge (SLiC):          max(0, 1 - β * delta_score)
          ipo (IPO, Azar et al.): (delta_score - 1/(2β))²
        """
        # TODO:
        # if self.loss_type == "sigmoid":
        #     return -F.logsigmoid(self.beta * delta_score).mean()
        # elif self.loss_type == "hinge":
        #     return F.relu(1.0 - self.beta * delta_score).mean()
        # elif self.loss_type == "ipo":
        #     return ((delta_score - 1.0 / (2.0 * self.beta)) ** 2).mean()
        # else:
        #     raise ValueError(f"Unknown loss_type: {self.loss_type}")

        raise NotImplementedError  # TODO

    # =========================================================================
    # Log-probability computation
    # =========================================================================

    def _segment_logprob(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Calcola log π_θ(τ) = Σ_t log π_θ(a_t | s_t) su un segmento.

        Analogo RL di "sequence log-probability" in NLP.
        Solo le azioni (completion) contribuiscono — le osservazioni sono il "prompt".

        Returns:
            scalar Tensor (con grad): somma dei log-prob sul segmento
        """
        # TODO:
        # obs = np.stack([t.obs for t in trajectory.transitions])
        # acts = np.array([t.action for t in trajectory.transitions])
        # obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # act_t = torch.as_tensor(acts, dtype=torch.long if self.discrete_actions
        #                              else torch.float32, device=self.device)
        # _, log_probs, _ = self.agent.policy.evaluate_actions(obs_t, act_t)
        # return log_probs.sum()   # Σ_t log π_θ(a_t | s_t)

        raise NotImplementedError  # TODO

    def _ref_segment_logprob(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Calcola log π_ref(τ) = Σ_t log π_ref(a_t | s_t) con π_ref frozen.

        Sempre chiamato in no_grad.
        """
        # TODO: come _segment_logprob ma con self._ref_policy al posto di self.agent.policy

        raise NotImplementedError  # TODO

    # =========================================================================
    # Expert interaction
    # =========================================================================

    def _query_preferences(self, segment_pairs: List[SegmentPair]) -> List[Preference]:
        """
        Query l'expert oracle per preferenze tra coppie di segmenti.

        Lo score per ogni segmento è calcolato con _expert_segment_score:
          - DQN:  mean Q_expert(s_t, a_t) sul segmento
          - PPO:  mean log π_expert(a_t | s_t) sul segmento
        """
        # TODO: implementare
        # return [
        #     Preference((1, 0) if self._expert_segment_score(p.seg1) >=
        #                          self._expert_segment_score(p.seg2) else (0, 1))
        #     for p in segment_pairs
        # ]

        raise NotImplementedError  # TODO

    def _expert_segment_score(self, segment: Trajectory) -> float:
        """
        Calcola lo score dell'expert su un segmento (normalizzato per lunghezza).

        - DQN:  mean_t Q_expert(s_t, a_t)
        - PPO:  mean_t log π_expert(a_t | s_t)
        """
        # TODO: implementare (vedi HumLrnAlgorithm._expert_segment_score)

        raise NotImplementedError  # TODO

    def _build_chosen_rejected_pairs(
        self,
        segment_pairs: List[SegmentPair],
        preferences: List[Preference],
    ) -> List[Tuple[Trajectory, Trajectory]]:
        """
        Costruisce coppie (τ_w, τ_l) da SegmentPair + Preference.

        Preference.label == (1, 0) → seg1 è chosen (w), seg2 è rejected (l).
        Preference.label == (0, 1) → seg2 è chosen (w), seg1 è rejected (l).
        """
        # TODO:
        # return [
        #     (p.seg1, p.seg2) if pref.label[0] > pref.label[1] else (p.seg2, p.seg1)
        #     for p, pref in zip(segment_pairs, preferences)
        # ]

        raise NotImplementedError  # TODO

    # =========================================================================
    # Rollout collection
    # =========================================================================

    def _collect_rollout(self, total_timesteps_target: int) -> List[Trajectory]:
        """
        Raccoglie traiettorie con l'agente corrente nell'env.
        """
        # TODO: implementare (vedi ChristianoAlgorithm._collect_rollout)

        raise NotImplementedError  # TODO

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _evaluate(self, n_eval_episodes: int) -> dict:
        """
        Valuta l'agente deterministicamente per n_eval_episodes.
        Ritorna mean_ep_reward e mean_ep_length sull'env reale.
        """
        # TODO: implementare (vedi DaggerAlgorithm._evaluate)

        raise NotImplementedError  # TODO

    # =========================================================================
    # Helpers
    # =========================================================================

    def _train_val_split(
        self,
        pairs: List[SegmentPair],
        preferences: List[Preference],
        split_ratio: float = 0.7,
    ):
        """Split 70/30 tra training e validation set."""
        # TODO: implementare (vedi ChristianoAlgorithm._train_val_split)

        raise NotImplementedError  # TODO