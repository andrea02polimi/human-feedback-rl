"""Hybrid reward learning from expert demonstrations *and* human preferences.

``HybridAlgorithm`` trains a single, shared reward/cost model from two feedback
sources at once:

* **Demonstrations** — a Guided-Cost-Learning-style MaxEnt IRL loss
  (``loss_type="maxent_2"`` by default) that pushes the reward up on expert
  trajectories and down on the sampled partition of model trajectories. This is
  exactly the loss already implemented in :class:`DemoAlgorithm`.
* **Preferences** — the Bradley-Terry loss over trajectory-fragment pairs used
  by :class:`PreferenceAlgorithm` (Christiano et al., 2017). Synthetic
  preferences are generated from the environment's true reward via the existing
  :class:`PreferenceGathererFromReward`.

The two losses are combined per ensemble member, per gradient step::

    total_loss = lambda_demo * gcl_loss
               + lambda_pref * preference_loss
               + l2_regularization        (via Adam weight_decay)

``lambda_demo`` and ``lambda_pref`` are hyper-parameters that select the
experimental mode:

============  ===========  ===========
mode          lambda_demo  lambda_pref
============  ===========  ===========
pref_only          0            > 0
demo_only        > 0             0
hybrid           > 0            > 0
============  ===========  ===========

Everything else — SAC policy optimization on the learned reward, reward
normalization, imitation diagnostics (expert-action RMSE), reward validation,
checkpointing and W&B logging — is inherited unchanged from
:class:`DemoAlgorithm`, so the hybrid framework logs the same rich metric set
plus the loss decomposition below.

Differences vs. the original algorithms
---------------------------------------
* The GCL/MaxEnt term is identical to ``DemoAlgorithm`` (same ``_reward_loss``).
* The preference term reproduces ``PreferenceAlgorithm``'s Bradley-Terry loss
  and bootstrap sampling, but the gradient step is *shared* with the demo term
  (one ``optimizer.step`` per member per iteration on the summed loss) instead of
  living in a separate optimizer loop. When ``lambda_demo=0`` this reduces to the
  original preference training; when ``lambda_pref=0`` it reduces to the original
  demonstration training.
"""

import time
from typing import List, Optional

import numpy as np
import torch as th

from human_feedback_rl.algorithms.demo_algorithm import DemoAlgorithm
from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import (
    HighVariancePairFragmenter,
    RandomPairFragmenter,
)
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.types import Trajectory


class HybridAlgorithm(DemoAlgorithm):
    """Alternating reward-learning (demonstrations + preferences) and SAC loop."""

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: List[Trajectory],
        # --- hybrid loss weights -------------------------------------------
        lambda_demo: float = 1.0,
        lambda_pref: float = 1.0,
        # --- demonstration (GCL / MaxEnt IRL) branch -----------------------
        loss_type: str = "maxent_2",
        batch_size_expert: int = 32,
        batch_size_model: int = 64,
        # --- preference (Bradley-Terry) branch -----------------------------
        pref_fragmenter_type: str = "random",
        pref_labels_type: str = "binary",
        pref_fragment_length: int = 1,
        pref_temperature: float = 1.0,
        pref_batch_size: int = 32,
        queries_per_iteration: int = 200,
        pref_train_frac: float = 0.8,
        pref_queue_size: int = 1_000_000,
        # --- shared reward-model optimization ------------------------------
        lr_rew: float = 0.001,
        gradient_steps_rew: int = 10,
        l2_rew: float = 0.01,
        temperature: float = 1.0,
        fragment_length: Optional[int] = None,
        # --- policy / rollout ----------------------------------------------
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
    ):
        if lambda_demo < 0 or lambda_pref < 0:
            raise ValueError("lambda_demo and lambda_pref must be non-negative.")
        if lambda_demo == 0 and lambda_pref == 0:
            raise ValueError(
                "At least one of lambda_demo / lambda_pref must be > 0 "
                "(otherwise there is no reward-learning signal)."
            )

        super().__init__(
            env=env,
            agent=agent,
            expert_trajectories=expert_trajectories,
            loss_type=loss_type,
            lr_rew=lr_rew,
            gradient_steps_rew=gradient_steps_rew,
            batch_size_expert=batch_size_expert,
            batch_size_model=batch_size_model,
            l2_rew=l2_rew,
            temperature=temperature,
            fragment_length=fragment_length,
            initial_agent_timesteps=initial_agent_timesteps,
            exploration_frac=exploration_frac,
            exploration_eps=exploration_eps,
            reward_model_kwargs=reward_model_kwargs,
            rng=rng,
            log_folder=log_folder,
            output_formats=output_formats,
            debug_dataset=debug_dataset,
            rollout_env=rollout_env,
            relabel_rewards=relabel_rewards,
            normalize_agent_reward=normalize_agent_reward,
        )

        self.lambda_demo = float(lambda_demo)
        self.lambda_pref = float(lambda_pref)
        self.mode = self._resolve_mode(self.lambda_demo, self.lambda_pref)

        # ---- preference machinery (only wired when it is actually used) ----
        self.pref_fragment_length = pref_fragment_length
        self.pref_temperature = pref_temperature
        self.pref_batch_size = pref_batch_size
        self.queries_per_iteration = queries_per_iteration
        self.pref_train_frac = pref_train_frac

        self.preference_gatherer = PreferenceGathererFromReward(
            logger=self.logger,
            labels_type=pref_labels_type,
            temperature=pref_temperature,
        )
        self.pref_fragmenter = self._make_pref_fragmenter(pref_fragmenter_type)
        self.dataset_train = PreferenceDataset(queue_size=pref_queue_size, rng=self.rng)
        self.dataset_val = PreferenceDataset(queue_size=pref_queue_size, rng=self.rng)

        print(
            f"[hybrid] mode={self.mode} "
            f"lambda_demo={self.lambda_demo} lambda_pref={self.lambda_pref} "
            f"demo_loss={self.loss_type}"
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_mode(lambda_demo: float, lambda_pref: float) -> str:
        if lambda_demo > 0 and lambda_pref > 0:
            return "hybrid"
        if lambda_demo > 0:
            return "demo_only"
        return "pref_only"

    def _make_pref_fragmenter(self, fragmenter_type: str):
        if fragmenter_type == "active":
            return HighVariancePairFragmenter(
                reward_ensemble=self.reward_model,
                oversample=5,
                logger=self.logger,
                rng=self.rng,
            )
        if fragmenter_type == "random":
            return RandomPairFragmenter(logger=self.logger, rng=self.rng)
        raise ValueError(f"Unknown pref_fragmenter_type: {fragmenter_type!r}")

    @property
    def _use_pref(self) -> bool:
        return self.lambda_pref > 0

    @property
    def _use_demo(self) -> bool:
        return self.lambda_demo > 0

    # ------------------------------------------------------------------
    # Preference data collection (mirrors PreferenceAlgorithm.push_data)
    # ------------------------------------------------------------------

    def _collect_preferences(self) -> None:
        """Fragment the current rollout, gather synthetic preferences, store them."""
        if not self._use_pref or not self.trajectories:
            return

        t0 = time.perf_counter()
        fragment_pairs = self.pref_fragmenter(
            self.trajectories, self.pref_fragment_length, self.queries_per_iteration
        )
        preferences = self.preference_gatherer(fragment_pairs)

        idx = self.rng.permutation(len(fragment_pairs))
        fragment_pairs = [fragment_pairs[i] for i in idx]
        preferences = [preferences[i] for i in idx]
        n_train = int(self.pref_train_frac * len(fragment_pairs))
        self.dataset_train.push(fragment_pairs[:n_train], preferences[:n_train])
        self.dataset_val.push(fragment_pairs[n_train:], preferences[n_train:])

        self.logger.record("pref/n_new_queries", len(fragment_pairs))
        self.logger.record("pref/dataset_train", len(self.dataset_train))
        self.logger.record("pref/dataset_val", len(self.dataset_val))
        self.logger.record("time/collect_preferences", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _preference_loss(self, member, batch) -> th.Tensor:
        """Bradley-Terry cross-entropy for one ensemble member on a batch.

        Identical objective to ``PreferenceAlgorithm.train_reward_model`` — the
        per-fragment average *raw* reward feeds a sigmoid comparison, matched
        against the (soft or hard) preference labels.
        """
        r1 = th.stack([member.fragment_avg_reward(p.frag1) for p in batch.fragment_pairs])
        r2 = th.stack([member.fragment_avg_reward(p.frag2) for p in batch.fragment_pairs])
        prob1 = th.sigmoid(r1 - r2)
        bt_probs = th.stack([prob1, 1 - prob1], dim=1)
        labels = th.tensor(
            [[p.pref1, p.pref2] for p in batch.preferences], dtype=th.float32
        )
        return -(labels * bt_probs.clamp(min=1e-7).log()).sum(dim=1).mean()

    # ------------------------------------------------------------------
    # Combined reward-model training (overrides RewardTrainingMixin)
    # ------------------------------------------------------------------

    def _train_reward_model(self) -> None:
        """One round of gradient updates on the shared reward model.

        Per member, per gradient step, the demonstration (GCL) and preference
        (Bradley-Terry) losses are combined into a single scalar and back-propped
        together, so both feedback sources shape the *same* reward network.
        """
        if not self.trajectories:
            return

        self._maxent_corrected_steps = []
        self._maxent_selfnorm_steps = []
        self._collect_preferences()
        use_pref = self._use_pref and len(self.dataset_train) > 0

        t0 = time.perf_counter()
        demo_losses, pref_losses, total_losses, grad_norms = [], [], [], []

        for member, optimizer in zip(self.reward_model.members, self.optimizers):
            member.train()
            boot = self.dataset_train.bootstrap() if use_pref else None
            for _ in range(self.gradient_steps_rew):
                loss = th.zeros((), dtype=th.float32)
                demo_val = float("nan")
                pref_val = float("nan")

                if self._use_demo:
                    gcl = self._reward_loss(member)
                    loss = loss + self.lambda_demo * gcl
                    demo_val = float(gcl.detach())

                if use_pref:
                    batch = boot.sample(self.pref_batch_size)
                    pref = self._preference_loss(member, batch)
                    loss = loss + self.lambda_pref * pref
                    pref_val = float(pref.detach())

                if not th.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite hybrid reward loss (mode={self.mode}): {loss.item()}"
                    )

                optimizer.zero_grad()
                loss.backward()
                grad_norm = self._grad_norm(member)
                if not np.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite hybrid reward gradient norm.")
                optimizer.step()

                demo_losses.append(demo_val)
                pref_losses.append(pref_val)
                total_losses.append(float(loss.detach()))
                grad_norms.append(grad_norm)

        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._log_hybrid_losses(demo_losses, pref_losses, total_losses, use_pref)
        self._log_preference_metrics(use_pref)
        # Reuse the demo-side reward diagnostics (maxent_2 partition, ESS, ...).
        self._log_reward_loss_diagnostics()
        self._log_maxent_selfnorm_step_diagnostics()
        self.logger.record("reward/grad_norm", float(np.mean(grad_norms)), exclude="stdout")
        self.logger.record("reward/grad_norm_max", float(np.max(grad_norms)), exclude="stdout")
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)
        self.logger.record_sum("time/loggings", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_hybrid_losses(self, demo_losses, pref_losses, total_losses, use_pref) -> None:
        self.logger.record("reward/lambda_demo", self.lambda_demo)
        self.logger.record("reward/lambda_pref", self.lambda_pref)
        self.logger.record("reward/total_reward_model_loss", float(np.mean(total_losses)))

        if self._use_demo:
            self.logger.record("reward/gcl_loss", float(np.nanmean(demo_losses)))
        if use_pref:
            self.logger.record("reward/preference_loss", float(np.nanmean(pref_losses)))

        # Numeric mode indicator so the three experimental modes are trivially
        # separable in W&B (0=pref_only, 1=demo_only, 2=hybrid).
        mode_code = {"pref_only": 0, "demo_only": 1, "hybrid": 2}[self.mode]
        self.logger.record("hybrid/mode_code", mode_code, exclude="stdout")

    def _log_preference_metrics(self, use_pref) -> None:
        if not use_pref:
            return
        loss_train, acc_train = self._evaluate_preferences(self.dataset_train.get_all())
        if loss_train is not None:
            self.logger.record("pref/loss_train", loss_train, exclude="stdout")
            self.logger.record("pref/acc_train", acc_train, exclude="stdout")
        if len(self.dataset_val) > 0:
            loss_val, acc_val = self._evaluate_preferences(self.dataset_val.get_all())
            if loss_val is not None:
                self.logger.record("pref/loss_val", loss_val, exclude="stdout")
                self.logger.record("pref/acc_val", acc_val, exclude="stdout")

    def _evaluate_preferences(self, data):
        """Bradley-Terry loss and accuracy of the full ensemble on a split."""
        if not data.fragment_pairs:
            return None, None
        self.reward_model.eval()
        with th.no_grad():
            r1 = th.stack([self.reward_model.fragment_avg_reward(p.frag1) for p in data.fragment_pairs])
            r2 = th.stack([self.reward_model.fragment_avg_reward(p.frag2) for p in data.fragment_pairs])
            prob1 = th.sigmoid(r1 - r2)
            bt_probs = th.stack([prob1, 1 - prob1], dim=1)
        self.reward_model.train()
        labels = th.tensor([[p.pref1, p.pref2] for p in data.preferences], dtype=th.float32)
        loss = -(labels * bt_probs.clamp(min=1e-7).log()).sum(dim=1).mean().item()
        acc = (bt_probs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
        return loss, acc

    # ------------------------------------------------------------------
    # Evaluation (deterministic policy rollout on the learned reward)
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 20, log_prefix: str = "eval") -> dict:
        """Run deterministic episodes and report task-level performance.

        Complements the per-iteration diagnostics with a clean, deterministic
        estimate of the policy trained on the learned reward. Uses the dedicated
        ``rollout_env`` when available so it never desynchronizes SAC's training
        env. Returns (and logs) mean true return, success rate and expert-action
        RMSE.
        """
        sampling_venv = self.trajectory_generator.sampling_venv
        obs = sampling_venv.reset()
        returns, successes, lengths = [], [], []
        cur_return = np.zeros(sampling_venv.num_envs, dtype=np.float64)
        cur_len = np.zeros(sampling_venv.num_envs, dtype=np.int64)

        while len(returns) < n_episodes:
            actions, _ = self.agent.predict(obs, deterministic=True)
            obs, rewards, dones, infos = sampling_venv.step(actions)
            cur_return += np.asarray(rewards, dtype=np.float64)
            cur_len += 1
            for i, done in enumerate(dones):
                if not done:
                    continue
                info = infos[i]
                true_r = info.get("episode", {}).get("r", cur_return[i])
                returns.append(float(true_r))
                status = info.get("ego_status", None)
                successes.append(float(status == 0))  # 0 == ARRIVED
                lengths.append(int(cur_len[i]))
                cur_return[i] = 0.0
                cur_len[i] = 0

        observations, expert_actions = self._flatten_expert_transitions()
        rmse = float("nan")
        if len(observations):
            agent_actions, _ = self.agent.predict(observations, deterministic=True)
            agent_actions = np.asarray(agent_actions, dtype=np.float64).reshape(len(observations), -1)
            expert_actions = expert_actions.reshape(len(observations), -1)
            rmse = float(np.sqrt(np.mean((agent_actions - expert_actions) ** 2)))

        metrics = {
            "fast_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "expert_action_rmse": rmse,
            "ep_length": float(np.mean(lengths)),
            "n_episodes": len(returns),
        }
        for key, value in metrics.items():
            self.logger.record(f"{log_prefix}/{key}", value)
        self.logger.dump()
        print(f"[hybrid] {log_prefix}: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing (extend demo checkpoint with hybrid metadata)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        super()._save_checkpoint(checkpoint_dir, iteration)
        import os

        th.save(
            {
                "iteration": iteration,
                "mode": self.mode,
                "lambda_demo": self.lambda_demo,
                "lambda_pref": self.lambda_pref,
                "loss_type": self.loss_type,
            },
            os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}", "hybrid_meta.pt"),
        )
