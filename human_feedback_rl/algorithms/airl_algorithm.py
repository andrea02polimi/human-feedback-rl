import numpy as np
import torch as th
import torch.nn.functional as F

from .gail_algorithm import GailAlgorithm
from ..common.reward_nets import make_airl_reward_ensemble


class AirlAlgorithm(GailAlgorithm):
    """AIRL with the discriminator from Fu et al. (2018)."""

    def __init__(self, *args, **kwargs):
        kwargs["reward_model_factory"] = make_airl_reward_ensemble
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Reward normalization before agent training
    # ------------------------------------------------------------------

    def before_agent_training(self) -> None:
        """Center the discriminator logit mean to 0 on the current rollout."""
        if not self.trajectories:
            return
        try:
            obs, acts, next_obs, terminated = self._batch_airl_transitions(self.trajectories, 4096)
        except ValueError:
            return
        self.reward_model.eval()
        with th.no_grad():
            for member in self.reward_model.members:
                mean = member.discriminator_logit(obs, acts, next_obs, terminated).mean().item()
                member.set_mean(mean)
        self.reward_model.train()

    # ------------------------------------------------------------------
    # AIRL discriminator loss
    # ------------------------------------------------------------------

    def _compute_reward_loss(self, member, eval_batch=None) -> th.Tensor:
        # AIRL samples its own transition batches from the current rollout, so the
        # trajectory-based ``eval_batch`` snapshot is not used here.
        obs_e, act_e, next_obs_e, term_e = self._batch_airl_transitions(
            self.expert_trajectories,
            self.batch_size_expert,
        )

        obs_a, act_a, next_obs_a, term_a = self._sample_agent_batch()

        f_e = member.shaped_reward(obs_e, act_e, next_obs_e, term_e)
        f_a = member.shaped_reward(obs_a, act_a, next_obs_a, term_a)

        logits_e = f_e - self._policy_log_prob(obs_e, act_e)
        logits_a = f_a - self._policy_log_prob(obs_a, act_a)

        # Absorbing states (DAC): terminating episodes of BOTH expert and agent feed
        # into a shared absorbing state, so termination carries no class information.
        # Add one absorbing self-loop per terminal transition to each batch.
        n_abs_e = int(term_e.sum().item())
        n_abs_a = int(term_a.sum().item())
        if n_abs_e:
            logits_e = th.cat([logits_e, member.absorbing_reward(n_abs_e, obs_e.device, obs_e.dtype)])
        if n_abs_a:
            logits_a = th.cat([logits_a, member.absorbing_reward(n_abs_a, obs_a.device, obs_a.dtype)])

        loss_e = F.binary_cross_entropy_with_logits(logits_e, th.ones_like(logits_e))
        loss_a = F.binary_cross_entropy_with_logits(logits_a, th.zeros_like(logits_a))

        return loss_e + loss_a

    def _evaluate_reward_model(self) -> float:
        """Validation loss for AIRL: the discriminator BCE on the current batches.

        DemoAlgorithm's inherited ``_evaluate_reward_model`` runs the MaxEnt IRL
        validation (``_traj_sum_reward`` → ``member(obs, action, next_status, done)``),
        which passes the 7-dim one-hot ``next_status`` as the third argument. The
        AIRL net interprets that argument as the actual ``next_state`` observation
        and feeds it to ``value_net`` (expects ``obs_dim+1``), so the MaxEnt path
        is both semantically wrong and dimensionally invalid here. AIRL is a
        discriminator method, so we report the same loss used for training,
        evaluated under ``no_grad``.
        """
        if not self.trajectories or not self.expert_trajectories:
            return float("nan")
        self.reward_model.eval()
        with th.no_grad():
            losses = [self._compute_reward_loss(m).item() for m in self.reward_model.members]
        self.reward_model.train()
        return float(np.mean(losses))

    def _policy_log_prob(self, obs, actions):
        policy = self.trajectory_generator.agent.policy
        _, log_prob, _ = policy.evaluate_actions(obs, actions)
        return log_prob.detach()

    def _sample_agent_batch(self):
        return self._batch_airl_transitions(
            self.trajectories,
            self.batch_size_model,
        )

    def _batch_airl_transitions(self, trajectories, n):
        transitions = []
        next_observations = []

        for traj in trajectories:
            for i, transition in enumerate(traj):
                next_obs = getattr(transition, "next_observation", None)
                if next_obs is None and i + 1 < len(traj):
                    next_obs = traj[i + 1].observation
                if next_obs is None:
                    if transition.done:
                        # V(next_state) is zeroed by (1-done) in shaped_reward, placeholder is safe
                        next_obs = np.zeros_like(transition.observation)
                    else:
                        continue
                transitions.append(transition)
                next_observations.append(next_obs)

        if not transitions:
            raise ValueError("AIRL requires transitions with next observations.")

        n = min(n, len(transitions))
        idx = self.rng.choice(len(transitions), size=n, replace=False)
        return self._airl_transitions_to_tensors(
            [transitions[i] for i in idx],
            [next_observations[i] for i in idx],
        )

    # next_status one-hot index 3 = "timeout" (truncation, not a true terminal).
    _TIMEOUT_IDX = 3

    @classmethod
    def _is_terminated(cls, transition) -> float:
        """True-terminal mask: episode ended *and* not by timeout (truncation)."""
        if not transition.done:
            return 0.0
        ns = getattr(transition, "next_status", None)
        if ns is None:
            return 1.0
        return 0.0 if float(ns[cls._TIMEOUT_IDX]) == 1.0 else 1.0

    @classmethod
    def _airl_transitions_to_tensors(cls, transitions, next_observations):
        obs = th.tensor(np.array([t.observation for t in transitions]), dtype=th.float32)
        acts = th.tensor(np.array([t.action for t in transitions]), dtype=th.float32)
        next_obs = th.tensor(np.array(next_observations), dtype=th.float32)
        terminated = th.tensor(np.array([cls._is_terminated(t) for t in transitions]), dtype=th.float32)
        return obs, acts, next_obs, terminated
