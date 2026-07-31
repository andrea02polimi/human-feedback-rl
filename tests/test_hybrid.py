import numpy as np
import pytest
import torch as th
from stable_baselines3 import SAC

from human_feedback_rl.algorithms import HybridAlgorithm
from human_feedback_rl.algorithms.hybrid.gradient_statistics import HybridGradientStats
from human_feedback_rl.algorithms.hybrid.reward_training import RewardTrainingMixin
from human_feedback_rl.common.replay_buffers import RewardRelabelReplayBuffer

from conftest import FakeVecEnv, make_trajectories

RM_KWARGS = dict(n_ensembles=2, net_arch=[8])


def _sac(env):
    return SAC(
        "MlpPolicy", env, buffer_size=500, learning_starts=0, batch_size=16,
        train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[16]),
        replay_buffer_class=RewardRelabelReplayBuffer, seed=0, verbose=0,
    )


def _hybrid(rng, **overrides):
    env = FakeVecEnv(num_envs=2, episode_len=10)
    kwargs = dict(
        expert_trajectories=make_trajectories(rng, [10, 10, 10]),
        loss_type="demo_2",
        gradient_steps_rew=2,
        batch_size_expert=2,
        batch_size_model=2,
        batch_size_pref=4,
        total_queries=8,
        preference_fragment_length=3,
        relabel_rewards=True,
        reward_model_kwargs=RM_KWARGS,
        rng=np.random.default_rng(0),
        output_formats=[],
    )
    kwargs.update(overrides)
    return HybridAlgorithm(env, _sac(env), **kwargs)


# ---------------------------------------------------------------------------
# Norm-balanced fusion math on hand-built gradients
# ---------------------------------------------------------------------------

class _TwoParam(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = th.nn.Parameter(th.zeros(2))


class _StepShim:
    """Carries only what _reward_step needs."""

    _reward_step = HybridAlgorithm._reward_step
    _demo_anchor_coefficients = HybridAlgorithm._demo_anchor_coefficients
    _flatten = staticmethod(HybridAlgorithm._flatten)
    _flatten_grads = HybridAlgorithm._flatten_grads
    _grad_norm = staticmethod(RewardTrainingMixin._grad_norm)

    def __init__(
        self,
        demo_weight=1.0,
        max_balance_scale=100.0,
        gradient_fusion="norm_balanced",
        reliability_lambda_max=1.0,
        reliability_state=None,
    ):
        self.demo_weight = demo_weight
        self.max_balance_scale = max_balance_scale
        self.balance_eps = 1e-12
        self.gradient_fusion = gradient_fusion
        self.reliability_lambda_max = reliability_lambda_max
        self._reliability_state = reliability_state or {}


def _run_step(g_pref, g_demo, grad_stats=None, **shim_kwargs):
    """Apply _reward_step with losses whose gradients are exactly g_pref/g_demo."""
    member = _TwoParam()
    optimizer = th.optim.SGD(member.parameters(), lr=0.0)
    pref_loss = (th.tensor(g_pref) * member.w).sum()
    demo_loss = (th.tensor(g_demo) * member.w).sum()
    shim = _StepShim(**shim_kwargs)
    stats = shim._reward_step(member, optimizer, pref_loss, demo_loss, grad_stats=grad_stats)
    return member.w.grad.detach().numpy(), stats


def test_norm_balanced_sum():
    g_p, g_d = [1.0, 0.0], [-0.5, 0.5]
    grad, stats = _run_step(g_p, g_d)
    scale = np.linalg.norm(g_p) / np.linalg.norm(g_d)  # demo_weight=1
    expected = np.array(g_p) + scale * np.array(g_d)
    assert np.allclose(grad, expected, atol=1e-6)
    assert stats["scale"] == pytest.approx(scale, rel=1e-6)


def test_matches_combined_backward():
    """Per-parameter composition == backward of pref + scale*demo."""
    g_p, g_d = [0.3, -0.7], [0.9, 0.4]
    grad, stats = _run_step(g_p, g_d)

    member = _TwoParam()
    combined = (th.tensor(g_p) * member.w).sum() + stats["scale"] * (th.tensor(g_d) * member.w).sum()
    combined.backward()
    assert np.allclose(grad, member.w.grad.numpy(), atol=1e-6)


def test_balance_scale_uses_demo_weight_and_clamp():
    _, stats = _run_step([2.0, 0.0], [0.5, 0.0], demo_weight=3.0)
    assert stats["scale"] == pytest.approx(3.0 * 2.0 / 0.5, rel=1e-6)
    _, stats = _run_step([2.0, 0.0], [0.001, 0.0], max_balance_scale=10.0)
    assert stats["scale"] == 10.0


def test_demo_anchor_warmup_is_exact_demo_only():
    g_p, g_d = [7.0, -2.0], [0.5, 1.5]
    grad, stats = _run_step(g_p, g_d, gradient_fusion="demo_anchor_reliability")
    assert np.allclose(grad, g_d)
    assert stats["pref_scale"] == 0.0
    assert stats["reliability_lambda"] == 0.0
    assert stats["reliability_fallback"] == 1.0


def test_demo_anchor_reliability_math_leaves_demo_unscaled():
    member = _TwoParam()
    optimizer = th.optim.SGD(member.parameters(), lr=0.0)
    state = {
        id(member): {
            "pref_norm": 4.0,
            "demo_norm": 2.0,
            "pref_var_ratio": 16.0,
            "demo_var_ratio": 4.0,
        }
    }
    shim = _StepShim(
        gradient_fusion="demo_anchor_reliability", reliability_state=state
    )
    g_p, g_d = np.array([4.0, 0.0]), np.array([0.0, 2.0])
    stats = shim._reward_step(
        member,
        optimizer,
        (th.tensor(g_p) * member.w).sum(),
        (th.tensor(g_d) * member.w).sum(),
    )
    # lambda=sqrt(4/16)=0.5 and EMA norm ratio=2/4, so pref_scale=0.25.
    assert stats["reliability_lambda"] == pytest.approx(0.5)
    assert stats["pref_scale"] == pytest.approx(0.25)
    assert stats["demo_scale"] == 1.0
    assert np.allclose(member.w.grad.numpy(), g_d + 0.25 * g_p)


def test_demo_anchor_noisy_preference_is_downweighted():
    member = _TwoParam()
    state = {
        id(member): {
            "pref_norm": 1.0,
            "demo_norm": 1.0,
            "pref_var_ratio": 100.0,
            "demo_var_ratio": 1.0,
        }
    }
    shim = _StepShim(
        gradient_fusion="demo_anchor_reliability", reliability_state=state
    )
    coefficients = shim._demo_anchor_coefficients(member, pref_norm=1.0)
    assert coefficients["reliability_lambda"] == pytest.approx(0.1)
    assert coefficients["pref_scale"] == pytest.approx(0.1)


def test_demo_anchor_zero_preference_gradient_falls_back_to_demo():
    member = _TwoParam()
    state = {
        id(member): {
            "pref_norm": 1.0,
            "demo_norm": 1.0,
            "pref_var_ratio": 1.0,
            "demo_var_ratio": 1.0,
        }
    }
    shim = _StepShim(
        gradient_fusion="demo_anchor_reliability", reliability_state=state
    )
    coefficients = shim._demo_anchor_coefficients(member, pref_norm=0.0)
    assert coefficients["pref_scale"] == 0.0
    assert coefficients["reliability_fallback"] == 1.0


def test_pref_only_and_demo_only_steps():
    member = _TwoParam()
    optimizer = th.optim.SGD(member.parameters(), lr=0.0)
    shim = _StepShim()
    stats = shim._reward_step(member, optimizer, (th.tensor([1.0, 2.0]) * member.w).sum(), None)
    assert np.allclose(member.w.grad.numpy(), [1.0, 2.0])
    assert np.isnan(stats["demo_loss"])
    stats = shim._reward_step(member, optimizer, None, (th.tensor([3.0, 0.0]) * member.w).sum())
    assert np.allclose(member.w.grad.numpy(), [3.0, 0.0])
    assert np.isnan(stats["pref_loss"])


# ---------------------------------------------------------------------------
# Gradient diagnostics wiring
# ---------------------------------------------------------------------------

def test_grad_stats_record_the_raw_gradients_not_the_balanced_ones():
    """The diagnostics must describe the two feedback signals themselves.

    ``scale`` rescales the demo gradient by two orders of magnitude here; a
    diagnostic taken after the balancing would report that instead of the
    scale conflict it is supposed to expose.
    """
    g_p, g_d = [3.0, 4.0], [0.1, 0.0]
    grad_stats = HybridGradientStats()
    grad, stats = _run_step(g_p, g_d, grad_stats=grad_stats)

    assert stats["scale"] == pytest.approx(50.0, rel=1e-6)  # demo upscaled 50x
    assert grad_stats.pref.sq_norm == pytest.approx(25.0, rel=1e-9)
    # 0.01, the raw demo gradient: not 25.0, which is what the balanced
    # gradient would report once scale has matched it to the preference norm.
    assert grad_stats.demo.sq_norm == pytest.approx(0.01, rel=1e-6)
    # The composed gradient that was actually applied is a different vector.
    assert not np.allclose(grad, g_p)


def test_grad_stats_cosine_from_a_full_step():
    grad_stats = HybridGradientStats()
    _run_step([1.0, 0.0], [0.0, 2.0], grad_stats=grad_stats)
    assert grad_stats.metrics()["grad_cosine"] == pytest.approx(0.0, abs=1e-9)


def test_grad_stats_record_the_single_channel_of_degenerate_arms():
    member = _TwoParam()
    optimizer = th.optim.SGD(member.parameters(), lr=0.0)
    shim = _StepShim()

    pref_only = HybridGradientStats()
    shim._reward_step(
        member, optimizer, (th.tensor([1.0, 2.0]) * member.w).sum(), None,
        grad_stats=pref_only,
    )
    assert pref_only.pref.count == 1 and pref_only.demo.count == 0
    assert pref_only.pref.sq_norm == pytest.approx(5.0, rel=1e-9)

    demo_only = HybridGradientStats()
    shim._reward_step(
        member, optimizer, None, (th.tensor([3.0, 0.0]) * member.w).sum(),
        grad_stats=demo_only,
    )
    assert demo_only.demo.count == 1 and demo_only.pref.count == 0
    assert demo_only.demo.sq_norm == pytest.approx(9.0, rel=1e-9)


def _train_reward_model_once(algo, n_queries=8):
    algo.trajectories = make_trajectories(np.random.default_rng(5), [10, 10, 10])
    algo._collect_feedback(n_queries)
    algo._train_reward_model()
    return algo.logger.name_to_value


def test_hybrid_iteration_logs_every_requested_curve(rng):
    recorded = _train_reward_model_once(_hybrid(rng, gradient_steps_rew=4))
    for key in (
        "reward/grad_var_pref", "reward/grad_var_demo",
        "reward/grad_sq_norm_pref", "reward/grad_sq_norm_demo",
        "reward/grad_var_ratio_pref", "reward/grad_var_ratio_demo",
        "reward/grad_cosine",
    ):
        assert key in recorded, key
        assert np.isfinite(recorded[key]), key
    # Variance over the gradient steps of the iteration, one point per
    # iteration: 4 steps means 4 samples per member.
    assert recorded["reward/grad_diagnostics_samples"] == pytest.approx(4.0)


def test_demo_anchor_uses_previous_call_state_and_logs_coefficients(rng):
    algo = _hybrid(
        rng,
        gradient_steps_rew=4,
        gradient_fusion="demo_anchor_reliability",
        reliability_ema_beta=0.9,
    )
    first = _train_reward_model_once(algo)
    assert first["reward/hybrid_reliability_fallback_rate"] == 1.0
    assert len(algo._reliability_state) == len(algo.reward_model.members)

    second = _train_reward_model_once(algo)
    assert second["reward/hybrid_reliability_fallback_rate"] == 0.0
    assert 0.0 <= second["reward/hybrid_reliability_lambda"] <= 1.0
    assert second["reward/hybrid_demo_scale"] == 1.0
    assert np.isfinite(second["reward/hybrid_pref_scale"])


def test_grad_diagnostics_interval_subsamples_the_steps(rng):
    recorded = _train_reward_model_once(
        _hybrid(rng, gradient_steps_rew=8, grad_diagnostics_interval=4)
    )
    assert recorded["reward/grad_diagnostics_samples"] == pytest.approx(2.0)


def test_grad_diagnostics_can_be_turned_off(rng):
    recorded = _train_reward_model_once(
        _hybrid(rng, gradient_steps_rew=4, grad_diagnostics_interval=0)
    )
    assert not [key for key in recorded if key.startswith("reward/grad_var")]
    # The pre-existing norm logging is untouched by the switch.
    assert "reward/grad_norm_pref" in recorded


def test_composed_gradient_and_directional_curves_are_logged(rng):
    recorded = _train_reward_model_once(_hybrid(rng, gradient_steps_rew=4))
    for key in (
        "reward/grad_dir_var_pref", "reward/grad_dir_var_demo",
        "reward/grad_dir_cosine_of_means",
        "reward/grad_var_total", "reward/grad_sq_norm_total",
        "reward/grad_var_ratio_total", "reward/grad_dir_var_total",
    ):
        assert key in recorded, key
        assert np.isfinite(recorded[key]), key


def test_total_channel_records_the_balanced_gradient_not_a_raw_one():
    """_total must be g_pref + scale*g_demo, i.e. what .grad actually holds."""
    g_p, g_d = [3.0, 4.0], [0.1, 0.0]
    grad_stats = HybridGradientStats()
    grad, stats = _run_step(g_p, g_d, grad_stats=grad_stats)

    applied_sq_norm = float(np.dot(grad, grad))
    assert grad_stats.total.sq_norm == pytest.approx(applied_sq_norm, rel=1e-6)
    # Distinct from both sources: the balancing changed the vector.
    assert grad_stats.total.sq_norm != pytest.approx(grad_stats.pref.sq_norm, rel=1e-3)
    assert grad_stats.total.sq_norm != pytest.approx(grad_stats.demo.sq_norm, rel=1e-3)


def test_degenerate_arms_report_total_equal_to_their_only_channel(rng):
    recorded = _train_reward_model_once(
        _hybrid(rng, total_queries=0, gradient_steps_rew=4), n_queries=0
    )
    assert recorded["reward/grad_var_total"] == pytest.approx(
        recorded["reward/grad_var_demo"], rel=1e-9
    )


def test_noise_fraction_curves_are_logged(rng):
    """Var / E[||g||^2], per channel, alongside the existing ratio."""
    recorded = _train_reward_model_once(_hybrid(rng, gradient_steps_rew=4))
    for channel in ("pref", "demo", "total"):
        key = f"reward/grad_noise_fraction_{channel}"
        assert key in recorded, key
        assert 0.0 <= recorded[key] <= 4 / 3 + 1e-9, recorded[key]
    # The pre-existing ratio is untouched by the addition.
    assert "reward/grad_var_ratio_pref" in recorded


def test_adam_style_curves_are_logged_in_both_beta_variants(rng):
    recorded = _train_reward_model_once(_hybrid(rng, gradient_steps_rew=4))
    for prefix in ("reward/grad_adam", "reward/grad_adam_eq"):
        for channel in ("pref", "demo"):
            for quantity in ("var", "sq_norm", "mean_sq_norm", "var_ratio"):
                key = f"{prefix}_{quantity}_{channel}"
                assert key in recorded, key
                assert np.isfinite(recorded[key]), key


def test_adam_state_persists_across_iterations(rng):
    """Adam's averages are only meaningful if they are never reset."""
    algo = _hybrid(rng, gradient_steps_rew=4)
    _train_reward_model_once(algo)
    after_one = [
        stats.variants["grad_adam"]["pref"].count
        for stats in algo._adam_grad_stats.values()
    ]
    _train_reward_model_once(algo)
    after_two = [
        stats.variants["grad_adam"]["pref"].count
        for stats in algo._adam_grad_stats.values()
    ]

    assert after_one == [4, 4]  # 4 gradient steps, one estimator per member
    assert after_two == [8, 8]
    # One estimator per ensemble member, kept apart.
    assert len(algo._adam_grad_stats) == len(algo.reward_model.members)


def test_adam_betas_come_from_the_reward_optimizer(rng):
    algo = _hybrid(rng, gradient_steps_rew=2)
    _train_reward_model_once(algo)
    expected = algo.optimizers[0].param_groups[0]["betas"]
    estimator = next(iter(algo._adam_grad_stats.values())).variants["grad_adam"]["pref"]
    assert (estimator.beta1, estimator.beta2) == pytest.approx(expected)


def test_eq_beta_variant_uses_the_configured_beta(rng):
    algo = _hybrid(rng, gradient_steps_rew=2, grad_diagnostics_eq_beta=0.5)
    _train_reward_model_once(algo)
    estimator = next(iter(algo._adam_grad_stats.values())).variants["grad_adam_eq"]["pref"]
    assert (estimator.beta1, estimator.beta2) == (0.5, 0.5)


def test_demo_only_arm_logs_its_channel_without_a_cosine(rng):
    recorded = _train_reward_model_once(
        _hybrid(rng, total_queries=0, gradient_steps_rew=4), n_queries=0
    )
    assert "reward/grad_var_demo" in recorded
    assert "reward/grad_var_pref" not in recorded
    assert "reward/grad_cosine" not in recorded


# ---------------------------------------------------------------------------
# Constructor semantics
# ---------------------------------------------------------------------------

def test_pref_temperature_reaches_the_gatherer(rng):
    algo = _hybrid(rng, pref_temperature=20.0)
    assert algo.preference_gatherer.temperature == 20.0


@pytest.mark.parametrize("bad", [
    dict(demo_mode="nope"),
    dict(pref_temperature=0.0), dict(demo_pref_batch_fraction=1.5),
    dict(gradient_fusion="nope"), dict(reliability_ema_beta=1.0),
    dict(reliability_lambda_max=1.1),
])
def test_invalid_kwargs_raise(rng, bad):
    with pytest.raises(ValueError):
        _hybrid(rng, **bad)


# ---------------------------------------------------------------------------
# Demos-as-preferences baseline
# ---------------------------------------------------------------------------

def test_demo_preference_pairs_expert_first_with_hard_labels(rng):
    algo = _hybrid(rng, demo_mode="preferences", demo_pref_pairs_per_iteration=12)
    algo.trajectories = make_trajectories(np.random.default_rng(5), [10, 10])
    algo._collect_demo_preference_pairs(12)

    n = len(algo.dataset_demo_prefs_train) + len(algo.dataset_demo_prefs_val)
    assert n == 12
    expert_transitions = {id(t) for traj in algo.expert_trajectories for t in traj}
    batch = algo.dataset_demo_prefs_train.get_all()
    for pair, pref in zip(batch.fragment_pairs, batch.preferences):
        assert (pref.pref1, pref.pref2) == (1.0, 0.0)
        assert all(id(t) in expert_transitions for t in pair.frag1)
        assert all(id(t) not in expert_transitions for t in pair.frag2)


# ---------------------------------------------------------------------------
# End-to-end smoke per mode
# ---------------------------------------------------------------------------

def test_hybrid_gcl_trains(rng):
    algo = _hybrid(rng)
    agent = algo.train(
        total_timesteps=64, timesteps_per_iteration=32, log_interval=100, scatter_interval=0
    )
    assert agent is algo.agent
    assert len(algo.dataset_train) > 0


def test_hybrid_demos_as_preferences_trains(rng):
    algo = _hybrid(rng, demo_mode="preferences", demo_pref_pairs_per_iteration=8)
    agent = algo.train(
        total_timesteps=64, timesteps_per_iteration=32, log_interval=100, scatter_interval=0
    )
    assert agent is algo.agent
    assert len(algo.dataset_demo_prefs_train) > 0


def test_hybrid_demo_only_arm_trains(rng):
    """total_queries=0 must still train the reward model on the demo loss alone."""
    algo = _hybrid(rng, total_queries=0)
    algo.train(
        total_timesteps=64, timesteps_per_iteration=32, log_interval=100, scatter_interval=0
    )
    assert len(algo.dataset_train) == 0
    assert algo.reward_model.normalization_std > 0


def test_hybrid_pref_only_arm_trains(rng):
    algo = _hybrid(rng, demo_weight=0.0)
    algo.train(
        total_timesteps=64, timesteps_per_iteration=32, log_interval=100, scatter_interval=0
    )
    assert len(algo.dataset_train) > 0
