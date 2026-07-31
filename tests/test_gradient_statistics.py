"""Gradient variance, squared norms and channel angle, against closed forms."""

import math

import numpy as np
import pytest
import torch as th

from human_feedback_rl.algorithms.hybrid.gradient_statistics import (
    AdamGradientStats,
    AdamVarianceEstimator,
    GradientChannelStats,
    HybridGradientStats,
    average_metrics,
)


def gradients(rng, n, dim=6, scale=1.0, offset=0.0):
    return [th.tensor(rng.normal(offset, scale, dim), dtype=th.float32) for _ in range(n)]


# ---------------------------------------------------------------------------
# One channel
# ---------------------------------------------------------------------------

def test_variance_and_norms_match_numpy():
    rng = np.random.default_rng(0)
    grads = gradients(rng, 40)
    stats = GradientChannelStats()
    for g in grads:
        stats.update(g)

    stacked = np.stack([g.numpy().astype(np.float64) for g in grads])
    mean = stacked.mean(axis=0)
    # tr(Cov): the variance of each component, summed.
    expected_var = stacked.var(axis=0, ddof=1).sum()

    assert stats.count == 40
    assert stats.var == pytest.approx(expected_var, rel=1e-9)
    assert stats.mean_sq_norm == pytest.approx(mean @ mean, rel=1e-9)
    assert stats.sq_norm == pytest.approx((stacked ** 2).sum(axis=1).mean(), rel=1e-9)
    assert stats.var_ratio == pytest.approx(expected_var / (mean @ mean), rel=1e-9)


def test_identical_gradients_have_zero_variance():
    stats = GradientChannelStats()
    for _ in range(5):
        stats.update(th.tensor([1.0, -2.0, 0.5]))
    assert stats.var == pytest.approx(0.0, abs=1e-12)
    assert stats.var_ratio == pytest.approx(0.0, abs=1e-12)
    assert stats.mean_sq_norm == pytest.approx(5.25, rel=1e-9)
    assert stats.sq_norm == pytest.approx(5.25, rel=1e-9)


def test_variance_is_undefined_below_two_samples():
    stats = GradientChannelStats()
    assert math.isnan(stats.var)
    assert math.isnan(stats.sq_norm)
    stats.update(th.tensor([1.0, 0.0]))
    assert math.isnan(stats.var)
    assert stats.sq_norm == pytest.approx(1.0)


def test_ratio_is_undefined_when_the_mean_gradient_vanishes():
    """Symmetric gradients cancel: a ratio there would divide by zero."""
    stats = GradientChannelStats()
    stats.update(th.tensor([1.0, 0.0]))
    stats.update(th.tensor([-1.0, 0.0]))
    assert stats.mean_sq_norm == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(stats.var_ratio)
    assert stats.var == pytest.approx(2.0, rel=1e-9)


def test_streaming_matches_a_two_pass_computation_on_a_large_offset():
    """Welford holds where E[||g||^2] - ||E[g]||^2 loses its significant digits."""
    rng = np.random.default_rng(1)
    grads = gradients(rng, 60, dim=8, scale=1e-3, offset=1e4)
    stats = GradientChannelStats()
    for g in grads:
        stats.update(g)

    stacked = np.stack([g.numpy().astype(np.float64) for g in grads])
    assert stats.var == pytest.approx(stacked.var(axis=0, ddof=1).sum(), rel=1e-6)


def test_shapes_are_flattened():
    """Gradients arrive per-parameter-tensor; only the flat vector matters."""
    flat = GradientChannelStats()
    nested = GradientChannelStats()
    for values in ([1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 5.0, 2.0]):
        flat.update(th.tensor(values))
        nested.update(th.tensor(values).reshape(2, 2))
    assert nested.var == pytest.approx(flat.var, rel=1e-12)
    assert nested.mean_sq_norm == pytest.approx(flat.mean_sq_norm, rel=1e-12)


# ---------------------------------------------------------------------------
# The angle between the two channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g_demo,expected", [
    ([1.0, 0.0], 1.0),      # aligned
    ([-1.0, 0.0], -1.0),    # exactly opposed: the conflict case
    ([0.0, 1.0], 0.0),      # orthogonal
    ([1.0, 1.0], math.sqrt(0.5)),  # 45 degrees
])
def test_cosine_of_known_angles(g_demo, expected):
    stats = HybridGradientStats()
    stats.update(th.tensor([1.0, 0.0]), th.tensor(g_demo))
    assert stats.metrics()["grad_cosine"] == pytest.approx(expected, abs=1e-9)


def test_cosine_is_invariant_to_the_channel_magnitudes():
    """It must describe the angle, not the scale conflict the norms show."""
    small = HybridGradientStats()
    small.update(th.tensor([1.0, 1.0]), th.tensor([0.0, 1.0]))
    large = HybridGradientStats()
    large.update(th.tensor([1.0, 1.0]), th.tensor([0.0, 1e6]))
    assert small.metrics()["grad_cosine"] == pytest.approx(large.metrics()["grad_cosine"])


def test_mean_cosine_and_cosine_of_means_differ_when_the_angle_flips():
    """Channels that alternate agreement average to zero but do not cancel."""
    stats = HybridGradientStats()
    stats.update(th.tensor([1.0, 0.0]), th.tensor([1.0, 0.0]))
    stats.update(th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]))
    metrics = stats.metrics()
    assert metrics["grad_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["grad_cosine_std"] == pytest.approx(math.sqrt(2.0), rel=1e-9)
    # The mean demo gradient vanished, so the angle between the means is gone.
    assert "grad_cosine_of_means" not in metrics


def test_vanishing_gradient_produces_no_cosine():
    stats = HybridGradientStats()
    stats.update(th.tensor([1.0, 0.0]), th.tensor([0.0, 0.0]))
    assert "grad_cosine" not in stats.metrics()


# ---------------------------------------------------------------------------
# Noise fraction: var / E[||g||^2]
# ---------------------------------------------------------------------------

def test_noise_fraction_matches_its_definition():
    rng = np.random.default_rng(51)
    grads = gradients(rng, 30, dim=6, scale=1.5, offset=0.8)
    stats = GradientChannelStats()
    for g in grads:
        stats.update(g)

    stacked = np.stack([g.numpy().astype(np.float64) for g in grads])
    expected = (
        stacked.var(axis=0, ddof=1).sum() / (stacked ** 2).sum(axis=1).mean()
    )
    assert stats.noise_fraction == pytest.approx(expected, rel=1e-9)


def test_noise_fraction_is_a_reparametrization_of_var_ratio():
    """noise_fraction = r / (1 + r*(T-1)/T): same information, bounded scale."""
    rng = np.random.default_rng(53)
    grads = gradients(rng, 40, dim=5, scale=2.0, offset=1.2)
    stats = GradientChannelStats()
    for g in grads:
        stats.update(g)

    n = len(grads)
    r = stats.var_ratio
    assert stats.noise_fraction == pytest.approx(r / (1 + r * (n - 1) / n), rel=1e-9)


def test_noise_fraction_is_zero_without_noise():
    stats = GradientChannelStats()
    for _ in range(6):
        stats.update(th.tensor([2.0, -1.0]))
    assert stats.noise_fraction == pytest.approx(0.0, abs=1e-12)


def test_noise_fraction_reaches_its_ceiling_when_the_mean_vanishes():
    """The maximum is T/(T-1), not 1: var is unbiased, the identity is not."""
    stats = GradientChannelStats()
    stats.update(th.tensor([1.0, 0.0]))
    stats.update(th.tensor([-1.0, 0.0]))
    assert stats.mean_sq_norm == pytest.approx(0.0, abs=1e-12)
    assert stats.noise_fraction == pytest.approx(2 / 1, rel=1e-12)  # T=2


@pytest.mark.parametrize("n", [3, 10, 50])
def test_noise_fraction_never_exceeds_the_ceiling(n):
    rng = np.random.default_rng(59)
    stats = GradientChannelStats()
    for g in gradients(rng, n, dim=4, scale=3.0, offset=0.0):
        stats.update(g)
    assert 0.0 <= stats.noise_fraction <= n / (n - 1) + 1e-9


def test_noise_fraction_is_invariant_to_rescaling():
    """Scale-free, which is the point of reporting a ratio at all."""
    rng = np.random.default_rng(61)
    grads = gradients(rng, 12, dim=4, scale=1.0, offset=0.5)
    plain, scaled = GradientChannelStats(), GradientChannelStats()
    for g in grads:
        plain.update(g)
        scaled.update(g * 1e4)
    assert plain.noise_fraction == pytest.approx(scaled.noise_fraction, rel=1e-6)


def test_noise_fraction_is_undefined_without_data():
    stats = GradientChannelStats()
    assert math.isnan(stats.noise_fraction)
    stats.update(th.tensor([1.0, 0.0]))
    assert math.isnan(stats.noise_fraction)  # var needs two samples


# ---------------------------------------------------------------------------
# Directional (unit-normalized) variance
# ---------------------------------------------------------------------------

def test_directional_variance_matches_the_closed_form():
    """For unit vectors tr(Cov) = (T/(T-1)) * (1 - R^2), R = ||mean direction||."""
    rng = np.random.default_rng(31)
    grads = gradients(rng, 25, dim=5, scale=2.0, offset=0.7)
    stats = GradientChannelStats()
    for g in grads:
        stats.update(g)

    units = np.stack([
        g.numpy().astype(np.float64) / np.linalg.norm(g.numpy().astype(np.float64))
        for g in grads
    ])
    resultant = units.mean(axis=0)
    n = len(grads)
    expected = (n / (n - 1)) * (1.0 - resultant @ resultant)
    assert stats.dir_var == pytest.approx(expected, rel=1e-9)


def test_directional_variance_ignores_magnitude_fluctuations():
    """Same directions, wildly different norms -> no directional variance.

    This is exactly what normalizing per step discards, and why it is
    reported alongside the raw variance rather than instead of it.
    """
    stats = GradientChannelStats()
    for k in (1e-6, 1.0, 1e3, 7.5):
        stats.update(th.tensor([3.0, 4.0]) * k)
    assert stats.dir_var == pytest.approx(0.0, abs=1e-12)
    # The raw variance, by contrast, is dominated by those magnitudes.
    assert stats.var > 1e5


def test_directional_variance_is_maximal_for_opposed_directions():
    stats = GradientChannelStats()
    stats.update(th.tensor([1.0, 0.0]))
    stats.update(th.tensor([-1.0, 0.0]))
    # R = 0, so tr(Cov) = (2/1) * (1 - 0) = 2.
    assert stats.dir_var == pytest.approx(2.0, rel=1e-12)


def test_zero_gradient_has_no_direction_to_record():
    stats = GradientChannelStats()
    stats.update(th.tensor([1.0, 0.0]))
    stats.update(th.zeros(2))
    stats.update(th.tensor([0.0, 1.0]))
    assert stats.count == 3          # the raw stream keeps every step
    assert stats.unit.count == 2     # the direction stream skips the null one


def test_normalized_stream_degenerates_as_documented():
    """Why sq_norm and var_ratio are NOT reported on the unit stream."""
    stats = GradientChannelStats()
    for g in gradients(np.random.default_rng(37), 10, dim=4, scale=2.0, offset=1.0):
        stats.update(g)
    # ||u||^2 = 1 identically, so a "normalized squared norm" carries nothing.
    assert stats.unit.sq_norm == pytest.approx(1.0, rel=1e-12)
    # And var/||mean||^2 on that stream is just a reparametrization of dir_var.
    resultant_sq = stats.unit.mean_sq_norm
    assert stats.unit.var / resultant_sq == pytest.approx(
        stats.dir_var / resultant_sq, rel=1e-12
    )


def test_directional_cosine_of_means_is_not_magnitude_weighted():
    """One huge step must not decide the sign of the systematic angle.

    Preference and demo agree on a single enormous step and disagree on many
    small ones. The magnitude-weighted angle follows the big step; the
    directional one follows the majority.
    """
    stats = HybridGradientStats()
    stats.update(th.tensor([1.0, 0.0]) * 1000.0, th.tensor([1.0, 0.0]) * 1000.0)
    for _ in range(10):
        stats.update(th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]))
    metrics = stats.metrics()
    assert metrics["grad_cosine_of_means"] > 0.9      # dragged by the big step
    assert metrics["grad_dir_cosine_of_means"] < -0.5  # follows the majority


def test_per_step_cosine_is_unaffected_by_normalization():
    """The cosine is already scale-free, so there is nothing to normalize."""
    plain = HybridGradientStats()
    scaled = HybridGradientStats()
    rng = np.random.default_rng(41)
    for g_pref, g_demo in zip(gradients(rng, 6), gradients(rng, 6)):
        plain.update(g_pref, g_demo)
        scaled.update(g_pref * 1e5, g_demo * 1e-3)
    # rel=1e-6: rescaling float32 tensors by 1e5 and 1e-3 costs a few ulp.
    assert plain.metrics()["grad_cosine"] == pytest.approx(
        scaled.metrics()["grad_cosine"], rel=1e-6
    )


# ---------------------------------------------------------------------------
# The composed gradient
# ---------------------------------------------------------------------------

def test_total_channel_is_tracked_separately_from_the_two_sources():
    stats = HybridGradientStats()
    g_pref = th.tensor([1.0, 0.0])
    g_demo = th.tensor([0.0, 2.0])
    for _ in range(4):
        stats.update(g_pref=g_pref, g_demo=g_demo, g_total=g_pref + 0.5 * g_demo)
    metrics = stats.metrics()
    assert metrics["grad_sq_norm_pref"] == pytest.approx(1.0, rel=1e-9)
    assert metrics["grad_sq_norm_demo"] == pytest.approx(4.0, rel=1e-9)
    assert metrics["grad_sq_norm_total"] == pytest.approx(2.0, rel=1e-9)


def test_total_channel_absent_when_not_recorded():
    stats = HybridGradientStats()
    stats.update(g_pref=th.tensor([1.0, 0.0]), g_demo=th.tensor([0.0, 1.0]))
    assert not [key for key in stats.metrics() if key.endswith("_total")]


# ---------------------------------------------------------------------------
# Adam-style estimator
# ---------------------------------------------------------------------------

def test_adam_moments_follow_the_update_rule():
    """m and v against the recurrence, worked out by hand.

    b1 = b2 = 0.5, gradients [1] then [3]:
        t=1  m = .5,          v = .5
             m_hat = 1,       v_hat = 1        -> var = 0
        t=2  m = .5(.5)+.5(3) = 1.75
             v = .5(.5)+.5(9) = 4.75
             m_hat = 1.75/.75 = 7/3
             v_hat = 4.75/.75 = 19/3           -> var = 19/3 - 49/9 = 8/9
    """
    estimator = AdamVarianceEstimator(0.5, 0.5)
    estimator.update(th.tensor([1.0]))
    assert estimator.var == pytest.approx(0.0, abs=1e-12)
    assert estimator.mean_sq_norm == pytest.approx(1.0, rel=1e-12)
    assert estimator.sq_norm == pytest.approx(1.0, rel=1e-12)

    estimator.update(th.tensor([3.0]))
    assert estimator.mean_sq_norm == pytest.approx((7 / 3) ** 2, rel=1e-12)
    assert estimator.sq_norm == pytest.approx(19 / 3, rel=1e-12)
    assert estimator.var == pytest.approx(8 / 9, rel=1e-12)
    assert estimator.var_ratio == pytest.approx((8 / 9) / (7 / 3) ** 2, rel=1e-12)


@pytest.mark.parametrize("betas", [(0.5, 0.5), (0.9, 0.999), (0.0, 0.9)])
def test_bias_correction_makes_the_first_step_exact(betas):
    """At t=1, m_hat = g and v_hat = g^2 whatever the betas, so var is 0."""
    estimator = AdamVarianceEstimator(*betas)
    estimator.update(th.tensor([2.0, -1.0]))
    assert estimator.mean_sq_norm == pytest.approx(5.0, rel=1e-12)
    assert estimator.sq_norm == pytest.approx(5.0, rel=1e-12)
    assert estimator.var == pytest.approx(0.0, abs=1e-12)


def test_constant_gradient_has_no_variance():
    estimator = AdamVarianceEstimator(0.9, 0.999)
    for _ in range(50):
        estimator.update(th.tensor([1.0, -2.0]))
    assert estimator.var == pytest.approx(0.0, abs=1e-9)
    assert estimator.mean_sq_norm == pytest.approx(5.0, rel=1e-6)


def test_mismatched_betas_can_yield_a_negative_variance():
    """The documented caveat, pinned so it is never mistaken for a bug.

    b1 = 0.5, b2 = 0.9, gradients [1] then [3]:
        m_hat = 1.75/.75 = 7/3,  v_hat = 0.99/0.19
        var = 0.99/0.19 - 49/9 < 0
    """
    estimator = AdamVarianceEstimator(0.5, 0.9)
    estimator.update(th.tensor([1.0]))
    estimator.update(th.tensor([3.0]))
    assert estimator.var == pytest.approx(0.99 / 0.19 - (7 / 3) ** 2, rel=1e-12)
    assert estimator.var < 0


def test_equal_betas_keep_the_variance_non_negative():
    """The reason the grad_adam_eq_* variant exists."""
    rng = np.random.default_rng(7)
    estimator = AdamVarianceEstimator(0.9, 0.9)
    for g in gradients(rng, 200, dim=5, scale=3.0, offset=0.5):
        estimator.update(g)
        assert estimator.var >= -1e-9


def test_variance_is_undefined_before_any_update():
    estimator = AdamVarianceEstimator(0.9, 0.999)
    assert math.isnan(estimator.var)
    assert math.isnan(estimator.sq_norm)
    assert math.isnan(estimator.var_ratio)


def test_adam_variance_tracks_a_step_change_in_noise():
    """A gradient that starts noisy and goes quiet must show the drop."""
    rng = np.random.default_rng(11)
    estimator = AdamVarianceEstimator(0.9, 0.9)
    for g in gradients(rng, 100, dim=4, scale=5.0, offset=1.0):
        estimator.update(g)
    noisy = estimator.var
    for _ in range(100):
        estimator.update(th.ones(4))
    assert estimator.var < noisy / 10


@pytest.mark.parametrize("betas", [(1.0, 0.9), (0.9, 1.0), (-0.1, 0.9)])
def test_invalid_betas_raise(betas):
    with pytest.raises(ValueError):
        AdamVarianceEstimator(*betas)


# ---------------------------------------------------------------------------
# Both estimators side by side
# ---------------------------------------------------------------------------

def test_adam_state_survives_across_iterations():
    """Reset per iteration would pin the averages to their warm-up."""
    adam = AdamGradientStats(betas=(0.9, 0.999), eq_beta=0.99)
    rng = np.random.default_rng(13)
    for _ in range(3):  # three iterations, a fresh Welford object each time
        stats = HybridGradientStats(adam=adam)
        for g_pref, g_demo in zip(gradients(rng, 4), gradients(rng, 4)):
            stats.update(g_pref, g_demo)
    assert adam.variants["grad_adam"]["pref"].count == 12
    assert adam.variants["grad_adam_eq"]["demo"].count == 12


def test_both_estimators_see_exactly_the_same_gradients():
    adam = AdamGradientStats()
    stats = HybridGradientStats(adam=adam)
    for g in gradients(np.random.default_rng(17), 6):
        stats.update(g_pref=g)
    assert adam.variants["grad_adam"]["pref"].count == stats.pref.count
    assert adam.variants["grad_adam"]["demo"].count == 0


def test_metrics_include_both_beta_variants():
    adam = AdamGradientStats(betas=(0.9, 0.999), eq_beta=0.99)
    stats = HybridGradientStats(adam=adam)
    rng = np.random.default_rng(19)
    for g_pref, g_demo in zip(gradients(rng, 8), gradients(rng, 8)):
        stats.update(g_pref, g_demo)
    metrics = stats.metrics()
    for prefix in ("grad_adam", "grad_adam_eq"):
        for channel in ("pref", "demo"):
            for quantity in ("var", "sq_norm", "mean_sq_norm", "var_ratio"):
                key = f"{prefix}_{quantity}_{channel}"
                assert key in metrics, key
                assert math.isfinite(metrics[key]), key
    # The Welford estimator is still reported alongside them.
    assert "grad_var_pref" in metrics


def test_the_two_estimators_agree_on_a_stationary_gradient():
    """Different windows, same answer when there is nothing to disagree about."""
    adam = AdamGradientStats(betas=(0.9, 0.9), eq_beta=0.9)
    stats = HybridGradientStats(adam=adam)
    for _ in range(200):
        stats.update(g_pref=th.tensor([2.0, -3.0]))
    metrics = stats.metrics()
    assert metrics["grad_var_pref"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["grad_adam_var_pref"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["grad_mean_sq_norm_pref"] == pytest.approx(13.0, rel=1e-6)
    assert metrics["grad_adam_mean_sq_norm_pref"] == pytest.approx(13.0, rel=1e-6)


def test_adam_metrics_absent_when_no_estimator_is_attached():
    stats = HybridGradientStats()
    for g in gradients(np.random.default_rng(23), 4):
        stats.update(g_pref=g)
    assert not [key for key in stats.metrics() if "adam" in key]


# ---------------------------------------------------------------------------
# Metric assembly
# ---------------------------------------------------------------------------

def test_single_channel_arms_report_only_their_channel():
    pref_only = HybridGradientStats()
    for g in gradients(np.random.default_rng(2), 4):
        pref_only.update(g_pref=g)
    metrics = pref_only.metrics()
    assert "grad_var_pref" in metrics
    assert not any(key.endswith("_demo") for key in metrics)
    assert "grad_cosine" not in metrics


def test_metrics_cover_every_requested_quantity():
    stats = HybridGradientStats()
    rng = np.random.default_rng(3)
    for g_pref, g_demo in zip(gradients(rng, 5), gradients(rng, 5)):
        stats.update(g_pref, g_demo)
    metrics = stats.metrics()
    for key in (
        "grad_var_pref", "grad_var_demo",
        "grad_sq_norm_pref", "grad_sq_norm_demo",
        "grad_mean_sq_norm_pref", "grad_mean_sq_norm_demo",
        "grad_var_ratio_pref", "grad_var_ratio_demo",
        "grad_dir_var_pref", "grad_dir_var_demo",
        "grad_cosine", "grad_cosine_of_means", "grad_dir_cosine_of_means",
    ):
        assert key in metrics, key
        assert math.isfinite(metrics[key]), key


def test_average_metrics_skips_missing_and_non_finite_values():
    averaged = average_metrics([
        {"a": 1.0, "b": 2.0},
        {"a": 3.0, "b": math.nan},
        {"a": 5.0},
    ])
    assert averaged["a"] == pytest.approx(3.0)
    assert averaged["b"] == pytest.approx(2.0)


def test_average_metrics_drops_a_key_with_no_finite_value():
    assert average_metrics([{"a": math.nan}, {"a": math.inf}]) == {}


def test_metrics_keep_every_previously_reported_key():
    """Guard against a new metric silently displacing an existing one."""
    stats = GradientChannelStats()
    for g in gradients(np.random.default_rng(67), 5):
        stats.update(g)
    keys = set(stats.metrics("pref"))
    assert keys == {
        "grad_var_pref", "grad_sq_norm_pref", "grad_mean_sq_norm_pref",
        "grad_var_ratio_pref", "grad_dir_var_pref", "grad_noise_fraction_pref",
    }
