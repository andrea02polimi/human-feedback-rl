import numpy as np
import pytest
import torch as th
from stable_baselines3 import SAC

from human_feedback_rl.algorithms import HybridAlgorithm
from human_feedback_rl.algorithms.demo.reward_training import RewardTrainingMixin
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
    _flatten = staticmethod(HybridAlgorithm._flatten)
    _grad_norm = staticmethod(RewardTrainingMixin._grad_norm)

    def __init__(self, demo_weight=1.0, max_balance_scale=100.0):
        self.demo_weight = demo_weight
        self.max_balance_scale = max_balance_scale
        self.balance_eps = 1e-12


def _run_step(g_pref, g_demo, **shim_kwargs):
    """Apply _reward_step with losses whose gradients are exactly g_pref/g_demo."""
    member = _TwoParam()
    optimizer = th.optim.SGD(member.parameters(), lr=0.0)
    pref_loss = (th.tensor(g_pref) * member.w).sum()
    demo_loss = (th.tensor(g_demo) * member.w).sum()
    shim = _StepShim(**shim_kwargs)
    stats = shim._reward_step(member, optimizer, pref_loss, demo_loss)
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
# Constructor semantics
# ---------------------------------------------------------------------------

def test_pref_temperature_reaches_the_gatherer(rng):
    algo = _hybrid(rng, pref_temperature=20.0)
    assert algo.preference_gatherer.temperature == 20.0


@pytest.mark.parametrize("bad", [
    dict(demo_mode="nope"),
    dict(pref_temperature=0.0), dict(demo_pref_batch_fraction=1.5),
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
