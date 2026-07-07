import numpy as np
import pytest
import torch as th

from human_feedback_rl.common.batching import (
    fragment_avg_rewards,
    fragment_sum_rewards,
    per_step_rewards,
    stacked_transitions,
)

from conftest import ConstantRewardNet, make_trajectories


def test_stacked_transitions_shapes_and_memoization(rng):
    (traj,) = make_trajectories(rng, [7])
    tensors = stacked_transitions(traj)
    assert tensors[0].shape == (7, 4) and tensors[1].shape == (7, 2)
    assert tensors[2].shape == (7, 7) and tensors[3].shape == (7,)
    # Memoized: the same tuple object comes back.
    assert stacked_transitions(traj) is tensors


def test_stacked_transitions_cache_invalidated_on_growth(rng):
    (traj,) = make_trajectories(rng, [4])
    first = stacked_transitions(traj)
    (extra,) = make_trajectories(rng, [1])
    traj.add_transition(extra[0])
    second = stacked_transitions(traj)
    assert second is not first
    assert second[0].shape == (5, 4)


def test_fragment_avg_rewards_matches_per_fragment_loop(rng, tiny_reward_ensemble):
    frags = make_trajectories(rng, [3, 8, 5, 1])  # unequal lengths
    member = tiny_reward_ensemble.members[0]
    with th.no_grad():
        batched = fragment_avg_rewards(member, frags)
        looped = th.stack([member.fragment_avg_reward(f) for f in frags])
    assert th.allclose(batched, looped, rtol=1e-6, atol=1e-6)


def test_per_step_rewards_splits_match_per_traj_forward(rng, tiny_reward_ensemble):
    trajs = make_trajectories(rng, [4, 9, 2])
    member = tiny_reward_ensemble.members[0]
    with th.no_grad():
        batched = per_step_rewards(member, trajs)
        for traj, steps in zip(trajs, batched):
            direct = member(*stacked_transitions(traj))
            assert steps.shape == (len(traj),)
            assert th.allclose(steps, direct, rtol=1e-6, atol=1e-6)


def test_empty_fragment_list():
    assert per_step_rewards(ConstantRewardNet(), []) == []


def test_fragment_sum_rewards_gradients_match_loop(rng, tiny_reward_ensemble):
    """Backward through the batched path produces the same gradients as the loop."""
    frags = make_trajectories(rng, [5, 3, 7])
    member = tiny_reward_ensemble.members[0]

    member.zero_grad()
    fragment_sum_rewards(member, frags).sum().backward()
    batched_grads = [p.grad.clone() for p in member.parameters()]

    member.zero_grad()
    th.stack([member(*stacked_transitions(f)).sum() for f in frags]).sum().backward()
    looped_grads = [p.grad.clone() for p in member.parameters()]

    for got, expected in zip(batched_grads, looped_grads):
        assert th.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_score_trajectories_matches_per_trajectory_predict(rng, tiny_reward_ensemble):
    from human_feedback_rl.common.base_reward_learning_algorithm import (
        BaseRewardLearningAlgorithm,
    )

    class _Shim:
        reward_model = tiny_reward_ensemble
        _score_trajectories = BaseRewardLearningAlgorithm._score_trajectories

    trajs = make_trajectories(rng, [4, 9, 2, 6])  # unequal lengths
    batched = _Shim()._score_trajectories(trajs)
    for traj, got in zip(trajs, batched):
        obs, acts, status, done = (t.numpy() for t in stacked_transitions(traj))
        expected = float(tiny_reward_ensemble.predict(obs, acts, status, done).sum())
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert _Shim()._score_trajectories([]) == []


def test_hand_computed_values_with_constant_net(rng):
    frags = make_trajectories(rng, [2, 3])
    net = ConstantRewardNet()
    with th.no_grad():
        sums = fragment_sum_rewards(net, frags)
    for i, frag in enumerate(frags):
        expected = sum(
            float(t.observation.sum()) + 0.5 * float(t.action.sum()) for t in frag
        )
        assert sums[i].item() == pytest.approx(expected, rel=1e-5)
