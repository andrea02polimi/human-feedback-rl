import numpy as np
import pytest
import torch as th

from human_feedback_rl.common.reward_nets import (
    NormalizedRewardNet,
    RewardEnsemble,
    SumoRewardNet,
    make_reward_ensemble,
)
from human_feedback_rl.common.status import STATUS_DIM

from conftest import OBS_DIM, ACT_DIM, ConstantRewardNet, make_trajectories


def _batch(rng, n=6):
    obs = rng.normal(size=(n, OBS_DIM)).astype(np.float32)
    act = rng.normal(size=(n, ACT_DIM)).astype(np.float32)
    status = np.zeros((n, STATUS_DIM), dtype=np.float32)
    status[:, 4] = 1.0
    done = np.zeros(n, dtype=np.float32)
    return obs, act, status, done


def test_sumo_reward_net_output_shape(fake_env, rng):
    th.manual_seed(0)
    net = SumoRewardNet(fake_env.observation_space, fake_env.action_space, net_arch=[8])
    obs, act, status, done = _batch(rng)
    out = net.predict(obs, act, status, done)
    assert out.shape == (6,)


def test_ensemble_forward_is_member_mean(rng):
    members = [ConstantRewardNet(offset=0.0), ConstantRewardNet(offset=2.0)]
    ensemble = RewardEnsemble(members[0].observation_space, members[0].action_space, members)
    obs, act, status, done = _batch(rng)
    mean = ensemble.predict(obs, act, status, done)
    per_member = ensemble.predict_all(obs, act, status, done)
    assert per_member.shape == (6, 2)
    assert np.allclose(mean, per_member.mean(axis=1))
    assert np.allclose(per_member[:, 1] - per_member[:, 0], 2.0)


def test_normalized_predict_and_raw_forward(rng):
    inner = ConstantRewardNet()
    norm = NormalizedRewardNet(inner, alpha=1)
    norm.set_mean(1.0)
    norm.set_std(4.0)
    obs, act, status, done = _batch(rng)
    raw = inner.predict(obs, act, status, done)
    assert np.allclose(norm.predict_unnormalized(obs, act, status, done), raw)
    assert np.allclose(norm.predict(obs, act, status, done), (raw - 1.0) / (4.0 + 1e-8))
    # forward stays raw so reward-model training is unaffected by the stats.
    fwd = norm(*norm.preprocess(obs, act, status, done)).detach().numpy()
    assert np.allclose(fwd, raw)


def test_normalization_ema_update():
    norm = NormalizedRewardNet(ConstantRewardNet(), alpha=0.5)
    norm.set_mean(10.0)  # 0.5*0 + 0.5*10
    assert norm.normalization_mean == pytest.approx(5.0)
    norm.set_std(3.0)  # 0.5*1 + 0.5*3
    assert norm.normalization_std == pytest.approx(2.0)


def test_fragment_avg_reward_matches_manual_mean(rng):
    net = ConstantRewardNet()
    (traj,) = make_trajectories(rng, [5])
    avg = net.fragment_avg_reward(traj).item()
    manual = np.mean([
        float(t.observation.sum()) + 0.5 * float(t.action.sum()) for t in traj
    ])
    assert avg == pytest.approx(manual, rel=1e-5)


def test_old_layout_checkpoint_loads(fake_env):
    """Pre-v0.2 checkpoints (per-member NormalizedRewardNet) must still load."""
    th.manual_seed(0)
    rm = make_reward_ensemble(fake_env, n_ensembles=2, net_arch=[8])
    new_sd = rm.state_dict()

    old_sd = {}
    for key, value in new_sd.items():
        if key.startswith("net.members."):
            parts = key.split(".")
            old_sd[".".join(parts[:3]) + ".net." + ".".join(parts[3:])] = value.clone()
        else:
            old_sd[key] = value.clone()
    for i in range(2):
        old_sd[f"net.members.{i}._mean"] = th.tensor(3.0)
        old_sd[f"net.members.{i}._std"] = th.tensor(9.0)

    th.manual_seed(1)
    reloaded = make_reward_ensemble(fake_env, n_ensembles=2, net_arch=[8])
    reloaded.load_state_dict(old_sd)
    for key in new_sd:
        assert th.equal(reloaded.state_dict()[key], new_sd[key]), key


def test_new_layout_checkpoint_roundtrip(fake_env):
    th.manual_seed(0)
    rm = make_reward_ensemble(fake_env, n_ensembles=2, net_arch=[8])
    rm.set_mean(0.7)
    sd = rm.state_dict()

    th.manual_seed(1)
    reloaded = make_reward_ensemble(fake_env, n_ensembles=2, net_arch=[8])
    reloaded.load_state_dict(sd)
    assert reloaded.normalization_mean == pytest.approx(0.7)
    for key in sd:
        assert th.equal(reloaded.state_dict()[key], sd[key]), key
