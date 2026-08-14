import numpy as np
import pytest
import torch as th
from stable_baselines3 import SAC

from human_feedback_rl.algorithms import HybridAlgorithm
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
    _flatten = staticmethod(HybridAlgorithm._flatten)
    _grad_norm = staticmethod(RewardTrainingMixin._grad_norm)

    _set_flat_grad = staticmethod(HybridAlgorithm._set_flat_grad)
    _alpha_weight = HybridAlgorithm._alpha_weight

    def __init__(self, demo_weight=1.0, max_balance_scale=100.0,
                 gcl_fusion="norm_balance"):
        self.demo_weight = demo_weight
        self.max_balance_scale = max_balance_scale
        self.balance_eps = 1e-12
        self.gcl_fusion = gcl_fusion
        self.alpha_eps = 1e-8
        self._alpha_current = {}


def _run_step(g_pref, g_demo, alpha=None, **shim_kwargs):
    """Apply _reward_step with losses whose gradients are exactly g_pref/g_demo."""
    member = _TwoParam()
    optimizer = th.optim.SGD(member.parameters(), lr=0.0)
    pref_loss = (th.tensor(g_pref) * member.w).sum()
    demo_loss = (th.tensor(g_demo) * member.w).sum()
    shim = _StepShim(**shim_kwargs)
    stats = shim._reward_step(member, optimizer, pref_loss, demo_loss, alpha=alpha)
    return member.w.grad.detach().numpy(), stats


# ---------------------------------------------------------------------------
# Fusione prova 1: direzioni unitarie pesate da alpha, un solo Adam
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_prova1_combina_direzioni_unitarie_con_alpha(alpha):
    """g_fin = (1-a) * g_p/||g_p|| + a * g_d/||g_d||, passato a UN solo optimizer."""
    g_pref, g_demo = [3.0, 4.0], [0.0, 2.0]      # norme 5 e 2
    grad, stats = _run_step(g_pref, g_demo, alpha=alpha,
                            gcl_fusion="alpha_norm_single_adam")
    atteso = (1 - alpha) * _unit(g_pref) + alpha * _unit(g_demo)
    assert np.allclose(grad, atteso, atol=1e-6)
    assert stats["alpha"] == pytest.approx(alpha)


def test_prova1_con_alpha_uno_resta_solo_la_direzione_delle_dimostrazioni():
    """Il fallback sotto la soglia deve ridursi al canale dimostrazioni."""
    grad, _ = _run_step([3.0, 4.0], [0.0, 2.0], alpha=1.0,
                        gcl_fusion="alpha_norm_single_adam")
    assert np.allclose(grad, _unit([0.0, 2.0]), atol=1e-6)


def test_prova1_butta_via_le_norme_dei_due_canali():
    """Scalare un canale non cambia l'update: sopravvive solo la direzione."""
    a, _ = _run_step([3.0, 4.0], [0.0, 2.0], alpha=0.5,
                     gcl_fusion="alpha_norm_single_adam")
    b, _ = _run_step([300.0, 400.0], [0.0, 0.02], alpha=0.5,
                     gcl_fusion="alpha_norm_single_adam")
    assert np.allclose(a, b, atol=1e-6)


def test_prova1_alpha_viene_clampato_in_zero_uno():
    grad, stats = _run_step([1.0, 0.0], [0.0, 1.0], alpha=5.0,
                            gcl_fusion="alpha_norm_single_adam")
    assert stats["alpha"] == pytest.approx(1.0)
    assert np.allclose(grad, [0.0, 1.0], atol=1e-6)


def test_la_fusione_storica_resta_il_default():
    """Chi non chiede la prova 1 deve avere il comportamento di prima."""
    grad, stats = _run_step([1.0, 0.0], [0.0, 1.0])
    assert np.isnan(stats["alpha"])
    assert np.allclose(grad, [1.0, 1.0], atol=1e-6)   # somma bilanciata di norma


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

    assert len(algo.dataset_demo_prefs_train) == 12
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


# ---------------------------------------------------------------------------
# alpha dentro un'iterazione vera
# ---------------------------------------------------------------------------

def _train_once(algo, n_queries=8):
    algo.trajectories = algo.sample_rollout(64)
    algo._collect_feedback(n_queries)
    algo._train_reward_model()


def test_l_iterazione_pubblica_le_quantita_che_formano_alpha(rng):
    """Contratto di logging: senza queste chiavi il sanity check e' invisibile."""
    algo = _hybrid(rng, gcl_fusion="alpha_norm_single_adam", total_queries=12,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    _train_once(algo, n_queries=12)
    scritte = {k for k, _ in algo.logger.name_to_value.items()}
    attese = {
        "alpha/V_pref", "alpha/S_pref", "alpha/cv2_pref",
        "alpha/gradmean_norm_sq_pref", "alpha/n_pref", "alpha/batch_pref",
        "alpha/V_demo", "alpha/S_demo", "alpha/cv2_demo",
        "alpha/gradmean_norm_sq_demo", "alpha/n_demo", "alpha/batch_demo",
        "reward/hybrid_alpha", "reward/hybrid_alpha_active",
    }
    assert attese <= scritte, sorted(attese - scritte)


def test_le_identita_fra_le_quantita_loggate_tengono(rng):
    """S = V/B e CV^2 = S/||gbar||^2, come misurate e pubblicate."""
    algo = _hybrid(rng, gcl_fusion="alpha_norm_single_adam", total_queries=12,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    _train_once(algo, n_queries=12)
    v = algo.logger.name_to_value
    for canale in ("pref", "demo"):
        assert v[f"alpha/S_{canale}"] == pytest.approx(
            v[f"alpha/V_{canale}"] / v[f"alpha/batch_{canale}"], rel=1e-6)
        assert v[f"alpha/cv2_{canale}"] == pytest.approx(
            v[f"alpha/S_{canale}"] / v[f"alpha/gradmean_norm_sq_{canale}"], rel=1e-6)


def test_sotto_la_soglia_alpha_e_uno_e_non_risulta_attivo(rng):
    """Con pochi confronti il peso e' fissato, e il log lo dichiara."""
    algo = _hybrid(rng, gcl_fusion="alpha_norm_single_adam", total_queries=2,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    _train_once(algo, n_queries=2)
    assert algo._alpha_is_active() is False
    assert algo.logger.name_to_value["reward/hybrid_alpha"] == pytest.approx(1.0)
    assert algo.logger.name_to_value["reward/hybrid_alpha_active"] == pytest.approx(0.0)


def test_alpha_weight_e_sola_lettura(rng):
    """_alpha_weight legge la stima corrente; produrla spetta a _estimate_alpha."""
    algo = _hybrid(rng, gcl_fusion="alpha_norm_single_adam",
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    member = algo.reward_model.members[0]
    assert algo._alpha_weight(member) == 1.0        # nessuna stima ancora
    _train_once(algo, n_queries=12)
    assert 0.0 <= algo._alpha_weight(member) <= 1.0


def test_la_stima_non_consuma_l_rng_del_training(rng):
    """Estrarre il rollout per la stima non deve spostare le estrazioni di training."""
    algo = _hybrid(rng, gcl_fusion="alpha_norm_single_adam", total_queries=12,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    algo.trajectories = algo.sample_rollout(64)
    algo._collect_feedback(12)
    prima = algo.rng.bit_generator.state
    algo._estimate_alpha()
    assert algo.rng.bit_generator.state == prima


def test_la_fusione_storica_non_stima_alpha(rng):
    """norm_balance non usa alpha: non deve nemmeno pagarne il costo."""
    algo = _hybrid(rng, gcl_fusion="norm_balance", total_queries=12,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    _train_once(algo, n_queries=12)
    assert algo._alpha_current == {}
    assert "alpha/S_demo" not in algo.logger.name_to_value


def test_demo_1_non_e_supportato_dalla_stima(rng):
    """La decomposizione implementata e' quella di demo_2: meglio fallire chiaro."""
    algo = _hybrid(rng, gcl_fusion="alpha_norm_single_adam", loss_type="demo_1",
                   total_queries=12, reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    algo.trajectories = algo.sample_rollout(64)
    algo._collect_feedback(12)
    with pytest.raises(NotImplementedError):
        algo._estimate_alpha()


def test_label_smoothing_solo_per_le_etichette_bernoulliane(rng):
    algo_b = _hybrid(rng, labels_type="binary_bernoulli", label_smoothing=0.1)
    algo_s = _hybrid(rng, labels_type="soft", label_smoothing=0.1)
    hard = th.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert th.allclose(algo_b._smoothed_labels(hard), th.tensor([[0.95, 0.05], [0.05, 0.95]]))
    assert th.allclose(algo_s._smoothed_labels(hard), hard)


def test_gcl_fusion_sconosciuta_viene_rifiutata(rng):
    with pytest.raises(ValueError, match="gcl_fusion"):
        _hybrid(rng, gcl_fusion="non_esiste")

# --- il bootstrap serve all'ensemble, non al membro singolo -----------------
# Un ricampionamento con rimpiazzo di n elementi su n ne contiene in media solo
# il 63.2% distinti. Con piu' membri e' il prezzo della decorrelazione; con un
# membro solo e' un terzo dei confronti buttato via a ogni iterazione, mentre
# il canale dimostrazioni continua a vedere tutto il suo budget.


def test_con_un_solo_membro_si_usa_tutto_il_dataset(rng):
    algo = _hybrid(rng, total_queries=12,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    _train_once(algo, n_queries=12)
    vista = algo._training_view(algo.dataset_train)
    assert vista is algo.dataset_train, "con un membro non si ricampiona"


def test_con_piu_membri_ogni_membro_vede_un_ricampionamento(rng):
    algo = _hybrid(rng, total_queries=12,
                   reward_model_kwargs=dict(n_ensembles=2, net_arch=[8]))
    _train_once(algo, n_queries=12)
    a = algo._training_view(algo.dataset_train)
    b = algo._training_view(algo.dataset_train)
    assert a is not algo.dataset_train and len(a) == len(algo.dataset_train)
    # due estrazioni indipendenti: e' cio' che decorrela i membri
    assert [id(x) for x in a.get_all().fragment_pairs] != \
           [id(x) for x in b.get_all().fragment_pairs]


def test_il_membro_singolo_non_perde_confronti(rng):
    """La proprieta' che conta: nessun confronto raccolto resta inutilizzato."""
    algo = _hybrid(rng, total_queries=12,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    _train_once(algo, n_queries=12)
    vista = algo._training_view(algo.dataset_train)
    distinti = {id(x) for x in vista.get_all().fragment_pairs}
    tutti = {id(x) for x in algo.dataset_train.get_all().fragment_pairs}
    assert distinti == tutti

# --- stream di RNG indipendenti --------------------------------------------
# Con un unico RNG, ogni estrazione dell'ottimizzazione spostava lo stato di
# quello che sceglie i frammenti e che estrae le etichette. Due run diverse
# solo per gradient_steps_rew ricevevano cosi' feedback diverso, e il confronto
# fra iperparametri misurava anche quale realizzazione era capitata.


def _feedback_raccolto(gradient_steps_rew):
    algo = _hybrid(np.random.default_rng(0), total_queries=8, initial_queries=2,
                   gradient_steps_rew=gradient_steps_rew,
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    _train_once(algo, n_queries=4)
    batch = algo.dataset_train.get_all()
    frammenti = [round(float(fp.frag1[0].true_reward), 8) for fp in batch.fragment_pairs]
    etichette = [(p.pref1, p.pref2) for p in batch.preferences]
    return frammenti, etichette


def test_gli_iperparametri_di_ottimizzazione_non_cambiano_il_feedback():
    """La proprieta' che rende confrontabili due trial di tuning."""
    a = _feedback_raccolto(2)
    b = _feedback_raccolto(9)
    assert a[0] == b[0], "i frammenti scelti dipendono da gradient_steps_rew"
    assert a[1] == b[1], "le etichette dipendono da gradient_steps_rew"


def test_gli_stream_sono_distinti_fra_loro():
    algo = _hybrid(np.random.default_rng(0),
                   reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    stream = [algo._rng_query, algo._rng_oracle, algo._rng_train, algo._grad_probe_rng]
    assert len({id(r) for r in stream}) == 4
    estratti = [r.integers(0, 2**62) for r in stream]
    assert len(set(estratti)) == 4


def test_gli_stream_restano_riproducibili_dal_seed():
    """Indipendenti fra loro, ma deterministici dato run.seed."""
    a = _hybrid(np.random.default_rng(3),
                reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    b = _hybrid(np.random.default_rng(3),
                reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]))
    assert a._rng_query.integers(0, 2**62) == b._rng_query.integers(0, 2**62)
    assert a._rng_oracle.integers(0, 2**62) == b._rng_oracle.integers(0, 2**62)

