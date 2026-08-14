"""Il peso di affidabilita' alpha, dalla dispersione per campione al minibatch.

Questi test sono il contratto del metodo: fissano le due definizioni di
varianza, il ruolo distinto di N e B, l'esattezza della decomposizione delle
preferenze, la definizione adottata per le dimostrazioni, la soglia sotto la
quale alpha resta fissato, e le chiavi che devono comparire nei log.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from human_feedback_rl.algorithms.hybrid.alpha_estimation import (
    _dispersion,
    demonstration_sample_gradients,
    estimate_alpha,
    preference_sample_gradients,
)
from human_feedback_rl.algorithms.hybrid.demonstration_losses import demo_2_loss
from human_feedback_rl.common.batching import fragment_avg_rewards, fragment_sum_rewards
from human_feedback_rl.common.datasets import PreferenceBatch
from human_feedback_rl.common.preference_losses import (
    bradley_terry_probs,
    preference_labels_tensor,
    preference_nll,
)
from human_feedback_rl.common.types import FragmentPair, Preference

from conftest import make_trajectories


# --- la ricetta della dispersione -------------------------------------------

def test_dispersione_usa_n_meno_1_e_divide_per_il_minibatch():
    """I due divisori hanno ruoli diversi e non vanno confusi.

    V stima quanto disperde il processo che genera i dati (divisore N-1);
    S = V/B descrive il rumore del gradiente EFFETTIVAMENTE applicato, e B e'
    la dimensione del minibatch, non il numero di campioni disponibili.
    """
    g = th.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])   # media (3,0)
    d = _dispersion(g, batch=2, eps=1e-12)
    atteso_V = ((1 - 3) ** 2 + (3 - 3) ** 2 + (5 - 3) ** 2) / (3 - 1)
    assert d.n == 3 and d.batch == 2
    assert d.process_var == pytest.approx(atteso_V)
    assert d.mean_var == pytest.approx(atteso_V / 2)
    assert d.mean_norm_sq == pytest.approx(9.0)
    assert d.cv2 == pytest.approx(atteso_V / 2 / 9.0)


def test_la_varianza_della_media_cala_quando_il_minibatch_cresce():
    """Il sanity check chiesto dal relatore: a parita' di processo, S cala con B."""
    g = th.randn(50, 4, generator=th.Generator().manual_seed(0))
    s = [_dispersion(g, batch=b, eps=1e-12).mean_var for b in (2, 10, 50)]
    assert s[0] > s[1] > s[2]
    # e V non dipende da B
    v = {_dispersion(g, batch=b, eps=1e-12).process_var for b in (2, 10, 50)}
    assert len(v) == 1


def test_dispersione_non_definita_sotto_due_campioni():
    assert _dispersion(th.zeros(1, 3), batch=1, eps=1e-12) is None
    assert _dispersion(th.zeros(5, 3), batch=0, eps=1e-12) is None


def test_gradienti_identici_danno_varianza_nulla():
    g = th.ones(6, 3)
    assert _dispersion(g, batch=3, eps=1e-12).process_var == pytest.approx(0.0)


# --- decomposizione delle preferenze: esatta --------------------------------

def _pref_batch(rng, n, net):
    trajs = make_trajectories(rng, [4] * (2 * n))
    pairs = [FragmentPair(trajs[2 * i], trajs[2 * i + 1]) for i in range(n)]
    prefs = [Preference(1.0, 0.0) if i % 2 else Preference(0.0, 1.0) for i in range(n)]
    return PreferenceBatch(pairs, prefs)


def test_i_gradienti_per_confronto_ricompongono_il_gradiente_full_batch(
    rng, tiny_reward_ensemble
):
    """La decomposizione delle preferenze e' esatta, non approssimata."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 6, member)
    labels = preference_labels_tensor(batch.preferences)

    per_sample = preference_sample_gradients(member, batch, labels, params)
    media = per_sample.mean(dim=0)

    r1 = fragment_avg_rewards(member, [p.frag1 for p in batch.fragment_pairs])
    r2 = fragment_avg_rewards(member, [p.frag2 for p in batch.fragment_pairs])
    loss = preference_nll(bradley_terry_probs(r1, r2), labels)
    grads = th.autograd.grad(loss, params)
    full = th.cat([g.reshape(-1) for g in grads])

    assert th.allclose(media, full, atol=1e-5)


# --- definizione adottata per le dimostrazioni ------------------------------

def test_il_gradiente_per_dimostrazione_e_quello_della_loss_a_un_esperto(
    rng, tiny_reward_ensemble
):
    """Definizione (a): un esperto per volta, con tutto il rollout congelato."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    esperti = make_trajectories(rng, [5, 5, 5])
    rollout = make_trajectories(np.random.default_rng(7), [5, 5])

    per_sample = demonstration_sample_gradients(member, esperti, rollout, params)

    for i, traj in enumerate(esperti):
        loss_i = demo_2_loss(
            fragment_sum_rewards(member, [traj]),
            fragment_sum_rewards(member, rollout),
        )
        atteso = th.cat([g.reshape(-1) for g in th.autograd.grad(loss_i, params)])
        assert th.allclose(per_sample[i], atteso, atol=1e-5), f"campione {i}"


def test_il_rollout_e_condiviso_da_tutti_i_campioni(rng, tiny_reward_ensemble):
    """Cambiare il rollout cambia TUTTE le righe: non e' una nuisance per campione."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    esperti = make_trajectories(rng, [5, 5])
    a = demonstration_sample_gradients(
        member, esperti, make_trajectories(np.random.default_rng(1), [5, 5]), params)
    b = demonstration_sample_gradients(
        member, esperti, make_trajectories(np.random.default_rng(2), [5, 5]), params)
    assert not th.allclose(a, b)


# --- da CV^2 ad alpha -------------------------------------------------------

def test_alpha_resta_fissato_a_uno_sotto_la_soglia_di_confronti(
    rng, tiny_reward_ensemble
):
    """Con pochissimi confronti la dispersione delle preferenze non e' stimabile."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 3, member)          # 3 < 5
    est = estimate_alpha(
        member, params, batch, preference_labels_tensor(batch.preferences),
        make_trajectories(rng, [5, 5]), make_trajectories(rng, [5, 5]),
        batch_size_pref=8, batch_size_expert=8, min_prefs=5, eps=1e-8,
    )
    assert est.alpha == 1.0 and est.pinned is True
    assert est.pref is None and est.demo is None


def test_alpha_e_il_rapporto_fra_i_due_cv2(rng, tiny_reward_ensemble):
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 8, member)
    est = estimate_alpha(
        member, params, batch, preference_labels_tensor(batch.preferences),
        make_trajectories(rng, [5, 5, 5]), make_trajectories(rng, [5, 5]),
        batch_size_pref=4, batch_size_expert=2, min_prefs=5, eps=1e-8,
    )
    assert est.pinned is False
    assert 0.0 <= est.alpha <= 1.0
    atteso = est.pref.cv2 / (est.pref.cv2 + est.demo.cv2)
    assert est.alpha == pytest.approx(atteso)


def test_il_minibatch_e_il_minimo_fra_batch_size_e_campioni(rng, tiny_reward_ensemble):
    """B = min(batch_size, N): non si puo' mediare su piu' campioni di quanti ce ne siano."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 6, member)
    est = estimate_alpha(
        member, params, batch, preference_labels_tensor(batch.preferences),
        make_trajectories(rng, [5, 5, 5]), make_trajectories(rng, [5, 5]),
        batch_size_pref=999, batch_size_expert=999, min_prefs=5, eps=1e-8,
    )
    assert est.pref.batch == 6 and est.pref.n == 6
    assert est.demo.batch == 3 and est.demo.n == 3


def test_alpha_sale_quando_le_preferenze_disperdono_di_piu():
    """Il senso del peso: chi disperde di piu' ne prende di meno."""
    poco = _dispersion(th.tensor([[1.0, 0.0], [1.1, 0.0]]), batch=2, eps=1e-12)
    molto = _dispersion(th.tensor([[1.0, 0.0], [9.0, 0.0]]), batch=2, eps=1e-12)
    alpha_pref_rumorose = molto.cv2 / (molto.cv2 + poco.cv2)
    alpha_demo_rumorose = poco.cv2 / (poco.cv2 + molto.cv2)
    assert alpha_pref_rumorose > 0.5 > alpha_demo_rumorose
