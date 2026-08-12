"""La proprieta' da cui dipendono le curve di budget: stesso budget -> stesse dimostrazioni."""
from __future__ import annotations

import numpy as np
import pytest

from human_feedback_rl.common.demo_subsampling import (
    DEMO_SUBSAMPLE_SEED,
    dataset_fingerprint,
    indices_fingerprint,
    select_demo_indices,
    subsample_manifest,
)

N_AVAILABLE = 500
LENGTHS = [10 + i % 7 for i in range(N_AVAILABLE)]


def indices(n_trajectories=None, n_transitions=None, seed=None):
    return select_demo_indices(
        N_AVAILABLE, lengths=LENGTHS, n_trajectories=n_trajectories,
        n_transitions=n_transitions, seed=seed,
    )


def test_stesso_budget_stesse_dimostrazioni():
    """Un braccio demo e uno ibrido che chiedono n=10 devono ricevere LE STESSE 10."""
    a, b = indices(n_trajectories=10), indices(n_trajectories=10)
    assert np.array_equal(a, b)


def test_la_selezione_ignora_il_seed_di_training():
    """Niente della selezione puo' dipendere da run.seed."""
    assert np.array_equal(indices(n_trajectories=25), indices(n_trajectories=25, seed=None))
    assert indices_fingerprint(indices(n_trajectories=25)) == indices_fingerprint(
        select_demo_indices(N_AVAILABLE, lengths=LENGTHS, n_trajectories=25,
                            seed=DEMO_SUBSAMPLE_SEED))


def test_seed_none_e_la_costante_condivisa():
    assert np.array_equal(
        indices(n_trajectories=10),
        select_demo_indices(N_AVAILABLE, n_trajectories=10, seed=DEMO_SUBSAMPLE_SEED),
    )


@pytest.mark.parametrize("budgets", [(10, 100), (1, 10, 100, 500), (20, 50, 200)])
def test_i_budget_sono_annidati(budgets):
    """Un budget piu' grande AGGIUNGE dimostrazioni, non le scambia."""
    for piccolo, grande in zip(budgets, budgets[1:]):
        a = set(indices(n_trajectories=piccolo).tolist())
        b = set(indices(n_trajectories=grande).tolist())
        assert a <= b, f"{piccolo} non e' contenuto in {grande}"


def test_seed_diversi_selezionano_dimostrazioni_diverse():
    """La garanzia e' una proprieta' del seed condiviso, non un caso fortunato."""
    a = set(indices(n_trajectories=50, seed=1).tolist())
    b = set(indices(n_trajectories=50, seed=2).tolist())
    assert a != b


def test_budget_in_transizioni_rispetta_il_cap_ed_e_annidato():
    sel = indices(n_transitions=200)
    assert sum(LENGTHS[i] for i in sel) <= 200
    piccolo = set(indices(n_transitions=100).tolist())
    grande = set(indices(n_transitions=400).tolist())
    assert piccolo <= grande


def test_budget_in_transizioni_da_almeno_una_traiettoria():
    """Un cap sotto la traiettoria piu' corta restituisce comunque qualcosa di usabile."""
    sel = indices(n_transitions=1)
    assert len(sel) == 1


def test_i_due_budget_condividono_la_permutazione():
    """Leggono lo stesso ordine, quindi i due assi restano confrontabili."""
    per_traj = indices(n_trajectories=1)
    per_trans = indices(n_transitions=1)
    assert per_traj[0] == per_trans[0]


def test_senza_budget_si_prende_tutto_il_dataset():
    assert np.array_equal(select_demo_indices(N_AVAILABLE), np.arange(N_AVAILABLE))


@pytest.mark.parametrize("kwargs", [
    dict(n_trajectories=0),
    dict(n_trajectories=N_AVAILABLE + 1),
    dict(n_transitions=0),
    dict(n_trajectories=10, n_transitions=100),
])
def test_budget_non_validi_sollevano(kwargs):
    with pytest.raises(ValueError):
        indices(**kwargs)


def test_budget_in_transizioni_richiede_le_lunghezze():
    with pytest.raises(ValueError, match="lengths"):
        select_demo_indices(N_AVAILABLE, n_transitions=100)


def test_l_impronta_identifica_l_insieme_non_l_ordine():
    a = [3, 1, 2]
    b = [2, 3, 1]
    assert indices_fingerprint(a) == indices_fingerprint(b)
    assert indices_fingerprint(a) != indices_fingerprint([1, 2, 4])


def test_l_impronta_del_dataset_vede_un_dataset_cambiato():
    assert dataset_fingerprint(LENGTHS) != dataset_fingerprint(LENGTHS[:-1])


def test_il_manifest_descrive_la_selezione():
    sel = indices(n_trajectories=10)
    m = subsample_manifest(sel, LENGTHS, seed=None, n_trajectories=10,
                           dataset_name="expert")
    assert m["n_selected"] == 10
    # nomi vincolati dai consumatori in scripts/
    assert m["subsample_seed"] == DEMO_SUBSAMPLE_SEED
    assert m["budget_n_trajectories"] == 10
    assert m["fingerprint"] == indices_fingerprint(sel)
    assert m["n_transitions_selected"] == sum(LENGTHS[i] for i in sel)
    assert m["dataset_fingerprint"] == dataset_fingerprint(LENGTHS)


def test_i_manifest_di_due_bracci_coincidono_allo_stesso_budget():
    """Forma end-to-end della verifica che le run rendono possibile."""
    def manifest_for(training_seed):
        # il seed di training non entra nella selezione
        sel = select_demo_indices(N_AVAILABLE, lengths=LENGTHS, n_trajectories=100)
        return subsample_manifest(sel, LENGTHS, seed=None, n_trajectories=100)

    assert manifest_for(1)["fingerprint"] == manifest_for(7)["fingerprint"]
