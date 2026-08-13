import pytest

from human_feedback_rl.common.base_reward_learning_algorithm import QUERY_SCHEDULES


class _ScheduleOnly:
    """Carries just what build_query_schedule needs."""

    from human_feedback_rl.common.base_reward_learning_algorithm import (
        BaseRewardLearningAlgorithm as _B,
    )
    build_query_schedule = _B.build_query_schedule

    def __init__(self, schedule_name, initial_queries=0):
        self.query_schedule = QUERY_SCHEDULES[schedule_name]
        # il nome, non solo la funzione: build_query_schedule tratta "constant"
        # a parte, perche' li' i resti frazionari sono tutti pari
        self.query_schedule_name = schedule_name
        self.initial_queries = initial_queries


@pytest.mark.parametrize("name", sorted(QUERY_SCHEDULES))
@pytest.mark.parametrize("n_iterations,total", [(1, 10), (7, 100), (100, 10_000), (13, 5)])
def test_schedule_length_and_sum(name, n_iterations, total):
    schedule = _ScheduleOnly(name).build_query_schedule(n_iterations, total)
    # Regression: the old schedule had n_iterations + 1 entries and np.round
    # did not preserve the total.
    assert len(schedule) == n_iterations
    assert sum(schedule) == total
    assert all(q >= 0 for q in schedule)


def test_initial_queries_added_to_first_iteration():
    schedule = _ScheduleOnly("constant", initial_queries=50).build_query_schedule(4, 90)
    assert sum(schedule) == 90
    assert schedule[0] == 50 + 10  # 40 remaining spread over 4 iterations


def test_decaying_schedules_front_load_queries():
    for name in ("hyperbolic", "inverse_quadratic"):
        schedule = _ScheduleOnly(name).build_query_schedule(10, 1000)
        assert schedule[0] == max(schedule)
        assert schedule[0] > schedule[-1]


# --- distribuzione quando il budget e' piu' piccolo del numero di iterazioni ---
# Regime dei budget bassi della tesi: 9 query su 100 iterazioni. Ogni quota
# esatta vale 0.09, tutte floor a zero e tutti i resti pari; ordinare i resti
# significa ordinare pareggi, e un argsort stabile consegna l'intero budget alle
# ULTIME iterazioni. Le posizioni attese sono quelle delle run di riferimento:
# 11, 22, ..., 99.


def _positions(total, initial, n_iterations=100, name="constant"):
    algo = _ScheduleOnly(name, initial_queries=initial)
    schedule = algo.build_query_schedule(n_iterations, total)
    schedule[0] = max(schedule[0] - initial, 0)      # il loop sottrae il bootstrap
    return schedule, [i for i, v in enumerate(schedule) if v]


def test_budget_piccolo_cade_su_indici_equispaziati():
    schedule, posizioni = _positions(10, 1)
    assert posizioni == [11, 22, 33, 44, 55, 66, 77, 88, 99]
    assert sum(schedule) + 1 == 10


def test_la_quinta_preferenza_arriva_a_meta_corsa():
    """Soglia di alpha: sotto 5 confronti resta fissato a 1, deve sbloccarsi presto."""
    schedule, _ = _positions(10, 1)
    totale = 1
    for i, v in enumerate(schedule):
        totale += v
        if totale >= 5:
            assert i == 44
            break
    else:
        raise AssertionError("mai raggiunte 5 preferenze")


@pytest.mark.parametrize("total,initial", [(10, 1), (100, 10), (1000, 100), (37, 3), (7, 0)])
def test_la_somma_resta_esatta_coi_budget_della_tesi(total, initial):
    algo = _ScheduleOnly("constant", initial_queries=initial)
    schedule = algo.build_query_schedule(100, total)
    assert sum(schedule) == total
    assert min(schedule) >= 0


def test_l_ultima_query_non_supera_l_ultima_iterazione():
    """La formula intera deve atterrare esattamente su n-1, non oltre."""
    for total in (2, 9, 31, 77):
        _, posizioni = _positions(total, 0)
        assert max(posizioni) <= 99


def test_lo_schedule_decrescente_resta_decrescente():
    """Il caso speciale vale solo per constant: gli altri tengono i resti."""
    algo = _ScheduleOnly("hyperbolic")
    schedule = algo.build_query_schedule(100, 500)
    assert sum(schedule[:25]) > sum(schedule[75:])
    assert sum(schedule) == 500


def test_budget_nullo_da_uno_schedule_vuoto_di_query():
    """Il braccio solo-dimostrazioni: nessuna query, nessuna eccezione."""
    algo = _ScheduleOnly("constant", initial_queries=0)
    assert algo.build_query_schedule(100, 0) == [0] * 100
