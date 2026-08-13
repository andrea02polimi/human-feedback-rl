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
# E' il regime dei budget bassi della tesi: 9 query su 100 iterazioni. Qui
# l'arrotondamento per-iterazione falliva in silenzio: ogni quota esatta valeva
# 0.09, tutte floor a zero e tutti i resti pari, e il tie-break consegnava le 9
# query alle ULTIME nove iterazioni. La run restava senza feedback fino alla 91.


def _collected(schedule, initial):
    """Preferenze disponibili dopo ogni iterazione, come le vede il loop."""
    schedule = list(schedule)
    schedule[0] = max(schedule[0] - initial, 0)   # il loop sottrae il bootstrap
    total, out = initial, []
    for v in schedule:
        total += v
        out.append(total)
    return schedule, out


def test_budget_piccolo_si_distribuisce_su_tutta_la_corsa():
    schedule = _ScheduleOnly("constant", initial_queries=1).build_query_schedule(100, 10)
    schedule, _ = _collected(schedule, 1)
    posizioni = [i for i, v in enumerate(schedule) if v]
    assert len(posizioni) == 9
    assert posizioni[0] < 10, f"prima query troppo tardi: {posizioni}"
    assert posizioni[-1] > 80, f"ultima query troppo presto: {posizioni}"
    salti = [b - a for a, b in zip(posizioni, posizioni[1:])]
    assert max(salti) <= 22, f"buco troppo lungo fra le query: {salti}"


def test_la_quinta_preferenza_arriva_entro_meta_corsa():
    """Sotto 5 confronti alpha resta fissato a 1: deve sbloccarsi presto."""
    schedule = _ScheduleOnly("constant", initial_queries=1).build_query_schedule(100, 10)
    _, cumulate = _collected(schedule, 1)
    prima = next(i for i, tot in enumerate(cumulate) if tot >= 5)
    assert prima < 50, f"la 5a preferenza arriva all'iterazione {prima}"


@pytest.mark.parametrize("total,initial", [(10, 1), (100, 10), (1000, 100), (37, 3), (7, 0)])
def test_la_somma_resta_esatta_coi_budget_della_tesi(total, initial):
    schedule = _ScheduleOnly("constant", initial_queries=initial).build_query_schedule(100, total)
    assert sum(schedule) == total
    assert min(schedule) >= 0


def test_lo_schedule_decrescente_resta_decrescente():
    """La correzione non deve appiattire gli schedule non costanti."""
    schedule = _ScheduleOnly("hyperbolic").build_query_schedule(100, 500)
    assert sum(schedule[:25]) > sum(schedule[75:])


def test_budget_nullo_da_uno_schedule_vuoto_di_query():
    """Il braccio solo-dimostrazioni: nessuna query, nessuna eccezione."""
    schedule = _ScheduleOnly("constant", initial_queries=0).build_query_schedule(100, 0)
    assert schedule == [0] * 100
