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
        # the name, not just the function: build_query_schedule treats
        # "constant" specially, because there the fractional remainders all tie
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


# --- how a budget smaller than the number of iterations is spread -----------
# The low-budget regime: 9 queries over 100 iterations. Every exact quota is
# 0.09, all floor to zero and all the remainders tie; sorting the remainders
# means sorting ties, and a stable argsort hands the whole budget to the LAST
# iterations. The expected positions are the ones of the reference runs:
# 11, 22, ..., 99.


def _positions(total, initial, n_iterations=100, name="constant"):
    algo = _ScheduleOnly(name, initial_queries=initial)
    schedule = algo.build_query_schedule(n_iterations, total)
    schedule[0] = max(schedule[0] - initial, 0)      # the loop subtracts the bootstrap
    return schedule, [i for i, v in enumerate(schedule) if v]


def test_a_small_budget_lands_on_evenly_spaced_indices():
    schedule, positions = _positions(10, 1)
    assert positions == [11, 22, 33, 44, 55, 66, 77, 88, 99]
    assert sum(schedule) + 1 == 10


def test_the_fifth_preference_arrives_by_halfway():
    """The alpha threshold: below 5 comparisons it stays pinned to 1, so it
    has to be cleared early."""
    schedule, _ = _positions(10, 1)
    total = 1
    for i, v in enumerate(schedule):
        total += v
        if total >= 5:
            assert i == 44
            break
    else:
        raise AssertionError("never reached 5 preferences")


@pytest.mark.parametrize("total,initial", [(10, 1), (100, 10), (1000, 100), (37, 3), (7, 0)])
def test_the_sum_stays_exact_across_budgets(total, initial):
    algo = _ScheduleOnly("constant", initial_queries=initial)
    schedule = algo.build_query_schedule(100, total)
    assert sum(schedule) == total
    assert min(schedule) >= 0


def test_the_last_query_does_not_pass_the_last_iteration():
    """The integer formula must land exactly on n-1, never beyond."""
    for total in (2, 9, 31, 77):
        _, positions = _positions(total, 0)
        assert max(positions) <= 99


def test_a_decaying_schedule_stays_decaying():
    """The special case is only for constant: the others keep their remainders."""
    algo = _ScheduleOnly("hyperbolic")
    schedule = algo.build_query_schedule(100, 500)
    assert sum(schedule[:25]) > sum(schedule[75:])
    assert sum(schedule) == 500


def test_a_zero_budget_gives_a_schedule_with_no_queries():
    """The demonstration-only method: no queries, and no exception."""
    algo = _ScheduleOnly("constant", initial_queries=0)
    assert algo.build_query_schedule(100, 0) == [0] * 100
