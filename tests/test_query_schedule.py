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


def test_sparse_constant_schedule_spreads_queries_over_training():
    schedule = _ScheduleOnly("constant", initial_queries=2).build_query_schedule(100, 10)

    # Initial queries are collected before the main training loop.
    schedule[0] -= 2
    query_iterations = [i for i, count in enumerate(schedule) if count]

    assert query_iterations == [12, 24, 37, 49, 62, 74, 87, 99]
    assert all(schedule[i] == 1 for i in query_iterations)


@pytest.mark.parametrize(
    "total,expected_extra_iterations",
    [
        (1_002, []),
        (1_005, [33, 66, 99]),
    ],
)
def test_dense_constant_schedule_stays_uniform(
    total, expected_extra_iterations
):
    schedule = _ScheduleOnly("constant", initial_queries=2).build_query_schedule(100, total)
    schedule[0] -= 2

    assert min(schedule) == 10
    assert max(schedule) <= 11
    assert [i for i, count in enumerate(schedule) if count == 11] == (
        expected_extra_iterations
    )


@pytest.mark.parametrize(
    "total,initial_queries",
    [
        (10, 2),
        (100, 20),
        (250, 50),
        (500, 100),
        (1_000, 200),
        (2_000, 250),
        (5_000, 250),
        (10_000, 250),
    ],
)
def test_constant_schedule_tracks_uniform_cumulative_target(total, initial_queries):
    schedule = _ScheduleOnly("constant", initial_queries).build_query_schedule(100, total)
    schedule[0] -= initial_queries
    remaining = total - initial_queries

    cumulative = 0
    for iteration, count in enumerate(schedule, start=1):
        cumulative += count
        assert cumulative == iteration * remaining // len(schedule)


def test_decaying_schedules_front_load_queries():
    for name in ("hyperbolic", "inverse_quadratic"):
        schedule = _ScheduleOnly(name).build_query_schedule(10, 1000)
        assert schedule[0] == max(schedule)
        assert schedule[0] > schedule[-1]
