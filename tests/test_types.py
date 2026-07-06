import numpy as np
import pytest

from human_feedback_rl.common.types import Preference, Trajectory, Transition


def _transition(reward=0.0):
    return Transition(observation=np.zeros(2), action=np.zeros(1), true_reward=reward)


class TestTrajectory:
    def test_distinct_trajectories_are_not_equal(self):
        # Regression: the old @dataclass-generated __eq__ made ALL trajectories equal.
        t1 = Trajectory([_transition(1.0)])
        t2 = Trajectory([_transition(2.0), _transition(3.0)])
        assert t1 != t2
        assert len(t1) == 1 and len(t2) == 2

    def test_total_reward_and_length(self):
        traj = Trajectory([_transition(1.0), _transition(2.5)])
        assert traj.total_reward() == 3.5
        assert traj.length() == 2

    def test_slicing_behaves_like_list(self):
        transitions = [_transition(float(i)) for i in range(5)]
        traj = Trajectory(transitions)
        assert list(traj[1:3]) == transitions[1:3]

    def test_add_transition(self):
        traj = Trajectory()
        traj.add_transition(_transition(7.0))
        assert traj.total_reward() == 7.0


class TestPreference:
    def test_valid(self):
        p = Preference(0.3, 0.7)
        assert (p.pref1, p.pref2) == (0.3, 0.7)

    @pytest.mark.parametrize("p1,p2", [(-0.1, 1.1), (1.5, -0.5), (0.5, 0.6), (1.0, 0.5)])
    def test_invalid_raises(self, p1, p2):
        with pytest.raises(ValueError):
            Preference(p1, p2)
