import numpy as np

from human_feedback_rl.common import status


def test_onehot_shape_and_orthogonality():
    vectors = np.stack([status.ego_status_to_onehot(name) for name in status.STATUS_NAMES])
    assert vectors.shape == (status.STATUS_DIM, status.STATUS_DIM)
    assert np.array_equal(vectors, np.eye(status.STATUS_DIM, dtype=np.float32))


def test_unknown_status_maps_to_running():
    unknown = status.ego_status_to_onehot("not-a-status")
    assert unknown[status.STATUS_RUNNING] == 1.0
    assert unknown.sum() == 1.0


def test_index_constants_match_names_order():
    assert status.STATUS_NAMES[status.STATUS_ARRIVED] == "arrived"
    assert status.STATUS_NAMES[status.STATUS_COLLIDED] == "collided"
    assert status.STATUS_NAMES[status.STATUS_OFFROAD] == "offroad"
    assert status.STATUS_NAMES[status.STATUS_TIMEOUT] == "timeout"
    assert status.STATUS_NAMES[status.STATUS_RUNNING] == "running"
    assert status.STATUS_NAMES[status.STATUS_TELEPORTED] == "teleported"
    assert status.STATUS_NAMES[status.STATUS_REMOVED_UNKNOWN] == "removed_unknown"
