"""Single source of truth for the ego-vehicle terminal status encoding.

The environment reports the ego status as a plain string in
``info["ego_status"]``; reward networks consume it as a one-hot vector whose
component order is fixed by ``STATUS_NAMES``. Every status index used anywhere
in the package must come from this module.

The string values mirror ``sumo_gym_ego.EgoStatus`` (e.g. ``"offroad"`` for
``OFF_ROAD``), but are duplicated here so the package can be imported and
tested without the SUMO stack installed.
"""

import numpy as np

STATUS_NAMES = (
    "arrived",
    "collided",
    "offroad",
    "timeout",
    "running",
    "teleported",
    "removed_unknown",
)

STATUS_DIM = len(STATUS_NAMES)

STATUS_ARRIVED = STATUS_NAMES.index("arrived")
STATUS_COLLIDED = STATUS_NAMES.index("collided")
STATUS_OFFROAD = STATUS_NAMES.index("offroad")
STATUS_TIMEOUT = STATUS_NAMES.index("timeout")
STATUS_RUNNING = STATUS_NAMES.index("running")
STATUS_TELEPORTED = STATUS_NAMES.index("teleported")
STATUS_REMOVED_UNKNOWN = STATUS_NAMES.index("removed_unknown")

_STATUS_ONEHOT = {
    name: np.eye(STATUS_DIM, dtype=np.float32)[i] for i, name in enumerate(STATUS_NAMES)
}


def ego_status_to_onehot(status: str) -> np.ndarray:
    """One-hot encode an ego status string; unknown statuses map to "running"."""
    return _STATUS_ONEHOT.get(status, _STATUS_ONEHOT["running"])
