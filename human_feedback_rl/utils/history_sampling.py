import numpy as np
import random


def sample_pair_from_history(history):

    steps_by_state = {}

    for traj in history:

        for step in traj.steps:

            key = tuple(np.round(step.state,3))

            steps_by_state.setdefault(key,[]).append(step)

    valid_states = [
        s for s,steps in steps_by_state.items()
        if len({step.action for step in steps}) > 1
    ]

    if not valid_states:
        return None

    s = random.choice(valid_states)

    steps = steps_by_state[s]

    a,b = random.sample(steps,2)

    if a.action == b.action:
        return None

    return a,b