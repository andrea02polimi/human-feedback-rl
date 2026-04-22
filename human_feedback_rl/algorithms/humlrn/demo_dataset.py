"""
DemonstrationDataset
====================
Buffer di traiettorie dimostrative raccolte dall'esperto.
Usato dal RewardTrainerHumLrn per la demonstration loss.
"""

from typing import List
from human_feedback_rl.common.types import Trajectory


class DemonstrationDataset:
    """
    Circular buffer di traiettorie esperto.

    Ogni traiettoria è una sequenza di Transition (obs, action, reward).
    Il reward è quello dell'esperto e NON viene usato dal reward model —
    serve solo per estrarre coppie (obs, action) per la demonstration loss.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.trajectories: List[Trajectory] = []

    def push(self, trajectories: List[Trajectory]) -> None:
        self.trajectories.extend(trajectories)
        if len(self.trajectories) > self.capacity:
            self.trajectories = self.trajectories[-self.capacity:]

    def __len__(self) -> int:
        return len(self.trajectories)

    # TODO: aggiungere metodo sample() per campionare segmenti casuali
    #       da usare durante il training batch della demonstration loss.