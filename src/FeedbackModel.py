from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Tuple, Union

#==============================================================
# ENUMERATIONS
#==============================================================

class FeedbackScope(Enum):
    STEP = auto()
    TRAJECTORY = auto()

class FeedbackMode(Enum):
    ABSOLUTE = auto()
    RELATIVE = auto()

#==============================================================
# INTERFACES
#==============================================================

class FeedbackModel(ABC):

    def __init__(self, env, history):
        self.env = env
        self.history = history

    @abstractmethod
    def required_object_count(self) -> Union[int, Tuple[int, int]]:
        pass

    @abstractmethod
    def mode(self) -> FeedbackMode:
        pass

    @abstractmethod
    def scope(self) -> FeedbackScope:
        pass

class StepFeedbackModel(FeedbackModel):

    def __init__(self, env, history):
        super().__init__(env, history)

    def scope(self) -> FeedbackScope:
        return FeedbackScope.STEP

class TrajectoryFeedbackModel(FeedbackModel):

    def __init__(self, env, history):
        super().__init__(env, history)

    def scope(self) -> FeedbackScope:
        return FeedbackScope.TRAJECTORY

class AbsoluteFeedbackModel(FeedbackModel):

    def __init__(self, env, history):
        super().__init__(env, history)

class RelativeFeedbackModel(FeedbackModel):

    def __init__(self, env, history):
        super().__init__(env, history)

#==============================================================
# CONCRETE FEEDBACKMODELS
#==============================================================