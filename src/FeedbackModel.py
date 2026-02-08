from abc import ABC, abstractmethod
from typing import Tuple, Union


class FeedbackModel(ABC):

    def __init__(self, env, history):
        self.env = env
        self.history = history

    @abstractmethod
    def required_object_count(self) -> Union[int, Tuple[int, int]]:
        pass

    @abstractmethod
    def mode(self):
        pass

    @abstractmethod
    def scope(self):
        pass