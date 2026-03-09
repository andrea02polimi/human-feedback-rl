from abc import ABC
from typing import TypeVar, Generic

T = TypeVar('T')


class Feedback(ABC, Generic[T]):
    """Base class for all feedback types."""

    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value!r})"