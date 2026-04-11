class InverseSchedule:
    """
    Decays an integer value according to 1 / (1 + decay * t).
    Returns at least 1.

    Used to reduce the number of preference queries over time as the reward
    model becomes more accurate (fewer queries needed for fine-tuning).
    """

    def __init__(self, initial_value: int, decay: float = 1.0):
        self.initial_value = initial_value
        self.decay = decay

    def __call__(self, t: int) -> int:
        return max(1, int(self.initial_value / (1.0 + self.decay * t)))