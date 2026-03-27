class InverseSchedule:
    def __init__(self, initial_value, final_value, decay_rate=1.0):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_rate = decay_rate

    def __call__(self, progress_remaining):
        progress = 1 - progress_remaining

        delta = self.initial_value - self.final_value

        return self.final_value + delta / (1 + self.decay_rate * progress)