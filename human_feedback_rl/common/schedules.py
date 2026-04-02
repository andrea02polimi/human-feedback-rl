class InverseSchedule:
    def __init__(self, initial_value, final_value, decay_rate=1.0):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_rate = decay_rate

    def __call__(self, progress_remaining):
        progress = 1 - progress_remaining

        inv = 1 / (1 + self.decay_rate * progress)
        inv_0 = 1
        inv_1 = 1 / (1 + self.decay_rate)

        normalized = (inv - inv_1) / (inv_0 - inv_1)

        return self.final_value + (self.initial_value - self.final_value) * normalized